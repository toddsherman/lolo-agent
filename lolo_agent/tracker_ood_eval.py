"""WP5 tracker out-of-distribution evaluation (learnings section 4.31 item a).

Measures how well the learned controllable-region tracker localizes on
Room 3 frames it never trained on, scored against DETECTOR-FREE
counterfactual ground truth reconstructed from the paired-probe runs'
telemetry (``entity-v322`` .. ``entity-v326``).  This is an evaluation-only
use of assisted-collected run archives: the ground truth itself derives
purely from counterfactual branch structure (factual action endpoints
versus duration-matched ``NOOP`` control endpoints recorded from the same
saved emulator state), never from the goal-prior player detector, and no
training artifact is produced.

Ground-truth construction
-------------------------

The evaluation runs' option search saves a parent emulator state, then for
every action edge loads that state, steps the action for its duration, and
records the endpoint (``human_prior_option_branch_verified``: the executed
edge is ``path[-1]`` / ``durations[-1]`` from ``parent_state_id``).  The
same search also records a duration-matched ``NOOP`` endpoint from the
same parent state (``human_prior_option_local_neutral_verified``), and at
depth one a root-level ``NOOP`` endpoint
(``human_prior_option_neutral_verified``).  Every such record is therefore
a first-step counterfactual edge in exactly the sense of
``counterfactual_labels``, with the saved parent state as the causal root.

The edges are regrouped into ``CounterfactualRoot`` values and labeled by
``counterfactual_labels.label_counterfactual_root`` itself -- the same
endpoint-difference, 4-connected-component, and leave-one-action-out
corroboration rule that produced the tracker's training labels, including
its explicit censoring (``absent_control``, ``ambiguous_endpoint``,
``ambiguous_control``, ``no_sibling_corroboration``).  Labeled arms whose
controllable mask is empty carry no localization evidence and are excluded
from scoring, mirroring ``arm_examples_from_records``.

Metrics (report-only, no gate)
------------------------------

Per scored arm the tracker predicts on the arm's factual endpoint frame
(the training input convention) and is scored against the arm's true
controllable cells:

- ``hit_rate``: the true component's argmax cell (the true cell with the
  highest tracker probability) lies inside the tracker's ``>= 0.5`` mask,
  i.e. the thresholded mask touches the true component at all.
- ``argmax_in_true_rate``: the tracker's global argmax cell lies inside
  the true controllable cells.
- ``cell_auc``: pooled per-cell ROC AUC against the true cells (residual
  cells count as negatives, as in ``validate_controllable_tracker``).
- ``iou``: per-arm intersection-over-union between the thresholded mask
  and the true cells, reported as a distribution.
- ``true_cell_probability``: tracker probability at the true component's
  argmax cell and averaged over the true cells.  Detector-free ground
  truth localizes the controllable component (vacated plus occupied
  cells), not the single player cell, so the per-arm maximum is an upper
  bound on the probability assigned to the true player cell.

As a clearly-labeled secondary reference only (assisted telemetry,
evaluation-only per learnings section 4.31), the probability at the
player cell recorded by the assisted detector at each factual endpoint is
also reported for the Room 3 runs.

The same metrics are computed on a held-in sample of ``lolo1-medium``
validation arms (the tracker's own hash-stable run-held-out split, so the
frames are unseen but in-distribution) to quantify the gap on one axis.

Usage::

    python -m lolo_agent.tracker_ood_eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .counterfactual_labels import (
    CounterfactualRoot,
    STATUS_LABELED,
    _eligible_factual_arms,
    _roots_from_edges,
    label_counterfactual_root,
    root_frame_digests,
)
from .controllable_tracker import (
    ControllableArmExample,
    _roc_auc,
    arm_examples_from_records,
    sample_arm_examples,
)
from .environment import Action
from .experience_import import decode_logged_png
from .pixels import Frame
from .tracker_substitution_replay import (
    DEFAULT_BACKBONE,
    DEFAULT_CHECKPOINT,
    LEARNED_MASK_PROBABILITY_THRESHOLD,
    load_replay_tracker,
)

Cell = Tuple[int, int]
ProbabilityMap = Tuple[Tuple[float, ...], ...]

EVAL_VERSION = 1
EVAL_KIND = "wp5-tracker-ood-eval"
_DIGEST_PREFIX = f"{EVAL_KIND}:v{EVAL_VERSION}:"

GRID_COLUMNS = 16
GRID_ROWS = 15

BRANCH_EVENT = "human_prior_option_branch_verified"
LOCAL_NEUTRAL_EVENT = "human_prior_option_local_neutral_verified"
NEUTRAL_EVENT = "human_prior_option_neutral_verified"
STATE_SAVED_EVENT = "state_saved"

DEFAULT_ROOM3_RUNS = (
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v322-room3-paired-probe-arm-a-pushed-d12",
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v323-room3-paired-probe-arm-b-prepush-d12",
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v324-room3-paired-probe-arm-b-rerun-certified-d12",
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v325-room3-object-removed-probe-d12",
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v326-room3-object-removed-repetition-d12",
)
DEFAULT_LABELS = "experiments/lolo1-wp5/wp5-labels-full.jsonl"
DEFAULT_DATASET = "experiments/lolo1-medium/dataset"
DEFAULT_REPORT = "experiments/lolo1-wp5/tracker-ood-report.json"

# Held-in reference convention: the tracker's own run-held-out validation
# split (training CLI default modulus 5) sampled with the uncapped run's
# validation sampling seed (training seed 17 + 1, per the trainer).
HELD_IN_VALIDATION_MODULUS = 5
HELD_IN_SAMPLE_SEED = 18
DEFAULT_HELD_IN_ARMS = 400


@dataclass(frozen=True)
class ProbeEdge:
    """One first-step counterfactual edge recorded by the option search."""

    parent_state_id: str
    action: str
    duration: int
    endpoint_digest: str
    # Assisted detector's player cell at the endpoint (branch events only);
    # secondary evaluation-only reference, never part of the ground truth.
    assisted_player_cell: Optional[Cell] = None


def _event_cell(event: Mapping[str, Any]) -> Optional[Cell]:
    slot = event.get("human_prior_target_player_slot")
    width = event.get("frame_width")
    height = event.get("frame_height")
    if slot is None or not width or not height:
        return None
    return (
        int(slot[0]) * GRID_COLUMNS // int(width),
        int(slot[1]) * GRID_ROWS // int(height),
    )


def probe_first_step_edges(
    events: Iterable[Mapping[str, Any]],
) -> Tuple[ProbeEdge, ...]:
    """Extract every recorded first-step counterfactual edge.

    Branch events contribute their executed edge (``path[-1]`` from
    ``parent_state_id``); local-neutral events contribute the
    duration-matched ``NOOP`` edge from the same parent; root-neutral
    events contribute a ``NOOP`` edge only at depth one, because deeper
    root neutrals record the endpoint of the full elapsed duration, not of
    a single step.  Unrecognized events are ignored.
    """

    edges: List[ProbeEdge] = []
    for event in events:
        kind = event.get("event")
        if kind == BRANCH_EVENT:
            path = event["path"]
            durations = event["durations"]
            if not path or not durations or len(path) != len(durations):
                raise ValueError("branch event path/durations malformed")
            edges.append(
                ProbeEdge(
                    parent_state_id=str(event["parent_state_id"]),
                    action=str(path[-1]),
                    duration=int(durations[-1]),
                    endpoint_digest=str(event["frame"]),
                    assisted_player_cell=_event_cell(event),
                )
            )
        elif kind == LOCAL_NEUTRAL_EVENT:
            edges.append(
                ProbeEdge(
                    parent_state_id=str(event["parent_state_id"]),
                    action=str(event["action"]),
                    duration=int(event["action_frames"]),
                    endpoint_digest=str(event["frame"]),
                )
            )
        elif kind == NEUTRAL_EVENT:
            path = event["path"]
            if len(path) != 1:
                continue
            edges.append(
                ProbeEdge(
                    parent_state_id=str(event["source_state_id"]),
                    action=str(path[0]),
                    duration=int(event["durations"][0]),
                    endpoint_digest=str(event["frame"]),
                )
            )
    return tuple(edges)


def state_frame_index(events: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """Map saved state ids to their recorded frame digests."""

    index: Dict[str, str] = {}
    for event in events:
        if event.get("event") != STATE_SAVED_EVENT:
            continue
        state_id = event.get("state_id")
        frame = event.get("frame")
        if state_id and frame:
            index[str(state_id)] = str(frame)
    return index


def state_group(state_id: str) -> int:
    """Deterministic integer group key from a ``state-NNNNNNNN`` id."""

    _, _, suffix = state_id.rpartition("-")
    if not suffix.isdigit():
        raise ValueError(f"cannot derive a group key from state id {state_id!r}")
    return int(suffix)


def collect_probe_roots(
    run_id: str,
    edges: Sequence[ProbeEdge],
    state_frames: Mapping[str, str],
) -> Tuple[CounterfactualRoot, ...]:
    """Regroup probe edges into causal roots keyed by saved parent state.

    A saved state id denotes exactly one emulator state within a run
    (state ids are never reused), so grouping by parent state is the
    strict analog of the label generator's shared-root-frame grouping.
    The root digest is the parent state's recorded frame digest when the
    save event carries one, else the state id itself; either way the
    ``(run, group, digest)`` key stays unique per saved state.
    """

    return _roots_from_edges(
        (
            run_id,
            state_group(edge.parent_state_id),
            state_frames.get(edge.parent_state_id, edge.parent_state_id),
            Action(edge.action),
            edge.duration,
            edge.endpoint_digest,
        )
        for edge in edges
    )


def assisted_cell_index(
    edges: Sequence[ProbeEdge],
) -> Dict[Tuple[str, str, int], Optional[Cell]]:
    """Recorded assisted player cell per (parent state, action, duration).

    ``None`` marks an arm whose recordings disagree or carry no detection;
    such arms are simply absent from the secondary reference metrics.
    """

    index: Dict[Tuple[str, str, int], Optional[Cell]] = {}
    for edge in edges:
        if edge.assisted_player_cell is None:
            continue
        key = (edge.parent_state_id, edge.action, edge.duration)
        if key in index and index[key] != edge.assisted_player_cell:
            index[key] = None
        else:
            index.setdefault(key, edge.assisted_player_cell)
    return index


class RunFrameCache:
    """LRU-decoded content-addressed frames from one run's ``frames/`` dir."""

    def __init__(self, frames_dir: Path, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("frame cache capacity must be positive")
        self.frames_dir = Path(frames_dir)
        self.capacity = capacity
        self._cache: "OrderedDict[str, Frame]" = OrderedDict()

    def get(self, digest: str) -> Frame:
        cached = self._cache.get(digest)
        if cached is not None:
            self._cache.move_to_end(digest)
            return cached
        path = self.frames_dir / f"{digest}.png"
        if not path.exists():
            raise KeyError(f"missing content-addressed frame {digest}")
        frame = decode_logged_png(path)
        if frame.digest != digest:
            raise ValueError(f"frame digest mismatch for {path}")
        self._cache[digest] = frame
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return frame


def label_probe_roots(
    roots: Sequence[CounterfactualRoot],
    frame_getter: Any,
    *,
    columns: int = GRID_COLUMNS,
    rows: int = GRID_ROWS,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Label eligible roots with the counterfactual-labels rule itself.

    Returns payload records (the label generator's documented contract)
    plus extraction counts.  Roots without any control-paired factual arm
    produce no record, exactly as ``generate_labels`` skips them.
    """

    records: List[Dict[str, Any]] = []
    skipped_roots = 0
    for root in sorted(roots, key=lambda item: item.sort_key):
        if not _eligible_factual_arms(root):
            skipped_roots += 1
            continue
        frames = {
            digest: frame_getter(digest)
            for digest in root_frame_digests(root)
        }
        records.append(
            label_counterfactual_root(
                root, frames, columns=columns, rows=rows
            ).payload()
        )
    return records, {
        "roots": len(roots),
        "roots_labeled": len(records),
        "roots_without_eligible_arms": skipped_roots,
    }


def censor_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        for arm in record["arms"]:
            if arm["status"] != STATUS_LABELED:
                reason = str(arm["censor_reason"])
                counts[reason] = counts.get(reason, 0) + 1
    return {reason: counts[reason] for reason in sorted(counts)}


def predict_probability_maps(
    tracker: Any,
    digests: Iterable[str],
    frame_getter: Any,
    batch_size: int = 32,
) -> Dict[str, ProbabilityMap]:
    """Batched ensemble-mean probability maps, keyed by frame digest.

    Digests are processed in sorted order with a fixed batch size so a
    rerun issues identical batches and the report digest is reproducible;
    frames are fetched batch by batch so the corpus never resides in
    memory at once.
    """

    import torch

    from .neural_world_model import frame_tensor

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = next(tracker.heads.parameters()).device
    ordered = sorted(set(digests))
    maps: Dict[str, ProbabilityMap] = {}
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        stacked = torch.stack(
            [frame_tensor(frame_getter(digest)) for digest in batch]
        ).to(device)
        mean, _variance = tracker.predict_map(stacked)
        for digest, grid in zip(batch, mean.cpu().tolist()):
            maps[digest] = tuple(
                tuple(float(value) for value in row) for row in grid
            )
    return maps


def arm_localization_metrics(
    probabilities: ProbabilityMap,
    true_cells: Sequence[Cell],
    threshold: float = LEARNED_MASK_PROBABILITY_THRESHOLD,
) -> Dict[str, Any]:
    """Score one arm's probability map against its true controllable cells."""

    if not true_cells:
        raise ValueError("localization metrics require true cells")
    rows = len(probabilities)
    columns = len(probabilities[0])
    true_set = set(true_cells)

    def probability(cell: Cell) -> float:
        return probabilities[cell[1]][cell[0]]

    # Deterministic argmax: highest probability, ties to smallest (row, column).
    argmax_cell = min(
        (
            (column, row)
            for row in range(rows)
            for column in range(columns)
        ),
        key=lambda cell: (-probability(cell), cell[1], cell[0]),
    )
    true_argmax_cell = min(
        sorted(true_set), key=lambda cell: (-probability(cell), cell[1], cell[0])
    )
    mask = {
        (column, row)
        for row in range(rows)
        for column in range(columns)
        if probabilities[row][column] >= threshold
    }
    intersection = len(mask & true_set)
    union = len(mask | true_set)
    true_values = [probability(cell) for cell in sorted(true_set)]
    return {
        "hit": true_argmax_cell in mask,
        "argmax_in_true": argmax_cell in true_set,
        "iou": intersection / union,
        "true_max_probability": probability(true_argmax_cell),
        "true_mean_probability": sum(true_values) / len(true_values),
        "predicted_mask_cells": len(mask),
        "true_cells": len(true_set),
    }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile of empty sequence")
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def score_examples(
    examples: Sequence[ControllableArmExample],
    maps: Mapping[str, ProbabilityMap],
    threshold: float = LEARNED_MASK_PROBABILITY_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[float], List[int]]:
    """Per-arm metric rows plus pooled cell probability/label arrays."""

    rows: List[Dict[str, Any]] = []
    pooled_probabilities: List[float] = []
    pooled_labels: List[int] = []
    for example in sorted(examples, key=lambda item: item.sort_key):
        probabilities = maps[example.endpoint_digest]
        row = arm_localization_metrics(
            probabilities, example.controllable_cells, threshold
        )
        row["key"] = (
            example.source_run_id,
            example.group,
            example.action,
            example.duration,
        )
        rows.append(row)
        true_set = set(example.controllable_cells)
        for cell_row in range(example.rows):
            for column in range(example.columns):
                pooled_probabilities.append(probabilities[cell_row][column])
                pooled_labels.append(int((column, cell_row) in true_set))
    return rows, pooled_probabilities, pooled_labels


def summarize_metrics(
    rows: Sequence[Mapping[str, Any]],
    pooled_probabilities: Sequence[float],
    pooled_labels: Sequence[int],
    threshold: float = LEARNED_MASK_PROBABILITY_THRESHOLD,
) -> Dict[str, Any]:
    """Aggregate per-arm rows into the reported metric block."""

    if not rows:
        return {"arms": 0}
    ious = sorted(row["iou"] for row in rows)
    true_max = sorted(row["true_max_probability"] for row in rows)
    true_mean = sorted(row["true_mean_probability"] for row in rows)
    controllable = [
        value
        for value, label in zip(pooled_probabilities, pooled_labels)
        if label
    ]
    background = [
        value
        for value, label in zip(pooled_probabilities, pooled_labels)
        if not label
    ]
    return {
        "arms": len(rows),
        "threshold": threshold,
        "hit_rate": sum(row["hit"] for row in rows) / len(rows),
        "argmax_in_true_rate": (
            sum(row["argmax_in_true"] for row in rows) / len(rows)
        ),
        "cell_auc": _roc_auc(list(pooled_probabilities), list(pooled_labels)),
        "iou": {
            "mean": sum(ious) / len(ious),
            "min": ious[0],
            "p25": _quantile(ious, 0.25),
            "median": _quantile(ious, 0.5),
            "p75": _quantile(ious, 0.75),
            "max": ious[-1],
            "fraction_zero": sum(value == 0.0 for value in ious) / len(ious),
            "fraction_ge_0_5": sum(value >= 0.5 for value in ious) / len(ious),
        },
        "true_cell_probability": {
            "max_mean": sum(true_max) / len(true_max),
            "max_median": _quantile(true_max, 0.5),
            "mean_mean": sum(true_mean) / len(true_mean),
            "mean_median": _quantile(true_mean, 0.5),
        },
        "mean_controllable_probability": (
            sum(controllable) / len(controllable) if controllable else 0.0
        ),
        "mean_background_probability": (
            sum(background) / len(background) if background else 0.0
        ),
        "predicted_mask_cells_mean": (
            sum(row["predicted_mask_cells"] for row in rows) / len(rows)
        ),
        "true_cells_mean": sum(row["true_cells"] for row in rows) / len(rows),
    }


def assisted_reference_metrics(
    examples: Sequence[ControllableArmExample],
    maps: Mapping[str, ProbabilityMap],
    assisted_cells: Mapping[Tuple[str, str, int], Optional[Cell]],
    group_to_state: Mapping[int, str],
    threshold: float = LEARNED_MASK_PROBABILITY_THRESHOLD,
) -> Dict[str, Any]:
    """Secondary evaluation-only reference: tracker probability at the
    assisted detector's recorded player cell (never part of ground truth)."""

    values: List[float] = []
    in_true = 0
    for example in sorted(examples, key=lambda item: item.sort_key):
        state_id = group_to_state.get(example.group)
        if state_id is None:
            continue
        cell = assisted_cells.get((state_id, example.action, example.duration))
        if cell is None:
            continue
        probabilities = maps[example.endpoint_digest]
        values.append(probabilities[cell[1]][cell[0]])
        if cell in set(example.controllable_cells):
            in_true += 1
    if not values:
        return {"arms_with_assisted_cell": 0}
    ordered = sorted(values)
    return {
        "arms_with_assisted_cell": len(values),
        "probability_mean": sum(values) / len(values),
        "probability_median": _quantile(ordered, 0.5),
        "fraction_ge_threshold": (
            sum(value >= threshold for value in values) / len(values)
        ),
        "assisted_cell_in_true_cells_rate": in_true / len(values),
    }


def _round_floats(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item, digits) for item in value]
    return value


def content_digest(payload: Mapping[str, Any]) -> str:
    """Deterministic digest over the report minus its own digest field."""

    body = {
        key: value for key, value in payload.items() if key != "content_digest"
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(
        (_DIGEST_PREFIX + canonical).encode("utf-8")
    ).hexdigest()


def build_report(
    room3_runs: Sequence[Mapping[str, Any]],
    room3_pooled: Mapping[str, Any],
    held_in: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble the deterministic content-digested evaluation report."""

    pooled_metrics = room3_pooled.get("metrics", {})
    held_in_metrics = held_in.get("metrics", {})
    gap: Dict[str, Any] = {}
    if pooled_metrics.get("arms") and held_in_metrics.get("arms"):
        gap = {
            "hit_rate": held_in_metrics["hit_rate"] - pooled_metrics["hit_rate"],
            "cell_auc": held_in_metrics["cell_auc"] - pooled_metrics["cell_auc"],
            "iou_mean": (
                held_in_metrics["iou"]["mean"] - pooled_metrics["iou"]["mean"]
            ),
            "true_cell_probability_max_mean": (
                held_in_metrics["true_cell_probability"]["max_mean"]
                - pooled_metrics["true_cell_probability"]["max_mean"]
            ),
        }
    report = {
        "version": EVAL_VERSION,
        "kind": EVAL_KIND,
        "basis": "docs/learnings.md section 4.31 plan-change item (a)",
        "evaluation_only": True,
        "ground_truth": (
            "detector-free counterfactual localization: factual versus "
            "duration-matched NOOP endpoints from shared saved states, "
            "labeled by counterfactual_labels.label_counterfactual_root"
        ),
        "provenance": dict(provenance),
        "grid": {"columns": GRID_COLUMNS, "rows": GRID_ROWS},
        "threshold": {
            "mask_probability": LEARNED_MASK_PROBABILITY_THRESHOLD,
            "rationale": (
                "the checkpoint's validation operating point, as pinned by "
                "the substitution replay"
            ),
        },
        "room3_runs": [dict(run) for run in room3_runs],
        "room3_pooled": dict(room3_pooled),
        "held_in": dict(held_in),
        "gap_held_in_minus_room3": gap,
        "result": "report-only-no-gate",
    }
    report = _round_floats(report)
    report["content_digest"] = content_digest(report)
    return report


def _read_events(events_path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with Path(events_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def evaluate_room3_run(
    run_dir: Path,
    tracker: Any,
    *,
    batch_size: int = 32,
    frame_cache_capacity: int = 1024,
) -> Dict[str, Any]:
    """Extract ground truth from one probe run and score the tracker."""

    run_dir = Path(run_dir)
    events = _read_events(run_dir / "events.jsonl")
    edges = probe_first_step_edges(events)
    state_frames = state_frame_index(events)
    roots = collect_probe_roots(run_dir.name, edges, state_frames)
    cache = RunFrameCache(run_dir / "frames", capacity=frame_cache_capacity)
    records, root_stats = label_probe_roots(roots, cache.get)
    examples, example_stats = arm_examples_from_records(records)
    unique_digests = sorted({item.endpoint_digest for item in examples})
    maps = predict_probability_maps(
        tracker, unique_digests, cache.get, batch_size
    )
    rows, pooled_probabilities, pooled_labels = score_examples(examples, maps)
    group_to_state = {
        state_group(edge.parent_state_id): edge.parent_state_id
        for edge in edges
    }
    assisted = assisted_reference_metrics(
        examples, maps, assisted_cell_index(edges), group_to_state
    )
    return {
        "run_id": run_dir.name,
        "extraction": {
            "events": len(events),
            "probe_edges": len(edges),
            **root_stats,
            **example_stats,
            "censored_by_reason": censor_counts(records),
            "scored_arms": len(rows),
            "unique_factual_endpoint_frames": len(unique_digests),
        },
        "metrics": summarize_metrics(rows, pooled_probabilities, pooled_labels),
        "assisted_reference": assisted,
        "_rows": rows,
        "_pooled_probabilities": pooled_probabilities,
        "_pooled_labels": pooled_labels,
    }


def evaluate_held_in_sample(
    labels_path: Path,
    dataset_path: Path,
    tracker: Any,
    *,
    sample_arms: int = DEFAULT_HELD_IN_ARMS,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Score the tracker on held-out lolo1-medium arms (in-distribution)."""

    from .controllable_tracker import (
        decode_arm_examples,
        load_labeled_arm_examples,
    )
    from .counterfactual_labels import open_strict_store
    from .ensemble_world_model import split_sequence_runs

    store = open_strict_store(Path(dataset_path))
    examples, manifest, _statistics = load_labeled_arm_examples(
        Path(labels_path)
    )
    _training, validation = split_sequence_runs(
        examples, validation_modulus=HELD_IN_VALIDATION_MODULUS
    )
    sampled = sample_arm_examples(validation, sample_arms, HELD_IN_SAMPLE_SEED)
    decoded = decode_arm_examples(store, sampled)
    frames_by_digest: Dict[str, Frame] = {
        item.endpoint_digest: item.frame for item in decoded if item.frame
    }
    maps = predict_probability_maps(
        tracker, frames_by_digest, frames_by_digest.__getitem__, batch_size
    )
    rows, pooled_probabilities, pooled_labels = score_examples(decoded, maps)
    return {
        "corpus": "lolo1-medium validation arms (run-held-out split)",
        "labels": str(labels_path),
        "label_manifest_sha256": str(manifest["content_digest"]),
        "validation_modulus": HELD_IN_VALIDATION_MODULUS,
        "sample_seed": HELD_IN_SAMPLE_SEED,
        "sampled_arms": len(sampled),
        "validation_arms_available": len(validation),
        "validation_source_runs": sorted(
            {item.source_run_id for item in sampled}
        ),
        "metrics": summarize_metrics(rows, pooled_probabilities, pooled_labels),
    }


def run_evaluation(
    run_dirs: Sequence[Path],
    checkpoint_path: Path,
    backbone_path: Path,
    labels_path: Path,
    dataset_path: Path,
    report_path: Path,
    *,
    held_in_arms: int = DEFAULT_HELD_IN_ARMS,
    batch_size: int = 32,
) -> Dict[str, Any]:
    tracker, provenance = load_replay_tracker(checkpoint_path, backbone_path)
    run_results: List[Dict[str, Any]] = []
    pooled_rows: List[Dict[str, Any]] = []
    pooled_probabilities: List[float] = []
    pooled_labels: List[int] = []
    for run_dir in run_dirs:
        result = evaluate_room3_run(
            Path(run_dir), tracker, batch_size=batch_size
        )
        pooled_rows.extend(result.pop("_rows"))
        pooled_probabilities.extend(result.pop("_pooled_probabilities"))
        pooled_labels.extend(result.pop("_pooled_labels"))
        run_results.append(result)
    room3_pooled = {
        "runs": [result["run_id"] for result in run_results],
        "metrics": summarize_metrics(
            pooled_rows, pooled_probabilities, pooled_labels
        ),
    }
    held_in = evaluate_held_in_sample(
        Path(labels_path),
        Path(dataset_path),
        tracker,
        sample_arms=held_in_arms,
        batch_size=batch_size,
    )
    report = build_report(run_results, room3_pooled, held_in, provenance)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "WP5 tracker out-of-distribution evaluation against detector-"
            "free counterfactual ground truth from Room 3 probe telemetry"
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        default=None,
        help="Room 3 probe run directory (repeatable)",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument(
        "--held-in-arms", type=int, default=DEFAULT_HELD_IN_ARMS
    )
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args(argv)
    if arguments.held_in_arms <= 0 or arguments.batch_size <= 0:
        parser.error("sample and batch sizes must be positive")
    runs = [Path(path) for path in (arguments.runs or DEFAULT_ROOM3_RUNS)]
    report = run_evaluation(
        runs,
        Path(arguments.checkpoint),
        Path(arguments.backbone),
        Path(arguments.labels),
        Path(arguments.dataset),
        Path(arguments.report),
        held_in_arms=arguments.held_in_arms,
        batch_size=arguments.batch_size,
    )
    summary = {
        "room3_pooled": report["room3_pooled"]["metrics"],
        "held_in": report["held_in"]["metrics"],
        "gap_held_in_minus_room3": report["gap_held_in_minus_room3"],
        "content_digest": report["content_digest"],
        "report": str(arguments.report),
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
