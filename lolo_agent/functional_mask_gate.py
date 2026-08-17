"""WP5-final FUNCTIONAL promotion gate (learnings section 4.35 plan-change).

The replication gates are retired: section 4.35 records that requiring the
learned mask to reproduce the assisted mask's bytes conflates "masks
correctly" with "reproduces the assisted mask including its defects".
This gate instead scores FUNCTION -- does tracking built on the learned
masking convention produce correct outcomes -- judged against
DETECTOR-FREE counterfactual ground truth (the corroborated controllable
component per probe arm, extracted by the ``tracker_ood_eval`` machinery
from the paired-probe telemetry), never against the assisted mask's
bytes.  The assisted convention is scored alongside as the incumbent
reference, but ground truth is the referee, not the assisted pipeline.

Ground truth
------------

For each probe corpus the option-search telemetry is regrouped into
counterfactual roots and labeled by
``counterfactual_labels.label_counterfactual_root`` itself (via the
``tracker_ood_eval`` extraction helpers, reused not reimplemented): every
scored arm is a factual endpoint paired with a duration-matched ``NOOP``
control endpoint from the same saved emulator state, whose corroborated
controllable cells are the byte-exact, leave-one-action-out-corroborated
GROUND-TRUTH manipulation locus.  The label rule also certifies, per arm,
exactly which cells did NOT change between the factual and control
endpoints -- the byte-level identity used by the stability and
preservation bits below.

Masking conventions compared
----------------------------

- LEARNED: the pinned pixel-mask reconstruction of ``pixel_mask_head``
  (frozen tracker v4 cell anchor at the 0.5 operating point, dilation one
  cell; head positives at 0.5 inside the anchor; Chebyshev halo 3), with
  the mask pixels and reference slot recovered through the unchanged
  substitution-replay helpers exactly as in the mask-sensitive gate v2.
  An empty reconstruction leaves the frame explicitly unmasked.
- ASSISTED: the recorded convention -- ``PixelHeartGoalPrior``
  ``detect_player`` plus ``player_pixel_mask``, fresh per frame.  A frame
  without a detection is explicitly unmasked.

Each convention masks every frame it evaluates with its own mask for that
frame; no quantity ever mixes conventions.

Preregistered functional bits (per corpus, all must pass)
---------------------------------------------------------

(a) MANIPULATION DETECTION, ground-truth-refereed.  Per deduplicated
    measurement (factual digest, control digest, component cells): a
    convention DETECTS the manipulation when the
    ``object_tracks.world_effect_cells_state_signature`` over the
    component cells differs between the factual and the control endpoint
    under that convention's masks.  The same evidence is lifted through
    ``object_correspondence`` (per-cell signatures and
    ``masked_cell_fingerprint`` values as ``CellEvidence``,
    ``endpoint_relative_state``, ``observations_from_evidence``) so the
    track-state view is derived on every factual/control pair and its
    consistency with the signature view is reported.  Gate: the learned
    convention detects on >= ``AGREEMENT_RATE_THRESHOLD`` (0.95) of ALL
    ground-truth measurements AND on >= 0.95 of the measurements where
    the assisted convention detects.  Both directions are reported
    (assisted-vs-ground-truth and assisted-given-learned), so a learned
    convention that out-detects the assisted one is visible rather than
    penalized.

(b) FINGERPRINT STABILITY under the learned convention
    (self-consistency, not cross-convention equality).  Instances: for
    each root, each scored arm ``i`` and each labeled sibling arm ``j``
    of a different (action, duration) with a non-empty changed-cell set,
    every cell of ``component_i`` minus ``changed_j`` -- cells where the
    label rule certifies the factual and control endpoints of arm ``j``
    are byte-identical while the player is at different positions.
    Deduplicated per (factual digest, control digest, cell).  Any
    feature motion at such a cell is pure masking-convention noise.
    Gate: the learned convention's normalized-L1 feature distance across
    the pair is within ``APPEARANCE_L1_THRESHOLD`` (0.08) on >= 0.95 of
    measurements.  The assisted convention's rate is reported for
    context only.

(c) NO PLAYER-ABSORPTION REGRESSION (the v316/v317 defect class).
    Instances: for each scored arm, every in-grid cell at Chebyshev
    distance one from the ground-truth component that is NOT in the
    arm's changed-cell set -- byte-certified player-free cells directly
    adjacent to where ground truth localizes the player's action.
    Deduplicated per (factual digest, cell).  A convention PRESERVES the
    adjacent appearance when its masked feature stays within the 0.08
    bound of the unmasked feature of the same frame and cell; a mask
    that leaks into the neighbouring cell and erases an adjacent
    object's appearance fails.  Gate: the learned preservation rate is
    at least the assisted preservation rate (both reported).

Every bit additionally requires >= ``MINIMUM_MEASUREMENTS`` (50)
deduplicated measurements per corpus -- perfect agreement over fewer is
vacuous, not a pass (the section 4.31 instrument lesson).

All thresholds are prior published operating points reused by import
(0.5 mask probability, 0.08 appearance L1, 0.95 agreement rate, 50
minimum instances); nothing is tuned against these corpora.

One preregistered run writes a deterministic content-digested report;
verdict PROMOTE-to-shadow (with mask-divergence telemetry) or NO-PROMOTE
with every failing mechanism named.

Usage::

    python -m lolo_agent.functional_mask_gate
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
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .counterfactual_labels import STATUS_LABELED
from .entity_behavior import AnonymousEntityBehaviorModel
from .object_correspondence import (
    CellEvidence,
    endpoint_relative_state,
    observations_from_evidence,
)
from .object_tracks import (
    masked_cell_fingerprint,
    world_effect_cells_state_signature,
)
from .pixels import Frame
from .tracker_ood_eval import (
    RunFrameCache,
    _read_events,
    collect_probe_roots,
    label_probe_roots,
    censor_counts,
    probe_first_step_edges,
    state_frame_index,
)
from .tracker_substitution_replay import (
    APPEARANCE_L1_THRESHOLD,
    DEFAULT_BACKBONE,
    LEARNED_MASK_PROBABILITY_THRESHOLD,
    learned_pixel_mask,
    learned_reference_slot,
    load_replay_tracker,
    mask_divergence,
)
from .mask_sensitive_gate import (
    AGREEMENT_RATE_THRESHOLD,
    DEFAULT_CHECKPOINT as DEFAULT_TRACKER_CHECKPOINT,
    MINIMUM_MATTERING_FRAMES,
)
from .unlabeled_entities import UnlabeledEntityMemory

Cell = Tuple[int, int]
Pixel = Tuple[int, int]

GATE_VERSION = 1
GATE_KIND = "wp5-functional-mask-gate"
_DIGEST_PREFIX = f"{GATE_KIND}:v{GATE_VERSION}:"

# Preregistered constants.  The agreement rate, appearance bound, mask
# operating point, and minimum-measurement count are the prior published
# operating points, reused by import so they cannot drift; the adjacency
# radius is the minimal cell neighbourhood (Chebyshev one) in which the
# documented v316/v317 absorption defect manifests.
MINIMUM_MEASUREMENTS = MINIMUM_MATTERING_FRAMES
ADJACENCY_CHEBYSHEV_RADIUS = 1

LEARNED_CONVENTION = "learned"
ASSISTED_CONVENTION = "assisted"

DEFAULT_CORPORA = (
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v322-room3-paired-probe-arm-a-pushed-d12",
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v323-room3-paired-probe-arm-b-prepush-d12",
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v325-room3-object-removed-probe-d12",
)
DEFAULT_HEAD_CHECKPOINT = "experiments/lolo1-wp5/pixel-mask-head-v1.pt"
DEFAULT_REPORT = "experiments/lolo1-wp5/functional-gate-report.json"

MASK_SOURCE = (
    "reconstructed-pixel-silhouette: frozen tracker v4 cell anchor "
    "(threshold 0.5, dilation 1 cell) + pixel_mask_head positives "
    "(threshold 0.5) + Chebyshev halo dilation 3; mask pixels and "
    "reference slot recovered through the unchanged substitution-replay "
    "helpers at the pinned 0.5 threshold"
)


# ---------------------------------------------------------------------------
# Masking conventions
# ---------------------------------------------------------------------------


class AssistedGoalPriorConvention:
    """The recorded assisted convention: goal-prior detection per frame."""

    name = ASSISTED_CONVENTION

    def __init__(self) -> None:
        from .goal_prior import PixelHeartGoalPrior

        self._prior = PixelHeartGoalPrior()

    def mask(self, frame: Frame) -> Tuple[Optional[Pixel], FrozenSet[Pixel]]:
        slot = self._prior.detect_player(frame)
        if slot is None:
            return None, frozenset()
        pixels = frozenset(
            (int(x), int(y))
            for x, y in self._prior.player_pixel_mask(frame, slot)
        )
        return (int(slot[0]), int(slot[1])), pixels


class LearnedReconstructionConvention:
    """The pinned pixel-mask reconstruction as the learned convention.

    The predictor is ``pixel_mask_head.PixelSilhouettePredictor`` (frozen
    tracker v4 anchor + refinement head + fixed halo); the mask pixels
    and the reference slot are recovered through the unchanged
    substitution-replay helpers at the pinned 0.5 threshold, exactly as
    the mask-sensitive gate v2 substitution did.
    """

    name = LEARNED_CONVENTION

    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor

    def mask(self, frame: Frame) -> Tuple[Optional[Pixel], FrozenSet[Pixel]]:
        prediction = self._predictor.predict(frame)
        pixels = learned_pixel_mask(prediction, frame.width, frame.height)
        slot = learned_reference_slot(prediction, frame.width, frame.height)
        return slot, pixels


class CachedConvention:
    """LRU cache of (slot, pixels) per content-addressed frame digest."""

    def __init__(self, inner: Any, capacity: int = 512) -> None:
        if capacity <= 0:
            raise ValueError("convention cache capacity must be positive")
        self.name = inner.name
        self._inner = inner
        self._capacity = capacity
        self._cache: "OrderedDict[str, Tuple[Optional[Pixel], FrozenSet[Pixel]]]" = (
            OrderedDict()
        )

    def mask(self, frame: Frame) -> Tuple[Optional[Pixel], FrozenSet[Pixel]]:
        cached = self._cache.get(frame.digest)
        if cached is not None:
            self._cache.move_to_end(frame.digest)
            return cached
        result = self._inner.mask(frame)
        self._cache[frame.digest] = result
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return result


# ---------------------------------------------------------------------------
# Per-frame convention quantities (object_tracks pure functions)
# ---------------------------------------------------------------------------


def convention_feature(
    frame: Frame,
    cell: Cell,
    slot: Optional[Pixel],
    pixels: FrozenSet[Pixel],
    memory: UnlabeledEntityMemory,
) -> Tuple[int, ...]:
    """One cell's pooled appearance under one convention's mask.

    Mirrors the established convention exactly: the mask participates
    only when the convention produced an anchor slot; ``slot=None`` is
    the explicitly unmasked computation.
    """

    ignored = pixels if slot is not None else None
    return tuple(memory.feature_at(frame, *cell, ignored))


@dataclass(frozen=True)
class ConventionCellQuantities:
    """Downstream tracking quantities of one frame under one convention."""

    joint_signature: str
    cell_signatures: Mapping[Cell, str]
    fingerprints: Mapping[Cell, str]
    features: Mapping[Cell, Tuple[int, ...]]


def convention_cell_quantities(
    frame: Frame,
    cells: Sequence[Cell],
    slot: Optional[Pixel],
    pixels: FrozenSet[Pixel],
    memory: UnlabeledEntityMemory,
) -> ConventionCellQuantities:
    """Signatures, fingerprints and features at cells under one mask.

    A fresh behaviour model is used per computation so the stateful
    ``classify`` side channel cannot leak between frames (the
    mask-sensitive gate's own precaution, kept).
    """

    model = AnonymousEntityBehaviorModel()

    def mask_callable(_frame: Frame, _slot: Pixel) -> FrozenSet[Pixel]:
        return pixels

    ordered = tuple(sorted(cells))
    joint = world_effect_cells_state_signature(
        frame, ordered, slot, memory, mask_callable
    )
    cell_signatures: Dict[Cell, str] = {}
    fingerprints: Dict[Cell, str] = {}
    features: Dict[Cell, Tuple[int, ...]] = {}
    for cell in ordered:
        cell_signatures[cell] = world_effect_cells_state_signature(
            frame, (cell,), slot, memory, mask_callable
        )
        fingerprint, _type_id = masked_cell_fingerprint(
            frame, cell, slot, memory, model, mask_callable
        )
        fingerprints[cell] = fingerprint
        features[cell] = convention_feature(frame, cell, slot, pixels, memory)
    return ConventionCellQuantities(
        joint_signature=joint,
        cell_signatures=cell_signatures,
        fingerprints=fingerprints,
        features=features,
    )


# ---------------------------------------------------------------------------
# Bit (a): ground-truth-refereed manipulation detection
# ---------------------------------------------------------------------------


def score_detection(
    factual: Frame,
    control: Frame,
    component_cells: Sequence[Cell],
    convention: Any,
    memory: UnlabeledEntityMemory,
) -> Dict[str, Any]:
    """Detect the ground-truth manipulation on one factual/control pair.

    Detection is the planner's own world-state comparison: the joint
    ``world_effect_cells_state_signature`` over the component cells
    differs between the two endpoints under this convention's per-frame
    masks.  The same evidence is lifted through ``object_correspondence``
    into an endpoint-relative track state whose non-empty
    ``current_cells`` view must agree with the signature view modulo
    hash collision; the consistency flag is reported.
    """

    slot_f, pixels_f = convention.mask(factual)
    slot_c, pixels_c = convention.mask(control)
    factual_quantities = convention_cell_quantities(
        factual, component_cells, slot_f, pixels_f, memory
    )
    control_quantities = convention_cell_quantities(
        control, component_cells, slot_c, pixels_c, memory
    )
    evidence = tuple(
        CellEvidence(
            cell=cell,
            appearance_signature=factual_quantities.cell_signatures[cell],
            baseline_signature=control_quantities.cell_signatures[cell],
            appearance_fingerprint=factual_quantities.fingerprints[cell],
            frame_index=0,
        )
        for cell in sorted(component_cells)
    )
    state = endpoint_relative_state(evidence)
    observations = observations_from_evidence(evidence)
    detected = bool(
        factual_quantities.joint_signature
        != control_quantities.joint_signature
    )
    return {
        "detected": detected,
        "current_cells": len(state.current_cells),
        "observations": len(observations),
        "track_state_signature": state.signature,
        "views_consistent": bool(
            (len(state.current_cells) > 0) == detected
        ),
        "factual_masked": bool(slot_f is not None and pixels_f),
        "control_masked": bool(slot_c is not None and pixels_c),
    }


# ---------------------------------------------------------------------------
# Bit (b): fingerprint stability; bit (c): absorption preservation
# ---------------------------------------------------------------------------


def stability_l1(
    first: Frame,
    second: Frame,
    cell: Cell,
    convention: Any,
    memory: UnlabeledEntityMemory,
) -> float:
    """Feature motion at a byte-identical cell across two frames."""

    slot_a, pixels_a = convention.mask(first)
    slot_b, pixels_b = convention.mask(second)
    return UnlabeledEntityMemory.feature_distance(
        convention_feature(first, cell, slot_a, pixels_a, memory),
        convention_feature(second, cell, slot_b, pixels_b, memory),
    )


def preservation_l1(
    frame: Frame,
    cell: Cell,
    convention: Any,
    memory: UnlabeledEntityMemory,
) -> float:
    """Masked-versus-unmasked feature distance at one player-free cell."""

    slot, pixels = convention.mask(frame)
    return UnlabeledEntityMemory.feature_distance(
        convention_feature(frame, cell, slot, pixels, memory),
        tuple(memory.feature_at(frame, *cell, None)),
    )


# ---------------------------------------------------------------------------
# Ground-truth measurement enumeration from label records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabeledArm:
    """One labeled arm of one root, as read back from a label record."""

    group: int
    action: str
    duration: int
    factual_digest: str
    control_digest: str
    component_cells: Tuple[Cell, ...]
    changed_cells: Tuple[Cell, ...]

    @property
    def scored(self) -> bool:
        return bool(self.component_cells)


def _cells(payload: Sequence[Sequence[int]]) -> Tuple[Cell, ...]:
    return tuple(sorted((int(cell[0]), int(cell[1])) for cell in payload))


def labeled_arms_from_record(
    record: Mapping[str, Any]
) -> Tuple[LabeledArm, ...]:
    """Every labeled arm of one record (scored arms have components)."""

    arms: List[LabeledArm] = []
    for arm in record["arms"]:
        if arm["status"] != STATUS_LABELED:
            continue
        arms.append(
            LabeledArm(
                group=int(record["group"]),
                action=str(arm["action"]),
                duration=int(arm["duration"]),
                factual_digest=str(arm["endpoint_digests"][0]),
                control_digest=str(arm["control_digest"]),
                component_cells=_cells(arm["controllable_cells"]),
                changed_cells=_cells(arm["changed_cells"]),
            )
        )
    return tuple(arms)


DetectionKey = Tuple[str, str, Tuple[Cell, ...]]
StabilityKey = Tuple[str, str, Cell]
PreservationKey = Tuple[str, Cell]


def detection_measurements(
    records: Sequence[Mapping[str, Any]]
) -> "OrderedDict[DetectionKey, int]":
    """Deduplicated (factual, control, component) measurements, counted."""

    measurements: Dict[DetectionKey, int] = {}
    for record in records:
        for arm in labeled_arms_from_record(record):
            if not arm.scored:
                continue
            key = (
                arm.factual_digest,
                arm.control_digest,
                arm.component_cells,
            )
            measurements[key] = measurements.get(key, 0) + 1
    return OrderedDict(
        (key, measurements[key]) for key in sorted(measurements)
    )


def stability_measurements(
    records: Sequence[Mapping[str, Any]]
) -> "OrderedDict[StabilityKey, int]":
    """Deduplicated byte-certified stability cells, counted.

    For scored arm ``i`` and labeled sibling ``j`` of a different
    (action, duration) with a non-empty changed set, every cell of
    ``component_i`` minus ``changed_j`` is byte-identical between arm
    ``j``'s factual and control endpoints by the label rule itself.
    """

    measurements: Dict[StabilityKey, int] = {}
    for record in records:
        arms = labeled_arms_from_record(record)
        for i in arms:
            if not i.scored:
                continue
            for j in arms:
                if (j.action, j.duration) == (i.action, i.duration):
                    continue
                if not j.changed_cells:
                    continue
                stable_cells = set(i.component_cells) - set(j.changed_cells)
                for cell in stable_cells:
                    key = (j.factual_digest, j.control_digest, cell)
                    measurements[key] = measurements.get(key, 0) + 1
    return OrderedDict(
        (key, measurements[key]) for key in sorted(measurements)
    )


def preservation_measurements(
    records: Sequence[Mapping[str, Any]]
) -> "OrderedDict[PreservationKey, int]":
    """Deduplicated player-free cells adjacent to the ground-truth locus.

    Chebyshev-1 neighbours of the component that are outside the arm's
    changed-cell set are byte-certified to contain no player pixels at
    the factual endpoint (any sprite spill would have differed from the
    control endpoint, landing the cell in the changed set).
    """

    measurements: Dict[PreservationKey, int] = {}
    for record in records:
        columns = int(record["columns"])
        rows = int(record["rows"])
        for arm in labeled_arms_from_record(record):
            if not arm.scored:
                continue
            changed = set(arm.changed_cells)
            adjacent: Set[Cell] = set()
            for column, row in arm.component_cells:
                for dx in range(-ADJACENCY_CHEBYSHEV_RADIUS, ADJACENCY_CHEBYSHEV_RADIUS + 1):
                    for dy in range(-ADJACENCY_CHEBYSHEV_RADIUS, ADJACENCY_CHEBYSHEV_RADIUS + 1):
                        neighbour = (column + dx, row + dy)
                        if (
                            0 <= neighbour[0] < columns
                            and 0 <= neighbour[1] < rows
                            and neighbour not in changed
                        ):
                            adjacent.add(neighbour)
            for cell in adjacent:
                key = (arm.factual_digest, cell)
                measurements[key] = measurements.get(key, 0) + 1
    return OrderedDict(
        (key, measurements[key]) for key in sorted(measurements)
    )


# ---------------------------------------------------------------------------
# Rates, per-corpus gate bits
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else numerator / denominator


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile of empty sequence")
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _distribution(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "p75": _quantile(ordered, 0.75),
        "max": ordered[-1],
    }


def detection_bit(
    rows: Sequence[Mapping[str, Any]],
    *,
    agreement_rate_threshold: float = AGREEMENT_RATE_THRESHOLD,
    minimum_measurements: int = MINIMUM_MEASUREMENTS,
) -> Dict[str, Any]:
    """Bit (a): both gated conditions plus both reported directions."""

    total = len(rows)
    learned = [row[LEARNED_CONVENTION]["detected"] for row in rows]
    assisted = [row[ASSISTED_CONVENTION]["detected"] for row in rows]
    learned_count = sum(learned)
    assisted_count = sum(assisted)
    learned_given_assisted = _rate(
        sum(l for l, a in zip(learned, assisted) if a), assisted_count
    )
    assisted_given_learned = _rate(
        sum(a for l, a in zip(learned, assisted) if l), learned_count
    )
    learned_gt_rate = _rate(learned_count, total)
    sufficient = total >= minimum_measurements
    gt_condition = bool(
        learned_gt_rate is not None
        and learned_gt_rate >= agreement_rate_threshold
    )
    # If the assisted convention detects nothing, the assisted-conditioned
    # requirement is vacuous (flagged): ground truth stays the referee.
    assisted_condition = bool(
        assisted_count == 0
        or (
            learned_given_assisted is not None
            and learned_given_assisted >= agreement_rate_threshold
        )
    )
    return {
        "measurements": total,
        "measurements_sufficient": sufficient,
        "minimum_measurements": minimum_measurements,
        "agreement_rate_threshold": agreement_rate_threshold,
        "learned_detected": learned_count,
        "assisted_detected": assisted_count,
        "learned_rate_vs_ground_truth": learned_gt_rate,
        "assisted_rate_vs_ground_truth": _rate(assisted_count, total),
        "learned_rate_given_assisted": learned_given_assisted,
        "assisted_rate_given_learned": assisted_given_learned,
        "assisted_condition_vacuous": bool(assisted_count == 0),
        "gt_condition": gt_condition,
        "assisted_condition": assisted_condition,
        "passed": bool(sufficient and gt_condition and assisted_condition),
    }


def stability_bit(
    learned_within: Sequence[bool],
    assisted_within: Sequence[bool],
    *,
    agreement_rate_threshold: float = AGREEMENT_RATE_THRESHOLD,
    minimum_measurements: int = MINIMUM_MEASUREMENTS,
) -> Dict[str, Any]:
    """Bit (b): learned self-consistency rate; assisted reported only."""

    total = len(learned_within)
    learned_rate = _rate(sum(learned_within), total)
    sufficient = total >= minimum_measurements
    rate_condition = bool(
        learned_rate is not None
        and learned_rate >= agreement_rate_threshold
    )
    return {
        "measurements": total,
        "measurements_sufficient": sufficient,
        "minimum_measurements": minimum_measurements,
        "agreement_rate_threshold": agreement_rate_threshold,
        "appearance_l1_threshold": APPEARANCE_L1_THRESHOLD,
        "learned_stability_rate": learned_rate,
        "assisted_stability_rate": _rate(sum(assisted_within), total),
        "passed": bool(sufficient and rate_condition),
    }


def preservation_bit(
    learned_preserved: Sequence[bool],
    assisted_preserved: Sequence[bool],
    *,
    minimum_measurements: int = MINIMUM_MEASUREMENTS,
) -> Dict[str, Any]:
    """Bit (c): learned preservation must not regress below assisted."""

    if len(learned_preserved) != len(assisted_preserved):
        raise ValueError("preservation flag sequences must be paired")
    total = len(learned_preserved)
    learned_rate = _rate(sum(learned_preserved), total)
    assisted_rate = _rate(sum(assisted_preserved), total)
    sufficient = total >= minimum_measurements
    no_regression = bool(
        learned_rate is not None
        and assisted_rate is not None
        and learned_rate >= assisted_rate
    )
    return {
        "measurements": total,
        "measurements_sufficient": sufficient,
        "minimum_measurements": minimum_measurements,
        "appearance_l1_threshold": APPEARANCE_L1_THRESHOLD,
        "learned_preservation_rate": learned_rate,
        "assisted_preservation_rate": assisted_rate,
        "passed": bool(sufficient and no_regression),
    }


# ---------------------------------------------------------------------------
# Corpus scoring
# ---------------------------------------------------------------------------


def score_corpus(
    run_dir: Path,
    learned_convention: Any,
    assisted_convention: Any,
    *,
    appearance_threshold: float = APPEARANCE_L1_THRESHOLD,
    frame_cache_capacity: int = 1024,
) -> Dict[str, Any]:
    """Score one probe corpus on all three preregistered functional bits.

    Ground truth comes from the corpus's own telemetry through the
    ``tracker_ood_eval`` extraction helpers and the counterfactual label
    rule itself; every enumeration is sorted so a rerun is byte-identical.
    """

    if learned_convention.name != LEARNED_CONVENTION:
        raise ValueError("the learned convention must be named 'learned'")
    if assisted_convention.name != ASSISTED_CONVENTION:
        raise ValueError("the assisted convention must be named 'assisted'")
    run_dir = Path(run_dir)
    events = _read_events(run_dir / "events.jsonl")
    edges = probe_first_step_edges(events)
    state_frames = state_frame_index(events)
    roots = collect_probe_roots(run_dir.name, edges, state_frames)
    cache = RunFrameCache(run_dir / "frames", capacity=frame_cache_capacity)
    records, root_stats = label_probe_roots(roots, cache.get)
    memory = UnlabeledEntityMemory()
    for record in records:
        if (int(record["columns"]), int(record["rows"])) != (
            memory.columns,
            memory.rows,
        ):
            raise ValueError("label grid does not match the entity grid")
    labeled_arms = sum(
        len(labeled_arms_from_record(record)) for record in records
    )
    scored_arms = sum(
        sum(arm.scored for arm in labeled_arms_from_record(record))
        for record in records
    )
    conventions = (
        (LEARNED_CONVENTION, learned_convention),
        (ASSISTED_CONVENTION, assisted_convention),
    )

    # Bit (a): detection on every deduplicated ground-truth pair.
    detection_keys = detection_measurements(records)
    detection_rows: List[Dict[str, Any]] = []
    for (factual_digest, control_digest, component), count in detection_keys.items():
        factual = cache.get(factual_digest)
        control = cache.get(control_digest)
        row: Dict[str, Any] = {
            "factual": factual_digest,
            "control": control_digest,
            "component_cells": len(component),
            "arms": count,
        }
        for name, convention in conventions:
            row[name] = score_detection(
                factual, control, component, convention, memory
            )
        detection_rows.append(row)

    # Feature memo for the cell-level bits (pure lookup, no result change).
    feature_memo: Dict[Tuple[str, Cell, str], Tuple[int, ...]] = {}

    def cached_feature(
        frame: Frame, cell: Cell, convention: Any
    ) -> Tuple[int, ...]:
        key = (frame.digest, cell, convention.name)
        cached = feature_memo.get(key)
        if cached is None:
            slot, pixels = convention.mask(frame)
            cached = convention_feature(frame, cell, slot, pixels, memory)
            feature_memo[key] = cached
        return cached

    def cached_unmasked(frame: Frame, cell: Cell) -> Tuple[int, ...]:
        key = (frame.digest, cell, "unmasked")
        cached = feature_memo.get(key)
        if cached is None:
            cached = tuple(memory.feature_at(frame, *cell, None))
            feature_memo[key] = cached
        return cached

    # Bit (b): stability across byte-certified identical cells.
    stability_keys = stability_measurements(records)
    stability_values: Dict[str, List[float]] = {
        name: [] for name, _convention in conventions
    }
    for (factual_digest, control_digest, cell), _count in stability_keys.items():
        factual = cache.get(factual_digest)
        control = cache.get(control_digest)
        for name, convention in conventions:
            stability_values[name].append(
                UnlabeledEntityMemory.feature_distance(
                    cached_feature(factual, cell, convention),
                    cached_feature(control, cell, convention),
                )
            )
    stability_within = {
        name: [value <= appearance_threshold for value in values]
        for name, values in stability_values.items()
    }

    # Bit (c): preservation at player-free adjacent cells.
    preservation_keys = preservation_measurements(records)
    preservation_values: Dict[str, List[float]] = {
        name: [] for name, _convention in conventions
    }
    for (factual_digest, cell), _count in preservation_keys.items():
        factual = cache.get(factual_digest)
        unmasked = cached_unmasked(factual, cell)
        for name, convention in conventions:
            preservation_values[name].append(
                UnlabeledEntityMemory.feature_distance(
                    cached_feature(factual, cell, convention), unmasked
                )
            )
    preservation_within = {
        name: [value <= appearance_threshold for value in values]
        for name, values in preservation_values.items()
    }

    # Divergence telemetry over the unique factual endpoint frames.
    factual_digests = sorted(
        {key[0] for key in detection_keys}
    )
    divergence_rows: List[Dict[str, Any]] = []
    for digest in factual_digests:
        frame = cache.get(digest)
        _slot_l, learned_pixels = learned_convention.mask(frame)
        _slot_a, assisted_pixels = assisted_convention.mask(frame)
        divergence_rows.append(
            mask_divergence(learned_pixels, assisted_pixels)
        )
    divergence_ious = [
        row["iou"] for row in divergence_rows if row["iou"] is not None
    ]

    gate = {
        "bit_a_detection": detection_bit(detection_rows),
        "bit_b_stability": stability_bit(
            stability_within[LEARNED_CONVENTION],
            stability_within[ASSISTED_CONVENTION],
        ),
        "bit_c_preservation": preservation_bit(
            preservation_within[LEARNED_CONVENTION],
            preservation_within[ASSISTED_CONVENTION],
        ),
    }
    gate["passed"] = bool(all(bit["passed"] for bit in gate.values()))
    return {
        "run_id": run_dir.name,
        "extraction": {
            "events": len(events),
            "probe_edges": len(edges),
            **root_stats,
            "censored_by_reason": censor_counts(records),
            "labeled_arms": labeled_arms,
            "scored_arms": scored_arms,
        },
        "detection": {
            "measurements": len(detection_rows),
            "duplicate_arms": sum(detection_keys.values())
            - len(detection_keys),
            "rows": [
                {
                    "factual": row["factual"],
                    "control": row["control"],
                    "component_cells": row["component_cells"],
                    "arms": row["arms"],
                    "learned_detected": row[LEARNED_CONVENTION]["detected"],
                    "assisted_detected": row[ASSISTED_CONVENTION]["detected"],
                }
                for row in detection_rows
            ],
            "views_consistent": {
                name: sum(
                    row[name]["views_consistent"] for row in detection_rows
                )
                for name, _convention in conventions
            },
            "unmasked_frames": {
                name: {
                    "factual": sum(
                        not row[name]["factual_masked"]
                        for row in detection_rows
                    ),
                    "control": sum(
                        not row[name]["control_masked"]
                        for row in detection_rows
                    ),
                }
                for name, _convention in conventions
            },
            "track_state": {
                name: {
                    "current_cells_mean": (
                        sum(
                            row[name]["current_cells"]
                            for row in detection_rows
                        )
                        / len(detection_rows)
                        if detection_rows
                        else None
                    ),
                    "observations_mean": (
                        sum(
                            row[name]["observations"]
                            for row in detection_rows
                        )
                        / len(detection_rows)
                        if detection_rows
                        else None
                    ),
                }
                for name, _convention in conventions
            },
        },
        "stability": {
            "measurements": len(stability_keys),
            "duplicate_instances": sum(stability_keys.values())
            - len(stability_keys),
            "l1": {
                name: _distribution(values)
                for name, values in stability_values.items()
            },
        },
        "preservation": {
            "measurements": len(preservation_keys),
            "duplicate_instances": sum(preservation_keys.values())
            - len(preservation_keys),
            "l1": {
                name: _distribution(values)
                for name, values in preservation_values.items()
            },
        },
        "divergence_telemetry": {
            "frames": len(divergence_rows),
            "mask_iou": _distribution(divergence_ious),
            "empty_learned_masks": sum(
                row["learned_pixels"] == 0 for row in divergence_rows
            ),
            "empty_assisted_masks": sum(
                row["assisted_pixels"] == 0 for row in divergence_rows
            ),
            "learned_pixels": _distribution(
                [float(row["learned_pixels"]) for row in divergence_rows]
            ),
            "assisted_pixels": _distribution(
                [float(row["assisted_pixels"]) for row in divergence_rows]
            ),
        },
        "gate": gate,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


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
        key: value
        for key, value in payload.items()
        if key != "content_digest"
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(
        (_DIGEST_PREFIX + canonical).encode("utf-8")
    ).hexdigest()


def _failing_mechanisms(result: Mapping[str, Any]) -> List[str]:
    """Name every failing mechanism of one corpus result."""

    mechanisms: List[str] = []
    run_id = result["run_id"]
    gate = result["gate"]
    bit_a = gate["bit_a_detection"]
    if not bit_a["passed"]:
        if not bit_a["measurements_sufficient"]:
            mechanisms.append(
                f"{run_id}: bit-a insufficient measurements "
                f"({bit_a['measurements']} < {bit_a['minimum_measurements']})"
            )
        if bit_a["measurements_sufficient"] and not bit_a["gt_condition"]:
            mechanisms.append(
                f"{run_id}: bit-a learned convention misses ground-truth "
                f"manipulations (rate {bit_a['learned_rate_vs_ground_truth']} "
                f"< {bit_a['agreement_rate_threshold']})"
            )
        if bit_a["measurements_sufficient"] and not bit_a["assisted_condition"]:
            mechanisms.append(
                f"{run_id}: bit-a learned convention misses manipulations "
                "the assisted convention detects (rate "
                f"{bit_a['learned_rate_given_assisted']} "
                f"< {bit_a['agreement_rate_threshold']})"
            )
    bit_b = gate["bit_b_stability"]
    if not bit_b["passed"]:
        if not bit_b["measurements_sufficient"]:
            mechanisms.append(
                f"{run_id}: bit-b insufficient measurements "
                f"({bit_b['measurements']} < {bit_b['minimum_measurements']})"
            )
        else:
            mechanisms.append(
                f"{run_id}: bit-b learned-convention fingerprints unstable "
                "across byte-identical cells (rate "
                f"{bit_b['learned_stability_rate']} "
                f"< {bit_b['agreement_rate_threshold']})"
            )
    bit_c = gate["bit_c_preservation"]
    if not bit_c["passed"]:
        if not bit_c["measurements_sufficient"]:
            mechanisms.append(
                f"{run_id}: bit-c insufficient measurements "
                f"({bit_c['measurements']} < {bit_c['minimum_measurements']})"
            )
        else:
            mechanisms.append(
                f"{run_id}: bit-c player-absorption regression -- learned "
                "preservation rate "
                f"{bit_c['learned_preservation_rate']} below assisted "
                f"{bit_c['assisted_preservation_rate']}"
            )
    return mechanisms


def build_report(
    corpus_results: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble the deterministic content-digested gate report."""

    per_corpus = {
        result["run_id"]: dict(result["gate"]) for result in corpus_results
    }
    passed = bool(corpus_results) and all(
        gate["passed"] for gate in per_corpus.values()
    )
    mechanisms: List[str] = []
    for result in corpus_results:
        mechanisms.extend(_failing_mechanisms(result))
    if not corpus_results:
        mechanisms.append("no corpora scored")
    report = {
        "version": GATE_VERSION,
        "kind": GATE_KIND,
        "preregistration": "docs/wp5-tracker-training-2026-08-16.md",
        "basis": "docs/learnings.md section 4.35 plan-change",
        "design_principle": (
            "functional promotion: tracking outcomes under the learned "
            "masking convention are judged against detector-free "
            "counterfactual ground truth, never against the assisted "
            "mask's bytes; replication gates retired per section 4.35"
        ),
        "provenance": dict(provenance),
        "thresholds": {
            "learned_mask_probability": LEARNED_MASK_PROBABILITY_THRESHOLD,
            "appearance_l1": APPEARANCE_L1_THRESHOLD,
            "agreement_rate": AGREEMENT_RATE_THRESHOLD,
            "minimum_measurements": MINIMUM_MEASUREMENTS,
            "adjacency_chebyshev_radius": ADJACENCY_CHEBYSHEV_RADIUS,
            "rationale": (
                "all prior published operating points reused by import; "
                "nothing tuned against these corpora"
            ),
        },
        "corpora": [dict(result) for result in corpus_results],
        "result": {
            "per_corpus": per_corpus,
            "gate": "PASS" if passed else "FAIL",
            "failing_mechanisms": mechanisms,
            "verdict": (
                "PROMOTE-to-shadow (learned masking convention with mask-"
                "divergence telemetry; explicitly gated convention change "
                "per section 4.35, claim boundary unmoved)"
                if passed
                else "NO-PROMOTE (functional bits failed: "
                + "; ".join(mechanisms)
                + ")"
            ),
        },
    }
    report = _round_floats(report)
    report["content_digest"] = content_digest(report)
    return report


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_conventions(
    tracker_checkpoint: Path,
    backbone_path: Path,
    head_checkpoint: Path,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load pinned artifacts, verify their digests, compose conventions."""

    from .pixel_mask_head import (
        PixelSilhouettePredictor,
        load_pixel_mask_head_checkpoint,
    )

    tracker, replay_provenance = load_replay_tracker(
        Path(tracker_checkpoint), Path(backbone_path)
    )
    head, head_provenance = load_pixel_mask_head_checkpoint(
        Path(head_checkpoint), device="cpu", frozen=True
    )
    if head_provenance["tracker_parameter_sha256"] != tracker.checkpoint_digest:
        raise ValueError(
            "pixel head was trained against a different tracker checkpoint"
        )
    if (
        head_provenance["backbone_parameter_sha256"]
        != replay_provenance["backbone_parameter_sha256"]
    ):
        raise ValueError(
            "pixel head was trained against a different spatial backbone"
        )
    if (
        head_provenance["cell_columns"],
        head_provenance["cell_rows"],
    ) != (tracker.columns, tracker.rows):
        raise ValueError("pixel head cell grid does not match the tracker grid")
    predictor = PixelSilhouettePredictor(tracker, head, device="cpu")
    learned = CachedConvention(LearnedReconstructionConvention(predictor))
    assisted = CachedConvention(AssistedGoalPriorConvention())
    provenance = dict(replay_provenance)
    provenance.update(
        {
            "pixel_mask_head_checkpoint": str(head_checkpoint),
            "pixel_mask_head_parameter_sha256": head.checkpoint_digest,
            "pixel_mask_head_label_manifest_sha256": head_provenance[
                "label_manifest_sha256"
            ],
            "pixel_mask_head_pixel_targets_sha256": head_provenance[
                "pixel_targets_sha256"
            ],
            "mask_source": MASK_SOURCE,
            "ground_truth": (
                "detector-free counterfactual components: factual versus "
                "duration-matched NOOP endpoints from shared saved states, "
                "labeled by counterfactual_labels.label_counterfactual_root "
                "via the tracker_ood_eval extraction helpers"
            ),
        }
    )
    return learned, assisted, provenance


def run_gate(
    corpus_dirs: Sequence[Path],
    tracker_checkpoint: Path,
    backbone_path: Path,
    head_checkpoint: Path,
    report_path: Path,
) -> Dict[str, Any]:
    learned, assisted, provenance = build_conventions(
        tracker_checkpoint, backbone_path, head_checkpoint
    )
    results = [
        score_corpus(Path(run_dir), learned, assisted)
        for run_dir in corpus_dirs
    ]
    report = build_report(results, provenance)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "WP5-final functional promotion gate: tracking outcomes under "
            "the learned masking convention judged against detector-free "
            "counterfactual ground truth (learnings section 4.35)"
        )
    )
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpora",
        default=None,
        help="probe corpus run directory (repeatable)",
    )
    parser.add_argument(
        "--tracker-checkpoint", default=DEFAULT_TRACKER_CHECKPOINT
    )
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--head-checkpoint", default=DEFAULT_HEAD_CHECKPOINT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    corpora = [
        Path(path) for path in (arguments.corpora or DEFAULT_CORPORA)
    ]
    report = run_gate(
        corpora,
        Path(arguments.tracker_checkpoint),
        Path(arguments.backbone),
        Path(arguments.head_checkpoint),
        Path(arguments.report),
    )
    summary = {
        "gate": report["result"]["gate"],
        "verdict": report["result"]["verdict"],
        "failing_mechanisms": report["result"]["failing_mechanisms"],
        "per_corpus": {
            run_id: {
                "bit_a": gate["bit_a_detection"]["passed"],
                "bit_b": gate["bit_b_stability"]["passed"],
                "bit_c": gate["bit_c_preservation"]["passed"],
                "passed": gate["passed"],
            }
            for run_id, gate in report["result"]["per_corpus"].items()
        },
        "content_digest": report["content_digest"],
        "report": str(arguments.report),
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
