"""Counterfactual controllable-region pseudo-labels from strict branch groups.

Direction-review 2026-08-16 Amendment B phase 1 (roadmap section 17 item 7):
the mechanized WP5 label generator.  It walks a strict-bound
``SequenceStore`` and derives per-cell pseudo-labels purely from
counterfactual branch structure -- factual action endpoints compared
against duration-matched ``NOOP`` control endpoints recorded from the same
causal root.  No supplied goal prior, player detector, or any other
semantic anchor participates anywhere in the label path.

Method
------

Every stored sequence contributes its first recorded edge: the shared root
frame, the first action with its duration, and the endpoint frame that
action reached.  Later steps of multi-step sequences start from
arm-specific states and are not counterfactually paired, so they are
ignored.  Edges sharing ``(source_run_id, group, root frame digest)`` are
sibling arms of one causal root.  For each factual (non-``NOOP``) arm with
a duration-matched ``NOOP`` sibling:

1. The counterfactual changed-cell set is the exact per-cell pixel
   difference between the factual endpoint and the control endpoint on a
   coarse grid (default 16 x 15, matching the anonymous entity grid).  The
   comparison is endpoint-relative by construction; no accumulated history
   participates (learnings section 4.29).
2. Controllable-region mask candidates are the 4-connected components of
   that changed-cell set that survive leave-one-action-out corroboration:
   a component qualifies only if it intersects the intersection of the
   changed-cell sets of every corroborating sibling arm, where
   corroborating siblings are eligible arms of a *different* primitive
   action with a non-empty changed-cell set.  Arms sharing an action never
   corroborate each other, so no action can certify its own idiosyncratic
   effect.  Siblings with an empty changed-cell set abstain: a blocked
   action carries no localization evidence.
3. The non-controllable residual-change mask is the changed-cell set minus
   the controllable candidate cells.

Censoring is explicit.  A factual arm is emitted censored, with empty
masks, when its control is absent (``absent_control``: no duration-matched
``NOOP`` endpoint at the root), when its own endpoint is ambiguous
(``ambiguous_endpoint``: the same action and duration reached distinct
endpoints from the same root), when the matched control is ambiguous
(``ambiguous_control``), or when no corroborating sibling action exists
(``no_sibling_corroboration``).  Roots with no controlled factual arm at
all are counted but produce no output record.

Known phase-1 limitations, accepted for pseudo-labels: a changed region
adjacent to the controllable region merges into its connected component,
and status regions that respond to every action (for example step
counters) satisfy the consistency rule.  Both surface as extra candidate
components for downstream training to discount.

Output format (documented contract)
-----------------------------------

``generate_labels`` / ``generate_store_labels`` produce
``CounterfactualRootLabels`` values; ``write_labels`` persists them as
JSONL, one canonical JSON object per causal root, ordered by
``(source_run_id, group, root_digest)`` with arms ordered by
``(action, duration)``.  Every line is the canonical serialization
(``sort_keys=True``, compact separators) of::

    {
      "version": 1,
      "source_run_id": str,
      "group": int,
      "root_digest": str,           # content digest of the shared root frame
      "columns": int,
      "rows": int,                  # coarse label grid geometry
      "arms": [
        {
          "action": str,            # primitive action value, never "noop"
          "duration": int,
          "endpoint_digests": [str, ...],   # distinct observed endpoints
          "control_digest": str | null,     # duration-matched NOOP endpoint
          "status": "labeled" | "censored",
          "censor_reason": str | null,
          "corroborating_arms": int,
          "changed_cells": [[column, row], ...],
          "controllable_components": [[[column, row], ...], ...],
          "controllable_cells": [[column, row], ...],
          "residual_cells": [[column, row], ...]
        },
        ...
      ],
      "content_digest": str         # sha256 over the record minus this field
    }

All cell lists are sorted, all digests are deterministic functions of the
serialized content, and regenerating from the same store yields
byte-identical output.  ``write_labels`` also writes a
``<name>.manifest.json`` sidecar carrying aggregate counts, the dataset
reward track, and a digest over all record digests.

Smoke usage (read-only against the strict store)::

    python -m lolo_agent.counterfactual_labels \
        --dataset experiments/lolo1-medium/dataset --maximum-roots 8
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .environment import Action
from .pixels import Frame
from .sequence_store import SequenceStore

Cell = Tuple[int, int]

LABEL_VERSION = 1
DEFAULT_COLUMNS = 16
DEFAULT_ROWS = 15
DEFAULT_DURATION = 4

STATUS_LABELED = "labeled"
STATUS_CENSORED = "censored"
CENSOR_ABSENT_CONTROL = "absent_control"
CENSOR_AMBIGUOUS_ENDPOINT = "ambiguous_endpoint"
CENSOR_AMBIGUOUS_CONTROL = "ambiguous_control"
CENSOR_NO_SIBLING_CORROBORATION = "no_sibling_corroboration"

_DIGEST_PREFIX = b"lolo-counterfactual-labels-v1:"

# (source_run_id, group, root digest, action, duration, endpoint digest)
_FirstStepEdge = Tuple[str, int, str, Action, int, str]


@dataclass(frozen=True)
class CounterfactualArm:
    """One (action, duration) branch recorded from a shared causal root."""

    action: Action
    duration: int
    endpoint_digests: Tuple[str, ...]

    @property
    def ambiguous(self) -> bool:
        return len(self.endpoint_digests) != 1


@dataclass(frozen=True)
class CounterfactualRoot:
    """All sibling first-step branches recorded from one root frame."""

    source_run_id: str
    group: int
    root_digest: str
    arms: Tuple[CounterfactualArm, ...]

    @property
    def sort_key(self) -> Tuple[str, int, str]:
        return (self.source_run_id, self.group, self.root_digest)


@dataclass(frozen=True)
class CounterfactualArmLabel:
    """Pseudo-label (or explicit censoring) for one factual arm."""

    action: Action
    duration: int
    endpoint_digests: Tuple[str, ...]
    control_digest: Optional[str]
    status: str
    censor_reason: Optional[str]
    corroborating_arms: int
    changed_cells: Tuple[Cell, ...]
    controllable_components: Tuple[Tuple[Cell, ...], ...]
    controllable_cells: Tuple[Cell, ...]
    residual_cells: Tuple[Cell, ...]


@dataclass(frozen=True)
class CounterfactualRootLabels:
    """Deterministic per-root label record; see the module docstring."""

    source_run_id: str
    group: int
    root_digest: str
    columns: int
    rows: int
    arms: Tuple[CounterfactualArmLabel, ...]

    def payload(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "version": LABEL_VERSION,
            "source_run_id": self.source_run_id,
            "group": self.group,
            "root_digest": self.root_digest,
            "columns": self.columns,
            "rows": self.rows,
            "arms": [
                {
                    "action": arm.action.value,
                    "duration": arm.duration,
                    "endpoint_digests": list(arm.endpoint_digests),
                    "control_digest": arm.control_digest,
                    "status": arm.status,
                    "censor_reason": arm.censor_reason,
                    "corroborating_arms": arm.corroborating_arms,
                    "changed_cells": _cells_payload(arm.changed_cells),
                    "controllable_components": [
                        _cells_payload(component)
                        for component in arm.controllable_components
                    ],
                    "controllable_cells": _cells_payload(arm.controllable_cells),
                    "residual_cells": _cells_payload(arm.residual_cells),
                }
                for arm in self.arms
            ],
        }
        body["content_digest"] = content_digest(body)
        return body


def _cells_payload(cells: Iterable[Cell]) -> List[List[int]]:
    return [[column, row] for column, row in cells]


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def content_digest(body: Mapping[str, Any]) -> str:
    """Digest a payload's canonical serialization, excluding any prior digest."""

    undigested = {key: value for key, value in body.items() if key != "content_digest"}
    return sha256(_DIGEST_PREFIX + canonical_json(undigested).encode("utf-8")).hexdigest()


def cell_difference(
    first: Frame, second: Frame, columns: int, rows: int
) -> FrozenSet[Cell]:
    """Return coarse grid cells whose pixel bytes differ exactly.

    Endpoint frames from the deterministic emulator are byte-exact, so no
    tolerance threshold is applied; a threshold would add a free parameter
    without evidence.
    """

    if columns <= 0 or rows <= 0:
        raise ValueError("label grid dimensions must be positive")
    if (first.width, first.height, first.channels) != (
        second.width,
        second.height,
        second.channels,
    ):
        raise ValueError("cannot difference frames with mismatched geometry")
    if columns > first.width or rows > first.height:
        raise ValueError("label grid cannot be finer than the frame")
    if first.pixels == second.pixels:
        return frozenset()
    row_bytes = first.width * first.channels
    spans = [
        (
            column * first.width // columns * first.channels,
            (column + 1) * first.width // columns * first.channels,
        )
        for column in range(columns)
    ]
    changed: Set[Cell] = set()
    for cell_row in range(rows):
        pending = set(range(columns))
        for y in range(cell_row * first.height // rows, (cell_row + 1) * first.height // rows):
            offset = y * row_bytes
            line_first = first.pixels[offset : offset + row_bytes]
            line_second = second.pixels[offset : offset + row_bytes]
            if line_first == line_second:
                continue
            for column in tuple(pending):
                start, stop = spans[column]
                if line_first[start:stop] != line_second[start:stop]:
                    changed.add((column, cell_row))
                    pending.discard(column)
            if not pending:
                break
    return frozenset(changed)


def connected_components(cells: AbstractSet[Cell]) -> Tuple[Tuple[Cell, ...], ...]:
    """Split cells into 4-connected components, deterministically ordered."""

    remaining = set(cells)
    components: List[Tuple[Cell, ...]] = []
    for seed in sorted(cells):
        if seed not in remaining:
            continue
        remaining.discard(seed)
        component = {seed}
        frontier = [seed]
        while frontier:
            column, row = frontier.pop()
            for neighbor in (
                (column - 1, row),
                (column + 1, row),
                (column, row - 1),
                (column, row + 1),
            ):
                if neighbor in remaining:
                    remaining.discard(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _roots_from_edges(edges: Iterable[_FirstStepEdge]) -> Tuple[CounterfactualRoot, ...]:
    grouped: Dict[Tuple[str, int, str], Dict[Tuple[Action, int], Set[str]]] = {}
    for run_id, group, root_digest, action, duration, endpoint_digest in edges:
        grouped.setdefault((run_id, group, root_digest), {}).setdefault(
            (action, duration), set()
        ).add(endpoint_digest)
    roots = []
    for key in sorted(grouped):
        arms = tuple(
            CounterfactualArm(action, duration, tuple(sorted(endpoints)))
            for (action, duration), endpoints in sorted(
                grouped[key].items(),
                key=lambda item: (item[0][0].value, item[0][1]),
            )
        )
        roots.append(CounterfactualRoot(key[0], key[1], key[2], arms))
    return tuple(roots)


def collect_counterfactual_roots(sequences: Iterable[Any]) -> Tuple[CounterfactualRoot, ...]:
    """Group the first-step edges of decoded sequences into causal roots."""

    def edges() -> Iterable[_FirstStepEdge]:
        for sequence in sequences:
            if not sequence.actions:
                continue
            duration = (
                int(sequence.durations[0]) if sequence.durations else DEFAULT_DURATION
            )
            yield (
                sequence.source_run_id,
                int(sequence.group),
                sequence.frames[0].digest,
                Action(sequence.actions[0]),
                duration,
                sequence.frames[1].digest,
            )

    return _roots_from_edges(edges())


def _control_digests(root: CounterfactualRoot) -> Dict[int, Optional[str]]:
    """Map duration to its unambiguous NOOP endpoint (None when ambiguous)."""

    controls: Dict[int, Optional[str]] = {}
    for arm in root.arms:
        if arm.action is Action.NOOP:
            controls[arm.duration] = (
                None if arm.ambiguous else arm.endpoint_digests[0]
            )
    return controls


def _eligible_arm_controls(
    root: CounterfactualRoot,
) -> Tuple[Tuple[CounterfactualArm, str], ...]:
    """Pair each unambiguous factual arm with its unambiguous matched control."""

    controls = _control_digests(root)
    pairs = []
    for arm in root.arms:
        if arm.action is Action.NOOP or arm.ambiguous:
            continue
        control_digest = controls.get(arm.duration)
        if control_digest is not None:
            pairs.append((arm, control_digest))
    return tuple(pairs)


def _eligible_factual_arms(root: CounterfactualRoot) -> Tuple[CounterfactualArm, ...]:
    return tuple(arm for arm, _ in _eligible_arm_controls(root))


def root_frame_digests(root: CounterfactualRoot) -> FrozenSet[str]:
    """Frame digests required to label this root (eligible arms and controls)."""

    digests: Set[str] = set()
    for arm, control_digest in _eligible_arm_controls(root):
        digests.add(arm.endpoint_digests[0])
        digests.add(control_digest)
    return frozenset(digests)


def _resolve_frame(frames: Mapping[str, Frame], digest: str) -> Frame:
    try:
        return frames[digest]
    except KeyError:
        raise KeyError(f"missing frame payload for digest {digest}") from None


def _censored_arm(
    arm: CounterfactualArm, control_digest: Optional[str], reason: str
) -> CounterfactualArmLabel:
    return CounterfactualArmLabel(
        action=arm.action,
        duration=arm.duration,
        endpoint_digests=arm.endpoint_digests,
        control_digest=control_digest,
        status=STATUS_CENSORED,
        censor_reason=reason,
        corroborating_arms=0,
        changed_cells=(),
        controllable_components=(),
        controllable_cells=(),
        residual_cells=(),
    )


def label_counterfactual_root(
    root: CounterfactualRoot,
    frames: Mapping[str, Frame],
    *,
    columns: int = DEFAULT_COLUMNS,
    rows: int = DEFAULT_ROWS,
) -> CounterfactualRootLabels:
    """Label every factual arm of one causal root, censoring explicitly."""

    controls = _control_digests(root)
    eligible = _eligible_arm_controls(root)
    changed: Dict[Tuple[Action, int], FrozenSet[Cell]] = {}
    for arm, control_digest in eligible:
        changed[(arm.action, arm.duration)] = cell_difference(
            _resolve_frame(frames, arm.endpoint_digests[0]),
            _resolve_frame(frames, control_digest),
            columns,
            rows,
        )
    labels: List[CounterfactualArmLabel] = []
    for arm in root.arms:
        if arm.action is Action.NOOP:
            continue
        if arm.ambiguous:
            labels.append(_censored_arm(arm, None, CENSOR_AMBIGUOUS_ENDPOINT))
            continue
        if arm.duration not in controls:
            labels.append(_censored_arm(arm, None, CENSOR_ABSENT_CONTROL))
            continue
        control_digest = controls[arm.duration]
        if control_digest is None:
            labels.append(_censored_arm(arm, None, CENSOR_AMBIGUOUS_CONTROL))
            continue
        arm_changed = changed[(arm.action, arm.duration)]
        corroborating = [
            changed[(other.action, other.duration)]
            for other, _ in eligible
            if other.action is not arm.action
            and changed[(other.action, other.duration)]
        ]
        if not corroborating:
            labels.append(
                _censored_arm(arm, control_digest, CENSOR_NO_SIBLING_CORROBORATION)
            )
            continue
        corroborated = frozenset.intersection(*corroborating)
        components = connected_components(arm_changed)
        controllable_components = tuple(
            component
            for component in components
            if corroborated.intersection(component)
        )
        controllable_cells = sorted(
            {cell for component in controllable_components for cell in component}
        )
        residual_cells = sorted(arm_changed.difference(controllable_cells))
        labels.append(
            CounterfactualArmLabel(
                action=arm.action,
                duration=arm.duration,
                endpoint_digests=arm.endpoint_digests,
                control_digest=control_digest,
                status=STATUS_LABELED,
                censor_reason=None,
                corroborating_arms=len(corroborating),
                changed_cells=tuple(sorted(arm_changed)),
                controllable_components=controllable_components,
                controllable_cells=tuple(controllable_cells),
                residual_cells=tuple(residual_cells),
            )
        )
    return CounterfactualRootLabels(
        source_run_id=root.source_run_id,
        group=root.group,
        root_digest=root.root_digest,
        columns=columns,
        rows=rows,
        arms=tuple(labels),
    )


def generate_labels(
    sequences: Iterable[Any],
    *,
    columns: int = DEFAULT_COLUMNS,
    rows: int = DEFAULT_ROWS,
) -> Tuple[CounterfactualRootLabels, ...]:
    """Label every causal root with a controlled factual arm among sequences."""

    materialized = list(sequences)
    frames: Dict[str, Frame] = {}
    for sequence in materialized:
        for frame in sequence.frames:
            frames.setdefault(frame.digest, frame)
    labels = []
    for root in collect_counterfactual_roots(materialized):
        if not _eligible_factual_arms(root):
            continue
        labels.append(
            label_counterfactual_root(root, frames, columns=columns, rows=rows)
        )
    return tuple(labels)


def _store_first_step_edges(store: SequenceStore) -> Iterable[_FirstStepEdge]:
    """Yield first-step edges from segment metadata without decoding frames.

    Group-level metadata is not part of the store's public surface; this
    reader deliberately reuses the store's internal record iterator and
    run-provenance helper so group keys cannot drift from the store's own
    conventions (legacy version-1 records inherit segment-derived run IDs).
    """

    for record in store._records():
        actions = record["actions"]
        if not actions:
            continue
        durations = record.get("durations") or []
        yield (
            SequenceStore._record_source_run_id(record),
            int(record["group"]),
            str(record["frames"][0]["digest"]),
            Action(actions[0]),
            int(durations[0]) if durations else DEFAULT_DURATION,
            str(record["frames"][1]["digest"]),
        )


def require_strict_store(store: SequenceStore) -> None:
    track = store.reward_track
    if track != "strict":
        raise ValueError(
            f"counterfactual labels require a strict-bound dataset, found {track!r}"
        )


def open_strict_store(root: Path) -> SequenceStore:
    """Open a sequence store, refusing anything not bound to the strict track."""

    store = SequenceStore(root)
    require_strict_store(store)
    return store


def collect_store_roots(store: SequenceStore) -> Tuple[CounterfactualRoot, ...]:
    """Collect causal roots from store metadata alone (no frame decoding)."""

    require_strict_store(store)
    return _roots_from_edges(_store_first_step_edges(store))


def store_root_statistics(roots: Sequence[CounterfactualRoot]) -> Dict[str, int]:
    """Describe counterfactual coverage of collected roots."""

    return {
        "causal_roots": len(roots),
        "counterfactual_roots": sum(len(root.arms) >= 2 for root in roots),
        "noop_control_roots": sum(
            any(arm.action is Action.NOOP for arm in root.arms)
            and any(arm.action is not Action.NOOP for arm in root.arms)
            for root in roots
        ),
        "control_paired_roots": sum(
            bool(_eligible_factual_arms(root)) for root in roots
        ),
        "factual_arms": sum(
            sum(arm.action is not Action.NOOP for arm in root.arms) for root in roots
        ),
        "eligible_factual_arms": sum(
            len(_eligible_factual_arms(root)) for root in roots
        ),
    }


def generate_store_labels(
    store: SequenceStore,
    *,
    columns: int = DEFAULT_COLUMNS,
    rows: int = DEFAULT_ROWS,
    maximum_roots: Optional[int] = None,
    root_batch_size: int = 32,
    roots: Optional[Sequence[CounterfactualRoot]] = None,
) -> Tuple[CounterfactualRootLabels, ...]:
    """Label store roots in deterministic order, decoding only needed frames.

    Frames are loaded in root batches through ``load_frame_subset`` so the
    full RGB corpus never resides in memory at once.
    """

    require_strict_store(store)
    if maximum_roots is not None and maximum_roots <= 0:
        raise ValueError("maximum root count must be positive")
    if root_batch_size <= 0:
        raise ValueError("root batch size must be positive")
    if roots is None:
        roots = collect_store_roots(store)
    selected = [root for root in roots if _eligible_factual_arms(root)]
    selected.sort(key=lambda root: root.sort_key)
    if maximum_roots is not None:
        selected = selected[:maximum_roots]
    labels: List[CounterfactualRootLabels] = []
    for start in range(0, len(selected), root_batch_size):
        batch = selected[start : start + root_batch_size]
        digests: Set[str] = set()
        for root in batch:
            digests.update(root_frame_digests(root))
        frames = store.load_frame_subset(digests)
        labels.extend(
            label_counterfactual_root(root, frames, columns=columns, rows=rows)
            for root in batch
        )
    return tuple(labels)


def labels_manifest(
    labels: Sequence[CounterfactualRootLabels],
    *,
    reward_track: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate deterministic counts and a digest over all label records."""

    payloads = [item.payload() for item in labels]
    censored: Dict[str, int] = {}
    labeled_arms = 0
    arms_with_candidates = 0
    controllable_cells = 0
    residual_cells = 0
    for payload in payloads:
        for arm in payload["arms"]:
            if arm["status"] == STATUS_LABELED:
                labeled_arms += 1
                controllable_cells += len(arm["controllable_cells"])
                residual_cells += len(arm["residual_cells"])
                if arm["controllable_cells"]:
                    arms_with_candidates += 1
            else:
                reason = str(arm["censor_reason"])
                censored[reason] = censored.get(reason, 0) + 1
    aggregate = sha256(
        _DIGEST_PREFIX
        + "\n".join(payload["content_digest"] for payload in payloads).encode("utf-8")
    ).hexdigest()
    return {
        "version": LABEL_VERSION,
        "generator": "counterfactual_labels",
        "reward_track": reward_track,
        "roots": len(payloads),
        "arms": sum(len(payload["arms"]) for payload in payloads),
        "labeled_arms": labeled_arms,
        "arms_with_controllable_candidates": arms_with_candidates,
        "controllable_cells": controllable_cells,
        "residual_cells": residual_cells,
        "censored_arms": {reason: censored[reason] for reason in sorted(censored)},
        "content_digest": aggregate,
    }


def write_labels(
    labels: Sequence[CounterfactualRootLabels],
    destination: Path,
    *,
    reward_track: Optional[str] = None,
) -> Dict[str, Any]:
    """Write label records as canonical JSONL plus a manifest sidecar."""

    destination = Path(destination).expanduser().resolve()
    manifest_path = destination.with_name(destination.name + ".manifest.json")
    if destination.exists():
        raise FileExistsError(destination)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = labels_manifest(labels, reward_track=reward_track)
    ordered = sorted(labels, key=lambda item: (item.source_run_id, item.group, item.root_digest))
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in ordered:
            handle.write(canonical_json(item.payload()) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate counterfactual controllable-region pseudo-labels from a "
            "strict-bound pixel sequence dataset"
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--maximum-roots", type=int, default=8)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--destination", type=Path, default=None)
    args = parser.parse_args()
    store = open_strict_store(args.dataset)
    roots = collect_store_roots(store)
    labels = generate_store_labels(
        store,
        columns=args.columns,
        rows=args.rows,
        maximum_roots=args.maximum_roots,
        roots=roots,
    )
    if args.destination is not None:
        manifest = write_labels(
            labels, args.destination, reward_track=store.reward_track
        )
    else:
        manifest = labels_manifest(labels, reward_track=store.reward_track)
    print(
        json.dumps(
            {"dataset": store_root_statistics(roots), "labels": manifest},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
