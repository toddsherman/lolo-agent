"""One-shot WP9a milestone-discovery scoring runner (telemetry reduction).

ENGINEERING-ONLY. This thin runner reduces stored ``events.jsonl`` telemetry
to the matched factual/NOOP endpoint pairs consumed by the preregistered pure
scorer in :mod:`lolo_agent.milestone_discovery`, exactly as preregistered in
``docs/milestone-scoring-2026-08-16.md`` (direction-review Amendment D step 3,
roadmap section 17 item 7). It inherits the assisted-footprint caveat: the
label-bearing corpora are assisted-track, so every output is an engineering
artifact of the offline spike, not strict-track evidence.

The runner never touches an emulator or frame pixels. The pooled arrays are
the existing 8x8 ``visual_signature`` fields (64 cells, mean//16, 0-15)
decoded from their hex form. All scoring, event signatures, and valence come
from the module's pure functions; this file only joins telemetry rows into
:class:`~lolo_agent.milestone_discovery.MatchedEndpointPair` values and writes
one deterministic, content-digested report JSON.

Labeling is evaluator/engineering-only: committed goal-semantic fields
(collected-count increases, confirmed-loss flags, coarse-scene changes) are
read for recall bookkeeping and never enter the scorer's inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from hashlib import sha256
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from lolo_agent.milestone_discovery import (
    EventProvenance,
    ExtractedEvent,
    MatchedEndpointPair,
    MilestoneScoreConfig,
    PooledArray,
    SignatureScore,
    VALENCE_NEGATIVE,
    VALENCE_POSITIVE,
    content_signature,
    escape_divergence_cells,
    extract_component_event,
    extract_component_events,
    extract_event,
    extract_events,
    score_events,
    score_events_v2,
)

# Preregistered constants (docs/milestone-scoring-2026-08-16.md section 1).
SUCCESSOR_WINDOW = 8
TOP_RANK_WINDOW = 10
REPORT_TOP_SIGNATURES = 25
POSITIVE_RECALL_THRESHOLD = 0.80
NEGATIVE_RECALL_MINIMUM = 10
DRIFT_DOMINATION_FRACTION = 0.5

# V2 constants (docs/milestone-scoring-v2-2026-08-16.md section 1). The
# recall/domination thresholds above are shared unchanged; the falsification
# frame is identical to v1.
BRANCH_FOLLOWUP_WINDOW = SUCCESSOR_WINDOW
PARENT_WALK_LIMIT = 100000
V2_REPORT_BASENAME = "milestone-scoring-v2-report.json"

KIND_COLLECTION = "collection_increase"
KIND_LIFE_LOSS = "life_loss"
KIND_SCENE_TRANSITION = "scene_transition"

REASON_MAPPED = "mapped"
REASON_RESTORE_COMMIT = "restore_commit"
REASON_MISSING_ROOT = "missing_root"
REASON_NO_VISUAL_CHANGE = "no_visual_change"

# Preregistered corpus C anchor: the annotated Floor 1 clear transition.
CORPUS_C_ANCHOR_RUN = "cycle-000010-floor1-resume-d879-finite-causal-bfs-1000"
CORPUS_C_ANCHOR_DECISIONS = (506, 507, 508)

_PARSED_EVENTS = (
    "decision_started",
    "decision_committed",
    "branch_verified",
    "matched_neutral_verified",
    "human_prior_option_branch_verified",
    "human_prior_option_neutral_verified",
    "human_prior_option_local_neutral_verified",
    "state_saved",
    "env_reset",
    "env_step",
)
_EVENT_MARKERS = {name: f'"event":"{name}"' for name in _PARSED_EVENTS}
_BOOTSTRAP_MARKER = '"phase":"bootstrap"'


def decode_visual_signature(value: object) -> Optional[PooledArray]:
    """Decode a hex ``visual_signature`` into a pooled integer array."""

    if not isinstance(value, str) or not value or len(value) % 2:
        return None
    try:
        return tuple(bytes.fromhex(value))
    except ValueError:
        return None


@dataclass(frozen=True)
class CommittedRecord:
    """One committed decision reduced to pair-assembly fields."""

    attempt: int
    decision: int
    seq: int
    action: str
    action_frames: int
    frame: str
    array: Optional[PooledArray]
    root_frame: Optional[str]
    restored: bool
    is_option: bool
    hearts: Optional[int]
    life_loss: bool
    scene_signature: Optional[str]


@dataclass(frozen=True)
class InstanceRecord:
    """One evaluator-side ground-truth instance with its mapping outcome."""

    run_id: str
    attempt: int
    decision: int
    kind: str
    event_signature: Optional[str]
    reason: str


@dataclass(frozen=True)
class RunReduction:
    """All rows one run contributes, before pair assembly."""

    run_id: str
    state_frame: Mapping[str, str]
    frame_array: Mapping[str, PooledArray]
    decision_roots: Mapping[Tuple[int, int], str]
    committed: Tuple[CommittedRecord, ...]
    strict_branches: Tuple[Mapping[str, object], ...]
    strict_noop_endpoints: Mapping[Tuple[int, int, int], str]
    option_branches: Tuple[Mapping[str, object], ...]
    option_full_neutrals: Mapping[Tuple[str, int], str]
    option_local_neutrals: Mapping[Tuple[str, int], str]
    pool_frames: Tuple[str, ...]
    skipped_rows: int


@dataclass(frozen=True)
class RunPairs:
    """One run's assembled pairs plus evaluator-side bookkeeping."""

    run_id: str
    pairs: Tuple[MatchedEndpointPair, ...]
    drift_signatures: FrozenSet[str]
    pool_signatures: FrozenSet[str]
    instances: Tuple[InstanceRecord, ...]
    committed_signatures: Mapping[Tuple[int, int], str]
    counters: Mapping[str, int]


def reduce_run_events(
    events: Iterable[Mapping[str, object]], run_id: str
) -> RunReduction:
    """Reduce one run's parsed telemetry rows to pair-assembly records.

    ``events`` must be in file (seq) order. Rows lacking required fields are
    counted and skipped; schema drift never aborts the reduction.
    """

    state_frame: Dict[str, str] = {}
    frame_array: Dict[str, PooledArray] = {}
    decision_roots: Dict[Tuple[int, int], str] = {}
    committed: List[CommittedRecord] = []
    strict_branches: List[Mapping[str, object]] = []
    strict_noop: Dict[Tuple[int, int, int], str] = {}
    option_branches: List[Mapping[str, object]] = []
    option_full: Dict[Tuple[str, int], str] = {}
    option_local: Dict[Tuple[str, int], str] = {}
    pool_frames: List[str] = []
    first_root_seen: set = set()
    skipped = 0

    for event in events:
        name = event.get("event")
        frame = event.get("frame")
        array = decode_visual_signature(event.get("visual_signature"))
        if isinstance(frame, str) and array is not None:
            frame_array[frame] = array
        attempt = event.get("attempt")
        attempt = attempt if isinstance(attempt, int) else 1

        if name == "state_saved":
            state_id = event.get("state_id")
            if isinstance(state_id, str) and isinstance(frame, str):
                state_frame[state_id] = frame
            else:
                skipped += 1
        elif name == "env_reset":
            if isinstance(frame, str) and array is not None:
                pool_frames.append(frame)
        elif name == "env_step":
            if event.get("phase") == "bootstrap" and isinstance(frame, str):
                if array is not None:
                    pool_frames.append(frame)
        elif name == "decision_started":
            decision = event.get("decision")
            if isinstance(decision, int) and isinstance(frame, str):
                decision_roots[(attempt, decision)] = frame
                if attempt not in first_root_seen and array is not None:
                    first_root_seen.add(attempt)
                    pool_frames.append(frame)
            else:
                skipped += 1
        elif name == "decision_committed":
            decision = event.get("decision")
            action = event.get("action")
            frames = event.get("action_frames")
            seq = event.get("seq")
            if not (
                isinstance(decision, int)
                and isinstance(action, str)
                and isinstance(frames, int)
                and isinstance(frame, str)
                and isinstance(seq, int)
            ):
                skipped += 1
                continue
            parent_frame = event.get("parent_frame")
            root_frame = (
                parent_frame
                if isinstance(parent_frame, str)
                else decision_roots.get((attempt, decision))
            )
            hearts = event.get("human_prior_collected_hearts")
            is_option = (
                event.get("human_prior_verified_option") is not None
                or event.get("human_prior_option_depth") is not None
            )
            scene = event.get("scene_signature")
            committed.append(
                CommittedRecord(
                    attempt=attempt,
                    decision=decision,
                    seq=seq,
                    action=action,
                    action_frames=frames,
                    frame=frame,
                    array=array,
                    root_frame=root_frame,
                    restored=bool(event.get("restored_archive")),
                    is_option=is_option,
                    hearts=hearts if isinstance(hearts, int) else None,
                    life_loss=bool(
                        event.get("human_prior_life_loss_confirmed")
                    ),
                    scene_signature=scene if isinstance(scene, str) else None,
                )
            )
        elif name == "branch_verified":
            decision = event.get("decision")
            action = event.get("action")
            frames = event.get("action_frames")
            if not (
                isinstance(decision, int)
                and isinstance(action, str)
                and isinstance(frames, int)
                and frames > 0
                and isinstance(frame, str)
                and array is not None
            ):
                skipped += 1
                continue
            if action == "noop":
                strict_noop.setdefault((attempt, decision, frames), frame)
            else:
                strict_branches.append(
                    {
                        "attempt": attempt,
                        "decision": decision,
                        "action": action,
                        "frames": frames,
                        "frame": frame,
                        "branch_id": event.get("branch_id")
                        or f"branch-seq-{event.get('seq', 0)}",
                        "seq": event.get("seq", 0),
                    }
                )
        elif name == "matched_neutral_verified":
            decision = event.get("decision")
            frames = event.get("action_frames")
            if (
                isinstance(decision, int)
                and isinstance(frames, int)
                and frames > 0
                and isinstance(frame, str)
                and array is not None
            ):
                strict_noop.setdefault((attempt, decision, frames), frame)
            else:
                skipped += 1
        elif name == "human_prior_option_branch_verified":
            decision = event.get("decision")
            source = event.get("source_state_id")
            path = event.get("path")
            durations = event.get("durations")
            if not (
                isinstance(decision, int)
                and isinstance(source, str)
                and isinstance(path, list)
                and isinstance(durations, list)
                and path
                and len(path) == len(durations)
                and all(isinstance(d, int) and d > 0 for d in durations)
                and all(isinstance(a, str) for a in path)
                and isinstance(frame, str)
                and array is not None
            ):
                skipped += 1
                continue
            option_branches.append(
                {
                    "attempt": attempt,
                    "decision": decision,
                    "source": source,
                    "path": tuple(path),
                    "durations": tuple(durations),
                    "frame": frame,
                    "seq": event.get("seq", 0),
                }
            )
        elif name == "human_prior_option_neutral_verified":
            source = event.get("source_state_id")
            elapsed = event.get("elapsed_frames")
            if (
                isinstance(source, str)
                and isinstance(elapsed, int)
                and elapsed > 0
                and isinstance(frame, str)
                and array is not None
            ):
                option_full.setdefault((source, elapsed), frame)
            else:
                skipped += 1
        elif name == "human_prior_option_local_neutral_verified":
            parent = event.get("parent_state_id")
            frames = event.get("action_frames")
            if (
                isinstance(parent, str)
                and isinstance(frames, int)
                and frames > 0
                and isinstance(frame, str)
                and array is not None
            ):
                option_local.setdefault((parent, frames), frame)
            else:
                skipped += 1

    return RunReduction(
        run_id=run_id,
        state_frame=state_frame,
        frame_array=frame_array,
        decision_roots=decision_roots,
        committed=tuple(committed),
        strict_branches=tuple(strict_branches),
        strict_noop_endpoints=strict_noop,
        option_branches=tuple(option_branches),
        option_full_neutrals=option_full,
        option_local_neutrals=option_local,
        pool_frames=tuple(pool_frames),
        skipped_rows=skipped,
    )


def _successor_chains(
    committed: Sequence[CommittedRecord],
) -> Dict[Tuple[int, int], Tuple[PooledArray, ...]]:
    """Successor arrays per committed decision, truncated at restores."""

    by_attempt: Dict[int, List[CommittedRecord]] = {}
    for record in committed:
        by_attempt.setdefault(record.attempt, []).append(record)
    chains: Dict[Tuple[int, int], Tuple[PooledArray, ...]] = {}
    for attempt, records in by_attempt.items():
        records.sort(key=lambda record: record.seq)
        for index, record in enumerate(records):
            if record.restored:
                continue
            successors: List[PooledArray] = []
            for later in records[index + 1 :]:
                if later.restored:
                    break
                if later.array is not None:
                    successors.append(later.array)
                if len(successors) >= SUCCESSOR_WINDOW:
                    break
            chains[(attempt, record.decision)] = tuple(successors)
    return chains


def assemble_run_pairs(reduction: RunReduction) -> RunPairs:
    """Assemble one run's matched endpoint pairs per the preregistration."""

    run_id = reduction.run_id
    frame_array = reduction.frame_array
    counters: Dict[str, int] = {
        "strict_branch_pairs": 0,
        "option_branch_pairs": 0,
        "committed_only_pairs": 0,
        "controls_resolved": 0,
        "controls_unresolved": 0,
        "pairs_dropped_unresolved_root": 0,
        "pairs_dropped_length_mismatch": 0,
        "restored_commits_excluded": 0,
        "noop_commits_excluded": 0,
        "committed_decisions": len(reduction.committed),
        "skipped_rows": reduction.skipped_rows,
    }

    chains = _successor_chains(reduction.committed)
    committed_by_key: Dict[Tuple[int, int], CommittedRecord] = {}
    for record in reduction.committed:
        committed_by_key.setdefault((record.attempt, record.decision), record)

    def root_array_for_frame(frame: Optional[str]) -> Optional[PooledArray]:
        if frame is None:
            return None
        return frame_array.get(frame)

    def committed_match(
        attempt: int,
        decision: int,
        root_frame: Optional[str],
        endpoint_frame: str,
    ) -> Tuple[bool, Tuple[PooledArray, ...]]:
        """Whether a pair is its decision's committed transition, plus its
        committed successors (possibly empty even when matched)."""

        record = committed_by_key.get((attempt, decision))
        if record is None or record.restored:
            return False, ()
        if record.frame != endpoint_frame:
            return False, ()
        if root_frame is None or record.root_frame != root_frame:
            return False, ()
        return True, chains.get((attempt, decision), ())

    drift_pairs: set = set()
    pairs: List[MatchedEndpointPair] = []
    covered_transitions: set = set()

    def add_pair(
        root: PooledArray,
        factual: PooledArray,
        control: Optional[PooledArray],
        successors: Tuple[PooledArray, ...],
        decision: int,
        branch_id: str,
        action: str,
        duration: int,
        counter: str,
    ) -> None:
        if len(factual) != len(root) or (
            control is not None and len(control) != len(root)
        ):
            counters["pairs_dropped_length_mismatch"] += 1
            return
        successors = tuple(
            successor for successor in successors if len(successor) == len(root)
        )
        if control is not None:
            counters["controls_resolved"] += 1
            drift_pairs.add((root, control))
        else:
            counters["controls_unresolved"] += 1
        pairs.append(
            MatchedEndpointPair(
                provenance=EventProvenance(
                    run_id=run_id,
                    decision=max(0, decision),
                    branch_id=branch_id,
                    action=action,
                    duration=duration,
                    source="telemetry",
                ),
                root=root,
                factual=factual,
                control=control,
                successors=successors,
            )
        )
        counters[counter] += 1

    for branch in reduction.strict_branches:
        attempt = branch["attempt"]
        decision = branch["decision"]
        root_frame = reduction.decision_roots.get((attempt, decision))
        root = root_array_for_frame(root_frame)
        if root is None:
            counters["pairs_dropped_unresolved_root"] += 1
            continue
        factual = frame_array.get(branch["frame"])
        if factual is None:
            counters["pairs_dropped_unresolved_root"] += 1
            continue
        control_frame = reduction.strict_noop_endpoints.get(
            (attempt, decision, branch["frames"])
        )
        control = root_array_for_frame(control_frame)
        matched, successors = committed_match(
            attempt, decision, root_frame, branch["frame"]
        )
        if matched:
            covered_transitions.add((attempt, decision))
        add_pair(
            root,
            factual,
            control,
            successors,
            decision,
            str(branch["branch_id"]),
            str(branch["action"]),
            int(branch["frames"]),
            "strict_branch_pairs",
        )

    for branch in reduction.option_branches:
        attempt = branch["attempt"]
        decision = branch["decision"]
        path: Tuple[str, ...] = branch["path"]
        durations: Tuple[int, ...] = branch["durations"]
        if all(action == "noop" for action in path):
            continue
        root_frame = reduction.state_frame.get(branch["source"])
        root = root_array_for_frame(root_frame)
        if root is None:
            counters["pairs_dropped_unresolved_root"] += 1
            continue
        factual = frame_array.get(branch["frame"])
        if factual is None:
            counters["pairs_dropped_unresolved_root"] += 1
            continue
        total = sum(durations)
        control_frame = reduction.option_full_neutrals.get(
            (branch["source"], total)
        )
        if control_frame is None and len(path) == 1:
            control_frame = reduction.option_local_neutrals.get(
                (branch["source"], durations[0])
            )
        control = root_array_for_frame(control_frame)
        matched, successors = committed_match(
            attempt, decision, root_frame, branch["frame"]
        )
        if matched:
            covered_transitions.add((attempt, decision))
        add_pair(
            root,
            factual,
            control,
            successors,
            decision,
            f"option-seq-{branch['seq']}",
            ",".join(path),
            total,
            "option_branch_pairs",
        )

    committed_signatures: Dict[Tuple[int, int], str] = {}
    instances: List[InstanceRecord] = []
    last_hearts: Dict[int, Optional[int]] = {}
    last_scene: Dict[int, Optional[str]] = {}
    ordered_committed = sorted(
        reduction.committed, key=lambda record: (record.attempt, record.seq)
    )

    for record in ordered_committed:
        key = (record.attempt, record.decision)
        if record.restored:
            counters["restored_commits_excluded"] += 1
        elif record.action == "noop":
            counters["noop_commits_excluded"] += 1

        signature: Optional[str] = None
        reason = REASON_MAPPED
        root = root_array_for_frame(record.root_frame)
        if record.restored:
            reason = REASON_RESTORE_COMMIT
        elif root is None or record.array is None:
            reason = REASON_MISSING_ROOT
        else:
            probe = extract_event(
                MatchedEndpointPair(
                    provenance=EventProvenance(
                        run_id=run_id,
                        decision=max(0, record.decision),
                        branch_id=f"committed-{record.attempt}-{record.decision}",
                        action=record.action,
                        duration=max(1, record.action_frames),
                        source="telemetry",
                    ),
                    root=root,
                    factual=record.array,
                )
            )
            if probe is None:
                reason = REASON_NO_VISUAL_CHANGE
            else:
                signature = probe.signature
                committed_signatures[key] = signature
                if (
                    key not in covered_transitions
                    and record.action != "noop"
                ):
                    # An option commit executed a whole multi-action path;
                    # a first-action-duration NOOP is not its matched
                    # control, so option commits stay dependence-censored.
                    control = (
                        None
                        if record.is_option
                        else root_array_for_frame(
                            reduction.strict_noop_endpoints.get(
                                (
                                    record.attempt,
                                    record.decision,
                                    record.action_frames,
                                )
                            )
                        )
                    )
                    add_pair(
                        root,
                        record.array,
                        control,
                        chains.get(key, ()),
                        record.decision,
                        f"committed-{record.attempt}-{record.decision}",
                        record.action,
                        max(1, record.action_frames),
                        "committed_only_pairs",
                    )

        previous_hearts = last_hearts.get(record.attempt)
        if record.hearts is not None:
            if previous_hearts is not None and record.hearts > previous_hearts:
                instances.append(
                    InstanceRecord(
                        run_id=run_id,
                        attempt=record.attempt,
                        decision=record.decision,
                        kind=KIND_COLLECTION,
                        event_signature=signature,
                        reason=reason,
                    )
                )
            last_hearts[record.attempt] = record.hearts
        if record.life_loss:
            instances.append(
                InstanceRecord(
                    run_id=run_id,
                    attempt=record.attempt,
                    decision=record.decision,
                    kind=KIND_LIFE_LOSS,
                    event_signature=signature,
                    reason=reason,
                )
            )
        previous_scene = last_scene.get(record.attempt)
        if record.scene_signature is not None:
            if (
                previous_scene is not None
                and record.scene_signature != previous_scene
            ):
                instances.append(
                    InstanceRecord(
                        run_id=run_id,
                        attempt=record.attempt,
                        decision=record.decision,
                        kind=KIND_SCENE_TRANSITION,
                        event_signature=signature,
                        reason=reason,
                    )
                )
            last_scene[record.attempt] = record.scene_signature

    drift_signatures: set = set()
    for root, control in drift_pairs:
        probe = extract_event(
            MatchedEndpointPair(
                provenance=EventProvenance(
                    run_id=run_id,
                    decision=0,
                    branch_id="neutral-drift",
                    action="noop",
                    duration=1,
                    source="telemetry",
                ),
                root=root,
                factual=control,
            )
        )
        if probe is not None:
            drift_signatures.add(probe.signature)

    pool_signatures = frozenset(
        content_signature(frame_array[frame])
        for frame in reduction.pool_frames
        if frame in frame_array
    )
    return RunPairs(
        run_id=run_id,
        pairs=tuple(pairs),
        drift_signatures=frozenset(drift_signatures),
        pool_signatures=pool_signatures,
        instances=tuple(instances),
        committed_signatures=committed_signatures,
        counters=counters,
    )


# ---------------------------------------------------------------------------
# V2 assembly (docs/milestone-scoring-v2-2026-08-16.md). Additive: the v1
# functions above are unchanged and keep reproducing the v1 report.
# ---------------------------------------------------------------------------


def _parent_map(reduction: RunReduction) -> Dict[str, str]:
    """Frame-level parent links for lineage tests.

    A committed non-restored frame descends from its root frame; a verified
    branch or option endpoint descends from its search root. Restored
    commits are deliberately NOT parent-linked: a restore is not a
    controller action, and linking it would falsely make the restored
    archive state a descendant of the pre-restore timeline. First writer
    wins, in deterministic construction order (committed rows in seq order,
    then strict branches, then option branches, both in file order).
    """

    parent: Dict[str, str] = {}
    for record in sorted(
        reduction.committed, key=lambda row: (row.attempt, row.seq)
    ):
        if record.restored or record.root_frame is None:
            continue
        if record.frame != record.root_frame:
            parent.setdefault(record.frame, record.root_frame)
    for branch in reduction.strict_branches:
        root_frame = reduction.decision_roots.get(
            (branch["attempt"], branch["decision"])
        )
        if isinstance(root_frame, str) and branch["frame"] != root_frame:
            parent.setdefault(str(branch["frame"]), root_frame)
    for branch in reduction.option_branches:
        root_frame = reduction.state_frame.get(str(branch["source"]))
        if isinstance(root_frame, str) and branch["frame"] != root_frame:
            parent.setdefault(str(branch["frame"]), root_frame)
    return parent


def _downstream_checker(target_frame: str, parent: Mapping[str, str]):
    """Memoized test for whether a frame descends from ``target_frame``."""

    memo: Dict[str, bool] = {target_frame: True}

    def check(frame: str) -> bool:
        chain = []
        chain_set = set()
        current: Optional[str] = frame
        steps = 0
        while (
            current is not None
            and current not in memo
            and current not in chain_set
            and steps < PARENT_WALK_LIMIT
        ):
            chain.append(current)
            chain_set.add(current)
            current = parent.get(current)
            steps += 1
        value = memo.get(current, False) if current is not None else False
        for visited in chain:
            memo[visited] = value
        return value

    return check


def _successor_windows_v2(
    reduction: RunReduction, parent: Mapping[str, str]
) -> Dict[Tuple[int, int], Tuple[PooledArray, ...]]:
    """Lineage-filtered committed successor windows (requirement 2).

    For each non-restored committed decision, later committed arrays in the
    same attempt count as successors when their root frame descends from
    the event's endpoint frame. Restored rows are skipped (never counted,
    never truncating); rows on a non-descendant lineage (after a restore to
    an unrelated archive) are skipped as well, and the window resumes if
    the timeline later returns to a descendant state. The window is capped
    at ``SUCCESSOR_WINDOW`` counted observations.
    """

    by_attempt: Dict[int, List[CommittedRecord]] = {}
    for record in reduction.committed:
        by_attempt.setdefault(record.attempt, []).append(record)
    windows: Dict[Tuple[int, int], Tuple[PooledArray, ...]] = {}
    for attempt, records in by_attempt.items():
        records.sort(key=lambda record: record.seq)
        for index, record in enumerate(records):
            if record.restored:
                continue
            check = _downstream_checker(record.frame, parent)
            successors: List[PooledArray] = []
            for later in records[index + 1 :]:
                if later.restored or later.array is None:
                    continue
                if later.root_frame is None:
                    continue
                if check(later.root_frame):
                    successors.append(later.array)
                    if len(successors) >= SUCCESSOR_WINDOW:
                        break
            windows[(attempt, record.decision)] = tuple(successors)
    return windows


def _branch_followups(
    reduction: RunReduction,
) -> Dict[str, Tuple[PooledArray, ...]]:
    """Verified endpoints per root frame, in seq order (fallback windows).

    When an event's lineage window is empty (the committed timeline never
    continued from its endpoint), the verified branch and option endpoints
    rooted at that endpoint frame are the only stored observations of its
    future; the first ``BRANCH_FOLLOWUP_WINDOW`` of them form the fallback
    successor window.
    """

    staged: Dict[str, List[Tuple[int, str]]] = {}
    for branch in reduction.strict_branches:
        root_frame = reduction.decision_roots.get(
            (branch["attempt"], branch["decision"])
        )
        if isinstance(root_frame, str):
            staged.setdefault(root_frame, []).append(
                (int(branch.get("seq", 0)), str(branch["frame"]))
            )
    for branch in reduction.option_branches:
        root_frame = reduction.state_frame.get(str(branch["source"]))
        if isinstance(root_frame, str):
            staged.setdefault(root_frame, []).append(
                (int(branch.get("seq", 0)), str(branch["frame"]))
            )
    followups: Dict[str, Tuple[PooledArray, ...]] = {}
    for root_frame, endpoints in staged.items():
        endpoints.sort()
        arrays: List[PooledArray] = []
        for _seq, frame in endpoints:
            array = reduction.frame_array.get(frame)
            if array is not None:
                arrays.append(array)
                if len(arrays) >= BRANCH_FOLLOWUP_WINDOW:
                    break
        if arrays:
            followups[root_frame] = tuple(arrays)
    return followups


def _escape_flags(
    reduction: RunReduction, config: MilestoneScoreConfig
) -> Dict[Tuple[int, int], bool]:
    """Per-decision escape divergence over control-resolved verified arms.

    ``True`` when some verified non-neutral arm at the decision kept at
    least ``escape_cell_minimum`` cells at their root values while the
    equal-duration control changed them; ``False`` when arms were
    evaluable but none did. Decisions with no control-resolved arm are
    absent (censored).
    """

    flags: Dict[Tuple[int, int], bool] = {}

    def note(key: Tuple[int, int], escaped: bool) -> None:
        flags[key] = flags.get(key, False) or escaped

    for branch in reduction.strict_branches:
        key = (branch["attempt"], branch["decision"])
        root_frame = reduction.decision_roots.get(key)
        root = (
            reduction.frame_array.get(root_frame)
            if isinstance(root_frame, str)
            else None
        )
        factual = reduction.frame_array.get(str(branch["frame"]))
        control_frame = reduction.strict_noop_endpoints.get(
            (branch["attempt"], branch["decision"], branch["frames"])
        )
        control = (
            reduction.frame_array.get(control_frame)
            if isinstance(control_frame, str)
            else None
        )
        if root is None or factual is None or control is None:
            continue
        if len(factual) != len(root) or len(control) != len(root):
            continue
        cells = escape_divergence_cells(root, factual, control)
        note(key, cells >= config.escape_cell_minimum)
    for branch in reduction.option_branches:
        key = (branch["attempt"], branch["decision"])
        source = str(branch["source"])
        durations: Tuple[int, ...] = branch["durations"]
        root_frame = reduction.state_frame.get(source)
        root = (
            reduction.frame_array.get(root_frame)
            if isinstance(root_frame, str)
            else None
        )
        factual = reduction.frame_array.get(str(branch["frame"]))
        total = sum(durations)
        control_frame = reduction.option_full_neutrals.get((source, total))
        if control_frame is None and len(durations) == 1:
            control_frame = reduction.option_local_neutrals.get(
                (source, durations[0])
            )
        control = (
            reduction.frame_array.get(control_frame)
            if isinstance(control_frame, str)
            else None
        )
        if root is None or factual is None or control is None:
            continue
        if len(factual) != len(root) or len(control) != len(root):
            continue
        cells = escape_divergence_cells(root, factual, control)
        note(key, cells >= config.escape_cell_minimum)
    return flags


def _escape_lookback(
    flags: Mapping[Tuple[int, int], bool],
    attempt: int,
    decision: int,
    lookback: int,
) -> Optional[bool]:
    """Escape divergence within the decision lookback window (inclusive)."""

    any_evaluable = False
    for earlier in range(decision - lookback + 1, decision + 1):
        flag = flags.get((attempt, earlier))
        if flag is None:
            continue
        any_evaluable = True
        if flag:
            return True
    return False if any_evaluable else None


def _history_by_decision(
    reduction: RunReduction,
) -> Dict[Tuple[int, int], Tuple[PooledArray, ...]]:
    """Deduplicated pre-root observed arrays per committed decision.

    The rewind reference set: the run's pre-intervention pool arrays plus
    every committed array observed strictly before the decision's root in
    the same attempt (restored rows included — a restored state was
    observed on the timeline).
    """

    pool_arrays: List[PooledArray] = []
    pool_seen = set()
    for frame in reduction.pool_frames:
        array = reduction.frame_array.get(frame)
        if array is not None and array not in pool_seen:
            pool_seen.add(array)
            pool_arrays.append(array)
    by_attempt: Dict[int, List[CommittedRecord]] = {}
    for record in reduction.committed:
        by_attempt.setdefault(record.attempt, []).append(record)
    history: Dict[Tuple[int, int], Tuple[PooledArray, ...]] = {}
    for attempt, records in by_attempt.items():
        records.sort(key=lambda record: record.seq)
        seen = set(pool_seen)
        arrays = list(pool_arrays)
        for record in records:
            history.setdefault((attempt, record.decision), tuple(arrays))
            if record.array is not None and record.array not in seen:
                seen.add(record.array)
                arrays.append(record.array)
    return history


def assemble_run_pairs_v2(
    reduction: RunReduction,
    config: Optional[MilestoneScoreConfig] = None,
) -> RunPairs:
    """Assemble one run's pairs per the v2 preregistration.

    Pair construction (roots, controls, exclusion of restore and NOOP
    commits, option-commit dependence censoring) is identical to v1. The
    v2 differences: successor windows are lineage-filtered committed
    windows with branch-level follow-up fallback, every pair carries the
    per-decision escape-lookback flag, pairs with successors carry the
    rewind history, and committed-transition signatures come from the same
    control-aware per-component extraction the scorer uses.
    """

    if config is None:
        config = MilestoneScoreConfig()
    run_id = reduction.run_id
    frame_array = reduction.frame_array
    counters: Dict[str, int] = {
        "strict_branch_pairs": 0,
        "option_branch_pairs": 0,
        "committed_only_pairs": 0,
        "controls_resolved": 0,
        "controls_unresolved": 0,
        "pairs_dropped_unresolved_root": 0,
        "pairs_dropped_length_mismatch": 0,
        "restored_commits_excluded": 0,
        "noop_commits_excluded": 0,
        "committed_decisions": len(reduction.committed),
        "skipped_rows": reduction.skipped_rows,
        "v2_lineage_successor_windows": 0,
        "v2_fallback_successor_windows": 0,
        "v2_empty_successor_windows": 0,
        "escape_decisions_evaluable": 0,
        "escape_decisions_true": 0,
    }

    parent = _parent_map(reduction)
    lineage_windows = _successor_windows_v2(reduction, parent)
    followups = _branch_followups(reduction)
    escape_flags = _escape_flags(reduction, config)
    history_by_decision = _history_by_decision(reduction)
    counters["escape_decisions_evaluable"] = len(escape_flags)
    counters["escape_decisions_true"] = sum(
        1 for flag in escape_flags.values() if flag
    )

    committed_by_key: Dict[Tuple[int, int], CommittedRecord] = {}
    for record in reduction.committed:
        committed_by_key.setdefault((record.attempt, record.decision), record)

    window_counted: set = set()

    def window_for(record: CommittedRecord) -> Tuple[PooledArray, ...]:
        key = (record.attempt, record.decision)
        first_visit = key not in window_counted
        window_counted.add(key)
        lineage = lineage_windows.get(key, ())
        if lineage:
            if first_visit:
                counters["v2_lineage_successor_windows"] += 1
            return lineage
        fallback = followups.get(record.frame, ())
        if fallback:
            if first_visit:
                counters["v2_fallback_successor_windows"] += 1
            return fallback
        if first_visit:
            counters["v2_empty_successor_windows"] += 1
        return ()

    def root_array_for_frame(frame: Optional[str]) -> Optional[PooledArray]:
        if frame is None:
            return None
        return frame_array.get(frame)

    def committed_match(
        attempt: int,
        decision: int,
        root_frame: Optional[str],
        endpoint_frame: str,
    ) -> Tuple[bool, Optional[CommittedRecord]]:
        record = committed_by_key.get((attempt, decision))
        if record is None or record.restored:
            return False, None
        if record.frame != endpoint_frame:
            return False, None
        if root_frame is None or record.root_frame != root_frame:
            return False, None
        return True, record

    drift_pairs: set = set()
    pairs: List[MatchedEndpointPair] = []
    covered_transitions: set = set()
    covered_signatures: Dict[Tuple[int, int], str] = {}

    def add_pair(
        root: PooledArray,
        factual: PooledArray,
        control: Optional[PooledArray],
        successors: Tuple[PooledArray, ...],
        attempt: int,
        decision: int,
        branch_id: str,
        action: str,
        duration: int,
        counter: str,
    ) -> Optional[MatchedEndpointPair]:
        if len(factual) != len(root) or (
            control is not None and len(control) != len(root)
        ):
            counters["pairs_dropped_length_mismatch"] += 1
            return None
        successors = tuple(
            successor for successor in successors if len(successor) == len(root)
        )
        if control is not None:
            counters["controls_resolved"] += 1
            drift_pairs.add((root, control))
        else:
            counters["controls_unresolved"] += 1
        history: Tuple[PooledArray, ...] = ()
        if successors:
            history = tuple(
                reference
                for reference in history_by_decision.get(
                    (attempt, decision), ()
                )
                if len(reference) == len(root)
            )
        pair = MatchedEndpointPair(
            provenance=EventProvenance(
                run_id=run_id,
                decision=max(0, decision),
                branch_id=branch_id,
                action=action,
                duration=duration,
                source="telemetry",
            ),
            root=root,
            factual=factual,
            control=control,
            successors=successors,
            history=history,
            escape_lookback=_escape_lookback(
                escape_flags, attempt, decision, config.divergence_lookback
            ),
        )
        pairs.append(pair)
        counters[counter] += 1
        return pair

    def note_covered(
        record: CommittedRecord, pair: MatchedEndpointPair
    ) -> None:
        key = (record.attempt, record.decision)
        covered_transitions.add(key)
        if key not in covered_signatures:
            event = extract_component_event(pair, config)
            if event is not None:
                covered_signatures[key] = event.signature

    for branch in reduction.strict_branches:
        attempt = branch["attempt"]
        decision = branch["decision"]
        root_frame = reduction.decision_roots.get((attempt, decision))
        root = root_array_for_frame(root_frame)
        if root is None:
            counters["pairs_dropped_unresolved_root"] += 1
            continue
        factual = frame_array.get(branch["frame"])
        if factual is None:
            counters["pairs_dropped_unresolved_root"] += 1
            continue
        control_frame = reduction.strict_noop_endpoints.get(
            (attempt, decision, branch["frames"])
        )
        control = root_array_for_frame(control_frame)
        matched, record = committed_match(
            attempt, decision, root_frame, branch["frame"]
        )
        successors: Tuple[PooledArray, ...] = ()
        if matched and record is not None:
            successors = window_for(record)
        pair = add_pair(
            root,
            factual,
            control,
            successors,
            attempt,
            decision,
            str(branch["branch_id"]),
            str(branch["action"]),
            int(branch["frames"]),
            "strict_branch_pairs",
        )
        if matched and record is not None and pair is not None:
            note_covered(record, pair)

    for branch in reduction.option_branches:
        attempt = branch["attempt"]
        decision = branch["decision"]
        path: Tuple[str, ...] = branch["path"]
        durations: Tuple[int, ...] = branch["durations"]
        if all(action == "noop" for action in path):
            continue
        root_frame = reduction.state_frame.get(branch["source"])
        root = root_array_for_frame(root_frame)
        if root is None:
            counters["pairs_dropped_unresolved_root"] += 1
            continue
        factual = frame_array.get(branch["frame"])
        if factual is None:
            counters["pairs_dropped_unresolved_root"] += 1
            continue
        total = sum(durations)
        control_frame = reduction.option_full_neutrals.get(
            (branch["source"], total)
        )
        if control_frame is None and len(path) == 1:
            control_frame = reduction.option_local_neutrals.get(
                (branch["source"], durations[0])
            )
        control = root_array_for_frame(control_frame)
        matched, record = committed_match(
            attempt, decision, root_frame, branch["frame"]
        )
        successors = ()
        if matched and record is not None:
            successors = window_for(record)
        pair = add_pair(
            root,
            factual,
            control,
            successors,
            attempt,
            decision,
            f"option-seq-{branch['seq']}",
            ",".join(path),
            total,
            "option_branch_pairs",
        )
        if matched and record is not None and pair is not None:
            note_covered(record, pair)

    committed_signatures: Dict[Tuple[int, int], str] = {}
    instances: List[InstanceRecord] = []
    last_hearts: Dict[int, Optional[int]] = {}
    last_scene: Dict[int, Optional[str]] = {}
    ordered_committed = sorted(
        reduction.committed, key=lambda record: (record.attempt, record.seq)
    )

    for record in ordered_committed:
        key = (record.attempt, record.decision)
        if record.restored:
            counters["restored_commits_excluded"] += 1
        elif record.action == "noop":
            counters["noop_commits_excluded"] += 1

        signature: Optional[str] = None
        reason = REASON_MAPPED
        root = root_array_for_frame(record.root_frame)
        if record.restored:
            reason = REASON_RESTORE_COMMIT
        elif root is None or record.array is None:
            reason = REASON_MISSING_ROOT
        elif key in covered_signatures:
            signature = covered_signatures[key]
            committed_signatures[key] = signature
        else:
            control = (
                None
                if record.is_option or record.action == "noop"
                # An option commit executed a whole multi-action path; a
                # first-action-duration NOOP is not its matched control, so
                # option commits stay dependence-censored (v1 rule). NOOP
                # commits are the control arm by definition.
                else root_array_for_frame(
                    reduction.strict_noop_endpoints.get(
                        (
                            record.attempt,
                            record.decision,
                            record.action_frames,
                        )
                    )
                )
            )
            if (
                len(record.array) != len(root)
                or (control is not None and len(control) != len(root))
            ):
                reason = REASON_MISSING_ROOT
            else:
                probe_successors = window_for(record)
                probe_history: Tuple[PooledArray, ...] = ()
                if probe_successors:
                    probe_history = tuple(
                        reference
                        for reference in history_by_decision.get(key, ())
                        if len(reference) == len(root)
                    )
                probe_pair = MatchedEndpointPair(
                    provenance=EventProvenance(
                        run_id=run_id,
                        decision=max(0, record.decision),
                        branch_id=(
                            f"committed-{record.attempt}-{record.decision}"
                        ),
                        action=record.action,
                        duration=max(1, record.action_frames),
                        source="telemetry",
                    ),
                    root=root,
                    factual=record.array,
                    control=control,
                    successors=tuple(
                        successor
                        for successor in probe_successors
                        if len(successor) == len(root)
                    ),
                    history=probe_history,
                    escape_lookback=_escape_lookback(
                        escape_flags,
                        record.attempt,
                        record.decision,
                        config.divergence_lookback,
                    ),
                )
                probe = extract_component_event(probe_pair, config)
                if probe is None:
                    reason = REASON_NO_VISUAL_CHANGE
                else:
                    signature = probe.signature
                    committed_signatures[key] = signature
                    if record.action != "noop":
                        if control is not None:
                            counters["controls_resolved"] += 1
                            drift_pairs.add((root, control))
                        else:
                            counters["controls_unresolved"] += 1
                        pairs.append(probe_pair)
                        counters["committed_only_pairs"] += 1

        previous_hearts = last_hearts.get(record.attempt)
        if record.hearts is not None:
            if previous_hearts is not None and record.hearts > previous_hearts:
                instances.append(
                    InstanceRecord(
                        run_id=run_id,
                        attempt=record.attempt,
                        decision=record.decision,
                        kind=KIND_COLLECTION,
                        event_signature=signature,
                        reason=reason,
                    )
                )
            last_hearts[record.attempt] = record.hearts
        if record.life_loss:
            instances.append(
                InstanceRecord(
                    run_id=run_id,
                    attempt=record.attempt,
                    decision=record.decision,
                    kind=KIND_LIFE_LOSS,
                    event_signature=signature,
                    reason=reason,
                )
            )
        previous_scene = last_scene.get(record.attempt)
        if record.scene_signature is not None:
            if (
                previous_scene is not None
                and record.scene_signature != previous_scene
            ):
                instances.append(
                    InstanceRecord(
                        run_id=run_id,
                        attempt=record.attempt,
                        decision=record.decision,
                        kind=KIND_SCENE_TRANSITION,
                        event_signature=signature,
                        reason=reason,
                    )
                )
            last_scene[record.attempt] = record.scene_signature

    drift_signatures: set = set()
    for root, control in drift_pairs:
        probe = extract_event(
            MatchedEndpointPair(
                provenance=EventProvenance(
                    run_id=run_id,
                    decision=0,
                    branch_id="neutral-drift",
                    action="noop",
                    duration=1,
                    source="telemetry",
                ),
                root=root,
                factual=control,
            )
        )
        if probe is not None:
            drift_signatures.add(probe.signature)

    pool_signatures = frozenset(
        content_signature(frame_array[frame])
        for frame in reduction.pool_frames
        if frame in frame_array
    )
    return RunPairs(
        run_id=run_id,
        pairs=tuple(pairs),
        drift_signatures=frozenset(drift_signatures),
        pool_signatures=pool_signatures,
        instances=tuple(instances),
        committed_signatures=committed_signatures,
        counters=counters,
    )


def iter_run_events(path: str) -> Iterable[Mapping[str, object]]:
    """Yield only the parsed telemetry rows the reduction consumes."""

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for name, marker in _EVENT_MARKERS.items():
                if marker in line:
                    if name == "env_step" and _BOOTSTRAP_MARKER not in line:
                        break
                    try:
                        event = json.loads(line)
                    except ValueError:
                        break
                    if event.get("event") == name:
                        yield event
                    break


def _merge_counters(
    total: Dict[str, int], counters: Mapping[str, int]
) -> None:
    for key, value in counters.items():
        total[key] = total.get(key, 0) + value


def canonical_json(payload: object) -> str:
    """Deterministic JSON encoding used for both the digest and the file."""

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def content_digest(payload: object) -> str:
    return sha256(canonical_json(payload).encode()).hexdigest()


def _signature_row(
    rank: int,
    score: SignatureScore,
    drift: FrozenSet[str],
    candidate_signatures: Mapping[str, Tuple[str, ...]],
) -> Dict[str, object]:
    samples = [
        {
            "run_id": provenance.run_id,
            "decision": provenance.decision,
            "action": provenance.action,
            "duration": provenance.duration,
        }
        for provenance in score.provenance[:3]
    ]
    return {
        "rank": rank,
        "signature": score.signature,
        "occurrences": score.occurrences,
        "score": score.score,
        "log_rarity": score.log_rarity,
        "action_dependence_rate": score.action_dependence_rate,
        "dependence_evaluable": score.dependence_evaluable,
        "dependence_censored": score.dependence_censored,
        "censored_non_return_factor": score.censored_non_return_factor,
        "return_evaluable": score.return_evaluable,
        "return_censored": score.return_censored,
        "successor_novelty_margin": score.successor_novelty_margin,
        "reversion_to_seen_rate": score.reversion_to_seen_rate,
        "persistence_rate": score.persistence_rate,
        "valence": score.valence,
        "valence_basis": score.valence_basis,
        "neutral_drift_member": score.signature in drift,
        "candidate_event_kinds": sorted(
            candidate_signatures.get(score.signature, ())
        ),
        "sample_provenance": samples,
    }


def score_corpus(
    corpus_id: str,
    run_directories: Sequence[str],
    config: Optional[MilestoneScoreConfig] = None,
) -> Dict[str, object]:
    """Reduce, pair, and score one corpus; returns the report section."""

    if config is None:
        config = MilestoneScoreConfig()
    events: List[ExtractedEvent] = []
    total_pairs = 0
    counters: Dict[str, int] = {}
    drift: set = set()
    pool: set = set()
    instances: List[InstanceRecord] = []
    committed_signatures: Dict[Tuple[str, int, int], str] = {}
    runs_scanned = 0
    runs_missing_events = 0

    for run_directory in sorted(run_directories):
        run_id = os.path.basename(run_directory.rstrip(os.sep))
        events_path = os.path.join(run_directory, "events.jsonl")
        if not os.path.exists(events_path):
            runs_missing_events += 1
            continue
        runs_scanned += 1
        reduction = reduce_run_events(iter_run_events(events_path), run_id)
        run_pairs = assemble_run_pairs(reduction)
        total_pairs += len(run_pairs.pairs)
        events.extend(extract_events(run_pairs.pairs))
        _merge_counters(counters, run_pairs.counters)
        drift.update(run_pairs.drift_signatures)
        pool.update(run_pairs.pool_signatures)
        instances.extend(run_pairs.instances)
        for key, signature in run_pairs.committed_signatures.items():
            committed_signatures[(run_id, key[0], key[1])] = signature

    scores = score_events(events, frozenset(pool), config)
    rank_by_signature = {
        score.signature: (rank, score)
        for rank, score in enumerate(scores, 1)
    }
    drift_frozen = frozenset(drift)

    candidate_signatures: Dict[str, Tuple[str, ...]] = {}
    for instance in instances:
        if instance.event_signature is None:
            continue
        kinds = set(candidate_signatures.get(instance.event_signature, ()))
        kinds.add(instance.kind)
        candidate_signatures[instance.event_signature] = tuple(sorted(kinds))

    top = scores[:TOP_RANK_WINDOW]
    top_nonzero = [score for score in top if score.score > 0.0]
    drift_in_top = sum(
        1 for score in top_nonzero if score.signature in drift_frozen
    )
    nonzero_signatures = sum(1 for score in scores if score.score > 0.0)
    drift_fraction = (
        drift_in_top / len(top_nonzero) if top_nonzero else 0.0
    )
    animation_dominated = (
        nonzero_signatures < TOP_RANK_WINDOW
        or drift_fraction >= DRIFT_DOMINATION_FRACTION
    )
    precision_matches = sum(
        1 for score in top if score.signature in candidate_signatures
    )

    instance_rows = []
    for instance in sorted(
        instances,
        key=lambda row: (row.kind, row.run_id, row.attempt, row.decision),
    ):
        row: Dict[str, object] = {
            "run_id": instance.run_id,
            "attempt": instance.attempt,
            "decision": instance.decision,
            "kind": instance.kind,
            "reason": instance.reason,
            "event_signature": instance.event_signature,
        }
        if instance.event_signature is not None:
            ranked = rank_by_signature.get(instance.event_signature)
            if ranked is not None:
                rank, score = ranked
                row["rank"] = rank
                row["score"] = score.score
                row["valence"] = score.valence
                row["valence_basis"] = score.valence_basis
        instance_rows.append(row)

    valence_counts: Dict[str, int] = {}
    for score in scores:
        valence_counts[score.valence] = valence_counts.get(score.valence, 0) + 1

    anchor_rows = []
    if corpus_id == "C":
        for decision in CORPUS_C_ANCHOR_DECISIONS:
            found = None
            for (run_id, _attempt, dec), signature in sorted(
                committed_signatures.items()
            ):
                if run_id == CORPUS_C_ANCHOR_RUN and dec == decision:
                    found = signature
                    break
            row = {
                "run_id": CORPUS_C_ANCHOR_RUN,
                "decision": decision,
                "event_signature": found,
            }
            if found is not None and found in rank_by_signature:
                rank, score = rank_by_signature[found]
                row["rank"] = rank
                row["score"] = score.score
                row["valence"] = score.valence
                row["valence_basis"] = score.valence_basis
            anchor_rows.append(row)

    section: Dict[str, object] = {
        "corpus_id": corpus_id,
        "runs_scanned": runs_scanned,
        "runs_missing_events": runs_missing_events,
        "total_pairs": total_pairs,
        "total_events": len(events),
        "pairs_without_event": total_pairs - len(events),
        "counters": dict(sorted(counters.items())),
        "seen_pool_size": len(pool),
        "neutral_drift_signatures": len(drift_frozen),
        "distinct_signatures": len(scores),
        "nonzero_score_signatures": nonzero_signatures,
        "valence_counts": dict(sorted(valence_counts.items())),
        "top_signatures": [
            _signature_row(rank, score, drift_frozen, candidate_signatures)
            for rank, score in enumerate(
                scores[:REPORT_TOP_SIGNATURES], 1
            )
        ],
        "top10_nonzero_count": len(top_nonzero),
        "top10_neutral_drift_fraction": drift_fraction,
        "animation_dominated": animation_dominated,
        "top10_candidate_event_precision": precision_matches
        / max(1, len(top)),
        "instances": instance_rows,
        "instance_counts": {
            kind: sum(1 for row in instances if row.kind == kind)
            for kind in (
                KIND_COLLECTION,
                KIND_LIFE_LOSS,
                KIND_SCENE_TRANSITION,
            )
        },
    }
    if anchor_rows:
        section["floor1_clear_anchor"] = anchor_rows
    return section


def _signature_row_v2(
    rank: int,
    score: SignatureScore,
    drift: FrozenSet[str],
    candidate_signatures: Mapping[str, Tuple[str, ...]],
) -> Dict[str, object]:
    row = _signature_row(rank, score, drift, candidate_signatures)
    row["negative_divergence_rate"] = score.negative_divergence_rate
    row["rewound_occurrences"] = score.rewound_occurrences
    row["escape_lookback_occurrences"] = score.escape_lookback_occurrences
    return row


def score_corpus_v2(
    corpus_id: str,
    run_directories: Sequence[str],
    config: Optional[MilestoneScoreConfig] = None,
) -> Dict[str, object]:
    """Reduce, pair, and score one corpus with the v2 semantics."""

    if config is None:
        config = MilestoneScoreConfig()
    events: List[ExtractedEvent] = []
    total_pairs = 0
    counters: Dict[str, int] = {}
    drift: set = set()
    pool: set = set()
    instances: List[InstanceRecord] = []
    committed_signatures: Dict[Tuple[str, int, int], str] = {}
    runs_scanned = 0
    runs_missing_events = 0

    for run_directory in sorted(run_directories):
        run_id = os.path.basename(run_directory.rstrip(os.sep))
        events_path = os.path.join(run_directory, "events.jsonl")
        if not os.path.exists(events_path):
            runs_missing_events += 1
            continue
        runs_scanned += 1
        reduction = reduce_run_events(iter_run_events(events_path), run_id)
        run_pairs = assemble_run_pairs_v2(reduction, config)
        total_pairs += len(run_pairs.pairs)
        events.extend(extract_component_events(run_pairs.pairs, config))
        _merge_counters(counters, run_pairs.counters)
        drift.update(run_pairs.drift_signatures)
        pool.update(run_pairs.pool_signatures)
        instances.extend(run_pairs.instances)
        for key, signature in run_pairs.committed_signatures.items():
            committed_signatures[(run_id, key[0], key[1])] = signature

    scores = score_events_v2(events, frozenset(pool), config)
    rank_by_signature = {
        score.signature: (rank, score)
        for rank, score in enumerate(scores, 1)
    }
    drift_frozen = frozenset(drift)

    candidate_signatures: Dict[str, Tuple[str, ...]] = {}
    for instance in instances:
        if instance.event_signature is None:
            continue
        kinds = set(candidate_signatures.get(instance.event_signature, ()))
        kinds.add(instance.kind)
        candidate_signatures[instance.event_signature] = tuple(sorted(kinds))

    top = scores[:TOP_RANK_WINDOW]
    top_nonzero = [score for score in top if score.score > 0.0]
    drift_in_top = sum(
        1 for score in top_nonzero if score.signature in drift_frozen
    )
    nonzero_signatures = sum(1 for score in scores if score.score > 0.0)
    drift_fraction = (
        drift_in_top / len(top_nonzero) if top_nonzero else 0.0
    )
    animation_dominated = (
        nonzero_signatures < TOP_RANK_WINDOW
        or drift_fraction >= DRIFT_DOMINATION_FRACTION
    )
    precision_matches = sum(
        1 for score in top if score.signature in candidate_signatures
    )

    instance_rows = []
    for instance in sorted(
        instances,
        key=lambda row: (row.kind, row.run_id, row.attempt, row.decision),
    ):
        row: Dict[str, object] = {
            "run_id": instance.run_id,
            "attempt": instance.attempt,
            "decision": instance.decision,
            "kind": instance.kind,
            "reason": instance.reason,
            "event_signature": instance.event_signature,
        }
        if instance.event_signature is not None:
            ranked = rank_by_signature.get(instance.event_signature)
            if ranked is not None:
                rank, score = ranked
                row["rank"] = rank
                row["score"] = score.score
                row["valence"] = score.valence
                row["valence_basis"] = score.valence_basis
        instance_rows.append(row)

    valence_counts: Dict[str, int] = {}
    for score in scores:
        valence_counts[score.valence] = valence_counts.get(score.valence, 0) + 1

    anchor_rows = []
    if corpus_id == "C":
        for decision in CORPUS_C_ANCHOR_DECISIONS:
            found = None
            for (run_id, _attempt, dec), signature in sorted(
                committed_signatures.items()
            ):
                if run_id == CORPUS_C_ANCHOR_RUN and dec == decision:
                    found = signature
                    break
            row = {
                "run_id": CORPUS_C_ANCHOR_RUN,
                "decision": decision,
                "event_signature": found,
            }
            if found is not None and found in rank_by_signature:
                rank, score = rank_by_signature[found]
                row["rank"] = rank
                row["score"] = score.score
                row["valence"] = score.valence
                row["valence_basis"] = score.valence_basis
            anchor_rows.append(row)

    section: Dict[str, object] = {
        "corpus_id": corpus_id,
        "runs_scanned": runs_scanned,
        "runs_missing_events": runs_missing_events,
        "total_pairs": total_pairs,
        "total_events": len(events),
        "pairs_without_event": total_pairs - len(events),
        "counters": dict(sorted(counters.items())),
        "seen_pool_size": len(pool),
        "neutral_drift_signatures": len(drift_frozen),
        "distinct_signatures": len(scores),
        "nonzero_score_signatures": nonzero_signatures,
        "valence_counts": dict(sorted(valence_counts.items())),
        "top_signatures": [
            _signature_row_v2(rank, score, drift_frozen, candidate_signatures)
            for rank, score in enumerate(
                scores[:REPORT_TOP_SIGNATURES], 1
            )
        ],
        "top10_nonzero_count": len(top_nonzero),
        "top10_neutral_drift_fraction": drift_fraction,
        "animation_dominated": animation_dominated,
        "top10_candidate_event_precision": precision_matches
        / max(1, len(top)),
        "instances": instance_rows,
        "instance_counts": {
            kind: sum(1 for row in instances if row.kind == kind)
            for kind in (
                KIND_COLLECTION,
                KIND_LIFE_LOSS,
                KIND_SCENE_TRANSITION,
            )
        },
    }
    if anchor_rows:
        section["floor1_clear_anchor"] = anchor_rows
    return section


def evaluate_gates(
    section_a: Mapping[str, object], section_b: Mapping[str, object]
) -> Dict[str, object]:
    """Pooled A+B recall gates per the preregistration."""

    def gate_rows(section: Mapping[str, object], kind: str):
        return [
            row
            for row in section["instances"]  # type: ignore[index]
            if row["kind"] == kind
        ]

    collections = gate_rows(section_a, KIND_COLLECTION) + gate_rows(
        section_b, KIND_COLLECTION
    )
    losses = gate_rows(section_a, KIND_LIFE_LOSS) + gate_rows(
        section_b, KIND_LIFE_LOSS
    )
    positive_hits = sum(
        1
        for row in collections
        if row.get("valence") == VALENCE_POSITIVE
        and float(row.get("score", 0.0)) > 0.0
    )
    negative_hits = sum(
        1 for row in losses if row.get("valence") == VALENCE_NEGATIVE
    )
    positive_recall = positive_hits / len(collections) if collections else 0.0
    return {
        "collection_instances": len(collections),
        "collection_positive_nonzero": positive_hits,
        "positive_recall": positive_recall,
        "positive_recall_threshold": POSITIVE_RECALL_THRESHOLD,
        "positive_recall_gate_passed": positive_recall
        >= POSITIVE_RECALL_THRESHOLD,
        "life_loss_instances": len(losses),
        "life_loss_negative": negative_hits,
        "negative_recall_minimum": NEGATIVE_RECALL_MINIMUM,
        "negative_recall_gate_passed": negative_hits
        >= NEGATIVE_RECALL_MINIMUM,
    }


def build_report(
    repo_root: str, config: Optional[MilestoneScoreConfig] = None
) -> Dict[str, object]:
    """Run the full preregistered scoring pass and build the report payload."""

    if config is None:
        config = MilestoneScoreConfig()
    corpus_a_root = os.path.join(
        repo_root, "experiments", "lolo1-entity-v10", "evaluations"
    )
    extended_root = os.path.join(
        repo_root, "experiments", "lolo1-medium", "extended_evaluations"
    )
    corpus_a_runs = [
        os.path.join(corpus_a_root, name)
        for name in sorted(os.listdir(corpus_a_root))
        if os.path.isdir(os.path.join(corpus_a_root, name))
    ]
    extended_runs = [
        name
        for name in sorted(os.listdir(extended_root))
        if os.path.isdir(os.path.join(extended_root, name))
    ]
    corpus_b_runs = [
        os.path.join(extended_root, name)
        for name in extended_runs
        if "human-prior" in name
    ]
    corpus_c_runs = [
        os.path.join(extended_root, name)
        for name in extended_runs
        if "human-prior" not in name
    ]

    section_a = score_corpus("A", corpus_a_runs, config)
    section_b = score_corpus("B", corpus_b_runs, config)
    section_c = score_corpus("C", corpus_c_runs, config)
    gates = evaluate_gates(section_a, section_b)

    falsification = {
        "timer_animation_domination": {
            "A": section_a["animation_dominated"],
            "B": section_b["animation_dominated"],
            "C": section_c["animation_dominated"],
        },
        "heart_inseparability": not gates["positive_recall_gate_passed"],
        "wp9_step1_falsified_as_written": bool(
            section_a["animation_dominated"]
            or section_b["animation_dominated"]
            or section_c["animation_dominated"]
            or not gates["positive_recall_gate_passed"]
        ),
    }

    payload: Dict[str, object] = {
        "schema": "milestone-scoring-report/1",
        "preregistration": "docs/milestone-scoring-2026-08-16.md",
        "census": "docs/milestone-event-census-2026-08-16.md",
        "provenance_note": (
            "engineering-only offline spike; assisted-footprint caveat "
            "applies; corpora A/B assisted-track, corpus C excluded from "
            "precision/recall thresholds per census section 5"
        ),
        "config": {
            "novelty_baseline": config.novelty_baseline,
            "negative_reversion_threshold": (
                config.negative_reversion_threshold
            ),
            "positive_persistence_threshold": (
                config.positive_persistence_threshold
            ),
            "positive_novelty_threshold": config.positive_novelty_threshold,
            "successor_window": SUCCESSOR_WINDOW,
            "top_rank_window": TOP_RANK_WINDOW,
            "drift_domination_fraction": DRIFT_DOMINATION_FRACTION,
            "seen_pool": "pre_intervention",
        },
        "corpora": {"A": section_a, "B": section_b, "C": section_c},
        "gates": gates,
        "falsification": falsification,
    }
    payload["content_digest"] = content_digest(payload)
    return payload


def build_report_v2(
    repo_root: str, config: Optional[MilestoneScoreConfig] = None
) -> Dict[str, object]:
    """Run the preregistered v2 scoring pass and build the report payload.

    Same corpora, gates, and falsification frame as v1
    (docs/milestone-scoring-2026-08-16.md); redesigned mechanisms and
    thresholds per docs/milestone-scoring-v2-2026-08-16.md.
    """

    if config is None:
        config = MilestoneScoreConfig()
    corpus_a_root = os.path.join(
        repo_root, "experiments", "lolo1-entity-v10", "evaluations"
    )
    extended_root = os.path.join(
        repo_root, "experiments", "lolo1-medium", "extended_evaluations"
    )
    corpus_a_runs = [
        os.path.join(corpus_a_root, name)
        for name in sorted(os.listdir(corpus_a_root))
        if os.path.isdir(os.path.join(corpus_a_root, name))
    ]
    extended_runs = [
        name
        for name in sorted(os.listdir(extended_root))
        if os.path.isdir(os.path.join(extended_root, name))
    ]
    corpus_b_runs = [
        os.path.join(extended_root, name)
        for name in extended_runs
        if "human-prior" in name
    ]
    corpus_c_runs = [
        os.path.join(extended_root, name)
        for name in extended_runs
        if "human-prior" not in name
    ]

    section_a = score_corpus_v2("A", corpus_a_runs, config)
    section_b = score_corpus_v2("B", corpus_b_runs, config)
    section_c = score_corpus_v2("C", corpus_c_runs, config)
    gates = evaluate_gates(section_a, section_b)

    falsification = {
        "timer_animation_domination": {
            "A": section_a["animation_dominated"],
            "B": section_b["animation_dominated"],
            "C": section_c["animation_dominated"],
        },
        "heart_inseparability": not gates["positive_recall_gate_passed"],
        "wp9_step1_falsified_as_redesigned": bool(
            section_a["animation_dominated"]
            or section_b["animation_dominated"]
            or section_c["animation_dominated"]
            or not gates["positive_recall_gate_passed"]
        ),
    }

    payload: Dict[str, object] = {
        "schema": "milestone-scoring-v2-report/1",
        "preregistration": "docs/milestone-scoring-v2-2026-08-16.md",
        "v1_preregistration": "docs/milestone-scoring-2026-08-16.md",
        "census": "docs/milestone-event-census-2026-08-16.md",
        "provenance_note": (
            "engineering-only offline spike; assisted-footprint caveat "
            "applies; corpora A/B assisted-track, corpus C excluded from "
            "precision/recall thresholds per census section 5; v2 redesign "
            "per docs/learnings.md section 4.33"
        ),
        "config": {
            "novelty_baseline": config.novelty_baseline,
            "negative_reversion_threshold": (
                config.negative_reversion_threshold
            ),
            "positive_persistence_threshold": (
                config.positive_persistence_threshold
            ),
            "positive_novelty_threshold": config.positive_novelty_threshold,
            "negative_divergence_threshold": (
                config.negative_divergence_threshold
            ),
            "rewind_transient_floor": config.rewind_transient_floor,
            "rewind_proximity_ceiling": config.rewind_proximity_ceiling,
            "escape_cell_minimum": config.escape_cell_minimum,
            "divergence_lookback": config.divergence_lookback,
            "successor_window": SUCCESSOR_WINDOW,
            "branch_followup_window": BRANCH_FOLLOWUP_WINDOW,
            "top_rank_window": TOP_RANK_WINDOW,
            "drift_domination_fraction": DRIFT_DOMINATION_FRACTION,
            "seen_pool": "pre_intervention",
            "successor_windows": "lineage_filtered_with_branch_followups",
            "event_extraction": "per_component",
            "negative_valence": "delayed_divergence",
        },
        "corpora": {"A": section_a, "B": section_b, "C": section_c},
        "gates": gates,
        "falsification": falsification,
    }
    payload["content_digest"] = content_digest(payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot WP9a milestone-discovery scoring pass over the "
            "census-qualified corpora (engineering-only)."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=os.getcwd(),
        help="Repository root containing experiments/ (default: cwd).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Report path relative to the repository root (default: the "
            "preregistered v1 or v2 report path)."
        ),
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help=(
            "Run the section-4.33 redesign preregistered in "
            "docs/milestone-scoring-v2-2026-08-16.md instead of the v1 pass."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.output is None:
        arguments.output = os.path.join(
            "experiments",
            "lolo1-wp5",
            V2_REPORT_BASENAME
            if arguments.v2
            else "milestone-scoring-report.json",
        )
    if arguments.v2:
        payload = build_report_v2(arguments.repo_root)
    else:
        payload = build_report(arguments.repo_root)
    output_path = os.path.join(arguments.repo_root, arguments.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(canonical_json(payload))
        handle.write("\n")
    print(f"wrote {output_path}")
    print(f"content_digest={payload['content_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
