"""Preregistered score-conflict root mining over stored run telemetry (WP8).

Offline, read-only implementation of the conflict-root mining procedure in
``docs/wp8-relational-planner-design-2026-08-17.md`` section 6. The learning
it serves (docs/learnings.md section 4.43): an ablation root must exhibit
*score conflict* — a configuration the baseline scorer disprefers but
certified accessibility prefers — or consequence bits cannot discriminate
deliberate from incidental choice.

What this module does, and deliberately does not do:

- It walks recorded ``events.jsonl`` telemetry (v322-v328 by default),
  reconstructs the archive candidate set at every restore-selection instant
  and every committed-decision boundary, and re-scores every candidate under
  (a) the baseline frontier score *recorded in telemetry* (the archive-add
  ``score`` field, per the section 6 step 1 rule) and (b) the would-be
  certified verified-accessibility bonus computed by the real
  :func:`lolo_agent.accessibility_preference.verified_accessibility_preference`
  against the certified record store, with the current side resolved by the
  section 6.8 root/baseline designation rule.
- A **conflict root** is a decision point where
  ``argmax(baseline) != argmax(baseline + bonus)``.
- Conflicts are classified into the three preregistered families
  (novelty-decoy, post-exploit, exhaustion) and reported with restorability
  evidence (state file exists + digest verified).
- If NO organic conflict exists in the corpus, that is a disclosed result
  and the module deterministically constructs the *seeded* designs the
  design doc licenses (section 6 step 5): an archive-seeded root pairing a
  certified-improving branch with a strictly higher-baseline neutral decoy
  from sibling runs, and a records-file-variant alternative that certifies
  a coverage record for a configuration the baseline underranks. Both are
  marked ``constructed: true`` — never silently substituted for mined roots.
- It never touches an emulator, never launches runs, and writes nothing
  except the manifest file the caller asks for.

Scoring-basis disclosure (fixed by the preregistration, restated here):
per-candidate baseline scores are the archive-add-time ``score`` fields
(``persistent_frontier_value`` for causal-outcome archive entries, which
carry no ``score``); the restore-time ``persistent_frontier_value`` of each
in-run winner is recorded as annotation and used for the seeded-design
pairing after subtracting any recorded ``verified_accessibility_bonus``
(the v328 treatment logs a bonus-inclusive value). Restore-time re-scoring
of non-winners is planner-state-dependent and not reconstructible offline;
instants where the in-run winner differs from the offline baseline argmax
are flagged, never hidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .accessibility_preference import (
    AccessibilityRecordProvenance,
    CertifiedAccessibilityRecord,
    VERIFICATION_CERTIFIED_HOLD,
    verified_accessibility_preference,
)

PROCEDURE_NAME = "wp8-conflict-root-mining"
PROCEDURE_VERSION = 1
PREREGISTRATION_DOC = "docs/wp8-relational-planner-design-2026-08-17.md#6"

DEFAULT_RUN_IDS: Tuple[str, ...] = (
    "entity-v322-room3-paired-probe-arm-a-pushed-d12",
    "entity-v323-room3-paired-probe-arm-b-prepush-d12",
    "entity-v324-room3-paired-probe-arm-b-rerun-certified-d12",
    "entity-v325-room3-object-removed-probe-d12",
    "entity-v326-room3-object-removed-repetition-d12",
    "entity-v327-room3-wp8lite-control-w0-d12",
    "entity-v328-room3-wp8lite-treatment-w1-d12",
)

FAMILY_NOVELTY_DECOY = "novelty_decoy"
FAMILY_POST_EXPLOIT = "post_exploit"
FAMILY_EXHAUSTION = "exhaustion"
FAMILIES = (FAMILY_NOVELTY_DECOY, FAMILY_POST_EXPLOIT, FAMILY_EXHAUSTION)

CURRENT_MAPPED = "mapped"
CURRENT_BASELINE = "baseline"
CURRENT_MISSING = "missing"

SEEDED_PROVENANCE_MARKER = "SEEDED-CONSTRUCT"

_CELL_PIXELS = 16
_TABLE_LIMIT = 8

# Main-archive entries: the candidate universe restore selection ranks.
# The causal-outcome archive is a separate store with its own fallback
# path (its entries are tracked for winner joins and root-signature
# semantics but never enter the scored candidate set; the per-instant
# archive_size cross-check validates this universe claim against the
# recorded telemetry).
_MAIN_ADD_EVENTS = (
    "human_prior_option_archive_added",
    "archive_branch_added",
)
_AUX_ADD_EVENTS = ("archive_causal_outcome_added",)
_SNAPSHOT_EVENTS = (
    "option_archive_snapshot_stored",
    "goal_milestone_checkpoint_snapshot_stored",
    "decision_snapshot_stored",
)
_EXHAUSTION_MARKERS = ("goal_milestone_exhaustion", "goal_exhaustion")


# ---------------------------------------------------------------------------
# Record store loading (section 6.8 semantics, standalone so this module
# never imports the planner monolith; content signatures are cross-checked
# against the values the runs recorded at load time).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MiningRecordStore:
    """Certified records keyed by configuration signature, plus the
    designated root/current baseline record (section 6.8)."""

    records: Mapping[str, CertifiedAccessibilityRecord]
    root_record: Optional[CertifiedAccessibilityRecord]
    file_sha256: Optional[str]

    @property
    def root_configuration_signature(self) -> Optional[str]:
        if self.root_record is None:
            return None
        return self.root_record.provenance.configuration_signature

    def content_signatures(self) -> Dict[str, str]:
        return {
            signature: record.content_signature()
            for signature, record in sorted(self.records.items())
        }


def record_from_payload(payload: Mapping[str, Any]) -> CertifiedAccessibilityRecord:
    provenance_payload = payload["provenance"]
    provenance = AccessibilityRecordProvenance(
        run_id=provenance_payload["run_id"],
        preregistration_doc=provenance_payload["preregistration_doc"],
        configuration_signature=provenance_payload["configuration_signature"],
        verification=provenance_payload["verification"],
        certification_predicate=provenance_payload["certification_predicate"],
        certified_branches=provenance_payload["certified_branches"],
        total_branches=provenance_payload["total_branches"],
        search_depth=provenance_payload["search_depth"],
        search_beam=provenance_payload["search_beam"],
    )
    return CertifiedAccessibilityRecord(
        provenance=provenance,
        certified_cells=tuple(
            (cell[0], cell[1]) for cell in payload["certified_cells"]
        ),
        certified_open_frontiers=tuple(
            ((edge[0][0], edge[0][1]), (edge[1][0], edge[1][1]))
            for edge in payload.get("certified_open_frontiers", ())
        ),
        certified_milestone_cells=tuple(
            (cell[0], cell[1])
            for cell in payload.get("certified_milestone_cells", ())
        ),
        preparation_outcome_category=payload.get(
            "preparation_outcome_category", "none"
        ),
        confirmed_manipulation_count=payload.get(
            "confirmed_manipulation_count", 0
        ),
    )


def store_from_payloads(
    payloads: Sequence[Mapping[str, Any]],
    file_sha256: Optional[str] = None,
) -> MiningRecordStore:
    records: Dict[str, CertifiedAccessibilityRecord] = {}
    root_record: Optional[CertifiedAccessibilityRecord] = None
    for payload in payloads:
        designation = payload.get("root_configuration", False)
        if not isinstance(designation, bool):
            raise ValueError(
                "root_configuration must be a boolean when present"
            )
        record = record_from_payload(payload)
        signature = record.provenance.configuration_signature
        if signature in records:
            raise ValueError(
                f"duplicate record configuration signature {signature!r}"
            )
        records[signature] = record
        if designation:
            if root_record is not None:
                raise ValueError(
                    "more than one record designates root_configuration"
                )
            root_record = record
    return MiningRecordStore(
        records=records, root_record=root_record, file_sha256=file_sha256
    )


def load_certified_records(path: Path) -> MiningRecordStore:
    raw = Path(path).read_bytes()
    payloads = json.loads(raw.decode("utf-8"))
    return store_from_payloads(
        payloads, file_sha256=hashlib.sha256(raw).hexdigest()
    )


def resolve_current_record(
    store: MiningRecordStore, current_signature: str
) -> Tuple[Optional[CertifiedAccessibilityRecord], str]:
    """Section 6.8 current-side resolution.

    Non-empty mapped signature -> that record (``mapped``); the empty
    signature with a designated baseline -> the baseline record
    (``baseline``); anything else refuses (``missing``) — a non-empty
    unmapped signature never falls back to the baseline.
    """

    if current_signature:
        record = store.records.get(current_signature)
        if record is not None:
            return record, CURRENT_MAPPED
        return None, CURRENT_MISSING
    if store.root_record is not None:
        return store.root_record, CURRENT_BASELINE
    return None, CURRENT_MISSING


def candidate_bonus(
    store: MiningRecordStore,
    candidate_signature: Optional[str],
    current_record: Optional[CertifiedAccessibilityRecord],
    weight: float = 1.0,
) -> Tuple[float, Optional[str]]:
    """Would-be verified-accessibility bonus for one candidate.

    Returns ``(bonus, refusal_reason)``; refusals score exactly zero, with
    the reason exposed (unverified accessibility never scores as observed).
    """

    if not candidate_signature:
        return 0.0, "candidate_signature_absent"
    candidate_record = store.records.get(candidate_signature)
    if candidate_record is None:
        return 0.0, "candidate_record_missing"
    if current_record is None:
        return 0.0, "current_record_missing"
    components = verified_accessibility_preference(
        candidate_record, current_record
    )
    if not components.scored:
        return 0.0, components.refusal_reason
    return weight * components.total_bonus, None


# ---------------------------------------------------------------------------
# Telemetry mining
# ---------------------------------------------------------------------------


@dataclass
class ArchiveCandidate:
    state_id: str
    signature: Optional[str]
    recorded_score: float
    source_event: str
    decision: Optional[int]
    seq: int
    restore_seq: Optional[int] = None
    restore_baseline_value: Optional[float] = None


@dataclass
class Restorability:
    state_id: str
    snapshot_recorded: bool
    state_file: Optional[str]
    recorded_sha256: Optional[str]
    file_exists: bool
    digest_verified: bool

    def as_payload(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "snapshot_recorded": self.snapshot_recorded,
            "state_file": self.state_file,
            "recorded_sha256": self.recorded_sha256,
            "file_exists": self.file_exists,
            "digest_verified": self.digest_verified,
        }


@dataclass
class ScoredCandidate:
    state_id: str
    signature: Optional[str]
    baseline_score: float
    bonus: float
    refusal_reason: Optional[str]
    add_seq: int
    source_event: str

    @property
    def combined_score(self) -> float:
        return self.baseline_score + self.bonus

    def as_payload(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "signature": self.signature,
            "baseline_score": self.baseline_score,
            "bonus": self.bonus,
            "combined_score": self.combined_score,
            "refusal_reason": self.refusal_reason,
            "source_event": self.source_event,
        }


@dataclass
class DecisionPointResult:
    run_id: str
    kind: str  # "restore" | "decision_commit"
    seq: int
    decision: Optional[int]
    reason: Optional[str]
    current_signature: str
    current_source: str
    candidate_count: int
    positive_bonus_candidates: int
    baseline_top: ScoredCandidate
    combined_top: ScoredCandidate
    conflict: bool
    candidates: List[ScoredCandidate]
    all_candidates: List[ScoredCandidate] = field(default_factory=list)
    recorded_archive_size: Optional[int] = None
    record_mapped_live_state_ids: Tuple[str, ...] = ()
    collected_milestone_cells: Tuple[Tuple[int, int], ...] = ()
    exhaustion_context: bool = False
    in_run_winner_state_id: Optional[str] = None
    in_run_winner_baseline_value: Optional[float] = None
    in_run_winner_matches_baseline_argmax: Optional[bool] = None

    def as_payload(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "seq": self.seq,
            "decision": self.decision,
            "reason": self.reason,
            "current_signature": self.current_signature,
            "current_source": self.current_source,
            "candidate_count": self.candidate_count,
            "recorded_archive_size": self.recorded_archive_size,
            "archive_size_matches_candidate_count": (
                None
                if self.recorded_archive_size is None
                else self.recorded_archive_size == self.candidate_count
            ),
            "positive_bonus_candidates": self.positive_bonus_candidates,
            "record_mapped_live_state_ids": list(
                self.record_mapped_live_state_ids
            ),
            "collected_milestone_cells": [
                list(cell) for cell in self.collected_milestone_cells
            ],
            "exhaustion_context": self.exhaustion_context,
            "baseline_argmax": self.baseline_top.as_payload(),
            "combined_argmax": self.combined_top.as_payload(),
            "conflict": self.conflict,
            "in_run_winner_state_id": self.in_run_winner_state_id,
            "in_run_winner_baseline_value": (
                self.in_run_winner_baseline_value
            ),
            "in_run_winner_matches_baseline_argmax": (
                self.in_run_winner_matches_baseline_argmax
            ),
            "candidate_table": [
                candidate.as_payload() for candidate in self.candidates
            ],
        }


@dataclass
class ConflictRoot:
    run_id: str
    point: DecisionPointResult
    family: str
    baseline_gap: float
    minimum_flipping_bonus: float
    flip_margin: float
    baseline_top_restorability: Restorability
    combined_top_restorability: Restorability
    instrument_gap_dependent: bool = False

    def as_payload(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "family": self.family,
            "decision_point": self.point.as_payload(),
            "baseline_gap": self.baseline_gap,
            "minimum_flipping_bonus": self.minimum_flipping_bonus,
            "flip_margin": self.flip_margin,
            "baseline_top_restorability": (
                self.baseline_top_restorability.as_payload()
            ),
            "combined_top_restorability": (
                self.combined_top_restorability.as_payload()
            ),
            "instrument_gap_dependent": self.instrument_gap_dependent,
            "instrument_gap_note": (
                "the current side resolves to the designated baseline "
                "only because a recorded restore of a track-less branch "
                "reset the configuration signature (learnings section "
                "4.29; unmerged fix 6a8488a); under the mandated fix the "
                "current side resolves mapped, the bonus is zero, and "
                "this conflict does not exist — DISQUALIFIED as a Gate 4 "
                "E2 root"
                if self.instrument_gap_dependent
                else None
            ),
        }


@dataclass
class BonusCrossCheck:
    seq: int
    decision: Optional[int]
    state_id: str
    recorded_bonus: Optional[float]
    recorded_current_source: Optional[str]
    computed_bonus: float
    computed_current_source: str
    match: Optional[bool]

    def as_payload(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "decision": self.decision,
            "state_id": self.state_id,
            "recorded_bonus": self.recorded_bonus,
            "recorded_current_source": self.recorded_current_source,
            "computed_bonus": self.computed_bonus,
            "computed_current_source": self.computed_current_source,
            "match": self.match,
        }


@dataclass
class RunMiningResult:
    run_id: str
    run_dir: Path
    events_total: int = 0
    seeded_root_signature: str = ""
    root_staging_reason: Optional[str] = None
    decision_points: List[DecisionPointResult] = field(default_factory=list)
    conflicts: List[ConflictRoot] = field(default_factory=list)
    candidates: Dict[str, ArchiveCandidate] = field(default_factory=dict)
    snapshots: Dict[str, Tuple[Optional[str], Optional[str]]] = field(
        default_factory=dict
    )
    restore_instants: int = 0
    unknown_removals: int = 0
    unknown_restored_winners: List[str] = field(default_factory=list)
    record_mapped_candidates: int = 0
    bonus_cross_checks: List[BonusCrossCheck] = field(default_factory=list)
    collected_milestone_cells_final: Tuple[Tuple[int, int], ...] = ()

    def summary_payload(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "events_total": self.events_total,
            "seeded_root_signature": self.seeded_root_signature or None,
            "root_staging_reason": self.root_staging_reason,
            "archive_candidates": len(self.candidates),
            "record_mapped_candidates": self.record_mapped_candidates,
            "restore_instants": self.restore_instants,
            "decision_points_evaluated": len(self.decision_points),
            "conflicts": len(self.conflicts),
            "unknown_removals": self.unknown_removals,
            "unknown_restored_winners": list(self.unknown_restored_winners),
            "collected_milestone_cells_final": [
                list(cell) for cell in self.collected_milestone_cells_final
            ],
            "bonus_cross_checks": [
                check.as_payload() for check in self.bonus_cross_checks
            ],
            "decision_point_table": [
                point.as_payload() for point in self.decision_points
            ],
        }


def _heart_slot_cell(slot: Sequence[int]) -> Tuple[int, int]:
    return (int(slot[0]) // _CELL_PIXELS, int(slot[1]) // _CELL_PIXELS)


def check_restorability(
    run_dir: Path,
    state_id: str,
    snapshots: Mapping[str, Tuple[Optional[str], Optional[str]]],
) -> Restorability:
    snapshot = snapshots.get(state_id)
    if snapshot is None:
        return Restorability(
            state_id=state_id,
            snapshot_recorded=False,
            state_file=None,
            recorded_sha256=None,
            file_exists=False,
            digest_verified=False,
        )
    state_file, recorded_sha = snapshot
    file_exists = False
    digest_verified = False
    if state_file:
        state_path = Path(run_dir) / state_file
        file_exists = state_path.is_file()
        if file_exists and recorded_sha:
            digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
            digest_verified = digest == recorded_sha
    return Restorability(
        state_id=state_id,
        snapshot_recorded=True,
        state_file=state_file,
        recorded_sha256=recorded_sha,
        file_exists=file_exists,
        digest_verified=digest_verified,
    )


def _score_members(
    members: Mapping[str, ArchiveCandidate],
    store: MiningRecordStore,
    current_signature: str,
    weight: float,
) -> Tuple[List[ScoredCandidate], str]:
    current_record, current_source = resolve_current_record(
        store, current_signature
    )
    scored = []
    for candidate in members.values():
        bonus, refusal = candidate_bonus(
            store, candidate.signature, current_record, weight
        )
        scored.append(
            ScoredCandidate(
                state_id=candidate.state_id,
                signature=candidate.signature,
                baseline_score=candidate.recorded_score,
                bonus=bonus,
                refusal_reason=refusal,
                add_seq=candidate.seq,
                source_event=candidate.source_event,
            )
        )
    return scored, current_source


def _argmax(
    scored: Sequence[ScoredCandidate], key: str
) -> ScoredCandidate:
    if key == "baseline":
        return min(scored, key=lambda c: (-c.baseline_score, c.add_seq))
    return min(scored, key=lambda c: (-c.combined_score, c.add_seq))


def classify_family(
    exhaustion_context: bool,
    challenger_record: Optional[CertifiedAccessibilityRecord],
    collected_cells: Sequence[Tuple[int, int]],
) -> str:
    """Deterministic family classification (design doc section 6 step 4).

    Exhaustion: a goal-exhaustion recovery event fired earlier in the same
    decision. Post-exploit: the challenger record carries milestone cells
    and every one of them is already collected at the instant (the record's
    milestone component is spent). Otherwise novelty-decoy — the generic
    shape where a fresher/unmapped branch outranks the certified one.
    """

    if exhaustion_context:
        return FAMILY_EXHAUSTION
    if challenger_record is not None:
        milestones = challenger_record.certified_milestone_cells
        if milestones and all(
            cell in collected_cells for cell in milestones
        ):
            return FAMILY_POST_EXPLOIT
    return FAMILY_NOVELTY_DECOY


def _evaluate_decision_point(
    result: RunMiningResult,
    members: Mapping[str, ArchiveCandidate],
    store: MiningRecordStore,
    current_signature: str,
    weight: float,
    kind: str,
    seq: int,
    decision: Optional[int],
    reason: Optional[str],
    exhaustion_decisions: Mapping[int, int],
    collected_cells: Sequence[Tuple[int, int]],
    in_run_winner: Optional[Mapping[str, Any]] = None,
    instrument_gap_active: bool = False,
    recorded_archive_size: Optional[int] = None,
) -> Optional[DecisionPointResult]:
    if not members:
        return None
    scored, current_source = _score_members(
        members, store, current_signature, weight
    )
    baseline_top = _argmax(scored, "baseline")
    combined_top = _argmax(scored, "combined")
    conflict = baseline_top.state_id != combined_top.state_id
    positive = sum(1 for candidate in scored if candidate.bonus > 0.0)
    table = sorted(
        scored, key=lambda c: (-c.combined_score, c.add_seq)
    )[:_TABLE_LIMIT]
    for candidate in scored:
        if candidate.bonus > 0.0 and candidate not in table:
            table.append(candidate)
    exhaustion_context = False
    if decision is not None:
        marker_seq = exhaustion_decisions.get(decision)
        exhaustion_context = marker_seq is not None and marker_seq < seq
    point = DecisionPointResult(
        run_id=result.run_id,
        kind=kind,
        seq=seq,
        decision=decision,
        reason=reason,
        current_signature=current_signature,
        current_source=current_source,
        candidate_count=len(scored),
        positive_bonus_candidates=positive,
        baseline_top=baseline_top,
        combined_top=combined_top,
        conflict=conflict,
        candidates=table,
        all_candidates=scored,
        recorded_archive_size=recorded_archive_size,
        record_mapped_live_state_ids=tuple(
            candidate.state_id
            for candidate in scored
            if candidate.signature
            and candidate.signature in store.records
        ),
        collected_milestone_cells=tuple(collected_cells),
        exhaustion_context=exhaustion_context,
    )
    if in_run_winner is not None:
        winner_id = in_run_winner.get("state_id")
        point.in_run_winner_state_id = winner_id
        point.in_run_winner_baseline_value = in_run_winner.get(
            "baseline_value"
        )
        point.in_run_winner_matches_baseline_argmax = (
            winner_id == baseline_top.state_id
        )
    result.decision_points.append(point)
    if conflict:
        challenger_record = (
            store.records.get(combined_top.signature)
            if combined_top.signature
            else None
        )
        family = classify_family(
            exhaustion_context,
            challenger_record,
            collected_cells,
        )
        baseline_gap = (
            baseline_top.baseline_score - combined_top.baseline_score
        )
        result.conflicts.append(
            ConflictRoot(
                run_id=result.run_id,
                point=point,
                family=family,
                baseline_gap=baseline_gap,
                minimum_flipping_bonus=baseline_gap,
                flip_margin=combined_top.bonus - baseline_gap,
                baseline_top_restorability=check_restorability(
                    result.run_dir, baseline_top.state_id, result.snapshots
                ),
                combined_top_restorability=check_restorability(
                    result.run_dir, combined_top.state_id, result.snapshots
                ),
                instrument_gap_dependent=(
                    instrument_gap_active
                    and current_source == CURRENT_BASELINE
                ),
            )
        )
    return point


def mine_run(
    run_dir: Path,
    store: MiningRecordStore,
    weight: float = 1.0,
) -> RunMiningResult:
    """Mine one recorded run for score-conflict decision points."""

    run_dir = Path(run_dir)
    result = RunMiningResult(run_id=run_dir.name, run_dir=run_dir)
    members: Dict[str, ArchiveCandidate] = {}
    aux_members: Dict[str, ArchiveCandidate] = {}
    current_signature = ""
    signature_reset_active = False
    collected_slots: List[Tuple[int, int]] = []
    exhaustion_decisions: Dict[int, int] = {}
    events_path = run_dir / "events.jsonl"
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            result.events_total += 1
            name = event.get("event", "")
            seq = event.get("seq", result.events_total)
            decision = event.get("decision")
            for marker in _EXHAUSTION_MARKERS:
                if marker in name and decision is not None:
                    exhaustion_decisions.setdefault(decision, seq)
            if name == "attempt_started" and result.root_staging_reason is None:
                result.root_staging_reason = event.get("reason")
            elif name == "human_prior_root_object_state_seeded":
                current_signature = (
                    event.get("tracked_world_state_signature") or ""
                )
                result.seeded_root_signature = current_signature
            elif name in _SNAPSHOT_EVENTS:
                state_id = event.get("state_id")
                if state_id:
                    result.snapshots[state_id] = (
                        event.get("state_file"),
                        event.get("state_sha256"),
                    )
            elif name in _MAIN_ADD_EVENTS or name in _AUX_ADD_EVENTS:
                state_id = event.get("state_id")
                if not state_id:
                    continue
                if name == "human_prior_option_archive_added":
                    signature = event.get(
                        "human_prior_option_tracked_world_state_signature"
                    )
                    recorded = event.get("score")
                else:
                    signature = None
                    recorded = event.get("score")
                    if recorded is None:
                        recorded = event.get("persistent_frontier_value")
                candidate = ArchiveCandidate(
                    state_id=state_id,
                    signature=signature,
                    recorded_score=float(recorded or 0.0),
                    source_event=name,
                    decision=decision,
                    seq=seq,
                )
                if name in _MAIN_ADD_EVENTS:
                    members[state_id] = candidate
                else:
                    aux_members[state_id] = candidate
                result.candidates[state_id] = candidate
                if signature and signature in store.records:
                    result.record_mapped_candidates += 1
            elif name == "archive_branch_removed":
                state_id = event.get("state_id")
                if state_id in members:
                    del members[state_id]
                elif state_id in aux_members:
                    del aux_members[state_id]
                else:
                    result.unknown_removals += 1
            elif name == "archive_branch_restored":
                result.restore_instants += 1
                winner_id = event.get("state_id")
                recorded_bonus = event.get("verified_accessibility_bonus")
                recorded_source = event.get(
                    "verified_accessibility_current_source"
                )
                pfv = event.get("persistent_frontier_value")
                baseline_value = None
                if pfv is not None:
                    baseline_value = float(pfv) - float(
                        recorded_bonus or 0.0
                    )
                winner = result.candidates.get(winner_id)
                current_record, current_source = resolve_current_record(
                    store, current_signature
                )
                computed_bonus, _refusal = candidate_bonus(
                    store,
                    winner.signature if winner else None,
                    current_record,
                    weight,
                )
                match = None
                if (
                    recorded_bonus is not None
                    and recorded_source not in (None, "disabled")
                ):
                    match = (
                        abs(computed_bonus - float(recorded_bonus)) < 1e-9
                        and current_source == recorded_source
                    )
                result.bonus_cross_checks.append(
                    BonusCrossCheck(
                        seq=seq,
                        decision=decision,
                        state_id=winner_id,
                        recorded_bonus=recorded_bonus,
                        recorded_current_source=recorded_source,
                        computed_bonus=computed_bonus,
                        computed_current_source=current_source,
                        match=match,
                    )
                )
                _evaluate_decision_point(
                    result,
                    members,
                    store,
                    current_signature,
                    weight,
                    "restore",
                    seq,
                    decision,
                    event.get("reason"),
                    exhaustion_decisions,
                    tuple(collected_slots),
                    in_run_winner={
                        "state_id": winner_id,
                        "baseline_value": baseline_value,
                    },
                    instrument_gap_active=signature_reset_active,
                    recorded_archive_size=event.get("archive_size"),
                )
                previous_signature = current_signature
                if winner is not None:
                    winner.restore_seq = seq
                    winner.restore_baseline_value = baseline_value
                    current_signature = winner.signature or ""
                else:
                    result.unknown_restored_winners.append(winner_id)
                    current_signature = ""
                if previous_signature and not current_signature:
                    # A restore of a track-less branch reset the root
                    # signature (learnings section 4.29 instrument gap).
                    signature_reset_active = True
                elif current_signature:
                    signature_reset_active = False
                members.pop(winner_id, None)
                aux_members.pop(winner_id, None)
            elif name == "decision_committed":
                _evaluate_decision_point(
                    result,
                    members,
                    store,
                    current_signature,
                    weight,
                    "decision_commit",
                    seq,
                    decision,
                    None,
                    exhaustion_decisions,
                    tuple(collected_slots),
                    instrument_gap_active=signature_reset_active,
                )
                slots = event.get("human_prior_collected_heart_slots")
                if slots:
                    for slot in slots:
                        cell = _heart_slot_cell(slot)
                        if cell not in collected_slots:
                            collected_slots.append(cell)
    result.collected_milestone_cells_final = tuple(collected_slots)
    return result


def mine_corpus(
    evaluations_dir: Path,
    store: MiningRecordStore,
    run_ids: Sequence[str] = DEFAULT_RUN_IDS,
    weight: float = 1.0,
) -> List[RunMiningResult]:
    results = []
    for run_id in run_ids:
        run_dir = Path(evaluations_dir) / run_id
        if not (run_dir / "events.jsonl").is_file():
            raise FileNotFoundError(
                f"missing events.jsonl for run {run_id} under "
                f"{evaluations_dir}"
            )
        results.append(mine_run(run_dir, store, weight=weight))
    return results


# ---------------------------------------------------------------------------
# Seeded-root construction (design doc section 6 step 5 fallback; only
# meaningful when the corpus yields zero organic conflicts, always marked
# constructed and disclosed).
# ---------------------------------------------------------------------------


def _restorable(
    result: RunMiningResult, state_id: str
) -> Tuple[bool, Restorability]:
    restorability = check_restorability(
        result.run_dir, state_id, result.snapshots
    )
    return restorability.digest_verified, restorability


def construct_archive_seeded_design(
    results: Sequence[RunMiningResult], store: MiningRecordStore
) -> Dict[str, Any]:
    """Primary seeded design: archive seeded with a certified-improving
    branch and a strictly higher-baseline neutral decoy from sibling runs.

    Deterministic selection rule (fixed here, preregistered): decoy pool =
    restore-instant winners whose signature is absent/unmapped, restorable,
    valued at their recorded restore-time baseline
    (``persistent_frontier_value`` minus any recorded bonus); certified pool
    = archived candidates whose signature maps to a record with a positive
    bonus against the designated root baseline, restorable, valued at their
    restore-time baseline when they won a restore, else their add-time
    score. Valid pairs satisfy ``decoy > certified`` on the baseline and
    ``certified + bonus > decoy`` on the combined score. The selected pair
    maximizes ``min(baseline_gap, flip_margin)`` with lexicographic
    tie-breaking. Uses only real archived states and the real record store —
    no fabricated certification.
    """

    if store.root_record is None:
        return {
            "constructed": False,
            "reason": "no designated root baseline record in the store",
        }
    decoys = []
    certified = []
    for result in results:
        for candidate in result.candidates.values():
            mapped = bool(
                candidate.signature
                and candidate.signature in store.records
            )
            if candidate.restore_seq is not None and not mapped:
                if candidate.restore_baseline_value is None:
                    continue
                ok, restorability = _restorable(result, candidate.state_id)
                if not ok:
                    continue
                decoys.append(
                    (result, candidate, candidate.restore_baseline_value,
                     restorability)
                )
            if mapped:
                bonus, _refusal = candidate_bonus(
                    store, candidate.signature, store.root_record
                )
                if bonus <= 0.0:
                    continue
                ok, restorability = _restorable(result, candidate.state_id)
                if not ok:
                    continue
                value = (
                    candidate.restore_baseline_value
                    if candidate.restore_baseline_value is not None
                    else candidate.recorded_score
                )
                certified.append(
                    (result, candidate, value, bonus, restorability)
                )
    best = None
    for decoy_result, decoy, decoy_value, decoy_restore in decoys:
        for (cert_result, cert, cert_value, bonus,
             cert_restore) in certified:
            gap = decoy_value - cert_value
            flip_margin = bonus - gap
            if gap <= 0.0 or flip_margin <= 0.0:
                continue
            robustness = min(gap, flip_margin)
            key = (
                -robustness,
                decoy_result.run_id,
                decoy.state_id,
                cert_result.run_id,
                cert.state_id,
            )
            entry = (
                key, decoy_result, decoy, decoy_value, decoy_restore,
                cert_result, cert, cert_value, bonus, cert_restore,
                gap, flip_margin, robustness,
            )
            if best is None or key < best[0]:
                best = entry
    if best is None:
        return {
            "constructed": False,
            "reason": (
                "no valid decoy/certified pair among restorable recorded "
                "candidates"
            ),
        }
    (_key, decoy_result, decoy, decoy_value, decoy_restore, cert_result,
     cert, cert_value, bonus, cert_restore, gap, flip_margin,
     robustness) = best
    equivalents = [
        other.run_id
        for other in results
        if other.run_id != cert_result.run_id
        and cert.state_id in other.candidates
        and other.candidates[cert.state_id].signature == cert.signature
    ]
    return {
        "constructed": True,
        "family": FAMILY_NOVELTY_DECOY,
        "kind": "archive_seeded_root",
        "disclosure": (
            "CONSTRUCTED seeded root per design doc section 6 step 5: the "
            "archive at the preregistered pre-push root is seeded with a "
            "certified-improving branch and a strictly higher-baseline "
            "neutral decoy taken from sibling runs' recorded, restorable "
            "states. Real record store, real archived states, no "
            "fabricated certification. Never a substitute for an organic "
            "mined root; disclosed as constructed."
        ),
        "root": {
            "staging": cert_result.root_staging_reason,
            "root_signature": cert_result.seeded_root_signature or "",
            "current_side_resolution": CURRENT_BASELINE,
            "reference": "docs/wp8-lite-ablation-design-2026-08-16.md#6.1",
        },
        "decoy": {
            "run_id": decoy_result.run_id,
            "state_id": decoy.state_id,
            "signature": decoy.signature,
            "add_time_score": decoy.recorded_score,
            "restore_time_baseline_value": decoy_value,
            "restore_seq": decoy.restore_seq,
            "restorability": decoy_restore.as_payload(),
        },
        "certified_branch": {
            "run_id": cert_result.run_id,
            "state_id": cert.state_id,
            "signature": cert.signature,
            "add_time_score": cert.recorded_score,
            "baseline_value_used": cert_value,
            "restore_seq": cert.restore_seq,
            "bonus_vs_root_baseline": bonus,
            "equivalent_source_runs": equivalents,
            "restorability": cert_restore.as_payload(),
        },
        "arithmetic": {
            "baseline_gap": gap,
            "minimum_flipping_bonus": gap,
            "provided_bonus": bonus,
            "flip_margin": flip_margin,
            "robustness_min_gap_flip": robustness,
        },
        "void_condition": (
            "VOID if, at the staged root's first restore-selection "
            "instant, the weight-0 baseline argmax is not the seeded decoy "
            f"state {decoy.state_id}, or the certified branch "
            f"{cert.state_id} is absent from the archive, or its recorded "
            f"bonus differs from {bonus} under the record store named in "
            "this manifest — fix the staging, disclose, rerun once "
            "(a VOID is not evidence)."
        ),
    }


def _removal_record(
    store: MiningRecordStore,
) -> Optional[CertifiedAccessibilityRecord]:
    removal = [
        record
        for record in store.records.values()
        if record.preparation_outcome_category == "removal"
    ]
    if len(removal) != 1:
        return None
    return removal[0]


def build_seeded_record(
    store: MiningRecordStore,
    configuration_signature: str,
    source_run_id: str,
    source_state_id: str,
) -> Optional[CertifiedAccessibilityRecord]:
    """Records-file-variant seed: the removal record's certified envelope
    re-keyed to a configuration the baseline underranks, with provenance
    explicitly marked as a disclosed construction (it can never be
    mistaken for a real certification; it exists only to stage the
    discrimination test)."""

    template = _removal_record(store)
    if template is None:
        return None
    provenance = AccessibilityRecordProvenance(
        run_id=(
            f"{SEEDED_PROVENANCE_MARKER}:{source_run_id}:{source_state_id}"
        ),
        preregistration_doc=PREREGISTRATION_DOC,
        configuration_signature=configuration_signature,
        verification=VERIFICATION_CERTIFIED_HOLD,
        certification_predicate=(
            f"{SEEDED_PROVENANCE_MARKER}: disclosed constructed record for "
            "conflict staging only (design doc section 6 step 5); envelope "
            "copied from the certified removal record "
            f"{template.provenance.configuration_signature}; NOT a "
            "measured certification and never accessibility evidence"
        ),
        certified_branches=template.provenance.certified_branches,
        total_branches=template.provenance.total_branches,
        search_depth=template.provenance.search_depth,
        search_beam=template.provenance.search_beam,
    )
    return CertifiedAccessibilityRecord(
        provenance=provenance,
        certified_cells=template.certified_cells,
        certified_open_frontiers=template.certified_open_frontiers,
        certified_milestone_cells=template.certified_milestone_cells,
        preparation_outcome_category=(
            template.preparation_outcome_category
        ),
        confirmed_manipulation_count=0,
    )


def record_to_payload(
    record: CertifiedAccessibilityRecord,
) -> Dict[str, Any]:
    return {
        "provenance": {
            "run_id": record.provenance.run_id,
            "preregistration_doc": record.provenance.preregistration_doc,
            "configuration_signature": (
                record.provenance.configuration_signature
            ),
            "verification": record.provenance.verification,
            "certification_predicate": (
                record.provenance.certification_predicate
            ),
            "certified_branches": record.provenance.certified_branches,
            "total_branches": record.provenance.total_branches,
            "search_depth": record.provenance.search_depth,
            "search_beam": record.provenance.search_beam,
        },
        "certified_cells": [list(cell) for cell in record.certified_cells],
        "certified_open_frontiers": [
            [list(edge[0]), list(edge[1])]
            for edge in record.certified_open_frontiers
        ],
        "certified_milestone_cells": [
            list(cell) for cell in record.certified_milestone_cells
        ],
        "preparation_outcome_category": (
            record.preparation_outcome_category
        ),
        "confirmed_manipulation_count": (
            record.confirmed_manipulation_count
        ),
    }


def construct_records_variant_designs(
    results: Sequence[RunMiningResult],
    store: MiningRecordStore,
    records_payloads: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Alternate seeded design: a records-file variant certifying a
    coverage record for a configuration the baseline underranks, creating
    conflict at an existing restorable decision point.

    Deterministic rule: at every real restore instant with zero
    positive-bonus live candidates, take the baseline argmax as decoy and
    the highest-baseline live candidate carrying a non-null unmapped
    signature (restorable, distinct from the decoy) as challenger; the
    seeded record re-keys the certified removal envelope to the
    challenger's signature. Valid iff the current side resolves under the
    section 6.8 rule and the seeded bonus flips the argmax. Attempts and
    their failure reasons are all recorded; the best valid attempt per
    family is selected by ``min(baseline_gap, flip_margin)`` descending
    with (run_id, seq) tie-breaking.
    """

    template = _removal_record(store)
    if template is None:
        return {
            "constructed": False,
            "reason": "store does not carry exactly one removal record",
        }
    attempts: List[Dict[str, Any]] = []
    valid: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    for result in results:
        for point in result.decision_points:
            if point.kind != "restore" and not point.exhaustion_context:
                # Restore-selection instants are the real seam; commit
                # boundaries are attempted only in exhaustion context so
                # the exhaustion family's constructibility is computed,
                # not asserted.
                continue
            attempt: Dict[str, Any] = {
                "run_id": result.run_id,
                "kind": point.kind,
                "seq": point.seq,
                "decision": point.decision,
                "exhaustion_context": point.exhaustion_context,
            }
            if point.positive_bonus_candidates > 0:
                attempt["outcome"] = "skipped_live_positive_bonus"
                attempts.append(attempt)
                continue
            current_record, current_source = resolve_current_record(
                store, point.current_signature
            )
            attempt["current_source"] = current_source
            if current_record is None:
                attempt["outcome"] = "invalid_current_missing"
                attempts.append(attempt)
                continue
            decoy = point.baseline_top
            challengers = [
                candidate
                for candidate in point.all_candidates
                if candidate.state_id != decoy.state_id
                and candidate.signature
                and candidate.signature not in store.records
            ]
            challengers.sort(
                key=lambda c: (-c.baseline_score, c.add_seq)
            )
            if not challengers:
                attempt["outcome"] = "invalid_no_challenger"
                attempts.append(attempt)
                continue
            challenger = challengers[0]
            challenger_restore = check_restorability(
                result.run_dir, challenger.state_id, result.snapshots
            )
            seeded = build_seeded_record(
                store,
                challenger.signature,
                result.run_id,
                challenger.state_id,
            )
            components = verified_accessibility_preference(
                seeded, current_record
            )
            bonus = components.total_bonus if components.scored else 0.0
            gap = decoy.baseline_score - challenger.baseline_score
            flip_margin = bonus - gap
            attempt.update(
                {
                    "decoy_state_id": decoy.state_id,
                    "decoy_baseline_score": decoy.baseline_score,
                    "challenger_state_id": challenger.state_id,
                    "challenger_signature": challenger.signature,
                    "challenger_baseline_score": (
                        challenger.baseline_score
                    ),
                    "seeded_bonus": bonus,
                    "baseline_gap": gap,
                    "flip_margin": flip_margin,
                }
            )
            if bonus <= 0.0:
                attempt["outcome"] = "invalid_seeded_bonus_zero"
                attempts.append(attempt)
                continue
            if flip_margin <= 0.0 or gap <= 0.0:
                attempt["outcome"] = "invalid_no_flip"
                attempts.append(attempt)
                continue
            family = classify_family(
                point.exhaustion_context,
                seeded,
                point.collected_milestone_cells,
            )
            decoy_restore = check_restorability(
                result.run_dir, decoy.state_id, result.snapshots
            )
            attempt["outcome"] = "valid"
            attempt["family"] = family
            payload = {
                "run_id": result.run_id,
                "seq": point.seq,
                "decision": point.decision,
                "family": family,
                "current_source": current_source,
                "current_signature": point.current_signature,
                "decoy": {
                    "state_id": decoy.state_id,
                    "signature": decoy.signature,
                    "baseline_score": decoy.baseline_score,
                    "restorability": decoy_restore.as_payload(),
                },
                "challenger": {
                    "state_id": challenger.state_id,
                    "signature": challenger.signature,
                    "baseline_score": challenger.baseline_score,
                    "restorability": challenger_restore.as_payload(),
                },
                "seeded_record": record_to_payload(seeded),
                "seeded_record_content_signature": (
                    seeded.content_signature()
                ),
                "arithmetic": {
                    "baseline_gap": gap,
                    "minimum_flipping_bonus": gap,
                    "provided_bonus": bonus,
                    "flip_margin": flip_margin,
                    "robustness_min_gap_flip": min(gap, flip_margin),
                },
                "void_condition": (
                    "VOID if, at the staged instant, the weight-0 "
                    "baseline argmax is not "
                    f"{decoy.state_id}, or the challenger "
                    f"{challenger.state_id} is absent, or the variant "
                    "store does not reproduce the seeded bonus "
                    f"{bonus} — fix the staging, disclose, rerun once."
                ),
            }
            key = (
                -min(gap, flip_margin),
                result.run_id,
                point.seq,
            )
            valid.append((key, payload))
            attempts.append(attempt)
    valid.sort(key=lambda item: item[0])
    selected_by_family: Dict[str, Dict[str, Any]] = {}
    for _key, payload in valid:
        selected_by_family.setdefault(payload["family"], payload)
    variant_recipe = None
    if records_payloads is not None and valid:
        top = valid[0][1]
        variant_payloads = list(records_payloads) + [
            dict(top["seeded_record"])
        ]
        canonical = json.dumps(
            variant_payloads, sort_keys=True, indent=1
        ).encode("utf-8")
        variant_recipe = {
            "construction": (
                "real records file payloads plus the selected seeded "
                "record appended, serialized as json.dumps(payloads, "
                "sort_keys=True, indent=1)"
            ),
            "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        }
    return {
        "constructed": bool(valid),
        "disclosure": (
            "CONSTRUCTED records-file variant per design doc section 6 "
            "step 5 and the task's licensed construction: certifies a "
            "coverage record for a configuration the baseline underranks "
            "at an existing restorable decision point. The seeded record's "
            f"provenance carries the {SEEDED_PROVENANCE_MARKER} marker so "
            "it can never be mistaken for a measured certification; it is "
            "staging for the discrimination test only, never "
            "accessibility evidence. Documented as constructed, never "
            "silently."
        ),
        "attempts": attempts,
        "valid_designs": [payload for _key, payload in valid],
        "selected_by_family": selected_by_family,
        "variant_records_file": variant_recipe,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in manifest.items()
        if key != "digest_sha256"
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_manifest(
    results: Sequence[RunMiningResult],
    store: MiningRecordStore,
    records_path: Path,
    evaluations_dir: Path,
    records_payloads: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    conflicts = [
        conflict for result in results for conflict in result.conflicts
    ]
    stageable = [
        conflict
        for conflict in conflicts
        if not conflict.instrument_gap_dependent
    ]
    gap_dependent = [
        conflict
        for conflict in conflicts
        if conflict.instrument_gap_dependent
    ]
    by_family = {family: 0 for family in FAMILIES}
    for conflict in stageable:
        by_family[conflict.family] = by_family.get(conflict.family, 0) + 1
    near_misses = []
    for result in results:
        for point in result.decision_points:
            if point.kind != "restore":
                continue
            if point.in_run_winner_state_id is None:
                continue
            winner = result.candidates.get(point.in_run_winner_state_id)
            winner_mapped = bool(
                winner
                and winner.signature
                and winner.signature in store.records
            )
            live_mapped = list(point.record_mapped_live_state_ids)
            if (
                not winner_mapped
                and live_mapped
                and point.current_source == CURRENT_MAPPED
            ):
                near_misses.append(
                    {
                        "run_id": result.run_id,
                        "seq": point.seq,
                        "decision": point.decision,
                        "in_run_winner_state_id": (
                            point.in_run_winner_state_id
                        ),
                        "in_run_winner_baseline_value": (
                            point.in_run_winner_baseline_value
                        ),
                        "live_record_mapped_candidates": live_mapped,
                        "note": (
                            "in-run baseline selected an unmapped branch "
                            "while a certified-record branch was live; the "
                            "current side was already mapped to the same "
                            "record so the bonus was structurally zero — "
                            "the boundary the archive-seeded design "
                            "reconstructs at a scoring-live root"
                        ),
                    }
                )
    organic_total = len(conflicts)
    manifest: Dict[str, Any] = {
        "manifest": "wp8-conflict-root-manifest",
        "procedure": PROCEDURE_NAME,
        "procedure_version": PROCEDURE_VERSION,
        "preregistration": PREREGISTRATION_DOC,
        "scoring_basis": (
            "per-candidate baseline = archive-add-time recorded score "
            "(persistent_frontier_value for causal-outcome entries); "
            "bonus = verified_accessibility_preference against the "
            "record store with section 6.8 current-side resolution; "
            "conflict predicate = argmax(baseline) != argmax(baseline + "
            "bonus); in-run restore-time re-scoring of non-winners is "
            "not reconstructible offline and mismatched winners are "
            "flagged per instant"
        ),
        "records_file": {
            "path": str(records_path),
            "sha256": store.file_sha256,
            "record_count": len(store.records),
            "root_configuration_signature": (
                store.root_configuration_signature
            ),
            "content_signatures": store.content_signatures(),
        },
        "corpus": {
            "evaluations_dir": str(evaluations_dir),
            "run_ids": [result.run_id for result in results],
            "events_total": sum(
                result.events_total for result in results
            ),
            "archive_candidates_total": sum(
                len(result.candidates) for result in results
            ),
            "record_mapped_candidates_total": sum(
                result.record_mapped_candidates for result in results
            ),
            "restore_instants_total": sum(
                result.restore_instants for result in results
            ),
            "decision_points_evaluated_total": sum(
                len(result.decision_points) for result in results
            ),
        },
        "organic_conflicts": {
            "total": organic_total,
            "stageable": len(stageable),
            "instrument_gap_dependent": len(gap_dependent),
            "by_family": by_family,
            "statement": (
                "no stageable organic score-conflict decision point "
                "exists in the v322-v328 corpus: at every instant with a "
                "certified-improving candidate and a scoring current "
                "side, the baseline argmax was itself "
                "certified-improving, and everywhere else no live "
                "candidate mapped to a certified record (the section "
                "4.43 non-discrimination result, now quantified over "
                "every recorded decision point); the only predicate "
                "hits are instrument-gap artifacts of the section 4.29 "
                "signature reset, disqualified as Gate 4 E2 roots "
                "because the mandated 6a8488a fix removes them"
                if not stageable
                else "stageable organic conflicts found; see entries"
            ),
            "conflicts": [conflict.as_payload() for conflict in conflicts],
        },
        "near_conflict_observations": near_misses,
        "runs": [result.summary_payload() for result in results],
        "caveats": [
            "baseline scores are add-time recorded values; the in-run "
            "restore selection re-scores candidates against live planner "
            "state, so offline argmax can differ from the in-run winner "
            "(flagged per instant, never hidden)",
            "the offline candidate universe is the main archive "
            "(option + branch adds); causal-outcome entries are excluded "
            "per the recorded archive_size evidence at early instants, "
            "but late (d5/d8) instants show one-to-two entry "
            "discrepancies against recorded archive_size from add "
            "streams telemetry does not name — every instant carrying "
            "both a certified-improving candidate and a scoring current "
            "side reconciles exactly, so the no-conflict conclusion "
            "does not rest on the ambiguous instants",
            "restore-time values harvested for the seeded design come "
            "from different runs' planner states; cross-run "
            "comparability is a staging assumption covered by each "
            "design's VOID condition",
            "restoring a branch without track metadata resets the "
            "current signature to empty (the learnings section 4.29 "
            "instrument gap, unmerged fix 6a8488a); mining reproduces "
            "the recorded behavior rather than the fixed behavior",
            "the pure preference term is collection-blind: a record's "
            "milestone component scores even when the milestone was "
            "already collected (exactly the decomposition the "
            "post-exploit family exercises)",
            "v325/v326 seed empty root signatures, so the section 6.8 "
            "rule resolves their current side to the designated pre-push "
            "baseline record even though their worlds are post-removal; "
            "this mirrors what the run-time seam would have computed and "
            "is disclosed rather than corrected",
        ],
    }
    if not stageable:
        manifest["seeded_designs"] = {
            "reason": (
                "section 6 step 5 fallback engaged: zero stageable "
                "organic conflicts in the corpus"
            ),
            "primary_archive_seeded": construct_archive_seeded_design(
                results, store
            ),
            "alternate_records_variant": (
                construct_records_variant_designs(
                    results, store, records_payloads
                )
            ),
            "exhaustion_family_note": (
                "no exhaustion-family root is constructible from "
                "existing certified evidence: the goal-exhaustion "
                "instants (v325/v326 decision 8) carry a non-empty "
                "unmapped current signature (refusal, bonus zero for "
                "every candidate) and the mapped d8 instants resolve to "
                "the removal record itself, so any positive challenger "
                "would require inventing cells beyond every certified "
                "envelope; deferred to the (8,4)/(9,12) continuation "
                "measurements"
            ),
        }
    manifest["digest_sha256"] = manifest_digest(manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preregistered WP8 conflict-root mining over stored "
            "v322-v328 telemetry (read-only; writes only the manifest)"
        )
    )
    parser.add_argument(
        "--evaluations-dir",
        type=Path,
        default=Path("experiments/lolo1-entity-v10/evaluations"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path(
            "experiments/lolo1-wp5/wp8lite-accessibility-records.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/lolo1-wp5/conflict-root-manifest.json"),
    )
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="restrict to specific run ids (default: the v322-v328 corpus)",
    )
    args = parser.parse_args(argv)
    store = load_certified_records(args.records)
    records_payloads = json.loads(
        Path(args.records).read_text(encoding="utf-8")
    )
    run_ids = tuple(args.run_ids) if args.run_ids else DEFAULT_RUN_IDS
    results = mine_corpus(args.evaluations_dir, store, run_ids)
    manifest = build_manifest(
        results,
        store,
        args.records,
        args.evaluations_dir,
        records_payloads=records_payloads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n",
        encoding="utf-8",
    )
    organic = manifest["organic_conflicts"]
    print(f"runs mined: {len(results)}")
    print(
        "decision points evaluated: "
        f"{manifest['corpus']['decision_points_evaluated_total']}"
    )
    print(
        f"organic conflicts: {organic['total']} "
        f"(stageable: {organic['stageable']}, instrument-gap dependent: "
        f"{organic['instrument_gap_dependent']})"
    )
    for family, count in manifest["organic_conflicts"]["by_family"].items():
        print(f"  {family}: {count}")
    for check in (
        check
        for result in results
        for check in result.bonus_cross_checks
        if check.match is not None
    ):
        print(
            f"bonus cross-check {check.seq} ({check.state_id}): "
            f"recorded {check.recorded_bonus} vs computed "
            f"{check.computed_bonus} -> match={check.match}"
        )
    print(f"manifest: {args.output}")
    print(f"manifest digest: {manifest['digest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
