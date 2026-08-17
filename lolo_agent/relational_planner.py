"""WP8 relational hypothesis planner — the chained-preparation slice.

ENGINEERING-ONLY until the preregistered Gate 4 experiments run. This pure
module is the extraction declared by direction-review Amendment E and
engaged by the preregistered WP8-lite FAIL (docs/learnings.md section 4.43):
deliberateness at a single restore is behaviorally redundant where novelty
and certified value agree, and the planner's real gap is sustaining and
*chaining* preparations. The module therefore implements the smallest
hypothesis-planning slice that can demonstrate chained deliberate
preparation (docs/wp8-relational-planner-design-2026-08-17.md sections 2-3):

- three hypothesis kinds — ``establish_configuration`` (realize a
  configuration with a certified accessibility record), ``hold_configuration``
  (keep it intact across verified transitions), ``exploit_configuration``
  (reach certified newly-reachable cells or milestone cells while the hold
  predicate is satisfied);
- a deterministic bounded hypothesis queue (``propose``) and a
  propose/advance state machine (``advance``) driven exclusively by
  verified-event summaries — exact outcomes override priors, so a verified
  transition contradicting the active hypothesis forces a replan, never a
  silent retry;
- the roadmap section 7 WP8 hypothesis score with EVERY component exposed
  separately, the accessibility term computed by the existing pure
  ``verified_accessibility_preference`` (inheriting its churn exclusion,
  censoring discipline, and predicted-provenance refusal: an uncertified
  record contributes exactly zero, with the refusal exposed);
- declarative :class:`RealizationObjective` values the monolith's option
  search interprets — this module never searches, never touches an
  emulator, files, telemetry streams, or planner state;
- relational, room-transferable option persistence: a stored
  :class:`RealizedOption` carries initiation/termination conditions and
  transfer-evidence counts, and structurally CANNOT carry controller
  sequences or absolute coordinates (roadmap WP8 item 8; the room-scoped
  cells live only in the in-run objective payloads and the episodic record
  store, exactly as today).

Inputs are narrow read-only views (:class:`RelationalStateView`,
:class:`ArchiveCandidateView`, :class:`TransitionRuleView`) — never planner
objects. Certified accessibility enters exclusively as
``accessibility_preference.CertifiedAccessibilityRecord`` through the
caller's provenance-checked store; the section 6.8 root/baseline
designation rule is honored through the store's duck-typed ``root_record``
attribute, so the empty in-run root signature can resolve to the
designated baseline and nothing else.

The vocabulary is anonymous: cells, configurations, frontiers, milestone
cells, and the sanctioned outcome categories. No supplied game-semantic
name and no room-specific coordinate constant appears in this module
(strict-lineage lint must report ``assisted: false``; the *records* feeding
it remain assisted-lineage, exactly as in WP8-lite).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
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
    AccessibilityPreferenceConfig,
    CertifiedAccessibilityRecord,
    verified_accessibility_preference,
)

Cell = Tuple[int, int]

AUTHORITY_OFF = "off"
AUTHORITY_TELEMETRY = "telemetry"
AUTHORITY_SELECTION = "selection"
AUTHORITIES = (AUTHORITY_OFF, AUTHORITY_TELEMETRY, AUTHORITY_SELECTION)

REALIZATION_RESTORE_ARCHIVE = "restore_archive"
REALIZATION_REACH_CELLS_UNDER_HOLD = "reach_cells_under_hold"
REALIZATION_REPRODUCE_TRANSITION = "reproduce_transition"
REALIZATION_KINDS = (
    REALIZATION_RESTORE_ARCHIVE,
    REALIZATION_REACH_CELLS_UNDER_HOLD,
    REALIZATION_REPRODUCE_TRANSITION,
)

RELATION_DIFFERS_FROM_RECORD = "differs_from_record"
RELATION_MAPS_TO_RECORD = "maps_to_record"
CONFIGURATION_RELATIONS = (
    RELATION_DIFFERS_FROM_RECORD,
    RELATION_MAPS_TO_RECORD,
)

ACHIEVED_CONFIGURATION_MAPS = "configuration_maps_to_record"
ACHIEVED_HELD_ACROSS_TRANSITION = (
    "configuration_held_across_verified_transition"
)
ACHIEVED_CERTIFIED_CELL_REACHED = "certified_target_cell_reached"
ACHIEVED_CONDITIONS = (
    ACHIEVED_CONFIGURATION_MAPS,
    ACHIEVED_HELD_ACROSS_TRANSITION,
    ACHIEVED_CERTIFIED_CELL_REACHED,
)

VIOLATED_NEVER = ""
VIOLATED_CONFIGURATION_DEPARTS = "configuration_departs_record"
VIOLATED_CONDITIONS = (VIOLATED_NEVER, VIOLATED_CONFIGURATION_DEPARTS)

ADVANCE_CONTINUE = "continue"
ADVANCE_HYPOTHESIS_ACHIEVED = "hypothesis_achieved"
ADVANCE_HOLD_VIOLATED = "hold_violated"
ADVANCE_BUDGET_EXHAUSTED = "budget_exhausted"
ADVANCE_REPLAN = "replan"
ADVANCE_OUTCOMES = (
    ADVANCE_CONTINUE,
    ADVANCE_HYPOTHESIS_ACHIEVED,
    ADVANCE_HOLD_VIOLATED,
    ADVANCE_BUDGET_EXHAUSTED,
    ADVANCE_REPLAN,
)

TERMINATED_ACHIEVED = "achieved"
TERMINATED_HOLD_VIOLATED = "hold_violated"
TERMINATED_BUDGET_EXHAUSTED = "budget_exhausted"
TERMINATED_REPLANNED = "replanned"
TERMINATED_CONTRADICTED = "contradicted"

SUMMARY_COMMITTED_DECISION = "committed_decision"
SUMMARY_ARCHIVE_RESTORE = "archive_restore"

CURRENT_SOURCE_MAPPED = "mapped"
CURRENT_SOURCE_BASELINE = "baseline"
CURRENT_SOURCE_MISSING = "missing"

REFUSAL_CURRENT_RECORD_MISSING = "current_record_missing"

LOG_FIELD_PREFIX = "relational_hypothesis_"


class HypothesisKind(Enum):
    ESTABLISH_CONFIGURATION = "establish_configuration"
    HOLD_CONFIGURATION = "hold_configuration"
    EXPLOIT_CONFIGURATION = "exploit_configuration"


@dataclass(frozen=True)
class RelationalStateView:
    """What the hypothesis layer may know about the current root."""

    configuration_signature: str
    track_set_signature: str
    player_cell: Optional[Cell]
    remaining_milestone_cells: Tuple[Cell, ...]
    decision_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "remaining_milestone_cells",
            _canonical_cells(self.remaining_milestone_cells),
        )


@dataclass(frozen=True)
class ArchiveCandidateView:
    """One archived branch, as restore-selection sees it."""

    state_id: str
    configuration_signature: str
    baseline_score: float
    verified_option: bool


@dataclass(frozen=True)
class TransitionRuleView:
    """One behavior-model rule summary (posterior, not authority)."""

    interaction_signature: str
    transition_kind: str
    posterior: float
    samples: int
    inert_probability: float
    causal_hazard_probability: float = 0.0


@dataclass(frozen=True)
class RelationalPlannerConfig:
    """Bounds and budgets of the hypothesis layer, preregistered defaults.

    The per-kind branch budgets are the ``RealizationObjective`` slices of
    the monolith's search budget; ``decision_budget`` bounds how many
    verified transitions one active hypothesis may consume before the state
    machine reports ``budget_exhausted``. ``search_cost_per_branch`` scales
    the subtractive ``search_cost`` score component from the branch budget.
    """

    max_queue: int = 4
    establish_branch_budget: int = 48
    hold_branch_budget: int = 8
    exploit_branch_budget: int = 48
    decision_budget: int = 4
    search_cost_per_branch: float = 0.001
    preference: AccessibilityPreferenceConfig = AccessibilityPreferenceConfig()

    def __post_init__(self) -> None:
        if self.max_queue <= 0:
            raise ValueError("max queue must be positive")
        if self.decision_budget <= 0:
            raise ValueError("decision budget must be positive")
        for name in (
            "establish_branch_budget",
            "hold_branch_budget",
            "exploit_branch_budget",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not (self.search_cost_per_branch >= 0.0):
            raise ValueError("search cost per branch must be non-negative")


@dataclass(frozen=True)
class InitiationCondition:
    """Relational predicate over a :class:`RelationalStateView`.

    Deliberately signature- and record-relative: configuration relation,
    certified-record availability (by content signature), chain-parent
    verification, and milestone-cell membership *relative to the record* —
    never an absolute coordinate (roadmap WP8 item 8).
    """

    configuration_relation: str
    required_record_signature: str
    requires_chain_parent_verified: bool = False
    requires_uncollected_certified_milestone: bool = False

    def __post_init__(self) -> None:
        if self.configuration_relation not in CONFIGURATION_RELATIONS:
            raise ValueError(
                "configuration relation must be one of "
                f"{CONFIGURATION_RELATIONS}"
            )
        if not self.required_record_signature:
            raise ValueError(
                "initiation requires the certified record's content "
                "signature"
            )


@dataclass(frozen=True)
class TerminationCondition:
    """Achieved / violated / budget-exhausted, in relational vocabulary."""

    achieved_when: str
    violated_when: str
    decision_budget: int

    def __post_init__(self) -> None:
        if self.achieved_when not in ACHIEVED_CONDITIONS:
            raise ValueError(
                f"achieved condition must be one of {ACHIEVED_CONDITIONS}"
            )
        if self.violated_when not in VIOLATED_CONDITIONS:
            raise ValueError(
                f"violated condition must be one of {VIOLATED_CONDITIONS}"
            )
        if self.decision_budget <= 0:
            raise ValueError("termination decision budget must be positive")


@dataclass(frozen=True)
class RealizationObjective:
    """Declarative objective the monolith's exact search interprets.

    ``payload`` may carry in-run room-scoped cells (target cell sets for
    ``reach_cells_under_hold``) exactly as the episodic record store does;
    objectives are never persisted as options. Exact save-state search
    remains the acceptance oracle; model posteriors only rank.
    """

    kind: str
    payload: Mapping[str, Any]
    branch_budget: int

    def __post_init__(self) -> None:
        if self.kind not in REALIZATION_KINDS:
            raise ValueError(
                f"realization kind must be one of {REALIZATION_KINDS}"
            )
        if self.branch_budget < 0:
            raise ValueError("branch budget must be non-negative")


@dataclass(frozen=True)
class HypothesisScore:
    """The roadmap WP8 score with every component separately exposed.

    ``accessibility_scored`` / ``accessibility_refusal_reason`` expose the
    inherited predicted-provenance refusal: an uncertified record on either
    side of the comparison contributes exactly zero, and milestone evidence
    from an uncertified record is refused with it. Subtractive components
    are stored as non-negative magnitudes and subtracted in :attr:`total`.
    """

    verified_milestone_evidence: float
    expected_accessibility_improvement: float
    information_gain: float
    option_transfer_evidence: float
    reversibility_confidence: float
    causal_terminal_risk: float
    predicted_inert_probability: float
    search_cost: float
    repeated_experiment_count: float
    accessibility_scored: bool = True
    accessibility_refusal_reason: Optional[str] = None

    @property
    def total(self) -> float:
        return (
            self.verified_milestone_evidence
            + self.expected_accessibility_improvement
            + self.information_gain
            + self.option_transfer_evidence
            + self.reversibility_confidence
            - self.causal_terminal_risk
            - self.predicted_inert_probability
            - self.search_cost
            - self.repeated_experiment_count
        )

    def log_fields(self) -> Dict[str, Any]:
        """Flat, prefixed, JSON-serializable decomposition for telemetry."""

        return {
            f"{LOG_FIELD_PREFIX}verified_milestone_evidence": (
                self.verified_milestone_evidence
            ),
            f"{LOG_FIELD_PREFIX}expected_accessibility_improvement": (
                self.expected_accessibility_improvement
            ),
            f"{LOG_FIELD_PREFIX}information_gain": self.information_gain,
            f"{LOG_FIELD_PREFIX}option_transfer_evidence": (
                self.option_transfer_evidence
            ),
            f"{LOG_FIELD_PREFIX}reversibility_confidence": (
                self.reversibility_confidence
            ),
            f"{LOG_FIELD_PREFIX}causal_terminal_risk": (
                self.causal_terminal_risk
            ),
            f"{LOG_FIELD_PREFIX}predicted_inert_probability": (
                self.predicted_inert_probability
            ),
            f"{LOG_FIELD_PREFIX}search_cost": self.search_cost,
            f"{LOG_FIELD_PREFIX}repeated_experiment_count": (
                self.repeated_experiment_count
            ),
            f"{LOG_FIELD_PREFIX}accessibility_scored": (
                self.accessibility_scored
            ),
            f"{LOG_FIELD_PREFIX}accessibility_refusal_reason": (
                self.accessibility_refusal_reason
            ),
            f"{LOG_FIELD_PREFIX}total_score": self.total,
        }


@dataclass(frozen=True)
class RelationalHypothesis:
    kind: HypothesisKind
    hypothesis_id: str
    target_configuration_signature: str
    initiation: InitiationCondition
    termination: TerminationCondition
    realization: RealizationObjective
    score: HypothesisScore
    chain_parent_id: Optional[str]
    target_is_designated_baseline: bool = False


@dataclass(frozen=True)
class HypothesisPlan:
    """Bounded, deterministically ordered queue (WP8 test requirement).

    ``achieved_ids``, ``active_decisions``, and the propose-time
    ``remaining_milestone_cells`` snapshot are the state machine's
    bookkeeping; ``advance`` returns updated copies, never mutates.
    """

    hypotheses: Tuple[RelationalHypothesis, ...]
    active_id: Optional[str]
    achieved_ids: Tuple[str, ...] = ()
    active_decisions: int = 0
    remaining_milestone_cells: Tuple[Cell, ...] = ()


@dataclass(frozen=True)
class VerifiedTransitionSummary:
    """One verified transition, as the feedback seam reports it.

    Only verified-event material appears here: committed decision
    endpoints, restore selections, and the tracked configuration signature
    after the step. Nothing predicted is ever summarized.
    """

    kind: str
    decision_index: int
    configuration_signature: str
    track_set_signature: str
    player_cell: Optional[Cell]
    remaining_milestone_cells: Tuple[Cell, ...]
    restored_state_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "remaining_milestone_cells",
            _canonical_cells(self.remaining_milestone_cells),
        )


@dataclass(frozen=True)
class HypothesisAdvance:
    """The state machine's verdict for one verified transition."""

    outcome: str
    plan: HypothesisPlan
    reason: str = ""
    achieved_id: Optional[str] = None
    realized_id: Optional[str] = None
    activated_id: Optional[str] = None
    terminated: Tuple[Tuple[str, str], ...] = ()
    collected_cells: Tuple[Cell, ...] = ()


@dataclass(frozen=True)
class RealizedOption:
    """Persistable option evidence for one realized hypothesis family.

    Structurally relational: initiation/termination conditions plus
    transfer-evidence counts. There is no field that could carry a
    controller sequence or an absolute coordinate, so no universal macro
    can be minted from one room-specific trajectory.
    """

    kind: str
    target_configuration_signature: str
    record_content_signature: str
    initiation: InitiationCondition
    termination: TerminationCondition
    transfer_evidence_count: int = 0
    attempt_count: int = 0

    def __post_init__(self) -> None:
        if self.kind not in tuple(item.value for item in HypothesisKind):
            raise ValueError("realized option kind must name a hypothesis kind")
        if self.transfer_evidence_count < 0 or self.attempt_count < 0:
            raise ValueError("option evidence counts must be non-negative")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "target_configuration_signature": (
                self.target_configuration_signature
            ),
            "record_content_signature": self.record_content_signature,
            "initiation": {
                "configuration_relation": (
                    self.initiation.configuration_relation
                ),
                "required_record_signature": (
                    self.initiation.required_record_signature
                ),
                "requires_chain_parent_verified": (
                    self.initiation.requires_chain_parent_verified
                ),
                "requires_uncollected_certified_milestone": (
                    self.initiation.requires_uncollected_certified_milestone
                ),
            },
            "termination": {
                "achieved_when": self.termination.achieved_when,
                "violated_when": self.termination.violated_when,
                "decision_budget": self.termination.decision_budget,
            },
            "transfer_evidence_count": self.transfer_evidence_count,
            "attempt_count": self.attempt_count,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RealizedOption":
        allowed = {
            "kind",
            "target_configuration_signature",
            "record_content_signature",
            "initiation",
            "termination",
            "transfer_evidence_count",
            "attempt_count",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                "realized option payload carries unsupported fields "
                f"{sorted(unknown)}: controller sequences and coordinates "
                "are structurally refused"
            )
        initiation = payload["initiation"]
        termination = payload["termination"]
        return cls(
            kind=str(payload["kind"]),
            target_configuration_signature=str(
                payload["target_configuration_signature"]
            ),
            record_content_signature=str(
                payload["record_content_signature"]
            ),
            initiation=InitiationCondition(
                configuration_relation=str(
                    initiation["configuration_relation"]
                ),
                required_record_signature=str(
                    initiation["required_record_signature"]
                ),
                requires_chain_parent_verified=bool(
                    initiation["requires_chain_parent_verified"]
                ),
                requires_uncollected_certified_milestone=bool(
                    initiation["requires_uncollected_certified_milestone"]
                ),
            ),
            termination=TerminationCondition(
                achieved_when=str(termination["achieved_when"]),
                violated_when=str(termination["violated_when"]),
                decision_budget=int(termination["decision_budget"]),
            ),
            transfer_evidence_count=int(
                payload["transfer_evidence_count"]
            ),
            attempt_count=int(payload["attempt_count"]),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_cells(cells: Any) -> Tuple[Cell, ...]:
    canonical = []
    for cell in cells:
        pair = tuple(cell)
        if len(pair) != 2 or not all(isinstance(v, int) for v in pair):
            raise ValueError("cells must be integer (x, y) pairs")
        canonical.append((pair[0], pair[1]))
    return tuple(sorted(set(canonical)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def option_key(hypothesis: RelationalHypothesis) -> str:
    """Stable evidence key: one option family per (kind, configuration)."""

    return (
        f"{hypothesis.kind.value}:"
        f"{hypothesis.target_configuration_signature}"
    )


def resolve_current_record(
    state: RelationalStateView,
    records: Mapping[str, CertifiedAccessibilityRecord],
) -> Tuple[Optional[CertifiedAccessibilityRecord], str]:
    """Resolve the current-side record, naming the resolution path.

    Mirrors the monolith's section 6.8 rule: a non-empty configuration
    signature resolves only by direct lookup; the empty in-run root
    signature resolves only to the store's designated root/baseline record
    (duck-typed ``root_record``); anything else is ``missing`` and every
    downstream comparison refuses to score.
    """

    signature = state.configuration_signature
    if signature:
        record = records.get(signature)
        if record is not None:
            return record, CURRENT_SOURCE_MAPPED
        return None, CURRENT_SOURCE_MISSING
    baseline = getattr(records, "root_record", None)
    if baseline is not None:
        return baseline, CURRENT_SOURCE_BASELINE
    return None, CURRENT_SOURCE_MISSING


def configuration_maps(
    configuration_signature: str, hypothesis: RelationalHypothesis
) -> bool:
    """Whether a verified configuration signature maps to the hypothesis.

    String equality, plus the section 6.8 equivalence: the empty in-run
    signature maps only to a target that is the store's designated
    root/baseline record. A non-empty unknown signature never maps.
    """

    if configuration_signature == hypothesis.target_configuration_signature:
        return True
    return bool(
        configuration_signature == ""
        and hypothesis.target_is_designated_baseline
    )


def _hypothesis_id(
    kind: HypothesisKind,
    target_configuration_signature: str,
    target_is_designated_baseline: bool,
    initiation: InitiationCondition,
    termination: TerminationCondition,
    realization: RealizationObjective,
    chain_parent_id: Optional[str],
) -> str:
    """Deterministic content digest naming one hypothesis (Gate 4 rule)."""

    payload = json.dumps(
        {
            "kind": kind.value,
            "target_configuration_signature": (
                target_configuration_signature
            ),
            "target_is_designated_baseline": target_is_designated_baseline,
            "initiation": {
                "configuration_relation": initiation.configuration_relation,
                "required_record_signature": (
                    initiation.required_record_signature
                ),
                "requires_chain_parent_verified": (
                    initiation.requires_chain_parent_verified
                ),
                "requires_uncollected_certified_milestone": (
                    initiation.requires_uncollected_certified_milestone
                ),
            },
            "termination": {
                "achieved_when": termination.achieved_when,
                "violated_when": termination.violated_when,
                "decision_budget": termination.decision_budget,
            },
            "realization": {
                "kind": realization.kind,
                "payload": _json_safe(realization.payload),
                "branch_budget": realization.branch_budget,
            },
            "chain_parent_id": chain_parent_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(b"relational-hypothesis:" + payload.encode()).hexdigest()


def _build_hypothesis(
    kind: HypothesisKind,
    target_configuration_signature: str,
    target_is_designated_baseline: bool,
    initiation: InitiationCondition,
    termination: TerminationCondition,
    realization: RealizationObjective,
    score: HypothesisScore,
    chain_parent_id: Optional[str],
) -> RelationalHypothesis:
    return RelationalHypothesis(
        kind=kind,
        hypothesis_id=_hypothesis_id(
            kind,
            target_configuration_signature,
            target_is_designated_baseline,
            initiation,
            termination,
            realization,
            chain_parent_id,
        ),
        target_configuration_signature=target_configuration_signature,
        initiation=initiation,
        termination=termination,
        realization=realization,
        score=score,
        chain_parent_id=chain_parent_id,
        target_is_designated_baseline=target_is_designated_baseline,
    )


def score_hypothesis_candidate(
    kind: HypothesisKind,
    candidate: CertifiedAccessibilityRecord,
    current: Optional[CertifiedAccessibilityRecord],
    state: RelationalStateView,
    config: RelationalPlannerConfig,
    realization_kind: str,
    branch_budget: int,
    rule: Optional[TransitionRuleView] = None,
    option: Optional[RealizedOption] = None,
) -> HypothesisScore:
    """Compute the WP8 hypothesis score for one candidate record.

    The accessibility term is ``verified_accessibility_preference`` with
    its refusal semantics intact; milestone evidence is refused alongside
    it whenever the candidate record is not certified (prediction can gate
    measurement, never preference). ``information_gain`` and
    ``reversibility_confidence`` are structural zeroes in this first cut,
    present and logged so the decomposition is complete.
    """

    accessibility_scored = False
    refusal_reason: Optional[str] = None
    improvement = 0.0
    milestone_evidence = 0.0
    if current is None:
        refusal_reason = REFUSAL_CURRENT_RECORD_MISSING
    else:
        components = verified_accessibility_preference(
            candidate, current, config.preference
        )
        accessibility_scored = components.scored
        refusal_reason = components.refusal_reason
        improvement = components.total_bonus
    if candidate.provenance.certified:
        milestone_evidence = float(
            len(
                set(candidate.certified_milestone_cells)
                & set(state.remaining_milestone_cells)
            )
        )
    transfer_evidence = 0.0
    repeated_experiments = 0.0
    if option is not None:
        transfer_evidence = float(option.transfer_evidence_count)
        repeated_experiments = float(
            max(0, option.attempt_count - option.transfer_evidence_count)
        )
    predicted_inert = 0.0
    terminal_risk = 0.0
    if realization_kind == REALIZATION_REPRODUCE_TRANSITION and rule is not None:
        predicted_inert = float(rule.inert_probability)
        terminal_risk = float(rule.causal_hazard_probability)
    return HypothesisScore(
        verified_milestone_evidence=milestone_evidence,
        expected_accessibility_improvement=improvement,
        information_gain=0.0,
        option_transfer_evidence=transfer_evidence,
        reversibility_confidence=0.0,
        causal_terminal_risk=terminal_risk,
        predicted_inert_probability=predicted_inert,
        search_cost=config.search_cost_per_branch * float(branch_budget),
        repeated_experiment_count=repeated_experiments,
        accessibility_scored=accessibility_scored,
        accessibility_refusal_reason=refusal_reason,
    )


def initiation_satisfied(
    hypothesis: RelationalHypothesis,
    configuration_signature: str,
    remaining_milestone_cells: Sequence[Cell],
    achieved_ids: Sequence[str],
) -> bool:
    """Evaluate one relational initiation condition against verified facts."""

    initiation = hypothesis.initiation
    maps = configuration_maps(configuration_signature, hypothesis)
    if initiation.configuration_relation == RELATION_DIFFERS_FROM_RECORD:
        if maps:
            return False
    elif not maps:
        return False
    if initiation.requires_chain_parent_verified:
        if (
            hypothesis.chain_parent_id is None
            or hypothesis.chain_parent_id not in tuple(achieved_ids)
        ):
            return False
    if initiation.requires_uncollected_certified_milestone:
        target_cells = tuple(
            tuple(cell)
            for cell in hypothesis.realization.payload.get(
                "target_cells", ()
            )
        )
        remaining = set(tuple(cell) for cell in remaining_milestone_cells)
        if not any(cell in remaining for cell in target_cells):
            return False
    return True


def option_initiation_satisfied(
    option: RealizedOption,
    state: RelationalStateView,
    records: Mapping[str, CertifiedAccessibilityRecord],
) -> bool:
    """Whether a persisted option's relational initiation matches a state.

    Purely relational: certified-record availability, the configuration
    relation, and milestone-cell membership *relative to the record* the
    state resolves to. A translated layout — the same preparation
    certified at translated cells under different signatures — matches
    identically, because no absolute coordinate and no layout-specific
    digest participates in the predicate. The option's stored
    ``record_content_signature`` is provenance for telemetry, never a
    matching key.
    """

    if option.initiation.configuration_relation == (
        RELATION_DIFFERS_FROM_RECORD
    ):
        # Establishment transfers wherever some certified record names a
        # configuration other than the current one.
        for signature in sorted(records):
            record = records[signature]
            if not record.provenance.certified:
                continue
            if signature != state.configuration_signature:
                return True
        return False
    current, _source = resolve_current_record(state, records)
    if current is None or not current.provenance.certified:
        return False
    if option.initiation.requires_uncollected_certified_milestone:
        if not (
            set(current.certified_milestone_cells)
            & set(state.remaining_milestone_cells)
        ):
            return False
    return True


def record_attempt(
    prior: Optional[RealizedOption], hypothesis: RelationalHypothesis
) -> RealizedOption:
    """Count one activation of this hypothesis family."""

    if prior is not None:
        return replace(prior, attempt_count=prior.attempt_count + 1)
    return RealizedOption(
        kind=hypothesis.kind.value,
        target_configuration_signature=(
            hypothesis.target_configuration_signature
        ),
        record_content_signature=(
            hypothesis.initiation.required_record_signature
        ),
        initiation=hypothesis.initiation,
        termination=hypothesis.termination,
        transfer_evidence_count=0,
        attempt_count=1,
    )


def record_success(
    prior: Optional[RealizedOption], hypothesis: RelationalHypothesis
) -> RealizedOption:
    """Count one verified achievement of this hypothesis family."""

    base = (
        prior
        if prior is not None
        else record_attempt(None, hypothesis)
    )
    return replace(
        base,
        transfer_evidence_count=base.transfer_evidence_count + 1,
    )


def hypothesis_log_fields(
    hypothesis: RelationalHypothesis,
) -> Dict[str, Any]:
    """Flat JSON-serializable telemetry row for one hypothesis."""

    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "hypothesis_kind": hypothesis.kind.value,
        "chain_parent_id": hypothesis.chain_parent_id,
        "target_configuration_signature": (
            hypothesis.target_configuration_signature
        ),
        "target_is_designated_baseline": (
            hypothesis.target_is_designated_baseline
        ),
        "initiation_configuration_relation": (
            hypothesis.initiation.configuration_relation
        ),
        "initiation_required_record_signature": (
            hypothesis.initiation.required_record_signature
        ),
        "initiation_requires_chain_parent_verified": (
            hypothesis.initiation.requires_chain_parent_verified
        ),
        "initiation_requires_uncollected_certified_milestone": (
            hypothesis.initiation.requires_uncollected_certified_milestone
        ),
        "termination_achieved_when": hypothesis.termination.achieved_when,
        "termination_violated_when": hypothesis.termination.violated_when,
        "termination_decision_budget": (
            hypothesis.termination.decision_budget
        ),
        "realization_kind": hypothesis.realization.kind,
        "realization_payload": _json_safe(hypothesis.realization.payload),
        "realization_branch_budget": hypothesis.realization.branch_budget,
        **hypothesis.score.log_fields(),
    }


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------


def _establish_realization(
    signature: str,
    candidate: CertifiedAccessibilityRecord,
    archive: Sequence[ArchiveCandidateView],
    rules: Sequence[TransitionRuleView],
    config: RelationalPlannerConfig,
) -> Tuple[Optional[RealizationObjective], Optional[TransitionRuleView]]:
    """Restore an archived branch carrying the signature, else reproduce.

    Deterministic: archive candidates rank by (baseline score, state id);
    behavior-model rules rank by (posterior, samples, signature) and must
    positively support the record's sanctioned outcome category.
    """

    matching = sorted(
        (
            view
            for view in archive
            if view.configuration_signature == signature
        ),
        key=lambda view: (-view.baseline_score, view.state_id),
    )
    if matching:
        return (
            RealizationObjective(
                kind=REALIZATION_RESTORE_ARCHIVE,
                payload={
                    "configuration_signature": signature,
                    "state_id": matching[0].state_id,
                },
                branch_budget=config.establish_branch_budget,
            ),
            None,
        )
    supporting = sorted(
        (
            rule
            for rule in rules
            if rule.transition_kind
            == candidate.preparation_outcome_category
            and rule.posterior > 0.0
            and rule.samples > 0
        ),
        key=lambda rule: (
            -rule.posterior,
            -rule.samples,
            rule.interaction_signature,
        ),
    )
    if supporting:
        rule = supporting[0]
        return (
            RealizationObjective(
                kind=REALIZATION_REPRODUCE_TRANSITION,
                payload={
                    "interaction_signature": rule.interaction_signature,
                    "expected_transition_kind": rule.transition_kind,
                },
                branch_budget=config.establish_branch_budget,
            ),
            rule,
        )
    return None, None


def _exploit_target_cells(
    candidate: CertifiedAccessibilityRecord,
    current: Optional[CertifiedAccessibilityRecord],
    state: RelationalStateView,
) -> Tuple[Tuple[Cell, ...], bool]:
    """Certified target cells for exploitation, milestone cells first.

    Returns ``(cells, milestone_targets)``. Uncollected certified
    milestone cells take priority; otherwise certified newly-reachable
    cells relative to the current record direct the frontier. Cells come
    exclusively from certified records and goal telemetry — the in-run
    objective payload is their only carrier.
    """

    if not candidate.provenance.certified:
        return (), False
    milestones = tuple(
        sorted(
            set(candidate.certified_milestone_cells)
            & set(state.remaining_milestone_cells)
        )
    )
    if milestones:
        return milestones, True
    if current is None or not current.provenance.certified:
        return (), False
    newly_reachable = tuple(
        sorted(
            set(candidate.certified_cells) - set(current.certified_cells)
        )
    )
    return newly_reachable, False


def _chain_for_candidate(
    signature: str,
    candidate: CertifiedAccessibilityRecord,
    current: Optional[CertifiedAccessibilityRecord],
    is_current: bool,
    target_is_baseline: bool,
    state: RelationalStateView,
    archive: Sequence[ArchiveCandidateView],
    rules: Sequence[TransitionRuleView],
    options_index: Mapping[str, RealizedOption],
    config: RelationalPlannerConfig,
) -> Tuple[RelationalHypothesis, ...]:
    record_signature = candidate.content_signature()
    chain: List[RelationalHypothesis] = []
    parent_id: Optional[str] = None

    if not is_current:
        realization, rule = _establish_realization(
            signature, candidate, archive, rules, config
        )
        if realization is None:
            return ()
        initiation = InitiationCondition(
            configuration_relation=RELATION_DIFFERS_FROM_RECORD,
            required_record_signature=record_signature,
        )
        termination = TerminationCondition(
            achieved_when=ACHIEVED_CONFIGURATION_MAPS,
            violated_when=VIOLATED_NEVER,
            decision_budget=config.decision_budget,
        )
        establish = _build_hypothesis(
            HypothesisKind.ESTABLISH_CONFIGURATION,
            signature,
            target_is_baseline,
            initiation,
            termination,
            realization,
            score_hypothesis_candidate(
                HypothesisKind.ESTABLISH_CONFIGURATION,
                candidate,
                current,
                state,
                config,
                realization_kind=realization.kind,
                branch_budget=realization.branch_budget,
                rule=rule,
                option=options_index.get(
                    f"{HypothesisKind.ESTABLISH_CONFIGURATION.value}:"
                    f"{signature}"
                ),
            ),
            chain_parent_id=None,
        )
        if establish.score.total <= 0.0 or (
            establish.score.verified_milestone_evidence <= 0.0
            and establish.score.expected_accessibility_improvement <= 0.0
        ):
            return ()
        chain.append(establish)
        parent_id = establish.hypothesis_id

    exploit_cells, milestone_targets = _exploit_target_cells(
        candidate, current, state
    )
    if is_current and not exploit_cells:
        # Nothing left to hold the current configuration for: fail open
        # to no hypothesis rather than churn on an empty objective.
        return ()

    hold_initiation = InitiationCondition(
        configuration_relation=RELATION_MAPS_TO_RECORD,
        required_record_signature=record_signature,
        requires_chain_parent_verified=parent_id is not None,
    )
    hold_termination = TerminationCondition(
        achieved_when=ACHIEVED_HELD_ACROSS_TRANSITION,
        violated_when=VIOLATED_CONFIGURATION_DEPARTS,
        decision_budget=config.decision_budget,
    )
    hold_realization = RealizationObjective(
        kind=REALIZATION_REACH_CELLS_UNDER_HOLD,
        payload={
            "hold_configuration_signature": signature,
            "target_cells": (),
        },
        branch_budget=config.hold_branch_budget,
    )
    hold = _build_hypothesis(
        HypothesisKind.HOLD_CONFIGURATION,
        signature,
        target_is_baseline,
        hold_initiation,
        hold_termination,
        hold_realization,
        score_hypothesis_candidate(
            HypothesisKind.HOLD_CONFIGURATION,
            candidate,
            current,
            state,
            config,
            realization_kind=REALIZATION_REACH_CELLS_UNDER_HOLD,
            branch_budget=config.hold_branch_budget,
            option=options_index.get(
                f"{HypothesisKind.HOLD_CONFIGURATION.value}:{signature}"
            ),
        ),
        chain_parent_id=parent_id,
    )
    chain.append(hold)

    if exploit_cells:
        exploit_initiation = InitiationCondition(
            configuration_relation=RELATION_MAPS_TO_RECORD,
            required_record_signature=record_signature,
            requires_chain_parent_verified=True,
            requires_uncollected_certified_milestone=milestone_targets,
        )
        exploit_termination = TerminationCondition(
            achieved_when=ACHIEVED_CERTIFIED_CELL_REACHED,
            violated_when=VIOLATED_CONFIGURATION_DEPARTS,
            decision_budget=config.decision_budget,
        )
        exploit_realization = RealizationObjective(
            kind=REALIZATION_REACH_CELLS_UNDER_HOLD,
            payload={
                "hold_configuration_signature": signature,
                "target_cells": exploit_cells,
            },
            branch_budget=config.exploit_branch_budget,
        )
        chain.append(
            _build_hypothesis(
                HypothesisKind.EXPLOIT_CONFIGURATION,
                signature,
                target_is_baseline,
                exploit_initiation,
                exploit_termination,
                exploit_realization,
                score_hypothesis_candidate(
                    HypothesisKind.EXPLOIT_CONFIGURATION,
                    candidate,
                    current,
                    state,
                    config,
                    realization_kind=REALIZATION_REACH_CELLS_UNDER_HOLD,
                    branch_budget=config.exploit_branch_budget,
                    option=options_index.get(
                        f"{HypothesisKind.EXPLOIT_CONFIGURATION.value}:"
                        f"{signature}"
                    ),
                ),
                chain_parent_id=hold.hypothesis_id,
            )
        )
    return tuple(chain)


def propose(
    state: RelationalStateView,
    records: Mapping[str, CertifiedAccessibilityRecord],
    archive: Sequence[ArchiveCandidateView],
    rules: Sequence[TransitionRuleView],
    realized_options: Sequence[RealizedOption],
    config: RelationalPlannerConfig,
) -> HypothesisPlan:
    """Propose a bounded, deterministically ordered hypothesis queue.

    Pure over its inputs: identical inputs produce a byte-identical
    :class:`HypothesisPlan`. Chains keep establish -> hold -> exploit
    ordering; chains rank by their lead hypothesis's total score with the
    target configuration signature as the deterministic tie-break. With no
    certified record — or no realization path — the queue is empty (fail
    open to nothing).
    """

    current, _current_source = resolve_current_record(state, records)
    current_signature = (
        None if current is None else current.content_signature()
    )
    root_signature = getattr(records, "root_configuration_signature", None)
    options_index = {
        f"{option.kind}:{option.target_configuration_signature}": option
        for option in realized_options
    }
    chains: List[Tuple[Tuple[float, str], Tuple[RelationalHypothesis, ...]]]
    chains = []
    for signature in sorted(records):
        candidate = records[signature]
        if not candidate.provenance.certified:
            # Structural refusal, mirrored from the preference term: a
            # predicted record can never seed a hypothesis.
            continue
        is_current = bool(
            current is not None
            and candidate.content_signature() == current_signature
        )
        chain = _chain_for_candidate(
            signature,
            candidate,
            current,
            is_current,
            bool(root_signature is not None and signature == root_signature),
            state,
            archive,
            rules,
            options_index,
            config,
        )
        if chain:
            chains.append(((-chain[0].score.total, signature), chain))
    chains.sort(key=lambda item: item[0])
    hypotheses: List[RelationalHypothesis] = []
    for _key, chain in chains:
        hypotheses.extend(chain)
    hypotheses = hypotheses[: config.max_queue]
    active_id = None
    for hypothesis in hypotheses:
        if initiation_satisfied(
            hypothesis,
            state.configuration_signature,
            state.remaining_milestone_cells,
            (),
        ):
            active_id = hypothesis.hypothesis_id
            break
    return HypothesisPlan(
        hypotheses=tuple(hypotheses),
        active_id=active_id,
        achieved_ids=(),
        active_decisions=0,
        remaining_milestone_cells=state.remaining_milestone_cells,
    )


# ---------------------------------------------------------------------------
# advance
# ---------------------------------------------------------------------------


def _chain_successor_ids(
    plan: HypothesisPlan, root_id: str
) -> Tuple[str, ...]:
    """Transitive successors of one hypothesis, in queue order."""

    successors: List[str] = []
    frontier = {root_id}
    changed = True
    while changed:
        changed = False
        for hypothesis in plan.hypotheses:
            if (
                hypothesis.chain_parent_id in frontier
                and hypothesis.hypothesis_id not in frontier
            ):
                frontier.add(hypothesis.hypothesis_id)
                successors.append(hypothesis.hypothesis_id)
                changed = True
    return tuple(successors)


def _terminate_chain(
    plan: HypothesisPlan,
    active: RelationalHypothesis,
    reason: str,
    updated_remaining: Tuple[Cell, ...],
) -> Tuple[HypothesisPlan, Tuple[Tuple[str, str], ...]]:
    """Drop the active hypothesis and its dependents.

    The surviving plan carries no active hypothesis: fallback chains are
    deliberately NOT auto-activated mid-transition — the caller's next
    propose cycle re-evaluates every chain against the verified root
    (replanning after every verified transition, design section 3.5).
    """

    dropped = {active.hypothesis_id}
    terminated: List[Tuple[str, str]] = [(active.hypothesis_id, reason)]
    for successor_id in _chain_successor_ids(plan, active.hypothesis_id):
        dropped.add(successor_id)
        terminated.append((successor_id, TERMINATED_REPLANNED))
    survivors = tuple(
        hypothesis
        for hypothesis in plan.hypotheses
        if hypothesis.hypothesis_id not in dropped
    )
    updated = replace(
        plan,
        hypotheses=survivors,
        active_id=None,
        active_decisions=0,
        remaining_milestone_cells=updated_remaining,
    )
    return updated, tuple(terminated)


def advance(
    plan: HypothesisPlan, verified: VerifiedTransitionSummary
) -> HypothesisAdvance:
    """Advance the chain state machine by one verified transition.

    Consumes only verified-event summaries. Exact outcomes override
    priors: a restore that was expected to realize an establish hypothesis
    but produced a non-mapping configuration forces ``replan`` with the
    hypothesis terminated as ``contradicted`` — never a silent retry. Hold
    violation (the configuration signature no longer maps to the held
    record, with the section 6.8 baseline equivalence honored) terminates
    the dependent chain with the reason exposed.
    """

    updated_remaining = verified.remaining_milestone_cells
    collected = tuple(
        sorted(
            set(plan.remaining_milestone_cells) - set(updated_remaining)
        )
    )
    by_id = {
        hypothesis.hypothesis_id: hypothesis
        for hypothesis in plan.hypotheses
    }
    active = by_id.get(plan.active_id) if plan.active_id else None
    if active is None:
        return HypothesisAdvance(
            outcome=ADVANCE_REPLAN,
            plan=replace(
                plan,
                active_id=None,
                remaining_milestone_cells=updated_remaining,
            ),
            reason=TERMINATED_REPLANNED,
            collected_cells=collected,
        )
    maps = configuration_maps(verified.configuration_signature, active)
    successors = tuple(
        hypothesis
        for hypothesis in plan.hypotheses
        if hypothesis.chain_parent_id == active.hypothesis_id
    )

    def achieve_and_activate(
        realized: bool,
    ) -> HypothesisAdvance:
        achieved_ids = plan.achieved_ids + (active.hypothesis_id,)
        successor = successors[0] if successors else None
        if successor is not None and initiation_satisfied(
            successor,
            verified.configuration_signature,
            updated_remaining,
            achieved_ids,
        ):
            updated = replace(
                plan,
                active_id=successor.hypothesis_id,
                achieved_ids=achieved_ids,
                active_decisions=0,
                remaining_milestone_cells=updated_remaining,
            )
            return HypothesisAdvance(
                outcome=ADVANCE_HYPOTHESIS_ACHIEVED,
                plan=updated,
                reason=TERMINATED_ACHIEVED,
                achieved_id=active.hypothesis_id,
                realized_id=active.hypothesis_id if realized else None,
                activated_id=successor.hypothesis_id,
                collected_cells=collected,
            )
        terminated: Tuple[Tuple[str, str], ...] = ()
        survivors = plan.hypotheses
        if successor is not None:
            dropped = {successor.hypothesis_id}
            terminated = (
                (successor.hypothesis_id, TERMINATED_REPLANNED),
            ) + tuple(
                (successor_id, TERMINATED_REPLANNED)
                for successor_id in _chain_successor_ids(
                    plan, successor.hypothesis_id
                )
            )
            dropped.update(item[0] for item in terminated)
            survivors = tuple(
                hypothesis
                for hypothesis in plan.hypotheses
                if hypothesis.hypothesis_id not in dropped
            )
        updated = replace(
            plan,
            hypotheses=survivors,
            active_id=None,
            achieved_ids=achieved_ids,
            active_decisions=0,
            remaining_milestone_cells=updated_remaining,
        )
        return HypothesisAdvance(
            outcome=ADVANCE_HYPOTHESIS_ACHIEVED,
            plan=updated,
            reason=TERMINATED_ACHIEVED,
            achieved_id=active.hypothesis_id,
            realized_id=active.hypothesis_id if realized else None,
            terminated=terminated,
            collected_cells=collected,
        )

    def exhaust_or_continue(outcome_reason: str) -> HypothesisAdvance:
        next_decisions = plan.active_decisions + 1
        if next_decisions >= active.termination.decision_budget:
            updated, terminated = _terminate_chain(
                plan,
                active,
                TERMINATED_BUDGET_EXHAUSTED,
                updated_remaining,
            )
            return HypothesisAdvance(
                outcome=ADVANCE_BUDGET_EXHAUSTED,
                plan=updated,
                reason=TERMINATED_BUDGET_EXHAUSTED,
                terminated=terminated,
                collected_cells=collected,
            )
        return HypothesisAdvance(
            outcome=ADVANCE_CONTINUE,
            plan=replace(
                plan,
                active_decisions=next_decisions,
                remaining_milestone_cells=updated_remaining,
            ),
            reason=outcome_reason,
            collected_cells=collected,
        )

    if active.kind is HypothesisKind.ESTABLISH_CONFIGURATION:
        if maps:
            return achieve_and_activate(realized=True)
        if (
            verified.kind == SUMMARY_ARCHIVE_RESTORE
            and active.realization.kind == REALIZATION_RESTORE_ARCHIVE
        ):
            # A restore executed and the verified configuration does not
            # map: the exact outcome contradicts the hypothesis.
            updated, terminated = _terminate_chain(
                plan, active, TERMINATED_CONTRADICTED, updated_remaining
            )
            return HypothesisAdvance(
                outcome=ADVANCE_REPLAN,
                plan=updated,
                reason=TERMINATED_CONTRADICTED,
                terminated=terminated,
                collected_cells=collected,
            )
        return exhaust_or_continue(ADVANCE_CONTINUE)

    if not maps:
        updated, terminated = _terminate_chain(
            plan, active, TERMINATED_HOLD_VIOLATED, updated_remaining
        )
        return HypothesisAdvance(
            outcome=ADVANCE_HOLD_VIOLATED,
            plan=updated,
            reason=TERMINATED_HOLD_VIOLATED,
            terminated=terminated,
            collected_cells=collected,
        )

    if active.kind is HypothesisKind.HOLD_CONFIGURATION:
        if successors:
            # Held across one verified transition with a dependent
            # successor ready: the sustain link is verified.
            return achieve_and_activate(realized=False)
        next_decisions = plan.active_decisions + 1
        if next_decisions >= active.termination.decision_budget:
            return achieve_and_activate(realized=False)
        return HypothesisAdvance(
            outcome=ADVANCE_CONTINUE,
            plan=replace(
                plan,
                active_decisions=next_decisions,
                remaining_milestone_cells=updated_remaining,
            ),
            reason=ADVANCE_CONTINUE,
            collected_cells=collected,
        )

    # Exploit under hold: achieved when a certified target cell is
    # verifiably reached or its milestone collected.
    target_cells = tuple(
        tuple(cell)
        for cell in active.realization.payload.get("target_cells", ())
    )
    reached = bool(
        verified.player_cell is not None
        and tuple(verified.player_cell) in target_cells
    )
    collected_hit = any(cell in target_cells for cell in collected)
    if reached or collected_hit:
        return achieve_and_activate(realized=True)
    return exhaust_or_continue(ADVANCE_CONTINUE)
