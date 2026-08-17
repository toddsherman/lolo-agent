"""Verified-accessibility preference term for archive/restore ranking (WP8-lite).

ENGINEERING-ONLY until the preregistered ablation passes. This pure module is
WP8-lite (docs/roadmap.md section 17 item 1): a verified-accessibility
preference term for the existing archive/restore-selection seams, adopted
under direction-review Amendment A (docs/direction-review-2026-08-16.md
section 3.A) and promoted only through the preregistered matched-budget
paired ablation in docs/wp8-lite-ablation-design-2026-08-16.md (mixed result
= FAIL). It computes a separately-loggable additive bonus comparing one
candidate configuration's certified accessibility record against the current
configuration's record. It never touches an emulator, files, telemetry
streams, or planner state; callers reduce probe results to records.

Evidence base and record semantics follow the certified paired-probe series
v322-v326 (docs/paired-accessibility-probe-2026-08-16.md,
docs/object-removed-probe-2026-08-16.md): a record's cells are coarse cells
reached under certified configuration-hold — branches whose anonymous object
track cells and tracked state signature match the probe root throughout —
never cells reached by configuration-departed branches, and never predicted
cells.

Scoring rule (WP8, docs/roadmap.md section 7): every component is exposed
separately, and unverified predicted accessibility must never score as if it
were an observed improvement. This module enforces the rule structurally:

- A record whose provenance is not ``certified_hold`` scores exactly zero,
  with the refusal exposed (``scored=False`` plus a reason), on either side
  of the comparison. Prediction can gate *measurement*, never preference.
- The success metric is hardened per Amendment A: previously-unreachable
  cells, previously-unreachable frontiers, and previously-unreachable
  milestone-bearing cells. Raw new-affordance counts are deliberately
  unscored — configuration churn can mint affordances (new frontier edges,
  confirmed manipulations) at already-reachable cells without changing what
  is reachable, and such churn must never outscore a certified new
  reachable cell. Churn evidence is exposed in the components for telemetry
  and contributes exactly zero.
- Absence is censored, never negative (docs/learnings.md section 2): a cell
  certified reachable in the current configuration but absent from the
  candidate's certified set is logged as censored, and no component may be
  negative.

The vocabulary is anonymous: cells, frontier edges, milestone events, and
the sanctioned outcome categories (transformation, displacement, removal,
expulsion). No supplied game-semantic names appear in code, fields, or
telemetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from typing import (
    AbstractSet,
    Any,
    Dict,
    FrozenSet,
    Optional,
    Tuple,
)


Cell = Tuple[int, int]
FrontierEdge = Tuple[Cell, Cell]

VERIFICATION_CERTIFIED_HOLD = "certified_hold"
VERIFICATION_PREDICTED = "predicted"
VERIFICATIONS = (VERIFICATION_CERTIFIED_HOLD, VERIFICATION_PREDICTED)

OUTCOME_TRANSFORMATION = "transformation"
OUTCOME_DISPLACEMENT = "displacement"
OUTCOME_REMOVAL = "removal"
OUTCOME_EXPULSION = "expulsion"
OUTCOME_NONE = "none"
OUTCOME_CATEGORIES = (
    OUTCOME_TRANSFORMATION,
    OUTCOME_DISPLACEMENT,
    OUTCOME_REMOVAL,
    OUTCOME_EXPULSION,
    OUTCOME_NONE,
)

REFUSAL_CANDIDATE_NOT_CERTIFIED = "candidate_not_certified"
REFUSAL_CURRENT_NOT_CERTIFIED = "current_not_certified"


def _canonical_cells(cells: Any, label: str) -> Tuple[Cell, ...]:
    canonical = []
    for cell in cells:
        pair = tuple(cell)
        if len(pair) != 2 or not all(isinstance(v, int) for v in pair):
            raise ValueError(f"{label} must contain integer (x, y) pairs")
        canonical.append((pair[0], pair[1]))
    return tuple(sorted(set(canonical)))


def _canonical_frontiers(edges: Any, label: str) -> Tuple[FrontierEdge, ...]:
    canonical = []
    for edge in edges:
        pair = tuple(edge)
        if len(pair) != 2:
            raise ValueError(f"{label} must contain (source, target) edges")
        source = _canonical_cells((pair[0],), label)[0]
        target = _canonical_cells((pair[1],), label)[0]
        if source == target:
            raise ValueError(f"{label} edges must join two distinct cells")
        canonical.append((source, target))
    return tuple(sorted(set(canonical)))


@dataclass(frozen=True)
class AccessibilityRecordProvenance:
    """Where one certified accessibility record came from.

    ``verification`` is explicit so a predicted record can never masquerade
    as a certified one; ``certification_predicate`` names the exact hold
    predicate (e.g. ``anonymous_object_track_cells == []``) under which the
    record's cells were reached. ``configuration_signature`` is the digest
    of the configuration the record describes — the key restore-selection
    seams use to match archived branches to records.
    """

    run_id: str
    preregistration_doc: str
    configuration_signature: str
    verification: str
    certification_predicate: str
    certified_branches: int
    total_branches: int
    search_depth: int
    search_beam: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("provenance requires a run id")
        if not self.preregistration_doc:
            raise ValueError("provenance requires a preregistration doc")
        if not self.configuration_signature:
            raise ValueError("provenance requires a configuration signature")
        if self.verification not in VERIFICATIONS:
            raise ValueError(
                "provenance verification must be one of "
                f"{VERIFICATIONS}, got {self.verification!r}"
            )
        if self.certified_branches < 0:
            raise ValueError("certified branch count must be non-negative")
        if self.total_branches < self.certified_branches:
            raise ValueError(
                "total branch count cannot be below the certified count"
            )
        if self.verification == VERIFICATION_CERTIFIED_HOLD:
            if not self.certification_predicate:
                raise ValueError(
                    "certified-hold provenance requires the certification "
                    "predicate"
                )
            if self.certified_branches <= 0:
                raise ValueError(
                    "certified-hold provenance requires at least one "
                    "certified branch"
                )
        if self.search_depth <= 0 or self.search_beam <= 0:
            raise ValueError("search budget dimensions must be positive")

    @property
    def certified(self) -> bool:
        return self.verification == VERIFICATION_CERTIFIED_HOLD


@dataclass(frozen=True)
class CertifiedAccessibilityRecord:
    """Cells, frontiers, and milestone cells certified for one configuration.

    ``certified_cells`` are coarse cells reached under certified
    configuration-hold at the provenance budget. ``certified_open_frontiers``
    are (source, target) edges positively probed open from a certified cell
    into a cell the probe did not commit-occupy under hold.
    ``certified_milestone_cells`` are certified cells at which a positive
    milestone event was observed within a certified branch.

    ``preparation_outcome_category`` names the sanctioned outcome category
    of the manipulation that produced the configuration and
    ``confirmed_manipulation_count`` counts confirmed manipulations observed
    during the probe. Both are provenance for telemetry only; scoring
    ignores them by design (churn resistance).
    """

    provenance: AccessibilityRecordProvenance
    certified_cells: Tuple[Cell, ...]
    certified_open_frontiers: Tuple[FrontierEdge, ...] = ()
    certified_milestone_cells: Tuple[Cell, ...] = ()
    preparation_outcome_category: str = OUTCOME_NONE
    confirmed_manipulation_count: int = 0

    def __post_init__(self) -> None:
        cells = _canonical_cells(self.certified_cells, "certified cells")
        frontiers = _canonical_frontiers(
            self.certified_open_frontiers, "certified open frontiers"
        )
        milestones = _canonical_cells(
            self.certified_milestone_cells, "certified milestone cells"
        )
        object.__setattr__(self, "certified_cells", cells)
        object.__setattr__(self, "certified_open_frontiers", frontiers)
        object.__setattr__(self, "certified_milestone_cells", milestones)
        cell_set = set(cells)
        for milestone in milestones:
            if milestone not in cell_set:
                raise ValueError(
                    "certified milestone cells must be certified cells: "
                    f"{milestone} is not"
                )
        for source, _target in frontiers:
            if source not in cell_set:
                raise ValueError(
                    "certified frontier edges must start at a certified "
                    f"cell: {source} is not"
                )
        if self.preparation_outcome_category not in OUTCOME_CATEGORIES:
            raise ValueError(
                "preparation outcome category must be one of "
                f"{OUTCOME_CATEGORIES}, got "
                f"{self.preparation_outcome_category!r}"
            )
        if self.confirmed_manipulation_count < 0:
            raise ValueError(
                "confirmed manipulation count must be non-negative"
            )

    def content_signature(self) -> str:
        """Deterministic content signature of the record."""

        parts = [
            self.provenance.run_id,
            self.provenance.preregistration_doc,
            self.provenance.configuration_signature,
            self.provenance.verification,
            self.provenance.certification_predicate,
            str(self.provenance.certified_branches),
            str(self.provenance.total_branches),
            str(self.provenance.search_depth),
            str(self.provenance.search_beam),
            ";".join(f"{x},{y}" for x, y in self.certified_cells),
            ";".join(
                f"{sx},{sy}>{tx},{ty}"
                for (sx, sy), (tx, ty) in self.certified_open_frontiers
            ),
            ";".join(f"{x},{y}" for x, y in self.certified_milestone_cells),
            self.preparation_outcome_category,
            str(self.confirmed_manipulation_count),
        ]
        payload = "|".join(parts).encode()
        return sha256(b"accessibility-record:" + payload).hexdigest()


@dataclass(frozen=True)
class AccessibilityPreferenceConfig:
    """Preregistered weights of the additive bonus.

    The defaults are the values preregistered for the WP8-lite ablation
    (docs/wp8-lite-ablation-design-2026-08-16.md); the seam multiplies the
    resulting total by its own single on/off weight, so these stay fixed
    across arms.
    """

    new_cell_weight: float = 1.0
    new_frontier_weight: float = 1.0
    new_milestone_weight: float = 8.0

    def __post_init__(self) -> None:
        for name in (
            "new_cell_weight",
            "new_frontier_weight",
            "new_milestone_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class AccessibilityPreferenceComponents:
    """The fully decomposed bonus for one candidate-vs-current comparison.

    Every scored quantity and every deliberately unscored quantity is a
    separate field so restore-selection telemetry can log the complete
    decomposition (WP8 scoring rule). ``scored`` is False when either
    record failed the certification requirement; every scored component is
    then zero and ``refusal_reason`` says why.
    """

    candidate_signature: str
    current_signature: str
    scored: bool
    refusal_reason: Optional[str]
    newly_reachable_cells: Tuple[Cell, ...]
    new_cell_bonus: float
    newly_open_frontiers: Tuple[FrontierEdge, ...]
    new_frontier_bonus: float
    newly_reachable_milestone_cells: Tuple[Cell, ...]
    new_milestone_bonus: float
    censored_current_only_cells: Tuple[Cell, ...]
    excluded_cells: Tuple[Cell, ...]
    churn_excluded_frontiers: Tuple[FrontierEdge, ...]
    candidate_confirmed_manipulation_count: int
    candidate_outcome_category: str
    total_bonus: float

    def log_fields(self) -> Dict[str, Any]:
        """Flat, prefixed, JSON-serializable fields for telemetry."""

        return {
            "verified_accessibility_candidate_signature": (
                self.candidate_signature
            ),
            "verified_accessibility_current_signature": (
                self.current_signature
            ),
            "verified_accessibility_scored": self.scored,
            "verified_accessibility_refusal_reason": self.refusal_reason,
            "verified_accessibility_new_cells": list(
                list(cell) for cell in self.newly_reachable_cells
            ),
            "verified_accessibility_new_cell_count": len(
                self.newly_reachable_cells
            ),
            "verified_accessibility_new_cell_bonus": self.new_cell_bonus,
            "verified_accessibility_new_frontiers": list(
                [list(source), list(target)]
                for source, target in self.newly_open_frontiers
            ),
            "verified_accessibility_new_frontier_count": len(
                self.newly_open_frontiers
            ),
            "verified_accessibility_new_frontier_bonus": (
                self.new_frontier_bonus
            ),
            "verified_accessibility_new_milestone_cells": list(
                list(cell) for cell in self.newly_reachable_milestone_cells
            ),
            "verified_accessibility_new_milestone_count": len(
                self.newly_reachable_milestone_cells
            ),
            "verified_accessibility_new_milestone_bonus": (
                self.new_milestone_bonus
            ),
            "verified_accessibility_censored_current_only_cells": list(
                list(cell) for cell in self.censored_current_only_cells
            ),
            "verified_accessibility_excluded_cells": list(
                list(cell) for cell in self.excluded_cells
            ),
            "verified_accessibility_churn_excluded_frontiers": list(
                [list(source), list(target)]
                for source, target in self.churn_excluded_frontiers
            ),
            "verified_accessibility_confirmed_manipulation_count": (
                self.candidate_confirmed_manipulation_count
            ),
            "verified_accessibility_outcome_category": (
                self.candidate_outcome_category
            ),
            "verified_accessibility_total_bonus": self.total_bonus,
        }


def _refused(
    candidate: CertifiedAccessibilityRecord,
    current: CertifiedAccessibilityRecord,
    reason: str,
) -> AccessibilityPreferenceComponents:
    return AccessibilityPreferenceComponents(
        candidate_signature=candidate.content_signature(),
        current_signature=current.content_signature(),
        scored=False,
        refusal_reason=reason,
        newly_reachable_cells=(),
        new_cell_bonus=0.0,
        newly_open_frontiers=(),
        new_frontier_bonus=0.0,
        newly_reachable_milestone_cells=(),
        new_milestone_bonus=0.0,
        censored_current_only_cells=(),
        excluded_cells=(),
        churn_excluded_frontiers=(),
        candidate_confirmed_manipulation_count=(
            candidate.confirmed_manipulation_count
        ),
        candidate_outcome_category=candidate.preparation_outcome_category,
        total_bonus=0.0,
    )


def verified_accessibility_preference(
    candidate: CertifiedAccessibilityRecord,
    current: CertifiedAccessibilityRecord,
    config: AccessibilityPreferenceConfig = AccessibilityPreferenceConfig(),
    excluded_cells: AbstractSet[Cell] = frozenset(),
    known_milestone_cells: AbstractSet[Cell] = frozenset(),
) -> AccessibilityPreferenceComponents:
    """Compute the verified-accessibility preference bonus.

    ``excluded_cells`` applies the paired-probe rule-1 discipline (cells —
    typically a manipulated object's own source/destination footprint —
    excluded from every claimed delta). ``known_milestone_cells`` are cells
    at which positive milestone events were previously *observed* (never
    predicted); they extend the candidate record's own certified milestone
    cells for milestone credit.

    Scoring, hardened per Amendment A:

    - new cell: certified-reachable in the candidate, not certified-reachable
      in the current configuration, not excluded;
    - new frontier: certified-open edge in the candidate, absent from the
      current record's certified frontier set, whose target cell is
      certified-reachable in *neither* configuration (a genuinely new
      boundary of reachable space) and whose cells are not excluded;
    - new milestone: new cell that carries observed milestone evidence.

    Frontier edges whose target is already certified-reachable are churn
    (minted affordances at reachable cells) and are exposed unscored.
    Censored absences are exposed unscored. No component is ever negative.
    """

    if not candidate.provenance.certified:
        return _refused(candidate, current, REFUSAL_CANDIDATE_NOT_CERTIFIED)
    if not current.provenance.certified:
        return _refused(candidate, current, REFUSAL_CURRENT_NOT_CERTIFIED)

    excluded: FrozenSet[Cell] = frozenset(
        _canonical_cells(excluded_cells, "excluded cells")
    )
    milestone_pool: FrozenSet[Cell] = frozenset(
        candidate.certified_milestone_cells
    ) | frozenset(
        _canonical_cells(known_milestone_cells, "known milestone cells")
    )

    candidate_cells = frozenset(candidate.certified_cells) - excluded
    current_cells = frozenset(current.certified_cells) - excluded

    newly_reachable = tuple(sorted(candidate_cells - current_cells))
    censored_current_only = tuple(sorted(current_cells - candidate_cells))
    newly_reachable_milestones = tuple(
        sorted(frozenset(newly_reachable) & milestone_pool)
    )

    current_frontiers = frozenset(current.certified_open_frontiers)
    newly_open = []
    churn_excluded = []
    candidate_certified_cells = frozenset(candidate.certified_cells)
    for edge in candidate.certified_open_frontiers:
        source, target = edge
        if source in excluded or target in excluded:
            continue
        if edge in current_frontiers:
            continue
        if target in current_cells or target in candidate_certified_cells:
            # An edge into space that is already certified-reachable (in
            # either configuration) is not a previously-unreachable
            # frontier: reachable targets are credited through the cell
            # component or not at all. Configuration churn mints exactly
            # these edges.
            churn_excluded.append(edge)
            continue
        newly_open.append(edge)

    new_cell_bonus = config.new_cell_weight * len(newly_reachable)
    new_frontier_bonus = config.new_frontier_weight * len(newly_open)
    new_milestone_bonus = config.new_milestone_weight * len(
        newly_reachable_milestones
    )
    total = new_cell_bonus + new_frontier_bonus + new_milestone_bonus

    return AccessibilityPreferenceComponents(
        candidate_signature=candidate.content_signature(),
        current_signature=current.content_signature(),
        scored=True,
        refusal_reason=None,
        newly_reachable_cells=newly_reachable,
        new_cell_bonus=new_cell_bonus,
        newly_open_frontiers=tuple(sorted(newly_open)),
        new_frontier_bonus=new_frontier_bonus,
        newly_reachable_milestone_cells=newly_reachable_milestones,
        new_milestone_bonus=new_milestone_bonus,
        censored_current_only_cells=censored_current_only,
        excluded_cells=tuple(sorted(excluded)),
        churn_excluded_frontiers=tuple(sorted(churn_excluded)),
        candidate_confirmed_manipulation_count=(
            candidate.confirmed_manipulation_count
        ),
        candidate_outcome_category=candidate.preparation_outcome_category,
        total_bonus=total,
    )
