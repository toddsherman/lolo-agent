"""WP6 productized certified-accessibility instrument (roadmap section 17 item 4).

Pure, stdlib-only encoding of the methodology that closed Gate 3
(``docs/object-removed-probe-2026-08-16.md``): certified configuration-held
paired probes measured over ``human_prior_option_branch_verified`` telemetry
records carrying the ``anonymous_object_track_*`` block.  Task G scope: this
module has no policy authority.  It never touches an emulator, a planner, or
a file; every function maps telemetry record dicts (or values derived from
them) to frozen result dataclasses so the next paired probe is scriptable
instead of hand-scored.

Methodology provenance
----------------------

- Certification predicate (``docs/paired-accessibility-probe-2026-08-16.md``
  section 7, ``docs/object-removed-probe-2026-08-16.md``): a branch is
  configuration-held iff its ``anonymous_object_track_cells`` equal the
  root's tracked cells and its tracked/confirmed state signatures match the
  root's.
- Causal-restore validity window (``docs/object-removed-probe-2026-08-16.md``,
  learnings section 4.29): ``archive_branch_restored`` events carry no track
  fields and silently reset the tracker, so certification is valid only for
  branches ordered before the first such restore; later branches are
  reported but never certified (they are censored, not classified).
- Footprint exclusion (``docs/direction-review-2026-08-16.md`` Amendment A):
  the vacated-cell delta is trivially nonzero and proves nothing, so every
  delta claim excludes a declared footprint set.  ``delta`` therefore
  requires the footprint as an explicit argument.
- Censoring discipline (learnings section 2/4.14, roadmap WP6): a
  budget-exhausted non-reach is censored scoped evidence, never
  "unreachable".  Result fields are named for what was measured (reach) and
  carry explicit censoring markers.
- Repetition gate (``docs/object-removed-probe-2026-08-16.md`` repetition
  preregistration): substantial agreement is >= 0.8 Jaccard between the
  certified coverage sets of two runs of the same configuration.

Input shape
-----------

The record dicts are exactly what the probe runs emit (v324-v326):
``human_prior_option_branch_verified`` events with ``seq``, ``decision``,
``human_prior_target_player_slot``, ``frame_width``/``frame_height``, the
``anonymous_object_track_*`` keys, and
``human_prior_option_tracked_world_state_signature``.  Root records are
``decision_committed``-shaped dicts (or any mapping with the same track
keys); a root missing the track keys seeds empty, which is
correct-by-construction for hold certification (v325 resume path).  A
*branch* missing the track keys predates the instrument fix and can never
be certified.

Anonymous terminology only: no supplied object or game identities appear
anywhere in this module; transformation, displacement, removal, and
expulsion are the sanctioned outcome categories referenced by the probe
documents.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

Cell = Tuple[int, int]

INSTRUMENT_VERSION = 1

GRID_COLUMNS = 16
GRID_ROWS = 15

BRANCH_EVENT = "human_prior_option_branch_verified"
CAUSAL_RESTORE_EVENT = "archive_branch_restored"

TRACK_CELLS_KEY = "anonymous_object_track_cells"
TRACKED_STATE_SIGNATURE_KEY = "human_prior_option_tracked_world_state_signature"
CONFIRMED_EFFECT_SIGNATURE_KEY = (
    "anonymous_object_track_confirmed_world_effect_signature"
)

CERTIFIED_HELD = "certified_configuration_held"
CONFIGURATION_DEPARTED = "configuration_departed"
CERTIFICATION_CENSORED = "certification_censored"

REASON_TRACK_CELLS_DEPARTED = "track_cells_departed"
REASON_TRACKED_STATE_SIGNATURE_DEPARTED = "tracked_state_signature_departed"
REASON_CONFIRMED_EFFECT_SIGNATURE_DEPARTED = (
    "confirmed_effect_signature_departed"
)
REASON_POST_CAUSAL_RESTORE = "post_causal_restore"
REASON_MISSING_TRACK_KEYS = "missing_track_keys"
REASON_UNORDERED_AGAINST_RESTORE = "unordered_against_causal_restore"

# Gate 3 closure criterion: certified coverage agreement across a repetition
# from a fresh restore (object-removed probe repetition preregistration).
REPETITION_JACCARD_THRESHOLD = 0.8


def _content_signature(kind: str, canonical: Mapping[str, Any]) -> str:
    """Deterministic digest of a canonical payload with a versioned prefix."""

    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    prefix = f"{kind}:v{INSTRUMENT_VERSION}:"
    return hashlib.sha256((prefix + payload).encode("utf-8")).hexdigest()


def _canonical_cells(cells: Sequence[Cell]) -> List[List[int]]:
    return [[int(column), int(row)] for column, row in cells]


def _round_floats(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item, digits) for item in value]
    return value


def normalized_cells(value: Any) -> Optional[Tuple[Cell, ...]]:
    """Sorted cell tuple from a telemetry cell list; ``None`` if absent.

    The telemetry serializes cells as ``[[column, row], ...]``.  ``None``
    is returned only for a missing value so callers can distinguish "keys
    absent" (uncertifiable branch) from "empty track state" (``[]``).
    """

    if value is None:
        return None
    return tuple(
        sorted((int(cell[0]), int(cell[1])) for cell in value)
    )


def normalized_signature(value: Any) -> str:
    """Signature string with ``None``/missing normalized to empty."""

    return "" if value is None else str(value)


def branch_endpoint_cell(record: Mapping[str, Any]) -> Optional[Cell]:
    """Coarse endpoint player cell of a branch record, if recorded.

    Uses the same pixel-slot-to-grid derivation as the probe analyses
    (16x15 coarse grid over the recorded frame dimensions).  Branches
    without a detected player slot return ``None`` and are counted, not
    silently dropped, by :func:`coverage_from_branches`.
    """

    slot = record.get("human_prior_target_player_slot")
    width = record.get("frame_width")
    height = record.get("frame_height")
    if slot is None or not width or not height:
        return None
    return (
        int(slot[0]) * GRID_COLUMNS // int(width),
        int(slot[1]) * GRID_ROWS // int(height),
    )


@dataclass(frozen=True)
class CertificationWindow:
    """Validity window bounded by the first causal-archive restore.

    ``archive_branch_restored`` events carry no track fields and silently
    reset the tracker (learnings section 4.29), so hold certification is
    valid only for branches ordered strictly before the first such restore.
    An unbounded window (both fields ``None``) means no causal restore was
    observed in the analyzed stream.
    """

    restore_seq: Optional[int] = None
    restore_decision: Optional[int] = None

    @property
    def bounded(self) -> bool:
        return self.restore_seq is not None or self.restore_decision is not None

    def admits(self, record: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
        """Whether a branch record is inside the validity window.

        Ordering uses event ``seq`` when both sides carry it, falling back
        to strict ``decision`` ordering (the preregistration's phrasing:
        decisions before the first restore).  A branch that cannot be
        ordered against a bounded window is conservatively excluded.
        """

        if not self.bounded:
            return True, None
        seq = record.get("seq")
        if seq is not None and self.restore_seq is not None:
            if int(seq) < int(self.restore_seq):
                return True, None
            return False, REASON_POST_CAUSAL_RESTORE
        decision = record.get("decision")
        if decision is not None and self.restore_decision is not None:
            if int(decision) < int(self.restore_decision):
                return True, None
            return False, REASON_POST_CAUSAL_RESTORE
        return False, REASON_UNORDERED_AGAINST_RESTORE

    def _canonical(self) -> Dict[str, Any]:
        return {
            "restore_seq": self.restore_seq,
            "restore_decision": self.restore_decision,
        }


def certification_window(
    events: Iterable[Mapping[str, Any]],
) -> CertificationWindow:
    """Locate the first causal-archive restore in an event stream.

    Returns an unbounded window when the stream contains no
    ``archive_branch_restored`` event.  Callers analyzing partial streams
    must pass the full run's window explicitly; a filtered stream cannot
    prove the absence of a restore.
    """

    for event in events:
        if event.get("event") != CAUSAL_RESTORE_EVENT:
            continue
        seq = event.get("seq")
        decision = event.get("decision")
        return CertificationWindow(
            restore_seq=None if seq is None else int(seq),
            restore_decision=None if decision is None else int(decision),
        )
    return CertificationWindow()


@dataclass(frozen=True)
class RootTrackState:
    """Root configuration the certification predicate compares against."""

    track_cells: Tuple[Cell, ...]
    tracked_state_signature: str
    confirmed_effect_signature: str

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "RootTrackState":
        """Root track state from a ``decision_committed``-shaped record.

        A root record missing the track keys seeds empty: the v325 resume
        path carries no track keys on the decision event, so the root track
        reconstructs empty and branch effects measure against the new run's
        root frame -- correct-by-construction for hold certification
        (object-removed probe preregistration).
        """

        cells = normalized_cells(record.get(TRACK_CELLS_KEY))
        return cls(
            track_cells=cells if cells is not None else (),
            tracked_state_signature=normalized_signature(
                record.get(TRACKED_STATE_SIGNATURE_KEY)
            ),
            confirmed_effect_signature=normalized_signature(
                record.get(CONFIRMED_EFFECT_SIGNATURE_KEY)
            ),
        )

    def _canonical(self) -> Dict[str, Any]:
        return {
            "track_cells": _canonical_cells(self.track_cells),
            "tracked_state_signature": self.tracked_state_signature,
            "confirmed_effect_signature": self.confirmed_effect_signature,
        }


@dataclass(frozen=True)
class BranchCertification:
    """Outcome of the certification predicate for one branch record."""

    status: str
    reasons: Tuple[str, ...] = ()

    @property
    def certified(self) -> bool:
        return self.status == CERTIFIED_HELD


def certify_branch(
    branch_record: Mapping[str, Any],
    root_record: Union[Mapping[str, Any], RootTrackState],
    *,
    window: Optional[CertificationWindow] = None,
) -> BranchCertification:
    """Apply the preregistered certification predicate to one branch.

    Configuration-held iff the branch's ``anonymous_object_track_cells``
    equal the root's tracked cells AND its tracked world-state signature
    and confirmed world-effect signature match the root's.  Branches
    outside the causal-restore validity window, or missing the track keys
    entirely (pre-instrument-fix telemetry), are censored: reported but
    never certified and never classified as departed, because the
    instrument cannot observe their configuration.
    """

    root = (
        root_record
        if isinstance(root_record, RootTrackState)
        else RootTrackState.from_record(root_record)
    )
    if window is not None:
        admitted, reason = window.admits(branch_record)
        if not admitted:
            return BranchCertification(
                status=CERTIFICATION_CENSORED,
                reasons=(reason,) if reason else (),
            )
    branch_cells = normalized_cells(branch_record.get(TRACK_CELLS_KEY))
    if branch_cells is None:
        return BranchCertification(
            status=CERTIFICATION_CENSORED,
            reasons=(REASON_MISSING_TRACK_KEYS,),
        )
    reasons: List[str] = []
    if branch_cells != root.track_cells:
        reasons.append(REASON_TRACK_CELLS_DEPARTED)
    if (
        normalized_signature(branch_record.get(TRACKED_STATE_SIGNATURE_KEY))
        != root.tracked_state_signature
    ):
        reasons.append(REASON_TRACKED_STATE_SIGNATURE_DEPARTED)
    if (
        normalized_signature(
            branch_record.get(CONFIRMED_EFFECT_SIGNATURE_KEY)
        )
        != root.confirmed_effect_signature
    ):
        reasons.append(REASON_CONFIRMED_EFFECT_SIGNATURE_DEPARTED)
    if reasons:
        return BranchCertification(
            status=CONFIGURATION_DEPARTED, reasons=tuple(reasons)
        )
    return BranchCertification(status=CERTIFIED_HELD)


@dataclass(frozen=True)
class ProbeBudget:
    """Declared budgets a coverage measurement is scoped to.

    Non-reach at these budgets is censored scoped evidence, never
    "unreachable" (learnings section 2/4.14).  All fields are optional so
    callers declare exactly what they know; ``None`` means undeclared, not
    unlimited.
    """

    search_depth: Optional[int] = None
    beam_width: Optional[int] = None
    decisions: Optional[int] = None
    wall_clock_seconds: Optional[float] = None
    wall_clock_ceiling_seconds: Optional[float] = None
    event_count: Optional[int] = None
    event_ceiling: Optional[int] = None
    completed_within_ceilings: Optional[bool] = None

    def _canonical(self) -> Dict[str, Any]:
        return _round_floats(
            {
                "search_depth": self.search_depth,
                "beam_width": self.beam_width,
                "decisions": self.decisions,
                "wall_clock_seconds": self.wall_clock_seconds,
                "wall_clock_ceiling_seconds": (
                    self.wall_clock_ceiling_seconds
                ),
                "event_count": self.event_count,
                "event_ceiling": self.event_ceiling,
                "completed_within_ceilings": self.completed_within_ceilings,
            }
        )


@dataclass(frozen=True)
class CertifiedCoverage:
    """Reach coverage of one probe run split by certification tier.

    ``certified_cells`` is the reachable set the instrument may claim:
    endpoint cells of certified configuration-held branches.
    ``side_effect_only_cells`` were reached only by configuration-departed
    branches (a second manipulation invalidates the fixed-layout claim for
    those branches).  ``certification_censored_cells`` were reached only by
    branches the instrument could not certify (post-causal-restore or
    missing track keys) -- distinct from cells that were never reached at
    all, which appear nowhere.  Tiers are disjoint with priority certified
    > side-effect > censored.
    """

    root_state_signature: str
    root_track_state: RootTrackState
    window: CertificationWindow
    certified_cells: Tuple[Cell, ...]
    certified_cell_branch_counts: Tuple[Tuple[Cell, int], ...]
    side_effect_only_cells: Tuple[Cell, ...]
    certification_censored_cells: Tuple[Cell, ...]
    branches_total: int
    branches_certified: int
    branches_departed: int
    branches_censored: int
    branches_without_endpoint_cell: int
    budget: ProbeBudget = field(default_factory=ProbeBudget)

    def _canonical(self) -> Dict[str, Any]:
        return {
            "root_state_signature": self.root_state_signature,
            "root_track_state": self.root_track_state._canonical(),
            "window": self.window._canonical(),
            "certified_cells": _canonical_cells(self.certified_cells),
            "certified_cell_branch_counts": [
                [[int(cell[0]), int(cell[1])], int(count)]
                for cell, count in self.certified_cell_branch_counts
            ],
            "side_effect_only_cells": _canonical_cells(
                self.side_effect_only_cells
            ),
            "certification_censored_cells": _canonical_cells(
                self.certification_censored_cells
            ),
            "branches_total": self.branches_total,
            "branches_certified": self.branches_certified,
            "branches_departed": self.branches_departed,
            "branches_censored": self.branches_censored,
            "branches_without_endpoint_cell": (
                self.branches_without_endpoint_cell
            ),
            "budget": self.budget._canonical(),
        }

    @property
    def signature(self) -> str:
        """Deterministic content digest of the coverage measurement."""

        return _content_signature("certified-coverage", self._canonical())


def coverage_from_branches(
    records: Iterable[Mapping[str, Any]],
    root_record: Mapping[str, Any],
    *,
    root_state_signature: str = "",
    window: Optional[CertificationWindow] = None,
    budget: Optional[ProbeBudget] = None,
) -> CertifiedCoverage:
    """Certified reach coverage from probe telemetry records.

    ``records`` may be the full event stream or pre-filtered branch
    events; non-branch events are used only to locate the causal-restore
    validity window when ``window`` is not supplied.  When analyzing a
    partial stream, pass the full run's window explicitly -- a filtered
    stream cannot prove the absence of a restore.

    ``root_state_signature`` is the caller's identity for the root
    configuration (for example the archived save-state content digest) and
    is carried verbatim for provenance.
    """

    materialized = list(records)
    if window is None:
        window = certification_window(materialized)
    root = RootTrackState.from_record(root_record)

    certified_counts: Dict[Cell, int] = {}
    departed_cells: set = set()
    censored_cells: set = set()
    branches_total = 0
    branches_certified = 0
    branches_departed = 0
    branches_censored = 0
    branches_without_endpoint_cell = 0

    for record in materialized:
        if record.get("event") != BRANCH_EVENT:
            continue
        branches_total += 1
        certification = certify_branch(record, root, window=window)
        cell = branch_endpoint_cell(record)
        if cell is None:
            branches_without_endpoint_cell += 1
        if certification.status == CERTIFIED_HELD:
            branches_certified += 1
            if cell is not None:
                certified_counts[cell] = certified_counts.get(cell, 0) + 1
        elif certification.status == CONFIGURATION_DEPARTED:
            branches_departed += 1
            if cell is not None:
                departed_cells.add(cell)
        else:
            branches_censored += 1
            if cell is not None:
                censored_cells.add(cell)

    certified = set(certified_counts)
    side_effect_only = departed_cells - certified
    censored_only = censored_cells - certified - departed_cells
    return CertifiedCoverage(
        root_state_signature=root_state_signature,
        root_track_state=root,
        window=window,
        certified_cells=tuple(sorted(certified)),
        certified_cell_branch_counts=tuple(
            (cell, certified_counts[cell]) for cell in sorted(certified)
        ),
        side_effect_only_cells=tuple(sorted(side_effect_only)),
        certification_censored_cells=tuple(sorted(censored_only)),
        branches_total=branches_total,
        branches_certified=branches_certified,
        branches_departed=branches_departed,
        branches_censored=branches_censored,
        branches_without_endpoint_cell=branches_without_endpoint_cell,
        budget=budget if budget is not None else ProbeBudget(),
    )


@dataclass(frozen=True)
class AccessibilityDelta:
    """Roadmap section 6.5 contract, amended per section 17 item 4.

    Reachable sets are defined over certified configuration-held branches
    only, and every cell field excludes the declared footprint set (the
    vacated-cell delta is trivially nonzero and proves nothing).
    Amendments to the section 6.5 sketch, semantics preserved:

    - ``newly_unreachable_cells`` is renamed ``no_longer_reached_cells``:
      non-reach at budget is censored scoped evidence, never
      "unreachable" (``non_reach_censored`` is structurally ``True``).
    - ``verification_budget`` is superseded by the two per-side
      :class:`ProbeBudget` values.
    - The side-effect-reached and certification-censored splits of each
      side are carried so uncertified coverage is reported, not hidden.
    - ``newly_reachable_tracks`` / ``lost_reachable_tracks`` /
      ``goal_region_distance_delta`` keep the section 6.5 semantics and
      stay empty until WP2 multi-track correspondence populates them.
    """

    source_state_signature: str
    target_state_signature: str
    source_coverage_signature: str
    target_coverage_signature: str
    excluded_footprint_cells: Tuple[Cell, ...]
    newly_reachable_cells: Tuple[Cell, ...]
    no_longer_reached_cells: Tuple[Cell, ...]
    shared_certified_cells: Tuple[Cell, ...]
    source_side_effect_only_cells: Tuple[Cell, ...]
    target_side_effect_only_cells: Tuple[Cell, ...]
    source_certification_censored_cells: Tuple[Cell, ...]
    target_certification_censored_cells: Tuple[Cell, ...]
    source_budget: ProbeBudget
    target_budget: ProbeBudget
    non_reach_censored: bool = True
    newly_reachable_tracks: Tuple[str, ...] = ()
    lost_reachable_tracks: Tuple[str, ...] = ()
    goal_region_distance_delta: Tuple[Tuple[str, float], ...] = ()
    return_observed: bool = False
    return_search_exhausted: bool = False

    def _canonical(self) -> Dict[str, Any]:
        return {
            "source_state_signature": self.source_state_signature,
            "target_state_signature": self.target_state_signature,
            "source_coverage_signature": self.source_coverage_signature,
            "target_coverage_signature": self.target_coverage_signature,
            "excluded_footprint_cells": _canonical_cells(
                self.excluded_footprint_cells
            ),
            "newly_reachable_cells": _canonical_cells(
                self.newly_reachable_cells
            ),
            "no_longer_reached_cells": _canonical_cells(
                self.no_longer_reached_cells
            ),
            "shared_certified_cells": _canonical_cells(
                self.shared_certified_cells
            ),
            "source_side_effect_only_cells": _canonical_cells(
                self.source_side_effect_only_cells
            ),
            "target_side_effect_only_cells": _canonical_cells(
                self.target_side_effect_only_cells
            ),
            "source_certification_censored_cells": _canonical_cells(
                self.source_certification_censored_cells
            ),
            "target_certification_censored_cells": _canonical_cells(
                self.target_certification_censored_cells
            ),
            "source_budget": self.source_budget._canonical(),
            "target_budget": self.target_budget._canonical(),
            "non_reach_censored": self.non_reach_censored,
            "newly_reachable_tracks": list(self.newly_reachable_tracks),
            "lost_reachable_tracks": list(self.lost_reachable_tracks),
            "goal_region_distance_delta": _round_floats(
                [
                    [region, distance]
                    for region, distance in self.goal_region_distance_delta
                ]
            ),
            "return_observed": self.return_observed,
            "return_search_exhausted": self.return_search_exhausted,
        }

    @property
    def signature(self) -> str:
        """Deterministic content digest of the measured delta."""

        return _content_signature("accessibility-delta", self._canonical())


def delta(
    before: CertifiedCoverage,
    after: CertifiedCoverage,
    *,
    excluded_footprint_cells: Iterable[Cell],
) -> AccessibilityDelta:
    """Certified accessibility delta between two coverage measurements.

    ``excluded_footprint_cells`` is deliberately a required keyword: the
    manipulated object's own source/destination cells must be declared and
    are excluded from every cell field, because the vacated-cell delta is
    trivially nonzero and proves nothing (direction-review Amendment A).
    Declare an empty footprint explicitly if, and only if, the compared
    configurations genuinely share every object footprint.

    ``no_longer_reached_cells`` is censored scoped evidence at the target
    side's declared budgets -- never a claim of unreachability.
    """

    footprint = frozenset(
        (int(cell[0]), int(cell[1])) for cell in excluded_footprint_cells
    )

    def beyond_footprint(cells: Iterable[Cell]) -> frozenset:
        return frozenset(cells) - footprint

    before_certified = beyond_footprint(before.certified_cells)
    after_certified = beyond_footprint(after.certified_cells)
    return AccessibilityDelta(
        source_state_signature=before.root_state_signature,
        target_state_signature=after.root_state_signature,
        source_coverage_signature=before.signature,
        target_coverage_signature=after.signature,
        excluded_footprint_cells=tuple(sorted(footprint)),
        newly_reachable_cells=tuple(
            sorted(after_certified - before_certified)
        ),
        no_longer_reached_cells=tuple(
            sorted(before_certified - after_certified)
        ),
        shared_certified_cells=tuple(
            sorted(after_certified & before_certified)
        ),
        source_side_effect_only_cells=tuple(
            sorted(beyond_footprint(before.side_effect_only_cells))
        ),
        target_side_effect_only_cells=tuple(
            sorted(beyond_footprint(after.side_effect_only_cells))
        ),
        source_certification_censored_cells=tuple(
            sorted(beyond_footprint(before.certification_censored_cells))
        ),
        target_certification_censored_cells=tuple(
            sorted(beyond_footprint(after.certification_censored_cells))
        ),
        source_budget=before.budget,
        target_budget=after.budget,
    )


def band_cells(
    columns: Iterable[int], rows: Iterable[int]
) -> Tuple[Cell, ...]:
    """Declared target band as an explicit sorted cell set.

    Encodes the preregistration pattern "column >= 8, rows 5-7" as
    ``band_cells(range(8, GRID_COLUMNS), range(5, 8))`` so scored targets
    are fixed, enumerable cell sets rather than ad-hoc comparisons.
    """

    return tuple(
        sorted(
            (int(column), int(row)) for column in columns for row in rows
        )
    )


@dataclass(frozen=True)
class ScoredBit:
    """Result of one preregistered scored bit over certified coverage.

    ``reached`` is positive evidence: at least one certified
    configuration-held branch endpoint lies in the scored target set.
    ``non_reach_censored`` marks the negative case as censored at the
    declared budgets -- a certified NO at completed depth is
    censored-negative, never "unreachable" (probe preregistrations).
    """

    target_cells: Tuple[Cell, ...]
    excluded_footprint_cells: Tuple[Cell, ...]
    scored_cells: Tuple[Cell, ...]
    certified_cells_reached: Tuple[Cell, ...]
    certified_branch_count: int
    reached: bool
    non_reach_censored: bool
    coverage_signature: str

    def _canonical(self) -> Dict[str, Any]:
        return {
            "target_cells": _canonical_cells(self.target_cells),
            "excluded_footprint_cells": _canonical_cells(
                self.excluded_footprint_cells
            ),
            "scored_cells": _canonical_cells(self.scored_cells),
            "certified_cells_reached": _canonical_cells(
                self.certified_cells_reached
            ),
            "certified_branch_count": self.certified_branch_count,
            "reached": self.reached,
            "non_reach_censored": self.non_reach_censored,
            "coverage_signature": self.coverage_signature,
        }

    @property
    def signature(self) -> str:
        return _content_signature("scored-bit", self._canonical())


def score_target_bit(
    coverage: CertifiedCoverage,
    target_cells: Iterable[Cell],
    *,
    excluded_footprint_cells: Iterable[Cell] = (),
) -> ScoredBit:
    """Score a preregistered target bit against certified coverage.

    The bit is the probes' scored shape: does at least one certified
    configuration-held branch reach the declared target set?  Footprint
    cells are removed from the target before scoring so a target that
    overlaps the declared footprint cannot be trivially satisfied by the
    vacated cell.  Branch counts are exact because each branch record
    carries exactly one endpoint cell.
    """

    footprint = frozenset(
        (int(cell[0]), int(cell[1])) for cell in excluded_footprint_cells
    )
    target = frozenset(
        (int(cell[0]), int(cell[1])) for cell in target_cells
    )
    scored = target - footprint
    counts = dict(coverage.certified_cell_branch_counts)
    reached_cells = tuple(sorted(scored & set(counts)))
    branch_count = sum(counts[cell] for cell in reached_cells)
    reached = bool(reached_cells)
    return ScoredBit(
        target_cells=tuple(sorted(target)),
        excluded_footprint_cells=tuple(sorted(footprint)),
        scored_cells=tuple(sorted(scored)),
        certified_cells_reached=reached_cells,
        certified_branch_count=branch_count,
        reached=reached,
        non_reach_censored=not reached,
        coverage_signature=coverage.signature,
    )


def jaccard(cells_a: Iterable[Cell], cells_b: Iterable[Cell]) -> float:
    """Jaccard agreement between two cell sets.

    Two empty sets are identical and score 1.0; the repetition gate
    compares certified coverage sets, and identical envelopes must score
    perfect agreement regardless of size.
    """

    first = frozenset((int(cell[0]), int(cell[1])) for cell in cells_a)
    second = frozenset((int(cell[0]), int(cell[1])) for cell in cells_b)
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


@dataclass(frozen=True)
class RepetitionAgreement:
    """Certified-coverage agreement between a run and its repetition.

    The Gate 3 closure criterion: substantial agreement (Jaccard at or
    above the preregistered threshold) between the certified coverage sets
    of two runs from the same restored root.  Divergence is reported and
    scoped -- the disagreeing cells are carried explicitly -- not hidden.
    """

    jaccard: float
    threshold: float
    agreed: bool
    shared_cells: Tuple[Cell, ...]
    only_first_cells: Tuple[Cell, ...]
    only_second_cells: Tuple[Cell, ...]
    first_coverage_signature: str
    second_coverage_signature: str

    def _canonical(self) -> Dict[str, Any]:
        return _round_floats(
            {
                "jaccard": self.jaccard,
                "threshold": self.threshold,
                "agreed": self.agreed,
                "shared_cells": _canonical_cells(self.shared_cells),
                "only_first_cells": _canonical_cells(self.only_first_cells),
                "only_second_cells": _canonical_cells(
                    self.only_second_cells
                ),
                "first_coverage_signature": self.first_coverage_signature,
                "second_coverage_signature": self.second_coverage_signature,
            }
        )

    @property
    def signature(self) -> str:
        return _content_signature("repetition-agreement", self._canonical())


def repetition_agreement(
    first: CertifiedCoverage,
    second: CertifiedCoverage,
    *,
    threshold: float = REPETITION_JACCARD_THRESHOLD,
) -> RepetitionAgreement:
    """Score the repetition gate between two certified coverages."""

    first_cells = frozenset(first.certified_cells)
    second_cells = frozenset(second.certified_cells)
    agreement = jaccard(first_cells, second_cells)
    return RepetitionAgreement(
        jaccard=agreement,
        threshold=float(threshold),
        agreed=agreement >= float(threshold),
        shared_cells=tuple(sorted(first_cells & second_cells)),
        only_first_cells=tuple(sorted(first_cells - second_cells)),
        only_second_cells=tuple(sorted(second_cells - first_cells)),
        first_coverage_signature=first.signature,
        second_coverage_signature=second.signature,
    )
