"""Endpoint-relative correspondence for simultaneous anonymous tracks.

Work package 2 first cut, as amended by roadmap section 17 item 3
(endpoint-relative contract, exclusion of HUD regions and autonomous
patrol from manipulation credit) and the section 17 item 7 /
direction-review amendment E descope: at most ``K <= 4`` simultaneous
tracks, greedy minimum-cost correspondence with abstain-and-freeze on
ambiguity, no minimum-cost solver, no split/merge events, and no
hypothesis sets.  Gate 1 requires only two tracks; the 32-track
machinery of the full WP2 text is the target contract, not this module.

Contract highlights, each carrying its evidence:

- **Endpoint-relative track state** (learnings section 4.29): the
  accumulated ``anonymous_object_track_cells`` committed by v324 held
  six cells of which five had physically relaxed to baseline by
  decision 7.  ``EndpointRelativeTrackState`` therefore derives its
  current cells from present-frame evidence only and keeps an explicit
  separation between "still changed" (``current_cells``), "changed at
  some point" (``ever_changed_cells``), "relaxed back to baseline"
  (``relaxed_cells``) and "not observed at this endpoint"
  (``unobserved_cells``).  Accumulated history is provenance, never
  correspondence input.
- **Identical appearances stay distinct spatially and temporally**
  (roadmap risk register): correspondence cost combines appearance
  distance, displacement bounds, and temporal continuity, so repeated
  identical appearances resolve by nearest coherent motion, and
  equally plausible assignments abstain-and-freeze instead of
  swapping.  Appearance-only matching is the recorded Gate 1
  falsification mode.
- **Exclusion by measurement** (learnings section 4.29 decomposition):
  the HUD shot counter at cell ``(14, 5)`` and the autonomous
  patroller leak at ``(2, 6)``/``(3, 7)`` registered as accumulated
  track cells without any manipulation.  Callers supply HUD-region and
  autonomous-motion predicates; excluded observations are reported
  with their pruning reason and never earn correspondence credit.
- **Deterministic signatures**: results serialize to a canonical JSON
  payload hashed with SHA-256, and the updated track tuple hashes
  through :class:`~lolo_agent.object_tracks.ObjectTrackSet` itself so
  track-set signatures stay byte-compatible with the archive
  conventions established by work package 1.

Sanctioned outcome categories are ``displacement``, ``transformation``,
``removal``, and ``expulsion``; ``stationary`` and ``animation``
describe matched tracks whose configuration did not durably change.
This module never *confirms* a transformation: an appearance beyond the
match threshold at a persistent locus freezes the track as a
transformation candidate for work package 3's matched-control evidence.

Pure module: standard library only, frozen dataclasses, and imports
from :mod:`lolo_agent.object_tracks` (never edits to it).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .object_tracks import AnonymousObjectTrack, Cell, ObjectTrackSet

# Descoped first-cut bounds (direction-review amendment E; roadmap
# section 17 item 7).  The displacement bound restates the WP2 initial
# bound of four coarse cells per verified edge.
MAX_SIMULTANEOUS_TRACKS = 4
MAX_DISPLACEMENT_CELLS = 4
MAX_FRAME_GAP = 4

DEFAULT_APPEARANCE_MATCH_THRESHOLD = 0.5
DEFAULT_AMBIGUITY_MARGIN = 0.25
DEFAULT_TEMPORAL_WEIGHT = 0.125

# Sanctioned outcome categories.
OUTCOME_DISPLACEMENT = "displacement"
OUTCOME_TRANSFORMATION = "transformation"
OUTCOME_REMOVAL = "removal"
OUTCOME_EXPULSION = "expulsion"
# Matched, configuration-preserving categories.
OUTCOME_STATIONARY = "stationary"
OUTCOME_ANIMATION = "animation"

# Abstain-and-freeze reasons.
FREEZE_AMBIGUOUS = "ambiguous_correspondence"
FREEZE_TRACK_BOUND = "track_bound_exceeded"
FREEZE_OBSERVATION_BOUND = "observation_bound_exceeded"
FREEZE_MISSING_CURRENT_CELL = "missing_current_cell"
FREEZE_MISSING_EVIDENCE = "missing_present_evidence"
FREEZE_TRANSFORMATION_CANDIDATE = "transformation_candidate"
FREEZE_TEMPORAL_DISCONTINUITY = "temporal_discontinuity"
FREEZE_CELL_CONTESTED = "cell_contested"

# Unmatched-observation reasons.
UNMATCHED_NEW_TRACK_CANDIDATE = "new_track_candidate"
UNMATCHED_AMBIGUOUS = "ambiguous"
UNMATCHED_TRANSFORMATION_CANDIDATE = "transformation_candidate"
UNMATCHED_BOUND_ABSTENTION = "bound_abstention"

# Exclusion-hook pruning reasons.
EXCLUDED_HUD_REGION = "hud_region"
EXCLUDED_AUTONOMOUS_MOTION = "autonomous_motion"

_MISSING_CELL_SORT_KEY: Cell = (1 << 30, 1 << 30)


def _normalized_cell(value: Any) -> Cell:
    return int(value[0]), int(value[1])


def _canonical_cell(cell: Optional[Cell]) -> Optional[List[int]]:
    return None if cell is None else [int(cell[0]), int(cell[1])]


def _cell_sort_key(cell: Optional[Cell]) -> Cell:
    return _MISSING_CELL_SORT_KEY if cell is None else cell


def _rounded(value: float) -> float:
    return round(float(value), 9)


def _optional_rounded(value: Optional[float]) -> Optional[float]:
    return None if value is None else _rounded(value)


def _digest(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest()


def _appearance_key(signature: str, fingerprint: str) -> str:
    """Prefer the state signature; fall back to the fingerprint."""

    return signature or fingerprint


def _default_appearance_distance(a: str, b: str) -> float:
    return 0.0 if a == b else 1.0


@dataclass(frozen=True)
class CellEvidence:
    """Present-frame appearance evidence for one coarse cell.

    ``baseline_signature`` is the learned appearance of the same cell
    in the relaxed reference configuration; a cell is *still changed*
    exactly when its present appearance differs from that baseline.
    Both signatures come from the caller's learned pixel instruments;
    nothing here supplies semantics.
    """

    cell: Cell
    appearance_signature: str = ""
    baseline_signature: str = ""
    appearance_fingerprint: str = ""
    frame_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _normalized_cell(self.cell))
        object.__setattr__(self, "frame_index", int(self.frame_index))

    @property
    def still_changed(self) -> bool:
        return self.appearance_signature != self.baseline_signature


@dataclass(frozen=True)
class ObjectObservation:
    """One present-frame candidate anonymous object at one cell."""

    cell: Cell
    appearance_state_signature: str = ""
    appearance_fingerprint: str = ""
    frame_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _normalized_cell(self.cell))
        object.__setattr__(self, "frame_index", int(self.frame_index))

    def _canonical(self) -> Dict[str, Any]:
        return {
            "cell": _canonical_cell(self.cell),
            "appearance_state_signature": (
                self.appearance_state_signature
            ),
            "appearance_fingerprint": self.appearance_fingerprint,
            "frame_index": self.frame_index,
        }


@dataclass(frozen=True)
class EndpointRelativeTrackState:
    """Track-cell state derived from present-frame evidence only.

    The four tuples are disjoint views with an explicit "still changed
    versus changed at some point" separation (learnings section 4.29:
    five of six accumulated cells were stale).  ``ever_changed_cells``
    is provenance for the caller's records; only ``current_cells`` may
    feed correspondence or manipulation credit.
    """

    frame_index: int = 0
    current_cells: Tuple[Cell, ...] = ()
    relaxed_cells: Tuple[Cell, ...] = ()
    unobserved_cells: Tuple[Cell, ...] = ()
    ever_changed_cells: Tuple[Cell, ...] = ()

    @property
    def signature(self) -> str:
        """Deterministic content digest of the endpoint state."""

        return _digest(
            {
                "frame_index": self.frame_index,
                "current_cells": [
                    _canonical_cell(cell)
                    for cell in self.current_cells
                ],
                "relaxed_cells": [
                    _canonical_cell(cell)
                    for cell in self.relaxed_cells
                ],
                "unobserved_cells": [
                    _canonical_cell(cell)
                    for cell in self.unobserved_cells
                ],
                "ever_changed_cells": [
                    _canonical_cell(cell)
                    for cell in self.ever_changed_cells
                ],
            }
        )


def _normalized_evidence(
    present: Sequence[CellEvidence],
) -> Dict[Cell, CellEvidence]:
    """Index evidence by cell, rejecting conflicting duplicates."""

    indexed: Dict[Cell, CellEvidence] = {}
    for evidence in present:
        existing = indexed.get(evidence.cell)
        if existing is None:
            indexed[evidence.cell] = evidence
        elif existing != evidence:
            raise ValueError(
                "conflicting present-frame evidence for cell "
                f"{evidence.cell}"
            )
    frames = {evidence.frame_index for evidence in indexed.values()}
    if len(frames) > 1:
        raise ValueError(
            "present-frame evidence must share one frame index; got "
            f"{sorted(frames)}"
        )
    return indexed


def endpoint_relative_state(
    present: Sequence[CellEvidence],
    previously_changed: Iterable[Cell] = (),
    *,
    frame_index: Optional[int] = None,
) -> EndpointRelativeTrackState:
    """Derive endpoint-relative track state from present evidence.

    ``previously_changed`` is the accumulated changed-at-some-point
    record; it contributes only to ``ever_changed_cells`` (and, where
    the present frame shows baseline, ``relaxed_cells``) — never to
    ``current_cells``.
    """

    indexed = _normalized_evidence(present)
    if frame_index is None:
        frames = {
            evidence.frame_index for evidence in indexed.values()
        }
        frame_index = frames.pop() if frames else 0
    else:
        frame_index = int(frame_index)
        mismatched = {
            evidence.frame_index
            for evidence in indexed.values()
            if evidence.frame_index != frame_index
        }
        if mismatched:
            raise ValueError(
                "present-frame evidence frame indices "
                f"{sorted(mismatched)} do not match frame_index "
                f"{frame_index}"
            )
    previous = {
        _normalized_cell(cell) for cell in previously_changed
    }
    current = {
        cell
        for cell, evidence in indexed.items()
        if evidence.still_changed
    }
    observed = set(indexed)
    relaxed = {
        cell for cell in previous & observed if cell not in current
    }
    unobserved = previous - observed
    return EndpointRelativeTrackState(
        frame_index=frame_index,
        current_cells=tuple(sorted(current)),
        relaxed_cells=tuple(sorted(relaxed)),
        unobserved_cells=tuple(sorted(unobserved)),
        ever_changed_cells=tuple(sorted(previous | current)),
    )


def observations_from_evidence(
    present: Sequence[CellEvidence],
) -> Tuple[ObjectObservation, ...]:
    """Lift still-changed present-frame cells into observations."""

    indexed = _normalized_evidence(present)
    return tuple(
        ObjectObservation(
            cell=cell,
            appearance_state_signature=(
                evidence.appearance_signature
            ),
            appearance_fingerprint=evidence.appearance_fingerprint,
            frame_index=evidence.frame_index,
        )
        for cell, evidence in sorted(indexed.items())
        if evidence.still_changed
    )


def anonymous_track_id(observation: ObjectObservation) -> str:
    """Deterministic content-derived identifier for a new track."""

    return "anon-track-" + _digest(observation._canonical())[:12]


def track_from_observation(
    observation: ObjectObservation,
    *,
    track_id: Optional[str] = None,
) -> AnonymousObjectTrack:
    """Promote one unmatched observation to a fresh anonymous track."""

    return AnonymousObjectTrack(
        appearance_fingerprint=observation.appearance_fingerprint,
        current_cell=observation.cell,
        appearance_state_signature=(
            observation.appearance_state_signature
        ),
        track_id=(
            anonymous_track_id(observation)
            if track_id is None
            else track_id
        ),
        first_observed_frame=str(observation.frame_index),
        latest_observed_frame=str(observation.frame_index),
    )


def bootstrap_tracks(
    observations: Sequence[ObjectObservation],
    *,
    max_tracks: int = MAX_SIMULTANEOUS_TRACKS,
) -> Tuple[AnonymousObjectTrack, ...]:
    """Deterministically seed tracks from first observations."""

    indexed: Dict[Cell, ObjectObservation] = {}
    for observation in observations:
        existing = indexed.get(observation.cell)
        if existing is None:
            indexed[observation.cell] = observation
        elif existing != observation:
            raise ValueError(
                "conflicting observations for cell "
                f"{observation.cell}"
            )
    if len(indexed) > max_tracks:
        raise ValueError(
            f"{len(indexed)} observations exceed the "
            f"{max_tracks}-track bound"
        )
    return tuple(
        track_from_observation(observation)
        for _cell, observation in sorted(indexed.items())
    )


@dataclass(frozen=True)
class TrackAssignment:
    """One accepted track-to-observation correspondence."""

    track_id: Optional[str]
    source_cell: Cell
    target_cell: Cell
    displacement: Cell
    outcome: str
    source_appearance_signature: str = ""
    target_appearance_signature: str = ""
    appearance_cost: float = 0.0
    displacement_cost: float = 0.0
    temporal_cost: float = 0.0
    total_cost: float = 0.0
    ambiguity_margin: Optional[float] = None

    def _canonical(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "source_cell": _canonical_cell(self.source_cell),
            "target_cell": _canonical_cell(self.target_cell),
            "displacement": _canonical_cell(self.displacement),
            "outcome": self.outcome,
            "source_appearance_signature": (
                self.source_appearance_signature
            ),
            "target_appearance_signature": (
                self.target_appearance_signature
            ),
            "appearance_cost": self.appearance_cost,
            "displacement_cost": self.displacement_cost,
            "temporal_cost": self.temporal_cost,
            "total_cost": self.total_cost,
            "ambiguity_margin": self.ambiguity_margin,
        }


@dataclass(frozen=True)
class FrozenTrack:
    """One track held unchanged under abstain-and-freeze."""

    track_id: Optional[str]
    current_cell: Optional[Cell]
    reason: str
    best_cost: Optional[float] = None
    competitor_cost: Optional[float] = None

    def _canonical(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "current_cell": _canonical_cell(self.current_cell),
            "reason": self.reason,
            "best_cost": self.best_cost,
            "competitor_cost": self.competitor_cost,
        }


@dataclass(frozen=True)
class RemovedTrack:
    """One track whose cell relaxed to baseline at the endpoint."""

    track_id: Optional[str]
    last_cell: Cell
    outcome: str = OUTCOME_REMOVAL

    def _canonical(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "last_cell": _canonical_cell(self.last_cell),
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class UnmatchedObservation:
    """One observation that earned no correspondence this endpoint."""

    observation: ObjectObservation
    reason: str

    def _canonical(self) -> Dict[str, Any]:
        return {
            "observation": self.observation._canonical(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ExcludedObservation:
    """One observation pruned by a caller-supplied exclusion hook."""

    observation: ObjectObservation
    reason: str

    def _canonical(self) -> Dict[str, Any]:
        return {
            "observation": self.observation._canonical(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CorrespondenceResult:
    """Complete, deterministic outcome of one correspondence round."""

    frame_index: Optional[int] = None
    assignments: Tuple[TrackAssignment, ...] = ()
    frozen_tracks: Tuple[FrozenTrack, ...] = ()
    removed_tracks: Tuple[RemovedTrack, ...] = ()
    unmatched_observations: Tuple[UnmatchedObservation, ...] = ()
    excluded_observations: Tuple[ExcludedObservation, ...] = ()
    updated_tracks: Tuple[AnonymousObjectTrack, ...] = ()
    abstained: bool = False
    abstain_reason: str = ""

    @property
    def new_track_candidates(self) -> Tuple[ObjectObservation, ...]:
        return tuple(
            unmatched.observation
            for unmatched in self.unmatched_observations
            if unmatched.reason == UNMATCHED_NEW_TRACK_CANDIDATE
        )

    @property
    def track_set_signature(self) -> str:
        """Track-tuple digest under the ObjectTrackSet conventions."""

        return ObjectTrackSet(tracks=self.updated_tracks).signature

    @property
    def signature(self) -> str:
        """Deterministic content digest of the full result."""

        return _digest(
            {
                "frame_index": self.frame_index,
                "assignments": [
                    assignment._canonical()
                    for assignment in self.assignments
                ],
                "frozen_tracks": [
                    frozen._canonical()
                    for frozen in self.frozen_tracks
                ],
                "removed_tracks": [
                    removed._canonical()
                    for removed in self.removed_tracks
                ],
                "unmatched_observations": [
                    unmatched._canonical()
                    for unmatched in self.unmatched_observations
                ],
                "excluded_observations": [
                    excluded._canonical()
                    for excluded in self.excluded_observations
                ],
                "abstained": self.abstained,
                "abstain_reason": self.abstain_reason,
                "track_set_signature": self.track_set_signature,
            }
        )


@dataclass(frozen=True)
class _PairEvaluation:
    """Internal feasibility/cost record for one track/observation."""

    feasible: bool
    blocked_by: str = ""
    appearance_cost: float = 0.0
    displacement_cost: float = 0.0
    temporal_cost: float = 0.0
    total_cost: float = 0.0


@dataclass(frozen=True)
class _CostedPair:
    total: float
    track_index: int
    observation_index: int
    evaluation: _PairEvaluation = field(compare=False)


def _track_frame_index(track: AnonymousObjectTrack) -> Optional[int]:
    """Parse the decimal frame index this engine stores on tracks."""

    if track.latest_observed_frame is None:
        return None
    try:
        return int(str(track.latest_observed_frame))
    except ValueError:
        return None


def _evaluate_pair(
    track: AnonymousObjectTrack,
    observation: ObjectObservation,
    *,
    appearance_distance: Callable[[str, str], float],
    appearance_match_threshold: float,
    max_displacement: int,
    max_frame_gap: int,
    temporal_weight: float,
) -> _PairEvaluation:
    appearance_cost = float(
        appearance_distance(
            _appearance_key(
                track.appearance_state_signature,
                track.appearance_fingerprint,
            ),
            _appearance_key(
                observation.appearance_state_signature,
                observation.appearance_fingerprint,
            ),
        )
    )
    if appearance_cost > appearance_match_threshold:
        return _PairEvaluation(feasible=False, blocked_by="appearance")
    assert track.current_cell is not None
    manhattan = abs(
        observation.cell[0] - track.current_cell[0]
    ) + abs(observation.cell[1] - track.current_cell[1])
    if manhattan > max_displacement:
        return _PairEvaluation(
            feasible=False, blocked_by="displacement"
        )
    displacement_cost = manhattan / max(1, max_displacement)
    temporal_cost = 0.0
    track_frame = _track_frame_index(track)
    if track_frame is not None:
        gap = observation.frame_index - track_frame
        if gap < 0 or gap > max_frame_gap:
            return _PairEvaluation(
                feasible=False, blocked_by="temporal"
            )
        temporal_cost = max(0, gap - 1) * temporal_weight
    total = appearance_cost + displacement_cost + temporal_cost
    return _PairEvaluation(
        feasible=True,
        appearance_cost=_rounded(appearance_cost),
        displacement_cost=_rounded(displacement_cost),
        temporal_cost=_rounded(temporal_cost),
        total_cost=_rounded(total),
    )


def _track_sort_key(
    track: AnonymousObjectTrack,
) -> Tuple[str, Cell, str, str]:
    return (
        track.track_id or "",
        _cell_sort_key(track.current_cell),
        track.appearance_state_signature,
        track.appearance_fingerprint,
    )


def _observation_sort_key(
    observation: ObjectObservation,
) -> Tuple[Cell, str, str]:
    return (
        observation.cell,
        observation.appearance_state_signature,
        observation.appearance_fingerprint,
    )


def _updated_track(
    track: AnonymousObjectTrack,
    observation: ObjectObservation,
    delta: Cell,
) -> AnonymousObjectTrack:
    changes: Dict[str, Any] = {
        "previous_cell": track.current_cell,
        "current_cell": observation.cell,
        "persistence_steps": track.persistence_steps + 1,
        "latest_observed_frame": str(observation.frame_index),
    }
    if delta != (0, 0):
        changes["displacement"] = delta
    new_signature = observation.appearance_state_signature
    if new_signature and (
        new_signature != track.appearance_state_signature
    ):
        changes["previous_appearance_state_signature"] = (
            track.appearance_state_signature or None
        )
        changes["appearance_state_signature"] = new_signature
    if observation.appearance_fingerprint:
        changes["appearance_fingerprint"] = (
            observation.appearance_fingerprint
        )
    return replace(track, **changes)


def _boundary_cell(
    cell: Cell,
    grid_columns: Optional[int],
    grid_rows: Optional[int],
) -> bool:
    if grid_columns is None or grid_rows is None:
        return False
    return (
        cell[0] <= 0
        or cell[1] <= 0
        or cell[0] >= grid_columns - 1
        or cell[1] >= grid_rows - 1
    )


def _abstained_result(
    reason: str,
    tracks: Sequence[AnonymousObjectTrack],
    observations: Sequence[ObjectObservation],
    excluded: Sequence[ExcludedObservation],
    frame_index: Optional[int],
) -> CorrespondenceResult:
    """Freeze every track and observation under a global bound."""

    frozen = tuple(
        FrozenTrack(
            track_id=track.track_id,
            current_cell=track.current_cell,
            reason=reason,
        )
        for track in sorted(tracks, key=_track_sort_key)
    )
    unmatched = tuple(
        UnmatchedObservation(
            observation=observation,
            reason=UNMATCHED_BOUND_ABSTENTION,
        )
        for observation in sorted(
            observations, key=_observation_sort_key
        )
    )
    return CorrespondenceResult(
        frame_index=frame_index,
        frozen_tracks=frozen,
        unmatched_observations=unmatched,
        excluded_observations=tuple(excluded),
        updated_tracks=tuple(
            sorted(tracks, key=_track_sort_key)
        ),
        abstained=True,
        abstain_reason=reason,
    )


def correspond(
    tracks: Sequence[AnonymousObjectTrack],
    observations: Sequence[ObjectObservation],
    *,
    relaxed_cells: Iterable[Cell] = (),
    frame_index: Optional[int] = None,
    max_tracks: int = MAX_SIMULTANEOUS_TRACKS,
    max_displacement: int = MAX_DISPLACEMENT_CELLS,
    max_frame_gap: int = MAX_FRAME_GAP,
    appearance_distance: Optional[
        Callable[[str, str], float]
    ] = None,
    appearance_match_threshold: float = (
        DEFAULT_APPEARANCE_MATCH_THRESHOLD
    ),
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
    hud_region: Optional[Callable[[Cell], bool]] = None,
    autonomous_motion: Optional[Callable[[Cell], bool]] = None,
    grid_columns: Optional[int] = None,
    grid_rows: Optional[int] = None,
) -> CorrespondenceResult:
    """Greedy minimum-cost correspondence with abstain-and-freeze.

    ``observations`` are the still-changed cells of the present frame;
    ``relaxed_cells`` are cells positively observed at baseline this
    frame.  A track whose cell carries neither kind of evidence is
    frozen, not expired — expiration requires matched evidence, never
    one missing frame (WP2 initial bounds).  Ambiguity uses a strict
    margin: an available competing pair whose cost is within
    ``ambiguity_margin`` (strictly less) of the best pair freezes every
    involved track rather than guessing.  ``hud_region`` and
    ``autonomous_motion`` are the caller-supplied exclusion hooks of
    roadmap section 17 item 3; matching cells are pruned with an
    explicit reason before any credit is assigned.
    """

    if appearance_distance is None:
        appearance_distance = _default_appearance_distance
    tracks = tuple(tracks)
    relaxed = {_normalized_cell(cell) for cell in relaxed_cells}

    # Validate observations: one per cell, disjoint from relaxed.
    indexed_observations: Dict[Cell, ObjectObservation] = {}
    for observation in observations:
        existing = indexed_observations.get(observation.cell)
        if existing is None:
            indexed_observations[observation.cell] = observation
        elif existing != observation:
            raise ValueError(
                "conflicting observations for cell "
                f"{observation.cell}"
            )
    overlap = relaxed & set(indexed_observations)
    if overlap:
        raise ValueError(
            "cells cannot be both observed and relaxed: "
            f"{sorted(overlap)}"
        )

    if frame_index is None:
        frames = {
            observation.frame_index
            for observation in indexed_observations.values()
        }
        frame_index = frames.pop() if len(frames) == 1 else None

    # Exclusion hooks prune before any bound or cost is computed.
    excluded: List[ExcludedObservation] = []
    active_observations: List[ObjectObservation] = []
    for _cell, observation in sorted(indexed_observations.items()):
        if hud_region is not None and hud_region(observation.cell):
            excluded.append(
                ExcludedObservation(
                    observation=observation,
                    reason=EXCLUDED_HUD_REGION,
                )
            )
        elif autonomous_motion is not None and autonomous_motion(
            observation.cell
        ):
            excluded.append(
                ExcludedObservation(
                    observation=observation,
                    reason=EXCLUDED_AUTONOMOUS_MOTION,
                )
            )
        else:
            active_observations.append(observation)

    # Descoped simultaneous-track bound: abstain globally, never trim.
    if len(tracks) > max_tracks:
        return _abstained_result(
            FREEZE_TRACK_BOUND,
            tracks,
            active_observations,
            excluded,
            frame_index,
        )
    if len(active_observations) > max_tracks:
        return _abstained_result(
            FREEZE_OBSERVATION_BOUND,
            tracks,
            active_observations,
            excluded,
            frame_index,
        )

    frozen_records: List[FrozenTrack] = []
    frozen_indices: Dict[int, str] = {}
    correspondable: Dict[int, AnonymousObjectTrack] = {}
    for index, track in enumerate(tracks):
        if track.current_cell is None:
            # Appearance-only matching is the Gate 1 falsification
            # mode; a track without a cell cannot correspond spatially.
            frozen_indices[index] = FREEZE_MISSING_CURRENT_CELL
        else:
            correspondable[index] = track

    evaluations: Dict[Tuple[int, int], _PairEvaluation] = {}
    pairs: List[_CostedPair] = []
    for track_index, track in correspondable.items():
        for observation_index, observation in enumerate(
            active_observations
        ):
            evaluation = _evaluate_pair(
                track,
                observation,
                appearance_distance=appearance_distance,
                appearance_match_threshold=(
                    appearance_match_threshold
                ),
                max_displacement=max_displacement,
                max_frame_gap=max_frame_gap,
                temporal_weight=temporal_weight,
            )
            evaluations[(track_index, observation_index)] = evaluation
            if evaluation.feasible:
                pairs.append(
                    _CostedPair(
                        total=evaluation.total_cost,
                        track_index=track_index,
                        observation_index=observation_index,
                        evaluation=evaluation,
                    )
                )
    pairs.sort(
        key=lambda pair: (
            pair.total,
            _track_sort_key(tracks[pair.track_index]),
            _observation_sort_key(
                active_observations[pair.observation_index]
            ),
        )
    )

    assigned_tracks: Dict[int, Tuple[int, _PairEvaluation, Optional[float]]] = {}
    assigned_observations: Dict[int, int] = {}
    ambiguous_observations: Dict[int, bool] = {}

    def _available(pair: _CostedPair) -> bool:
        return (
            pair.track_index not in assigned_tracks
            and pair.track_index not in frozen_indices
            and pair.observation_index not in assigned_observations
            and pair.observation_index not in ambiguous_observations
        )

    for pair in pairs:
        if not _available(pair):
            continue
        competitors = [
            other
            for other in pairs
            if other is not pair
            and _available(other)
            and (
                other.track_index == pair.track_index
                or other.observation_index == pair.observation_index
            )
        ]
        close = [
            other
            for other in competitors
            if (other.total - pair.total) < ambiguity_margin
        ]
        if close:
            competitor_cost = min(other.total for other in close)
            involved_tracks = {pair.track_index} | {
                other.track_index for other in close
            }
            for track_index in sorted(involved_tracks):
                frozen_indices[track_index] = FREEZE_AMBIGUOUS
                track = tracks[track_index]
                frozen_records.append(
                    FrozenTrack(
                        track_id=track.track_id,
                        current_cell=track.current_cell,
                        reason=FREEZE_AMBIGUOUS,
                        best_cost=pair.total,
                        competitor_cost=competitor_cost,
                    )
                )
            ambiguous_observations[pair.observation_index] = True
            for other in close:
                ambiguous_observations[other.observation_index] = True
        else:
            margin = (
                min(
                    other.total - pair.total for other in competitors
                )
                if competitors
                else None
            )
            assigned_tracks[pair.track_index] = (
                pair.observation_index,
                pair.evaluation,
                margin,
            )
            assigned_observations[pair.observation_index] = (
                pair.track_index
            )

    # Resolve every unmatched, not-yet-frozen track.
    removed_records: List[RemovedTrack] = []
    removed_indices: set = set()
    observation_cells = {
        observation.cell: index
        for index, observation in enumerate(active_observations)
    }
    for track_index, track in correspondable.items():
        if (
            track_index in assigned_tracks
            or track_index in frozen_indices
        ):
            continue
        cell = track.current_cell
        assert cell is not None
        at_cell = observation_cells.get(cell)
        if at_cell is not None:
            if at_cell in ambiguous_observations:
                reason = FREEZE_AMBIGUOUS
            elif at_cell in assigned_observations:
                reason = FREEZE_CELL_CONTESTED
            else:
                evaluation = evaluations.get(
                    (track_index, at_cell)
                )
                if (
                    evaluation is not None
                    and evaluation.blocked_by == "temporal"
                ):
                    reason = FREEZE_TEMPORAL_DISCONTINUITY
                else:
                    # An unexplained appearance at a persistent locus
                    # is a transformation candidate; confirmation is
                    # WP3's matched-control evidence, so abstain.
                    reason = FREEZE_TRANSFORMATION_CANDIDATE
            frozen_indices[track_index] = reason
            frozen_records.append(
                FrozenTrack(
                    track_id=track.track_id,
                    current_cell=cell,
                    reason=reason,
                )
            )
        elif cell in relaxed:
            outcome = (
                OUTCOME_EXPULSION
                if _boundary_cell(cell, grid_columns, grid_rows)
                else OUTCOME_REMOVAL
            )
            removed_indices.add(track_index)
            removed_records.append(
                RemovedTrack(
                    track_id=track.track_id,
                    last_cell=cell,
                    outcome=outcome,
                )
            )
        else:
            # No present-frame evidence either way: freeze, do not
            # expire (expiration requires matched evidence).
            frozen_indices[track_index] = FREEZE_MISSING_EVIDENCE
            frozen_records.append(
                FrozenTrack(
                    track_id=track.track_id,
                    current_cell=cell,
                    reason=FREEZE_MISSING_EVIDENCE,
                )
            )
    for track_index, reason in sorted(frozen_indices.items()):
        if reason == FREEZE_MISSING_CURRENT_CELL:
            track = tracks[track_index]
            frozen_records.append(
                FrozenTrack(
                    track_id=track.track_id,
                    current_cell=None,
                    reason=reason,
                )
            )

    # Categorize every unmatched observation.  An observation at a
    # frozen track's cell inherits that freeze reason so telemetry
    # explains the abstention instead of inventing a candidate.
    frozen_cell_reasons = {
        tracks[index].current_cell: reason
        for index, reason in sorted(frozen_indices.items())
        if tracks[index].current_cell is not None
    }
    unmatched_records: List[UnmatchedObservation] = []
    for index, observation in enumerate(active_observations):
        if index in assigned_observations:
            continue
        if index in ambiguous_observations:
            reason = UNMATCHED_AMBIGUOUS
        elif observation.cell in frozen_cell_reasons:
            reason = frozen_cell_reasons[observation.cell]
        else:
            reason = UNMATCHED_NEW_TRACK_CANDIDATE
        unmatched_records.append(
            UnmatchedObservation(
                observation=observation, reason=reason
            )
        )

    # Build assignments and the endpoint-relative updated track tuple.
    assignment_records: List[TrackAssignment] = []
    updated: List[AnonymousObjectTrack] = []
    for track_index, track in enumerate(tracks):
        if track_index in removed_indices:
            continue
        if track_index in assigned_tracks:
            observation_index, evaluation, margin = assigned_tracks[
                track_index
            ]
            observation = active_observations[observation_index]
            source = track.current_cell
            assert source is not None
            delta = (
                observation.cell[0] - source[0],
                observation.cell[1] - source[1],
            )
            if delta != (0, 0):
                outcome = OUTCOME_DISPLACEMENT
            elif evaluation.appearance_cost > 0:
                outcome = OUTCOME_ANIMATION
            else:
                outcome = OUTCOME_STATIONARY
            assignment_records.append(
                TrackAssignment(
                    track_id=track.track_id,
                    source_cell=source,
                    target_cell=observation.cell,
                    displacement=delta,
                    outcome=outcome,
                    source_appearance_signature=_appearance_key(
                        track.appearance_state_signature,
                        track.appearance_fingerprint,
                    ),
                    target_appearance_signature=_appearance_key(
                        observation.appearance_state_signature,
                        observation.appearance_fingerprint,
                    ),
                    appearance_cost=evaluation.appearance_cost,
                    displacement_cost=evaluation.displacement_cost,
                    temporal_cost=evaluation.temporal_cost,
                    total_cost=evaluation.total_cost,
                    ambiguity_margin=_optional_rounded(margin),
                )
            )
            updated.append(
                _updated_track(track, observation, delta)
            )
        else:
            # Frozen: carried unchanged, explicitly not updated.
            updated.append(track)

    return CorrespondenceResult(
        frame_index=frame_index,
        assignments=tuple(
            sorted(
                assignment_records,
                key=lambda record: (
                    record.track_id or "",
                    record.source_cell,
                    record.target_cell,
                ),
            )
        ),
        frozen_tracks=tuple(
            sorted(
                frozen_records,
                key=lambda record: (
                    record.track_id or "",
                    _cell_sort_key(record.current_cell),
                    record.reason,
                ),
            )
        ),
        removed_tracks=tuple(
            sorted(
                removed_records,
                key=lambda record: (
                    record.track_id or "",
                    record.last_cell,
                ),
            )
        ),
        unmatched_observations=tuple(
            sorted(
                unmatched_records,
                key=lambda record: _observation_sort_key(
                    record.observation
                ),
            )
        ),
        excluded_observations=tuple(
            sorted(
                excluded,
                key=lambda record: _observation_sort_key(
                    record.observation
                ),
            )
        ),
        updated_tracks=tuple(sorted(updated, key=_track_sort_key)),
        abstained=False,
        abstain_reason="",
    )


def correspond_evidence(
    tracks: Sequence[AnonymousObjectTrack],
    evidence: Sequence[CellEvidence],
    *,
    previously_changed: Iterable[Cell] = (),
    **correspondence_options: Any,
) -> Tuple[EndpointRelativeTrackState, CorrespondenceResult]:
    """Endpoint-relative evidence pipeline in one call.

    Derives the endpoint state and the still-changed observations from
    one frame of cell evidence, then corresponds against ``tracks``
    with the state's relaxed cells as positive removal evidence.
    """

    state = endpoint_relative_state(
        evidence, previously_changed=previously_changed
    )
    result = correspond(
        tracks,
        observations_from_evidence(evidence),
        relaxed_cells=state.relaxed_cells,
        frame_index=state.frame_index,
        **correspondence_options,
    )
    return state, result


def track_set_signature(
    tracks: Sequence[AnonymousObjectTrack],
) -> str:
    """Digest a track tuple under the ObjectTrackSet conventions."""

    return ObjectTrackSet(tracks=tuple(tracks)).signature


def correspondence_telemetry(
    result: CorrespondenceResult,
) -> Dict[str, Any]:
    """Serializable telemetry payload for one correspondence round.

    Carries the WP2-required fields: retained/candidate counts,
    per-track source, target, displacement, and appearance transition,
    correspondence cost and ambiguity margin, pruning reasons, and the
    deterministic track-set signature.
    """

    return {
        "anonymous_correspondence_frame_index": result.frame_index,
        "anonymous_correspondence_track_count": len(
            result.updated_tracks
        )
        + len(result.removed_tracks),
        "anonymous_correspondence_retained_track_count": len(
            result.updated_tracks
        ),
        "anonymous_correspondence_assignments": [
            assignment._canonical()
            for assignment in result.assignments
        ],
        "anonymous_correspondence_frozen_tracks": [
            frozen._canonical() for frozen in result.frozen_tracks
        ],
        "anonymous_correspondence_removed_tracks": [
            removed._canonical() for removed in result.removed_tracks
        ],
        "anonymous_correspondence_unmatched_observations": [
            unmatched._canonical()
            for unmatched in result.unmatched_observations
        ],
        "anonymous_correspondence_excluded_observations": [
            excluded._canonical()
            for excluded in result.excluded_observations
        ],
        "anonymous_correspondence_abstained": result.abstained,
        "anonymous_correspondence_abstain_reason": (
            result.abstain_reason or None
        ),
        "anonymous_correspondence_track_set_signature": (
            result.track_set_signature
        ),
        "anonymous_correspondence_signature": result.signature,
    }
