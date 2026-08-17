from __future__ import annotations

import dataclasses
import json
import unittest
from typing import Optional

from lolo_agent.object_correspondence import (
    CellEvidence,
    CorrespondenceResult,
    EXCLUDED_AUTONOMOUS_MOTION,
    EXCLUDED_HUD_REGION,
    FREEZE_AMBIGUOUS,
    FREEZE_MISSING_CURRENT_CELL,
    FREEZE_MISSING_EVIDENCE,
    FREEZE_OBSERVATION_BOUND,
    FREEZE_TEMPORAL_DISCONTINUITY,
    FREEZE_TRACK_BOUND,
    FREEZE_TRANSFORMATION_CANDIDATE,
    MAX_SIMULTANEOUS_TRACKS,
    OUTCOME_ANIMATION,
    OUTCOME_DISPLACEMENT,
    OUTCOME_EXPULSION,
    OUTCOME_REMOVAL,
    OUTCOME_STATIONARY,
    UNMATCHED_AMBIGUOUS,
    UNMATCHED_BOUND_ABSTENTION,
    UNMATCHED_NEW_TRACK_CANDIDATE,
    ObjectObservation,
    anonymous_track_id,
    bootstrap_tracks,
    correspond,
    correspond_evidence,
    correspondence_telemetry,
    endpoint_relative_state,
    observations_from_evidence,
    track_from_observation,
    track_set_signature,
)
from lolo_agent.object_tracks import (
    AnonymousObjectTrack,
    ObjectTrackSet,
)


def _track(
    cell,
    *,
    fingerprint: str = "",
    signature: str = "",
    track_id: Optional[str] = None,
    frame: Optional[int] = None,
    steps: int = 0,
) -> AnonymousObjectTrack:
    return AnonymousObjectTrack(
        appearance_fingerprint=fingerprint,
        current_cell=cell,
        appearance_state_signature=signature,
        track_id=track_id,
        latest_observed_frame=None if frame is None else str(frame),
        persistence_steps=steps,
    )


def _obs(
    cell,
    *,
    fingerprint: str = "",
    signature: str = "",
    frame: int = 0,
) -> ObjectObservation:
    return ObjectObservation(
        cell=cell,
        appearance_state_signature=signature,
        appearance_fingerprint=fingerprint,
        frame_index=frame,
    )


class EndpointRelativeStateTests(unittest.TestCase):
    def test_stale_relaxed_cell_leaves_current_set(self) -> None:
        # The 4.29 lesson: five of six accumulated cells had relaxed
        # to baseline while the committed set still listed them.  A
        # cell observed at baseline must leave the current set and
        # stay only in the changed-at-some-point record.
        state = endpoint_relative_state(
            [
                CellEvidence(
                    cell=(7, 6),
                    appearance_signature="floor",
                    baseline_signature="floor",
                    frame_index=8,
                ),
                CellEvidence(
                    cell=(8, 6),
                    appearance_signature="blob",
                    baseline_signature="floor",
                    frame_index=8,
                ),
            ],
            previously_changed=[(7, 6), (8, 6), (14, 5)],
        )
        self.assertEqual(state.frame_index, 8)
        self.assertEqual(state.current_cells, ((8, 6),))
        self.assertEqual(state.relaxed_cells, ((7, 6),))
        self.assertEqual(state.unobserved_cells, ((14, 5),))
        self.assertEqual(
            state.ever_changed_cells, ((7, 6), (8, 6), (14, 5))
        )

    def test_current_cells_never_include_unobserved_history(
        self,
    ) -> None:
        state = endpoint_relative_state(
            [],
            previously_changed=[(2, 6), (3, 7)],
        )
        self.assertEqual(state.current_cells, ())
        self.assertEqual(state.unobserved_cells, ((2, 6), (3, 7)))
        self.assertEqual(
            state.ever_changed_cells, ((2, 6), (3, 7))
        )

    def test_conflicting_evidence_rejected(self) -> None:
        with self.assertRaises(ValueError):
            endpoint_relative_state(
                [
                    CellEvidence(
                        cell=(4, 4),
                        appearance_signature="a",
                        baseline_signature="floor",
                    ),
                    CellEvidence(
                        cell=(4, 4),
                        appearance_signature="b",
                        baseline_signature="floor",
                    ),
                ]
            )

    def test_mixed_frame_evidence_rejected(self) -> None:
        with self.assertRaises(ValueError):
            endpoint_relative_state(
                [
                    CellEvidence(cell=(4, 4), frame_index=1),
                    CellEvidence(cell=(5, 4), frame_index=2),
                ]
            )

    def test_state_signature_deterministic(self) -> None:
        evidence = [
            CellEvidence(
                cell=(8, 6),
                appearance_signature="blob",
                baseline_signature="floor",
                frame_index=3,
            ),
            CellEvidence(
                cell=(7, 6),
                appearance_signature="floor",
                baseline_signature="floor",
                frame_index=3,
            ),
        ]
        first = endpoint_relative_state(
            evidence, previously_changed=[(7, 6)]
        )
        second = endpoint_relative_state(
            list(reversed(evidence)), previously_changed=[(7, 6)]
        )
        self.assertEqual(first, second)
        self.assertEqual(first.signature, second.signature)

    def test_observations_derive_from_still_changed_only(
        self,
    ) -> None:
        observations = observations_from_evidence(
            [
                CellEvidence(
                    cell=(8, 6),
                    appearance_signature="blob",
                    baseline_signature="floor",
                    appearance_fingerprint="blob",
                    frame_index=3,
                ),
                CellEvidence(
                    cell=(7, 6),
                    appearance_signature="floor",
                    baseline_signature="floor",
                    frame_index=3,
                ),
            ]
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].cell, (8, 6))
        self.assertEqual(observations[0].frame_index, 3)


class CorrespondenceTests(unittest.TestCase):
    def test_two_identical_objects_move_independently_without_swap(
        self,
    ) -> None:
        # Identical-appearance-swap risk: correspondence must resolve
        # repeated identical appearances spatially, never by
        # appearance alone.  The cross pairings here are feasible
        # (manhattan 4 within the displacement bound) so only the
        # cost ordering prevents a swap.
        tracks = [
            _track((2, 2), fingerprint="blob", track_id="track-a"),
            _track((5, 2), fingerprint="blob", track_id="track-b"),
        ]
        observations = [
            _obs((3, 2), fingerprint="blob", frame=1),
            _obs((6, 2), fingerprint="blob", frame=1),
        ]
        result = correspond(tracks, observations)
        self.assertFalse(result.abstained)
        self.assertEqual(result.frozen_tracks, ())
        self.assertEqual(len(result.assignments), 2)
        by_id = {
            assignment.track_id: assignment
            for assignment in result.assignments
        }
        self.assertEqual(by_id["track-a"].target_cell, (3, 2))
        self.assertEqual(by_id["track-b"].target_cell, (6, 2))
        self.assertEqual(
            by_id["track-a"].outcome, OUTCOME_DISPLACEMENT
        )
        self.assertEqual(
            by_id["track-b"].outcome, OUTCOME_DISPLACEMENT
        )
        self.assertEqual(by_id["track-a"].displacement, (1, 0))
        updated = {
            track.track_id: track for track in result.updated_tracks
        }
        self.assertEqual(updated["track-a"].current_cell, (3, 2))
        self.assertEqual(updated["track-a"].previous_cell, (2, 2))
        self.assertEqual(updated["track-b"].current_cell, (6, 2))
        self.assertEqual(updated["track-a"].persistence_steps, 1)
        self.assertEqual(
            updated["track-a"].latest_observed_frame, "1"
        )

    def test_one_object_moving_while_another_animates(self) -> None:
        def appearance_distance(a: str, b: str) -> float:
            if {a, b} == {"blob-frame-1", "blob-frame-2"}:
                return 0.2
            return 0.0 if a == b else 1.0

        tracks = [
            _track((2, 2), fingerprint="mover", track_id="track-a"),
            _track(
                (10, 10),
                signature="blob-frame-1",
                track_id="track-b",
            ),
        ]
        observations = [
            _obs((3, 2), fingerprint="mover", frame=1),
            _obs((10, 10), signature="blob-frame-2", frame=1),
        ]
        result = correspond(
            tracks,
            observations,
            appearance_distance=appearance_distance,
        )
        by_id = {
            assignment.track_id: assignment
            for assignment in result.assignments
        }
        self.assertEqual(
            by_id["track-a"].outcome, OUTCOME_DISPLACEMENT
        )
        self.assertEqual(
            by_id["track-b"].outcome, OUTCOME_ANIMATION
        )
        self.assertEqual(by_id["track-b"].displacement, (0, 0))
        updated = {
            track.track_id: track for track in result.updated_tracks
        }
        self.assertEqual(
            updated["track-b"].appearance_state_signature,
            "blob-frame-2",
        )
        self.assertEqual(
            updated["track-b"].previous_appearance_state_signature,
            "blob-frame-1",
        )

    def test_equidistant_adjacent_observation_freezes_both_tracks(
        self,
    ) -> None:
        # Two identical tracks converging on one adjacent cell are
        # equally plausible: abstain-and-freeze, never guess.
        tracks = [
            _track((5, 5), fingerprint="blob", track_id="track-a"),
            _track((7, 5), fingerprint="blob", track_id="track-b"),
        ]
        observations = [_obs((6, 5), fingerprint="blob", frame=1)]
        result = correspond(tracks, observations)
        self.assertFalse(result.abstained)
        self.assertEqual(result.assignments, ())
        self.assertEqual(len(result.frozen_tracks), 2)
        for frozen in result.frozen_tracks:
            self.assertEqual(frozen.reason, FREEZE_AMBIGUOUS)
            self.assertEqual(frozen.best_cost, 0.25)
            self.assertEqual(frozen.competitor_cost, 0.25)
        self.assertEqual(len(result.unmatched_observations), 1)
        self.assertEqual(
            result.unmatched_observations[0].reason,
            UNMATCHED_AMBIGUOUS,
        )
        # Frozen tracks are carried unchanged.
        self.assertEqual(
            {track.current_cell for track in result.updated_tracks},
            {(5, 5), (7, 5)},
        )

    def test_ambiguity_margin_boundary_is_strict(self) -> None:
        tracks = [
            _track((5, 5), fingerprint="blob", track_id="track-a")
        ]
        observations = [
            _obs((6, 5), fingerprint="blob", frame=1),
            _obs((7, 5), fingerprint="blob", frame=1),
        ]
        # Competitor exactly one margin away: decision stands.
        decided = correspond(
            tracks, observations, ambiguity_margin=0.25
        )
        self.assertEqual(len(decided.assignments), 1)
        self.assertEqual(
            decided.assignments[0].target_cell, (6, 5)
        )
        self.assertEqual(
            decided.assignments[0].ambiguity_margin, 0.25
        )
        self.assertEqual(
            decided.unmatched_observations[0].reason,
            UNMATCHED_NEW_TRACK_CANDIDATE,
        )
        # Competitor strictly inside the margin: freeze.
        frozen = correspond(
            tracks, observations, ambiguity_margin=0.3
        )
        self.assertEqual(frozen.assignments, ())
        self.assertEqual(len(frozen.frozen_tracks), 1)
        self.assertEqual(
            frozen.frozen_tracks[0].reason, FREEZE_AMBIGUOUS
        )
        self.assertEqual(
            {
                unmatched.reason
                for unmatched in frozen.unmatched_observations
            },
            {UNMATCHED_AMBIGUOUS},
        )

    def test_track_bound_exceeded_abstains_and_freezes_all(
        self,
    ) -> None:
        tracks = [
            _track(
                (index, 0),
                fingerprint="blob",
                track_id=f"track-{index}",
            )
            for index in range(MAX_SIMULTANEOUS_TRACKS + 1)
        ]
        result = correspond(tracks, [])
        self.assertTrue(result.abstained)
        self.assertEqual(result.abstain_reason, FREEZE_TRACK_BOUND)
        self.assertEqual(
            len(result.frozen_tracks), MAX_SIMULTANEOUS_TRACKS + 1
        )
        for frozen in result.frozen_tracks:
            self.assertEqual(frozen.reason, FREEZE_TRACK_BOUND)
        self.assertEqual(result.assignments, ())
        self.assertEqual(
            len(result.updated_tracks), MAX_SIMULTANEOUS_TRACKS + 1
        )

    def test_observation_bound_exceeded_abstains(self) -> None:
        tracks = [
            _track((0, 0), fingerprint="blob", track_id="track-a")
        ]
        observations = [
            _obs((index, 9), fingerprint="blob")
            for index in range(MAX_SIMULTANEOUS_TRACKS + 1)
        ]
        result = correspond(tracks, observations)
        self.assertTrue(result.abstained)
        self.assertEqual(
            result.abstain_reason, FREEZE_OBSERVATION_BOUND
        )
        self.assertEqual(
            {
                unmatched.reason
                for unmatched in result.unmatched_observations
            },
            {UNMATCHED_BOUND_ABSTENTION},
        )
        self.assertEqual(
            result.frozen_tracks[0].reason,
            FREEZE_OBSERVATION_BOUND,
        )

    def test_four_tracks_process_without_abstention(self) -> None:
        tracks = [
            _track(
                (3 * index, 0),
                fingerprint="blob",
                track_id=f"track-{index}",
            )
            for index in range(MAX_SIMULTANEOUS_TRACKS)
        ]
        observations = [
            _obs((3 * index, 0), fingerprint="blob", frame=1)
            for index in range(MAX_SIMULTANEOUS_TRACKS)
        ]
        result = correspond(tracks, observations)
        self.assertFalse(result.abstained)
        self.assertEqual(
            len(result.assignments), MAX_SIMULTANEOUS_TRACKS
        )
        self.assertEqual(
            {
                assignment.outcome
                for assignment in result.assignments
            },
            {OUTCOME_STATIONARY},
        )

    def test_relaxed_cell_track_removed(self) -> None:
        # Removal requires positive present-frame evidence that the
        # cell relaxed to baseline, never a merely missing frame.
        tracks = [
            _track((7, 6), fingerprint="blob", track_id="track-a")
        ]
        result = correspond(
            tracks, [], relaxed_cells=[(7, 6)]
        )
        self.assertEqual(len(result.removed_tracks), 1)
        removed = result.removed_tracks[0]
        self.assertEqual(removed.track_id, "track-a")
        self.assertEqual(removed.last_cell, (7, 6))
        self.assertEqual(removed.outcome, OUTCOME_REMOVAL)
        self.assertEqual(result.updated_tracks, ())
        self.assertEqual(result.frozen_tracks, ())

    def test_boundary_relaxed_cell_track_expelled(self) -> None:
        # The 4.29 chain ended with the transformed object expelled
        # east off row 6; a relaxed cell on the grid boundary scores
        # as expulsion when the caller supplies grid bounds.
        tracks = [
            _track((15, 6), fingerprint="blob", track_id="track-a")
        ]
        result = correspond(
            tracks,
            [],
            relaxed_cells=[(15, 6)],
            grid_columns=16,
            grid_rows=15,
        )
        self.assertEqual(
            result.removed_tracks[0].outcome, OUTCOME_EXPULSION
        )

    def test_missing_evidence_freezes_track_instead_of_expiring(
        self,
    ) -> None:
        tracks = [
            _track((5, 5), fingerprint="blob", track_id="track-a")
        ]
        result = correspond(tracks, [])
        self.assertEqual(result.removed_tracks, ())
        self.assertEqual(len(result.frozen_tracks), 1)
        self.assertEqual(
            result.frozen_tracks[0].reason, FREEZE_MISSING_EVIDENCE
        )
        self.assertEqual(result.updated_tracks, tuple(tracks))

    def test_missing_current_cell_track_frozen(self) -> None:
        # Appearance-only matching is the recorded Gate 1
        # falsification mode; a cell-less track cannot correspond.
        tracks = [
            AnonymousObjectTrack(
                appearance_fingerprint="blob", track_id="track-a"
            )
        ]
        observations = [_obs((5, 5), fingerprint="blob")]
        result = correspond(tracks, observations)
        self.assertEqual(result.assignments, ())
        self.assertEqual(
            result.frozen_tracks[0].reason,
            FREEZE_MISSING_CURRENT_CELL,
        )
        self.assertEqual(
            result.unmatched_observations[0].reason,
            UNMATCHED_NEW_TRACK_CANDIDATE,
        )

    def test_transformation_candidate_freezes_track(self) -> None:
        # An unexplained appearance at a persistent locus abstains as
        # a transformation candidate for WP3's matched-control
        # confirmation; WP2 never confirms transformations.
        tracks = [
            _track((4, 4), signature="shape-a", track_id="track-a")
        ]
        observations = [_obs((4, 4), signature="shape-b", frame=1)]
        result = correspond(tracks, observations)
        self.assertEqual(result.assignments, ())
        self.assertEqual(
            result.frozen_tracks[0].reason,
            FREEZE_TRANSFORMATION_CANDIDATE,
        )
        self.assertEqual(
            result.unmatched_observations[0].reason,
            FREEZE_TRANSFORMATION_CANDIDATE,
        )
        self.assertEqual(result.updated_tracks, tuple(tracks))

    def test_temporal_discontinuity_freezes_track(self) -> None:
        tracks = [
            _track(
                (5, 5),
                fingerprint="blob",
                track_id="track-a",
                frame=0,
            )
        ]
        observations = [_obs((5, 5), fingerprint="blob", frame=6)]
        result = correspond(tracks, observations)
        self.assertEqual(result.assignments, ())
        self.assertEqual(
            result.frozen_tracks[0].reason,
            FREEZE_TEMPORAL_DISCONTINUITY,
        )
        self.assertEqual(
            result.unmatched_observations[0].reason,
            FREEZE_TEMPORAL_DISCONTINUITY,
        )

    def test_temporal_gap_costs_accumulate(self) -> None:
        tracks = [
            _track(
                (5, 5),
                fingerprint="blob",
                track_id="track-a",
                frame=0,
            )
        ]
        observations = [_obs((5, 5), fingerprint="blob", frame=3)]
        result = correspond(tracks, observations)
        self.assertEqual(len(result.assignments), 1)
        assignment = result.assignments[0]
        self.assertEqual(assignment.temporal_cost, 0.25)
        self.assertEqual(assignment.total_cost, 0.25)
        self.assertEqual(assignment.outcome, OUTCOME_STATIONARY)

    def test_hud_region_exclusion_hook(self) -> None:
        # The 4.29 decomposition found the HUD shot counter at
        # (14, 5) — outside the room — inside the accumulated track
        # set.  The caller-supplied predicate prunes it before any
        # correspondence credit.
        tracks = [
            _track((13, 5), fingerprint="counter", track_id="track-a")
        ]
        observations = [_obs((14, 5), fingerprint="counter", frame=1)]
        result = correspond(
            tracks,
            observations,
            hud_region=lambda cell: cell == (14, 5),
        )
        self.assertEqual(result.assignments, ())
        self.assertEqual(len(result.excluded_observations), 1)
        excluded = result.excluded_observations[0]
        self.assertEqual(excluded.observation.cell, (14, 5))
        self.assertEqual(excluded.reason, EXCLUDED_HUD_REGION)
        self.assertEqual(
            result.frozen_tracks[0].reason, FREEZE_MISSING_EVIDENCE
        )

    def test_autonomous_motion_exclusion_hook(self) -> None:
        # The 4.29 decomposition found the autonomous patroller leak
        # at (2, 6)/(3, 7); autonomous patrol earns no manipulation
        # credit and no new-track nomination.
        patrol = {(2, 6), (3, 7)}
        observations = [
            _obs((2, 6), fingerprint="walker", frame=1),
            _obs((3, 7), fingerprint="walker", frame=1),
        ]
        result = correspond(
            [],
            observations,
            autonomous_motion=lambda cell: cell in patrol,
        )
        self.assertEqual(
            {
                excluded.reason
                for excluded in result.excluded_observations
            },
            {EXCLUDED_AUTONOMOUS_MOTION},
        )
        self.assertEqual(result.new_track_candidates, ())

    def test_duplicate_observation_cells_rejected(self) -> None:
        with self.assertRaises(ValueError):
            correspond(
                [],
                [
                    _obs((4, 4), fingerprint="a"),
                    _obs((4, 4), fingerprint="b"),
                ],
            )

    def test_observed_and_relaxed_overlap_rejected(self) -> None:
        with self.assertRaises(ValueError):
            correspond(
                [],
                [_obs((3, 3), fingerprint="a")],
                relaxed_cells=[(3, 3)],
            )


class DeterminismTests(unittest.TestCase):
    def _scenario(
        self, *, reverse: bool
    ) -> CorrespondenceResult:
        tracks = [
            _track((2, 2), fingerprint="blob", track_id="track-a"),
            _track((5, 2), fingerprint="blob", track_id="track-b"),
            _track(
                (9, 9), signature="shape-a", track_id="track-c"
            ),
        ]
        observations = [
            _obs((3, 2), fingerprint="blob", frame=1),
            _obs((6, 2), fingerprint="blob", frame=1),
            _obs((14, 5), fingerprint="counter", frame=1),
        ]
        if reverse:
            tracks = list(reversed(tracks))
            observations = list(reversed(observations))
        return correspond(
            tracks,
            observations,
            relaxed_cells=[(9, 9)],
            hud_region=lambda cell: cell == (14, 5),
        )

    def test_result_invariant_under_input_permutation(self) -> None:
        first = self._scenario(reverse=False)
        second = self._scenario(reverse=True)
        self.assertEqual(first, second)
        self.assertEqual(first.signature, second.signature)
        self.assertEqual(
            first.track_set_signature, second.track_set_signature
        )

    def test_track_set_signature_matches_conventions(self) -> None:
        result = self._scenario(reverse=False)
        self.assertEqual(
            result.track_set_signature,
            ObjectTrackSet(tracks=result.updated_tracks).signature,
        )
        self.assertEqual(
            result.track_set_signature,
            track_set_signature(result.updated_tracks),
        )

    def test_result_signature_changes_with_input(self) -> None:
        base = self._scenario(reverse=False)
        moved = correspond(
            [_track((2, 2), fingerprint="blob", track_id="track-a")],
            [_obs((3, 2), fingerprint="blob", frame=1)],
        )
        self.assertNotEqual(base.signature, moved.signature)

    def test_result_dataclasses_frozen(self) -> None:
        result = self._scenario(reverse=False)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.abstained = True  # type: ignore[misc]
        observation = _obs((1, 1), fingerprint="blob")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            observation.cell = (2, 2)  # type: ignore[misc]
        evidence = CellEvidence(cell=(1, 1))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            evidence.cell = (2, 2)  # type: ignore[misc]


class EvidencePipelineTests(unittest.TestCase):
    def test_correspond_evidence_end_to_end(self) -> None:
        tracks = [
            _track(
                (8, 6),
                fingerprint="blob",
                signature="blob",
                track_id="track-obj",
                frame=7,
            ),
            _track(
                (7, 6),
                fingerprint="shape",
                signature="shape",
                track_id="track-gone",
                frame=7,
            ),
        ]
        evidence = [
            CellEvidence(
                cell=(7, 6),
                appearance_signature="floor",
                baseline_signature="floor",
                frame_index=8,
            ),
            CellEvidence(
                cell=(8, 6),
                appearance_signature="blob",
                baseline_signature="floor",
                appearance_fingerprint="blob",
                frame_index=8,
            ),
        ]
        state, result = correspond_evidence(
            tracks,
            evidence,
            previously_changed=[(7, 6), (8, 6)],
        )
        self.assertEqual(state.current_cells, ((8, 6),))
        self.assertEqual(state.relaxed_cells, ((7, 6),))
        self.assertEqual(result.frame_index, 8)
        self.assertEqual(len(result.assignments), 1)
        self.assertEqual(
            result.assignments[0].track_id, "track-obj"
        )
        self.assertEqual(
            result.assignments[0].outcome, OUTCOME_STATIONARY
        )
        self.assertEqual(len(result.removed_tracks), 1)
        self.assertEqual(
            result.removed_tracks[0].track_id, "track-gone"
        )
        self.assertEqual(
            result.removed_tracks[0].outcome, OUTCOME_REMOVAL
        )
        self.assertEqual(len(result.updated_tracks), 1)
        self.assertEqual(
            result.updated_tracks[0].latest_observed_frame, "8"
        )

    def test_bootstrap_tracks_deterministic_and_bounded(
        self,
    ) -> None:
        observations = [
            _obs((6, 2), fingerprint="blob", frame=1),
            _obs((3, 2), fingerprint="blob", frame=1),
        ]
        first = bootstrap_tracks(observations)
        second = bootstrap_tracks(list(reversed(observations)))
        self.assertEqual(first, second)
        self.assertEqual(
            [track.current_cell for track in first],
            [(3, 2), (6, 2)],
        )
        # Identical appearances at distinct cells receive distinct
        # content-derived identities.
        self.assertEqual(len({track.track_id for track in first}), 2)
        self.assertEqual(
            first[0].track_id, anonymous_track_id(observations[1])
        )
        self.assertEqual(first[0].first_observed_frame, "1")
        with self.assertRaises(ValueError):
            bootstrap_tracks(
                [
                    _obs((index, 0), fingerprint="blob")
                    for index in range(MAX_SIMULTANEOUS_TRACKS + 1)
                ]
            )

    def test_track_from_observation_provenance(self) -> None:
        observation = _obs(
            (3, 2), fingerprint="blob", signature="blob", frame=4
        )
        track = track_from_observation(observation)
        self.assertEqual(track.current_cell, (3, 2))
        self.assertEqual(track.first_observed_frame, "4")
        self.assertEqual(track.latest_observed_frame, "4")
        self.assertEqual(
            track.track_id, anonymous_track_id(observation)
        )

    def test_telemetry_payload_serializable(self) -> None:
        tracks = [
            _track((2, 2), fingerprint="blob", track_id="track-a"),
            _track((9, 9), fingerprint="shape", track_id="track-b"),
        ]
        observations = [_obs((3, 2), fingerprint="blob", frame=1)]
        result = correspond(
            tracks, observations, relaxed_cells=[(9, 9)]
        )
        payload = correspondence_telemetry(result)
        json.dumps(payload)
        self.assertEqual(
            payload["anonymous_correspondence_track_count"], 2
        )
        self.assertEqual(
            payload[
                "anonymous_correspondence_retained_track_count"
            ],
            1,
        )
        self.assertEqual(
            len(payload["anonymous_correspondence_assignments"]), 1
        )
        self.assertEqual(
            len(
                payload["anonymous_correspondence_removed_tracks"]
            ),
            1,
        )
        self.assertEqual(
            payload[
                "anonymous_correspondence_track_set_signature"
            ],
            result.track_set_signature,
        )
        self.assertIsNone(
            payload["anonymous_correspondence_abstain_reason"]
        )


if __name__ == "__main__":
    unittest.main()
