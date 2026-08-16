from __future__ import annotations

import dataclasses
import hashlib
import json
import unittest
from dataclasses import dataclass
from typing import Optional, Tuple

from lolo_agent.entity_behavior import AnonymousEntityBehaviorModel
from lolo_agent.environment import Action
from lolo_agent.neural_planner import NeuralPlan, _ArchivedBranch
from lolo_agent.object_tracks import (
    AnonymousObjectTrack,
    AnonymousObjectTransition,
    HumanPriorRootObjectState,
    ObjectTrackSet,
    apply_verified_transition,
    archived_track_fields,
    causal_spatial_cells,
    legacy_interaction_from_effect_bitmask,
    masked_cell_fingerprint,
    match,
    observe_frame,
    player_masked_world_effect_signature,
    world_effect_cells_state_signature,
)
from lolo_agent.pixels import Frame
from lolo_agent.unlabeled_entities import UnlabeledEntityMemory


# A 16x15 coarse effect bitmask with exactly cell (8, 6) set, matching the
# real world-effect signatures serialized by the v318/v321 evaluation runs.
_EFFECT_BITMASK = "00" * 104 + "01" + "00" * 135

_WORLD_CONTEXT = (
    "dd9de862b6c3832e23e6bd3de7aa1a593b7f9baadc36c3d158d63233c0f5e8a8"
)

# Hand-copied from a real v318 ``human_prior_option_archive_added`` payload
# (entity/world-effect subset).  v318 serialized no tracked cells and no
# interaction identity: everything must be reconstructed from the effect
# bitmask and the controlled path.
_V318_OPTION_ARCHIVE_METADATA_JSON = """
{
 "path": ["right"],
 "durations": [16],
 "human_prior_option_world_effect_signature": "%s",
 "human_prior_option_effect_confirmed_action_indices": [0],
 "human_prior_option_effect_frontier": true,
 "human_prior_option_effect_frontier_reason": "anonymous_entity_state_change",
 "human_prior_option_entity_frontier": true,
 "human_prior_option_entity_state_signature": "fbed5d3a014aa50c",
 "human_prior_world_target_context": "%s"
}
""" % (_EFFECT_BITMASK, _WORLD_CONTEXT)

# Hand-copied from a real v321 ``human_prior_option_archive_added`` payload
# (entity/world-effect subset).  v321 serializes the full interaction
# identity; the ``anonymous_entity_*`` keys are present but nullable.
_V321_OPTION_ARCHIVE_METADATA_JSON = """
{
 "path": ["left", "left"],
 "durations": [8, 8],
 "human_prior_option_world_effect_signature": "%s",
 "human_prior_option_world_effect_state_signature": null,
 "human_prior_option_tracked_world_effect_cells": [[8, 6]],
 "human_prior_option_tracked_world_state_signature": "fbed5d3a014aa50c",
 "human_prior_option_world_effect_changed_pixels": 0,
 "human_prior_option_effect_confirmed_action_indices": [0],
 "human_prior_option_effect_frontier": true,
 "human_prior_option_effect_frontier_reason": "anonymous_entity_state_change",
 "human_prior_option_entity_frontier": true,
 "human_prior_option_entity_state_signature": "fbed5d3a014aa50c",
 "human_prior_option_entity_interaction_signature": "fbed5d3a014aa50c",
 "human_prior_option_entity_interaction_action": "right",
 "human_prior_option_entity_interaction_action_index": 0,
 "human_prior_option_entity_interaction_direction": "right",
 "human_prior_option_entity_interaction_cell": [7, 6],
 "anonymous_entity_appearance_fingerprint": null,
 "anonymous_entity_type_id": null,
 "anonymous_entity_context_signature": null,
 "anonymous_entity_phase_signature": null,
 "anonymous_entity_neighborhood_signature": null,
 "human_prior_option_entity_effect_target_distance": 1,
 "human_prior_option_entity_persistence_observed": true,
 "human_prior_option_entity_persistence_steps": 1,
 "human_prior_world_target_context": "%s"
}
""" % (_EFFECT_BITMASK, _WORLD_CONTEXT)


def _v318_metadata() -> dict:
    return json.loads(_V318_OPTION_ARCHIVE_METADATA_JSON)


def _v321_metadata() -> dict:
    return json.loads(_V321_OPTION_ARCHIVE_METADATA_JSON)


def _cell_grid_frame(
    width: int, height: int, values: dict, background: int = 10
) -> Frame:
    pixels = bytearray([background]) * (width * height)
    for (x, y), value in values.items():
        pixels[y * width + x] = value
    return Frame(
        width=width, height=height, channels=1, pixels=bytes(pixels)
    )


def _cell_memory() -> UnlabeledEntityMemory:
    return UnlabeledEntityMemory(
        columns=16,
        rows=15,
        pooled_columns=1,
        pooled_rows=1,
        quantization=1,
    )


@dataclass(frozen=True)
class _StubGoalAnalysis:
    source_player_slot: Optional[Tuple[int, int]] = None
    target_player_slot: Optional[Tuple[int, int]] = None
    collected: Tuple[Tuple[int, int], ...] = ()
    chest_completed: bool = False
    chest_obtained: bool = False
    source_chest_slot: Optional[Tuple[int, int]] = None
    target_chest_slot: Optional[Tuple[int, int]] = None


class ObjectTrackSetArchiveMetadataTests(unittest.TestCase):
    def test_v318_fixture_reconstructs_legacy_track(self) -> None:
        memory = _cell_memory()
        model = AnonymousEntityBehaviorModel()
        frame = _cell_grid_frame(16, 15, {(8, 6): 200})
        resolved_cells = []

        def fingerprint_resolver(
            cell: Tuple[int, int]
        ) -> Tuple[str, Optional[int]]:
            resolved_cells.append(cell)
            return masked_cell_fingerprint(
                frame, cell, None, memory, model, None
            )

        tracks = ObjectTrackSet.from_archive_metadata(
            _v318_metadata(),
            fingerprint_resolver=fingerprint_resolver,
        )
        state = tracks.to_root_object_state()

        self.assertEqual(state.tracked_world_effect_cells, ((8, 6),))
        self.assertEqual(state.entity_interaction_action, Action.RIGHT)
        self.assertEqual(state.entity_interaction_direction, Action.RIGHT)
        self.assertEqual(state.entity_interaction_cell, (7, 6))
        self.assertEqual(state.entity_interaction_action_index, 0)
        self.assertEqual(state.entity_effect_target_distance, 1)
        self.assertTrue(state.entity_effect_persisted_in_search)
        self.assertEqual(state.entity_effect_persistence_steps, 1)
        self.assertEqual(
            state.entity_interaction_signature, "fbed5d3a014aa50c"
        )
        self.assertEqual(
            state.confirmed_entity_state_signature, "fbed5d3a014aa50c"
        )
        self.assertEqual(
            state.confirmed_world_effect_signature, _EFFECT_BITMASK
        )
        self.assertEqual(state.confirmed_world_context, _WORLD_CONTEXT)
        self.assertEqual(state.confirmed_action_indices, (0,))
        self.assertEqual(
            state.confirmed_effect_frontier_reason,
            "anonymous_entity_state_change",
        )
        self.assertEqual(resolved_cells, [(8, 6)])
        self.assertTrue(state.entity_interaction_appearance_fingerprint)
        self.assertEqual(len(tracks.tracks), 1)
        self.assertEqual(len(tracks.transitions), 1)
        self.assertEqual(tracks.tracks[0].current_cell, (8, 6))
        self.assertEqual(tracks.tracks[0].displacement, (1, 0))
        self.assertEqual(tracks.transitions[0].source_cell, (7, 6))

    def test_v321_fixture_round_trips_all_fields(self) -> None:
        tracks = ObjectTrackSet.from_archive_metadata(_v321_metadata())
        state = tracks.to_root_object_state()

        expected = HumanPriorRootObjectState(
            world_effect_signature=_EFFECT_BITMASK,
            world_effect_state_signature="fbed5d3a014aa50c",
            tracked_world_effect_cells=((8, 6),),
            tracked_world_state_signature="fbed5d3a014aa50c",
            world_effect_changed_pixels=0,
            confirmed_world_effect_signature=_EFFECT_BITMASK,
            confirmed_world_context=_WORLD_CONTEXT,
            confirmed_action_indices=(0,),
            confirmed_effect_frontier_reason=(
                "anonymous_entity_state_change"
            ),
            confirmed_entity_state_signature="fbed5d3a014aa50c",
            entity_interaction_signature="fbed5d3a014aa50c",
            entity_interaction_action=Action.RIGHT,
            entity_interaction_action_index=0,
            entity_interaction_direction=Action.RIGHT,
            entity_interaction_cell=(7, 6),
            entity_interaction_appearance_fingerprint="",
            entity_interaction_type_id=None,
            entity_interaction_context_signature="",
            entity_interaction_phase_signature="",
            entity_interaction_neighborhood_signature="",
            entity_effect_target_distance=1,
            entity_effect_persisted_in_search=True,
            entity_effect_persistence_steps=1,
        )
        self.assertEqual(state, expected)
        self.assertEqual(
            ObjectTrackSet.from_root_object_state(state), tracks
        )

    def test_telemetry_golden_digest_v318(self) -> None:
        tracks = ObjectTrackSet.from_archive_metadata(_v318_metadata())
        payload = json.dumps(tracks.to_telemetry(), sort_keys=True)
        self.assertEqual(
            hashlib.sha256(payload.encode("ascii")).hexdigest(),
            "583f086a22ea6b2a912415777dcdbbfbff91b959d20401aad412b154be75e9f0",
        )
        self.assertEqual(
            tracks.signature,
            "2650c86deb205125b8c9de490403b60e16a5242ec15e8d19cd0db7e45645c858",
        )

    def test_telemetry_golden_digest_v321(self) -> None:
        tracks = ObjectTrackSet.from_archive_metadata(_v321_metadata())
        payload = json.dumps(tracks.to_telemetry(), sort_keys=True)
        self.assertEqual(
            hashlib.sha256(payload.encode("ascii")).hexdigest(),
            "0e21787dc47d70be7b36fe574b3d34fa7f3867ddc8d2534872ecb4ed5436f77a",
        )
        self.assertEqual(
            tracks.signature,
            "d1d7944bb8e7b72907e5a5d7ae3bba62a208ab56a3953542103a8dd0824bdfe8",
        )

    def test_archive_round_trip_to_telemetry_from_archive_metadata(
        self,
    ) -> None:
        for metadata in (_v318_metadata(), _v321_metadata()):
            tracks = ObjectTrackSet.from_archive_metadata(metadata)
            reparsed = ObjectTrackSet.from_archive_metadata(
                tracks.to_telemetry()
            )
            self.assertEqual(reparsed, tracks)
            self.assertEqual(reparsed.signature, tracks.signature)

    def test_persistence_steps_default_when_persisted_flag_set(self) -> None:
        explicit = ObjectTrackSet.from_archive_metadata(
            {
                "human_prior_option_entity_persistence_observed": True,
            }
        ).to_root_object_state()
        self.assertTrue(explicit.entity_effect_persisted_in_search)
        self.assertEqual(explicit.entity_effect_persistence_steps, 1)

        implicit = ObjectTrackSet.from_archive_metadata(
            {
                "human_prior_option_effect_frontier": True,
                "human_prior_option_entity_state_signature": "entity-a",
            }
        ).to_root_object_state()
        self.assertTrue(implicit.entity_effect_persisted_in_search)
        self.assertEqual(implicit.entity_effect_persistence_steps, 1)

        preserved = ObjectTrackSet.from_archive_metadata(
            {
                "human_prior_option_entity_persistence_observed": True,
                "human_prior_option_entity_persistence_steps": 3,
            }
        ).to_root_object_state()
        self.assertEqual(preserved.entity_effect_persistence_steps, 3)

        absent = ObjectTrackSet.from_archive_metadata(
            {}
        ).to_root_object_state()
        self.assertFalse(absent.entity_effect_persisted_in_search)
        self.assertEqual(absent.entity_effect_persistence_steps, 0)

    def test_action_string_and_cell_list_coercion(self) -> None:
        stringly = ObjectTrackSet.from_archive_metadata(
            {
                "human_prior_option_entity_interaction_signature": "sig-a",
                "human_prior_option_entity_interaction_action": "right",
                "human_prior_option_entity_interaction_direction": "right",
                "human_prior_option_entity_interaction_cell": [7, 6],
            }
        )
        typed = ObjectTrackSet.from_archive_metadata(
            {
                "human_prior_option_entity_interaction_signature": "sig-a",
                "human_prior_option_entity_interaction_action": Action.RIGHT,
                "human_prior_option_entity_interaction_direction": (
                    Action.RIGHT
                ),
                "human_prior_option_entity_interaction_cell": (7, 6),
            }
        )
        self.assertEqual(stringly, typed)
        state = stringly.to_root_object_state()
        self.assertIs(state.entity_interaction_action, Action.RIGHT)
        self.assertEqual(state.entity_interaction_cell, (7, 6))

        fields = archived_track_fields(
            {
                "human_prior_option_entity_interaction_action": "up",
                "human_prior_option_entity_interaction_cell": [3, 4],
                "human_prior_option_tracked_world_effect_cells": [
                    [5, 2],
                    [1, 2],
                ],
            }
        )
        self.assertIs(
            fields["entity_interaction_action"], Action.UP
        )
        self.assertEqual(fields["entity_interaction_cell"], (3, 4))
        self.assertEqual(
            fields["tracked_world_effect_cells"], ((1, 2), (5, 2))
        )

    def test_archived_track_fields_parse_strictly_without_reconstruction(
        self,
    ) -> None:
        fields = archived_track_fields(_v318_metadata())
        # The strict archive parser restores only what was serialized: no
        # bitmask-derived cells, no path-derived direction, no implicit
        # persistence for legacy shapes.
        self.assertEqual(fields["tracked_world_effect_cells"], ())
        self.assertIsNone(fields["entity_interaction_action"])
        self.assertIsNone(fields["entity_interaction_direction"])
        self.assertIsNone(fields["entity_interaction_cell"])
        self.assertEqual(fields["entity_interaction_signature"], "")
        self.assertFalse(fields["entity_effect_persisted_in_search"])
        self.assertEqual(fields["entity_effect_persistence_steps"], 0)
        self.assertEqual(fields["confirmed_action_indices"], (0,))

        rich = archived_track_fields(_v321_metadata())
        self.assertEqual(rich["tracked_world_effect_cells"], ((8, 6),))
        self.assertIs(rich["entity_interaction_action"], Action.RIGHT)
        self.assertEqual(rich["entity_interaction_cell"], (7, 6))
        self.assertEqual(rich["entity_effect_target_distance"], 1)
        self.assertTrue(rich["entity_effect_persisted_in_search"])
        self.assertEqual(rich["entity_effect_persistence_steps"], 1)


class ObjectTrackSetStructureTests(unittest.TestCase):
    def test_signature_deterministic_and_order_independent(self) -> None:
        track_a = AnonymousObjectTrack(
            appearance_fingerprint="fp-a", source_cell=(1, 2)
        )
        track_b = AnonymousObjectTrack(
            appearance_fingerprint="fp-b", source_cell=(3, 4)
        )
        transition_a = AnonymousObjectTransition(
            action=Action.RIGHT, interaction_signature="sig-a"
        )
        transition_b = AnonymousObjectTransition(
            action=Action.UP, interaction_signature="sig-b"
        )
        first = ObjectTrackSet(
            tracks=(track_a, track_b),
            transitions=(transition_a, transition_b),
            world_effect_signature="effect",
        )
        reordered = ObjectTrackSet(
            tracks=(track_b, track_a),
            transitions=(transition_b, transition_a),
            world_effect_signature="effect",
        )
        rebuilt = ObjectTrackSet(
            tracks=(
                AnonymousObjectTrack(
                    appearance_fingerprint="fp-a", source_cell=(1, 2)
                ),
                track_b,
            ),
            transitions=(transition_a, transition_b),
            world_effect_signature="effect",
        )
        self.assertEqual(first.signature, reordered.signature)
        self.assertEqual(first.signature, rebuilt.signature)
        changed = dataclasses.replace(
            first, world_effect_changed_pixels=5
        )
        self.assertNotEqual(first.signature, changed.signature)

    def test_empty_track_set_signature_stable(self) -> None:
        self.assertEqual(ObjectTrackSet.empty(), ObjectTrackSet())
        self.assertEqual(
            ObjectTrackSet.empty().signature,
            "2a02232debf18056252ead1278bccdd69df811a6e7186c772b5d21d5eef25fff",
        )
        self.assertEqual(
            ObjectTrackSet.empty().to_root_object_state(),
            HumanPriorRootObjectState(),
        )

    def test_root_object_state_round_trip_is_lossless(self) -> None:
        states = (
            HumanPriorRootObjectState(),
            HumanPriorRootObjectState(entity_interaction_cell=(7, 6)),
            HumanPriorRootObjectState(
                entity_effect_persisted_in_search=True
            ),
            HumanPriorRootObjectState(
                tracked_world_effect_cells=((8, 6),),
                tracked_world_state_signature="layout-a",
            ),
            HumanPriorRootObjectState(
                world_effect_signature=_EFFECT_BITMASK,
                world_effect_state_signature="state-a",
                tracked_world_effect_cells=((8, 6), (9, 6)),
                tracked_world_state_signature="layout-a",
                world_effect_changed_pixels=7,
                confirmed_world_effect_signature=_EFFECT_BITMASK,
                confirmed_world_context="world-after",
                confirmed_action_indices=(0, 2),
                confirmed_effect_frontier_reason="delayed_causal_effect",
                confirmed_entity_state_signature="entity-state",
                entity_interaction_signature="entity-interaction",
                entity_interaction_action=Action.A,
                entity_interaction_action_index=2,
                entity_interaction_direction=Action.LEFT,
                entity_interaction_cell=(4, 4),
                entity_interaction_appearance_fingerprint="fp",
                entity_interaction_type_id=3,
                entity_interaction_context_signature="ctx",
                entity_interaction_phase_signature="phase",
                entity_interaction_neighborhood_signature="nbhd",
                entity_effect_target_distance=2,
                entity_effect_persisted_in_search=True,
                entity_effect_persistence_steps=4,
            ),
        )
        for state in states:
            with self.subTest(state=state):
                self.assertEqual(
                    ObjectTrackSet.from_root_object_state(
                        state
                    ).to_root_object_state(),
                    state,
                )

    def test_confirmed_identity_distinct_from_transient_candidate(
        self,
    ) -> None:
        confirmed = ObjectTrackSet.from_root_object_state(
            HumanPriorRootObjectState(
                confirmed_world_effect_signature="effect-1",
                confirmed_world_context="world-1",
                confirmed_entity_state_signature="confirmed-state",
                entity_interaction_signature="confirmed-interaction",
                entity_interaction_action=Action.RIGHT,
                entity_interaction_action_index=0,
                entity_interaction_direction=Action.RIGHT,
                entity_interaction_cell=(7, 6),
                entity_effect_target_distance=1,
            )
        )
        candidate = AnonymousObjectTransition(
            action=Action.UP,
            action_index=1,
            direction=Action.UP,
            source_cell=(6, 8),
            interaction_signature="later-interaction",
        )

        promoted = apply_verified_transition(
            confirmed,
            candidate,
            confirmed_world_effect_signature="effect-2",
            confirmed_world_context="world-2",
            confirmed_action_indices=(1,),
            confirmed_effect_frontier_reason=(
                "anonymous_entity_state_change"
            ),
            entity_state_signature="later-state",
        )

        # The unverified candidate never contaminates the stored identity.
        self.assertEqual(
            confirmed.transitions[0].interaction_signature,
            "confirmed-interaction",
        )
        self.assertEqual(
            confirmed.to_root_object_state().entity_interaction_action,
            Action.RIGHT,
        )
        # Promotion swaps the confirmed identity to the verified candidate.
        self.assertEqual(promoted.transitions, (candidate,))
        self.assertEqual(
            promoted.tracks[0].appearance_state_signature, "later-state"
        )
        self.assertEqual(
            promoted.confirmed_world_effect_signature, "effect-2"
        )
        self.assertEqual(promoted.confirmed_world_context, "world-2")
        self.assertEqual(promoted.confirmed_action_indices, (1,))
        promoted_state = promoted.to_root_object_state()
        self.assertEqual(
            promoted_state.entity_interaction_signature,
            "later-interaction",
        )
        self.assertEqual(promoted_state.entity_interaction_cell, (6, 8))

    def test_from_archived_branch_copies_track_state_fields(self) -> None:
        frame = Frame(width=1, height=1, channels=1, pixels=b"\x00")
        branch = _ArchivedBranch(
            state=object(),
            frame=frame,
            plan=NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
            score=1.0,
            scene="scene",
            created=0,
            goal_world_effect_signature="confirmed-effect",
            goal_target_world_context="world-after",
            human_prior_option_world_effect_signature="effect",
            human_prior_option_entity_state_signature="entity-state",
            human_prior_option_effect_frontier_reason=(
                "anonymous_entity_state_change"
            ),
            world_effect_state_signature="state-a",
            tracked_world_effect_cells=((8, 6),),
            tracked_world_state_signature="layout-a",
            world_effect_changed_pixels=3,
            confirmed_action_indices=(0,),
            entity_interaction_signature="entity-interaction",
            entity_interaction_action=Action.RIGHT,
            entity_interaction_action_index=0,
            entity_interaction_direction=Action.RIGHT,
            entity_interaction_cell=(7, 6),
            entity_interaction_appearance_fingerprint="fp",
            entity_interaction_type_id=3,
            entity_interaction_context_signature="ctx",
            entity_interaction_phase_signature="phase",
            entity_interaction_neighborhood_signature="nbhd",
            entity_effect_target_distance=1,
            entity_effect_persisted_in_search=True,
            entity_effect_persistence_steps=2,
        )

        state = ObjectTrackSet.from_archived_branch(
            branch
        ).to_root_object_state()

        self.assertEqual(
            state,
            HumanPriorRootObjectState(
                world_effect_signature="effect",
                world_effect_state_signature="state-a",
                tracked_world_effect_cells=((8, 6),),
                tracked_world_state_signature="layout-a",
                world_effect_changed_pixels=3,
                confirmed_world_effect_signature="confirmed-effect",
                confirmed_world_context="world-after",
                confirmed_action_indices=(0,),
                confirmed_effect_frontier_reason=(
                    "anonymous_entity_state_change"
                ),
                confirmed_entity_state_signature="entity-state",
                entity_interaction_signature="entity-interaction",
                entity_interaction_action=Action.RIGHT,
                entity_interaction_action_index=0,
                entity_interaction_direction=Action.RIGHT,
                entity_interaction_cell=(7, 6),
                entity_interaction_appearance_fingerprint="fp",
                entity_interaction_type_id=3,
                entity_interaction_context_signature="ctx",
                entity_interaction_phase_signature="phase",
                entity_interaction_neighborhood_signature="nbhd",
                entity_effect_target_distance=1,
                entity_effect_persisted_in_search=True,
                entity_effect_persistence_steps=2,
            ),
        )


class PureHelperTests(unittest.TestCase):
    def test_legacy_bitmask_derives_interaction_cell_and_direction(
        self,
    ) -> None:
        transition = legacy_interaction_from_effect_bitmask(
            _EFFECT_BITMASK, ["right"], 16, 15
        )
        self.assertIsNotNone(transition)
        self.assertIs(transition.action, Action.RIGHT)
        self.assertIs(transition.direction, Action.RIGHT)
        self.assertEqual(transition.action_index, 0)
        self.assertEqual(transition.source_cell, (7, 6))
        self.assertEqual(transition.effect_target_distance, 1)

        two_cells = "00" * 104 + "0101" + "00" * 134
        self.assertIsNone(
            legacy_interaction_from_effect_bitmask(
                two_cells, ["right"], 16, 15
            )
        )
        self.assertIsNone(
            legacy_interaction_from_effect_bitmask(
                _EFFECT_BITMASK, ["a"], 16, 15
            )
        )
        self.assertIsNone(
            legacy_interaction_from_effect_bitmask(
                _EFFECT_BITMASK, [], 16, 15
            )
        )
        self.assertIsNone(
            legacy_interaction_from_effect_bitmask(
                _EFFECT_BITMASK, ["right"], 8, 8
            )
        )
        self.assertIsNone(
            legacy_interaction_from_effect_bitmask(
                None, ["right"], 16, 15
            )
        )
        self.assertEqual(
            causal_spatial_cells(_EFFECT_BITMASK, 16), {(8, 6)}
        )

    def test_missing_fingerprint_recovered_via_masked_resolver(
        self,
    ) -> None:
        memory = _cell_memory()
        model = AnonymousEntityBehaviorModel()
        frame = _cell_grid_frame(16, 15, {(8, 6): 200})

        fingerprint, type_id = masked_cell_fingerprint(
            frame, (8, 6), None, memory, model, None
        )
        self.assertEqual(
            fingerprint, model.appearance_fingerprint((200,))
        )
        self.assertIsNone(type_id)

        model.observe(
            (200,),
            Action.NOOP,
            1,
            "stationary",
            autonomous=True,
        )
        _fingerprint, known_type = masked_cell_fingerprint(
            frame, (8, 6), None, memory, model, None
        )
        self.assertIsNotNone(known_type)

        self.assertEqual(
            masked_cell_fingerprint(frame, (8, 6), None, None, model),
            ("", None),
        )
        self.assertEqual(
            masked_cell_fingerprint(frame, (8, 6), None, memory, None),
            ("", None),
        )

        recovered = ObjectTrackSet.from_archive_metadata(
            _v318_metadata(),
            fingerprint_resolver=lambda cell: masked_cell_fingerprint(
                frame, cell, None, memory, model, None
            ),
        ).to_root_object_state()
        self.assertEqual(
            recovered.entity_interaction_appearance_fingerprint,
            fingerprint,
        )
        without_resolver = ObjectTrackSet.from_archive_metadata(
            _v318_metadata()
        ).to_root_object_state()
        self.assertEqual(
            without_resolver.entity_interaction_appearance_fingerprint, ""
        )

    def test_player_mask_excludes_player_pixels_from_fingerprint(
        self,
    ) -> None:
        memory = _cell_memory()
        model = AnonymousEntityBehaviorModel()
        frame = _cell_grid_frame(16, 15, {(8, 6): 200})
        masked_pixels = []

        def player_pixel_mask(mask_frame, player_slot):
            masked_pixels.append(player_slot)
            return {(8, 6)}

        unmasked, _unused = masked_cell_fingerprint(
            frame, (8, 6), None, memory, model, player_pixel_mask
        )
        masked, _unused = masked_cell_fingerprint(
            frame, (8, 6), (8, 6), memory, model, player_pixel_mask
        )
        self.assertEqual(masked_pixels, [(8, 6)])
        self.assertNotEqual(masked, unmasked)
        # A fully masked pool is encoded as zeroes, keeping the anonymous
        # cell identity independent of an overlapping controlled sprite.
        self.assertEqual(masked, model.appearance_fingerprint((0,)))

        with_player = world_effect_cells_state_signature(
            frame, ((8, 6),), (8, 6), memory, player_pixel_mask
        )
        without_player = world_effect_cells_state_signature(
            frame, ((8, 6),), None, memory, player_pixel_mask
        )
        self.assertNotEqual(with_player, without_player)

    def test_world_effect_signature_masks_player_cells(self) -> None:
        frame = _cell_grid_frame(16, 15, {})
        two_cells = "00" * 104 + "0101" + "00" * 134
        analysis = _StubGoalAnalysis(target_player_slot=(9, 6))

        masked = player_masked_world_effect_signature(
            two_cells,
            analysis,
            frame,
            Action.A,
            columns=16,
            rows=15,
        )
        self.assertEqual(masked, _EFFECT_BITMASK)
        # Directional presses without the exact-search audits never mint a
        # world state from cells beside the moving sprite.
        self.assertEqual(
            player_masked_world_effect_signature(
                two_cells,
                analysis,
                frame,
                Action.RIGHT,
                columns=16,
                rows=15,
            ),
            "",
        )
        self.assertEqual(
            player_masked_world_effect_signature(
                two_cells,
                analysis,
                frame,
                Action.RIGHT,
                True,
                columns=16,
                rows=15,
            ),
            _EFFECT_BITMASK,
        )

    def test_observe_frame_and_match_use_player_masked_state(self) -> None:
        memory = _cell_memory()
        frame = _cell_grid_frame(16, 15, {(8, 6): 200, (9, 6): 90})
        cells = ((8, 6), (9, 6))
        stored = world_effect_cells_state_signature(
            frame, cells, None, memory, None
        )
        tracks = ObjectTrackSet(
            tracked_world_effect_cells=cells,
            tracked_world_state_signature=stored,
        )

        observed = observe_frame(tracks, frame, memory=memory)
        self.assertEqual(observed, stored)
        self.assertTrue(match(tracks, observed))

        changed_frame = _cell_grid_frame(
            16, 15, {(8, 6): 30, (9, 6): 90}
        )
        changed = observe_frame(tracks, changed_frame, memory=memory)
        self.assertFalse(match(tracks, changed))
        self.assertFalse(match(ObjectTrackSet(), ""))


if __name__ == "__main__":
    unittest.main()
