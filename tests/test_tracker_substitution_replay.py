from __future__ import annotations

import json
import tempfile
import unittest

try:  # optional ML extra (imported transitively by the modules below)
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise unittest.SkipTest(
        "requires the optional 'ml' extra: pip install -e '.[ml]'"
    ) from exc
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from lolo_agent.environment import Action
from lolo_agent.object_tracks import ObjectTrackSet
from lolo_agent.pixels import Frame
from lolo_agent.run_logging import encode_png
from lolo_agent.tracker_substitution_replay import (
    APPEARANCE_L1_THRESHOLD,
    LEARNED_MASK_PROBABILITY_THRESHOLD,
    PREREGISTERED_DESTINATION_CELL,
    PREREGISTERED_SOURCE_CELL,
    ManipulationIdentity,
    appearance_comparison,
    build_report,
    content_digest,
    find_confirmed_archive_events,
    grid_cell_pixel_block,
    identity_equivalent,
    learned_mask_cells,
    learned_pixel_mask,
    learned_reference_slot,
    manipulation_identity,
    mask_divergence,
    replay_archive_event,
)

# A 16x15 coarse effect bitmask with exactly cell (8, 6) set, matching the
# real world-effect signatures serialized by the v318/v321 evaluation runs
# (same synthetic fixture family as tests/test_object_tracks.py).
_EFFECT_BITMASK = "00" * 104 + "01" + "00" * 135

_WORLD_CONTEXT = (
    "dd9de862b6c3832e23e6bd3de7aa1a593b7f9baadc36c3d158d63233c0f5e8a8"
)

# v318 archive shape: no tracked cells and no interaction identity were
# serialized; everything reconstructs from the bitmask and the path.
_V318_SHAPE_METADATA = {
    "path": ["right"],
    "durations": [16],
    "human_prior_option_world_effect_signature": _EFFECT_BITMASK,
    "human_prior_option_effect_confirmed_action_indices": [0],
    "human_prior_option_effect_frontier": True,
    "human_prior_option_effect_frontier_reason": (
        "anonymous_entity_state_change"
    ),
    "human_prior_option_entity_frontier": True,
    "human_prior_option_entity_state_signature": "fbed5d3a014aa50c",
    "human_prior_world_target_context": _WORLD_CONTEXT,
}

# v321 archive shape: the full interaction identity is serialized.
_V321_SHAPE_METADATA = {
    "path": ["left", "left"],
    "durations": [8, 8],
    "human_prior_option_world_effect_signature": _EFFECT_BITMASK,
    "human_prior_option_world_effect_state_signature": None,
    "human_prior_option_tracked_world_effect_cells": [[8, 6]],
    "human_prior_option_tracked_world_state_signature": "fbed5d3a014aa50c",
    "human_prior_option_world_effect_changed_pixels": 0,
    "human_prior_option_effect_confirmed_action_indices": [0],
    "human_prior_option_effect_frontier": True,
    "human_prior_option_effect_frontier_reason": (
        "anonymous_entity_state_change"
    ),
    "human_prior_option_entity_frontier": True,
    "human_prior_option_entity_state_signature": "fbed5d3a014aa50c",
    "human_prior_option_entity_interaction_signature": "fbed5d3a014aa50c",
    "human_prior_option_entity_interaction_action": "right",
    "human_prior_option_entity_interaction_action_index": 0,
    "human_prior_option_entity_interaction_direction": "right",
    "human_prior_option_entity_interaction_cell": [7, 6],
    "anonymous_entity_appearance_fingerprint": None,
    "anonymous_entity_type_id": None,
    "human_prior_option_entity_effect_target_distance": 1,
    "human_prior_option_entity_persistence_observed": True,
    "human_prior_option_entity_persistence_steps": 1,
    "human_prior_world_target_context": _WORLD_CONTEXT,
}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_ARCHIVES = (
    _REPO_ROOT
    / "experiments/lolo1-entity-v10/evaluations"
    / "entity-v318-room3-known-push-connected-mask-d2",
    _REPO_ROOT
    / "experiments/lolo1-entity-v10/evaluations"
    / "entity-v321-room3-confirmed-identity-d2",
)

_EXPECTED_IDENTITY = ManipulationIdentity(
    source_cell=(7, 6),
    destination_cell=(8, 6),
    direction=Action.RIGHT,
    effect_signature=_EFFECT_BITMASK,
)


@dataclass(frozen=True)
class _StubPrediction:
    columns: int
    rows: int
    probabilities: Tuple[Tuple[float, ...], ...]


def _prediction(
    columns: int, rows: int, cells: dict
) -> _StubPrediction:
    grid = [
        [float(cells.get((column, row), 0.0)) for column in range(columns)]
        for row in range(rows)
    ]
    return _StubPrediction(
        columns=columns,
        rows=rows,
        probabilities=tuple(tuple(row) for row in grid),
    )


@dataclass(frozen=True)
class _StubPredictor:
    prediction: _StubPrediction

    def predict(self, frame: Frame) -> _StubPrediction:
        return self.prediction


def _cell_grid_frame(
    width: int, height: int, values: dict, background: int = 10
) -> Frame:
    pixels = bytearray([background]) * (width * height)
    for (x, y), value in values.items():
        pixels[y * width + x] = value
    return Frame(
        width=width, height=height, channels=1, pixels=bytes(pixels)
    )


class ThresholdRuleTests(unittest.TestCase):
    def test_thresholds_pinned_to_documented_operating_points(self) -> None:
        # The learned-mask threshold is the checkpoint's own validation
        # operating point (validate_controllable_tracker scores predicted
        # cells at probability >= 0.5); the appearance threshold is the
        # established 0.08 appearance-match bound.  Pinning them here
        # keeps the replay from drifting to a tuned rule.
        self.assertEqual(LEARNED_MASK_PROBABILITY_THRESHOLD, 0.5)
        self.assertEqual(APPEARANCE_L1_THRESHOLD, 0.08)
        self.assertEqual(PREREGISTERED_SOURCE_CELL, (7, 6))
        self.assertEqual(PREREGISTERED_DESTINATION_CELL, (8, 6))

    def test_learned_mask_cells_threshold_boundary(self) -> None:
        prediction = _prediction(
            4, 3, {(0, 0): 0.4999, (1, 1): 0.5, (2, 2): 0.75}
        )
        self.assertEqual(
            learned_mask_cells(prediction), ((1, 1), (2, 2))
        )
        self.assertEqual(
            learned_mask_cells(prediction, threshold=0.8), ()
        )

    def test_learned_pixel_mask_uses_feature_grid_partition(self) -> None:
        prediction = _prediction(4, 3, {(1, 1): 1.0})
        width, height = 10, 9
        expected = {
            (x, y)
            for y in range(1 * height // 3, 2 * height // 3)
            for x in range(1 * width // 4, 2 * width // 4)
        }
        self.assertEqual(
            set(learned_pixel_mask(prediction, width, height)), expected
        )
        self.assertEqual(
            set(grid_cell_pixel_block((1, 1), width, height, 4, 3)),
            expected,
        )
        # Every pixel of the frame belongs to exactly one cell block.
        blocks = [
            grid_cell_pixel_block((column, row), width, height, 4, 3)
            for row in range(3)
            for column in range(4)
        ]
        self.assertEqual(
            sum(len(block) for block in blocks), width * height
        )
        self.assertEqual(
            set().union(*blocks),
            {(x, y) for y in range(height) for x in range(width)},
        )

    def test_learned_reference_slot_from_argmax_or_none(self) -> None:
        empty = _prediction(4, 3, {(1, 1): 0.4})
        self.assertIsNone(learned_reference_slot(empty, 10, 9))
        prediction = _prediction(4, 3, {(1, 1): 0.6, (3, 2): 0.9})
        self.assertEqual(
            learned_reference_slot(prediction, 10, 9),
            (3 * 10 // 4, 2 * 9 // 3),
        )


class ComparatorTests(unittest.TestCase):
    def test_manipulation_identity_from_both_archive_shapes(self) -> None:
        for metadata in (_V318_SHAPE_METADATA, _V321_SHAPE_METADATA):
            with self.subTest(shape=len(metadata)):
                tracks = ObjectTrackSet.from_archive_metadata(metadata)
                self.assertEqual(
                    manipulation_identity(tracks), _EXPECTED_IDENTITY
                )

    def test_identity_equivalence_requires_equality_and_completeness(
        self,
    ) -> None:
        self.assertTrue(
            identity_equivalent(_EXPECTED_IDENTITY, _EXPECTED_IDENTITY)
        )
        rotated = ManipulationIdentity(
            source_cell=(7, 6),
            destination_cell=(7, 5),
            direction=Action.UP,
            effect_signature=_EFFECT_BITMASK,
        )
        self.assertFalse(identity_equivalent(_EXPECTED_IDENTITY, rotated))
        shifted = ManipulationIdentity(
            source_cell=(6, 6),
            destination_cell=(7, 6),
            direction=Action.RIGHT,
            effect_signature=_EFFECT_BITMASK,
        )
        self.assertFalse(identity_equivalent(_EXPECTED_IDENTITY, shifted))
        # Two equal but incomplete identities never certify equivalence.
        empty = ManipulationIdentity()
        self.assertFalse(identity_equivalent(empty, empty))
        incomplete = ManipulationIdentity(
            source_cell=(7, 6),
            destination_cell=(8, 6),
            direction=None,
            effect_signature=_EFFECT_BITMASK,
        )
        self.assertFalse(identity_equivalent(incomplete, incomplete))

    def test_appearance_comparison_exact_recorded_precedence(self) -> None:
        result = appearance_comparison(
            "recorded-sig",
            "assisted-sig",
            "recorded-sig",
            (10, 10),
            (0, 0),
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["learned_matches_recorded"])
        self.assertFalse(result["assisted_matches_recorded"])
        self.assertEqual(result["basis"], "exact-recorded-signature")
        self.assertGreater(result["l1_assisted_vs_learned"], 0.08)

    def test_appearance_comparison_l1_fallback_boundary(self) -> None:
        # Normalized L1 = (|100 - 99| + |100 - 101|) / (max 101 * 2
        # features); see UnlabeledEntityMemory.feature_distance.
        near = appearance_comparison(
            "recorded-sig", "assisted", "learned", (100, 100), (99, 101)
        )
        self.assertEqual(near["basis"], "l1-vs-assisted-reference")
        self.assertAlmostEqual(
            near["l1_assisted_vs_learned"], 2.0 / (101.0 * 2.0)
        )
        self.assertLessEqual(
            near["l1_assisted_vs_learned"], APPEARANCE_L1_THRESHOLD
        )
        self.assertTrue(near["passed"])

        # |100 - 92| / (max 100 * 1 feature) sits exactly on the 0.08
        # boundary, which the established convention accepts as "same".
        exact_boundary = appearance_comparison(
            "", "assisted", "learned", (100,), (92,)
        )
        self.assertAlmostEqual(
            exact_boundary["l1_assisted_vs_learned"],
            APPEARANCE_L1_THRESHOLD,
        )
        self.assertTrue(exact_boundary["passed"])

        far = appearance_comparison(
            "recorded-sig", "assisted", "learned", (10, 10), (0, 0)
        )
        self.assertFalse(far["passed"])

    def test_appearance_comparison_without_recorded_signature(self) -> None:
        result = appearance_comparison("", "assisted", "learned", (5,), (5,))
        self.assertEqual(result["basis"], "l1-vs-assisted-reference")
        self.assertIsNone(result["recorded_signature"])
        self.assertFalse(result["learned_matches_recorded"])
        self.assertTrue(result["passed"])

    def test_mask_divergence_counts(self) -> None:
        first = frozenset({(0, 0), (1, 0), (2, 0)})
        second = frozenset({(1, 0), (2, 0), (3, 0)})
        divergence = mask_divergence(first, second)
        self.assertEqual(divergence["learned_pixels"], 3)
        self.assertEqual(divergence["assisted_pixels"], 3)
        self.assertEqual(divergence["intersection_pixels"], 2)
        self.assertEqual(divergence["union_pixels"], 4)
        self.assertAlmostEqual(divergence["iou"], 0.5)
        self.assertEqual(mask_divergence(first, first)["iou"], 1.0)
        self.assertEqual(
            mask_divergence(first, frozenset())["iou"], 0.0
        )
        self.assertIsNone(
            mask_divergence(frozenset(), frozenset())["iou"]
        )


class SyntheticReplayTests(unittest.TestCase):
    def _write_archive(
        self, root: Path, frame: Frame, metadata: dict
    ) -> dict:
        (root / "frames").mkdir(parents=True)
        digest = frame.digest
        (root / "frames" / f"{digest}.png").write_bytes(encode_png(frame))
        event = dict(metadata)
        event.update(
            {
                "event": "human_prior_option_archive_added",
                "seq": 7,
                "frame": digest,
                "human_prior_target_player_slot": [7, 6],
            }
        )
        (root / "events.jsonl").write_text(
            json.dumps(event) + "\n", encoding="utf-8"
        )
        return event

    def test_replay_archive_event_scores_synthetic_fixture(self) -> None:
        frame = _cell_grid_frame(16, 15, {(8, 6): 200})
        predictor = _StubPredictor(_prediction(16, 15, {(2, 2): 1.0}))
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "synthetic-run"
            event = self._write_archive(root, frame, _V318_SHAPE_METADATA)
            found = find_confirmed_archive_events(root / "events.jsonl")
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["frame"], event["frame"])

            result = replay_archive_event(root, found[0], predictor)

        identity_bit = result["bits"]["reconstruction_identity"]
        self.assertTrue(identity_bit["passed"])
        self.assertTrue(identity_bit["equivalent"])
        self.assertTrue(identity_bit["matches_preregistered_cells"])
        self.assertTrue(identity_bit["effect_signature_matches_recorded"])
        self.assertEqual(
            identity_bit["learned_identity"],
            _EXPECTED_IDENTITY.serialized(),
        )

        appearance_bit = result["bits"]["destination_appearance"]
        # The learned mask covers only cell (2, 2), far from the tracked
        # destination, so both reconstructions read identical destination
        # pixels: L1 distance 0 against the assisted reference.
        self.assertEqual(appearance_bit["basis"], "l1-vs-assisted-reference")
        self.assertEqual(appearance_bit["l1_assisted_vs_learned"], 0.0)
        self.assertTrue(appearance_bit["passed"])
        self.assertTrue(appearance_bit["fingerprints_equal"])
        # The synthetic frame cannot reproduce the archived hash; the
        # comparator must say so rather than borrowing the recorded value.
        self.assertEqual(
            appearance_bit["recorded_signature"], "fbed5d3a014aa50c"
        )
        self.assertFalse(appearance_bit["learned_matches_recorded"])
        self.assertFalse(appearance_bit["assisted_matches_recorded"])

        divergence = result["endpoint_mask_divergence"]
        self.assertEqual(divergence["learned_pixels"], 1)
        self.assertEqual(divergence["assisted_pixels"], 0)
        self.assertEqual(divergence["iou"], 0.0)
        self.assertEqual(result["learned"]["mask_cells"], [[2, 2]])
        self.assertEqual(result["learned"]["reference_slot"], [2, 2])

    def test_replay_rejects_frame_digest_mismatch(self) -> None:
        frame = _cell_grid_frame(16, 15, {(8, 6): 200})
        predictor = _StubPredictor(_prediction(16, 15, {}))
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "synthetic-run"
            event = self._write_archive(root, frame, _V318_SHAPE_METADATA)
            tampered = _cell_grid_frame(16, 15, {(8, 6): 30})
            (root / "frames" / f"{event['frame']}.png").write_bytes(
                encode_png(tampered)
            )
            with self.assertRaises(ValueError):
                replay_archive_event(root, event, predictor)

    def test_report_digest_deterministic_and_content_sensitive(self) -> None:
        frame = _cell_grid_frame(16, 15, {(8, 6): 200})
        predictor = _StubPredictor(_prediction(16, 15, {(2, 2): 1.0}))
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "synthetic-run"
            self._write_archive(root, frame, _V321_SHAPE_METADATA)
            events = find_confirmed_archive_events(root / "events.jsonl")
            result = replay_archive_event(root, events[0], predictor)

        provenance = {"checkpoint": "synthetic", "digest": "0" * 64}
        report = build_report([result], {"synthetic-run": []}, provenance)
        rebuilt = build_report([result], {"synthetic-run": []}, provenance)
        self.assertEqual(
            report["content_digest"], rebuilt["content_digest"]
        )
        self.assertEqual(
            report["content_digest"], content_digest(report)
        )
        self.assertIn(report["result"]["promotion_gate"], ("PASS", "FAIL"))
        self.assertEqual(
            report["result"]["bit3_divergence"], "reported-not-gated"
        )
        changed = build_report(
            [result],
            {"synthetic-run": []},
            {"checkpoint": "synthetic", "digest": "1" * 64},
        )
        self.assertNotEqual(
            report["content_digest"], changed["content_digest"]
        )


@unittest.skipUnless(
    all((path / "events.jsonl").exists() for path in _REAL_ARCHIVES),
    "real v318/v321 evaluation archives not present",
)
class RealArchiveMetadataTests(unittest.TestCase):
    def test_confirmed_events_carry_preregistered_identity(self) -> None:
        for run_dir in _REAL_ARCHIVES:
            with self.subTest(run=run_dir.name):
                events = find_confirmed_archive_events(
                    run_dir / "events.jsonl"
                )
                self.assertEqual(len(events), 1)
                identity = manipulation_identity(
                    ObjectTrackSet.from_archive_metadata(events[0])
                )
                self.assertEqual(
                    identity.source_cell, PREREGISTERED_SOURCE_CELL
                )
                self.assertEqual(
                    identity.destination_cell,
                    PREREGISTERED_DESTINATION_CELL,
                )
                self.assertIs(identity.direction, Action.RIGHT)
                self.assertTrue(identity.complete)


if __name__ == "__main__":
    unittest.main()
