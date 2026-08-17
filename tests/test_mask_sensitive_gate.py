from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from lolo_agent import tracker_substitution_replay
from lolo_agent.mask_sensitive_gate import (
    AGREEMENT_RATE_THRESHOLD,
    APPEARANCE_L1_THRESHOLD,
    LEARNED_MASK_PROBABILITY_THRESHOLD,
    MINIMUM_MATTERING_FRAMES,
    REASON_CHANGED,
    REASON_EMPTY_MASK,
    REASON_NO_DETECTION,
    REASON_UNCHANGED,
    build_report,
    cells_touched_by_pixels,
    content_digest,
    corpus_gate,
    frame_mask_sensitivity,
    masked_frame_quantities,
    pixel_cell,
    score_corpus,
    score_frame,
)
from lolo_agent.pixels import Frame
from lolo_agent.run_logging import encode_png
from lolo_agent.tracker_substitution_replay import grid_cell_pixel_block
from lolo_agent.unlabeled_entities import UnlabeledEntityMemory


@dataclass(frozen=True)
class _StubPrediction:
    columns: int
    rows: int
    probabilities: Tuple[Tuple[float, ...], ...]


def _prediction(columns: int, rows: int, cells: dict) -> _StubPrediction:
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


def _frame(
    width: int, height: int, values: Dict[Tuple[int, int], int], background: int = 10
) -> Frame:
    pixels = bytearray([background]) * (width * height)
    for (x, y), value in values.items():
        pixels[y * width + x] = value
    return Frame(width=width, height=height, channels=1, pixels=bytes(pixels))


def _block(x0: int, x1: int, y0: int, y1: int, value: int) -> Dict[Tuple[int, int], int]:
    return {(x, y): value for y in range(y0, y1) for x in range(x0, x1)}


# 32x30 single-channel frames over a 4x3 coarse grid: each cell is an
# 8x10 pixel block, matching the feature_at partition exactly.
_WIDTH, _HEIGHT = 32, 30
_COLUMNS, _ROWS = 4, 3
# Cell (1, 1) pixel block.
_SPRITE_BLOCK = frozenset(
    (x, y) for y in range(10, 20) for x in range(8, 16)
)
_SPRITE_SLOT = (8, 10)


def _memory() -> UnlabeledEntityMemory:
    return UnlabeledEntityMemory(columns=_COLUMNS, rows=_ROWS)


def _sprite_frame() -> Frame:
    """Bright sprite filling cell (1, 1) on a dark background."""

    return _frame(_WIDTH, _HEIGHT, _block(8, 16, 10, 20, 200))


class ThresholdPinningTests(unittest.TestCase):
    def test_preregistered_constants_pinned(self) -> None:
        # The mask and appearance thresholds are the substitution replay's
        # established operating points, reused by import; the agreement
        # rate and minimum mattering count are this gate's own
        # preregistration.  Pinning them keeps the gate from drifting to a
        # tuned rule after seeing results.
        self.assertEqual(LEARNED_MASK_PROBABILITY_THRESHOLD, 0.5)
        self.assertEqual(APPEARANCE_L1_THRESHOLD, 0.08)
        self.assertEqual(AGREEMENT_RATE_THRESHOLD, 0.95)
        self.assertEqual(MINIMUM_MATTERING_FRAMES, 50)
        self.assertIs(
            LEARNED_MASK_PROBABILITY_THRESHOLD,
            tracker_substitution_replay.LEARNED_MASK_PROBABILITY_THRESHOLD,
        )
        self.assertIs(
            APPEARANCE_L1_THRESHOLD,
            tracker_substitution_replay.APPEARANCE_L1_THRESHOLD,
        )


class GridPartitionTests(unittest.TestCase):
    def test_pixel_cell_inverts_feature_grid_partition(self) -> None:
        # Awkward non-divisible dimensions: every pixel of every cell's
        # block must map back to that cell.
        width, height, columns, rows = 10, 9, 4, 3
        for row in range(rows):
            for column in range(columns):
                for x, y in grid_cell_pixel_block(
                    (column, row), width, height, columns, rows
                ):
                    self.assertEqual(
                        pixel_cell(x, y, width, height, columns, rows),
                        (column, row),
                    )

    def test_cells_touched_by_pixels_sorted_unique_in_bounds(self) -> None:
        pixels = frozenset({(0, 0), (1, 1), (9, 8), (50, 50), (-1, 2)})
        self.assertEqual(
            cells_touched_by_pixels(pixels, 10, 9, 4, 3),
            ((0, 0), (3, 2)),
        )


class MatteringDetectorTests(unittest.TestCase):
    def test_detects_frame_where_masking_changes_fingerprint(self) -> None:
        # Erasing the bright sprite empties its cell's pools (encoded as
        # zeros), so the masked quantities differ from the unmasked ones:
        # the mask has causal influence and the frame is mattering.
        result = frame_mask_sensitivity(
            _sprite_frame(), _SPRITE_SLOT, _SPRITE_BLOCK, _memory()
        )
        self.assertTrue(result["mattering"])
        self.assertEqual(result["reason"], REASON_CHANGED)
        self.assertEqual(result["assisted_cells"], ((1, 1),))
        self.assertEqual(result["changed_cells"], ((1, 1),))
        self.assertNotEqual(
            result["assisted_signature"], result["unmasked_signature"]
        )

    def test_rejects_frame_where_masking_changes_nothing(self) -> None:
        # A uniform dim background quantizes to zero with or without the
        # mask (fully masked pools also encode zero), so the same mask has
        # no causal influence and the frame is non-mattering.
        result = frame_mask_sensitivity(
            _frame(_WIDTH, _HEIGHT, {}), _SPRITE_SLOT, _SPRITE_BLOCK, _memory()
        )
        self.assertFalse(result["mattering"])
        self.assertEqual(result["reason"], REASON_UNCHANGED)
        self.assertEqual(result["assisted_cells"], ((1, 1),))
        self.assertEqual(result["changed_cells"], ())

    def test_no_detection_and_empty_mask_are_never_mattering(self) -> None:
        frame = _sprite_frame()
        no_detection = frame_mask_sensitivity(
            frame, None, frozenset(), _memory()
        )
        self.assertFalse(no_detection["mattering"])
        self.assertEqual(no_detection["reason"], REASON_NO_DETECTION)
        empty_mask = frame_mask_sensitivity(
            frame, _SPRITE_SLOT, frozenset(), _memory()
        )
        self.assertFalse(empty_mask["mattering"])
        self.assertEqual(empty_mask["reason"], REASON_EMPTY_MASK)

    def test_signature_and_fingerprint_views_are_consistent(self) -> None:
        quantities = masked_frame_quantities(
            _sprite_frame(), ((1, 1),), _SPRITE_SLOT, _SPRITE_BLOCK, _memory()
        )
        self.assertEqual(set(quantities.features), {(1, 1)})
        self.assertEqual(set(quantities.fingerprints), {(1, 1)})
        # The fully masked cell encodes as zeros.
        self.assertEqual(set(quantities.features[(1, 1)]), {0})


class ScoreFrameTests(unittest.TestCase):
    def test_learned_mask_equal_to_assisted_agrees(self) -> None:
        prediction = _prediction(_COLUMNS, _ROWS, {(1, 1): 1.0})
        row = score_frame(
            _sprite_frame(),
            prediction,
            _memory(),
            _SPRITE_SLOT,
            _SPRITE_BLOCK,
        )
        self.assertTrue(row["mattering"])
        self.assertTrue(row["signature_equal"])
        self.assertTrue(row["l1_within"])
        self.assertTrue(row["agrees"])
        self.assertEqual(row["max_cell_l1"], 0.0)
        self.assertEqual(row["mask_divergence"]["iou"], 1.0)

    def test_empty_learned_mask_disagrees_on_mattering_frame(self) -> None:
        prediction = _prediction(_COLUMNS, _ROWS, {})
        row = score_frame(
            _sprite_frame(),
            prediction,
            _memory(),
            _SPRITE_SLOT,
            _SPRITE_BLOCK,
        )
        self.assertTrue(row["mattering"])
        self.assertFalse(row["signature_equal"])
        self.assertFalse(row["l1_within"])
        self.assertFalse(row["agrees"])
        self.assertGreater(row["max_cell_l1"], APPEARANCE_L1_THRESHOLD)
        self.assertEqual(row["mask_divergence"]["iou"], 0.0)

    def test_mislocalized_learned_mask_disagrees(self) -> None:
        # Learned mass on a different cell: the sprite cell is compared
        # unmasked under the learned convention and disagrees.
        prediction = _prediction(_COLUMNS, _ROWS, {(3, 0): 1.0})
        row = score_frame(
            _sprite_frame(),
            prediction,
            _memory(),
            _SPRITE_SLOT,
            _SPRITE_BLOCK,
        )
        self.assertTrue(row["mattering"])
        self.assertFalse(row["agrees"])
        self.assertEqual(row["comparison_cells"], 2)

    def test_learned_over_coverage_erasing_anonymous_object_disagrees(
        self,
    ) -> None:
        # Sprite at (1, 1) plus a bright anonymous object at (2, 1).  A
        # learned mask covering both cells erases the object's appearance,
        # which the assisted computation preserves -- caught because the
        # comparison cells are the union of both masks' touched cells.
        values = dict(_block(8, 16, 10, 20, 200))
        values.update(_block(16, 24, 10, 20, 200))
        frame = _frame(_WIDTH, _HEIGHT, values)
        prediction = _prediction(_COLUMNS, _ROWS, {(1, 1): 1.0, (2, 1): 1.0})
        row = score_frame(
            frame, prediction, _memory(), _SPRITE_SLOT, _SPRITE_BLOCK
        )
        self.assertTrue(row["mattering"])
        self.assertFalse(row["signature_equal"])
        self.assertFalse(row["l1_within"])
        self.assertFalse(row["agrees"])

    def test_non_mattering_frame_is_still_scored_not_gated(self) -> None:
        row = score_frame(
            _frame(_WIDTH, _HEIGHT, {}),
            _prediction(_COLUMNS, _ROWS, {(1, 1): 1.0}),
            _memory(),
            _SPRITE_SLOT,
            _SPRITE_BLOCK,
        )
        self.assertFalse(row["mattering"])
        self.assertTrue(row["agrees"])


class CorpusGateTests(unittest.TestCase):
    @staticmethod
    def _rows(agreeing: int, disagreeing: int) -> list:
        rows = []
        for index in range(agreeing + disagreeing):
            agrees = index < agreeing
            rows.append(
                {
                    "signature_equal": agrees,
                    "l1_within": agrees,
                    "agrees": agrees,
                }
            )
        return rows

    def test_rate_boundary_is_inclusive_at_0_95(self) -> None:
        gate = corpus_gate(self._rows(95, 5))
        self.assertTrue(gate["signature_bit"])
        self.assertTrue(gate["fingerprint_l1_bit"])
        self.assertTrue(gate["passed"])
        below = corpus_gate(self._rows(94, 6))
        self.assertFalse(below["signature_bit"])
        self.assertFalse(below["passed"])

    def test_insufficient_mattering_frames_fail_the_gate(self) -> None:
        # Perfect agreement on too few mattering frames is vacuous, not a
        # pass: the instrument cannot demonstrate mask sensitivity.
        gate = corpus_gate(self._rows(MINIMUM_MATTERING_FRAMES - 1, 0))
        self.assertTrue(gate["mattering_frames"] < MINIMUM_MATTERING_FRAMES)
        self.assertFalse(gate["mattering_frames_sufficient"])
        self.assertFalse(gate["passed"])
        sufficient = corpus_gate(self._rows(MINIMUM_MATTERING_FRAMES, 0))
        self.assertTrue(sufficient["passed"])

    def test_empty_corpus_fails(self) -> None:
        gate = corpus_gate([])
        self.assertFalse(gate["passed"])


class ReportTests(unittest.TestCase):
    def _corpus_results(self) -> list:
        rows = CorpusGateTests._rows(95, 5)
        for index, row in enumerate(rows):
            row.update(
                {
                    "frame": f"digest-{index:04d}",
                    "mask_divergence": {"iou": 0.9},
                    "max_cell_l1": 0.0,
                }
            )
        return [
            {
                "run_id": "corpus-a",
                "frames": 120,
                "frames_by_reason": {REASON_CHANGED: 100},
                "mattering": {"frames": 100},
                "non_mattering": {"frames": 20},
                "gate": corpus_gate(rows),
            }
        ]

    def test_report_digest_deterministic_and_content_sensitive(self) -> None:
        results = self._corpus_results()
        provenance = {"checkpoint": "tracker-v4.pt"}
        first = build_report(results, provenance)
        second = build_report(results, provenance)
        self.assertEqual(first["content_digest"], second["content_digest"])
        self.assertEqual(
            first["content_digest"], content_digest(first)
        )
        mutated = build_report(
            results, {"checkpoint": "tracker-v3.pt"}
        )
        self.assertNotEqual(
            first["content_digest"], mutated["content_digest"]
        )
        self.assertEqual(first["result"]["gate"], "PASS")
        self.assertIn("PROMOTE-to-shadow", first["result"]["verdict"])

    def test_failing_corpus_yields_no_promote(self) -> None:
        results = self._corpus_results()
        results[0]["gate"] = corpus_gate(CorpusGateTests._rows(80, 20))
        report = build_report(results, {})
        self.assertEqual(report["result"]["gate"], "FAIL")
        self.assertIn("NO-PROMOTE", report["result"]["verdict"])

    def test_empty_report_fails(self) -> None:
        report = build_report([], {})
        self.assertEqual(report["result"]["gate"], "FAIL")


class ScoreCorpusIntegrationTests(unittest.TestCase):
    def test_synthetic_corpus_end_to_end_deterministic(self) -> None:
        # Dark synthetic frames yield no assisted detection, so every
        # frame lands in the non-mattering bucket with trivial agreement
        # and the gate honestly fails for lack of mattering frames.
        frames = [
            _frame(_WIDTH, _HEIGHT, {(x, 0): 12 + x}) for x in range(3)
        ]
        with tempfile.TemporaryDirectory() as scratch:
            run_dir = Path(scratch) / "corpus"
            frames_dir = run_dir / "frames"
            frames_dir.mkdir(parents=True)
            for frame in frames:
                (frames_dir / f"{frame.digest}.png").write_bytes(
                    encode_png(frame)
                )
            predictor = _StubPredictor(_prediction(16, 15, {}))
            result = score_corpus(run_dir, predictor)
            again = score_corpus(run_dir, predictor)
        self.assertEqual(result, again)
        self.assertEqual(result["frames"], 3)
        self.assertEqual(
            result["frames_by_reason"], {REASON_NO_DETECTION: 3}
        )
        self.assertEqual(result["mattering"]["frames"], 0)
        self.assertEqual(result["non_mattering"]["frames"], 3)
        self.assertEqual(
            result["non_mattering"]["joint_agreement_rate"], 1.0
        )
        self.assertEqual(result["non_mattering"]["disagreeing_frames"], [])
        self.assertFalse(result["gate"]["passed"])
        report = build_report([result], {"checkpoint": "stub"})
        self.assertEqual(report["result"]["gate"], "FAIL")
        json.dumps(report, sort_keys=True, allow_nan=False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
