from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from lolo_agent import mask_sensitive_gate, tracker_substitution_replay
from lolo_agent.functional_mask_gate import (
    ADJACENCY_CHEBYSHEV_RADIUS,
    AGREEMENT_RATE_THRESHOLD,
    APPEARANCE_L1_THRESHOLD,
    ASSISTED_CONVENTION,
    DEFAULT_HEAD_CHECKPOINT_V2,
    DEFAULT_REPORT_V3,
    DETECTION_QUANTITY_V2,
    DIFFERENTIAL_FP_IDENTICAL_RATE_BOUND,
    DIFFERENTIAL_FP_UNCORROBORATED_RATE_BOUND,
    DIFFERENTIAL_MIN_CHANGED_PIXELS,
    GATE_VERSION_V2,
    LEARNED_CONVENTION,
    LEARNED_MASK_PROBABILITY_THRESHOLD,
    MINIMUM_MEASUREMENTS,
    NO_EFFECT_CLASS_IDENTICAL,
    NO_EFFECT_CLASS_UNCORROBORATED,
    _DIGEST_PREFIX_V2,
    build_report,
    build_report_v2,
    component_pixel_region,
    content_digest,
    detection_bit,
    detection_bit_v2,
    detection_measurements,
    differential_false_positive_bit,
    hidden_pixels,
    labeled_arms_from_record,
    masked_region_differential,
    no_effect_measurements,
    preservation_bit,
    preservation_l1,
    preservation_measurements,
    region_content_digest,
    score_corpus,
    score_corpus_v2,
    score_detection,
    score_detection_v2,
    stability_bit,
    stability_l1,
    stability_measurements,
)
from lolo_agent.pixels import Frame
from lolo_agent.run_logging import encode_png
from lolo_agent.unlabeled_entities import UnlabeledEntityMemory

Cell = Tuple[int, int]
Pixel = Tuple[int, int]


# ---------------------------------------------------------------------------
# Synthetic fixtures: 32x30 single-channel frames over a 4x3 coarse grid,
# each cell an exact 8x10 pixel block of the feature_at partition.  The
# background quantizes to a NON-zero feature value (40 // 16 = 2) so a
# fully masked pool (encoded as zeros) is distinguishable from background.
# ---------------------------------------------------------------------------

_WIDTH, _HEIGHT = 32, 30
_COLUMNS, _ROWS = 4, 3
_BACKGROUND = 40
_PLAYER_VALUE = 200
_OBJECT_VALUE = 180


def _frame(values: Dict[Pixel, int]) -> Frame:
    pixels = bytearray([_BACKGROUND]) * (_WIDTH * _HEIGHT)
    for (x, y), value in values.items():
        pixels[y * _WIDTH + x] = value
    return Frame(
        width=_WIDTH, height=_HEIGHT, channels=1, pixels=bytes(pixels)
    )


def _cell_block(cell: Cell) -> FrozenSet[Pixel]:
    column, row = cell
    return frozenset(
        (x, y)
        for y in range(row * 10, (row + 1) * 10)
        for x in range(column * 8, (column + 1) * 8)
    )


def _cell_values(cell: Cell, value: int) -> Dict[Pixel, int]:
    return {pixel: value for pixel in _cell_block(cell)}


def _memory() -> UnlabeledEntityMemory:
    return UnlabeledEntityMemory(columns=_COLUMNS, rows=_ROWS)


def _slot(cell: Cell) -> Pixel:
    return (cell[0] * 8, cell[1] * 10)


@dataclass(frozen=True)
class _StubConvention:
    """Mask convention driven by a digest-keyed table of fixed masks."""

    name: str
    masks: Mapping[str, Tuple[Optional[Pixel], FrozenSet[Pixel]]] = field(
        default_factory=dict
    )

    def mask(self, frame: Frame) -> Tuple[Optional[Pixel], FrozenSet[Pixel]]:
        return self.masks.get(frame.digest, (None, frozenset()))


def _convention(
    name: str, table: Dict[Frame, Tuple[Cell, Tuple[Cell, ...]]]
) -> _StubConvention:
    """Build a stub convention masking the given cells per frame."""

    masks = {}
    for frame, (slot_cell, masked_cells) in table.items():
        pixels: FrozenSet[Pixel] = frozenset().union(
            *(_cell_block(cell) for cell in masked_cells)
        )
        masks[frame.digest] = (_slot(slot_cell), pixels)
    return _StubConvention(name=name, masks=masks)


class ThresholdPinningTests(unittest.TestCase):
    def test_preregistered_constants_are_prior_operating_points(self) -> None:
        # Every threshold is a prior published operating point reused by
        # import, so the functional gate cannot drift to a tuned rule.
        self.assertEqual(LEARNED_MASK_PROBABILITY_THRESHOLD, 0.5)
        self.assertEqual(APPEARANCE_L1_THRESHOLD, 0.08)
        self.assertEqual(AGREEMENT_RATE_THRESHOLD, 0.95)
        self.assertEqual(MINIMUM_MEASUREMENTS, 50)
        self.assertEqual(ADJACENCY_CHEBYSHEV_RADIUS, 1)
        self.assertIs(
            APPEARANCE_L1_THRESHOLD,
            tracker_substitution_replay.APPEARANCE_L1_THRESHOLD,
        )
        self.assertIs(
            LEARNED_MASK_PROBABILITY_THRESHOLD,
            tracker_substitution_replay.LEARNED_MASK_PROBABILITY_THRESHOLD,
        )
        self.assertIs(
            AGREEMENT_RATE_THRESHOLD,
            mask_sensitive_gate.AGREEMENT_RATE_THRESHOLD,
        )
        self.assertIs(
            MINIMUM_MEASUREMENTS,
            mask_sensitive_gate.MINIMUM_MATTERING_FRAMES,
        )


class DetectionFixtureTests(unittest.TestCase):
    """Bit (a): a ground-truth change the convention must detect."""

    def setUp(self) -> None:
        # Control endpoint: player at (1, 1).  Factual endpoint: player
        # moved to (0, 1) and an object now occupies (2, 1) -- the
        # ground-truth component includes the genuinely changed object
        # cell (2, 1).
        self.control = _frame(_cell_values((1, 1), _PLAYER_VALUE))
        factual_values = dict(_cell_values((0, 1), _PLAYER_VALUE))
        factual_values.update(_cell_values((2, 1), _OBJECT_VALUE))
        self.factual = _frame(factual_values)
        self.component = ((0, 1), (1, 1), (2, 1))
        self.correct = _convention(
            LEARNED_CONVENTION,
            {
                self.factual: ((0, 1), ((0, 1),)),
                self.control: ((1, 1), ((1, 1),)),
            },
        )

    def test_correct_convention_detects_ground_truth_change(self) -> None:
        row = score_detection(
            self.factual, self.control, self.component, self.correct, _memory()
        )
        self.assertTrue(row["detected"])
        self.assertGreater(row["current_cells"], 0)
        self.assertTrue(row["views_consistent"])
        self.assertTrue(row["factual_masked"])
        self.assertTrue(row["control_masked"])

    def test_over_covering_convention_misses_and_scorer_catches_it(
        self,
    ) -> None:
        # A mask that erases the whole component in BOTH endpoints makes
        # the two states identical: the manipulation is missed, and the
        # scorer must report detected=False rather than letter-pass.
        absorbing = _convention(
            LEARNED_CONVENTION,
            {
                self.factual: ((0, 1), self.component),
                self.control: ((1, 1), self.component),
            },
        )
        row = score_detection(
            self.factual,
            self.control,
            self.component,
            absorbing,
            _memory(),
        )
        self.assertFalse(row["detected"])
        self.assertEqual(row["current_cells"], 0)
        self.assertEqual(row["observations"], 0)
        self.assertTrue(row["views_consistent"])

    def test_correspondence_view_lifts_changed_cells(self) -> None:
        row = score_detection(
            self.factual, self.control, self.component, self.correct, _memory()
        )
        # The object cell (2, 1) and the mask-swapped player cells all
        # register as still-changed observations under the track-state
        # derivation; the exact count is the per-cell signature diff.
        self.assertEqual(row["current_cells"], row["observations"])
        self.assertGreaterEqual(row["current_cells"], 1)


class StabilityFixtureTests(unittest.TestCase):
    """Bit (b): a self-consistency violation the scorer must catch."""

    def setUp(self) -> None:
        # Two frames byte-identical at the object cell (2, 1) while the
        # player stands at different cells.
        first_values = dict(_cell_values((1, 1), _PLAYER_VALUE))
        first_values.update(_cell_values((2, 1), _OBJECT_VALUE))
        self.first = _frame(first_values)
        second_values = dict(_cell_values((0, 1), _PLAYER_VALUE))
        second_values.update(_cell_values((2, 1), _OBJECT_VALUE))
        self.second = _frame(second_values)
        self.cell = (2, 1)

    def test_stable_convention_yields_zero_l1(self) -> None:
        stable = _convention(
            LEARNED_CONVENTION,
            {
                self.first: ((1, 1), ((1, 1),)),
                self.second: ((0, 1), ((0, 1),)),
            },
        )
        value = stability_l1(
            self.first, self.second, self.cell, stable, _memory()
        )
        self.assertEqual(value, 0.0)
        self.assertLessEqual(value, APPEARANCE_L1_THRESHOLD)

    def test_mask_spill_violates_the_bound_and_is_caught(self) -> None:
        # The convention's mask covers the object cell in one frame but
        # not the other: the same object bytes yield different
        # fingerprints -- pure masking-convention noise the scorer must
        # flag as a violation.
        spilling = _convention(
            LEARNED_CONVENTION,
            {
                self.first: ((1, 1), ((1, 1), (2, 1))),
                self.second: ((0, 1), ((0, 1),)),
            },
        )
        value = stability_l1(
            self.first, self.second, self.cell, spilling, _memory()
        )
        self.assertGreater(value, APPEARANCE_L1_THRESHOLD)


class PreservationFixtureTests(unittest.TestCase):
    """Bit (c): the v316/v317 player-absorption defect class."""

    def setUp(self) -> None:
        values = dict(_cell_values((1, 1), _PLAYER_VALUE))
        values.update(_cell_values((2, 1), _OBJECT_VALUE))
        self.frame = _frame(values)
        self.object_cell = (2, 1)

    def test_clean_mask_preserves_the_adjacent_object(self) -> None:
        clean = _convention(
            LEARNED_CONVENTION, {self.frame: ((1, 1), ((1, 1),))}
        )
        value = preservation_l1(
            self.frame, self.object_cell, clean, _memory()
        )
        self.assertEqual(value, 0.0)

    def test_absorbing_mask_erases_the_adjacent_object(self) -> None:
        absorbing = _convention(
            ASSISTED_CONVENTION,
            {self.frame: ((1, 1), ((1, 1), (2, 1)))},
        )
        value = preservation_l1(
            self.frame, self.object_cell, absorbing, _memory()
        )
        self.assertGreater(value, APPEARANCE_L1_THRESHOLD)

    def test_regression_ordering_is_scored_not_assumed(self) -> None:
        # learned at least as good as assisted -> pass; worse -> fail.
        no_regression = preservation_bit(
            [True] * MINIMUM_MEASUREMENTS,
            [True] * (MINIMUM_MEASUREMENTS - 5) + [False] * 5,
        )
        self.assertTrue(no_regression["passed"])
        equal = preservation_bit(
            [True] * MINIMUM_MEASUREMENTS, [True] * MINIMUM_MEASUREMENTS
        )
        self.assertTrue(equal["passed"])
        regression = preservation_bit(
            [True] * (MINIMUM_MEASUREMENTS - 5) + [False] * 5,
            [True] * MINIMUM_MEASUREMENTS,
        )
        self.assertFalse(regression["passed"])


def _record(
    arms: list,
    *,
    group: int = 1,
    columns: int = _COLUMNS,
    rows: int = _ROWS,
) -> Dict[str, object]:
    return {"group": group, "columns": columns, "rows": rows, "arms": arms}


def _arm(
    action: str,
    factual: str,
    control: str,
    changed: list,
    controllable: list,
    *,
    duration: int = 8,
    status: str = "labeled",
) -> Dict[str, object]:
    return {
        "action": action,
        "duration": duration,
        "endpoint_digests": [factual],
        "control_digest": control,
        "status": status,
        "censor_reason": None if status == "labeled" else "absent_control",
        "corroborating_arms": 1,
        "changed_cells": changed,
        "controllable_components": [controllable] if controllable else [],
        "controllable_cells": controllable,
        "residual_cells": [],
    }


class MeasurementEnumerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = _record(
            [
                _arm(
                    "left",
                    "f-left",
                    "c-8",
                    [[0, 1], [1, 1]],
                    [[0, 1], [1, 1]],
                ),
                _arm(
                    "right",
                    "f-right",
                    "c-8",
                    [[1, 1], [2, 1]],
                    [[1, 1], [2, 1]],
                ),
                # Labeled but unscored: no controllable component and an
                # empty changed set -- never a stability sibling.
                _arm("b", "f-b", "c-8", [], []),
                # Censored arms never participate.
                _arm("up", "f-up", "c-8", [[9, 9]], [[9, 9]], status="censored"),
            ]
        )

    def test_labeled_arms_reader_skips_censored(self) -> None:
        arms = labeled_arms_from_record(self.record)
        self.assertEqual([arm.action for arm in arms], ["left", "right", "b"])
        self.assertEqual(
            [arm.scored for arm in arms], [True, True, False]
        )

    def test_detection_measurements_deduplicate_with_counts(self) -> None:
        measurements = detection_measurements([self.record, self.record])
        self.assertEqual(len(measurements), 2)
        self.assertEqual(set(measurements.values()), {2})
        self.assertIn(
            ("f-left", "c-8", ((0, 1), (1, 1))), measurements
        )

    def test_stability_cells_are_component_minus_sibling_changed(self) -> None:
        measurements = stability_measurements([self.record])
        # i=left against j=right: component {(0,1),(1,1)} minus changed
        # {(1,1),(2,1)} leaves (0,1) measured on right's endpoint pair;
        # i=right against j=left leaves (2,1) on left's pair.  The "b"
        # arm has an empty changed set and contributes nothing.
        self.assertEqual(
            dict(measurements),
            {
                ("f-right", "c-8", (0, 1)): 1,
                ("f-left", "c-8", (2, 1)): 1,
            },
        )

    def test_stability_skips_same_action_and_duration_identity(self) -> None:
        twin = _record(
            [
                _arm(
                    "left",
                    "f-1",
                    "c-8",
                    [[0, 1]],
                    [[0, 1]],
                ),
                _arm(
                    "left",
                    "f-2",
                    "c-16",
                    [[1, 1]],
                    [[1, 1]],
                    duration=16,
                ),
            ]
        )
        measurements = stability_measurements([twin])
        # Different durations of the same action DO pair; identical
        # (action, duration) identities never self-pair.
        self.assertEqual(
            dict(measurements),
            {
                ("f-2", "c-16", (0, 1)): 1,
                ("f-1", "c-8", (1, 1)): 1,
            },
        )

    def test_preservation_cells_adjacent_in_grid_not_changed(self) -> None:
        measurements = preservation_measurements([self.record])
        left_cells = {
            cell for digest, cell in measurements if digest == "f-left"
        }
        # Chebyshev-1 neighbourhood of {(0,1),(1,1)} clipped to the 4x3
        # grid, minus the arm's changed cells.
        self.assertEqual(
            left_cells,
            {(0, 0), (1, 0), (2, 0), (2, 1), (0, 2), (1, 2), (2, 2)},
        )
        right_cells = {
            cell for digest, cell in measurements if digest == "f-right"
        }
        self.assertEqual(
            right_cells,
            {
                (0, 0), (1, 0), (2, 0), (3, 0),
                (0, 1), (3, 1),
                (0, 2), (1, 2), (2, 2), (3, 2),
            },
        )
        self.assertEqual(len(measurements), 17)

    def test_preservation_duplicates_counted_once_each_extra(self) -> None:
        measurements = preservation_measurements([self.record, self.record])
        self.assertEqual(len(measurements), 17)
        self.assertEqual(set(measurements.values()), {2})


class GateBitTests(unittest.TestCase):
    @staticmethod
    def _rows(learned: list, assisted: list) -> list:
        return [
            {
                LEARNED_CONVENTION: {"detected": l},
                ASSISTED_CONVENTION: {"detected": a},
            }
            for l, a in zip(learned, assisted)
        ]

    def test_detection_bit_inclusive_boundary_and_both_directions(self) -> None:
        learned = [True] * 95 + [False] * 5
        assisted = [True] * 100
        bit = detection_bit(self._rows(learned, assisted))
        self.assertEqual(bit["learned_rate_vs_ground_truth"], 0.95)
        self.assertEqual(bit["learned_rate_given_assisted"], 0.95)
        self.assertEqual(bit["assisted_rate_given_learned"], 1.0)
        self.assertTrue(bit["passed"])
        below = detection_bit(
            self._rows([True] * 94 + [False] * 6, assisted)
        )
        self.assertFalse(below["passed"])

    def test_detection_bit_ground_truth_is_the_referee(self) -> None:
        # The assisted convention detecting nothing does not excuse the
        # learned convention from the ground-truth condition -- and a
        # learned convention that out-detects assisted is not penalized.
        learned = [True] * 100
        assisted = [False] * 100
        bit = detection_bit(self._rows(learned, assisted))
        self.assertTrue(bit["assisted_condition_vacuous"])
        self.assertTrue(bit["passed"])
        self.assertEqual(bit["assisted_rate_vs_ground_truth"], 0.0)
        missing = detection_bit(self._rows([False] * 100, [False] * 100))
        self.assertFalse(missing["passed"])

    def test_detection_bit_insufficient_measurements_fail(self) -> None:
        rows = self._rows(
            [True] * (MINIMUM_MEASUREMENTS - 1),
            [True] * (MINIMUM_MEASUREMENTS - 1),
        )
        self.assertFalse(detection_bit(rows)["passed"])

    def test_stability_bit_gates_learned_only(self) -> None:
        learned = [True] * 95 + [False] * 5
        assisted = [False] * 100
        bit = stability_bit(learned, assisted)
        self.assertTrue(bit["passed"])
        self.assertEqual(bit["assisted_stability_rate"], 0.0)
        failing = stability_bit([True] * 94 + [False] * 6, assisted)
        self.assertFalse(failing["passed"])
        vacuous = stability_bit([True] * 10, [True] * 10)
        self.assertFalse(vacuous["passed"])

    def test_preservation_bit_insufficient_measurements_fail(self) -> None:
        bit = preservation_bit([True] * 10, [True] * 10)
        self.assertFalse(bit["passed"])
        self.assertFalse(bit["measurements_sufficient"])


class ReportTests(unittest.TestCase):
    @staticmethod
    def _corpus_result(
        run_id: str,
        *,
        detect_learned: int = 100,
        stability_learned: int = 100,
        preserve_learned: int = 100,
        preserve_assisted: int = 100,
    ) -> Dict[str, object]:
        total = 100
        rows = GateBitTests._rows(
            [index < detect_learned for index in range(total)],
            [True] * total,
        )
        gate = {
            "bit_a_detection": detection_bit(rows),
            "bit_b_stability": stability_bit(
                [index < stability_learned for index in range(total)],
                [True] * total,
            ),
            "bit_c_preservation": preservation_bit(
                [index < preserve_learned for index in range(total)],
                [index < preserve_assisted for index in range(total)],
            ),
        }
        gate["passed"] = all(bit["passed"] for bit in gate.values())
        return {"run_id": run_id, "gate": gate}

    def test_pass_report_promotes_to_shadow(self) -> None:
        report = build_report(
            [self._corpus_result("corpus-a")], {"checkpoint": "v4"}
        )
        self.assertEqual(report["result"]["gate"], "PASS")
        self.assertIn("PROMOTE-to-shadow", report["result"]["verdict"])
        self.assertEqual(report["result"]["failing_mechanisms"], [])

    def test_fail_report_names_the_failing_mechanism(self) -> None:
        report = build_report(
            [
                self._corpus_result(
                    "corpus-a",
                    preserve_learned=90,
                    preserve_assisted=100,
                )
            ],
            {},
        )
        self.assertEqual(report["result"]["gate"], "FAIL")
        self.assertIn("NO-PROMOTE", report["result"]["verdict"])
        mechanisms = report["result"]["failing_mechanisms"]
        self.assertEqual(len(mechanisms), 1)
        self.assertIn("player-absorption regression", mechanisms[0])
        detection_fail = build_report(
            [self._corpus_result("corpus-b", detect_learned=80)], {}
        )
        self.assertTrue(
            any(
                "misses ground-truth manipulations" in mechanism
                for mechanism in detection_fail["result"][
                    "failing_mechanisms"
                ]
            )
        )
        stability_fail = build_report(
            [self._corpus_result("corpus-c", stability_learned=80)], {}
        )
        self.assertTrue(
            any(
                "unstable" in mechanism
                for mechanism in stability_fail["result"][
                    "failing_mechanisms"
                ]
            )
        )

    def test_digest_deterministic_and_content_sensitive(self) -> None:
        results = [self._corpus_result("corpus-a")]
        first = build_report(results, {"checkpoint": "v4"})
        second = build_report(results, {"checkpoint": "v4"})
        self.assertEqual(first["content_digest"], second["content_digest"])
        self.assertEqual(first["content_digest"], content_digest(first))
        mutated = build_report(results, {"checkpoint": "v3"})
        self.assertNotEqual(
            first["content_digest"], mutated["content_digest"]
        )

    def test_empty_report_fails(self) -> None:
        report = build_report([], {})
        self.assertEqual(report["result"]["gate"], "FAIL")
        self.assertIn(
            "no corpora scored", report["result"]["failing_mechanisms"]
        )


# ---------------------------------------------------------------------------
# End-to-end corpus scoring on a synthetic probe archive: 16x15 frames map
# cells 1:1 to pixels; the events exercise the real tracker_ood_eval
# extraction path and the counterfactual label rule.
# ---------------------------------------------------------------------------

_E2E_COLUMNS, _E2E_ROWS = 16, 15
_E2E_BACKGROUND = 40


def _e2e_frame(values: Dict[Cell, int]) -> Frame:
    pixels = bytearray([_E2E_BACKGROUND]) * (_E2E_COLUMNS * _E2E_ROWS)
    for (column, row), value in values.items():
        pixels[row * _E2E_COLUMNS + column] = value
    return Frame(
        width=_E2E_COLUMNS,
        height=_E2E_ROWS,
        channels=1,
        pixels=bytes(pixels),
    )


class ScoreCorpusEndToEndTests(unittest.TestCase):
    def test_synthetic_corpus_deterministic_and_honestly_insufficient(
        self,
    ) -> None:
        control = _e2e_frame({(5, 5): _PLAYER_VALUE})
        factual_left = _e2e_frame({(4, 5): _PLAYER_VALUE})
        factual_right = _e2e_frame({(6, 5): _PLAYER_VALUE})
        events = [
            {
                "event": "human_prior_option_branch_verified",
                "parent_state_id": "state-00000001",
                "path": ["left"],
                "durations": [8],
                "frame": factual_left.digest,
                "frame_width": _E2E_COLUMNS,
                "frame_height": _E2E_ROWS,
            },
            {
                "event": "human_prior_option_branch_verified",
                "parent_state_id": "state-00000001",
                "path": ["right"],
                "durations": [8],
                "frame": factual_right.digest,
                "frame_width": _E2E_COLUMNS,
                "frame_height": _E2E_ROWS,
            },
            {
                "event": "human_prior_option_local_neutral_verified",
                "parent_state_id": "state-00000001",
                "action": "noop",
                "action_frames": 8,
                "frame": control.digest,
            },
        ]
        # Both conventions mask exactly the player pixel of each frame.
        tables = {
            control.digest: ((5, 5), frozenset({(5, 5)})),
            factual_left.digest: ((4, 5), frozenset({(4, 5)})),
            factual_right.digest: ((6, 5), frozenset({(6, 5)})),
        }
        learned = _StubConvention(name=LEARNED_CONVENTION, masks=tables)
        assisted = _StubConvention(name=ASSISTED_CONVENTION, masks=tables)
        with tempfile.TemporaryDirectory() as scratch:
            run_dir = Path(scratch) / "corpus"
            frames_dir = run_dir / "frames"
            frames_dir.mkdir(parents=True)
            for frame in (control, factual_left, factual_right):
                (frames_dir / f"{frame.digest}.png").write_bytes(
                    encode_png(frame)
                )
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            result = score_corpus(run_dir, learned, assisted)
            again = score_corpus(run_dir, learned, assisted)
        self.assertEqual(result, again)
        self.assertEqual(result["extraction"]["scored_arms"], 2)
        self.assertEqual(result["detection"]["measurements"], 2)
        # Both arms' manipulations register under the correct masks.
        bit_a = result["gate"]["bit_a_detection"]
        self.assertEqual(bit_a["learned_rate_vs_ground_truth"], 1.0)
        self.assertEqual(bit_a["assisted_rate_vs_ground_truth"], 1.0)
        self.assertEqual(bit_a["learned_rate_given_assisted"], 1.0)
        # component {(4,5),(5,5)} minus changed {(5,5),(6,5)} and the
        # mirror image: two byte-certified stability cells, both stable.
        self.assertEqual(result["stability"]["measurements"], 2)
        self.assertEqual(
            result["gate"]["bit_b_stability"]["learned_stability_rate"],
            1.0,
        )
        # Chebyshev-1 neighbourhoods of the two components minus their
        # changed sets; the clean masks preserve every adjacent cell.
        self.assertEqual(result["preservation"]["measurements"], 20)
        bit_c = result["gate"]["bit_c_preservation"]
        self.assertEqual(bit_c["learned_preservation_rate"], 1.0)
        self.assertEqual(bit_c["assisted_preservation_rate"], 1.0)
        # Perfect rates over too few measurements are vacuous: every bit
        # honestly fails the minimum-measurement requirement.
        self.assertFalse(bit_a["passed"])
        self.assertFalse(result["gate"]["passed"])
        # Divergence telemetry covers the unique factual endpoints.
        self.assertEqual(result["divergence_telemetry"]["frames"], 2)
        self.assertEqual(
            result["divergence_telemetry"]["mask_iou"]["mean"], 1.0
        )
        report = build_report([result], {"checkpoint": "stub"})
        self.assertEqual(report["result"]["gate"], "FAIL")
        json.dumps(report, sort_keys=True, allow_nan=False)

    def test_convention_names_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            run_dir = Path(scratch) / "corpus"
            (run_dir / "frames").mkdir(parents=True)
            (run_dir / "events.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                score_corpus(
                    run_dir,
                    _StubConvention(name="wrong"),
                    _StubConvention(name=ASSISTED_CONVENTION),
                )


# ---------------------------------------------------------------------------
# Detection quantity v2 (section 4.38): the masked-region differential
# channel, its false-positive discipline, and the v2 scorer plumbing.
# ---------------------------------------------------------------------------


class DetectionQuantityV2ConstantTests(unittest.TestCase):
    def test_v2_constants_are_pinned(self) -> None:
        # The differential fires on any byte difference (no free tolerance
        # parameter); its false-positive bounds are per class -- exact
        # zero on byte-identical pairs, the published 0.95 operating
        # point reused by import as the raw-frame-differencing ceiling on
        # uncorroborated-change pairs.
        self.assertEqual(DIFFERENTIAL_MIN_CHANGED_PIXELS, 1)
        self.assertEqual(DIFFERENTIAL_FP_IDENTICAL_RATE_BOUND, 0.0)
        self.assertEqual(DIFFERENTIAL_FP_UNCORROBORATED_RATE_BOUND, 0.95)
        self.assertIs(
            DIFFERENTIAL_FP_UNCORROBORATED_RATE_BOUND,
            AGREEMENT_RATE_THRESHOLD,
        )
        self.assertEqual(
            DETECTION_QUANTITY_V2,
            "outside-mask-signature-or-masked-region-differential",
        )
        self.assertEqual(GATE_VERSION_V2, 2)
        self.assertEqual(
            DEFAULT_REPORT_V3,
            "experiments/lolo1-wp5/functional-gate-v3-report.json",
        )
        self.assertEqual(
            DEFAULT_HEAD_CHECKPOINT_V2,
            "experiments/lolo1-wp5/pixel-mask-head-v2.pt",
        )


class MaskedRegionDifferentialTests(unittest.TestCase):
    """The new channel catches in-place erasure; discipline keeps it quiet."""

    def setUp(self) -> None:
        # IN-PLACE change: the player transforms at cell (1, 1) without
        # displacing -- the control shows pose A, the factual pose B, at
        # the same cell.  A convention that masks the cell fully in BOTH
        # endpoints hides the entire effect from the outside-mask
        # signature quantity (fully masked pools encode zeros on both
        # sides), which is exactly the section 4.38 failure class.
        self.control = _frame(_cell_values((1, 1), _PLAYER_VALUE))
        self.factual = _frame(_cell_values((1, 1), _OBJECT_VALUE))
        self.component = ((1, 1),)
        self.covering = _convention(
            LEARNED_CONVENTION,
            {
                self.factual: ((1, 1), ((1, 1),)),
                self.control: ((1, 1), ((1, 1),)),
            },
        )

    def test_signature_misses_in_place_change_and_differential_catches(
        self,
    ) -> None:
        row = score_detection_v2(
            self.factual,
            self.control,
            self.component,
            self.covering,
            _memory(),
        )
        # The v1 quantity is structurally blind here...
        self.assertFalse(row["signature"]["detected"])
        # ...the dually-masked component region still carries the bytes.
        self.assertTrue(row["differential"]["fired"])
        self.assertEqual(row["differential"]["region_pixels"], 80)
        self.assertEqual(row["differential"]["changed_pixels"], 80)
        self.assertTrue(row["detected"])
        # And the v1 scorer alone would have missed the manipulation.
        v1 = score_detection(
            self.factual,
            self.control,
            self.component,
            self.covering,
            _memory(),
        )
        self.assertFalse(v1["detected"])

    def test_fired_equals_region_digest_inequality(self) -> None:
        slot, pixels = self.covering.mask(self.factual)
        region = pixels & component_pixel_region(
            self.component, _WIDTH, _HEIGHT, _COLUMNS, _ROWS
        )
        differential = masked_region_differential(
            self.factual,
            self.control,
            slot,
            pixels,
            slot,
            pixels,
            region_limit=component_pixel_region(
                self.component, _WIDTH, _HEIGHT, _COLUMNS, _ROWS
            ),
        )
        self.assertEqual(
            differential["fired"],
            region_content_digest(self.factual, region)
            != region_content_digest(self.control, region),
        )

    def test_animation_under_mask_outside_component_does_not_fire(
        self,
    ) -> None:
        # Ambient animation at (2, 1) differs between the endpoints and
        # the mask covers it in both -- but the component anchor keeps
        # the differential quiet: masked content outside the ground-truth
        # locus can never claim a detection of THIS manipulation.
        control_values = dict(_cell_values((1, 1), _PLAYER_VALUE))
        control_values.update(_cell_values((2, 1), 120))
        factual_values = dict(_cell_values((1, 1), _PLAYER_VALUE))
        factual_values.update(_cell_values((2, 1), 140))
        control = _frame(control_values)
        factual = _frame(factual_values)
        component = ((1, 1),)
        spilling = _convention(
            LEARNED_CONVENTION,
            {
                factual: ((1, 1), ((1, 1), (2, 1))),
                control: ((1, 1), ((1, 1), (2, 1))),
            },
        )
        row = score_detection_v2(
            factual, control, component, spilling, _memory()
        )
        self.assertFalse(row["differential"]["fired"])
        self.assertFalse(row["signature"]["detected"])
        self.assertFalse(row["detected"])
        # Without the anchor the same masked bytes WOULD fire -- the
        # anchored region is what disciplines the channel; the unanchored
        # form is reserved for the bit-(d) false-positive measurement,
        # where it upper-bounds the anchored channel by construction.
        slot, pixels = spilling.mask(factual)
        unanchored = masked_region_differential(
            factual, control, slot, pixels, slot, pixels
        )
        self.assertTrue(unanchored["fired"])
        self.assertEqual(unanchored["changed_pixels"], 80)

    def test_phase_matched_identical_endpoints_cannot_fire(self) -> None:
        # Duration-matched byte-identical endpoints (the phase discipline
        # holding exactly): no region can fire, even fully masked.
        row = score_detection_v2(
            self.control,
            self.control,
            self.component,
            self.covering,
            _memory(),
        )
        self.assertFalse(row["differential"]["fired"])
        self.assertEqual(row["differential"]["region_pixels"], 80)
        self.assertEqual(row["differential"]["changed_pixels"], 0)
        self.assertFalse(row["detected"])

    def test_unmasked_frame_hides_nothing(self) -> None:
        # A convention without an anchor slot hides no pixels: the
        # dually-hidden region is empty and the differential is inert
        # (the signature channel already sees such frames whole).
        self.assertEqual(
            hidden_pixels(None, frozenset(_cell_block((1, 1)))), frozenset()
        )
        differential = masked_region_differential(
            self.factual,
            self.control,
            None,
            frozenset(_cell_block((1, 1))),
            (8, 10),
            frozenset(_cell_block((1, 1))),
        )
        self.assertEqual(differential["region_pixels"], 0)
        self.assertFalse(differential["fired"])

    def test_mismatched_geometry_raises(self) -> None:
        other = Frame(
            width=16, height=15, channels=1, pixels=bytes(16 * 15)
        )
        with self.assertRaises(ValueError):
            masked_region_differential(
                self.factual, other, None, frozenset(), None, frozenset()
            )


class NoEffectMeasurementTests(unittest.TestCase):
    def test_no_effect_pairs_are_classed_and_deduplicated(self) -> None:
        record = _record(
            [
                _arm(
                    "left",
                    "f-left",
                    "c-8",
                    [[0, 1], [1, 1]],
                    [[0, 1], [1, 1]],
                ),
                # Byte-identical endpoints: the no-change class.
                _arm("b", "c-8", "c-8", [], []),
                # Changed but uncorroborated: the ambient/animation class.
                _arm("up", "f-up", "c-8", [[3, 2]], []),
                # Censored arms never participate.
                _arm(
                    "down",
                    "f-down",
                    "c-8",
                    [[9, 9]],
                    [[9, 9]],
                    status="censored",
                ),
            ]
        )
        measurements, conflicting = no_effect_measurements([record, record])
        self.assertEqual(conflicting, 0)
        self.assertEqual(
            dict(measurements),
            {
                ("c-8", "c-8"): {
                    "class": NO_EFFECT_CLASS_IDENTICAL,
                    "arms": 2,
                },
                ("f-up", "c-8"): {
                    "class": NO_EFFECT_CLASS_UNCORROBORATED,
                    "arms": 2,
                },
            },
        )

    def test_pairs_scored_elsewhere_are_excluded_as_conflicts(self) -> None:
        unscored = _record(
            [
                _arm("up", "f-x", "c-8", [[3, 2]], []),
                _arm("left", "f-left", "c-8", [[0, 1]], [[0, 1]]),
            ]
        )
        scored = _record(
            [_arm("up", "f-x", "c-8", [[3, 2]], [[3, 2]])], group=2
        )
        measurements, conflicting = no_effect_measurements(
            [unscored, scored]
        )
        self.assertEqual(conflicting, 1)
        self.assertEqual(dict(measurements), {})


class DetectionBitV2Tests(unittest.TestCase):
    @staticmethod
    def _rows(
        learned_signature: list,
        learned_differential: list,
        assisted_signature: list,
        assisted_differential: list,
    ) -> list:
        rows = []
        for ls, ld, as_, ad in zip(
            learned_signature,
            learned_differential,
            assisted_signature,
            assisted_differential,
        ):
            rows.append(
                {
                    LEARNED_CONVENTION: {
                        "signature": {"detected": ls},
                        "differential": {"fired": ld},
                        "detected": bool(ls or ld),
                    },
                    ASSISTED_CONVENTION: {
                        "signature": {"detected": as_},
                        "differential": {"fired": ad},
                        "detected": bool(as_ or ad),
                    },
                }
            )
        return rows

    def test_or_combination_and_per_channel_accounting(self) -> None:
        # 60 signature-only, 35 differential-only, 5 total misses: the
        # combined rate is 0.95 (inclusive boundary) while the signature
        # channel alone would fail at 0.60 -- and the report must show
        # exactly which channel carried each detection.
        learned_signature = [True] * 60 + [False] * 40
        learned_differential = [False] * 60 + [True] * 35 + [False] * 5
        assisted = [True] * 100
        bit = detection_bit_v2(
            self._rows(
                learned_signature,
                learned_differential,
                assisted,
                [False] * 100,
            )
        )
        self.assertEqual(bit["detection_quantity"], DETECTION_QUANTITY_V2)
        self.assertEqual(bit["learned_rate_vs_ground_truth"], 0.95)
        self.assertEqual(bit["learned_rate_given_assisted"], 0.95)
        channels = bit["channels"]
        self.assertEqual(
            channels[LEARNED_CONVENTION]["signature_detected"], 60
        )
        self.assertEqual(
            channels[LEARNED_CONVENTION]["differential_fired"], 35
        )
        self.assertEqual(
            channels[LEARNED_CONVENTION]["differential_only_detected"], 35
        )
        self.assertEqual(
            channels[LEARNED_CONVENTION]["signature_rate_vs_ground_truth"],
            0.6,
        )
        self.assertEqual(
            channels[ASSISTED_CONVENTION]["differential_only_detected"], 0
        )
        self.assertTrue(bit["passed"])
        below = detection_bit_v2(
            self._rows(
                learned_signature,
                [False] * 60 + [True] * 34 + [False] * 6,
                assisted,
                [False] * 100,
            )
        )
        self.assertFalse(below["passed"])

    def test_assisted_gets_the_same_upgrade(self) -> None:
        # The incumbent's differential detections count toward its own
        # rate AND toward the learned-given-assisted condition -- the
        # comparison is not rigged in either direction.
        learned = self._rows(
            [True] * 95 + [False] * 5,
            [False] * 100,
            [False] * 100,
            [True] * 100,
        )
        bit = detection_bit_v2(learned)
        self.assertEqual(bit["assisted_rate_vs_ground_truth"], 1.0)
        self.assertEqual(bit["learned_rate_given_assisted"], 0.95)
        self.assertEqual(
            bit["channels"][ASSISTED_CONVENTION][
                "differential_only_detected"
            ],
            100,
        )
        self.assertTrue(bit["passed"])

    def test_vacuous_assisted_condition_and_insufficiency(self) -> None:
        vacuous = detection_bit_v2(
            self._rows(
                [True] * 100, [False] * 100, [False] * 100, [False] * 100
            )
        )
        self.assertTrue(vacuous["assisted_condition_vacuous"])
        self.assertTrue(vacuous["passed"])
        short = detection_bit_v2(
            self._rows(
                [True] * (MINIMUM_MEASUREMENTS - 1),
                [False] * (MINIMUM_MEASUREMENTS - 1),
                [True] * (MINIMUM_MEASUREMENTS - 1),
                [False] * (MINIMUM_MEASUREMENTS - 1),
            )
        )
        self.assertFalse(short["passed"])


class DifferentialFalsePositiveBitTests(unittest.TestCase):
    @staticmethod
    def _rows(
        total: int,
        learned_fired: int,
        assisted_fired: int = 0,
        cls: str = NO_EFFECT_CLASS_UNCORROBORATED,
    ) -> list:
        return [
            {
                "class": cls,
                LEARNED_CONVENTION: {"fired": index < learned_fired},
                ASSISTED_CONVENTION: {"fired": index < assisted_fired},
            }
            for index in range(total)
        ]

    def test_identical_class_must_never_fire(self) -> None:
        # A single fire on a byte-identical pair breaks the phase
        # discipline: an instrument defect, not a tolerable rate.
        clean = differential_false_positive_bit(
            self._rows(100, 0, 0, NO_EFFECT_CLASS_IDENTICAL)
        )
        self.assertTrue(clean["identical_condition"])
        self.assertTrue(clean["passed"])
        broken = differential_false_positive_bit(
            self._rows(100, 1, 1, NO_EFFECT_CLASS_IDENTICAL)
        )
        self.assertFalse(broken["identical_condition"])
        self.assertFalse(broken["passed"])

    def test_uncorroborated_ceiling_is_inclusive(self) -> None:
        at_bound = differential_false_positive_bit(self._rows(100, 95, 95))
        self.assertEqual(
            at_bound["by_class"][NO_EFFECT_CLASS_UNCORROBORATED][
                "learned_rate"
            ],
            0.95,
        )
        self.assertTrue(at_bound["uncorroborated_condition"])
        self.assertTrue(at_bound["passed"])
        over = differential_false_positive_bit(self._rows(100, 96, 96))
        self.assertFalse(over["uncorroborated_condition"])
        self.assertFalse(over["passed"])

    def test_learned_regression_against_assisted_fails(self) -> None:
        # The incumbent received the same upgrade: the learned convention
        # firing on MORE certified-null pairs is a regression even when
        # both rates sit under the ceiling.
        regression = differential_false_positive_bit(self._rows(100, 10, 5))
        self.assertFalse(regression["no_regression_condition"])
        self.assertFalse(regression["passed"])
        parity = differential_false_positive_bit(self._rows(100, 10, 10))
        self.assertTrue(parity["passed"])

    def test_by_class_accounting_and_vacuous_flags(self) -> None:
        rows = self._rows(60, 3, 3) + self._rows(
            40, 0, 0, NO_EFFECT_CLASS_IDENTICAL
        )
        bit = differential_false_positive_bit(rows)
        self.assertEqual(bit["measurements"], 100)
        self.assertEqual(
            bit["by_class"][NO_EFFECT_CLASS_UNCORROBORATED],
            {
                "measurements": 60,
                "learned_fired": 3,
                "assisted_fired": 3,
                "learned_rate": 0.05,
            },
        )
        self.assertEqual(
            bit["by_class"][NO_EFFECT_CLASS_IDENTICAL],
            {
                "measurements": 40,
                "learned_fired": 0,
                "assisted_fired": 0,
                "learned_rate": 0.0,
            },
        )
        self.assertFalse(bit["identical_condition_vacuous"])
        self.assertFalse(bit["uncorroborated_condition_vacuous"])
        self.assertTrue(bit["passed"])
        # A corpus without an uncorroborated class leaves that condition
        # vacuous (flagged), not failed.
        identical_only = differential_false_positive_bit(
            self._rows(60, 0, 0, NO_EFFECT_CLASS_IDENTICAL)
        )
        self.assertTrue(identical_only["uncorroborated_condition_vacuous"])
        self.assertTrue(identical_only["passed"])

    def test_insufficient_measurements_fail(self) -> None:
        bit = differential_false_positive_bit(self._rows(10, 0))
        self.assertFalse(bit["passed"])
        self.assertFalse(bit["measurements_sufficient"])


class ReportV2Tests(unittest.TestCase):
    @staticmethod
    def _corpus_result(
        run_id: str,
        *,
        detect_signature: int = 60,
        detect_differential: int = 40,
        stability_learned: int = 100,
        preserve_learned: int = 100,
        preserve_assisted: int = 100,
        fp_learned: int = 0,
        fp_assisted: int = 0,
        fp_total: int = 100,
    ) -> Dict[str, object]:
        total = 100
        rows = DetectionBitV2Tests._rows(
            [index < detect_signature for index in range(total)],
            [
                detect_signature <= index < detect_signature
                + detect_differential
                for index in range(total)
            ],
            [True] * total,
            [False] * total,
        )
        gate = {
            "bit_a_detection": detection_bit_v2(rows),
            "bit_b_stability": stability_bit(
                [index < stability_learned for index in range(total)],
                [True] * total,
            ),
            "bit_c_preservation": preservation_bit(
                [index < preserve_learned for index in range(total)],
                [index < preserve_assisted for index in range(total)],
            ),
            "bit_d_differential_false_positive": (
                differential_false_positive_bit(
                    DifferentialFalsePositiveBitTests._rows(
                        fp_total, fp_learned, fp_assisted
                    )
                )
            ),
        }
        gate["passed"] = all(bit["passed"] for bit in gate.values())
        return {"run_id": run_id, "gate": gate}

    def test_pass_report_promotes_with_v2_quantity_named(self) -> None:
        report = build_report_v2(
            [self._corpus_result("corpus-a")], {"checkpoint": "v4"}
        )
        self.assertEqual(report["version"], GATE_VERSION_V2)
        self.assertEqual(report["result"]["gate"], "PASS")
        self.assertIn("PROMOTE-to-shadow", report["result"]["verdict"])
        self.assertIn("detection quantity v2", report["result"]["verdict"])
        self.assertEqual(report["result"]["failing_mechanisms"], [])
        self.assertEqual(
            report["detection_quantity"]["name"], DETECTION_QUANTITY_V2
        )
        self.assertEqual(
            report["thresholds"]["differential_fp_identical_rate_bound"],
            DIFFERENTIAL_FP_IDENTICAL_RATE_BOUND,
        )
        self.assertEqual(
            report["thresholds"][
                "differential_fp_uncorroborated_rate_bound"
            ],
            DIFFERENTIAL_FP_UNCORROBORATED_RATE_BOUND,
        )

    def test_bit_d_failures_name_each_condition(self) -> None:
        regression = build_report_v2(
            [self._corpus_result("corpus-a", fp_learned=10, fp_assisted=5)],
            {},
        )
        self.assertEqual(regression["result"]["gate"], "FAIL")
        mechanisms = regression["result"]["failing_mechanisms"]
        self.assertEqual(len(mechanisms), 1)
        self.assertIn("fires on more no-effect pairs", mechanisms[0])
        ceiling = build_report_v2(
            [
                self._corpus_result(
                    "corpus-b", fp_learned=96, fp_assisted=96
                )
            ],
            {},
        )
        self.assertTrue(
            any(
                "v312/v313" in mechanism
                for mechanism in ceiling["result"]["failing_mechanisms"]
            )
        )

    def test_bits_a_to_c_mechanisms_still_named(self) -> None:
        report = build_report_v2(
            [
                self._corpus_result(
                    "corpus-b", detect_signature=50, detect_differential=30
                )
            ],
            {},
        )
        self.assertTrue(
            any(
                "misses ground-truth manipulations" in mechanism
                for mechanism in report["result"]["failing_mechanisms"]
            )
        )
        stability_fail = build_report_v2(
            [self._corpus_result("corpus-c", stability_learned=80)], {}
        )
        self.assertTrue(
            any(
                "unstable" in mechanism
                for mechanism in stability_fail["result"][
                    "failing_mechanisms"
                ]
            )
        )

    def test_v2_digest_deterministic_and_prefix_disjoint_from_v1(
        self,
    ) -> None:
        results = [self._corpus_result("corpus-a")]
        first = build_report_v2(results, {"checkpoint": "v4"})
        second = build_report_v2(results, {"checkpoint": "v4"})
        self.assertEqual(first["content_digest"], second["content_digest"])
        self.assertEqual(
            first["content_digest"],
            content_digest(first, prefix=_DIGEST_PREFIX_V2),
        )
        # The v1 and v2 digest domains can never alias.
        self.assertNotEqual(
            content_digest(first), content_digest(first, prefix=_DIGEST_PREFIX_V2)
        )
        mutated = build_report_v2(results, {"checkpoint": "v3"})
        self.assertNotEqual(
            first["content_digest"], mutated["content_digest"]
        )

    def test_empty_report_fails(self) -> None:
        report = build_report_v2([], {})
        self.assertEqual(report["result"]["gate"], "FAIL")
        self.assertIn(
            "no corpora scored", report["result"]["failing_mechanisms"]
        )


class ScoreCorpusV2EndToEndTests(unittest.TestCase):
    """The v2 scorer on a synthetic probe archive with an in-place arm."""

    def _events_and_frames(self):
        control = _e2e_frame({(5, 5): _PLAYER_VALUE})
        # Root 1: an in-place transformation ("up") and a displacement
        # ("left"), corroborating each other.
        factual_press = _e2e_frame({(5, 5): _OBJECT_VALUE})
        factual_left = _e2e_frame({(4, 5): _PLAYER_VALUE})
        # Root 2: an ambient-change arm ("down", changed cell (9, 9) far
        # from the player), a displacement arm ("right") that the ambient
        # arm fails to corroborate, and a no-change arm ("b") whose
        # endpoint is byte-identical to the control.
        factual_amb = _e2e_frame({(5, 5): _PLAYER_VALUE, (9, 9): 90})
        factual_right = _e2e_frame({(6, 5): _PLAYER_VALUE})

        def branch(state: str, action: str, frame: Frame) -> Dict[str, object]:
            return {
                "event": "human_prior_option_branch_verified",
                "parent_state_id": state,
                "path": [action],
                "durations": [8],
                "frame": frame.digest,
                "frame_width": _E2E_COLUMNS,
                "frame_height": _E2E_ROWS,
            }

        def neutral(state: str) -> Dict[str, object]:
            return {
                "event": "human_prior_option_local_neutral_verified",
                "parent_state_id": state,
                "action": "noop",
                "action_frames": 8,
                "frame": control.digest,
            }

        events = [
            branch("state-00000001", "up", factual_press),
            branch("state-00000001", "left", factual_left),
            neutral("state-00000001"),
            branch("state-00000002", "down", factual_amb),
            branch("state-00000002", "right", factual_right),
            branch("state-00000002", "b", control),
            neutral("state-00000002"),
        ]
        frames = (
            control,
            factual_press,
            factual_left,
            factual_amb,
            factual_right,
        )
        return events, frames

    def test_in_place_detection_and_priced_false_positive(self) -> None:
        events, frames = self._events_and_frames()
        control, factual_press, factual_left, factual_amb, factual_right = (
            frames
        )
        # The learned stub masks the player cell of each frame -- and on
        # the ambient pair it also spills over the animated cell (9, 9)
        # in both endpoints, so its unanchored differential must fire
        # there and be priced by bit (d).
        learned = _StubConvention(
            name=LEARNED_CONVENTION,
            masks={
                control.digest: ((5, 5), frozenset({(5, 5), (9, 9)})),
                factual_press.digest: ((5, 5), frozenset({(5, 5)})),
                factual_left.digest: ((4, 5), frozenset({(4, 5)})),
                factual_amb.digest: ((5, 5), frozenset({(5, 5), (9, 9)})),
                factual_right.digest: ((6, 5), frozenset({(6, 5)})),
            },
        )
        assisted = _StubConvention(
            name=ASSISTED_CONVENTION,
            masks={
                control.digest: ((5, 5), frozenset({(5, 5)})),
                factual_press.digest: ((5, 5), frozenset({(5, 5)})),
                factual_left.digest: ((4, 5), frozenset({(4, 5)})),
                factual_amb.digest: ((5, 5), frozenset({(5, 5)})),
                factual_right.digest: ((6, 5), frozenset({(6, 5)})),
            },
        )
        with tempfile.TemporaryDirectory() as scratch:
            run_dir = Path(scratch) / "corpus"
            frames_dir = run_dir / "frames"
            frames_dir.mkdir(parents=True)
            for frame in frames:
                (frames_dir / f"{frame.digest}.png").write_bytes(
                    encode_png(frame)
                )
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            result = score_corpus_v2(run_dir, learned, assisted)
            again = score_corpus_v2(run_dir, learned, assisted)
        self.assertEqual(result, again)
        self.assertEqual(result["detection_quantity"], DETECTION_QUANTITY_V2)
        # Root 1 scores both arms; root 2's arms are all uncorroborated
        # or unchanged, so they land in the no-effect population.
        self.assertEqual(result["extraction"]["labeled_arms"], 5)
        self.assertEqual(result["extraction"]["scored_arms"], 2)
        self.assertEqual(result["detection"]["measurements"], 2)

        bit_a = result["gate"]["bit_a_detection"]
        # The in-place arm is invisible to the signature channel under
        # the fully covering masks; the differential channel catches it,
        # for the incumbent as well (the symmetric upgrade).
        self.assertEqual(bit_a["learned_rate_vs_ground_truth"], 1.0)
        self.assertEqual(bit_a["assisted_rate_vs_ground_truth"], 1.0)
        for name in (LEARNED_CONVENTION, ASSISTED_CONVENTION):
            self.assertEqual(
                bit_a["channels"][name]["signature_rate_vs_ground_truth"],
                0.5,
            )
            self.assertEqual(
                bit_a["channels"][name]["differential_only_detected"], 1
            )
        press_rows = [
            row
            for row in result["detection"]["rows"]
            if row["factual"] == factual_press.digest
        ]
        self.assertEqual(len(press_rows), 1)
        self.assertFalse(press_rows[0]["learned_signature_detected"])
        self.assertTrue(press_rows[0]["learned_differential_fired"])
        self.assertEqual(
            press_rows[0]["learned_differential_changed_pixels"], 1
        )

        # No-effect population: the ambient pair, the failed-corroboration
        # displacement pair, and the byte-identical pair.
        no_effect = result["no_effect"]
        self.assertEqual(no_effect["measurements"], 3)
        self.assertEqual(no_effect["conflicting_pairs_excluded"], 0)
        bit_d = result["gate"]["bit_d_differential_false_positive"]
        self.assertEqual(
            bit_d["by_class"][NO_EFFECT_CLASS_IDENTICAL]["measurements"], 1
        )
        self.assertEqual(
            bit_d["by_class"][NO_EFFECT_CLASS_UNCORROBORATED][
                "measurements"
            ],
            2,
        )
        # The learned spill over the animated cell fires exactly once --
        # the raw-change pathology, caught and priced as a regression
        # against the tight assisted mask, which stays quiet; the
        # identical pair structurally cannot fire.
        self.assertEqual(bit_d["learned_fired"], 1)
        self.assertEqual(bit_d["assisted_fired"], 0)
        self.assertEqual(
            bit_d["by_class"][NO_EFFECT_CLASS_IDENTICAL]["learned_fired"], 0
        )
        self.assertTrue(bit_d["identical_condition"])
        self.assertTrue(bit_d["uncorroborated_condition"])
        self.assertFalse(bit_d["no_regression_condition"])
        self.assertEqual(no_effect["fired_rows"], [
            {
                "factual": factual_amb.digest,
                "control": control.digest,
                "class": NO_EFFECT_CLASS_UNCORROBORATED,
                "convention": LEARNED_CONVENTION,
                "region_pixels": 2,
                "changed_pixels": 1,
            }
        ])
        # Honest insufficiency: perfect-looking rates over tiny synthetic
        # populations never letter-pass.
        self.assertFalse(bit_a["passed"])
        self.assertFalse(bit_d["passed"])
        self.assertFalse(result["gate"]["passed"])
        report = build_report_v2([result], {"checkpoint": "stub"})
        self.assertEqual(report["result"]["gate"], "FAIL")
        json.dumps(report, sort_keys=True, allow_nan=False)

    def test_convention_names_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            run_dir = Path(scratch) / "corpus"
            (run_dir / "frames").mkdir(parents=True)
            (run_dir / "events.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                score_corpus_v2(
                    run_dir,
                    _StubConvention(name="wrong"),
                    _StubConvention(name=ASSISTED_CONVENTION),
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
