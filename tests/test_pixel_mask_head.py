import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch

from lolo_agent import tracker_substitution_replay
from lolo_agent.controllable_tracker import (
    ControllableRegionTracker,
    _roc_auc,
    arm_examples_from_records,
)
from lolo_agent.counterfactual_labels import (
    STATUS_CENSORED,
    STATUS_LABELED,
    generate_labels,
)
from lolo_agent.environment import Action
from lolo_agent.mask_sensitive_gate import pixel_cell, score_frame
from lolo_agent.pixel_mask_head import (
    ANCHOR_CELL_DILATION,
    ANCHOR_CELL_DILATION_V2,
    ANCHOR_CELL_PROBABILITY_THRESHOLD,
    OCCUPIED_SPLIT_NEIGHBORHOOD_RADIUS,
    PIXEL_MASK_PROBABILITY_THRESHOLD,
    SILHOUETTE_HALO_DILATION,
    TARGET_SEMANTICS_OCCUPIED_V2,
    TARGET_SEMANTICS_UNION_V1,
    PixelMaskExample,
    PixelMaskHead,
    PixelMaskPrediction,
    PixelSilhouettePredictor,
    anchor_pixel_region,
    anchored_cells,
    attach_cell_probabilities,
    cell_pixel_block,
    dilate_pixels,
    label_pixel_root,
    load_pixel_mask_head_checkpoint,
    pixel_difference,
    pixel_examples_from_store,
    pixel_grid_cell,
    pixel_neighborhood_equal,
    pixel_targets_digest,
    reconstruct_silhouette_pixels,
    save_pixel_mask_head_checkpoint,
    split_occupied_vacated,
    train_pixel_mask_head,
    validate_pixel_mask_head,
)
from lolo_agent.pixels import Frame
from lolo_agent.spatial_world_model import SpatialTokenDynamicsModel
from lolo_agent.strict_lineage import (
    audit_checkpoint_metadata,
    lint_strict_lineage,
)
from lolo_agent.tracker_substitution_replay import (
    grid_cell_pixel_block,
    learned_mask_cells,
    learned_pixel_mask,
    learned_reference_slot,
)
from lolo_agent.unlabeled_entities import UnlabeledEntityMemory

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "lolo_agent"

# 32x30 single-channel frames over a 4x3 coarse grid: each cell is an
# 8x10 pixel block, matching the pooled-feature partition exactly.
WIDTH, HEIGHT = 32, 30
COLUMNS, ROWS = 4, 3
BACKGROUND = 10
SPRITE = 200


def _frame(values: Dict[Tuple[int, int], int]) -> Frame:
    pixels = bytearray([BACKGROUND]) * (WIDTH * HEIGHT)
    for (x, y), value in values.items():
        pixels[y * WIDTH + x] = value
    return Frame(WIDTH, HEIGHT, 1, bytes(pixels))


def _block(x0: int, x1: int, y0: int, y1: int, value: int = SPRITE) -> Dict[Tuple[int, int], int]:
    return {(x, y): value for y in range(y0, y1) for x in range(x0, x1)}


def _block_pixels(x0: int, x1: int, y0: int, y1: int) -> frozenset:
    return frozenset((x, y) for y in range(y0, y1) for x in range(x0, x1))


# Sprite fills cell (1, 1); RIGHT moves it to cell (2, 1), UP to (1, 0).
ROOT = _frame(_block(8, 16, 10, 20))
RIGHT_ENDPOINT = _frame(_block(16, 24, 10, 20))
UP_ENDPOINT = _frame(_block(8, 16, 0, 10))
SOURCE_PIXELS = _block_pixels(8, 16, 10, 20)
RIGHT_DESTINATION = _block_pixels(16, 24, 10, 20)
UP_DESTINATION = _block_pixels(8, 16, 0, 10)


@dataclass(frozen=True)
class _StubSequence:
    group: int
    frames: Tuple[Frame, ...]
    actions: Tuple[Action, ...]
    durations: Tuple[int, ...]
    source_run_id: str


def _one_step(
    root: Frame,
    endpoint: Frame,
    action: Action,
    duration: int = 4,
    *,
    group: int = 0,
    run_id: str = "run-a",
) -> _StubSequence:
    return _StubSequence(group, (root, endpoint), (action,), (duration,), run_id)


def _sprite_record(extra: Tuple[_StubSequence, ...] = ()) -> Dict:
    sequences = (
        _one_step(ROOT, RIGHT_ENDPOINT, Action.RIGHT),
        _one_step(ROOT, UP_ENDPOINT, Action.UP),
        _one_step(ROOT, ROOT, Action.NOOP),
    ) + extra
    labels = generate_labels(sequences, columns=COLUMNS, rows=ROWS)
    assert len(labels) == 1
    return labels[0].payload()


def _frames_map(*frames: Frame) -> Dict[str, Frame]:
    return {frame.digest: frame for frame in frames}


@dataclass(frozen=True)
class _StubStore:
    frames: Mapping[str, Frame]

    def load_frame_subset(self, digests) -> Dict[str, Frame]:
        return {digest: self.frames[digest] for digest in set(digests)}


@dataclass(frozen=True)
class _StubCellPrediction:
    columns: int
    rows: int
    probabilities: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class _StubTracker:
    prediction: _StubCellPrediction

    def predict(self, frame: Frame) -> _StubCellPrediction:
        return self.prediction


def _cell_map(cells: Dict[Tuple[int, int], float]) -> Tuple[Tuple[float, ...], ...]:
    return tuple(
        tuple(float(cells.get((column, row), 0.0)) for column in range(COLUMNS))
        for row in range(ROWS)
    )


def _make_head(seed: int = 0, hidden_size: int = 8) -> PixelMaskHead:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return PixelMaskHead(hidden_size=hidden_size)


def _make_tracker(seed: int = 0) -> ControllableRegionTracker:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        backbone = SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=4,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        return ControllableRegionTracker(
            backbone, hidden_size=16, ensemble_size=2, columns=COLUMNS, rows=ROWS
        )


def _make_example(
    frame: Frame,
    target: Tuple[Tuple[int, int], ...],
    residual: Tuple[Tuple[int, int], ...] = (),
    *,
    run_id: str = "run-a",
    group: int = 0,
    cell_probabilities=None,
) -> PixelMaskExample:
    return PixelMaskExample(
        source_run_id=run_id,
        group=group,
        root_digest="root",
        action="right",
        duration=4,
        endpoint_digest=frame.digest,
        width=WIDTH,
        height=HEIGHT,
        columns=COLUMNS,
        rows=ROWS,
        target_pixels=target,
        residual_pixels=residual,
        frame=frame,
        cell_probabilities=cell_probabilities,
    )


class PixelDifferenceTests(unittest.TestCase):
    def test_exact_difference_and_identity(self) -> None:
        self.assertEqual(pixel_difference(ROOT, ROOT), frozenset())
        self.assertEqual(
            pixel_difference(RIGHT_ENDPOINT, ROOT),
            SOURCE_PIXELS | RIGHT_DESTINATION,
        )

    def test_mismatched_geometry_is_refused(self) -> None:
        other = Frame(WIDTH, HEIGHT - 10, 1, bytes(WIDTH * (HEIGHT - 10)))
        with self.assertRaises(ValueError):
            pixel_difference(ROOT, other)


class GridPartitionTests(unittest.TestCase):
    def test_helpers_match_the_gate_and_replay_partition(self) -> None:
        # Awkward non-divisible dimensions: the local helpers must agree
        # exactly with the unchanged gate/replay grid conventions.
        width, height, columns, rows = 10, 9, 4, 3
        for row in range(rows):
            for column in range(columns):
                block = cell_pixel_block((column, row), width, height, columns, rows)
                self.assertEqual(
                    block,
                    grid_cell_pixel_block((column, row), width, height, columns, rows),
                )
                for x, y in block:
                    self.assertEqual(
                        pixel_grid_cell(x, y, width, height, columns, rows),
                        (column, row),
                    )
                    self.assertEqual(
                        pixel_grid_cell(x, y, width, height, columns, rows),
                        pixel_cell(x, y, width, height, columns, rows),
                    )


class ReconstructionConventionTests(unittest.TestCase):
    def test_preregistered_constants_pinned(self) -> None:
        # The pixel threshold is the same pinned 0.5 operating point every
        # WP5 instrument uses; the anchor dilation and halo radius are the
        # spike's own preregistration, fixed before any gate execution.
        self.assertEqual(PIXEL_MASK_PROBABILITY_THRESHOLD, 0.5)
        self.assertEqual(
            PIXEL_MASK_PROBABILITY_THRESHOLD,
            tracker_substitution_replay.LEARNED_MASK_PROBABILITY_THRESHOLD,
        )
        self.assertEqual(ANCHOR_CELL_PROBABILITY_THRESHOLD, 0.5)
        self.assertEqual(ANCHOR_CELL_DILATION, 1)
        self.assertEqual(SILHOUETTE_HALO_DILATION, 3)


class PixelLabelTests(unittest.TestCase):
    def test_moved_sprite_yields_corroborated_silhouette(self) -> None:
        record = _sprite_record()
        frames = _frames_map(ROOT, RIGHT_ENDPOINT, UP_ENDPOINT)
        labels = label_pixel_root(record, frames)
        by_action = {label.action: label for label in labels}
        right = by_action["right"]
        self.assertEqual(right.status, STATUS_LABELED)
        self.assertEqual(right.corroborating_arms, 1)
        self.assertEqual(
            set(right.changed_pixels), SOURCE_PIXELS | RIGHT_DESTINATION
        )
        # The vacated+occupied silhouette is one 4-connected component and
        # intersects the leave-one-action-out intersection (the source
        # block shared with the UP arm), so all of it is controllable.
        self.assertEqual(len(right.controllable_components), 1)
        self.assertEqual(
            set(right.controllable_pixels), SOURCE_PIXELS | RIGHT_DESTINATION
        )
        self.assertEqual(right.residual_pixels, ())
        up = by_action["up"]
        self.assertEqual(set(up.controllable_pixels), SOURCE_PIXELS | UP_DESTINATION)

    def test_uncorroborated_change_is_residual(self) -> None:
        # RIGHT additionally flips an isolated blob no sibling action
        # touches: it survives as a changed component but fails
        # leave-one-action-out corroboration and lands in the residual.
        blob = _block(26, 30, 22, 28)
        right_with_blob = _frame({**_block(16, 24, 10, 20), **blob})
        record = _sprite_record()
        # Rebuild the record with the blobbed endpoint.
        sequences = (
            _one_step(ROOT, right_with_blob, Action.RIGHT),
            _one_step(ROOT, UP_ENDPOINT, Action.UP),
            _one_step(ROOT, ROOT, Action.NOOP),
        )
        record = generate_labels(sequences, columns=COLUMNS, rows=ROWS)[0].payload()
        frames = _frames_map(ROOT, right_with_blob, UP_ENDPOINT)
        labels = label_pixel_root(record, frames)
        right = {label.action: label for label in labels}["right"]
        self.assertEqual(right.status, STATUS_LABELED)
        self.assertEqual(
            set(right.controllable_pixels), SOURCE_PIXELS | RIGHT_DESTINATION
        )
        self.assertEqual(set(right.residual_pixels), _block_pixels(26, 30, 22, 28))

    def test_pixel_corroboration_is_stricter_than_cell(self) -> None:
        # Two sibling actions change DIFFERENT pixels of the SAME coarse
        # cell: the cell path corroborates the cell, but the pixel-level
        # intersection is empty, so no pixel component survives.
        right_endpoint = _frame({(0, 0): SPRITE})
        up_endpoint = _frame({(1, 1): SPRITE})
        sequences = (
            _one_step(ROOT2 := _frame({}), right_endpoint, Action.RIGHT),
            _one_step(ROOT2, up_endpoint, Action.UP),
            _one_step(ROOT2, ROOT2, Action.NOOP),
        )
        record = generate_labels(sequences, columns=COLUMNS, rows=ROWS)[0].payload()
        right_record = [arm for arm in record["arms"] if arm["action"] == "right"][0]
        self.assertEqual(right_record["status"], STATUS_LABELED)
        self.assertEqual(right_record["controllable_cells"], [[0, 0]])
        frames = _frames_map(ROOT2, right_endpoint, up_endpoint)
        labels = label_pixel_root(record, frames)
        right = {label.action: label for label in labels}["right"]
        self.assertEqual(right.status, STATUS_LABELED)
        self.assertEqual(right.controllable_pixels, ())
        self.assertEqual(set(right.residual_pixels), {(0, 0)})

    def test_censored_arms_inherit_reasons_and_emit_no_pixels(self) -> None:
        # UP has no duration-matched control (absent_control); RIGHT then
        # has no corroborating sibling (no_sibling_corroboration).  The
        # pixel path copies both censor statuses unchanged.
        sequences = (
            _one_step(ROOT, RIGHT_ENDPOINT, Action.RIGHT, 4),
            _one_step(ROOT, UP_ENDPOINT, Action.UP, 2),
            _one_step(ROOT, ROOT, Action.NOOP, 4),
        )
        record = generate_labels(sequences, columns=COLUMNS, rows=ROWS)[0].payload()
        frames = _frames_map(ROOT, RIGHT_ENDPOINT, UP_ENDPOINT)
        labels = label_pixel_root(record, frames)
        by_action = {label.action: label for label in labels}
        self.assertEqual(by_action["right"].status, STATUS_CENSORED)
        self.assertEqual(
            by_action["right"].censor_reason, "no_sibling_corroboration"
        )
        self.assertEqual(by_action["up"].status, STATUS_CENSORED)
        self.assertEqual(by_action["up"].censor_reason, "absent_control")
        for label in labels:
            self.assertEqual(label.changed_pixels, ())
            self.assertEqual(label.controllable_pixels, ())

    def test_record_cross_checks_fail_loudly(self) -> None:
        record = _sprite_record()
        frames = _frames_map(ROOT, RIGHT_ENDPOINT, UP_ENDPOINT)
        tampered = {**record, "arms": [dict(arm) for arm in record["arms"]]}
        tampered["arms"][0]["changed_cells"] = [[0, 0]]
        with self.assertRaises(ValueError):
            label_pixel_root(tampered, frames)
        tampered = {**record, "arms": [dict(arm) for arm in record["arms"]]}
        tampered["arms"][0]["corroborating_arms"] = 7
        with self.assertRaises(ValueError):
            label_pixel_root(tampered, frames)


class ExampleBuilderTests(unittest.TestCase):
    def test_examples_built_for_selected_arms_with_frames(self) -> None:
        record = _sprite_record()
        frames = _frames_map(ROOT, RIGHT_ENDPOINT, UP_ENDPOINT)
        selected, _statistics = arm_examples_from_records([record])
        examples, statistics = pixel_examples_from_store(
            _StubStore(frames), [record], selected
        )
        self.assertEqual(statistics["selected_arms"], 2)
        self.assertEqual(statistics["examples"], 2)
        self.assertEqual(statistics["empty_pixel_mask_arms"], 0)
        by_action = {example.action: example for example in examples}
        right = by_action["right"]
        self.assertEqual(set(right.target_pixels), SOURCE_PIXELS | RIGHT_DESTINATION)
        self.assertEqual(right.residual_pixels, ())
        self.assertIsNotNone(right.frame)
        self.assertEqual(right.frame.digest, RIGHT_ENDPOINT.digest)
        self.assertEqual((right.width, right.height), (WIDTH, HEIGHT))

    def test_empty_pixel_silhouettes_are_excluded_and_counted(self) -> None:
        root = _frame({})
        right_endpoint = _frame({(0, 0): SPRITE})
        up_endpoint = _frame({(1, 1): SPRITE})
        sequences = (
            _one_step(root, right_endpoint, Action.RIGHT),
            _one_step(root, up_endpoint, Action.UP),
            _one_step(root, root, Action.NOOP),
        )
        record = generate_labels(sequences, columns=COLUMNS, rows=ROWS)[0].payload()
        frames = _frames_map(root, right_endpoint, up_endpoint)
        selected, _statistics = arm_examples_from_records([record])
        self.assertEqual(len(selected), 2)  # cell path labels both arms
        examples, statistics = pixel_examples_from_store(
            _StubStore(frames), [record], selected
        )
        self.assertEqual(examples, [])
        self.assertEqual(statistics["empty_pixel_mask_arms"], 2)

    def test_pixel_targets_digest_deterministic_and_sensitive(self) -> None:
        first = _make_example(RIGHT_ENDPOINT, tuple(sorted(RIGHT_DESTINATION)))
        second = _make_example(UP_ENDPOINT, tuple(sorted(UP_DESTINATION)), group=1)
        digest = pixel_targets_digest([first, second])
        self.assertEqual(digest, pixel_targets_digest([second, first]))
        mutated = _make_example(
            RIGHT_ENDPOINT, tuple(sorted(RIGHT_DESTINATION - {(16, 10)}))
        )
        self.assertNotEqual(digest, pixel_targets_digest([mutated, second]))


class HeadForwardTests(unittest.TestCase):
    def test_forward_shape_and_determinism(self) -> None:
        head = _make_head()
        frames = torch.rand((2, 3, HEIGHT, WIDTH))
        cell_maps = torch.rand((2, ROWS, COLUMNS))
        logits = head(frames, cell_maps)
        self.assertEqual(tuple(logits.shape), (2, HEIGHT, WIDTH))
        again = head(frames, cell_maps)
        self.assertTrue(torch.equal(logits, again))

    def test_forward_rejects_bad_shapes(self) -> None:
        head = _make_head()
        with self.assertRaises(ValueError):
            head(torch.rand((3, HEIGHT, WIDTH)), torch.rand((1, ROWS, COLUMNS)))
        with self.assertRaises(ValueError):
            head(torch.rand((1, 1, HEIGHT, WIDTH)), torch.rand((1, ROWS, COLUMNS)))
        with self.assertRaises(ValueError):
            head(torch.rand((1, 3, HEIGHT, WIDTH)), torch.rand((ROWS, COLUMNS)))
        with self.assertRaises(ValueError):
            head(torch.rand((2, 3, HEIGHT, WIDTH)), torch.rand((1, ROWS, COLUMNS)))

    def test_frozen_inference_does_not_change_parameters(self) -> None:
        head = _make_head()
        head.freeze()
        digest = head.checkpoint_digest
        head(torch.rand((1, 3, HEIGHT, WIDTH)), torch.rand((1, ROWS, COLUMNS)))
        self.assertEqual(head.checkpoint_digest, digest)
        self.assertFalse(
            any(parameter.requires_grad for parameter in head.parameters())
        )


class TrainingTests(unittest.TestCase):
    def _sprite_training_examples(self):
        examples = []
        cells = ((0, 0), (1, 1), (2, 2), (3, 0), (0, 2), (2, 1), (1, 0), (3, 2))
        for index, (column, row) in enumerate(cells):
            x0, x1 = column * 8, column * 8 + 8
            y0, y1 = row * 10, row * 10 + 10
            frame = _frame(_block(x0, x1, y0, y1))
            examples.append(
                _make_example(
                    frame,
                    tuple(sorted(_block_pixels(x0, x1, y0, y1))),
                    run_id="run-a" if index % 2 == 0 else "run-b",
                    group=index,
                    cell_probabilities=_cell_map({(column, row): 1.0}),
                )
            )
        return examples

    def test_training_beats_the_untrained_baseline_on_synthetic_masks(self) -> None:
        examples = self._sprite_training_examples()
        baseline = _make_head(seed=1)
        before = validate_pixel_mask_head(baseline, examples, "cpu", batch_size=4)
        head = _make_head(seed=1)
        history = train_pixel_mask_head(
            head,
            examples,
            "cpu",
            epochs=8,
            batch_size=4,
            learning_rate=1e-2,
            seed=3,
        )
        self.assertTrue(history)
        after = validate_pixel_mask_head(head, examples, "cpu", batch_size=4)
        self.assertLess(after.loss, before.loss)
        self.assertGreater(after.roc_auc, before.roc_auc)
        self.assertGreater(
            after.mean_target_probability, after.mean_background_probability
        )

    def test_attach_cell_probabilities_keeps_tracker_frozen(self) -> None:
        tracker = _make_tracker()
        tracker.freeze()
        tracker_digest = tracker.checkpoint_digest
        backbone_digest = tracker.backbone.checkpoint_digest
        examples = [
            _make_example(RIGHT_ENDPOINT, tuple(sorted(RIGHT_DESTINATION)))
        ]
        attached = attach_cell_probabilities(tracker, examples, "cpu")
        self.assertEqual(tracker.checkpoint_digest, tracker_digest)
        self.assertEqual(tracker.backbone.checkpoint_digest, backbone_digest)
        self.assertEqual(len(attached), 1)
        cell_map = attached[0].cell_probabilities
        self.assertEqual(len(cell_map), ROWS)
        for row in cell_map:
            self.assertEqual(len(row), COLUMNS)
            for value in row:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_batches_require_frames_and_cell_maps(self) -> None:
        head = _make_head()
        missing_map = [_make_example(ROOT, ((8, 10),))]
        with self.assertRaises(ValueError):
            train_pixel_mask_head(head, missing_map, "cpu", epochs=1)


class ValidationTests(unittest.TestCase):
    def test_roc_auc_matches_the_cell_trainer_reference(self) -> None:
        generator = torch.Generator().manual_seed(11)
        probabilities = torch.rand(200, generator=generator)
        # Force ties so the tie-handling path is exercised.
        probabilities = (probabilities * 20).round() / 20
        labels = (torch.rand(200, generator=generator) > 0.7).to(torch.int64)
        from lolo_agent.pixel_mask_head import _roc_auc_from_tensors

        fast = _roc_auc_from_tensors(probabilities, labels.bool())
        reference = _roc_auc(probabilities.tolist(), labels.tolist())
        self.assertAlmostEqual(fast, reference, places=12)


class CheckpointTests(unittest.TestCase):
    _PINS = {
        "label_manifest_digest": "a" * 64,
        "pixel_targets_sha256": "b" * 64,
        "tracker_parameter_digest": "c" * 64,
        "backbone_parameter_digest": "d" * 64,
    }

    def test_save_load_roundtrip_and_digest_stability(self) -> None:
        head = _make_head()
        frames = torch.rand((1, 3, HEIGHT, WIDTH))
        cell_maps = torch.rand((1, ROWS, COLUMNS))
        expected = head(frames, cell_maps)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            saved = save_pixel_mask_head_checkpoint(
                head, path, cell_columns=COLUMNS, cell_rows=ROWS, **self._PINS
            )
            self.assertEqual(saved, head.checkpoint_digest)
            loaded, provenance = load_pixel_mask_head_checkpoint(path)
        self.assertEqual(loaded.checkpoint_digest, saved)
        self.assertTrue(torch.equal(loaded(frames, cell_maps), expected))
        self.assertFalse(
            any(parameter.requires_grad for parameter in loaded.parameters())
        )
        self.assertEqual(provenance["label_manifest_sha256"], "a" * 64)
        self.assertEqual(provenance["pixel_targets_sha256"], "b" * 64)
        self.assertEqual(provenance["tracker_parameter_sha256"], "c" * 64)
        self.assertEqual(provenance["backbone_parameter_sha256"], "d" * 64)
        self.assertEqual(provenance["cell_columns"], COLUMNS)
        self.assertEqual(provenance["cell_rows"], ROWS)

    def test_missing_provenance_digests_are_refused(self) -> None:
        head = _make_head()
        with tempfile.TemporaryDirectory() as directory:
            for missing in self._PINS:
                pins = {**self._PINS, missing: ""}
                with self.assertRaises(ValueError):
                    save_pixel_mask_head_checkpoint(
                        head,
                        Path(directory) / "head.pt",
                        cell_columns=COLUMNS,
                        cell_rows=ROWS,
                        **pins,
                    )

    def test_checkpoint_provenance_passes_the_strict_audit(self) -> None:
        head = _make_head()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            save_pixel_mask_head_checkpoint(
                head, path, cell_columns=COLUMNS, cell_rows=ROWS, **self._PINS
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["architecture"], "wp5-pixel-mask-head")
        self.assertEqual(payload["reward_track"], "strict")
        audit = audit_checkpoint_metadata(payload, label="pixel-mask-head")
        self.assertEqual(audit.violations, ())
        self.assertFalse(audit.assisted)
        self.assertEqual(
            audit.declared_fields,
            ("reward_track", "persistent_inputs", "excluded_inputs"),
        )


class LineageTests(unittest.TestCase):
    def test_head_module_lints_clean(self) -> None:
        report = lint_strict_lineage((_PACKAGE_ROOT / "pixel_mask_head.py",))
        verdicts = {entry.module: entry.assisted for entry in report.modules}
        self.assertEqual(verdicts, {"lolo_agent.pixel_mask_head": False})
        self.assertFalse(report.assisted)

    def test_train_module_couples_only_through_evaluation_imports(self) -> None:
        # The gate subcommand imports the (assisted-coupled) evaluation
        # instrument, so the entry module is assisted-coupled -- but no
        # assisted symbol may be referenced directly in either new module.
        report = lint_strict_lineage((_PACKAGE_ROOT / "pixel_mask_train.py",))
        entry = report.modules[0]
        self.assertEqual(entry.module, "lolo_agent.pixel_mask_train")
        self.assertTrue(entry.assisted)
        for finding in entry.findings:
            if finding.reference.decisive:
                self.assertGreater(
                    len(finding.chain),
                    1,
                    "assisted reference directly inside the new module: "
                    f"{finding}",
                )
                self.assertNotIn(
                    finding.chain[-1],
                    (
                        "lolo_agent.pixel_mask_train",
                        "lolo_agent.pixel_mask_head",
                    ),
                )


class ReconstructionTests(unittest.TestCase):
    def test_anchor_region_dilates_and_clips(self) -> None:
        cell_map = _cell_map({(0, 0): 0.9})
        self.assertEqual(
            anchored_cells(cell_map, COLUMNS, ROWS), ((0, 0),)
        )
        region = anchor_pixel_region(cell_map, COLUMNS, ROWS, WIDTH, HEIGHT)
        expected = frozenset()
        for cell in ((0, 0), (1, 0), (0, 1), (1, 1)):
            expected |= cell_pixel_block(cell, WIDTH, HEIGHT, COLUMNS, ROWS)
        self.assertEqual(region, expected)
        below = _cell_map({(0, 0): 0.49})
        self.assertEqual(
            anchor_pixel_region(below, COLUMNS, ROWS, WIDTH, HEIGHT), frozenset()
        )

    def test_reconstruction_thresholds_anchors_and_dilates(self) -> None:
        probabilities = [[0.0] * WIDTH for _ in range(HEIGHT)]
        probabilities[15][12] = 1.0  # inside anchor
        probabilities[25][28] = 1.0  # outside anchor: must be ignored
        anchor = cell_pixel_block((1, 1), WIDTH, HEIGHT, COLUMNS, ROWS)
        mask = reconstruct_silhouette_pixels(probabilities, anchor, WIDTH, HEIGHT)
        self.assertEqual(mask, dilate_pixels({(12, 15)}, WIDTH, HEIGHT, 3))
        self.assertEqual(len(mask), 49)  # full 7x7 Chebyshev halo
        self.assertNotIn((28, 25), mask)
        empty = reconstruct_silhouette_pixels(
            [[0.0] * WIDTH for _ in range(HEIGHT)], anchor, WIDTH, HEIGHT
        )
        self.assertEqual(empty, frozenset())

    def test_halo_clips_at_frame_borders(self) -> None:
        mask = dilate_pixels({(0, 0)}, WIDTH, HEIGHT, 3)
        self.assertEqual(mask, _block_pixels(0, 4, 0, 4))

    def test_pixel_prediction_round_trips_through_replay_helpers(self) -> None:
        # THE substitution mechanism: a pixel-resolution prediction makes
        # the UNCHANGED replay helpers recover exactly the reconstructed
        # mask at the pinned 0.5 threshold.
        mask = frozenset({(8, 10), (9, 10), (8, 11), (30, 29)})
        indicator = [[0.0] * WIDTH for _ in range(HEIGHT)]
        for x, y in mask:
            indicator[y][x] = 1.0
        prediction = PixelMaskPrediction(
            columns=WIDTH,
            rows=HEIGHT,
            probabilities=tuple(tuple(row) for row in indicator),
            mask=mask,
        )
        self.assertEqual(
            learned_pixel_mask(prediction, WIDTH, HEIGHT, 0.5), mask
        )
        self.assertEqual(len(learned_mask_cells(prediction, 0.5)), len(mask))
        slot = learned_reference_slot(prediction, WIDTH, HEIGHT, 0.5)
        self.assertIn(slot, mask)

    def test_unchanged_gate_scores_pixel_predictions(self) -> None:
        # End to end through the unchanged instrument: when the
        # reconstructed pixel mask equals the assisted mask on a mattering
        # frame the gate agrees; an empty reconstruction disagrees.
        memory = UnlabeledEntityMemory(columns=COLUMNS, rows=ROWS)
        sprite_pixels = frozenset(SOURCE_PIXELS)
        slot = (8, 10)
        indicator = [[0.0] * WIDTH for _ in range(HEIGHT)]
        for x, y in sprite_pixels:
            indicator[y][x] = 1.0
        matching = PixelMaskPrediction(
            columns=WIDTH,
            rows=HEIGHT,
            probabilities=tuple(tuple(row) for row in indicator),
            mask=sprite_pixels,
        )
        row = score_frame(ROOT, matching, memory, slot, sprite_pixels)
        self.assertTrue(row["mattering"])
        self.assertTrue(row["signature_equal"])
        self.assertTrue(row["l1_within"])
        self.assertTrue(row["agrees"])
        self.assertEqual(row["mask_divergence"]["iou"], 1.0)
        empty = PixelMaskPrediction(
            columns=WIDTH,
            rows=HEIGHT,
            probabilities=tuple(
                tuple(0.0 for _ in range(WIDTH)) for _ in range(HEIGHT)
            ),
            mask=frozenset(),
        )
        row = score_frame(ROOT, empty, memory, slot, sprite_pixels)
        self.assertTrue(row["mattering"])
        self.assertFalse(row["agrees"])

    def test_predictor_is_deterministic_and_anchor_bounded(self) -> None:
        tracker = _StubTracker(
            _StubCellPrediction(COLUMNS, ROWS, _cell_map({(1, 1): 0.9}))
        )
        predictor = PixelSilhouettePredictor(tracker, _make_head(), device="cpu")
        first = predictor.predict(ROOT)
        second = predictor.predict(ROOT)
        self.assertEqual(first, second)
        self.assertEqual((first.columns, first.rows), (WIDTH, HEIGHT))
        anchor = anchor_pixel_region(
            _cell_map({(1, 1): 0.9}), COLUMNS, ROWS, WIDTH, HEIGHT
        )
        allowed = dilate_pixels(anchor, WIDTH, HEIGHT, SILHOUETTE_HALO_DILATION)
        self.assertTrue(first.mask <= allowed)
        values = {value for row in first.probabilities for value in row}
        self.assertTrue(values <= {0.0, 1.0})

    def test_predictor_empty_tracker_map_yields_empty_mask(self) -> None:
        tracker = _StubTracker(_StubCellPrediction(COLUMNS, ROWS, _cell_map({})))
        predictor = PixelSilhouettePredictor(tracker, _make_head(), device="cpu")
        prediction = predictor.predict(ROOT)
        self.assertEqual(prediction.mask, frozenset())
        self.assertEqual(
            {value for row in prediction.probabilities for value in row}, {0.0}
        )


# ---------------------------------------------------------------------------
# Occupied/vacated disambiguation (label semantics v2, learnings 4.37)
# ---------------------------------------------------------------------------

# Exact expected split for the standard moved-sprite fixture.  The RIGHT
# and UP factual endpoints differ exactly on RIGHT_DESTINATION and
# UP_DESTINATION, so a source pixel is vacated unless its 3x3 window
# touches either destination block: the seam column x=15 (adjacent to the
# RIGHT destination) and the seam row y=10 (adjacent to the UP
# destination) stay occupied -- the rule's documented conservative
# one-pixel seam.
VACATED_INTERIOR = _block_pixels(8, 15, 11, 20)
RIGHT_SEAM = frozenset({(15, y) for y in range(10, 20)}) | frozenset(
    {(x, 10) for x in range(8, 15)}
)
# Transformation in place: the sprite recolors without moving.
TRANSFORM_ENDPOINT = _frame(_block(8, 16, 10, 20, 210))
# Removal: the sprite disappears; the sibling moves with a two-pixel gap
# (y0-7) so no 3x3 window of the source block touches the sibling sprite.
EMPTY_ENDPOINT = _frame({})
UP_GAP_ENDPOINT = _frame(_block(8, 16, 0, 8))


class NeighborhoodEqualityTests(unittest.TestCase):
    def test_neighborhood_equality_and_clipping(self) -> None:
        self.assertTrue(pixel_neighborhood_equal(ROOT, ROOT, 0, 0))
        # (16, 15) is inside the RIGHT destination: the frames differ at
        # the center, so every radius fails.
        self.assertFalse(
            pixel_neighborhood_equal(RIGHT_ENDPOINT, UP_ENDPOINT, 16, 15)
        )
        # (5, 5) is background in both frames, but radius 3 reaches the
        # UP destination at x=8.
        self.assertTrue(
            pixel_neighborhood_equal(RIGHT_ENDPOINT, UP_ENDPOINT, 5, 5, 1)
        )
        self.assertFalse(
            pixel_neighborhood_equal(RIGHT_ENDPOINT, UP_ENDPOINT, 5, 5, 3)
        )
        # Corner pixels clip instead of raising.
        self.assertTrue(
            pixel_neighborhood_equal(ROOT, ROOT, 0, HEIGHT - 1, 2)
        )

    def test_mismatched_geometry_and_negative_radius_refused(self) -> None:
        other = Frame(WIDTH, HEIGHT - 10, 1, bytes(WIDTH * (HEIGHT - 10)))
        with self.assertRaises(ValueError):
            pixel_neighborhood_equal(ROOT, other, 0, 0)
        with self.assertRaises(ValueError):
            pixel_neighborhood_equal(ROOT, ROOT, 0, 0, -1)


class OccupiedVacatedSplitTests(unittest.TestCase):
    def test_preregistered_v2_constants_pinned(self) -> None:
        self.assertEqual(OCCUPIED_SPLIT_NEIGHBORHOOD_RADIUS, 1)
        self.assertEqual(ANCHOR_CELL_DILATION_V2, 0)
        self.assertEqual(TARGET_SEMANTICS_UNION_V1, "union-v1")
        self.assertEqual(TARGET_SEMANTICS_OCCUPIED_V2, "occupied-v2")

    def test_moved_sprite_splits_into_destination_and_revealed_origin(
        self,
    ) -> None:
        record = _sprite_record()
        frames = _frames_map(ROOT, RIGHT_ENDPOINT, UP_ENDPOINT)
        labels = label_pixel_root(record, frames, occupied_split=True)
        right = {label.action: label for label in labels}["right"]
        self.assertEqual(
            set(right.occupied_pixels), RIGHT_DESTINATION | RIGHT_SEAM
        )
        self.assertEqual(set(right.vacated_pixels), VACATED_INTERIOR)
        # The split partitions the controllable silhouette exactly.
        self.assertEqual(
            set(right.occupied_pixels) | set(right.vacated_pixels),
            set(right.controllable_pixels),
        )
        self.assertEqual(
            set(right.occupied_pixels) & set(right.vacated_pixels), set()
        )
        # Destination pixels have no sibling evidence and default to
        # occupied -- the destination-region fallback.
        self.assertTrue(RIGHT_DESTINATION <= set(right.occupied_pixels))

    def test_split_function_direct_and_deterministic(self) -> None:
        changed_up = pixel_difference(UP_ENDPOINT, ROOT)
        occupied, vacated = split_occupied_vacated(
            SOURCE_PIXELS | RIGHT_DESTINATION,
            RIGHT_ENDPOINT,
            ((changed_up, UP_ENDPOINT),),
        )
        self.assertEqual(occupied, RIGHT_DESTINATION | RIGHT_SEAM)
        self.assertEqual(vacated, VACATED_INTERIOR)
        again = split_occupied_vacated(
            SOURCE_PIXELS | RIGHT_DESTINATION,
            RIGHT_ENDPOINT,
            ((changed_up, UP_ENDPOINT),),
        )
        self.assertEqual((occupied, vacated), again)

    def test_transformation_in_place_is_fully_occupied(self) -> None:
        sequences = (
            _one_step(ROOT, TRANSFORM_ENDPOINT, Action.RIGHT),
            _one_step(ROOT, UP_ENDPOINT, Action.UP),
            _one_step(ROOT, ROOT, Action.NOOP),
        )
        record = generate_labels(sequences, columns=COLUMNS, rows=ROWS)[0].payload()
        frames = _frames_map(ROOT, TRANSFORM_ENDPOINT, UP_ENDPOINT)
        labels = label_pixel_root(record, frames, occupied_split=True)
        right = {label.action: label for label in labels}["right"]
        self.assertEqual(set(right.controllable_pixels), SOURCE_PIXELS)
        self.assertEqual(set(right.occupied_pixels), SOURCE_PIXELS)
        self.assertEqual(right.vacated_pixels, ())

    def test_removal_has_no_occupied_pixels_at_the_source(self) -> None:
        sequences = (
            _one_step(ROOT, EMPTY_ENDPOINT, Action.RIGHT),
            _one_step(ROOT, UP_GAP_ENDPOINT, Action.UP),
            _one_step(ROOT, ROOT, Action.NOOP),
        )
        record = generate_labels(sequences, columns=COLUMNS, rows=ROWS)[0].payload()
        frames = _frames_map(ROOT, EMPTY_ENDPOINT, UP_GAP_ENDPOINT)
        labels = label_pixel_root(record, frames, occupied_split=True)
        right = {label.action: label for label in labels}["right"]
        self.assertEqual(set(right.controllable_pixels), SOURCE_PIXELS)
        self.assertEqual(right.occupied_pixels, ())
        self.assertEqual(set(right.vacated_pixels), SOURCE_PIXELS)

    def test_split_disabled_by_default_and_deterministic(self) -> None:
        record = _sprite_record()
        frames = _frames_map(ROOT, RIGHT_ENDPOINT, UP_ENDPOINT)
        plain = label_pixel_root(record, frames)
        for label in plain:
            self.assertEqual(label.occupied_pixels, ())
            self.assertEqual(label.vacated_pixels, ())
        first = label_pixel_root(record, frames, occupied_split=True)
        second = label_pixel_root(record, frames, occupied_split=True)
        self.assertEqual(first, second)
        for label in first:
            self.assertEqual(
                label.occupied_pixels, tuple(sorted(label.occupied_pixels))
            )
            self.assertEqual(
                label.vacated_pixels, tuple(sorted(label.vacated_pixels))
            )


class OccupiedTargetExampleTests(unittest.TestCase):
    def test_v2_examples_carry_occupied_targets_and_vacated_negatives(
        self,
    ) -> None:
        record = _sprite_record()
        frames = _frames_map(ROOT, RIGHT_ENDPOINT, UP_ENDPOINT)
        selected, _statistics = arm_examples_from_records([record])
        examples, statistics = pixel_examples_from_store(
            _StubStore(frames),
            [record],
            selected,
            target_semantics=TARGET_SEMANTICS_OCCUPIED_V2,
        )
        self.assertEqual(statistics["target_semantics"], "occupied-v2")
        self.assertEqual(statistics["examples"], 2)
        self.assertEqual(statistics["empty_occupied_arms"], 0)
        right = {example.action: example for example in examples}["right"]
        self.assertEqual(
            set(right.target_pixels), RIGHT_DESTINATION | RIGHT_SEAM
        )
        self.assertEqual(set(right.vacated_pixels), VACATED_INTERIOR)
        # v1 semantics remain byte-identical to the original spike.
        v1_examples, v1_statistics = pixel_examples_from_store(
            _StubStore(frames), [record], selected
        )
        v1_right = {example.action: example for example in v1_examples}["right"]
        self.assertEqual(
            set(v1_right.target_pixels), SOURCE_PIXELS | RIGHT_DESTINATION
        )
        self.assertEqual(v1_right.vacated_pixels, ())
        self.assertEqual(v1_statistics["empty_occupied_arms"], 0)

    def test_empty_occupied_arms_are_excluded_and_counted(self) -> None:
        sequences = (
            _one_step(ROOT, EMPTY_ENDPOINT, Action.RIGHT),
            _one_step(ROOT, UP_GAP_ENDPOINT, Action.UP),
            _one_step(ROOT, ROOT, Action.NOOP),
        )
        record = generate_labels(sequences, columns=COLUMNS, rows=ROWS)[0].payload()
        frames = _frames_map(ROOT, EMPTY_ENDPOINT, UP_GAP_ENDPOINT)
        selected, _statistics = arm_examples_from_records([record])
        examples, statistics = pixel_examples_from_store(
            _StubStore(frames),
            [record],
            selected,
            target_semantics=TARGET_SEMANTICS_OCCUPIED_V2,
        )
        # The removal arm is all-vacated; the UP arm's destination fell
        # into the uncorroborated residual (the disclosed
        # corroboration-granularity limit), leaving an origin-only,
        # all-vacated component.  Both arms are excluded and counted.
        self.assertEqual(examples, [])
        self.assertEqual(statistics["empty_occupied_arms"], 2)
        # The same arm IS a v1 example: the union silhouette is non-empty.
        v1_examples, _v1_statistics = pixel_examples_from_store(
            _StubStore(frames), [record], selected
        )
        self.assertIn(
            "right", {example.action for example in v1_examples}
        )

    def test_unknown_target_semantics_refused(self) -> None:
        record = _sprite_record()
        frames = _frames_map(ROOT, RIGHT_ENDPOINT, UP_ENDPOINT)
        selected, _statistics = arm_examples_from_records([record])
        with self.assertRaises(ValueError):
            pixel_examples_from_store(
                _StubStore(frames),
                [record],
                selected,
                target_semantics="occupied-v3",
            )
        with self.assertRaises(ValueError):
            pixel_targets_digest([], target_semantics="occupied-v3")

    def test_v2_targets_digest_pins_vacated_and_never_aliases_v1(self) -> None:
        occupied = tuple(sorted(RIGHT_DESTINATION))
        vacated = tuple(sorted(VACATED_INTERIOR))
        example = PixelMaskExample(
            source_run_id="run-a",
            group=0,
            root_digest="root",
            action="right",
            duration=4,
            endpoint_digest=RIGHT_ENDPOINT.digest,
            width=WIDTH,
            height=HEIGHT,
            columns=COLUMNS,
            rows=ROWS,
            target_pixels=occupied,
            residual_pixels=(),
            frame=RIGHT_ENDPOINT,
            vacated_pixels=vacated,
        )
        v1 = pixel_targets_digest([example])
        v2 = pixel_targets_digest(
            [example], target_semantics=TARGET_SEMANTICS_OCCUPIED_V2
        )
        self.assertNotEqual(v1, v2)
        self.assertEqual(
            v2,
            pixel_targets_digest(
                [example], target_semantics=TARGET_SEMANTICS_OCCUPIED_V2
            ),
        )
        from dataclasses import replace as dc_replace

        mutated = dc_replace(example, vacated_pixels=vacated[:-1])
        # The v1 digest ignores vacated pixels (byte-compatible with the
        # original spike); the v2 digest pins them.
        self.assertEqual(v1, pixel_targets_digest([mutated]))
        self.assertNotEqual(
            v2,
            pixel_targets_digest(
                [mutated], target_semantics=TARGET_SEMANTICS_OCCUPIED_V2
            ),
        )


class VacatedNegativeTrainingTests(unittest.TestCase):
    def _occupied_examples(self):
        # Sprite occupies one cell; the tracker cell map fires on the
        # sprite cell AND an adjacent vacated (background-valued) cell,
        # mirroring the union-blurred tracker anchor.  The head must
        # separate them by appearance.
        examples = []
        placements = ((1, 1, 2, 1), (2, 2, 1, 2), (1, 0, 2, 0), (2, 1, 1, 1))
        for index, (column, row, v_column, v_row) in enumerate(placements):
            x0, x1 = column * 8, column * 8 + 8
            y0, y1 = row * 10, row * 10 + 10
            vx0, vx1 = v_column * 8, v_column * 8 + 8
            vy0, vy1 = v_row * 10, v_row * 10 + 10
            frame = _frame(_block(x0, x1, y0, y1))
            examples.append(
                PixelMaskExample(
                    source_run_id="run-a" if index % 2 == 0 else "run-b",
                    group=index,
                    root_digest="root",
                    action="right",
                    duration=4,
                    endpoint_digest=frame.digest,
                    width=WIDTH,
                    height=HEIGHT,
                    columns=COLUMNS,
                    rows=ROWS,
                    target_pixels=tuple(sorted(_block_pixels(x0, x1, y0, y1))),
                    residual_pixels=(),
                    frame=frame,
                    cell_probabilities=_cell_map(
                        {(column, row): 1.0, (v_column, v_row): 1.0}
                    ),
                    vacated_pixels=tuple(
                        sorted(_block_pixels(vx0, vx1, vy0, vy1))
                    ),
                )
            )
        return examples

    def test_vacated_pixels_train_as_negatives_and_are_reported(self) -> None:
        examples = self._occupied_examples()
        head = _make_head(seed=5)
        history = train_pixel_mask_head(
            head,
            examples,
            "cpu",
            epochs=8,
            batch_size=2,
            learning_rate=1e-2,
            seed=7,
            vacated_weight=8.0,
        )
        self.assertTrue(history)
        self.assertGreaterEqual(history[-1].vacated_probability, 0.0)
        report = validate_pixel_mask_head(head, examples, "cpu", batch_size=2)
        self.assertEqual(report.vacated_pixels, 4 * 80)
        self.assertGreater(
            report.mean_target_probability, report.mean_vacated_probability
        )

    def test_default_vacated_weight_reproduces_v1_loss_behavior(self) -> None:
        # Without vacated pixels the vacated weight is inert: identical
        # seeds and examples give identical parameters either way.
        examples = [
            example
            for example in self._occupied_examples()
        ]
        stripped = [
            PixelMaskExample(
                source_run_id=example.source_run_id,
                group=example.group,
                root_digest=example.root_digest,
                action=example.action,
                duration=example.duration,
                endpoint_digest=example.endpoint_digest,
                width=example.width,
                height=example.height,
                columns=example.columns,
                rows=example.rows,
                target_pixels=example.target_pixels,
                residual_pixels=example.residual_pixels,
                frame=example.frame,
                cell_probabilities=example.cell_probabilities,
            )
            for example in examples
        ]
        first = _make_head(seed=9)
        train_pixel_mask_head(
            first, stripped, "cpu", epochs=2, batch_size=2, seed=11,
            vacated_weight=1.0,
        )
        second = _make_head(seed=9)
        train_pixel_mask_head(
            second, stripped, "cpu", epochs=2, batch_size=2, seed=11,
            vacated_weight=64.0,
        )
        self.assertEqual(first.checkpoint_digest, second.checkpoint_digest)

    def test_invalid_vacated_weight_refused(self) -> None:
        head = _make_head()
        with self.assertRaises(ValueError):
            train_pixel_mask_head(
                head,
                self._occupied_examples(),
                "cpu",
                epochs=1,
                vacated_weight=0.0,
            )


class ConventionV2CheckpointTests(unittest.TestCase):
    _PINS = {
        "label_manifest_digest": "a" * 64,
        "pixel_targets_sha256": "b" * 64,
        "tracker_parameter_digest": "c" * 64,
        "backbone_parameter_digest": "d" * 64,
    }

    def test_v2_checkpoint_pins_semantics_and_convention(self) -> None:
        head = _make_head()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head-v2.pt"
            save_pixel_mask_head_checkpoint(
                head,
                path,
                cell_columns=COLUMNS,
                cell_rows=ROWS,
                target_semantics=TARGET_SEMANTICS_OCCUPIED_V2,
                anchor_cell_dilation=ANCHOR_CELL_DILATION_V2,
                **self._PINS,
            )
            loaded, provenance = load_pixel_mask_head_checkpoint(path)
        self.assertEqual(provenance["target_semantics"], "occupied-v2")
        self.assertEqual(provenance["anchor_cell_dilation"], 0)
        self.assertEqual(loaded.target_semantics, "occupied-v2")
        self.assertEqual(loaded.anchor_cell_dilation, 0)

    def test_pre_v2_checkpoints_load_as_union_v1_convention(self) -> None:
        head = _make_head()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head-legacy.pt"
            save_pixel_mask_head_checkpoint(
                head, path, cell_columns=COLUMNS, cell_rows=ROWS, **self._PINS
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            # Simulate a checkpoint written before the v2 spike.
            del payload["target_semantics"]
            del payload["anchor_cell_dilation"]
            torch.save(payload, path)
            loaded, provenance = load_pixel_mask_head_checkpoint(path)
        self.assertEqual(provenance["target_semantics"], "union-v1")
        self.assertEqual(provenance["anchor_cell_dilation"], ANCHOR_CELL_DILATION)
        self.assertEqual(loaded.anchor_cell_dilation, ANCHOR_CELL_DILATION)

    def test_invalid_semantics_and_dilation_refused_on_save(self) -> None:
        head = _make_head()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_pixel_mask_head_checkpoint(
                    head,
                    Path(directory) / "head.pt",
                    cell_columns=COLUMNS,
                    cell_rows=ROWS,
                    target_semantics="occupied-v3",
                    **self._PINS,
                )
            with self.assertRaises(ValueError):
                save_pixel_mask_head_checkpoint(
                    head,
                    Path(directory) / "head.pt",
                    cell_columns=COLUMNS,
                    cell_rows=ROWS,
                    anchor_cell_dilation=-1,
                    **self._PINS,
                )


class ConventionV2PredictorTests(unittest.TestCase):
    def test_predictor_reads_the_convention_from_the_head(self) -> None:
        tracker = _StubTracker(
            _StubCellPrediction(COLUMNS, ROWS, _cell_map({(1, 1): 0.9}))
        )
        head = _make_head()
        head.anchor_cell_dilation = ANCHOR_CELL_DILATION_V2
        predictor = PixelSilhouettePredictor(tracker, head, device="cpu")
        self.assertEqual(predictor.anchor_cell_dilation, 0)
        prediction = predictor.predict(ROOT)
        undilated_anchor = anchor_pixel_region(
            _cell_map({(1, 1): 0.9}),
            COLUMNS,
            ROWS,
            WIDTH,
            HEIGHT,
            cell_dilation=0,
        )
        self.assertEqual(
            undilated_anchor, cell_pixel_block((1, 1), WIDTH, HEIGHT, COLUMNS, ROWS)
        )
        allowed = dilate_pixels(
            undilated_anchor, WIDTH, HEIGHT, SILHOUETTE_HALO_DILATION
        )
        self.assertTrue(prediction.mask <= allowed)

    def test_explicit_dilation_overrides_and_default_stays_v1(self) -> None:
        tracker = _StubTracker(
            _StubCellPrediction(COLUMNS, ROWS, _cell_map({(1, 1): 0.9}))
        )
        head = _make_head()
        # A head with no pinned convention keeps the v1 dilation.
        default_predictor = PixelSilhouettePredictor(tracker, head, device="cpu")
        self.assertEqual(
            default_predictor.anchor_cell_dilation, ANCHOR_CELL_DILATION
        )
        override = PixelSilhouettePredictor(
            tracker, head, device="cpu", anchor_cell_dilation=0
        )
        self.assertEqual(override.anchor_cell_dilation, 0)
        with self.assertRaises(ValueError):
            PixelSilhouettePredictor(
                tracker, head, device="cpu", anchor_cell_dilation=-1
            )


class FunctionalGateDriverTests(unittest.TestCase):
    def test_mask_source_description_reports_the_applied_convention(
        self,
    ) -> None:
        from lolo_agent.pixel_mask_train import (
            DEFAULT_FUNCTIONAL_GATE_REPORT,
            DEFAULT_HEAD_CHECKPOINT_V2,
            mask_source_description,
        )

        description = mask_source_description(TARGET_SEMANTICS_OCCUPIED_V2, 0)
        self.assertIn("dilation 0 cells", description)
        self.assertIn("occupied-v2", description)
        self.assertIn("unchanged substitution-replay helpers", description)
        self.assertEqual(
            DEFAULT_HEAD_CHECKPOINT_V2,
            "experiments/lolo1-wp5/pixel-mask-head-v2.pt",
        )
        self.assertEqual(
            DEFAULT_FUNCTIONAL_GATE_REPORT,
            "experiments/lolo1-wp5/functional-gate-v2-report.json",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
