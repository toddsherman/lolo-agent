import tempfile
import unittest

try:  # optional ML extra
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise unittest.SkipTest(
        "requires the optional 'ml' extra: pip install -e '.[ml]'"
    ) from exc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


from lolo_agent.controllable_tracker import (
    CHECKPOINT_ARCHITECTURE,
    ControllableArmExample,
    ControllableRegionTracker,
    arm_examples_from_records,
    load_controllable_tracker_checkpoint,
    load_labeled_arm_examples,
    sample_arm_examples,
    save_controllable_tracker_checkpoint,
    train_controllable_tracker,
    validate_controllable_tracker,
)
from lolo_agent.counterfactual_labels import (
    STATUS_LABELED,
    generate_labels,
    write_labels,
)
from lolo_agent.environment import Action
from lolo_agent.pixels import Frame
from lolo_agent.spatial_world_model import SpatialTokenDynamicsModel
from lolo_agent.strict_lineage import (
    audit_checkpoint_metadata,
    lint_strict_lineage,
)

GRID = 4
WIDTH = 8
HEIGHT = 8
_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "lolo_agent"


@dataclass(frozen=True)
class _StubSequence:
    """Minimal stand-in carrying the fields the label generator reads."""

    group: int
    frames: Tuple[Frame, ...]
    actions: Tuple[Action, ...]
    durations: Tuple[int, ...]
    source_run_id: str


def make_frame(marks: Dict[Tuple[int, int], int]) -> Frame:
    """Build a tiny frame whose 2x2-pixel cells carry the marked values."""

    pixels = bytearray([16] * (WIDTH * HEIGHT))
    for (column, row), value in marks.items():
        for y in range(row * 2, row * 2 + 2):
            for x in range(column * 2, column * 2 + 2):
                pixels[y * WIDTH + x] = value
    return Frame(WIDTH, HEIGHT, 1, bytes(pixels))


def one_step(
    root: Frame,
    endpoint: Frame,
    action: Action,
    duration: int,
    *,
    group: int = 0,
    run_id: str = "run-a",
) -> _StubSequence:
    return _StubSequence(group, (root, endpoint), (action,), (duration,), run_id)


def make_backbone(seed: int = 0) -> SpatialTokenDynamicsModel:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=GRID,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )


def make_tracker(seed: int = 0, backbone_seed: int = 0) -> ControllableRegionTracker:
    backbone = make_backbone(backbone_seed)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return ControllableRegionTracker(
            backbone, hidden_size=16, ensemble_size=2, columns=GRID, rows=GRID
        )


def make_example(
    frame: Frame,
    controllable: Tuple[Tuple[int, int], ...],
    residual: Tuple[Tuple[int, int], ...] = (),
    *,
    run_id: str = "run-a",
    group: int = 0,
    action: Action = Action.RIGHT,
) -> ControllableArmExample:
    return ControllableArmExample(
        source_run_id=run_id,
        group=group,
        root_digest="root",
        action=action.value,
        duration=4,
        endpoint_digest=frame.digest,
        columns=GRID,
        rows=GRID,
        controllable_cells=controllable,
        residual_cells=residual,
        frame=frame,
    )


def sprite_examples() -> Tuple[ControllableArmExample, ...]:
    """Sprite in varying cells; the target is always the sprite's cell."""

    examples = []
    cells = ((0, 0), (1, 1), (2, 2), (3, 3), (3, 0), (0, 3), (2, 1), (1, 2))
    for index, (column, row) in enumerate(cells):
        run_id = "run-a" if index % 2 == 0 else "run-b"
        examples.append(
            make_example(
                make_frame({(column, row): 200}),
                ((column, row),),
                run_id=run_id,
                group=index,
            )
        )
    return tuple(examples)


class ForwardApiTests(unittest.TestCase):
    def test_forward_shape_and_prediction_determinism(self) -> None:
        tracker = make_tracker()
        frames = torch.rand((2, 3, 128, 128))
        logits = tracker(frames)
        self.assertEqual(tuple(logits.shape), (2, 2, GRID, GRID))
        frame = make_frame({(1, 2): 200})
        first = tracker.predict(frame)
        second = tracker.predict(frame)
        self.assertEqual(first, second)
        self.assertEqual(first.columns, GRID)
        self.assertEqual(first.rows, GRID)
        self.assertEqual(len(first.probabilities), GRID)
        for row in first.probabilities:
            self.assertEqual(len(row), GRID)
            for value in row:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        self.assertGreaterEqual(first.confidence, 0.0)
        self.assertLessEqual(first.confidence, 1.0)

    def test_forward_rejects_bad_shapes(self) -> None:
        tracker = make_tracker()
        with self.assertRaises(ValueError):
            tracker(torch.rand((3, 128, 128)))
        with self.assertRaises(ValueError):
            tracker.forward_tokens(torch.rand((2, 8, 4)))


class CheckpointTests(unittest.TestCase):
    def test_save_load_roundtrip_and_digest_stability(self) -> None:
        tracker = make_tracker()
        manifest_digest = "d" * 64
        frame = make_frame({(2, 2): 200})
        expected = tracker.predict(frame)
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "tracker.pt"
            second_path = Path(directory) / "tracker-again.pt"
            saved = save_controllable_tracker_checkpoint(
                tracker, first_path, label_manifest_digest=manifest_digest
            )
            self.assertEqual(saved, tracker.checkpoint_digest)
            self.assertEqual(
                saved,
                save_controllable_tracker_checkpoint(
                    tracker, second_path, label_manifest_digest=manifest_digest
                ),
            )
            loaded, provenance = load_controllable_tracker_checkpoint(
                first_path, make_backbone(0)
            )
        self.assertEqual(loaded.checkpoint_digest, saved)
        self.assertEqual(provenance["label_manifest_sha256"], manifest_digest)
        self.assertEqual(loaded.predict(frame), expected)
        self.assertFalse(
            any(parameter.requires_grad for parameter in loaded.parameters())
        )

    def test_mismatched_backbone_is_refused(self) -> None:
        tracker = make_tracker()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.pt"
            save_controllable_tracker_checkpoint(
                tracker, path, label_manifest_digest="d" * 64
            )
            with self.assertRaises(ValueError):
                load_controllable_tracker_checkpoint(path, make_backbone(9))

    def test_missing_label_manifest_digest_is_refused(self) -> None:
        tracker = make_tracker()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_controllable_tracker_checkpoint(
                    tracker,
                    Path(directory) / "tracker.pt",
                    label_manifest_digest="",
                )

    def test_frozen_inference_does_not_change_parameters(self) -> None:
        tracker = make_tracker()
        tracker.freeze()
        digest = tracker.checkpoint_digest
        tracker.predict(make_frame({(1, 1): 200}))
        self.assertEqual(tracker.checkpoint_digest, digest)


class ProvenanceTests(unittest.TestCase):
    def test_checkpoint_provenance_fields_pass_the_strict_audit(self) -> None:
        tracker = make_tracker()
        manifest_digest = "e" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.pt"
            save_controllable_tracker_checkpoint(
                tracker, path, label_manifest_digest=manifest_digest
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["architecture"], CHECKPOINT_ARCHITECTURE)
        self.assertEqual(payload["reward_track"], "strict")
        self.assertEqual(payload["label_manifest_sha256"], manifest_digest)
        self.assertEqual(
            payload["backbone_parameter_sha256"],
            tracker.backbone.checkpoint_digest,
        )
        audit = audit_checkpoint_metadata(payload, label="tracker")
        self.assertEqual(audit.violations, ())
        self.assertFalse(audit.assisted)
        self.assertEqual(
            audit.declared_fields,
            ("reward_track", "persistent_inputs", "excluded_inputs"),
        )

    def test_tracker_modules_lint_clean(self) -> None:
        report = lint_strict_lineage(
            (
                _PACKAGE_ROOT / "controllable_tracker.py",
                _PACKAGE_ROOT / "controllable_tracker_train.py",
            )
        )
        verdicts = {entry.module: entry.assisted for entry in report.modules}
        self.assertEqual(
            verdicts,
            {
                "lolo_agent.controllable_tracker": False,
                "lolo_agent.controllable_tracker_train": False,
            },
        )
        self.assertFalse(report.assisted)


class FrozenBackboneTests(unittest.TestCase):
    def test_backbone_parameters_stay_frozen_through_training(self) -> None:
        tracker = make_tracker()
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in tracker.backbone.parameters()
            )
        )
        tracker.train()
        self.assertFalse(tracker.backbone.training)
        backbone_digest = tracker.backbone.checkpoint_digest
        head_digest = tracker.checkpoint_digest
        history = train_controllable_tracker(
            tracker,
            sprite_examples(),
            "cpu",
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            seed=3,
        )
        self.assertTrue(history)
        self.assertEqual(tracker.backbone.checkpoint_digest, backbone_digest)
        self.assertNotEqual(tracker.checkpoint_digest, head_digest)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in tracker.backbone.parameters()
            )
        )


class LabelRegressionTests(unittest.TestCase):
    """v316/v317-inspired failure modes as pure counterfactual-label cases."""

    def test_adjacent_same_appearance_region_is_excluded(self) -> None:
        # Two identical patches; only the one at (1,1) is action-correlated.
        root = make_frame({(1, 1): 200, (2, 1): 200})
        sequences = (
            one_step(root, make_frame({(0, 1): 200, (2, 1): 200}), Action.LEFT, 4),
            one_step(root, make_frame({(1, 0): 200, (2, 1): 200}), Action.UP, 4),
            one_step(root, make_frame({(1, 2): 200, (2, 1): 200}), Action.DOWN, 4),
            one_step(root, root, Action.NOOP, 4),
        )
        labels = generate_labels(sequences, columns=GRID, rows=GRID)
        self.assertEqual(len(labels), 1)
        twin = (2, 1)
        expectations = {
            Action.LEFT: ((0, 1), (1, 1)),
            Action.UP: ((1, 0), (1, 1)),
            Action.DOWN: ((1, 1), (1, 2)),
        }
        for arm in labels[0].arms:
            self.assertEqual(arm.status, STATUS_LABELED)
            self.assertNotIn(twin, arm.changed_cells)
            self.assertNotIn(twin, arm.controllable_cells)
            self.assertEqual(arm.controllable_cells, expectations[arm.action])
        examples, statistics = arm_examples_from_records(
            [record.payload() for record in labels]
        )
        self.assertEqual(statistics["examples"], 3)
        for example in examples:
            self.assertNotIn(twin, example.controllable_cells)
            self.assertNotIn(twin, example.residual_cells)

    def test_blocked_action_pose_change_stays_in_place(self) -> None:
        # UP is blocked: the region's appearance changes but its cell does not.
        root = make_frame({(1, 1): 200})
        sequences = (
            one_step(root, make_frame({(0, 1): 200}), Action.LEFT, 4),
            one_step(root, make_frame({(2, 1): 200}), Action.RIGHT, 4),
            one_step(root, make_frame({(1, 1): 190}), Action.UP, 4),
            one_step(root, root, Action.NOOP, 4),
        )
        labels = generate_labels(sequences, columns=GRID, rows=GRID)
        blocked = next(
            arm for arm in labels[0].arms if arm.action is Action.UP
        )
        self.assertEqual(blocked.status, STATUS_LABELED)
        self.assertEqual(blocked.changed_cells, ((1, 1),))
        self.assertEqual(blocked.controllable_cells, ((1, 1),))
        self.assertEqual(blocked.residual_cells, ())

    def test_autonomous_region_is_not_action_correlated(self) -> None:
        # A second region moves identically with or without input, so the
        # factual-versus-control difference cancels it everywhere.
        root = make_frame({(1, 1): 200, (3, 3): 90})
        drifted = {(3, 2): 90}
        sequences = (
            one_step(
                root, make_frame({(0, 1): 200, **drifted}), Action.LEFT, 4
            ),
            one_step(
                root, make_frame({(2, 1): 200, **drifted}), Action.RIGHT, 4
            ),
            one_step(root, make_frame({(1, 1): 200, **drifted}), Action.NOOP, 4),
        )
        labels = generate_labels(sequences, columns=GRID, rows=GRID)
        for arm in labels[0].arms:
            self.assertEqual(arm.status, STATUS_LABELED)
            self.assertNotIn((3, 3), arm.changed_cells)
            self.assertNotIn((3, 2), arm.changed_cells)
            self.assertNotIn((3, 3), arm.controllable_cells)
            self.assertNotIn((3, 2), arm.controllable_cells)


class LabelCorpusLoaderTests(unittest.TestCase):
    def _fixture_sequences(self) -> Tuple[_StubSequence, ...]:
        root = make_frame({(1, 1): 200})
        return (
            one_step(root, make_frame({(0, 1): 200}), Action.LEFT, 4),
            one_step(root, make_frame({(2, 1): 200}), Action.RIGHT, 4),
            one_step(root, root, Action.A, 4),
            one_step(root, root, Action.NOOP, 4),
        )

    def test_loader_verifies_digests_and_excludes_empty_masks(self) -> None:
        labels = generate_labels(
            self._fixture_sequences(), columns=GRID, rows=GRID
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "labels.jsonl"
            manifest = write_labels(labels, destination, reward_track="strict")
            examples, loaded_manifest, statistics = load_labeled_arm_examples(
                destination
            )
            self.assertEqual(loaded_manifest, manifest)
            self.assertEqual(statistics["labeled_arms"], 3)
            self.assertEqual(statistics["empty_mask_labeled_arms"], 1)
            self.assertEqual(statistics["examples"], 2)
            self.assertEqual(
                sorted(example.action for example in examples),
                [Action.LEFT.value, Action.RIGHT.value],
            )
            corrupted = destination.read_text(encoding="utf-8").replace(
                '"duration":4', '"duration":5', 1
            )
            destination.write_text(corrupted, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_labeled_arm_examples(destination)

    def test_manifest_digest_mismatch_is_refused(self) -> None:
        labels = generate_labels(
            self._fixture_sequences(), columns=GRID, rows=GRID
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "labels.jsonl"
            write_labels(labels, destination, reward_track="strict")
            manifest_path = Path(directory) / "labels.jsonl.manifest.json"
            tampered = manifest_path.read_text(encoding="utf-8").replace(
                '"content_digest": "', '"content_digest": "0', 1
            )
            manifest_path.write_text(tampered, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_labeled_arm_examples(destination)

    def test_sampling_is_deterministic_and_order_stable(self) -> None:
        examples = sprite_examples()
        first = sample_arm_examples(examples, 3, seed=7)
        second = sample_arm_examples(tuple(reversed(examples)), 3, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(
            sample_arm_examples(examples, 100, seed=7),
            sorted(examples, key=lambda example: example.sort_key),
        )


class SyntheticTrainingTests(unittest.TestCase):
    def test_training_beats_the_untrained_baseline_on_synthetic_masks(
        self,
    ) -> None:
        examples = sprite_examples()
        baseline = make_tracker(seed=11)
        before = validate_controllable_tracker(baseline, examples, "cpu", 4)
        tracker = make_tracker(seed=11)
        history = train_controllable_tracker(
            tracker,
            examples,
            "cpu",
            epochs=20,
            batch_size=4,
            learning_rate=5e-3,
            seed=5,
        )
        after = validate_controllable_tracker(tracker, examples, "cpu", 4)
        self.assertLess(history[-1].loss, history[0].loss)
        self.assertLess(after.loss, before.loss)
        self.assertGreater(after.roc_auc, before.roc_auc)
        self.assertGreater(
            after.mean_controllable_probability,
            after.mean_background_probability,
        )

    def test_grid_mismatch_and_missing_frames_are_rejected(self) -> None:
        tracker = make_tracker()
        mismatched = ControllableArmExample(
            source_run_id="run-a",
            group=0,
            root_digest="root",
            action=Action.RIGHT.value,
            duration=4,
            endpoint_digest="digest",
            columns=GRID + 1,
            rows=GRID,
            controllable_cells=((0, 0),),
            residual_cells=(),
            frame=make_frame({(1, 1): 200}),
        )
        with self.assertRaises(ValueError):
            validate_controllable_tracker(tracker, [mismatched], "cpu")
        undecoded = make_example(make_frame({(1, 1): 200}), ((1, 1),))
        undecoded = ControllableArmExample(
            **{**undecoded.__dict__, "frame": None}
        )
        with self.assertRaises(ValueError):
            train_controllable_tracker(tracker, [undecoded], "cpu", epochs=1)


if __name__ == "__main__":
    unittest.main()
