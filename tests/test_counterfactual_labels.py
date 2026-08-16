import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from lolo_agent.counterfactual_labels import (
    CENSOR_ABSENT_CONTROL,
    CENSOR_AMBIGUOUS_CONTROL,
    CENSOR_AMBIGUOUS_ENDPOINT,
    CENSOR_NO_SIBLING_CORROBORATION,
    STATUS_CENSORED,
    STATUS_LABELED,
    collect_counterfactual_roots,
    connected_components,
    content_digest,
    cell_difference,
    generate_labels,
    generate_store_labels,
    label_counterfactual_root,
    labels_manifest,
    open_strict_store,
    store_root_statistics,
    write_labels,
)
from lolo_agent.environment import Action
from lolo_agent.pixels import Frame

GRID = 4
WIDTH = 8
HEIGHT = 8


@dataclass(frozen=True)
class _StubSequence:
    """Minimal stand-in carrying the sequence fields the generator reads."""

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
    root: Frame, endpoint: Frame, action: Action, duration: int, *, group: int = 0,
    run_id: str = "run-a",
) -> _StubSequence:
    return _StubSequence(group, (root, endpoint), (action,), (duration,), run_id)


def sibling_fixture() -> Tuple[_StubSequence, ...]:
    """Root with a controlled sprite at (1,1) moved by three sibling actions."""

    root = make_frame({(1, 1): 200})
    return (
        one_step(root, make_frame({(0, 1): 200}), Action.LEFT, 4),
        one_step(root, make_frame({(2, 1): 200}), Action.RIGHT, 4),
        one_step(root, make_frame({(1, 0): 200}), Action.UP, 4),
        one_step(root, root, Action.NOOP, 4),
    )


class CellDifferenceTests(unittest.TestCase):
    def test_exact_cell_difference_and_geometry_guard(self) -> None:
        first = make_frame({(1, 1): 200})
        second = make_frame({(1, 1): 200, (3, 0): 40})
        self.assertEqual(cell_difference(first, first, GRID, GRID), frozenset())
        self.assertEqual(
            cell_difference(first, second, GRID, GRID), frozenset({(3, 0)})
        )
        with self.assertRaises(ValueError):
            cell_difference(first, Frame(4, 4, 1, bytes(16)), GRID, GRID)
        with self.assertRaises(ValueError):
            cell_difference(first, second, WIDTH + 1, GRID)

    def test_connected_components_use_four_connectivity(self) -> None:
        components = connected_components({(0, 0), (1, 0), (2, 2), (3, 3)})
        self.assertEqual(
            components, (((0, 0), (1, 0)), ((2, 2),), ((3, 3),))
        )


class ConsistencyRuleTests(unittest.TestCase):
    def test_sibling_consistent_component_is_controllable(self) -> None:
        labels = generate_labels(sibling_fixture(), columns=GRID, rows=GRID)
        self.assertEqual(len(labels), 1)
        record = labels[0]
        self.assertEqual(len(record.arms), 3)
        expectations = {
            Action.LEFT: ((0, 1), (1, 1)),
            Action.RIGHT: ((1, 1), (2, 1)),
            Action.UP: ((1, 0), (1, 1)),
        }
        for arm in record.arms:
            self.assertEqual(arm.status, STATUS_LABELED)
            self.assertEqual(arm.corroborating_arms, 2)
            self.assertEqual(arm.changed_cells, expectations[arm.action])
            self.assertEqual(arm.controllable_cells, expectations[arm.action])
            self.assertEqual(arm.controllable_components, (expectations[arm.action],))
            self.assertEqual(arm.residual_cells, ())

    def test_no_effect_arm_labels_empty_masks(self) -> None:
        root = make_frame({(1, 1): 200})
        sequences = sibling_fixture() + (
            one_step(root, root, Action.A, 4),
        )
        labels = generate_labels(sequences, columns=GRID, rows=GRID)
        a_arm = next(
            arm for arm in labels[0].arms if arm.action is Action.A
        )
        self.assertEqual(a_arm.status, STATUS_LABELED)
        self.assertEqual(a_arm.changed_cells, ())
        self.assertEqual(a_arm.controllable_cells, ())
        self.assertEqual(a_arm.residual_cells, ())


class LeaveOneActionOutTests(unittest.TestCase):
    def test_action_specific_change_falls_to_residual(self) -> None:
        root = make_frame({(1, 1): 200, (3, 3): 90})
        sequences = (
            one_step(root, make_frame({(0, 1): 200, (3, 3): 90}), Action.LEFT, 4),
            # RIGHT additionally removes the (3,3) region; no sibling action
            # corroborates that cell, so it must not become controllable.
            one_step(root, make_frame({(2, 1): 200}), Action.RIGHT, 4),
            one_step(root, make_frame({(1, 0): 200, (3, 3): 90}), Action.UP, 4),
            one_step(root, root, Action.NOOP, 4),
        )
        labels = generate_labels(sequences, columns=GRID, rows=GRID)
        right = next(arm for arm in labels[0].arms if arm.action is Action.RIGHT)
        self.assertEqual(right.status, STATUS_LABELED)
        self.assertEqual(right.changed_cells, ((1, 1), (2, 1), (3, 3)))
        self.assertEqual(right.controllable_cells, ((1, 1), (2, 1)))
        self.assertEqual(right.residual_cells, ((3, 3),))
        left = next(arm for arm in labels[0].arms if arm.action is Action.LEFT)
        self.assertEqual(left.residual_cells, ())

    def test_same_action_arms_never_corroborate_each_other(self) -> None:
        root = make_frame({(1, 1): 200})
        sequences = (
            one_step(root, make_frame({(0, 1): 200}), Action.LEFT, 4),
            one_step(root, make_frame({(0, 1): 201}), Action.LEFT, 8),
            one_step(root, root, Action.NOOP, 4),
            one_step(root, root, Action.NOOP, 8),
        )
        labels = generate_labels(sequences, columns=GRID, rows=GRID)
        self.assertEqual(len(labels), 1)
        for arm in labels[0].arms:
            self.assertEqual(arm.status, STATUS_CENSORED)
            self.assertEqual(arm.censor_reason, CENSOR_NO_SIBLING_CORROBORATION)
            self.assertEqual(arm.changed_cells, ())

    def test_blocked_siblings_abstain_from_corroboration(self) -> None:
        root = make_frame({(1, 1): 200})
        sequences = (
            one_step(root, make_frame({(0, 1): 200}), Action.LEFT, 4),
            one_step(root, make_frame({(2, 1): 200}), Action.RIGHT, 4),
            # Blocked action: endpoint identical to the control endpoint.
            one_step(root, root, Action.UP, 4),
            one_step(root, root, Action.NOOP, 4),
        )
        labels = generate_labels(sequences, columns=GRID, rows=GRID)
        left = next(arm for arm in labels[0].arms if arm.action is Action.LEFT)
        self.assertEqual(left.status, STATUS_LABELED)
        self.assertEqual(left.corroborating_arms, 1)
        self.assertEqual(left.controllable_cells, ((0, 1), (1, 1)))


class CensoringTests(unittest.TestCase):
    def test_absent_control_censors_whole_root(self) -> None:
        root = make_frame({(1, 1): 200})
        sequences = (
            one_step(root, make_frame({(0, 1): 200}), Action.LEFT, 4),
            one_step(root, make_frame({(2, 1): 200}), Action.RIGHT, 4),
            # The only NOOP has a different duration, so no arm is controlled.
            one_step(root, root, Action.NOOP, 8),
        )
        self.assertEqual(generate_labels(sequences, columns=GRID, rows=GRID), ())
        record = label_counterfactual_root(
            collect_counterfactual_roots(sequences)[0], {}, columns=GRID, rows=GRID
        )
        self.assertEqual(len(record.arms), 2)
        for arm in record.arms:
            self.assertEqual(arm.status, STATUS_CENSORED)
            self.assertEqual(arm.censor_reason, CENSOR_ABSENT_CONTROL)
            self.assertIsNone(arm.control_digest)

    def test_duration_mismatched_arm_is_censored_within_labeled_root(self) -> None:
        root = make_frame({(1, 1): 200})
        sequences = sibling_fixture() + (
            one_step(root, make_frame({(1, 3): 200}), Action.DOWN, 8),
        )
        labels = generate_labels(sequences, columns=GRID, rows=GRID)
        down = next(arm for arm in labels[0].arms if arm.action is Action.DOWN)
        self.assertEqual(down.status, STATUS_CENSORED)
        self.assertEqual(down.censor_reason, CENSOR_ABSENT_CONTROL)
        left = next(arm for arm in labels[0].arms if arm.action is Action.LEFT)
        self.assertEqual(left.status, STATUS_LABELED)

    def test_ambiguous_endpoint_and_ambiguous_control_censoring(self) -> None:
        root = make_frame({(1, 1): 200})
        sequences = sibling_fixture() + (
            one_step(root, make_frame({(3, 0): 70}), Action.A, 4),
            one_step(root, make_frame({(3, 0): 71}), Action.A, 4),
        )
        labels = generate_labels(sequences, columns=GRID, rows=GRID)
        a_arm = next(arm for arm in labels[0].arms if arm.action is Action.A)
        self.assertEqual(a_arm.status, STATUS_CENSORED)
        self.assertEqual(a_arm.censor_reason, CENSOR_AMBIGUOUS_ENDPOINT)
        self.assertEqual(len(a_arm.endpoint_digests), 2)

        ambiguous_control = (
            one_step(root, make_frame({(0, 1): 200}), Action.LEFT, 4, group=1),
            one_step(root, root, Action.NOOP, 4, group=1),
            one_step(root, make_frame({(3, 3): 5}), Action.NOOP, 4, group=1),
        )
        record = label_counterfactual_root(
            collect_counterfactual_roots(ambiguous_control)[0],
            {},
            columns=GRID,
            rows=GRID,
        )
        self.assertEqual(record.arms[0].status, STATUS_CENSORED)
        self.assertEqual(record.arms[0].censor_reason, CENSOR_AMBIGUOUS_CONTROL)


class DigestDeterminismTests(unittest.TestCase):
    def test_regeneration_and_input_order_do_not_change_digests(self) -> None:
        sequences = sibling_fixture()
        first = [record.payload() for record in generate_labels(sequences, columns=GRID, rows=GRID)]
        second = [
            record.payload()
            for record in generate_labels(tuple(reversed(sequences)), columns=GRID, rows=GRID)
        ]
        self.assertEqual(first, second)
        for payload in first:
            self.assertEqual(payload["content_digest"], content_digest(payload))
        self.assertEqual(
            labels_manifest(generate_labels(sequences, columns=GRID, rows=GRID)),
            labels_manifest(generate_labels(sequences, columns=GRID, rows=GRID)),
        )

    def test_content_change_changes_digest(self) -> None:
        baseline = generate_labels(sibling_fixture(), columns=GRID, rows=GRID)
        root = make_frame({(1, 1): 200})
        perturbed_sequences = (
            one_step(root, make_frame({(0, 1): 201}), Action.LEFT, 4),
            one_step(root, make_frame({(2, 1): 200}), Action.RIGHT, 4),
            one_step(root, make_frame({(1, 0): 200}), Action.UP, 4),
            one_step(root, root, Action.NOOP, 4),
        )
        perturbed = generate_labels(perturbed_sequences, columns=GRID, rows=GRID)
        self.assertNotEqual(
            baseline[0].payload()["content_digest"],
            perturbed[0].payload()["content_digest"],
        )


class StoreIntegrationTests(unittest.TestCase):
    def _visual_sequences(self):
        from lolo_agent.ensemble_world_model import VisualSequence

        return [
            VisualSequence(
                stub.group, stub.frames, stub.actions, stub.durations, stub.source_run_id
            )
            for stub in sibling_fixture()
        ]

    def test_store_walk_matches_pure_generation_and_writes_jsonl(self) -> None:
        from lolo_agent.sequence_store import SequenceStore

        sequences = self._visual_sequences()
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceStore(Path(directory) / "dataset")
            store.bind_reward_track("strict")
            store.append_segment("cycle-000001", sequences)
            store = open_strict_store(Path(directory) / "dataset")
            labels = generate_store_labels(store, columns=GRID, rows=GRID)
            statistics = store_root_statistics(
                collect_counterfactual_roots(sequences)
            )
            destination = Path(directory) / "labels.jsonl"
            manifest = write_labels(labels, destination, reward_track="strict")
            with self.assertRaises(FileExistsError):
                write_labels(labels, destination, reward_track="strict")
            lines = destination.read_text(encoding="utf-8").splitlines()
            sidecar = json.loads(
                (Path(directory) / "labels.jsonl.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        expected = [
            record.payload()
            for record in generate_labels(sequences, columns=GRID, rows=GRID)
        ]
        self.assertEqual([record.payload() for record in labels], expected)
        self.assertEqual([json.loads(line) for line in lines], expected)
        self.assertEqual(manifest, sidecar)
        self.assertEqual(manifest["reward_track"], "strict")
        self.assertEqual(manifest["roots"], 1)
        self.assertEqual(manifest["labeled_arms"], 3)
        self.assertEqual(statistics["causal_roots"], 1)
        self.assertEqual(statistics["control_paired_roots"], 1)
        self.assertEqual(statistics["eligible_factual_arms"], 3)

    def test_maximum_roots_caps_deterministically(self) -> None:
        from lolo_agent.sequence_store import SequenceStore

        sequences = self._visual_sequences()
        shifted = [
            type(sequence)(
                sequence.group + 1,
                sequence.frames,
                sequence.actions,
                sequence.durations,
                sequence.source_run_id,
            )
            for sequence in sequences
        ]
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceStore(Path(directory))
            store.bind_reward_track("strict")
            store.append_segment("cycle-000001", sequences + shifted)
            capped = generate_store_labels(
                store, columns=GRID, rows=GRID, maximum_roots=1
            )
            complete = generate_store_labels(store, columns=GRID, rows=GRID)
        self.assertEqual(len(capped), 1)
        self.assertEqual(len(complete), 2)
        self.assertEqual(capped[0].payload(), complete[0].payload())
        self.assertEqual(capped[0].group, 0)

    def test_non_strict_stores_are_refused(self) -> None:
        from lolo_agent.sequence_store import SequenceStore

        with tempfile.TemporaryDirectory() as directory:
            assisted = SequenceStore(Path(directory) / "assisted")
            assisted.bind_reward_track("assisted")
            with self.assertRaises(ValueError):
                open_strict_store(Path(directory) / "assisted")
            unbound = SequenceStore(Path(directory) / "unbound")
            with self.assertRaises(ValueError):
                generate_store_labels(unbound)


if __name__ == "__main__":
    unittest.main()
