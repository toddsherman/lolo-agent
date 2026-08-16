from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Sequence, Tuple

from lolo_agent.controllable_tracker import (
    _roc_auc,
    arm_examples_from_records,
)
from lolo_agent.pixels import Frame
from lolo_agent.run_logging import encode_png
from lolo_agent.tracker_ood_eval import (
    GRID_COLUMNS,
    GRID_ROWS,
    ProbeEdge,
    RunFrameCache,
    arm_localization_metrics,
    assisted_cell_index,
    assisted_reference_metrics,
    build_report,
    censor_counts,
    collect_probe_roots,
    content_digest,
    label_probe_roots,
    predict_probability_maps,
    probe_first_step_edges,
    score_examples,
    state_frame_index,
    state_group,
    summarize_metrics,
)

Cell = Tuple[int, int]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_RUN = (
    _REPO_ROOT
    / "experiments/lolo1-entity-v10/evaluations"
    / "entity-v322-room3-paired-probe-arm-a-pushed-d12"
)


def _frame(changed: Dict[Cell, int]) -> Frame:
    """A 16x15 single-channel frame; cells map 1:1 to pixels."""

    pixels = bytearray(GRID_COLUMNS * GRID_ROWS)
    for (column, row), value in changed.items():
        pixels[row * GRID_COLUMNS + column] = value
    return Frame(
        width=GRID_COLUMNS, height=GRID_ROWS, channels=1, pixels=bytes(pixels)
    )


def _branch_event(
    parent: str,
    path: Sequence[str],
    durations: Sequence[int],
    frame: str,
    slot: Tuple[int, int] | None = None,
) -> Dict[str, object]:
    event: Dict[str, object] = {
        "event": "human_prior_option_branch_verified",
        "parent_state_id": parent,
        "path": list(path),
        "durations": list(durations),
        "frame": frame,
        "frame_width": 256,
        "frame_height": 240,
    }
    if slot is not None:
        event["human_prior_target_player_slot"] = list(slot)
    return event


def _local_neutral_event(
    parent: str, duration: int, frame: str
) -> Dict[str, object]:
    return {
        "event": "human_prior_option_local_neutral_verified",
        "parent_state_id": parent,
        "action": "noop",
        "action_frames": duration,
        "frame": frame,
    }


def _probability_map(cells: Dict[Cell, float]) -> Tuple[Tuple[float, ...], ...]:
    return tuple(
        tuple(
            float(cells.get((column, row), 0.0))
            for column in range(GRID_COLUMNS)
        )
        for row in range(GRID_ROWS)
    )


class ProbeEdgeExtractionTests(unittest.TestCase):
    def test_branch_event_yields_last_step_edge(self) -> None:
        edges = probe_first_step_edges(
            [
                _branch_event(
                    "state-00000007",
                    ["up", "left"],
                    [8, 16],
                    "f" * 64,
                    slot=(112, 96),
                )
            ]
        )
        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(edge.parent_state_id, "state-00000007")
        self.assertEqual(edge.action, "left")
        self.assertEqual(edge.duration, 16)
        self.assertEqual(edge.endpoint_digest, "f" * 64)
        self.assertEqual(edge.assisted_player_cell, (7, 6))

    def test_local_neutral_yields_noop_edge(self) -> None:
        edges = probe_first_step_edges(
            [_local_neutral_event("state-00000007", 16, "a" * 64)]
        )
        self.assertEqual(
            edges,
            (
                ProbeEdge(
                    parent_state_id="state-00000007",
                    action="noop",
                    duration=16,
                    endpoint_digest="a" * 64,
                ),
            ),
        )

    def test_root_neutral_used_only_at_depth_one(self) -> None:
        depth_one = {
            "event": "human_prior_option_neutral_verified",
            "source_state_id": "state-00000003",
            "path": ["noop"],
            "durations": [16],
            "frame": "b" * 64,
        }
        depth_two = {
            "event": "human_prior_option_neutral_verified",
            "source_state_id": "state-00000003",
            "path": ["noop", "noop"],
            "durations": [16, 8],
            "frame": "c" * 64,
        }
        edges = probe_first_step_edges([depth_one, depth_two])
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].action, "noop")
        self.assertEqual(edges[0].parent_state_id, "state-00000003")
        self.assertEqual(edges[0].endpoint_digest, "b" * 64)

    def test_unrelated_events_ignored_and_malformed_rejected(self) -> None:
        self.assertEqual(
            probe_first_step_edges([{"event": "env_step", "frame": "d" * 64}]),
            (),
        )
        with self.assertRaises(ValueError):
            probe_first_step_edges(
                [_branch_event("state-00000001", [], [], "e" * 64)]
            )

    def test_state_frame_index_and_group(self) -> None:
        index = state_frame_index(
            [
                {
                    "event": "state_saved",
                    "state_id": "state-00000009",
                    "frame": "9" * 64,
                },
                {"event": "state_saved", "state_id": None, "frame": "0" * 64},
            ]
        )
        self.assertEqual(index, {"state-00000009": "9" * 64})
        self.assertEqual(state_group("state-00000009"), 9)
        with self.assertRaises(ValueError):
            state_group("no-numeric-suffix")

    def test_assisted_cell_index_marks_disagreement(self) -> None:
        agree = ProbeEdge("state-00000001", "up", 8, "a" * 64, (1, 2))
        disagree = ProbeEdge("state-00000001", "up", 8, "a" * 64, (3, 4))
        index = assisted_cell_index([agree, agree, disagree])
        self.assertIsNone(index[("state-00000001", "up", 8)])
        self.assertEqual(
            assisted_cell_index([agree])[("state-00000001", "up", 8)], (1, 2)
        )


class ProbeRootLabelingTests(unittest.TestCase):
    """The wiring into counterfactual_labels' own labeling rule."""

    def _fixture(self) -> Tuple[Dict[str, object], Dict[str, Frame]]:
        control = _frame({})
        right = _frame({(2, 3): 200, (3, 3): 200})
        down = _frame({(3, 3): 150, (3, 4): 150})
        # A blocked action reaches the control endpoint: empty changed-cell
        # set, so it abstains from corroboration instead of vetoing it.
        inert = _frame({})
        frames = {frame.digest: frame for frame in (control, right, down, inert)}
        events = [
            {
                "event": "state_saved",
                "state_id": "state-00000005",
                "frame": "root" + "0" * 60,
            },
            _local_neutral_event("state-00000005", 4, control.digest),
            _branch_event("state-00000005", ["right"], [4], right.digest),
            _branch_event("state-00000005", ["down"], [4], down.digest),
            _branch_event("state-00000005", ["a"], [4], inert.digest),
            # Duration 8 has no matched control: censored absent_control.
            _branch_event("state-00000005", ["up"], [8], right.digest),
        ]
        return {"events": events}, frames

    def test_labeling_matches_counterfactual_rule(self) -> None:
        fixture, frames = self._fixture()
        edges = probe_first_step_edges(fixture["events"])
        state_frames = state_frame_index(fixture["events"])
        roots = collect_probe_roots("synthetic-run", edges, state_frames)
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].root_digest, "root" + "0" * 60)
        records, stats = label_probe_roots(roots, frames.__getitem__)
        self.assertEqual(stats["roots"], 1)
        self.assertEqual(stats["roots_labeled"], 1)
        by_action = {arm["action"]: arm for arm in records[0]["arms"]}
        self.assertEqual(by_action["right"]["status"], "labeled")
        self.assertEqual(
            by_action["right"]["controllable_cells"], [[2, 3], [3, 3]]
        )
        self.assertEqual(by_action["down"]["status"], "labeled")
        self.assertEqual(
            by_action["down"]["controllable_cells"], [[3, 3], [3, 4]]
        )
        # The inert arm's change is not corroborated by any sibling action.
        self.assertEqual(by_action["a"]["status"], "labeled")
        self.assertEqual(by_action["a"]["controllable_cells"], [])
        self.assertEqual(by_action["up"]["censor_reason"], "absent_control")
        self.assertEqual(censor_counts(records), {"absent_control": 1})
        examples, example_stats = arm_examples_from_records(records)
        self.assertEqual(len(examples), 2)
        self.assertEqual(example_stats["empty_mask_labeled_arms"], 1)

    def test_ambiguous_endpoint_censored(self) -> None:
        fixture, frames = self._fixture()
        other = _frame({(9, 9): 40})
        frames[other.digest] = other
        fixture["events"].append(
            _branch_event("state-00000005", ["right"], [4], other.digest)
        )
        edges = probe_first_step_edges(fixture["events"])
        roots = collect_probe_roots("synthetic-run", edges, {})
        records, _stats = label_probe_roots(roots, frames.__getitem__)
        by_action = {arm["action"]: arm for arm in records[0]["arms"]}
        self.assertEqual(by_action["right"]["status"], "censored")
        self.assertEqual(
            by_action["right"]["censor_reason"], "ambiguous_endpoint"
        )

    def test_root_without_control_produces_no_record(self) -> None:
        right = _frame({(2, 3): 200})
        events = [
            _branch_event("state-00000006", ["right"], [4], right.digest)
        ]
        roots = collect_probe_roots(
            "synthetic-run", probe_first_step_edges(events), {}
        )
        records, stats = label_probe_roots(
            roots, {right.digest: right}.__getitem__
        )
        self.assertEqual(records, [])
        self.assertEqual(stats["roots_without_eligible_arms"], 1)


class MetricTests(unittest.TestCase):
    def test_arm_localization_metrics_hit(self) -> None:
        probabilities = _probability_map({(2, 3): 0.9, (5, 5): 0.6})
        metrics = arm_localization_metrics(
            probabilities, [(2, 3), (3, 3)], threshold=0.5
        )
        self.assertTrue(metrics["hit"])
        self.assertTrue(metrics["argmax_in_true"])
        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["true_max_probability"], 0.9)
        self.assertAlmostEqual(metrics["true_mean_probability"], 0.45)
        self.assertEqual(metrics["predicted_mask_cells"], 2)

    def test_arm_localization_metrics_miss(self) -> None:
        probabilities = _probability_map({(5, 5): 0.9})
        metrics = arm_localization_metrics(
            probabilities, [(2, 3)], threshold=0.5
        )
        self.assertFalse(metrics["hit"])
        self.assertFalse(metrics["argmax_in_true"])
        self.assertEqual(metrics["iou"], 0.0)
        self.assertEqual(metrics["true_max_probability"], 0.0)

    def test_requires_true_cells(self) -> None:
        with self.assertRaises(ValueError):
            arm_localization_metrics(_probability_map({}), [])

    def test_score_and_summarize(self) -> None:
        control = _frame({})
        right = _frame({(2, 3): 200, (3, 3): 200})
        down = _frame({(3, 3): 150, (3, 4): 150})
        events = [
            _local_neutral_event("state-00000005", 4, control.digest),
            _branch_event("state-00000005", ["right"], [4], right.digest),
            _branch_event("state-00000005", ["down"], [4], down.digest),
        ]
        frames = {frame.digest: frame for frame in (control, right, down)}
        roots = collect_probe_roots(
            "synthetic-run", probe_first_step_edges(events), {}
        )
        records, _stats = label_probe_roots(roots, frames.__getitem__)
        examples, _example_stats = arm_examples_from_records(records)
        maps = {
            right.digest: _probability_map({(2, 3): 0.9}),
            down.digest: _probability_map({(9, 9): 0.8}),
        }
        rows, probabilities, labels = score_examples(examples, maps)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(probabilities), 2 * GRID_COLUMNS * GRID_ROWS)
        self.assertEqual(sum(labels), 4)
        summary = summarize_metrics(rows, probabilities, labels)
        self.assertEqual(summary["arms"], 2)
        self.assertAlmostEqual(summary["hit_rate"], 0.5)
        self.assertAlmostEqual(summary["argmax_in_true_rate"], 0.5)
        self.assertAlmostEqual(summary["iou"]["fraction_zero"], 0.5)
        self.assertAlmostEqual(
            summary["cell_auc"], _roc_auc(probabilities, labels)
        )
        self.assertAlmostEqual(summary["true_cells_mean"], 2.0)

    def test_summarize_empty(self) -> None:
        self.assertEqual(summarize_metrics([], [], []), {"arms": 0})

    def test_assisted_reference_metrics(self) -> None:
        control = _frame({})
        right = _frame({(2, 3): 200, (3, 3): 200})
        down = _frame({(3, 3): 150, (3, 4): 150})
        events = [
            _local_neutral_event("state-00000005", 4, control.digest),
            _branch_event(
                "state-00000005", ["right"], [4], right.digest, slot=(32, 48)
            ),
            _branch_event("state-00000005", ["down"], [4], down.digest),
        ]
        frames = {frame.digest: frame for frame in (control, right, down)}
        edges = probe_first_step_edges(events)
        roots = collect_probe_roots("synthetic-run", edges, {})
        records, _stats = label_probe_roots(roots, frames.__getitem__)
        examples, _example_stats = arm_examples_from_records(records)
        maps = {
            right.digest: _probability_map({(2, 3): 0.9}),
            down.digest: _probability_map({}),
        }
        reference = assisted_reference_metrics(
            examples,
            maps,
            assisted_cell_index(edges),
            {5: "state-00000005"},
        )
        # Slot (32, 48) on a 256x240 frame is cell (2, 3).
        self.assertEqual(reference["arms_with_assisted_cell"], 1)
        self.assertAlmostEqual(reference["probability_mean"], 0.9)
        self.assertAlmostEqual(reference["fraction_ge_threshold"], 1.0)
        self.assertAlmostEqual(
            reference["assisted_cell_in_true_cells_rate"], 1.0
        )

    def test_predict_probability_maps_uses_getter_batches(self) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover
            self.skipTest("torch unavailable")

        class _StubTracker(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.heads = torch.nn.ModuleList(
                    [torch.nn.Linear(1, 1, bias=False)]
                )

            def predict_map(self, frames):
                batch = frames.shape[0]
                mean = torch.zeros((batch, GRID_ROWS, GRID_COLUMNS))
                mean[:, 3, 2] = 0.75
                return mean, torch.zeros_like(mean)

        frame = _frame({(1, 1): 10})
        maps = predict_probability_maps(
            _StubTracker(),
            [frame.digest],
            {frame.digest: frame}.__getitem__,
            batch_size=2,
        )
        self.assertAlmostEqual(maps[frame.digest][3][2], 0.75)
        self.assertAlmostEqual(maps[frame.digest][0][0], 0.0)


class FrameCacheTests(unittest.TestCase):
    def test_cache_decodes_and_verifies(self) -> None:
        frame = _frame({(4, 4): 77})
        with tempfile.TemporaryDirectory() as scratch:
            frames_dir = Path(scratch)
            (frames_dir / f"{frame.digest}.png").write_bytes(encode_png(frame))
            tampered = _frame({(4, 4): 78})
            (frames_dir / f"{'0' * 64}.png").write_bytes(encode_png(tampered))
            cache = RunFrameCache(frames_dir, capacity=2)
            self.assertEqual(cache.get(frame.digest).pixels, frame.pixels)
            # Second read is served from the cache.
            self.assertIs(cache.get(frame.digest), cache.get(frame.digest))
            with self.assertRaises(KeyError):
                cache.get("f" * 64)
            with self.assertRaises(ValueError):
                cache.get("0" * 64)


class ReportTests(unittest.TestCase):
    def _report(self) -> Dict[str, object]:
        run = {
            "run_id": "synthetic-run",
            "extraction": {"scored_arms": 2},
            "metrics": {
                "arms": 2,
                "hit_rate": 0.5,
                "cell_auc": 0.75,
                "iou": {"mean": 0.25},
                "true_cell_probability": {"max_mean": 0.45},
            },
            "assisted_reference": {"arms_with_assisted_cell": 0},
        }
        pooled = {"runs": ["synthetic-run"], "metrics": run["metrics"]}
        held_in = {
            "corpus": "synthetic",
            "metrics": {
                "arms": 3,
                "hit_rate": 1.0,
                "cell_auc": 0.999,
                "iou": {"mean": 0.75},
                "true_cell_probability": {"max_mean": 0.95},
            },
        }
        return build_report([run], pooled, held_in, {"checkpoint": "stub"})

    def test_report_digest_deterministic_and_sensitive(self) -> None:
        first = self._report()
        second = self._report()
        self.assertEqual(first["content_digest"], second["content_digest"])
        self.assertEqual(first["content_digest"], content_digest(first))
        self.assertEqual(first["result"], "report-only-no-gate")
        self.assertAlmostEqual(
            first["gap_held_in_minus_room3"]["hit_rate"], 0.5
        )
        self.assertAlmostEqual(
            first["gap_held_in_minus_room3"]["iou_mean"], 0.5
        )
        perturbed = self._report()
        perturbed["room3_pooled"]["metrics"] = dict(
            perturbed["room3_pooled"]["metrics"], hit_rate=0.6
        )
        self.assertNotEqual(
            content_digest(perturbed), first["content_digest"]
        )

    def test_report_serializes_canonically(self) -> None:
        report = self._report()
        json.dumps(report, sort_keys=True, allow_nan=False)


@unittest.skipUnless(_REAL_RUN.exists(), "real Room 3 probe run not present")
class RealTelemetryTests(unittest.TestCase):
    def test_v322_extraction_yields_ground_truth_pairs(self) -> None:
        events = []
        with (_REAL_RUN / "events.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        edges = probe_first_step_edges(events)
        self.assertGreater(len(edges), 1000)
        state_frames = state_frame_index(events)
        roots = collect_probe_roots(_REAL_RUN.name, edges, state_frames)
        self.assertGreater(len(roots), 100)
        cache = RunFrameCache(_REAL_RUN / "frames", capacity=64)
        eligible = [
            root
            for root in sorted(roots, key=lambda item: item.sort_key)
            if any(arm.action.value != "noop" for arm in root.arms)
        ][:2]
        records, stats = label_probe_roots(eligible, cache.get)
        self.assertEqual(stats["roots"], 2)
        self.assertTrue(records)
        for record in records:
            self.assertEqual(record["columns"], GRID_COLUMNS)
            self.assertEqual(record["rows"], GRID_ROWS)
            for arm in record["arms"]:
                self.assertIn(arm["status"], ("labeled", "censored"))


if __name__ == "__main__":
    unittest.main()
