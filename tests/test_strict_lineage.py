from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional, Tuple

from lolo_agent.entity_behavior import AnonymousEntityBehaviorModel
from lolo_agent.strict_lineage import (
    PackageGraph,
    StrictLineageReport,
    audit_checkpoint_json,
    audit_checkpoint_metadata,
    lint_strict_lineage,
    trace_module_lineage,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "lolo_agent"

_FIXTURE_SOURCES: Dict[str, str] = {
    "__init__.py": "",
    "goal_prior.py": (
        "HEART_PROTOTYPE = (1, 2, 3)\n"
        "\n"
        "\n"
        "class PixelHeartGoalPrior:\n"
        "    def detect_player(self, frame):\n"
        "        return None\n"
        "\n"
        "    def player_pixel_mask(self, frame, slot):\n"
        "        return frozenset()\n"
    ),
    "clean_leaf.py": "VALUE = 7\n",
    "clean_mid.py": (
        "from .clean_leaf import VALUE\n"
        "\n"
        "DOUBLE = VALUE * 2\n"
    ),
    "middle.py": (
        "from .goal_prior import PixelHeartGoalPrior\n"
        "\n"
        "PRIOR_CLASS = PixelHeartGoalPrior\n"
    ),
    "coupled_entry.py": "from .middle import PRIOR_CLASS\n",
    "absolute_entry.py": "import pkg.goal_prior\n",
    "dynamic_user.py": (
        "def mask_of(source):\n"
        '    return getattr(source, "player_pixel_mask", None)\n'
    ),
    "telemetry_reader.py": (
        'KEY = "human_prior_source_player_slot"\n'
        "\n"
        "\n"
        "def read(event):\n"
        "    return event.get(KEY)\n"
    ),
}


def _write_fixture_package(base: Path) -> Path:
    package_root = base / "pkg"
    package_root.mkdir()
    for name, source in _FIXTURE_SOURCES.items():
        (package_root / name).write_text(source, encoding="utf-8")
    return package_root


def _strict_metadata() -> Dict[str, object]:
    """Metadata shaped like the repository's strict `.pt` checkpoints."""

    return {
        "version": 8,
        "architecture": "unlabeled-spatial-token-dynamics",
        "persistent_inputs": ["pixels", "actions", "action_durations"],
        "excluded_inputs": [
            "RAM",
            "object_labels",
            "rewards",
            "level_annotations",
            "solutions",
        ],
        "actions": ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B"],
        "digest": "ab" * 32,
    }


_REAL_GRAPH: Optional[PackageGraph] = None


def _real_graph() -> PackageGraph:
    global _REAL_GRAPH
    if _REAL_GRAPH is None:
        _REAL_GRAPH = PackageGraph.scan(_PACKAGE_ROOT)
    return _REAL_GRAPH


class RealPackageLineageTests(unittest.TestCase):
    """The linter must report the current truth of the repository."""

    def test_neural_planner_is_detected_as_assisted_coupled(self) -> None:
        lineage = trace_module_lineage(
            "lolo_agent.neural_planner", _real_graph()
        )
        self.assertTrue(lineage.assisted)
        import_symbols = {
            finding.reference.symbol
            for finding in lineage.findings
            if finding.reference.kind == "import"
            and finding.chain == ("lolo_agent.neural_planner",)
        }
        self.assertIn(
            "lolo_agent.goal_prior.PixelHeartGoalPrior", import_symbols
        )
        definition_chains = {
            finding.chain
            for finding in lineage.findings
            if finding.reference.kind == "definition"
        }
        self.assertIn(
            ("lolo_agent.neural_planner", "lolo_agent.goal_prior"),
            definition_chains,
        )

    def test_partitions_is_clean(self) -> None:
        lineage = trace_module_lineage(
            "lolo_agent.partitions", _real_graph()
        )
        self.assertFalse(lineage.assisted)
        self.assertEqual(lineage.findings, ())

    def test_object_tracks_is_assisted_without_importing_goal_prior(
        self,
    ) -> None:
        graph = _real_graph()
        scan = graph.scan_for("lolo_agent.object_tracks")
        self.assertNotIn("lolo_agent.goal_prior", scan.imports)
        lineage = trace_module_lineage("lolo_agent.object_tracks", graph)
        self.assertTrue(lineage.assisted)
        symbols = {
            finding.reference.symbol
            for finding in lineage.findings
            if finding.reference.kind == "symbol"
            and finding.chain == ("lolo_agent.object_tracks",)
        }
        self.assertIn("player_pixel_mask", symbols)

    def test_goal_prior_reports_its_own_definitions(self) -> None:
        lineage = trace_module_lineage("lolo_agent.goal_prior", _real_graph())
        self.assertTrue(lineage.assisted)
        definitions = {
            finding.reference.symbol
            for finding in lineage.findings
            if finding.reference.kind == "definition"
            and finding.chain == ("lolo_agent.goal_prior",)
        }
        self.assertLessEqual(
            {
                "PixelHeartGoalPrior",
                "detect_player",
                "player_pixel_mask",
                "HEART_PROTOTYPE",
                "OPEN_CHEST_PROTOTYPE",
            },
            definitions,
        )


class FixturePackageLineageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.package_root = _write_fixture_package(Path(self._tempdir.name))
        self.graph = PackageGraph.scan(self.package_root)

    def test_transitive_chain_is_exact(self) -> None:
        lineage = trace_module_lineage("pkg.coupled_entry", self.graph)
        self.assertTrue(lineage.assisted)
        chains = {
            (finding.chain, finding.reference.kind)
            for finding in lineage.findings
        }
        self.assertIn(
            (("pkg.coupled_entry", "pkg.middle"), "import"), chains
        )
        self.assertIn(
            (
                ("pkg.coupled_entry", "pkg.middle", "pkg.goal_prior"),
                "definition",
            ),
            chains,
        )

    def test_absolute_import_of_assisted_module_is_decisive(self) -> None:
        lineage = trace_module_lineage("pkg.absolute_entry", self.graph)
        self.assertTrue(lineage.assisted)
        self.assertIn(
            ("pkg.absolute_entry",),
            {
                finding.chain
                for finding in lineage.findings
                if finding.reference.kind == "import"
                and finding.reference.symbol == "pkg.goal_prior"
            },
        )

    def test_clean_modules_have_no_findings(self) -> None:
        for module in ("pkg.clean_mid", "pkg.clean_leaf"):
            lineage = trace_module_lineage(module, self.graph)
            self.assertFalse(lineage.assisted, module)
            self.assertEqual(lineage.findings, (), module)

    def test_dynamic_string_access_is_decisive(self) -> None:
        lineage = trace_module_lineage("pkg.dynamic_user", self.graph)
        self.assertTrue(lineage.assisted)
        kinds = {finding.reference.kind for finding in lineage.findings}
        self.assertIn("dynamic_string", kinds)

    def test_telemetry_marker_is_advisory_only(self) -> None:
        lineage = trace_module_lineage("pkg.telemetry_reader", self.graph)
        self.assertFalse(lineage.assisted)
        references = [
            finding.reference for finding in lineage.findings
        ]
        self.assertTrue(references)
        for reference in references:
            self.assertEqual(reference.kind, "telemetry_marker")
            self.assertFalse(reference.decisive)


class CheckpointAuditTests(unittest.TestCase):
    def test_strict_pt_metadata_is_clean(self) -> None:
        audit = audit_checkpoint_metadata(
            _strict_metadata(), label="spatial-checkpoint"
        )
        self.assertEqual(audit.violations, ())
        self.assertFalse(audit.assisted)
        self.assertEqual(
            audit.declared_fields,
            ("persistent_inputs", "excluded_inputs"),
        )

    def test_anonymous_behavior_checkpoint_is_clean(self) -> None:
        payload = AnonymousEntityBehaviorModel().to_dict()
        audit = audit_checkpoint_metadata(
            payload, label="anonymous-behavior"
        )
        self.assertEqual(audit.violations, ())
        self.assertFalse(audit.assisted)

    def test_assisted_reward_track_is_a_violation(self) -> None:
        payload = _strict_metadata()
        payload["reward_track"] = "assisted"
        audit = audit_checkpoint_metadata(payload, label="bad-track")
        self.assertTrue(audit.assisted)
        self.assertEqual(
            [violation.location for violation in audit.violations],
            ["reward_track"],
        )

    def test_strict_reward_track_is_allowed(self) -> None:
        payload = _strict_metadata()
        payload["reward_track"] = "strict"
        audit = audit_checkpoint_metadata(payload, label="strict-track")
        self.assertEqual(audit.violations, ())
        self.assertEqual(
            audit.declared_fields,
            ("reward_track", "persistent_inputs", "excluded_inputs"),
        )

    def test_forbidden_and_unknown_persistent_inputs(self) -> None:
        payload = _strict_metadata()
        payload["persistent_inputs"] = [
            "pixels",
            "rewards",
            "supplied_room_maps",
        ]
        audit = audit_checkpoint_metadata(payload, label="bad-inputs")
        reasons = {
            violation.location: violation.reason
            for violation in audit.violations
        }
        self.assertEqual(
            reasons["persistent_inputs[1]"],
            "forbidden input declared persistent",
        )
        self.assertIn(
            "outside the strict allowlist",
            reasons["persistent_inputs[2]"],
        )

    def test_assisted_markers_in_fields_and_values(self) -> None:
        payload = _strict_metadata()
        payload["human_prior_remaining"] = 3
        payload["notes"] = "derived with goal_prior templates"
        audit = audit_checkpoint_metadata(payload, label="contaminated")
        locations = [
            violation.location for violation in audit.violations
        ]
        self.assertEqual(locations, ["human_prior_remaining", "notes"])

    def test_excluded_inputs_values_are_not_marker_scanned(self) -> None:
        payload = _strict_metadata()
        payload["excluded_inputs"] = ["supplied_heart_labels", "RAM"]
        audit = audit_checkpoint_metadata(payload, label="exclusions")
        self.assertEqual(audit.violations, ())

    def test_nested_marker_location_is_reported(self) -> None:
        payload = {
            "sources": [
                {"run_id": "run-a", "detail": "detect_player anchored"}
            ]
        }
        audit = audit_checkpoint_metadata(payload, label="nested")
        self.assertEqual(len(audit.violations), 1)
        self.assertEqual(
            audit.violations[0].location, "sources[0].detail"
        )

    def test_non_mapping_payload_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            audit_checkpoint_metadata(["pixels"], label="wrong-shape")


class LintEntryPointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.base = Path(self._tempdir.name)
        self.package_root = _write_fixture_package(self.base)
        self.clean_checkpoint = self.base / "clean-checkpoint.json"
        self.clean_checkpoint.write_text(
            json.dumps(_strict_metadata(), sort_keys=True),
            encoding="utf-8",
        )
        contaminated = _strict_metadata()
        contaminated["reward_track"] = "assisted"
        self.contaminated_checkpoint = self.base / "bad-checkpoint.json"
        self.contaminated_checkpoint.write_text(
            json.dumps(contaminated, sort_keys=True), encoding="utf-8"
        )

    def _paths(self) -> Tuple[Path, ...]:
        return (
            self.package_root / "coupled_entry.py",
            self.package_root / "clean_mid.py",
            self.contaminated_checkpoint,
            self.clean_checkpoint,
        )

    def test_report_is_deterministic_across_input_order(self) -> None:
        paths = self._paths()
        report = lint_strict_lineage(paths)
        reversed_report = lint_strict_lineage(tuple(reversed(paths)))
        self.assertEqual(report.to_dict(), reversed_report.to_dict())
        self.assertEqual(
            report.report_signature, reversed_report.report_signature
        )
        self.assertEqual(
            [entry.module for entry in report.modules],
            ["pkg.clean_mid", "pkg.coupled_entry"],
        )
        self.assertEqual(
            [entry.label for entry in report.checkpoints],
            sorted(
                str(path.resolve())
                for path in (
                    self.contaminated_checkpoint,
                    self.clean_checkpoint,
                )
            ),
        )

    def test_combined_verdict_and_event(self) -> None:
        report = lint_strict_lineage(self._paths())
        self.assertIsInstance(report, StrictLineageReport)
        self.assertTrue(report.assisted)
        event = report.to_event()
        self.assertEqual(event["event"], "strict_lineage_linted")
        self.assertTrue(event["assisted_lineage_detected"])
        self.assertEqual(event["assisted_modules"], ["pkg.coupled_entry"])
        self.assertEqual(event["module_count"], 2)
        self.assertEqual(event["checkpoint_count"], 2)
        self.assertEqual(event["checkpoint_violation_count"], 1)
        self.assertEqual(
            event["report_signature"], report.report_signature
        )

    def test_clean_targets_produce_clean_report(self) -> None:
        report = lint_strict_lineage(
            (self.package_root / "clean_mid.py", self.clean_checkpoint)
        )
        self.assertFalse(report.assisted)

    def test_checkpoint_json_audit_reports_label(self) -> None:
        audit = audit_checkpoint_json(self.contaminated_checkpoint)
        self.assertEqual(
            audit.label, str(self.contaminated_checkpoint.resolve())
        )
        self.assertTrue(audit.assisted)

    def test_pt_payloads_are_directed_to_metadata_audit(self) -> None:
        checkpoint = self.base / "model.pt"
        checkpoint.write_bytes(b"opaque")
        with self.assertRaises(ValueError) as raised:
            lint_strict_lineage((checkpoint,))
        self.assertIn("audit_checkpoint_metadata", str(raised.exception))

    def test_module_outside_a_package_is_rejected(self) -> None:
        stray = self.base / "stray.py"
        stray.write_text("VALUE = 1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            lint_strict_lineage((stray,))

    def test_lint_is_read_only(self) -> None:
        paths = self._paths()
        watched = sorted(self.package_root.glob("*.py")) + [
            self.clean_checkpoint,
            self.contaminated_checkpoint,
        ]
        before = {path: path.read_bytes() for path in watched}
        lint_strict_lineage(paths)
        after = {path: path.read_bytes() for path in watched}
        self.assertEqual(before, after)

    def test_real_package_mixture_matches_current_truth(self) -> None:
        report = lint_strict_lineage(
            (
                _PACKAGE_ROOT / "neural_planner.py",
                _PACKAGE_ROOT / "partitions.py",
            )
        )
        verdicts = {
            entry.module: entry.assisted for entry in report.modules
        }
        self.assertEqual(
            verdicts,
            {
                "lolo_agent.neural_planner": True,
                "lolo_agent.partitions": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
