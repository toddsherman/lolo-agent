from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

from lolo_agent.partitions import (
    ARTIFACT_CLASSES,
    BASELINE_VERSIONS,
    DEFAULT_MANIFEST_PATH,
    PARTITION_CATEGORIES,
    PartitionManifest,
    PartitionUpdateError,
    audit_persistent_artifacts,
    canonical_signature,
    digest_audit_event,
    verify_frozen_digests,
)
from lolo_agent.research_cycle import run_cycle

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _manifest_value() -> Dict[str, Any]:
    """A small, self-contained, valid manifest fixture."""

    return {
        "version": 1,
        "manifest_id": "fixture-partitions-v1",
        "created_at": "2026-08-16T00:00:00Z",
        "immutable": True,
        "partitions": {
            "training": {
                "game": "lolo1",
                "rooms": [4, 5, 6],
                "reward_tracks": ["strict", "assisted"],
                "update_authority": "trainable",
            },
            "development": {
                "game": "lolo1",
                "rooms": [1, 2, 3],
                "reward_tracks": ["strict", "assisted"],
                "update_authority": "trainable",
            },
            "withheld_lolo1": {
                "game": "lolo1",
                "rooms": [45, 50],
                "reward_tracks": ["strict"],
                "update_authority": "frozen",
            },
            "sequel": {
                "game": "lolo2",
                "rooms": [1, 2],
                "reward_tracks": ["strict"],
                "update_authority": "frozen",
            },
        },
        "frozen_baseline": {
            "artifacts": {
                "checkpoint": {
                    "name": "fixture-checkpoint.pt",
                    "file_sha256": _DIGEST_A,
                    "parameter_sha256": _DIGEST_B,
                },
                "behavior_checkpoint": {
                    "name": "fixture-behavior.json",
                    "file_sha256": _DIGEST_A,
                    "parameter_sha256": _DIGEST_B,
                },
                "native_host": {
                    "name": "fixture-host",
                    "file_sha256": _DIGEST_A,
                },
                "core": {
                    "name": "fixture-core.dylib",
                    "file_sha256": _DIGEST_A,
                },
                "rom": {
                    "name": "fixture-rom.nes",
                    "file_sha256": _DIGEST_A,
                },
            },
            "planning_configurations": {
                version: {"planning_config_sha256": _DIGEST_A}
                for version in ("v318", "v319", "v320", "v321")
            },
        },
    }


def _write_manifest(root: Path, value: Dict[str, Any]) -> Path:
    path = root / "partitions.json"
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


class PartitionManifestParsingTests(unittest.TestCase):
    def test_fixture_manifest_parses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_manifest(Path(temporary), _manifest_value())
            manifest = PartitionManifest.load(path)
        self.assertEqual(manifest.manifest_id, "fixture-partitions-v1")
        self.assertEqual(
            tuple(entry.category for entry in manifest.partitions),
            PARTITION_CATEGORIES,
        )
        self.assertEqual(len(manifest.file_sha256), 64)
        self.assertEqual(len(manifest.content_signature), 64)

    def test_committed_manifest_is_valid_and_room3_is_development(self) -> None:
        manifest = PartitionManifest.load(DEFAULT_MANIFEST_PATH)
        development = manifest.partition_for_room("lolo1", 3)
        self.assertEqual(development.category, "development")
        withheld = manifest.partition("withheld_lolo1")
        self.assertNotIn(3, withheld.rooms)
        self.assertTrue(withheld.frozen)
        self.assertEqual(withheld.reward_tracks, ("strict",))
        sequel = manifest.partition("sequel")
        self.assertEqual(sequel.game, "lolo2")
        self.assertTrue(sequel.frozen)
        self.assertEqual(manifest.baseline_versions, BASELINE_VERSIONS)
        self.assertEqual(
            [role for role, _ in manifest.baseline_artifacts],
            ["checkpoint", "behavior_checkpoint", "native_host", "core", "rom"],
        )

    def test_committed_manifest_covers_every_lolo1_room_once(self) -> None:
        manifest = PartitionManifest.load(DEFAULT_MANIFEST_PATH)
        lolo1_rooms = [
            room
            for entry in manifest.partitions
            if entry.game == "lolo1"
            for room in entry.rooms
        ]
        self.assertEqual(sorted(lolo1_rooms), list(range(1, 51)))
        self.assertEqual(
            sorted(manifest.partition("sequel").rooms), list(range(1, 51))
        )

    def test_manifest_values_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_manifest(Path(temporary), _manifest_value())
            manifest = PartitionManifest.load(path)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            manifest.manifest_id = "other"  # type: ignore[misc]
        partition = manifest.partition("withheld_lolo1")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            partition.update_authority = "trainable"  # type: ignore[misc]

    def test_loader_rejects_a_mutable_manifest(self) -> None:
        value = _manifest_value()
        value["immutable"] = False
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_manifest(Path(temporary), value)
            with self.assertRaisesRegex(ValueError, "immutable"):
                PartitionManifest.load(path)

    def test_loader_rejects_withholding_room_three(self) -> None:
        value = _manifest_value()
        value["partitions"]["development"]["rooms"] = [1, 2]
        value["partitions"]["withheld_lolo1"]["rooms"] = [3, 45, 50]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_manifest(Path(temporary), value)
            with self.assertRaisesRegex(ValueError, "room 3"):
                PartitionManifest.load(path)

    def test_loader_rejects_overlapping_room_assignments(self) -> None:
        value = _manifest_value()
        value["partitions"]["training"]["rooms"] = [3, 4, 5]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_manifest(Path(temporary), value)
            with self.assertRaisesRegex(ValueError, "two partitions"):
                PartitionManifest.load(path)

    def test_loader_rejects_assisted_track_on_withheld_partition(self) -> None:
        value = _manifest_value()
        value["partitions"]["withheld_lolo1"]["reward_tracks"] = [
            "strict",
            "assisted",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_manifest(Path(temporary), value)
            with self.assertRaisesRegex(ValueError, "strict-track only"):
                PartitionManifest.load(path)

    def test_loader_rejects_trainable_sequel_partition(self) -> None:
        value = _manifest_value()
        value["partitions"]["sequel"]["update_authority"] = "trainable"
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_manifest(Path(temporary), value)
            with self.assertRaisesRegex(ValueError, "frozen"):
                PartitionManifest.load(path)

    def test_loader_requires_every_baseline_planning_configuration(self) -> None:
        value = _manifest_value()
        del value["frozen_baseline"]["planning_configurations"]["v321"]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_manifest(Path(temporary), value)
            with self.assertRaisesRegex(ValueError, "v321"):
                PartitionManifest.load(path)

    def test_loader_requires_every_baseline_artifact_digest(self) -> None:
        value = _manifest_value()
        del value["frozen_baseline"]["artifacts"]["rom"]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_manifest(Path(temporary), value)
            with self.assertRaisesRegex(ValueError, "rom"):
                PartitionManifest.load(path)

    def test_content_signature_is_deterministic(self) -> None:
        value = _manifest_value()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = PartitionManifest.load(_write_manifest(root, value))
            reordered = {key: value[key] for key in reversed(list(value))}
            other = root / "reordered.json"
            other.write_text(json.dumps(reordered), encoding="utf-8")
            second = PartitionManifest.load(other)
        self.assertEqual(first.content_signature, second.content_signature)
        self.assertNotEqual(first.file_sha256, second.file_sha256)


class PartitionAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.manifest = PartitionManifest.load(
            _write_manifest(Path(self._temporary.name), _manifest_value())
        )

    def test_loaded_event_records_partition_provenance(self) -> None:
        partition = self.manifest.partition_for_room("lolo1", 2)
        event = self.manifest.loaded_event(partition, 2, "strict")
        self.assertEqual(event["event"], "evaluation_partition_loaded")
        self.assertEqual(event["evaluation_partition"], "development")
        self.assertEqual(event["partition_room"], 2)
        self.assertEqual(event["partition_reward_track"], "strict")
        self.assertEqual(event["partition_update_authority"], "trainable")
        self.assertEqual(
            event["partition_manifest_sha256"], self.manifest.file_sha256
        )

    def test_run_manifest_fields_are_explicit(self) -> None:
        partition = self.manifest.partition_for_room("lolo1", 45)
        fields = self.manifest.run_manifest_fields(partition, 45, "strict")
        self.assertEqual(fields["evaluation_partition"], "withheld_lolo1")
        self.assertEqual(fields["partition_reward_track"], "strict")
        self.assertEqual(fields["partition_update_authority"], "frozen")
        with self.assertRaisesRegex(ValueError, "assisted"):
            self.manifest.run_manifest_fields(partition, 45, "assisted")

    def test_training_write_is_allowed_from_training_partition(self) -> None:
        partition = self.manifest.authorize_training_write(
            "lolo1", 4, "strict"
        )
        self.assertEqual(partition.category, "training")

    def test_training_write_is_rejected_from_withheld_partition(self) -> None:
        with self.assertRaises(PartitionUpdateError) as caught:
            self.manifest.authorize_training_write("lolo1", 50, "strict")
        event = caught.exception.event
        self.assertEqual(event["event"], "partition_update_rejected")
        self.assertEqual(event["evaluation_partition"], "withheld_lolo1")
        self.assertEqual(
            event["rejected_operation"], "training_artifact_write"
        )

    def test_training_write_is_rejected_from_sequel_partition(self) -> None:
        with self.assertRaises(PartitionUpdateError) as caught:
            self.manifest.authorize_training_write("lolo2", 1, "strict")
        self.assertEqual(
            caught.exception.event["evaluation_partition"], "sequel"
        )

    def test_unassigned_room_cannot_authorize_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "not assigned"):
            self.manifest.authorize_training_write("lolo1", 49, "strict")

    def test_strict_withheld_corpus_rejects_assisted_events(self) -> None:
        partition = self.manifest.partition("withheld_lolo1")
        with self.assertRaises(PartitionUpdateError) as caught:
            self.manifest.authorize_corpus_import(
                partition, "strict", "assisted"
            )
        self.assertEqual(
            caught.exception.event["rejected_operation"], "corpus_import"
        )

    def test_test_only_override_admits_assisted_events(self) -> None:
        partition = self.manifest.partition("withheld_lolo1")
        self.manifest.authorize_corpus_import(
            partition, "strict", "assisted", test_only_override=True
        )

    def test_matching_strict_import_is_permitted(self) -> None:
        partition = self.manifest.partition("withheld_lolo1")
        self.manifest.authorize_corpus_import(partition, "strict", "strict")


class DigestAuditTests(unittest.TestCase):
    def _inventory(
        self, root: Path, relational: Optional[Path] = None
    ) -> Dict[str, Optional[Path]]:
        neural = root / "neural.pt"
        spatial = root / "spatial.json"
        entity = root / "entity.json"
        neural.write_bytes(b"neural-parameters")
        spatial.write_text('{"kind": "spatial"}', encoding="utf-8")
        entity.write_text('{"kind": "entity"}', encoding="utf-8")
        return {
            "neural": neural,
            "spatial": spatial,
            "entity": entity,
            "relational": relational,
        }

    def test_audit_covers_every_artifact_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            digests = audit_persistent_artifacts(
                self._inventory(Path(temporary))
            )
        self.assertEqual(
            tuple(entry.artifact_class for entry in digests),
            ARTIFACT_CLASSES,
        )
        by_class = {entry.artifact_class: entry for entry in digests}
        self.assertIsNone(by_class["relational"].file_sha256)
        for artifact_class in ("neural", "spatial", "entity"):
            self.assertEqual(len(by_class[artifact_class].file_sha256), 64)

    def test_audit_rejects_an_incomplete_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inventory = self._inventory(Path(temporary))
            del inventory["relational"]
            with self.assertRaisesRegex(ValueError, "relational"):
                audit_persistent_artifacts(inventory)

    def test_audit_event_payload_is_signed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            digests = audit_persistent_artifacts(
                self._inventory(Path(temporary))
            )
        event = digest_audit_event(digests, phase="cycle_start")
        self.assertEqual(
            event["event"], "persistent_artifact_digest_audited"
        )
        self.assertEqual(event["audit_phase"], "cycle_start")
        self.assertEqual(
            event["artifact_classes"], list(ARTIFACT_CLASSES)
        )
        self.assertEqual(
            event["audit_signature"],
            canonical_signature(event["artifacts"]),
        )

    def test_unchanged_digests_verify_and_mutations_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = self._inventory(root)
            before = audit_persistent_artifacts(inventory)
            verify_frozen_digests(
                before,
                audit_persistent_artifacts(inventory),
                "withheld_lolo1",
            )
            (root / "entity.json").write_text(
                '{"kind": "entity", "updated": true}', encoding="utf-8"
            )
            after = audit_persistent_artifacts(inventory)
            with self.assertRaises(PartitionUpdateError) as caught:
                verify_frozen_digests(before, after, "withheld_lolo1")
        event = caught.exception.event
        self.assertEqual(event["event"], "partition_update_rejected")
        self.assertEqual(
            event["rejected_operation"], "persistent_artifact_mutation"
        )
        self.assertEqual(event["changed_artifact_classes"], ["entity"])

    def test_newly_created_relational_artifact_breaks_the_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relational = root / "relational.json"
            inventory = self._inventory(root, relational=relational)
            before = audit_persistent_artifacts(inventory)
            relational.write_text('{"kind": "relational"}', encoding="utf-8")
            after = audit_persistent_artifacts(inventory)
            with self.assertRaises(PartitionUpdateError) as caught:
                verify_frozen_digests(before, after, "sequel")
        self.assertEqual(
            caught.exception.event["changed_artifact_classes"],
            ["relational"],
        )


class ResearchCyclePartitionTests(unittest.TestCase):
    def _plan_value(
        self,
        root: Path,
        cycle_id: str,
        manifest_path: Path,
        room: int,
        intent: str,
        command: list[str],
        game: str = "lolo1",
        reward_track: str = "strict",
    ) -> Path:
        artifacts = {
            "neural": str(root / "neural.pt"),
            "spatial": str(root / "spatial.json"),
            "entity": str(root / "entity.json"),
            "relational": None,
        }
        (root / "neural.pt").write_bytes(b"neural-parameters")
        (root / "spatial.json").write_text(
            '{"kind": "spatial"}', encoding="utf-8"
        )
        (root / "entity.json").write_text(
            '{"kind": "entity"}', encoding="utf-8"
        )
        value = {
            "version": 1,
            "cycle_id": cycle_id,
            "hypothesis": "A frozen smoke run leaves digests unchanged",
            "decision_question": "Did any persistent digest change?",
            "expected_evidence": ["Identical opening and closing audits"],
            "stop_conditions": ["The command exits"],
            "command": command,
            "working_directory": str(root),
            "telemetry_path": None,
            "prior_cycle_id": None,
            "budgets": {
                "max_wall_seconds": 30,
                "max_events": None,
                "hourly_rate_usd": 0,
                "max_cycle_cost_usd": 0,
                "max_campaign_cost_usd": 0,
            },
            "evaluation_partition": {
                "manifest_path": str(manifest_path),
                "game": game,
                "room": room,
                "intent": intent,
                "reward_track": reward_track,
                "audited_artifacts": artifacts,
            },
        }
        path = root / f"{cycle_id}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_frozen_smoke_cycle_verifies_digests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_manifest(root, _manifest_value())
            plan = self._plan_value(
                root,
                "frozen-smoke",
                manifest_path,
                room=50,
                intent="frozen_evaluation",
                command=[sys.executable, "-c", "pass"],
            )
            report = run_cycle(plan, root / "campaign")
        partition = report["evaluation_partition"]
        self.assertTrue(partition["frozen_digests_verified"])
        self.assertEqual(
            partition["loaded"]["event"], "evaluation_partition_loaded"
        )
        self.assertEqual(
            partition["loaded"]["evaluation_partition"], "withheld_lolo1"
        )
        self.assertEqual(
            partition["opening_audit"]["event"],
            "persistent_artifact_digest_audited",
        )
        self.assertEqual(
            partition["opening_audit"]["audit_signature"],
            partition["closing_audit"]["audit_signature"],
        )

    def test_training_cycle_from_withheld_room_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_manifest(root, _manifest_value())
            plan = self._plan_value(
                root,
                "withheld-training",
                manifest_path,
                room=50,
                intent="training",
                command=[sys.executable, "-c", "pass"],
            )
            campaign = root / "campaign"
            with self.assertRaises(PartitionUpdateError) as caught:
                run_cycle(plan, campaign)
            self.assertFalse(
                (campaign / "cycles" / "withheld-training").exists()
            )
        self.assertEqual(
            caught.exception.event["rejected_operation"], "training_cycle"
        )

    def test_training_cycle_from_sequel_room_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_manifest(root, _manifest_value())
            plan = self._plan_value(
                root,
                "sequel-training",
                manifest_path,
                room=1,
                intent="training",
                command=[sys.executable, "-c", "pass"],
                game="lolo2",
            )
            with self.assertRaises(PartitionUpdateError):
                run_cycle(plan, root / "campaign")

    def test_assisted_evaluation_on_withheld_room_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_manifest(root, _manifest_value())
            plan = self._plan_value(
                root,
                "assisted-withheld",
                manifest_path,
                room=50,
                intent="frozen_evaluation",
                command=[sys.executable, "-c", "pass"],
                reward_track="assisted",
            )
            with self.assertRaisesRegex(ValueError, "assisted"):
                run_cycle(plan, root / "campaign")

    def test_frozen_cycle_that_mutates_an_artifact_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_manifest(root, _manifest_value())
            entity_path = root / "entity.json"
            script = (
                "import pathlib; "
                f"pathlib.Path({str(entity_path)!r}).write_text("
                "'{\"kind\": \"entity\", \"updated\": true}')"
            )
            plan = self._plan_value(
                root,
                "frozen-mutation",
                manifest_path,
                room=50,
                intent="frozen_evaluation",
                command=[sys.executable, "-c", script],
            )
            campaign = root / "campaign"
            with self.assertRaises(PartitionUpdateError) as caught:
                run_cycle(plan, campaign)
            report = json.loads(
                (campaign / "cycles" / "frozen-mutation" / "report.json")
                .read_text(encoding="utf-8")
            )
        self.assertEqual(
            caught.exception.event["changed_artifact_classes"], ["entity"]
        )
        partition = report["evaluation_partition"]
        self.assertFalse(partition["frozen_digests_verified"])
        self.assertEqual(
            partition["violation"]["event"], "partition_update_rejected"
        )

    def test_plans_without_a_partition_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = json.loads(
                self._plan_value(
                    root,
                    "no-partition",
                    _write_manifest(root, _manifest_value()),
                    room=50,
                    intent="frozen_evaluation",
                    command=[sys.executable, "-c", "pass"],
                ).read_text(encoding="utf-8")
            )
            del value["evaluation_partition"]
            value["cycle_id"] = "plain"
            plan_path = root / "plain.json"
            plan_path.write_text(json.dumps(value), encoding="utf-8")
            report = run_cycle(plan_path, root / "campaign")
        self.assertNotIn("evaluation_partition", report)


if __name__ == "__main__":
    unittest.main()
