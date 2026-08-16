from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .run_logging import sha256_file

PARTITION_CATEGORIES = ("training", "development", "withheld_lolo1", "sequel")
FROZEN_PARTITIONS = frozenset({"withheld_lolo1", "sequel"})
REWARD_TRACKS = ("strict", "assisted")
UPDATE_AUTHORITIES = ("trainable", "frozen")
ARTIFACT_CLASSES = ("neural", "spatial", "entity", "relational")
PARTITION_INTENTS = ("frozen_evaluation", "training")
BASELINE_VERSIONS = ("v318", "v319", "v320", "v321")
BASELINE_ARTIFACT_ROLES = (
    "checkpoint",
    "behavior_checkpoint",
    "native_host",
    "core",
    "rom",
)
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "evaluation-partitions.json"
)

_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GAMES = ("lolo1", "lolo2")
_ROOM3_GAME = "lolo1"
_ROOM3 = 3


class PartitionUpdateError(RuntimeError):
    """Raised loudly when an update is attempted from a frozen partition.

    The exception carries the exact ``partition_update_rejected`` telemetry
    payload so callers can both log the rejection and fail closed.
    """

    def __init__(self, message: str, event: Dict[str, Any]) -> None:
        super().__init__(message)
        self.event = dict(event)


def canonical_signature(value: Any) -> str:
    """Deterministic, content-derived signature of a JSON-serializable value."""

    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _required_digest(value: Any, field: str) -> str:
    text = _required_text(value, field)
    if not _HEX_DIGEST.fullmatch(text):
        raise ValueError(f"{field} must be a 64-character sha256 hex digest")
    return text


def _room_tuple(value: Any, field: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    rooms = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{field} entries must be positive integers")
        rooms.append(item)
    if len(set(rooms)) != len(rooms):
        raise ValueError(f"{field} entries must be unique")
    return tuple(sorted(rooms))


def _reward_track_tuple(value: Any, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    tracks = tuple(str(item) for item in value)
    for track in tracks:
        if track not in REWARD_TRACKS:
            raise ValueError(
                f"{field} entries must be one of {sorted(REWARD_TRACKS)}"
            )
    if len(set(tracks)) != len(tracks):
        raise ValueError(f"{field} entries must be unique")
    return tracks


@dataclass(frozen=True)
class RoomPartition:
    """One immutable room-allocation category from the split manifest."""

    category: str
    game: str
    rooms: Tuple[int, ...]
    reward_tracks: Tuple[str, ...]
    update_authority: str

    @property
    def frozen(self) -> bool:
        return self.update_authority == "frozen"

    def allows_reward_track(self, reward_track: str) -> bool:
        return reward_track in self.reward_tracks


@dataclass(frozen=True)
class ArtifactDigest:
    """Content digest of one persistent learned artifact (or its absence)."""

    artifact_class: str
    path: Optional[str]
    file_sha256: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_class": self.artifact_class,
            "path": self.path,
            "file_sha256": self.file_sha256,
        }


@dataclass(frozen=True)
class PartitionManifest:
    """Parsed, validated, immutable room-allocation manifest."""

    manifest_id: str
    created_at: str
    source_path: Path
    file_sha256: str
    content_signature: str
    partitions: Tuple[RoomPartition, ...]
    baseline_versions: Tuple[str, ...]
    baseline_artifacts: Tuple[Tuple[str, str], ...]

    @classmethod
    def load(cls, path: Path) -> "PartitionManifest":
        source = Path(path).expanduser().resolve()
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("partition manifest must be an object")
        if value.get("version") != 1:
            raise ValueError("partition manifest version must be 1")
        if value.get("immutable") is not True:
            raise ValueError(
                "partition manifest must declare itself immutable"
            )
        manifest_id = _required_text(value.get("manifest_id"), "manifest_id")
        created_at = _required_text(value.get("created_at"), "created_at")
        partitions_value = value.get("partitions")
        if not isinstance(partitions_value, dict):
            raise ValueError("partitions must be an object")
        if tuple(sorted(partitions_value)) != tuple(
            sorted(PARTITION_CATEGORIES)
        ):
            raise ValueError(
                "partitions must define exactly the categories "
                f"{sorted(PARTITION_CATEGORIES)}"
            )
        partitions = []
        assigned: set[Tuple[str, int]] = set()
        for category in PARTITION_CATEGORIES:
            entry = partitions_value[category]
            if not isinstance(entry, dict):
                raise ValueError(f"partition {category} must be an object")
            game = _required_text(entry.get("game"), f"{category}.game")
            if game not in _GAMES:
                raise ValueError(
                    f"{category}.game must be one of {sorted(_GAMES)}"
                )
            rooms = _room_tuple(entry.get("rooms"), f"{category}.rooms")
            reward_tracks = _reward_track_tuple(
                entry.get("reward_tracks"), f"{category}.reward_tracks"
            )
            authority = _required_text(
                entry.get("update_authority"),
                f"{category}.update_authority",
            )
            if authority not in UPDATE_AUTHORITIES:
                raise ValueError(
                    f"{category}.update_authority must be one of "
                    f"{sorted(UPDATE_AUTHORITIES)}"
                )
            if category in FROZEN_PARTITIONS:
                if authority != "frozen":
                    raise ValueError(f"{category} must be frozen")
                if reward_tracks != ("strict",):
                    raise ValueError(
                        f"{category} must be strict-track only"
                    )
            elif authority != "trainable":
                raise ValueError(f"{category} must be trainable")
            if category == "withheld_lolo1" and game != "lolo1":
                raise ValueError("withheld_lolo1 must allocate lolo1 rooms")
            if category == "sequel" and game != "lolo2":
                raise ValueError("sequel must allocate lolo2 rooms")
            for room in rooms:
                key = (game, room)
                if key in assigned:
                    raise ValueError(
                        f"{game} room {room} is assigned to two partitions"
                    )
                assigned.add(key)
            if (
                category == "withheld_lolo1"
                and game == _ROOM3_GAME
                and _ROOM3 in rooms
            ):
                raise ValueError(
                    "lolo1 room 3 influenced engineering and must not be "
                    "withheld"
                )
            partitions.append(
                RoomPartition(
                    category=category,
                    game=game,
                    rooms=rooms,
                    reward_tracks=reward_tracks,
                    update_authority=authority,
                )
            )
        baseline = value.get("frozen_baseline")
        if not isinstance(baseline, dict):
            raise ValueError("frozen_baseline must be an object")
        artifacts_value = baseline.get("artifacts")
        if not isinstance(artifacts_value, dict) or tuple(
            sorted(artifacts_value)
        ) != tuple(sorted(BASELINE_ARTIFACT_ROLES)):
            raise ValueError(
                "frozen_baseline.artifacts must record exactly the roles "
                f"{sorted(BASELINE_ARTIFACT_ROLES)}"
            )
        baseline_artifacts = []
        for role in BASELINE_ARTIFACT_ROLES:
            entry = artifacts_value[role]
            if not isinstance(entry, dict):
                raise ValueError(f"frozen_baseline artifact {role} must be an object")
            _required_text(entry.get("name"), f"{role}.name")
            digest = _required_digest(
                entry.get("file_sha256"), f"{role}.file_sha256"
            )
            if role in ("checkpoint", "behavior_checkpoint"):
                _required_digest(
                    entry.get("parameter_sha256"), f"{role}.parameter_sha256"
                )
            baseline_artifacts.append((role, digest))
        configurations = baseline.get("planning_configurations")
        if not isinstance(configurations, dict) or tuple(
            sorted(configurations)
        ) != tuple(sorted(BASELINE_VERSIONS)):
            raise ValueError(
                "frozen_baseline.planning_configurations must record "
                f"exactly the versions {sorted(BASELINE_VERSIONS)}"
            )
        for version in BASELINE_VERSIONS:
            entry = configurations[version]
            if not isinstance(entry, dict):
                raise ValueError(
                    f"planning configuration {version} must be an object"
                )
            _required_digest(
                entry.get("planning_config_sha256"),
                f"{version}.planning_config_sha256",
            )
        return cls(
            manifest_id=manifest_id,
            created_at=created_at,
            source_path=source,
            file_sha256=sha256_file(source),
            content_signature=canonical_signature(value),
            partitions=tuple(partitions),
            baseline_versions=BASELINE_VERSIONS,
            baseline_artifacts=tuple(baseline_artifacts),
        )

    def partition(self, category: str) -> RoomPartition:
        for entry in self.partitions:
            if entry.category == category:
                return entry
        raise ValueError(f"unknown partition category: {category}")

    def partition_for_room(self, game: str, room: int) -> RoomPartition:
        if game not in _GAMES:
            raise ValueError(f"game must be one of {sorted(_GAMES)}")
        for entry in self.partitions:
            if entry.game == game and room in entry.rooms:
                return entry
        raise ValueError(
            f"{game} room {room} is not assigned to any partition"
        )

    def loaded_event(
        self, partition: RoomPartition, room: int, reward_track: str
    ) -> Dict[str, Any]:
        """Telemetry payload for `evaluation_partition_loaded`."""

        return {
            "event": "evaluation_partition_loaded",
            "partition_manifest_id": self.manifest_id,
            "partition_manifest_sha256": self.file_sha256,
            "partition_content_signature": self.content_signature,
            "evaluation_partition": partition.category,
            "partition_game": partition.game,
            "partition_room": room,
            "partition_reward_track": reward_track,
            "partition_update_authority": partition.update_authority,
        }

    def run_manifest_fields(
        self, partition: RoomPartition, room: int, reward_track: str
    ) -> Dict[str, Any]:
        """Explicit partition provenance fields for a run's manifest.json."""

        if not partition.allows_reward_track(reward_track):
            raise ValueError(
                f"partition {partition.category} does not permit the "
                f"{reward_track!r} reward track"
            )
        return {
            "evaluation_partition": partition.category,
            "partition_game": partition.game,
            "partition_room": room,
            "partition_reward_track": reward_track,
            "partition_update_authority": partition.update_authority,
            "partition_manifest_id": self.manifest_id,
            "partition_manifest_sha256": self.file_sha256,
        }

    def _rejection_event(
        self, partition: RoomPartition, operation: str, reason: str
    ) -> Dict[str, Any]:
        return {
            "event": "partition_update_rejected",
            "partition_manifest_id": self.manifest_id,
            "partition_manifest_sha256": self.file_sha256,
            "evaluation_partition": partition.category,
            "partition_game": partition.game,
            "partition_update_authority": partition.update_authority,
            "rejected_operation": operation,
            "rejection_reason": reason,
        }

    def require_update_authority(
        self, partition: RoomPartition, operation: str
    ) -> None:
        """Reject any persistent-artifact update from a frozen partition."""

        if not partition.frozen:
            return
        reason = (
            f"partition {partition.category} is frozen; persistent "
            "artifacts must not be updated from its runs"
        )
        raise PartitionUpdateError(
            reason, self._rejection_event(partition, operation, reason)
        )

    def authorize_training_write(
        self,
        game: str,
        room: int,
        reward_track: str,
        operation: str = "training_artifact_write",
    ) -> RoomPartition:
        """Authorize writing a training artifact from a run in one room."""

        if reward_track not in REWARD_TRACKS:
            raise ValueError(
                f"reward track must be one of {sorted(REWARD_TRACKS)}"
            )
        partition = self.partition_for_room(game, room)
        self.require_update_authority(partition, operation)
        if not partition.allows_reward_track(reward_track):
            reason = (
                f"partition {partition.category} does not permit the "
                f"{reward_track!r} reward track"
            )
            raise PartitionUpdateError(
                reason,
                self._rejection_event(partition, operation, reason),
            )
        return partition

    def authorize_corpus_import(
        self,
        partition: RoomPartition,
        corpus_reward_track: str,
        event_reward_track: str,
        test_only_override: bool = False,
    ) -> None:
        """Keep strict and assisted provenance separated per partition.

        Training or assisted events cannot enter a strict withheld or sequel
        corpus unless a test explicitly overrides the separation.
        """

        for field, track in (
            ("corpus_reward_track", corpus_reward_track),
            ("event_reward_track", event_reward_track),
        ):
            if track not in REWARD_TRACKS:
                raise ValueError(
                    f"{field} must be one of {sorted(REWARD_TRACKS)}"
                )
        if test_only_override:
            return
        operation = "corpus_import"
        if event_reward_track != corpus_reward_track:
            reason = (
                f"cannot import {event_reward_track!r} events into the "
                f"{corpus_reward_track!r} corpus of partition "
                f"{partition.category} without a test-only override"
            )
            raise PartitionUpdateError(
                reason,
                self._rejection_event(partition, operation, reason),
            )
        if not partition.allows_reward_track(event_reward_track):
            reason = (
                f"partition {partition.category} does not permit "
                f"{event_reward_track!r} events"
            )
            raise PartitionUpdateError(
                reason,
                self._rejection_event(partition, operation, reason),
            )


def audit_persistent_artifacts(
    inventory: Mapping[str, Optional[Path]],
) -> Tuple[ArtifactDigest, ...]:
    """Digest every persistent artifact class, recording absences explicitly.

    The inventory must declare every artifact class so a future class (for
    example a relational model) cannot silently escape the freeze audit.
    """

    if tuple(sorted(inventory)) != tuple(sorted(ARTIFACT_CLASSES)):
        raise ValueError(
            "the artifact inventory must declare exactly the classes "
            f"{sorted(ARTIFACT_CLASSES)}"
        )
    digests = []
    for artifact_class in ARTIFACT_CLASSES:
        path = inventory[artifact_class]
        if path is None:
            digests.append(
                ArtifactDigest(
                    artifact_class=artifact_class,
                    path=None,
                    file_sha256=None,
                )
            )
            continue
        resolved = Path(path).expanduser().resolve()
        digests.append(
            ArtifactDigest(
                artifact_class=artifact_class,
                path=str(resolved),
                file_sha256=(
                    sha256_file(resolved) if resolved.is_file() else None
                ),
            )
        )
    return tuple(digests)


def digest_audit_event(
    digests: Tuple[ArtifactDigest, ...], phase: str
) -> Dict[str, Any]:
    """Telemetry payload for `persistent_artifact_digest_audited`."""

    artifacts = [digest.to_dict() for digest in digests]
    return {
        "event": "persistent_artifact_digest_audited",
        "audit_phase": phase,
        "artifact_classes": [digest.artifact_class for digest in digests],
        "artifacts": artifacts,
        "audit_signature": canonical_signature(artifacts),
    }


def verify_frozen_digests(
    before: Tuple[ArtifactDigest, ...],
    after: Tuple[ArtifactDigest, ...],
    partition_category: str,
) -> None:
    """Fail loudly if a frozen run changed any persistent artifact digest."""

    if tuple(sorted(entry.artifact_class for entry in before)) != tuple(
        sorted(entry.artifact_class for entry in after)
    ):
        raise ValueError(
            "before/after audits must cover the same artifact classes"
        )
    closing = {entry.artifact_class: entry for entry in after}
    changed = sorted(
        entry.artifact_class
        for entry in before
        if closing[entry.artifact_class] != entry
    )
    if not changed:
        return
    reason = (
        f"frozen run in partition {partition_category} changed persistent "
        f"artifact digests: {changed}"
    )
    raise PartitionUpdateError(
        reason,
        {
            "event": "partition_update_rejected",
            "evaluation_partition": partition_category,
            "partition_update_authority": "frozen",
            "rejected_operation": "persistent_artifact_mutation",
            "changed_artifact_classes": changed,
            "rejection_reason": reason,
        },
    )


@dataclass(frozen=True)
class CyclePartitionBinding:
    """The evaluation-partition section of a research-cycle plan."""

    manifest_path: Path
    game: str
    room: int
    intent: str
    reward_track: str
    audited_artifacts: Tuple[Tuple[str, Optional[str]], ...]

    @classmethod
    def from_dict(
        cls, value: Any, base_directory: Path
    ) -> "CyclePartitionBinding":
        if not isinstance(value, dict):
            raise ValueError("evaluation_partition must be an object")
        manifest_path = Path(
            _required_text(value.get("manifest_path"), "manifest_path")
        ).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = base_directory / manifest_path
        game = _required_text(value.get("game"), "game")
        room_value = value.get("room")
        if (
            isinstance(room_value, bool)
            or not isinstance(room_value, int)
            or room_value <= 0
        ):
            raise ValueError("room must be a positive integer")
        intent = _required_text(value.get("intent"), "intent")
        if intent not in PARTITION_INTENTS:
            raise ValueError(
                f"intent must be one of {sorted(PARTITION_INTENTS)}"
            )
        reward_track = _required_text(
            value.get("reward_track"), "reward_track"
        )
        if reward_track not in REWARD_TRACKS:
            raise ValueError(
                f"reward_track must be one of {sorted(REWARD_TRACKS)}"
            )
        artifacts_value = value.get("audited_artifacts")
        if not isinstance(artifacts_value, dict) or tuple(
            sorted(artifacts_value)
        ) != tuple(sorted(ARTIFACT_CLASSES)):
            raise ValueError(
                "audited_artifacts must declare exactly the classes "
                f"{sorted(ARTIFACT_CLASSES)}"
            )
        audited = []
        for artifact_class in ARTIFACT_CLASSES:
            entry = artifacts_value[artifact_class]
            if entry is None:
                audited.append((artifact_class, None))
                continue
            path = Path(
                _required_text(entry, f"audited_artifacts.{artifact_class}")
            ).expanduser()
            if not path.is_absolute():
                path = base_directory / path
            audited.append((artifact_class, str(path.resolve())))
        return cls(
            manifest_path=manifest_path.resolve(),
            game=game,
            room=room_value,
            intent=intent,
            reward_track=reward_track,
            audited_artifacts=tuple(audited),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "game": self.game,
            "room": self.room,
            "intent": self.intent,
            "reward_track": self.reward_track,
            "audited_artifacts": {
                artifact_class: path
                for artifact_class, path in self.audited_artifacts
            },
        }

    def artifact_inventory(self) -> Dict[str, Optional[Path]]:
        return {
            artifact_class: None if path is None else Path(path)
            for artifact_class, path in self.audited_artifacts
        }


@dataclass(frozen=True)
class CyclePartitionContext:
    """Loaded partition state guarding one research cycle."""

    manifest: PartitionManifest
    partition: RoomPartition
    binding: CyclePartitionBinding
    loaded_event: Dict[str, Any]
    opening_audit: Tuple[ArtifactDigest, ...]


def prepare_cycle_partition(
    binding: CyclePartitionBinding,
) -> CyclePartitionContext:
    """Load the manifest and authorize one cycle before any side effects."""

    manifest = PartitionManifest.load(binding.manifest_path)
    partition = manifest.partition_for_room(binding.game, binding.room)
    if binding.intent == "training":
        manifest.authorize_training_write(
            binding.game,
            binding.room,
            binding.reward_track,
            operation="training_cycle",
        )
    elif not partition.allows_reward_track(binding.reward_track):
        raise ValueError(
            f"partition {partition.category} does not permit the "
            f"{binding.reward_track!r} reward track"
        )
    return CyclePartitionContext(
        manifest=manifest,
        partition=partition,
        binding=binding,
        loaded_event=manifest.loaded_event(
            partition, binding.room, binding.reward_track
        ),
        opening_audit=audit_persistent_artifacts(
            binding.artifact_inventory()
        ),
    )
