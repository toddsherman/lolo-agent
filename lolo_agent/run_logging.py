from __future__ import annotations

import json
import os
import platform
import struct
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .environment import Action, PixelSaveStateEnv
from .pixels import Frame, signature_key


SCHEMA_VERSION = "1.0.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Action):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Counter):
        return {
            str(key.value if isinstance(key, Action) else key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"telemetry value is not JSON-safe: {type(value).__name__}")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def encode_png(frame: Frame) -> bytes:
    """Encode an 8-bit Frame without adding an imaging dependency."""

    color_types = {1: 0, 2: 4, 3: 2, 4: 6}
    if frame.channels not in color_types:
        raise ValueError(f"PNG logging does not support {frame.channels} channels")
    stride = frame.width * frame.channels
    scanlines = b"".join(
        b"\x00" + frame.pixels[offset : offset + stride]
        for offset in range(0, len(frame.pixels), stride)
    )
    header = struct.pack(">IIBBBBB", frame.width, frame.height, 8, color_types[frame.channels], 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(
        b"IDAT", zlib.compress(scanlines, level=6)
    ) + _png_chunk(b"IEND", b"")


class RunLogger:
    """Append-only event log with content-addressed visual observations."""

    def __init__(
        self,
        root: Path,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        store_frames: bool = True,
        fsync_interval: int = 100,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_id = run_id or f"run-{stamp}"
        self.run_dir = Path(root).expanduser().resolve() / self.run_id
        self.frames_dir = self.run_dir / "frames"
        self.states_dir = self.run_dir / "states"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        if store_frames:
            self.frames_dir.mkdir()
        self.store_frames = store_frames
        self.fsync_interval = max(1, fsync_interval)
        self.sequence = 0
        self.attempt = 0
        self._start_monotonic = time.monotonic()
        self._events_path = self.run_dir / "events.jsonl"
        self._events = self._events_path.open("a", encoding="utf-8", buffering=1)
        self._closed = False
        self._stored_frames: set[str] = set()
        self._stored_states: set[str] = set()
        self._event_counts: Counter[str] = Counter()
        self.manifest: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "running",
            "started_at": utc_now(),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "frame_storage": "content-addressed-png" if store_frames else "digest-only",
            "metadata": _json_value(metadata or {}),
        }
        self._write_manifest()
        self.log("run_started", manifest_schema=SCHEMA_VERSION)

    def _write_manifest(self) -> None:
        destination = self.run_dir / "manifest.json"
        temporary = self.run_dir / ".manifest.json.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)

    def log(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("run logger is closed")
        self.sequence += 1
        event = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "seq": self.sequence,
            "time_utc": utc_now(),
            "elapsed_ms": round((time.monotonic() - self._start_monotonic) * 1000, 3),
            "event": event_type,
            "attempt": self.attempt,
        }
        event.update(_json_value(fields))
        self._events.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
        self._events.flush()
        if self.sequence % self.fsync_interval == 0:
            os.fsync(self._events.fileno())
        self._event_counts[event_type] += 1
        return event

    def start_attempt(self, reason: str = "environment_reset") -> int:
        self.attempt += 1
        self.log("attempt_started", reason=reason)
        return self.attempt

    def annotate_level(self, label: str, source: str = "evaluator", **fields: Any) -> None:
        """Write an external label to telemetry; callers must not expose it to the agent."""

        self.log("level_annotation", label=label, source=source, **fields)

    def store_frame(self, frame: Frame) -> str:
        digest = frame.digest
        if self.store_frames and digest not in self._stored_frames:
            destination = self.frames_dir / f"{digest}.png"
            if not destination.exists():
                temporary = self.frames_dir / f".{digest}.tmp"
                temporary.write_bytes(encode_png(frame))
                os.replace(temporary, destination)
            self._stored_frames.add(digest)
        return digest

    def frame_fields(self, frame: Frame) -> Dict[str, Any]:
        return {
            "frame": self.store_frame(frame),
            "frame_width": frame.width,
            "frame_height": frame.height,
            "frame_channels": frame.channels,
            "visual_signature": signature_key(frame.coarse_signature()),
            "scene_signature": signature_key(frame.coarse_signature(columns=3, rows=3)),
        }

    def store_decision_snapshot(
        self, decision: int, state: bytes, frame: Frame
    ) -> Dict[str, Any]:
        """Persist an opaque emulator checkpoint for constant-time episodic resume."""

        if decision <= 0:
            raise ValueError("snapshot decision must be positive")
        if not state:
            raise ValueError("snapshot state must not be empty")
        digest = sha256(state).hexdigest()
        relative = Path("states") / f"{digest}.state"
        if digest not in self._stored_states:
            self.states_dir.mkdir(exist_ok=True)
            destination = self.run_dir / relative
            if not destination.exists():
                temporary = self.states_dir / f".{digest}.tmp"
                temporary.write_bytes(state)
                os.replace(temporary, destination)
            self._stored_states.add(digest)
        return self.log(
            "decision_snapshot_stored",
            decision=decision,
            state_file=str(relative),
            state_sha256=digest,
            state_bytes=len(state),
            **self.frame_fields(frame),
        )

    def store_option_archive_snapshot(
        self,
        decision: int,
        state_id: str,
        state: bytes,
        frame: Frame,
    ) -> Dict[str, Any]:
        """Persist one promoted option without exposing its bytes to policy."""

        if decision < 0:
            raise ValueError("archive snapshot decision must be non-negative")
        if not state_id:
            raise ValueError("archive snapshot requires a state ID")
        if not state:
            raise ValueError("archive snapshot state must not be empty")
        digest = sha256(state).hexdigest()
        relative = Path("states") / f"{digest}.state"
        if digest not in self._stored_states:
            self.states_dir.mkdir(exist_ok=True)
            destination = self.run_dir / relative
            if not destination.exists():
                temporary = self.states_dir / f".{digest}.tmp"
                temporary.write_bytes(state)
                os.replace(temporary, destination)
            self._stored_states.add(digest)
        return self.log(
            "option_archive_snapshot_stored",
            decision=decision,
            state_id=state_id,
            state_file=str(relative),
            state_sha256=digest,
            state_bytes=len(state),
            agent_visible=False,
            **self.frame_fields(frame),
        )

    def store_goal_milestone_checkpoint_snapshot(
        self,
        decision: int,
        state_id: str,
        state: bytes,
        frame: Frame,
        **metadata: Any,
    ) -> Dict[str, Any]:
        """Persist a pre-milestone rollback state for episodic resume."""

        if decision < 0:
            raise ValueError("milestone snapshot decision must be non-negative")
        if not state_id:
            raise ValueError("milestone snapshot requires a state ID")
        if not state:
            raise ValueError("milestone snapshot state must not be empty")
        digest = sha256(state).hexdigest()
        relative = Path("states") / f"{digest}.state"
        if digest not in self._stored_states:
            self.states_dir.mkdir(exist_ok=True)
            destination = self.run_dir / relative
            if not destination.exists():
                temporary = self.states_dir / f".{digest}.tmp"
                temporary.write_bytes(state)
                os.replace(temporary, destination)
            self._stored_states.add(digest)
        return self.log(
            "goal_milestone_checkpoint_snapshot_stored",
            decision=decision,
            state_id=state_id,
            state_file=str(relative),
            state_sha256=digest,
            state_bytes=len(state),
            agent_visible=False,
            **metadata,
            **self.frame_fields(frame),
        )

    def close(self, status: str = "complete", error: Optional[str] = None) -> None:
        if self._closed:
            return
        self.log("run_finished", status=status, error=error)
        os.fsync(self._events.fileno())
        self._events.close()
        self.manifest.update(
            {
                "status": status,
                "finished_at": utc_now(),
                "event_count": self.sequence,
                "event_counts": dict(sorted(self._event_counts.items())),
                "attempt_count": self.attempt,
                "unique_frame_count": len(self._stored_frames),
                "unique_state_count": len(self._stored_states),
            }
        )
        if error is not None:
            self.manifest["error"] = error
        self._write_manifest()
        self._closed = True

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close("error" if exc is not None else "complete", None if exc is None else str(exc))


class LoggedEnvironment:
    """Environment decorator that records pixels and opaque state lifecycles."""

    def __init__(self, env: PixelSaveStateEnv, logger: RunLogger) -> None:
        self.env = env
        self.logger = logger
        self._frame: Optional[Frame] = None
        self._next_state = 0
        self._active_states: Dict[int, str] = {}
        self._persisted_archive_state_ids: set[str] = set()
        self._persisted_milestone_state_ids: set[str] = set()
        self.last_step_seq: Optional[int] = None
        self.last_state_event_seq: Optional[int] = None
        self.phase = "agent"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(
        self,
        *,
        start_attempt: bool = True,
        phase: str = "agent",
    ) -> Frame:
        self.phase = phase
        if start_attempt:
            self.logger.start_attempt()
        self._frame = self.env.reset()
        self.logger.log(
            "env_reset", phase=self.phase, **self.logger.frame_fields(self._frame)
        )
        return self._frame

    def start_attempt_from_current(
        self, frame: Frame, reason: str = "evaluator_initialization"
    ) -> Frame:
        self._frame = frame
        self.phase = "agent"
        self.logger.start_attempt(reason)
        self.logger.log(
            "env_attached",
            phase=self.phase,
            reason=reason,
            **self.logger.frame_fields(frame),
        )
        return frame

    def observe(self) -> Frame:
        frame = getattr(self.env, "observe")()
        self._frame = frame
        return frame

    def step(self, action: Action, frames: int = 1) -> Frame:
        source = self._frame
        target = self.env.step(action, frames)
        self._frame = target
        fields = self.logger.frame_fields(target)
        fields.update(
            {
                "action": action.value,
                "action_frames": frames,
                "source_frame": None if source is None else self.logger.store_frame(source),
                "target_frame": target.digest,
                "visual_change": None if source is None else source.mean_absolute_difference(target),
                "phase": self.phase,
            }
        )
        event = self.logger.log("env_step", **fields)
        self.last_step_seq = int(event["seq"])
        return target

    def save_state(self) -> object:
        state = self.env.save_state()
        self._next_state += 1
        alias = f"state-{self._next_state:08d}"
        self._active_states[id(state)] = alias
        event = self.logger.log(
            "state_saved",
            state_id=alias,
            frame=None if self._frame is None else self.logger.store_frame(self._frame),
        )
        self.last_state_event_seq = int(event["seq"])
        return state

    def persist_option_archive_state(
        self,
        state: object,
        frame: Frame,
        decision: int,
    ) -> Optional[Dict[str, Any]]:
        """Store an opaque promoted branch while restoring the live state."""

        export_state = getattr(self.env, "export_state", None)
        release_state = getattr(self.env, "release_state", None)
        if not callable(export_state) or not callable(release_state):
            return None
        state_id = self.state_id(state)
        if state_id is None or self._frame is None:
            return None
        if state_id in self._persisted_archive_state_ids:
            return None
        live_frame = self._frame
        live_state = self.env.save_state()
        try:
            archived_frame = self.env.load_state(state)
            if archived_frame.digest != frame.digest:
                raise RuntimeError(
                    "promoted option state does not match its archived frame"
                )
            payload = export_state()
            stored = self.logger.store_option_archive_snapshot(
                decision, state_id, payload, frame
            )
            self._persisted_archive_state_ids.add(state_id)
            return stored
        finally:
            restored = self.env.load_state(live_state)
            release_state(live_state)
            if restored.digest != live_frame.digest:
                raise RuntimeError(
                    "live emulator state diverged after archive snapshot"
                )
            self._frame = live_frame

    def import_option_archive_state(
        self,
        state: bytes,
        frame: Frame,
        *,
        source_run_id: str,
        source_state_id: str,
    ) -> object:
        """Import an evaluator-owned archive and restore the live state."""

        import_state = getattr(self.env, "import_state", None)
        release_state = getattr(self.env, "release_state", None)
        if not callable(import_state) or not callable(release_state):
            raise RuntimeError(
                "environment does not support persistent option archives"
            )
        if self._frame is None:
            raise RuntimeError("cannot import an archive before environment attach")
        live_frame = self._frame
        live_state = self.env.save_state()
        imported_handle: Optional[object] = None
        try:
            imported_frame = import_state(state, frame)
            if imported_frame.digest != frame.digest:
                raise RuntimeError(
                    "imported option state does not match its archived frame"
                )
            imported_handle = self.env.save_state()
            self._next_state += 1
            imported_state_id = f"state-{self._next_state:08d}"
            self._active_states[id(imported_handle)] = imported_state_id
        finally:
            restored = self.env.load_state(live_state)
            release_state(live_state)
            if restored.digest != live_frame.digest:
                if imported_handle is not None:
                    self.release_state(imported_handle)
                raise RuntimeError(
                    "live emulator state diverged after archive import"
                )
            self._frame = live_frame
        assert imported_handle is not None
        imported_state_id = self.state_id(imported_handle)
        assert imported_state_id is not None
        stored = self.logger.store_option_archive_snapshot(
            0, imported_state_id, state, frame
        )
        self._persisted_archive_state_ids.add(imported_state_id)
        saved = self.logger.log(
            "state_saved",
            state_id=imported_state_id,
            imported_option_archive=True,
            option_archive_state_file=stored["state_file"],
            option_archive_state_sha256=stored["state_sha256"],
            frame=self.logger.store_frame(frame),
        )
        self.last_state_event_seq = int(saved["seq"])
        self.logger.log(
            "episodic_option_archive_state_imported",
            source_run_id=source_run_id,
            source_state_id=source_state_id,
            state_id=imported_state_id,
            agent_visible=False,
            **self.logger.frame_fields(frame),
        )
        return imported_handle

    def persist_goal_milestone_checkpoint_state(
        self,
        state: object,
        frame: Frame,
        decision: int,
        **metadata: Any,
    ) -> Optional[Dict[str, Any]]:
        """Store a rollback capability without changing the live emulator."""

        export_state = getattr(self.env, "export_state", None)
        release_state = getattr(self.env, "release_state", None)
        if not callable(export_state) or not callable(release_state):
            return None
        state_id = self.state_id(state)
        if state_id is None or self._frame is None:
            return None
        live_frame = self._frame
        live_state = self.env.save_state()
        try:
            checkpoint_frame = self.env.load_state(state)
            if checkpoint_frame.digest != frame.digest:
                raise RuntimeError(
                    "goal milestone state does not match its checkpoint frame"
                )
            payload = export_state()
            stored = self.logger.store_goal_milestone_checkpoint_snapshot(
                decision,
                state_id,
                payload,
                frame,
                **metadata,
            )
            return stored
        finally:
            restored = self.env.load_state(live_state)
            release_state(live_state)
            if restored.digest != live_frame.digest:
                raise RuntimeError(
                    "live emulator state diverged after milestone snapshot"
                )
            self._frame = live_frame

    def import_goal_milestone_checkpoint_state(
        self,
        state: bytes,
        frame: Frame,
        *,
        source_run_id: str,
        source_state_id: str,
        metadata: Dict[str, Any],
    ) -> object:
        """Import a persisted rollback capability and restore live state."""

        import_state = getattr(self.env, "import_state", None)
        release_state = getattr(self.env, "release_state", None)
        if not callable(import_state) or not callable(release_state):
            raise RuntimeError(
                "environment does not support persistent milestone checkpoints"
            )
        if self._frame is None:
            raise RuntimeError(
                "cannot import a milestone checkpoint before environment attach"
            )
        live_frame = self._frame
        live_state = self.env.save_state()
        imported_handle: Optional[object] = None
        try:
            imported_frame = import_state(state, frame)
            if imported_frame.digest != frame.digest:
                raise RuntimeError(
                    "imported milestone state does not match its checkpoint frame"
                )
            imported_handle = self.env.save_state()
            self._next_state += 1
            imported_state_id = f"state-{self._next_state:08d}"
            self._active_states[id(imported_handle)] = imported_state_id
        finally:
            restored = self.env.load_state(live_state)
            release_state(live_state)
            if restored.digest != live_frame.digest:
                if imported_handle is not None:
                    self.release_state(imported_handle)
                raise RuntimeError(
                    "live emulator state diverged after milestone import"
                )
            self._frame = live_frame
        assert imported_handle is not None
        imported_state_id = self.state_id(imported_handle)
        assert imported_state_id is not None
        stored = self.logger.store_goal_milestone_checkpoint_snapshot(
            0,
            imported_state_id,
            state,
            frame,
            **metadata,
        )
        self._persisted_milestone_state_ids.add(imported_state_id)
        saved = self.logger.log(
            "state_saved",
            state_id=imported_state_id,
            imported_goal_milestone_checkpoint=True,
            goal_milestone_state_file=stored["state_file"],
            goal_milestone_state_sha256=stored["state_sha256"],
            frame=self.logger.store_frame(frame),
        )
        self.last_state_event_seq = int(saved["seq"])
        self.logger.log(
            "episodic_goal_milestone_checkpoint_state_imported",
            source_run_id=source_run_id,
            source_state_id=source_state_id,
            state_id=imported_state_id,
            agent_visible=False,
            **self.logger.frame_fields(frame),
        )
        return imported_handle

    def state_id(self, state: object) -> Optional[str]:
        return self._active_states.get(id(state))

    def load_state(self, state: object) -> Frame:
        alias = self.state_id(state)
        frame = self.env.load_state(state)
        self._frame = frame
        event = self.logger.log("state_loaded", state_id=alias, **self.logger.frame_fields(frame))
        self.last_state_event_seq = int(event["seq"])
        return frame

    def release_state(self, state: object) -> None:
        alias = self.state_id(state)
        release = getattr(self.env, "release_state", None)
        if release is not None:
            release(state)
        event = self.logger.log("state_released", state_id=alias)
        self.last_state_event_seq = int(event["seq"])
        self._active_states.pop(id(state), None)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "LoggedEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def read_events(run_dir: Path) -> Iterable[Dict[str, Any]]:
    with (Path(run_dir) / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
