from __future__ import annotations

import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional

from .environment import Action
from .pixels import Frame


_MAGIC = 0x4F4C4F4C
_VERSION = 1
_REQUEST = struct.Struct("<IHHI")
_RESPONSE = struct.Struct("<IHHHHI")
_FRAME = struct.Struct("<III")
_STEP = struct.Struct("<II")
_HELLO = struct.Struct("<IIIId")
_HANDLE = struct.Struct("<Q")
_IMPORT = struct.Struct("<QIII")

_HELLO_COMMAND = 1
_RESET_COMMAND = 2
_STEP_COMMAND = 3
_SAVE_COMMAND = 4
_LOAD_COMMAND = 5
_DROP_COMMAND = 6
_CLOSE_COMMAND = 7
_EXPORT_COMMAND = 8
_IMPORT_COMMAND = 9

_BUTTON_IDS = {
    Action.B: 0,
    Action.SELECT: 2,
    Action.START: 3,
    Action.UP: 4,
    Action.DOWN: 5,
    Action.LEFT: 6,
    Action.RIGHT: 7,
    Action.A: 8,
}


class NativeHostError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeStateHandle:
    """An opaque capability valid only for its originating host session."""

    _token: int = field(repr=False)
    _owner: object = field(repr=False)


class NativeLibretroEnv:
    """Pixel-only client for the isolated native libretro host process."""

    def __init__(self, host_path: Path, core_path: Path, rom_path: Path) -> None:
        self.host_path = Path(host_path).expanduser().resolve()
        self.core_path = Path(core_path).expanduser().resolve()
        self.rom_path = Path(rom_path).expanduser().resolve()
        for path in (self.host_path, self.core_path, self.rom_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self._owner = object()
        self._closed = False
        self._frame: Optional[Frame] = None
        self._process = subprocess.Popen(
            [
                str(self.host_path),
                "--core",
                str(self.core_path),
                "--rom",
                str(self.rom_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise NativeHostError("failed to open native host protocol pipes")
        self._input: BinaryIO = self._process.stdin
        self._output: BinaryIO = self._process.stdout
        payload = self._request(_HELLO_COMMAND)
        if len(payload) < _HELLO.size:
            raise NativeHostError("native host returned a truncated hello response")
        name_length, version_length, self.base_width, self.base_height, self.fps = (
            _HELLO.unpack_from(payload)
        )
        expected = _HELLO.size + name_length + version_length
        if len(payload) != expected:
            raise NativeHostError("native host returned invalid hello metadata")
        start = _HELLO.size
        self.core_name = payload[start : start + name_length].decode(errors="replace")
        start += name_length
        self.core_version = payload[start : start + version_length].decode(errors="replace")

    def reset(self) -> Frame:
        self._frame = self._decode_frame(self._request(_RESET_COMMAND))
        return self._frame

    def observe(self) -> Frame:
        if self._frame is None:
            raise NativeHostError("environment has not produced a frame")
        return self._frame

    def step(self, action: Action, frames: int = 1) -> Frame:
        if frames <= 0:
            raise ValueError("frames must be positive")
        button_id = _BUTTON_IDS.get(action)
        mask = 0 if button_id is None else 1 << button_id
        self._frame = self._decode_frame(self._request(_STEP_COMMAND, _STEP.pack(mask, frames)))
        return self._frame

    def save_state(self) -> NativeStateHandle:
        payload = self._request(_SAVE_COMMAND)
        if len(payload) != _HANDLE.size:
            raise NativeHostError("native host returned an invalid state handle")
        return NativeStateHandle(_HANDLE.unpack(payload)[0], self._owner)

    def load_state(self, state: object) -> Frame:
        handle = self._validate_handle(state)
        self._frame = self._decode_frame(
            self._request(_LOAD_COMMAND, _HANDLE.pack(handle._token))
        )
        return self._frame

    def release_state(self, state: object) -> None:
        handle = self._validate_handle(state)
        self._request(_DROP_COMMAND, _HANDLE.pack(handle._token))

    def export_state(self) -> bytes:
        """Export an opaque cross-session checkpoint for evaluator persistence."""

        return self._request(_EXPORT_COMMAND)

    def import_state(self, state: bytes, frame: Frame) -> Frame:
        """Restore an evaluator-owned checkpoint without exposing it to the agent."""

        if not state:
            raise ValueError("imported state must not be empty")
        if frame.channels != 3:
            raise ValueError("imported framebuffer must have three channels")
        payload = (
            _IMPORT.pack(len(state), frame.width, frame.height, frame.channels)
            + state
            + frame.pixels
        )
        self._frame = self._decode_frame(self._request(_IMPORT_COMMAND, payload))
        return self._frame

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                self._request(_CLOSE_COMMAND)
                self._process.wait(timeout=2)
        except (BrokenPipeError, EOFError, NativeHostError, subprocess.TimeoutExpired):
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
        finally:
            self._input.close()
            self._output.close()
            if self._process.stderr is not None:
                self._process.stderr.close()
            self._closed = True

    def __enter__(self) -> "NativeLibretroEnv":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _validate_handle(self, state: object) -> NativeStateHandle:
        if not isinstance(state, NativeStateHandle) or state._owner is not self._owner:
            raise NativeHostError("save-state handle belongs to another host session")
        return state

    def _request(self, command: int, payload: bytes = b"") -> bytes:
        if self._closed:
            raise NativeHostError("native host is closed")
        self._input.write(_REQUEST.pack(_MAGIC, _VERSION, command, len(payload)))
        if payload:
            self._input.write(payload)
        self._input.flush()
        header = self._read_exact(_RESPONSE.size)
        magic, version, status, response_command, _reserved, size = _RESPONSE.unpack(header)
        if magic != _MAGIC or version != _VERSION or response_command != command:
            raise NativeHostError("native host protocol response mismatch")
        response = self._read_exact(size)
        if status:
            raise NativeHostError(response.decode(errors="replace"))
        return response

    def _read_exact(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._output.read(size - len(chunks))
            if not chunk:
                details = ""
                if self._process.stderr is not None and self._process.poll() is not None:
                    details = self._process.stderr.read().decode(errors="replace").strip()
                raise NativeHostError(f"native host exited unexpectedly: {details}")
            chunks.extend(chunk)
        return bytes(chunks)

    @staticmethod
    def _decode_frame(payload: bytes) -> Frame:
        if len(payload) < _FRAME.size:
            raise NativeHostError("native host returned a truncated frame")
        width, height, channels = _FRAME.unpack_from(payload)
        pixels = payload[_FRAME.size :]
        return Frame(width, height, channels, pixels)
