from __future__ import annotations

import ctypes
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

from .environment import Action
from .pixels import Frame


# Stable libretro API v1 values used by this frontend. Deliberately absent are
# the memory-access functions: the environment has no path to CPU/PPU RAM.
_RETRO_DEVICE_JOYPAD = 1
_RETRO_ENVIRONMENT_GET_CAN_DUPE = 3
_RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY = 9
_RETRO_ENVIRONMENT_SET_PIXEL_FORMAT = 10
_RETRO_ENVIRONMENT_GET_VARIABLE = 15
_RETRO_ENVIRONMENT_SET_VARIABLES = 16
_RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE = 17
_RETRO_ENVIRONMENT_GET_INPUT_DEVICE_CAPABILITIES = 24
_RETRO_ENVIRONMENT_GET_CONTENT_DIRECTORY = 30
_RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY = 31
_RETRO_ENVIRONMENT_SET_GEOMETRY = 37
_RETRO_ENVIRONMENT_GET_LANGUAGE = 39
_RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION = 52
_RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2 = 67
_RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2_INTL = 68

_PIXEL_0RGB1555 = 0
_PIXEL_XRGB8888 = 1
_PIXEL_RGB565 = 2

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


class LibretroError(RuntimeError):
    pass


class _RetroGameInfo(ctypes.Structure):
    _fields_ = [
        ("path", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("meta", ctypes.c_char_p),
    ]


class _RetroSystemInfo(ctypes.Structure):
    _fields_ = [
        ("library_name", ctypes.c_char_p),
        ("library_version", ctypes.c_char_p),
        ("valid_extensions", ctypes.c_char_p),
        ("need_fullpath", ctypes.c_bool),
        ("block_extract", ctypes.c_bool),
    ]


class _RetroGameGeometry(ctypes.Structure):
    _fields_ = [
        ("base_width", ctypes.c_uint),
        ("base_height", ctypes.c_uint),
        ("max_width", ctypes.c_uint),
        ("max_height", ctypes.c_uint),
        ("aspect_ratio", ctypes.c_float),
    ]


class _RetroSystemTiming(ctypes.Structure):
    _fields_ = [("fps", ctypes.c_double), ("sample_rate", ctypes.c_double)]


class _RetroSystemAVInfo(ctypes.Structure):
    _fields_ = [("geometry", _RetroGameGeometry), ("timing", _RetroSystemTiming)]


class _RetroVariable(ctypes.Structure):
    _fields_ = [("key", ctypes.c_char_p), ("value", ctypes.c_char_p)]


_EnvironmentCallback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint, ctypes.c_void_p)
_VideoCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t
)
_AudioCallback = ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int16)
_AudioBatchCallback = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t
)
_InputPollCallback = ctypes.CFUNCTYPE(None)
_InputStateCallback = ctypes.CFUNCTYPE(
    ctypes.c_int16, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint
)


class LibretroEnv:
    """Direct pixel/controller/save-state environment for a libretro core.

    This loader binds only the public video, input, lifecycle, and serialization
    API. It intentionally does not bind libretro's optional memory inspection
    functions. State bytes are accepted only for same-process restoration; the
    production native host will replace them with opaque server-side handles.
    """

    def __init__(
        self,
        core_path: Path,
        rom_path: Path,
        system_directory: Optional[Path] = None,
        save_directory: Optional[Path] = None,
    ) -> None:
        self.core_path = Path(core_path).expanduser().resolve()
        self.rom_path = Path(rom_path).expanduser().resolve()
        if not self.core_path.is_file():
            raise FileNotFoundError(self.core_path)
        if not self.rom_path.is_file():
            raise FileNotFoundError(self.rom_path)

        self.system_directory = Path(system_directory or self.rom_path.parent).resolve()
        self.save_directory = Path(save_directory or self.rom_path.parent).resolve()
        self._system_directory_bytes = str(self.system_directory).encode()
        self._save_directory_bytes = str(self.save_directory).encode()
        self._rom_path_bytes = str(self.rom_path).encode()
        self._pixel_format = _PIXEL_0RGB1555
        self._button_mask = 0
        self._frame: Optional[Frame] = None
        self._saved_frames: Dict[str, Frame] = {}
        self._closed = False

        self._core = ctypes.CDLL(str(self.core_path))
        self._bind_functions()
        self._environment_callback = _EnvironmentCallback(self._environment)
        self._video_callback = _VideoCallback(self._video_refresh)
        self._audio_callback = _AudioCallback(self._audio_sample)
        self._audio_batch_callback = _AudioBatchCallback(self._audio_batch)
        self._input_poll_callback = _InputPollCallback(self._input_poll)
        self._input_state_callback = _InputStateCallback(self._input_state)

        self._core.retro_set_environment(self._environment_callback)
        self._core.retro_set_video_refresh(self._video_callback)
        self._core.retro_set_audio_sample(self._audio_callback)
        self._core.retro_set_audio_sample_batch(self._audio_batch_callback)
        self._core.retro_set_input_poll(self._input_poll_callback)
        self._core.retro_set_input_state(self._input_state_callback)
        self._core.retro_init()

        self.system_info = _RetroSystemInfo()
        self._core.retro_get_system_info(ctypes.byref(self.system_info))
        rom_bytes = self.rom_path.read_bytes()
        self._rom_buffer = ctypes.create_string_buffer(rom_bytes)
        game = _RetroGameInfo(
            self._rom_path_bytes,
            ctypes.cast(self._rom_buffer, ctypes.c_void_p),
            len(rom_bytes),
            None,
        )
        if not self._core.retro_load_game(ctypes.byref(game)):
            self._core.retro_deinit()
            raise LibretroError(f"core rejected content: {self.rom_path}")

        self.av_info = _RetroSystemAVInfo()
        self._core.retro_get_system_av_info(ctypes.byref(self.av_info))
        self._run_frame()

    @property
    def core_name(self) -> str:
        return (self.system_info.library_name or b"unknown").decode(errors="replace")

    @property
    def core_version(self) -> str:
        return (self.system_info.library_version or b"unknown").decode(errors="replace")

    @property
    def fps(self) -> float:
        return self.av_info.timing.fps

    def reset(self) -> Frame:
        self._ensure_open()
        self._button_mask = 0
        self._core.retro_reset()
        self._run_frame()
        return self.observe()

    def observe(self) -> Frame:
        if self._frame is None:
            raise LibretroError("core has not produced a video frame")
        return self._frame

    def step(self, action: Action, frames: int = 1) -> Frame:
        self._ensure_open()
        if frames <= 0:
            raise ValueError("frames must be positive")
        button_id = _BUTTON_IDS.get(action)
        self._button_mask = 0 if button_id is None else 1 << button_id
        for _ in range(frames):
            self._run_frame()
        self._button_mask = 0
        return self.observe()

    def save_state(self) -> bytes:
        self._ensure_open()
        size = self._core.retro_serialize_size()
        if size <= 0:
            raise LibretroError("core does not support serialization")
        buffer = ctypes.create_string_buffer(size)
        if not self._core.retro_serialize(buffer, size):
            raise LibretroError("core failed to serialize state")
        state = bytes(buffer.raw)
        self._saved_frames[sha256(state).hexdigest()] = self.observe()
        return state

    def load_state(self, state: bytes) -> Frame:
        self._ensure_open()
        if not state:
            raise ValueError("state must not be empty")
        buffer = ctypes.create_string_buffer(state, len(state))
        if not self._core.retro_unserialize(buffer, len(state)):
            raise LibretroError("core rejected serialized state")
        key = sha256(state).hexdigest()
        try:
            self._frame = self._saved_frames[key]
        except KeyError as exc:
            raise LibretroError("state was not created by this environment session") from exc
        self._button_mask = 0
        return self.observe()

    def close(self) -> None:
        if self._closed:
            return
        self._core.retro_unload_game()
        self._core.retro_deinit()
        self._closed = True

    def __enter__(self) -> "LibretroEnv":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise LibretroError("environment is closed")

    def _bind_functions(self) -> None:
        core = self._core
        core.retro_set_environment.argtypes = [_EnvironmentCallback]
        core.retro_set_video_refresh.argtypes = [_VideoCallback]
        core.retro_set_audio_sample.argtypes = [_AudioCallback]
        core.retro_set_audio_sample_batch.argtypes = [_AudioBatchCallback]
        core.retro_set_input_poll.argtypes = [_InputPollCallback]
        core.retro_set_input_state.argtypes = [_InputStateCallback]
        core.retro_init.argtypes = []
        core.retro_deinit.argtypes = []
        core.retro_reset.argtypes = []
        core.retro_run.argtypes = []
        core.retro_get_system_info.argtypes = [ctypes.POINTER(_RetroSystemInfo)]
        core.retro_get_system_av_info.argtypes = [ctypes.POINTER(_RetroSystemAVInfo)]
        core.retro_load_game.argtypes = [ctypes.POINTER(_RetroGameInfo)]
        core.retro_load_game.restype = ctypes.c_bool
        core.retro_unload_game.argtypes = []
        core.retro_serialize_size.argtypes = []
        core.retro_serialize_size.restype = ctypes.c_size_t
        core.retro_serialize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        core.retro_serialize.restype = ctypes.c_bool
        core.retro_unserialize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        core.retro_unserialize.restype = ctypes.c_bool

    def _environment(self, command: int, data: int) -> bool:
        if command == _RETRO_ENVIRONMENT_GET_CAN_DUPE:
            ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))[0] = True
            return True
        if command in (
            _RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY,
            _RETRO_ENVIRONMENT_GET_CONTENT_DIRECTORY,
        ):
            ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))[0] = self._system_directory_bytes
            return True
        if command == _RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
            ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))[0] = self._save_directory_bytes
            return True
        if command == _RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
            pixel_format = ctypes.cast(data, ctypes.POINTER(ctypes.c_int))[0]
            if pixel_format not in (_PIXEL_0RGB1555, _PIXEL_XRGB8888, _PIXEL_RGB565):
                return False
            self._pixel_format = pixel_format
            return True
        if command == _RETRO_ENVIRONMENT_GET_VARIABLE:
            ctypes.cast(data, ctypes.POINTER(_RetroVariable))[0].value = None
            return False
        if command == _RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE:
            ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))[0] = False
            return True
        if command == _RETRO_ENVIRONMENT_GET_INPUT_DEVICE_CAPABILITIES:
            ctypes.cast(data, ctypes.POINTER(ctypes.c_uint64))[0] = 1 << _RETRO_DEVICE_JOYPAD
            return True
        if command == _RETRO_ENVIRONMENT_GET_LANGUAGE:
            ctypes.cast(data, ctypes.POINTER(ctypes.c_uint))[0] = 0
            return True
        if command == _RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION:
            ctypes.cast(data, ctypes.POINTER(ctypes.c_uint))[0] = 2
            return True
        if command in (
            _RETRO_ENVIRONMENT_SET_VARIABLES,
            _RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2,
            _RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2_INTL,
            _RETRO_ENVIRONMENT_SET_GEOMETRY,
        ):
            return True
        return False

    def _video_refresh(self, data: int, width: int, height: int, pitch: int) -> None:
        if not data:
            return
        raw = ctypes.string_at(data, pitch * height)
        rgb = bytearray(width * height * 3)
        destination = 0
        for y in range(height):
            row = y * pitch
            for x in range(width):
                if self._pixel_format == _PIXEL_XRGB8888:
                    offset = row + x * 4
                    blue, green, red = raw[offset], raw[offset + 1], raw[offset + 2]
                else:
                    offset = row + x * 2
                    value = raw[offset] | (raw[offset + 1] << 8)
                    if self._pixel_format == _PIXEL_RGB565:
                        red = ((value >> 11) & 0x1F) * 255 // 31
                        green = ((value >> 5) & 0x3F) * 255 // 63
                        blue = (value & 0x1F) * 255 // 31
                    else:
                        red = ((value >> 10) & 0x1F) * 255 // 31
                        green = ((value >> 5) & 0x1F) * 255 // 31
                        blue = (value & 0x1F) * 255 // 31
                rgb[destination : destination + 3] = bytes((red, green, blue))
                destination += 3
        self._frame = Frame(width, height, 3, bytes(rgb))

    @staticmethod
    def _audio_sample(left: int, right: int) -> None:
        return None

    @staticmethod
    def _audio_batch(data: object, frames: int) -> int:
        return frames

    @staticmethod
    def _input_poll() -> None:
        return None

    def _input_state(self, port: int, device: int, index: int, button_id: int) -> int:
        if port != 0 or device != _RETRO_DEVICE_JOYPAD or index != 0:
            return 0
        return 1 if self._button_mask & (1 << button_id) else 0

    def _run_frame(self) -> None:
        self._core.retro_run()

