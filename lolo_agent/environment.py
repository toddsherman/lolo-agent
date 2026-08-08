from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from .pixels import Frame


class Action(str, Enum):
    """Primitive NES controls exposed as hardware affordances."""

    NOOP = "noop"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    A = "a"
    B = "b"
    START = "start"
    SELECT = "select"


@runtime_checkable
class PixelSaveStateEnv(Protocol):
    """The complete information boundary visible to an agent.

    Save-state bytes are capabilities for restoration, not observations. Agent
    code must treat them as opaque and must never parse their contents.
    """

    def reset(self) -> Frame:
        ...

    def step(self, action: Action, frames: int = 1) -> Frame:
        ...

    def save_state(self) -> bytes:
        ...

    def load_state(self, state: bytes) -> Frame:
        ...
