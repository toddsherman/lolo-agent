"""Evaluator fixture: hidden symbolic rules rendered as raw pixels.

Production agents must not import or inspect this module. It stands in for an
emulator whose internal state is likewise outside the agent information path.
"""

from __future__ import annotations

import json
from typing import Iterable, Sequence, Set, Tuple

from .environment import Action
from .pixels import Frame

Point = Tuple[int, int]


class MockPuzzleEnv:
    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        player: Point = (1, 2),
        crate: Point = (2, 2),
        goal: Point = (4, 2),
        walls: Iterable[Point] = (),
        tile_size: int = 3,
    ) -> None:
        self.width = width
        self.height = height
        self.initial_player = player
        self.initial_crate = crate
        self.goal = goal
        self.walls: Set[Point] = set(walls)
        self.tile_size = tile_size
        self.player = player
        self.crate = crate
        self.steps = 0

    def reset(self) -> Frame:
        self.player = self.initial_player
        self.crate = self.initial_crate
        self.steps = 0
        return self._render()

    def step(self, action: Action, frames: int = 1) -> Frame:
        for _ in range(frames):
            self._single_step(action)
        return self._render()

    def _single_step(self, action: Action) -> None:
        self.steps += 1
        if self.evaluator_solved():
            return
        delta = {
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0),
        }.get(action)
        if delta is None:
            return
        target = (self.player[0] + delta[0], self.player[1] + delta[1])
        if not self._open(target):
            return
        if target == self.crate:
            beyond = (target[0] + delta[0], target[1] + delta[1])
            if not self._open(beyond) or beyond == self.crate:
                return
            self.crate = beyond
        self.player = target

    def _open(self, point: Point) -> bool:
        x, y = point
        return 0 < x < self.width - 1 and 0 < y < self.height - 1 and point not in self.walls

    def save_state(self) -> bytes:
        return json.dumps(
            {"player": self.player, "crate": self.crate, "steps": self.steps},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def load_state(self, state: bytes) -> Frame:
        data = json.loads(state)
        self.player = tuple(data["player"])
        self.crate = tuple(data["crate"])
        self.steps = int(data["steps"])
        return self._render()

    def evaluator_solved(self) -> bool:
        """Ground truth for metrics; never returned through the agent API."""

        return self.crate == self.goal

    def _render(self) -> Frame:
        colors = {"floor": 15, "wall": 65, "goal": 115, "crate": 175, "player": 235}
        pixel_width = self.width * self.tile_size
        pixel_height = self.height * self.tile_size
        if self.evaluator_solved():
            # A visually obvious scene transition stands in for the next-room
            # animation. The agent sees only the pixels and is not told that
            # this pattern means success.
            pixels = bytes(
                245 if (x // self.tile_size + y // self.tile_size) % 2 else 25
                for y in range(pixel_height)
                for x in range(pixel_width)
            )
            return Frame(pixel_width, pixel_height, 1, pixels)
        pixels = bytearray(pixel_width * pixel_height)
        for y in range(self.height):
            for x in range(self.width):
                point = (x, y)
                if x in (0, self.width - 1) or y in (0, self.height - 1) or point in self.walls:
                    value = colors["wall"]
                elif point == self.player:
                    value = colors["player"]
                elif point == self.crate:
                    value = colors["crate"]
                elif point == self.goal:
                    value = colors["goal"]
                else:
                    value = colors["floor"]
                for py in range(y * self.tile_size, (y + 1) * self.tile_size):
                    start = py * pixel_width + x * self.tile_size
                    pixels[start : start + self.tile_size] = bytes([value]) * self.tile_size
        return Frame(pixel_width, pixel_height, 1, bytes(pixels))
