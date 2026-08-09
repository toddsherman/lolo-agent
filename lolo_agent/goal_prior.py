from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .pixels import Frame


HeartSlot = Tuple[int, int]

_PALETTE = {
    ".": (0, 0, 0),
    "B": (86, 29, 0),
    "M": (183, 30, 123),
    "W": (255, 255, 255),
}
_HEART_ROWS = (
    "................",
    "BBBWWWWWWWWWW.BB",
    "BBWWWWWWWWWWWW.B",
    "BWWW........WWW.",
    ".WW..MMMMMMM.WW.",
    ".WW.MWWMMWWMMWW.",
    ".WW.WWWWWWWWMWW.",
    ".WW.WWWWWWWWMWW.",
    ".WW.MWWWWWWMMWW.",
    "BWW.MMWWWWMMMWW.",
    "BWW.MMMWWMMMMWW.",
    "BWWW.MMMMMMMWWW.",
    ".MWWWWWWWWWWWWM.",
    "..MWWWWWWWWWWM..",
    ".B.MMMMMMMMMM..B",
    ".BB...........BB",
)
HEART_PROTOTYPE: Tuple[Tuple[int, int, int], ...] = tuple(
    _PALETTE[value] for row in _HEART_ROWS for value in row
)


@dataclass(frozen=True)
class HeartGoalAnalysis:
    reliable: bool
    known_slots: Tuple[HeartSlot, ...]
    source_present: Tuple[HeartSlot, ...]
    target_present: Tuple[HeartSlot, ...]
    collected: Tuple[HeartSlot, ...]
    target_similarities: Tuple[Tuple[int, int, float], ...]
    heart_reward: float
    all_hearts_reward: float
    total_reward: float
    global_visual_change: float
    target_intensity: float

    @property
    def remaining_hearts(self) -> int:
        return len(self.target_present)

    def telemetry(self) -> Dict[str, object]:
        return {
            "human_prior_reliable": self.reliable,
            "human_prior_known_heart_slots": self.known_slots,
            "human_prior_source_hearts": self.source_present,
            "human_prior_target_hearts": self.target_present,
            "human_prior_collected_heart_slots": self.collected,
            "human_prior_collected_hearts": len(self.collected),
            "human_prior_remaining_hearts": self.remaining_hearts,
            "human_prior_heart_similarities": [
                {"x": x, "y": y, "similarity": similarity}
                for x, y, similarity in self.target_similarities
            ],
            "human_prior_heart_reward": self.heart_reward,
            "human_prior_all_hearts_reward": self.all_hearts_reward,
            "human_prior_goal_reward": self.total_reward,
            "human_prior_global_visual_change": self.global_visual_change,
            "human_prior_target_intensity": self.target_intensity,
        }


class PixelHeartGoalPrior:
    """Explicit human-like heart prior operating only on screen pixels.

    This component is intentionally not used by the strict rule-free track. Its
    fixed sprite prototype and weights are frozen evaluator configuration.
    """

    def __init__(
        self,
        heart_reward: float = 25.0,
        all_hearts_reward: float = 75.0,
        discovery_similarity: float = 0.98,
        presence_similarity: float = 0.55,
        maximum_event_visual_change: float = 0.08,
        minimum_scene_intensity: float = 0.05,
    ) -> None:
        self.heart_reward = float(heart_reward)
        self.all_hearts_reward = float(all_hearts_reward)
        self.discovery_similarity = float(discovery_similarity)
        self.presence_similarity = float(presence_similarity)
        self.maximum_event_visual_change = float(maximum_event_visual_change)
        self.minimum_scene_intensity = float(minimum_scene_intensity)
        self.known_slots: set[HeartSlot] = set()
        self.current_present: set[HeartSlot] = set()
        self.initialized = False
        self.best_remaining_hearts: Optional[int] = None

    @staticmethod
    def mean_intensity(frame: Frame) -> float:
        return sum(frame.pixels) / (255.0 * len(frame.pixels))

    @staticmethod
    def _pixel(frame: Frame, x: int, y: int) -> Tuple[int, int, int]:
        offset = (y * frame.width + x) * frame.channels
        values = frame.pixels[offset : offset + frame.channels]
        if frame.channels == 1:
            return values[0], values[0], values[0]
        if frame.channels == 2:
            return values[0], values[0], values[0]
        return values[0], values[1], values[2]

    def similarity(self, frame: Frame, slot: HeartSlot) -> float:
        x, y = slot
        if x < 0 or y < 0 or x + 16 > frame.width or y + 16 > frame.height:
            return 0.0
        matches = 0
        index = 0
        for row in range(y, y + 16):
            for column in range(x, x + 16):
                matches += self._pixel(frame, column, row) == HEART_PROTOTYPE[index]
                index += 1
        return matches / len(HEART_PROTOTYPE)

    def discover(self, frame: Frame) -> Tuple[HeartSlot, ...]:
        if self.mean_intensity(frame) < self.minimum_scene_intensity:
            return ()
        slots = []
        for y in range(32, min(frame.height - 15, 208), 16):
            for x in range(32, min(frame.width - 15, 208), 16):
                slot = (x, y)
                if self.similarity(frame, slot) >= self.discovery_similarity:
                    slots.append(slot)
        return tuple(slots)

    def observe_room(self, frame: Frame) -> Tuple[HeartSlot, ...]:
        discovered = set(self.discover(frame))
        if discovered and not self.initialized:
            self.known_slots = discovered
            self.current_present = discovered
            self.initialized = True
            self.best_remaining_hearts = len(discovered)
        return tuple(sorted(discovered))

    def analyze(self, source: Frame, target: Frame) -> HeartGoalAnalysis:
        if not self.initialized:
            self.observe_room(source)
        target_intensity = self.mean_intensity(target)
        visual_change = source.mean_absolute_difference(target)
        similarities = tuple(
            (x, y, self.similarity(target, (x, y)))
            for x, y in sorted(self.known_slots)
        )
        reliable = bool(
            self.initialized
            and target_intensity >= self.minimum_scene_intensity
            and visual_change <= self.maximum_event_visual_change
        )
        target_present = (
            {
                (x, y)
                for x, y, similarity in similarities
                if similarity >= self.presence_similarity
            }
            if reliable
            else set(self.current_present)
        )
        collected = self.current_present - target_present if reliable else set()
        heart_reward = self.heart_reward * len(collected)
        all_hearts_reward = (
            self.all_hearts_reward
            if collected and not target_present
            else 0.0
        )
        return HeartGoalAnalysis(
            reliable=reliable,
            known_slots=tuple(sorted(self.known_slots)),
            source_present=tuple(sorted(self.current_present)),
            target_present=tuple(sorted(target_present)),
            collected=tuple(sorted(collected)),
            target_similarities=similarities,
            heart_reward=heart_reward,
            all_hearts_reward=all_hearts_reward,
            total_reward=heart_reward + all_hearts_reward,
            global_visual_change=visual_change,
            target_intensity=target_intensity,
        )

    def commit(self, analysis: HeartGoalAnalysis, frame: Frame) -> None:
        if not self.initialized:
            self.observe_room(frame)
            return
        if analysis.reliable:
            self.current_present = set(analysis.target_present)
            remaining = len(self.current_present)
            self.best_remaining_hearts = (
                remaining
                if self.best_remaining_hearts is None
                else min(self.best_remaining_hearts, remaining)
            )

    def restore(self, present: Sequence[HeartSlot], frame: Frame) -> None:
        discovered = set(self.discover(frame))
        if discovered and not self.initialized:
            self.known_slots = discovered
            self.initialized = True
        if self.initialized:
            self.current_present = set(present) if present else discovered

    def current_slots(self) -> Tuple[HeartSlot, ...]:
        return tuple(sorted(self.current_present))
