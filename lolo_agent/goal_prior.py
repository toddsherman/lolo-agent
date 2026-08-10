from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .pixels import Frame


HeartSlot = Tuple[int, int]

_PALETTE = {
    ".": (0, 0, 0),
    "B": (86, 29, 0),
    "M": (183, 30, 123),
    "P": (255, 110, 204),
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

# Pixel-observed open treasure from Lolo 1 room 2. Like the heart sprite, this
# is an explicitly labelled human prior and is never enabled in the strict
# rule-free track. The closed sprite changes into this form after all hearts
# disappear; the prior therefore does not need a room-specific coordinate.
_OPEN_CHEST_ROWS = (
    "W..P.P.P.P.P.PW.",
    "WWWWWWWWWWWWWWWW",
    "................",
    "WWWWWWWWWWWWWWWW",
    "W.............W.",
    "W.....PPP.....W.",
    "WB...PWWPP...BW.",
    "WB...PWWPP...BW.",
    "WBBB.PPPWP.BBBW.",
    "WBBB..PPP..BBBW.",
    "WBBBB.....BBBBW.",
    "WWWWWW...WWWWWW.",
    "W.MMMWWWWW.MMMW.",
    "BW.MMMW.W.MMMW..",
    "BWWWWWWWWWWWWW..",
    "BB..............",
)
OPEN_CHEST_PROTOTYPE: Tuple[Tuple[int, int, int], ...] = tuple(
    _PALETTE[value] for row in _OPEN_CHEST_ROWS for value in row
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
    chest_reward: float
    navigation_reward: float
    life_loss_penalty: float
    total_reward: float
    global_visual_change: float
    target_intensity: float
    source_player_slot: Optional[HeartSlot]
    target_player_slot: Optional[HeartSlot]
    source_heart_distance: Optional[float]
    target_heart_distance: Optional[float]
    source_chest_slot: Optional[HeartSlot]
    target_chest_slot: Optional[HeartSlot]
    source_chest_distance: Optional[float]
    target_chest_distance: Optional[float]
    chest_completed: bool
    source_life_signature: Optional[str]
    target_life_signature: Optional[str]
    life_counter_changed: bool
    dark_transition_started: bool
    life_loss_confirmed: bool

    @property
    def remaining_hearts(self) -> int:
        return len(self.target_present)

    @property
    def milestone_reward(self) -> float:
        return self.heart_reward + self.all_hearts_reward + self.chest_reward

    @property
    def outcome_reward(self) -> float:
        return self.milestone_reward + self.life_loss_penalty

    @property
    def goal_phase(self) -> str:
        if self.target_present:
            return "hearts"
        if self.chest_completed:
            return "chest_completed"
        if self.source_chest_slot is not None or self.target_chest_slot is not None:
            return "open_chest"
        if self.known_slots:
            return "awaiting_open_chest"
        return "uncalibrated"

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
            "human_prior_chest_reward": self.chest_reward,
            "human_prior_navigation_reward": self.navigation_reward,
            "human_prior_life_loss_penalty": self.life_loss_penalty,
            "human_prior_milestone_reward": self.milestone_reward,
            "human_prior_goal_reward": self.total_reward,
            "human_prior_goal_phase": self.goal_phase,
            "human_prior_source_player_slot": self.source_player_slot,
            "human_prior_target_player_slot": self.target_player_slot,
            "human_prior_source_heart_distance": self.source_heart_distance,
            "human_prior_target_heart_distance": self.target_heart_distance,
            "human_prior_source_chest_slot": self.source_chest_slot,
            "human_prior_target_chest_slot": self.target_chest_slot,
            "human_prior_source_chest_distance": self.source_chest_distance,
            "human_prior_target_chest_distance": self.target_chest_distance,
            "human_prior_chest_completed": self.chest_completed,
            "human_prior_source_life_signature": self.source_life_signature,
            "human_prior_target_life_signature": self.target_life_signature,
            "human_prior_life_counter_changed": self.life_counter_changed,
            "human_prior_dark_transition_started": self.dark_transition_started,
            "human_prior_life_loss_confirmed": self.life_loss_confirmed,
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
        chest_reward: float = 100.0,
        navigation_reward: float = 0.0,
        life_loss_penalty: float = 100.0,
        discovery_similarity: float = 0.98,
        presence_similarity: float = 0.55,
        maximum_event_visual_change: float = 0.08,
        minimum_scene_intensity: float = 0.05,
    ) -> None:
        self.heart_reward = float(heart_reward)
        self.all_hearts_reward = float(all_hearts_reward)
        self.chest_reward = float(chest_reward)
        self.navigation_reward = float(navigation_reward)
        self.life_loss_penalty = float(life_loss_penalty)
        self.discovery_similarity = float(discovery_similarity)
        self.presence_similarity = float(presence_similarity)
        self.maximum_event_visual_change = float(maximum_event_visual_change)
        self.minimum_scene_intensity = float(minimum_scene_intensity)
        self.known_slots: set[HeartSlot] = set()
        self.current_present: set[HeartSlot] = set()
        self.initialized = False
        self.best_remaining_hearts: Optional[int] = None
        self.current_life_signature: Optional[str] = None
        self.dark_transition_observed = False
        self._player_cache: OrderedDict[str, Optional[HeartSlot]] = OrderedDict()

    @staticmethod
    def _snap_to_tile(slot: HeartSlot) -> HeartSlot:
        return 16 * round(slot[0] / 16), 16 * round(slot[1] / 16)

    def detect_player(self, frame: Frame) -> Optional[HeartSlot]:
        """Locate Lolo from his visible palette, without reading emulator memory.

        The scan is deliberately part of the explicitly labelled human-prior
        track. Water uses Lolo's dark blue, so its two animated highlight
        colours are rejected before candidates are ranked.
        """

        digest = frame.digest
        if digest in self._player_cache:
            self._player_cache.move_to_end(digest)
            return self._player_cache[digest]
        if self.mean_intensity(frame) < self.minimum_scene_intensity:
            return None
        blue = (21, 95, 217)
        white = (255, 255, 255)
        black = (0, 0, 0)
        magenta = (183, 30, 123)
        water_highlights = {(100, 176, 255), (192, 223, 255)}
        best: Optional[Tuple[Tuple[int, ...], HeartSlot]] = None
        for y in range(28, min(frame.height - 15, 205), 4):
            for x in range(28, min(frame.width - 15, 205), 4):
                counts = {
                    blue: 0,
                    white: 0,
                    black: 0,
                    magenta: 0,
                }
                water_pixels = 0
                for row in range(y, y + 16):
                    for column in range(x, x + 16):
                        pixel = self._pixel(frame, column, row)
                        if pixel in counts:
                            counts[pixel] += 1
                        if pixel in water_highlights:
                            water_pixels += 1
                blue_pixels = counts[blue]
                white_pixels = counts[white]
                black_pixels = counts[black]
                magenta_pixels = counts[magenta]
                if not (
                    20 <= blue_pixels <= 120
                    and white_pixels >= 20
                    and black_pixels >= 25
                    and magenta_pixels < 35
                    and water_pixels < 10
                ):
                    continue
                rank = (
                    2 * blue_pixels + white_pixels,
                    blue_pixels,
                    white_pixels,
                    black_pixels,
                    -magenta_pixels,
                    x,
                    y,
                )
                if best is None or rank > best[0]:
                    best = (rank, (x, y))
        result = None if best is None else self._snap_to_tile(best[1])
        self._player_cache[digest] = result
        if len(self._player_cache) > 2048:
            self._player_cache.popitem(last=False)
        return result

    @staticmethod
    def _nearest_distance(
        player: Optional[HeartSlot], hearts: Iterable[HeartSlot]
    ) -> Optional[float]:
        if player is None:
            return None
        heart_slots = tuple(hearts)
        if not heart_slots:
            return None
        return min(
            (abs(player[0] - x) + abs(player[1] - y)) / 16.0
            for x, y in heart_slots
        )

    def distance_to_hearts(
        self,
        frame: Frame,
        hearts: Optional[Iterable[HeartSlot]] = None,
    ) -> Optional[float]:
        heart_slots = self.current_present if hearts is None else hearts
        return self._nearest_distance(self.detect_player(frame), heart_slots)

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
        return self._prototype_similarity(frame, slot, HEART_PROTOTYPE)

    def _prototype_similarity(
        self,
        frame: Frame,
        slot: HeartSlot,
        prototype: Sequence[Tuple[int, int, int]],
    ) -> float:
        x, y = slot
        if x < 0 or y < 0 or x + 16 > frame.width or y + 16 > frame.height:
            return 0.0
        matches = 0
        index = 0
        for row in range(y, y + 16):
            for column in range(x, x + 16):
                matches += self._pixel(frame, column, row) == prototype[index]
                index += 1
        return matches / len(prototype)

    def open_chest_similarity(self, frame: Frame, slot: HeartSlot) -> float:
        return self._prototype_similarity(frame, slot, OPEN_CHEST_PROTOTYPE)

    def detect_open_chest(self, frame: Frame) -> Optional[HeartSlot]:
        if self.mean_intensity(frame) < self.minimum_scene_intensity:
            return None
        best: Optional[Tuple[float, HeartSlot]] = None
        for y in range(32, min(frame.height - 15, 208), 16):
            for x in range(32, min(frame.width - 15, 208), 16):
                slot = (x, y)
                similarity = self.open_chest_similarity(frame, slot)
                if similarity < self.discovery_similarity:
                    continue
                if best is None or similarity > best[0]:
                    best = similarity, slot
        return None if best is None else best[1]

    def _life_signature(self, frame: Frame) -> Optional[str]:
        """Return the visible 8x8 HUD life glyph, never emulator memory.

        The signature records only white and magenta glyph pixels. Requiring
        both colours rejects dark transitions and partially drawn HUD frames.
        """

        if frame.width != 256 or frame.height != 240 or frame.channels < 3:
            return None
        values = []
        white_pixels = 0
        magenta_pixels = 0
        for y in range(48, 56):
            for x in range(232, 240):
                pixel = self._pixel(frame, x, y)
                if pixel == (255, 255, 255):
                    value = 1
                    white_pixels += 1
                elif pixel == (183, 30, 123):
                    value = 2
                    magenta_pixels += 1
                else:
                    value = 0
                values.append(value)
        if white_pixels < 10 or magenta_pixels < 2:
            return None
        return bytes(values).hex()

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
        life_signature = self._life_signature(frame)
        if life_signature is not None and self.current_life_signature is None:
            self.current_life_signature = life_signature
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
        target_dark = target_intensity <= 0.02
        visual_reliable = bool(
            target_intensity >= self.minimum_scene_intensity
            and visual_change <= self.maximum_event_visual_change
        )
        similarities = tuple(
            (x, y, self.similarity(target, (x, y)))
            for x, y in sorted(self.known_slots)
        )
        reliable = bool(
            self.initialized
            and visual_reliable
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
        source_player = (
            self.detect_player(source)
            if self.mean_intensity(source) >= self.minimum_scene_intensity
            else None
        )
        target_player = (
            self.detect_player(target)
            if target_intensity >= self.minimum_scene_intensity
            else None
        )
        source_distance = self._nearest_distance(
            source_player, target_present
        )
        target_distance = self._nearest_distance(
            target_player, target_present
        )
        source_chest = None
        target_chest = None
        source_chest_distance = None
        target_chest_distance = None
        chest_completed = False
        if not target_present:
            source_chest = self.detect_open_chest(source)
            target_chest = self.detect_open_chest(target)
            source_chest_distance = self._nearest_distance(
                source_player, () if source_chest is None else (source_chest,)
            )
            target_chest_distance = self._nearest_distance(
                target_player, () if target_chest is None else (target_chest,)
            )
            chest_completed = bool(
                source_chest is not None
                and target_chest is None
                and source_chest_distance is not None
                and source_chest_distance <= 1.0
                and (target_dark or visual_change > self.maximum_event_visual_change)
            )
        awarded_chest_reward = self.chest_reward if chest_completed else 0.0
        navigation_reward = 0.0
        if (
            reliable
            and not collected
            and source_distance is not None
            and target_distance is not None
        ):
            navigation_reward = self.navigation_reward * (
                source_distance - target_distance
            )
        elif (
            visual_reliable
            and not collected
            and source_chest is not None
            and target_chest is not None
            and source_chest_distance is not None
            and target_chest_distance is not None
        ):
            navigation_reward = self.navigation_reward * (
                source_chest_distance - target_chest_distance
            )
        source_life_signature = self._life_signature(source)
        target_life_signature = self._life_signature(target)
        life_counter_changed = bool(
            target_life_signature is not None
            and self.current_life_signature is not None
            and target_life_signature != self.current_life_signature
        )
        life_loss_confirmed = bool(
            life_counter_changed and self.dark_transition_observed
        )
        awarded_life_loss_penalty = (
            -self.life_loss_penalty if life_loss_confirmed else 0.0
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
            chest_reward=awarded_chest_reward,
            navigation_reward=navigation_reward,
            life_loss_penalty=awarded_life_loss_penalty,
            total_reward=(
                heart_reward
                + all_hearts_reward
                + awarded_chest_reward
                + navigation_reward
                + awarded_life_loss_penalty
            ),
            global_visual_change=visual_change,
            target_intensity=target_intensity,
            source_player_slot=source_player,
            target_player_slot=target_player,
            source_heart_distance=source_distance,
            target_heart_distance=target_distance,
            source_chest_slot=source_chest,
            target_chest_slot=target_chest,
            source_chest_distance=source_chest_distance,
            target_chest_distance=target_chest_distance,
            chest_completed=chest_completed,
            source_life_signature=source_life_signature,
            target_life_signature=target_life_signature,
            life_counter_changed=life_counter_changed,
            dark_transition_started=target_dark,
            life_loss_confirmed=life_loss_confirmed,
        )

    def commit(self, analysis: HeartGoalAnalysis, frame: Frame) -> None:
        if not self.initialized:
            self.observe_room(frame)
        if self.initialized and analysis.reliable:
            self.current_present = set(analysis.target_present)
            remaining = len(self.current_present)
            self.best_remaining_hearts = (
                remaining
                if self.best_remaining_hearts is None
                else min(self.best_remaining_hearts, remaining)
            )
        if analysis.dark_transition_started:
            self.dark_transition_observed = True
        elif analysis.target_life_signature is not None:
            if analysis.life_counter_changed:
                self.current_life_signature = analysis.target_life_signature
            self.dark_transition_observed = False

    def restore(self, present: Sequence[HeartSlot], frame: Frame) -> None:
        discovered = set(self.discover(frame))
        if discovered and not self.initialized:
            self.known_slots = discovered
            self.initialized = True
        if self.initialized:
            self.current_present = set(present) if present else discovered
        self.current_life_signature = self._life_signature(frame)
        self.dark_transition_observed = False

    def current_slots(self) -> Tuple[HeartSlot, ...]:
        return tuple(sorted(self.current_present))
