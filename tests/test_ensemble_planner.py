import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from lolo_agent.entity_behavior import AnonymousEntityBehaviorModel
from lolo_agent.ensemble_world_model import (
    EnsembleVisualDynamicsModel,
    VisualSequence,
    train_ensemble_model,
    validate_ensemble_model,
    load_ensemble_checkpoint,
    save_ensemble_checkpoint,
)
from lolo_agent.environment import Action
from lolo_agent.goal_prior import HeartGoalAnalysis
from lolo_agent.mock_puzzle import MockPuzzleEnv
from lolo_agent.neural_planner import (
    NeuralPlan,
    NeuralPlanningConfig,
    VerifiedNeuralAgent,
    _ArchivedBranch,
    _BehaviorProbeSelection,
    _HumanPriorOptionNode,
    _LifeHazardCheckpoint,
    _OptionCounterfactual,
    _TemporalOptionTrace,
)
from lolo_agent.pixels import Frame


class AutonomousAnimationEnv:
    def __init__(self) -> None:
        self.tick = 0

    def reset(self) -> Frame:
        self.tick = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        self.tick += frames
        return self._frame()

    def save_state(self) -> int:
        return self.tick

    def load_state(self, state: int) -> Frame:
        self.tick = state
        return self._frame()

    def _frame(self) -> Frame:
        return Frame(8, 8, 1, bytes([self.tick % 256]) * 64)


class PassiveRareEntityEnv:
    def __init__(
        self,
        duplicate: bool = False,
        minimum_motion_frames: int = 1,
    ) -> None:
        self.moved = False
        self.duplicate = duplicate
        self.minimum_motion_frames = minimum_motion_frames

    def reset(self) -> Frame:
        self.moved = False
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if (
            action == Action.NOOP
            and frames >= self.minimum_motion_frames
        ):
            self.moved = True
        return self._frame()

    def save_state(self) -> bool:
        return self.moved

    def load_state(self, state: bool) -> Frame:
        self.moved = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray([16] * (16 * 16))
        entity_column = 2 if self.moved else 1
        for y in range(4, 8):
            for x in range(entity_column * 4, entity_column * 4 + 4):
                pixels[y * 16 + x] = 224
        if self.duplicate:
            for y in range(12, 16):
                for x in range(12, 16):
                    pixels[y * 16 + x] = 224
        return Frame(16, 16, 1, bytes(pixels))


class CausalRareEntityEnv:
    def __init__(self, local_motion: bool = True) -> None:
        self.local_motion = local_motion
        self.player = (1, 3)
        self.entity_moved = False
        self.dead = False

    def reset(self) -> Frame:
        self.player = (1, 3)
        self.entity_moved = False
        self.dead = False
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if action == Action.RIGHT:
            self.player = (2, 3)
        elif action == Action.NOOP and self.player == (2, 3):
            if self.local_motion and frames >= 2:
                self.entity_moved = True
            if frames >= 3:
                self.dead = True
        return self._frame()

    def save_state(self) -> tuple[tuple[int, int], bool, bool]:
        return self.player, self.entity_moved, self.dead

    def load_state(
        self, state: tuple[tuple[int, int], bool, bool]
    ) -> Frame:
        self.player, self.entity_moved, self.dead = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray([16] * (16 * 16))
        entity = (2, 2) if self.entity_moved else (2, 1)
        for y in range(entity[1] * 4, entity[1] * 4 + 4):
            for x in range(entity[0] * 4, entity[0] * 4 + 4):
                pixels[y * 16 + x] = 224
        player_x = self.player[0] * 4 + 1
        player_y = self.player[1] * 4 + 1
        pixels[player_y * 16 + player_x] = 255
        if self.dead:
            pixels[0] = 80
        return Frame(16, 16, 1, bytes(pixels))


class AnimationPauseEnv:
    def __init__(self) -> None:
        self.tick = 0

    def reset(self) -> Frame:
        self.tick = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if self.tick >= 4 and action == Action.A:
            self.tick = 255
        elif self.tick < 4:
            self.tick = min(4, self.tick + frames)
        return self._frame()

    def save_state(self) -> int:
        return self.tick

    def load_state(self, state: int) -> Frame:
        self.tick = state
        return self._frame()

    def _frame(self) -> Frame:
        return Frame(
            8,
            8,
            1,
            bytes((index + self.tick) % 256 for index in range(64)),
        )


class DelayedCausalityEnv:
    def __init__(self) -> None:
        self.triggered = False
        self.tick = 0
        self.released = 0

    def reset(self) -> Frame:
        self.triggered = False
        self.tick = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if action == Action.START and self.tick == 0:
            self.triggered = True
        elif action == Action.NOOP:
            self.tick += frames
        return self._frame()

    def save_state(self) -> tuple[bool, int]:
        return self.triggered, self.tick

    def load_state(self, state: tuple[bool, int]) -> Frame:
        self.triggered, self.tick = state
        return self._frame()

    def release_state(self, state: tuple[bool, int]) -> None:
        self.released += 1

    def _frame(self) -> Frame:
        value = 255 if self.triggered and self.tick > 0 else 0
        return Frame(8, 8, 1, bytes([value]) * 64)


class ActionEffectEnv:
    def __init__(self) -> None:
        self.position = 0

    def reset(self) -> Frame:
        self.position = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if action == Action.RIGHT:
            self.position = min(63, self.position + frames)
        elif action == Action.SELECT:
            self.position = 63
        return self._frame()

    def save_state(self) -> int:
        return self.position

    def load_state(self, state: int) -> Frame:
        self.position = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[self.position] = 255
        return Frame(8, 8, 1, bytes(pixels))


class DynamicActionEffectEnv:
    def __init__(self) -> None:
        self.value = 0
        self.collapsed = False

    def reset(self) -> Frame:
        self.value = 0
        self.collapsed = False
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if self.collapsed:
            self.value = 64
        elif action == Action.SELECT:
            self.value = 128
        elif action == Action.RIGHT:
            self.value = min(255, self.value + 16)
            self.collapsed = True
        else:
            self.value = min(255, self.value + frames)
            self.collapsed = True
        return self._frame()

    def save_state(self) -> tuple[int, bool]:
        return self.value, self.collapsed

    def load_state(self, state: tuple[int, bool]) -> Frame:
        self.value, self.collapsed = state
        return self._frame()

    def _frame(self) -> Frame:
        return Frame(8, 8, 1, bytes([self.value]) * 64)


class TemporaryControlPauseEnv:
    def __init__(self) -> None:
        self.tick = 0

    def reset(self) -> Frame:
        self.tick = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if self.tick < 2:
            self.tick += 1
        elif action == Action.RIGHT:
            self.tick += 16
        return self._frame()

    def save_state(self) -> int:
        return self.tick

    def load_state(self, state: int) -> Frame:
        self.tick = state
        return self._frame()

    def _frame(self) -> Frame:
        return Frame(8, 8, 1, bytes([64 + self.tick]) * 64)


class NovelSceneTransitionEnv:
    def __init__(self) -> None:
        self.triggered = False
        self.tick = 0

    def reset(self) -> Frame:
        self.triggered = False
        self.tick = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if not self.triggered and action == Action.RIGHT:
            self.triggered = True
            self.tick = 0
        elif self.triggered:
            self.tick += 1
        return self._frame()

    def save_state(self) -> tuple[bool, int]:
        return self.triggered, self.tick

    def load_state(self, state: tuple[bool, int]) -> Frame:
        self.triggered, self.tick = state
        return self._frame()

    def _frame(self) -> Frame:
        if not self.triggered:
            pixels = bytes([64]) * 64
        elif self.tick < 2:
            pixels = bytes(64)
        else:
            pixels = bytes([224]) * 64
        return Frame(8, 8, 1, pixels)


class UniqueStateEnv(ActionEffectEnv):
    def __init__(self) -> None:
        super().__init__()
        self.serial = 0
        self.active_states = set()

    def reset(self) -> Frame:
        self.active_states = set()
        return super().reset()

    def save_state(self) -> tuple[int, int]:
        self.serial += 1
        state = (self.serial, self.position)
        self.active_states.add(state)
        return state

    def load_state(self, state: tuple[int, int]) -> Frame:
        if state not in self.active_states:
            raise RuntimeError("unknown save-state handle")
        self.position = state[1]
        return self._frame()

    def release_state(self, state: tuple[int, int]) -> None:
        if state not in self.active_states:
            raise RuntimeError("unknown save-state handle")
        self.active_states.remove(state)


class AutonomousPositionEnv:
    """Move the only tracked sprite with time, independent of input."""

    def __init__(self) -> None:
        self.tick = 0

    def reset(self) -> Frame:
        self.tick = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del action
        self.tick += frames
        return self._frame()

    def save_state(self) -> int:
        return self.tick

    def load_state(self, state: int) -> Frame:
        self.tick = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[self.tick % 8] = 255
        return Frame(8, 8, 1, bytes(pixels))


class DivergentPositionEnv:
    """Offer a high-reward short step and a farther regressive branch."""

    def __init__(self) -> None:
        self.position = 3

    def reset(self) -> Frame:
        self.position = 3
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.RIGHT:
            self.position = min(7, self.position + 1)
        elif action == Action.LEFT:
            self.position = 0
        elif action == Action.DOWN and self.position == 0:
            self.position = 1
        return self._frame()

    def save_state(self) -> int:
        return self.position

    def load_state(self, state: int) -> Frame:
        self.position = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[self.position] = 255
        return Frame(8, 8, 1, bytes(pixels))


class PositionGoalPrior:
    def __init__(self) -> None:
        self.known_slots = {(7, 0)}
        self.current_present = {(7, 0)}
        self.current_player_slot = (0, 0)
        self.best_remaining_hearts = 1
        self.navigation_reward = 1.0
        self.chest_obtained = False

    @staticmethod
    def _position(frame: Frame) -> tuple[int, int]:
        return frame.pixels.index(255), 0

    def current_slots(self):
        return tuple(sorted(self.current_present))

    def observe_room(self, frame: Frame):
        del frame
        return ()

    def analyze(
        self,
        source: Frame,
        target: Frame,
        *,
        target_player_reference=None,
    ) -> HeartGoalAnalysis:
        del target_player_reference
        source_player = self._position(source)
        target_player = self._position(target)
        navigation = float(target_player[0] - source_player[0])
        return HeartGoalAnalysis(
            reliable=True,
            known_slots=((7, 0),),
            source_present=((7, 0),),
            target_present=((7, 0),),
            collected=(),
            target_similarities=(),
            heart_reward=0.0,
            all_hearts_reward=0.0,
            chest_reward=0.0,
            navigation_reward=navigation,
            life_loss_penalty=0.0,
            total_reward=navigation,
            global_visual_change=source.mean_absolute_difference(target),
            target_intensity=1.0,
            source_player_slot=source_player,
            target_player_slot=target_player,
            source_heart_distance=float(7 - source_player[0]),
            target_heart_distance=float(7 - target_player[0]),
            source_chest_slot=None,
            target_chest_slot=None,
            source_chest_distance=None,
            target_chest_distance=None,
            chest_completed=False,
            source_life_signature="life",
            target_life_signature="life",
            life_counter_changed=False,
            dark_transition_started=False,
            life_loss_confirmed=False,
        )

    def restore(self, slots, frame: Frame, player_slot) -> None:
        self.current_present = set(slots)
        self.current_player_slot = player_slot or self._position(frame)

    def commit(self, analysis: HeartGoalAnalysis, frame: Frame) -> None:
        del frame
        self.current_present = set(analysis.target_present)
        self.current_player_slot = analysis.target_player_slot
        self.best_remaining_hearts = min(
            self.best_remaining_hearts,
            analysis.remaining_hearts,
        )

    def distance_to_hearts(self, frame: Frame, slots) -> float:
        player = self._position(frame)
        return float(min(abs(player[0] - slot[0]) for slot in slots))


class ReferenceSensitivePositionGoalPrior(PositionGoalPrior):
    """Require the prior tracked position to resolve a moved sprite."""

    def analyze(
        self,
        source: Frame,
        target: Frame,
        *,
        target_player_reference=None,
    ) -> HeartGoalAnalysis:
        analysis = super().analyze(
            source,
            target,
            target_player_reference=target_player_reference,
        )
        if (
            target_player_reference is None
            and analysis.target_player_slot != (0, 0)
        ):
            return replace(analysis, target_player_slot=(0, 0))
        return analysis


class HazardPositionGoalPrior(PositionGoalPrior):
    @staticmethod
    def _position(frame: Frame) -> tuple[int, int]:
        del frame
        return 0, 0

    def analyze(
        self,
        source: Frame,
        target: Frame,
        *,
        target_player_reference=None,
    ) -> HeartGoalAnalysis:
        analysis = super().analyze(
            source,
            target,
            target_player_reference=target_player_reference,
        )
        hazard = source.pixels != target.pixels
        return replace(
            analysis,
            target_life_signature="changed" if hazard else "life",
            life_counter_changed=hazard,
            life_loss_confirmed=hazard,
        )


class CausalEntityGoalPrior(PositionGoalPrior):
    @staticmethod
    def _position(frame: Frame) -> tuple[int, int]:
        index = frame.pixels.index(255)
        return index % frame.width, index // frame.width

    def analyze(
        self,
        source: Frame,
        target: Frame,
        *,
        target_player_reference=None,
    ) -> HeartGoalAnalysis:
        analysis = super().analyze(
            source,
            target,
            target_player_reference=target_player_reference,
        )
        hazard = target.pixels[0] == 80
        return replace(
            analysis,
            target_life_signature="changed" if hazard else "life",
            life_counter_changed=hazard,
            life_loss_confirmed=hazard,
        )


class OverlappingPlayerGoalPrior(PositionGoalPrior):
    @staticmethod
    def player_pixel_mask(
        frame: Frame,
        slot: tuple[int, int],
        search_padding: int = 12,
        dilation: int = 3,
    ) -> set[tuple[int, int]]:
        del frame, slot, search_padding, dilation
        return {(0, 0), (4, 0), (5, 0), (6, 0)}


class FootprintPositionGoalPrior(PositionGoalPrior):
    @staticmethod
    def player_pixel_mask(
        frame: Frame,
        slot: tuple[int, int],
        search_padding: int = 12,
        dilation: int = 3,
    ) -> set[tuple[int, int]]:
        del frame, search_padding, dilation
        return {slot}


class RegressivePositionGoalPrior(PositionGoalPrior):
    def analyze(
        self,
        source: Frame,
        target: Frame,
        *,
        target_player_reference=None,
    ) -> HeartGoalAnalysis:
        analysis = super().analyze(
            source,
            target,
            target_player_reference=target_player_reference,
        )
        regression = -abs(
            analysis.target_player_slot[0]
            - analysis.source_player_slot[0]
        )
        return replace(
            analysis,
            navigation_reward=float(regression),
            total_reward=float(regression),
        )


class OrderingPositionGoalPrior(PositionGoalPrior):
    def __init__(self) -> None:
        super().__init__()
        self.known_slots = {(-16, 0), (32, 0)}
        self.current_present = set(self.known_slots)
        self.best_remaining_hearts = 2

    def analyze(
        self,
        source: Frame,
        target: Frame,
        *,
        target_player_reference=None,
    ) -> HeartGoalAnalysis:
        analysis = super().analyze(
            source,
            target,
            target_player_reference=target_player_reference,
        )
        hearts = tuple(sorted(self.current_present))
        return replace(
            analysis,
            known_slots=tuple(sorted(self.known_slots)),
            source_present=hearts,
            target_present=hearts,
        )


class WorldEffectEnv:
    def __init__(
        self, persistent: bool, world_index: int | tuple[int, ...] = 63
    ) -> None:
        self.persistent = persistent
        self.world_indices = (
            (world_index,) if isinstance(world_index, int) else world_index
        )
        self.world_active = False

    def reset(self) -> Frame:
        self.world_active = False
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.RIGHT:
            self.world_active = True
        elif action == Action.NOOP and not self.persistent:
            self.world_active = False
        return self._frame()

    def save_state(self) -> bool:
        return self.world_active

    def load_state(self, state: bool) -> Frame:
        self.world_active = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[0] = 255
        if self.world_active:
            for world_index in self.world_indices:
                pixels[world_index] = 128
        return Frame(8, 8, 1, bytes(pixels))


class WorldEffectAndMovementEnv:
    def __init__(self) -> None:
        self.position = 0
        self.world_active = False

    def reset(self) -> Frame:
        self.position = 0
        self.world_active = False
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.RIGHT:
            self.position = min(1, self.position + 1)
        elif action == Action.A:
            self.world_active = True
        return self._frame()

    def save_state(self) -> tuple[int, bool]:
        return self.position, self.world_active

    def load_state(self, state: tuple[int, bool]) -> Frame:
        self.position, self.world_active = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[self.position] = 255
        if self.world_active:
            pixels[63] = 128
        return Frame(8, 8, 1, bytes(pixels))


class GoalDirectedEffectPriorityEnv:
    def __init__(self) -> None:
        self.position = 0
        self.effect = None

    def reset(self) -> Frame:
        self.position = 0
        self.effect = None
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.A:
            self.effect = 63
        elif action == Action.RIGHT:
            self.position = min(7, self.position + 1)
            self.effect = 62
        return self._frame()

    def save_state(self) -> tuple[int, int | None]:
        return self.position, self.effect

    def load_state(self, state: tuple[int, int | None]) -> Frame:
        self.position, self.effect = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[self.position] = 255
        if self.effect is not None:
            pixels[self.effect] = 128
        return Frame(8, 8, 1, bytes(pixels))


class ControllabilityGainEnv:
    def __init__(self) -> None:
        self.world_active = False
        self.position = 0

    def reset(self) -> Frame:
        self.world_active = False
        self.position = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.RIGHT:
            if self.world_active:
                self.position = 1
            else:
                self.world_active = True
        return self._frame()

    def save_state(self) -> tuple[bool, int]:
        return self.world_active, self.position

    def load_state(self, state: tuple[bool, int]) -> Frame:
        self.world_active, self.position = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[self.position] = 255
        if self.world_active:
            pixels[63] = 128
        return Frame(8, 8, 1, bytes(pixels))


class PoseControllabilityGainEnv:
    """Expose a false coarse-slot gain caused only by a different pose."""

    def __init__(self) -> None:
        self.pose = 0
        self.position = 0

    def reset(self) -> Frame:
        self.pose = 0
        self.position = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.LEFT:
            self.pose = 1
        elif action == Action.RIGHT and self.pose == 1:
            self.position = 1
        return self._frame()

    def save_state(self) -> tuple[int, int]:
        return self.pose, self.position

    def load_state(self, state: tuple[int, int]) -> Frame:
        self.pose, self.position = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[self.position] = 255
        pixels[10 + self.pose] = 255
        return Frame(8, 8, 1, bytes(pixels))


class PosePositionGoalPrior(PositionGoalPrior):
    @staticmethod
    def player_pixel_mask(
        frame: Frame,
        slot: tuple[int, int],
        search_padding: int = 12,
        dilation: int = 3,
    ) -> set[tuple[int, int]]:
        del slot, search_padding, dilation
        return {
            (index, 0)
            for index, value in enumerate(frame.pixels)
            if value == 255
        }


class LongPressMovementEnv:
    def __init__(self) -> None:
        self.position = 0

    def reset(self) -> Frame:
        self.position = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if action == Action.RIGHT and frames >= 16:
            self.position = 1
        return self._frame()

    def save_state(self) -> int:
        return self.position

    def load_state(self, state: int) -> Frame:
        self.position = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[self.position] = 255
        return Frame(8, 8, 1, bytes(pixels))


class DelayedControllabilityGainEnv:
    def __init__(self) -> None:
        self.world_active = False
        self.right_steps = 0
        self.position = 0

    def reset(self) -> Frame:
        self.world_active = False
        self.right_steps = 0
        self.position = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.A:
            self.world_active = True
        elif action == Action.RIGHT and self.world_active:
            self.right_steps += 1
            if self.right_steps >= 2:
                self.position = 1
        return self._frame()

    def save_state(self) -> tuple[bool, int, int]:
        return self.world_active, self.right_steps, self.position

    def load_state(self, state: tuple[bool, int, int]) -> Frame:
        self.world_active, self.right_steps, self.position = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[self.position] = 255
        if self.world_active:
            pixels[63] = 128
        return Frame(8, 8, 1, bytes(pixels))


class UnlabeledEntityTransformEnv:
    def __init__(
        self,
        entity_cell: tuple[int, int] = (1, 0),
        remote_display: bool = False,
    ) -> None:
        self.armed = False
        self.transformed = False
        self.entity_cell = entity_cell
        self.remote_display = remote_display

    def reset(self) -> Frame:
        self.armed = False
        self.transformed = False
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.RIGHT:
            self.armed = True
        elif action == Action.A and self.armed:
            self.transformed = True
        return self._frame()

    def save_state(self) -> tuple[bool, bool]:
        return self.armed, self.transformed

    def load_state(self, state: tuple[bool, bool]) -> Frame:
        self.armed, self.transformed = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(32 * 32)
        pixels[0] = 255
        entity_value = 224 if self.transformed else 32
        x_start = self.entity_cell[0] * 4
        y_start = self.entity_cell[1] * 4
        for y in range(y_start, y_start + 4):
            for x in range(x_start, x_start + 4):
                pixels[y * 32 + x] = entity_value
        if self.remote_display and self.transformed:
            for y in range(28, 32):
                for x in range(28, 32):
                    pixels[y * 32 + x] = 128
        return Frame(32, 32, 1, bytes(pixels))


class MovingEntitySettlesEnv:
    def __init__(self) -> None:
        self.armed = False
        self.transformed = False
        self.motion_step = 0

    def reset(self) -> Frame:
        self.armed = False
        self.transformed = False
        self.motion_step = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.RIGHT:
            self.armed = True
        elif action == Action.A and self.armed:
            self.transformed = True
            self.motion_step = 0
        elif action == Action.NOOP and self.transformed:
            self.motion_step = min(2, self.motion_step + 1)
        return self._frame()

    def save_state(self) -> tuple[bool, bool, int]:
        return self.armed, self.transformed, self.motion_step

    def load_state(self, state: tuple[bool, bool, int]) -> Frame:
        self.armed, self.transformed, self.motion_step = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(32 * 32)
        pixels[0] = 255
        entity_value = 224 if self.transformed else 32
        for y in range(4):
            for x in range(4, 8):
                pixels[y * 32 + x] = entity_value
        if self.transformed and self.motion_step < 2:
            moving_column = 2 + self.motion_step
            for y in range(4):
                for x in range(moving_column * 4, moving_column * 4 + 4):
                    pixels[y * 32 + x] = 128
        return Frame(32, 32, 1, bytes(pixels))


class MovingMilestoneSettlesEnv:
    def __init__(self) -> None:
        self.armed = False
        self.collected = False
        self.motion_step = 0

    def reset(self) -> Frame:
        self.armed = False
        self.collected = False
        self.motion_step = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.RIGHT and not self.collected:
            self.armed = True
        elif action == Action.A and self.armed:
            self.collected = True
            self.motion_step = 0
        elif action == Action.NOOP and self.collected:
            self.motion_step = min(2, self.motion_step + 1)
        return self._frame()

    def save_state(self) -> tuple[bool, bool, int]:
        return self.armed, self.collected, self.motion_step

    def load_state(self, state: tuple[bool, bool, int]) -> Frame:
        self.armed, self.collected, self.motion_step = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        if not self.collected:
            pixels[1 if self.armed else 0] = 255
            pixels[7] = 128
        elif self.motion_step >= 2:
            pixels[2] = 255
        return Frame(8, 8, 1, bytes(pixels))


class MovingMilestoneGoalPrior(PositionGoalPrior):
    @staticmethod
    def _optional_position(frame: Frame):
        try:
            return frame.pixels.index(255), 0
        except ValueError:
            return None

    def analyze(
        self,
        source: Frame,
        target: Frame,
        *,
        target_player_reference=None,
    ) -> HeartGoalAnalysis:
        del target_player_reference
        source_player = self._optional_position(source)
        target_player = self._optional_position(target)
        source_present = ((7, 0),) if source.pixels[7] == 128 else ()
        target_present = ((7, 0),) if target.pixels[7] == 128 else ()
        collected = (
            ((7, 0),) if source_present and not target_present else ()
        )
        heart_reward = 25.0 if collected else 0.0
        navigation = (
            0.0
            if source_player is None or target_player is None
            else float(target_player[0] - source_player[0])
        )
        return HeartGoalAnalysis(
            reliable=True,
            known_slots=((7, 0),),
            source_present=source_present,
            target_present=target_present,
            collected=collected,
            target_similarities=(),
            heart_reward=heart_reward,
            all_hearts_reward=0.0,
            chest_reward=0.0,
            navigation_reward=navigation,
            life_loss_penalty=0.0,
            total_reward=heart_reward + navigation,
            global_visual_change=source.mean_absolute_difference(target),
            target_intensity=1.0,
            source_player_slot=source_player,
            target_player_slot=target_player,
            source_heart_distance=None,
            target_heart_distance=None,
            source_chest_slot=None,
            target_chest_slot=None,
            source_chest_distance=None,
            target_chest_distance=None,
            chest_completed=False,
            source_life_signature="life",
            target_life_signature="life",
            life_counter_changed=False,
            dark_transition_started=False,
            life_loss_confirmed=False,
        )


class VisibleMovingMilestoneSettlesEnv(MovingMilestoneSettlesEnv):
    """Keep the moved player visible while its milestone animation settles."""

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        if not self.collected:
            pixels[1 if self.armed else 0] = 255
            pixels[7] = 128
        else:
            pixels[2] = 255
            if self.motion_step < 2:
                pixels[3 + self.motion_step] = 64
        return Frame(8, 8, 1, bytes(pixels))


class RecordingMovingMilestoneGoalPrior(MovingMilestoneGoalPrior):
    def __init__(self) -> None:
        super().__init__()
        self.analysis_calls = []

    def analyze(
        self,
        source: Frame,
        target: Frame,
        *,
        target_player_reference=None,
    ) -> HeartGoalAnalysis:
        self.analysis_calls.append(
            (source.digest, target.digest, target_player_reference)
        )
        return super().analyze(
            source,
            target,
            target_player_reference=target_player_reference,
        )


class PlayerOverlapEntityEnv:
    def __init__(self) -> None:
        self.armed = False
        self.transformed = False
        self.pose = Action.RIGHT

    def reset(self) -> Frame:
        self.armed = False
        self.transformed = False
        self.pose = Action.RIGHT
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action in (Action.UP, Action.DOWN, Action.RIGHT):
            self.pose = action
        if action == Action.RIGHT:
            self.armed = True
        elif action == Action.A and self.armed:
            self.transformed = True
        return self._frame()

    def save_state(self) -> tuple[bool, bool, Action]:
        return self.armed, self.transformed, self.pose

    def load_state(self, state: tuple[bool, bool, Action]) -> Frame:
        self.armed, self.transformed, self.pose = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(32 * 32)
        pixels[0] = 255
        entity_value = 224 if self.transformed else 32
        for y in range(4):
            for x in range(4, 8):
                pixels[y * 32 + x] = entity_value
        pose_x = {
            Action.UP: 4,
            Action.DOWN: 5,
            Action.RIGHT: 6,
        }[self.pose]
        pixels[pose_x] = 255
        return Frame(32, 32, 1, bytes(pixels))


class TemporalUnlabeledEntityTransformEnv:
    def __init__(self) -> None:
        self.armed = False
        self.primed = False
        self.ready = False
        self.transformed = False

    def reset(self) -> Frame:
        self.armed = False
        self.primed = False
        self.ready = False
        self.transformed = False
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.RIGHT:
            self.armed = True
        elif action == Action.NOOP and self.primed:
            self.ready = True
        elif action == Action.A and self.ready:
            self.transformed = True
        elif action == Action.A and self.armed:
            self.primed = True
        return self._frame()

    def save_state(self) -> tuple[bool, bool, bool, bool]:
        return self.armed, self.primed, self.ready, self.transformed

    def load_state(
        self, state: tuple[bool, bool, bool, bool]
    ) -> Frame:
        self.armed, self.primed, self.ready, self.transformed = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(32 * 32)
        pixels[0] = 255
        entity_value = 224 if self.transformed else 32
        for y in range(4):
            for x in range(4, 8):
                pixels[y * 32 + x] = entity_value
        display_value = 192 if self.transformed else 128 if self.ready else 64
        if self.primed:
            for y in range(28, 32):
                for x in range(28, 32):
                    pixels[y * 32 + x] = display_value
        return Frame(32, 32, 1, bytes(pixels))


class MultiStateUnlabeledEntityTransformEnv:
    def __init__(self) -> None:
        self.armed = False
        self.entity_state = 0

    def reset(self) -> Frame:
        self.armed = False
        self.entity_state = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.RIGHT:
            self.armed = True
        elif action == Action.A and self.armed:
            self.entity_state = min(2, self.entity_state + 1)
        return self._frame()

    def save_state(self) -> tuple[bool, int]:
        return self.armed, self.entity_state

    def load_state(self, state: tuple[bool, int]) -> Frame:
        self.armed, self.entity_state = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(32 * 32)
        pixels[0] = 255
        entity_value = (32, 128, 224)[self.entity_state]
        for y in range(4):
            for x in range(4, 8):
                pixels[y * 32 + x] = entity_value
        return Frame(32, 32, 1, bytes(pixels))


class PhaseShiftWorldEffectEnv:
    def __init__(self) -> None:
        self.tick = 0
        self.phase_shift = 0

    def reset(self) -> Frame:
        self.tick = 0
        self.phase_shift = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if action == Action.RIGHT:
            self.phase_shift = 1
        self.tick += frames
        return self._frame()

    def save_state(self) -> tuple[int, int]:
        return self.tick, self.phase_shift

    def load_state(self, state: tuple[int, int]) -> Frame:
        self.tick, self.phase_shift = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[0] = 255
        pixels[63] = (0, 64, 128, 192)[
            (self.tick + self.phase_shift) % 4
        ]
        return Frame(8, 8, 1, bytes(pixels))


class RecordingLogger:
    def __init__(self) -> None:
        self.events = []

    def log(self, event_type: str, **fields) -> None:
        self.events.append({"event": event_type, **fields})


class AdversarialSpatialShadow:
    """A shadow that strongly prefers the action the real planner ranks last."""

    def score_plans(self, _frame, plans):
        return [
            {
                "spatial_shadow_score": float(index * 1_000_000),
                "spatial_shadow_predicted_effect": float(index),
                "spatial_shadow_predicted_change": float(index),
                "spatial_shadow_uncertainty": 0.0,
            }
            for index, _plan in enumerate(plans)
        ]

    def evaluate_transition(self, _source, _action, _duration, _target):
        return {
            "spatial_shadow_pixel_l1": 0.1,
            "spatial_shadow_persistence_l1": 0.2,
            "spatial_shadow_predicted_pixel_change": 0.1,
            "spatial_shadow_effect_weighted_pixel_l1": 0.1,
            "spatial_shadow_effect_weighted_persistence_l1": 0.2,
            "spatial_shadow_beats_persistence": True,
            "spatial_shadow_effect_l1": 0.1,
            "spatial_shadow_effect_f1": 0.5,
            "spatial_shadow_predicted_effect": 0.2,
            "spatial_shadow_actual_effect": 0.3,
            "spatial_shadow_uncertainty": 0.0,
        }


class EnsemblePlannerTests(unittest.TestCase):
    def frame(self, offset: int) -> Frame:
        return Frame(32, 32, 3, bytes((index + offset) % 256 for index in range(32 * 32 * 3)))

    def test_multistep_training_and_validation(self) -> None:
        sequences = [
            VisualSequence(
                group=index,
                frames=(self.frame(index), self.frame(index + 1), self.frame(index + 2)),
                actions=(Action.RIGHT, Action.DOWN),
            )
            for index in range(2)
        ]
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        history = train_ensemble_model(model, sequences, "cpu", epochs=1, batch_size=2)
        report = validate_ensemble_model(model, sequences, "cpu", batch_size=2)
        self.assertTrue(history)
        self.assertEqual(len(report.horizon_pixel_l1), 2)
        self.assertEqual(len(report.horizon_uncertainty), 2)
        self.assertTrue(all(value >= 0 for value in report.horizon_uncertainty))

    def test_bidirectional_probe_cannot_change_committed_decision(self) -> None:
        torch.manual_seed(13)
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        base = dict(
            actions=(Action.RIGHT, Action.LEFT, Action.NOOP),
            planning_depth=1,
            beam_width=3,
            verify_actions=3,
            action_frames=1,
            visual_stagnation_visits=99,
        )
        control = VerifiedNeuralAgent(
            ActionEffectEnv(), model, "cpu", NeuralPlanningConfig(**base)
        )
        probed = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                **base,
                returnability_probe_depth=1,
                returnability_probe_beam_width=2,
                returnability_probe_pixel_l1_threshold=0.0,
            ),
        )
        control.reset()
        probed.reset()

        control_decision = control.decide()
        probed_decision = probed.decide()

        self.assertEqual(probed_decision.action, control_decision.action)
        self.assertEqual(probed_decision.action_frames, control_decision.action_frames)
        self.assertEqual(probed_decision.frame, control_decision.frame)
        self.assertAlmostEqual(probed_decision.score, control_decision.score)

    def test_mixed_horizon_training_and_validation(self) -> None:
        sequences = [
            VisualSequence(
                group=0,
                frames=(self.frame(0), self.frame(1)),
                actions=(Action.RIGHT,),
                durations=(1,),
            ),
            VisualSequence(
                group=1,
                frames=(self.frame(1), self.frame(2), self.frame(3)),
                actions=(Action.DOWN, Action.LEFT),
                durations=(2, 4),
            ),
        ]
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
        )
        history = train_ensemble_model(model, sequences, "cpu", epochs=1, batch_size=2)
        report = validate_ensemble_model(model, sequences, "cpu", batch_size=2)
        self.assertEqual(len(history), 2)
        self.assertEqual(len(report.horizon_pixel_l1), 2)

    def test_verified_planner_preserves_frozen_model(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                planning_depth=2,
                beam_width=4,
                verify_actions=2,
                action_frames=1,
            ),
        )
        before = model.checkpoint_digest
        agent.reset()
        decision = agent.decide()
        self.assertIn(decision.action, (Action.LEFT, Action.RIGHT))
        self.assertEqual(decision.branches_examined, 2)
        self.assertEqual(before, model.checkpoint_digest)

    def test_known_milestone_suppresses_only_repeated_goal_bonus(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = MovingMilestoneSettlesEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                human_prior_heart_reward=25.0,
                human_prior_navigation_reward=1.0,
                human_prior_intrinsic_clip=10.0,
            ),
        )
        source = env.reset()
        env.armed = True
        env.collected = True
        env.motion_step = 2
        target = env._frame()
        analysis = MovingMilestoneGoalPrior().analyze(source, target)

        novel_score, novel_intrinsic = agent._human_prior_score(
            500.0, analysis
        )
        outcome_key = agent._human_prior_milestone_outcome_key(analysis)
        agent.human_prior_milestone_outcomes.add(outcome_key)
        repeated_score, repeated_intrinsic = agent._human_prior_score(
            500.0, analysis
        )

        self.assertEqual(novel_intrinsic, repeated_intrinsic)
        self.assertEqual(novel_score - repeated_score, 25.0)
        self.assertEqual(repeated_score, repeated_intrinsic + 2.0)
        self.assertTrue(agent._human_prior_milestone_outcome_known(analysis))

    def test_spatial_shadow_is_logged_but_cannot_change_selection(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                action_frames=1,
                actual_novelty_weight=0.0,
                scene_novelty_weight=0.0,
                prediction_error_weight=0.0,
                actual_change_weight=0.0,
                action_effect_weight=0.0,
                causal_spatial_novelty_weight=0.0,
                frontier_score_weight=0.0,
                temporal_option_score_weight=0.0,
                action_coverage_weight=0.0,
                duration_coverage_weight=0.0,
                consecutive_repeat_weight=0.0,
            ),
            event_logger=logger,
            spatial_shadow=AdversarialSpatialShadow(),
        )
        agent.reset()
        agent.planner.plan = lambda _frame: [
            NeuralPlan((Action.LEFT,), (1,), 10.0, 0.0),
            NeuralPlan((Action.RIGHT,), (1,), 0.0, 0.0),
        ]

        decision = agent.decide()

        self.assertEqual(decision.action, Action.LEFT)
        candidates = next(
            event for event in logger.events if event["event"] == "planner_candidates"
        )["candidates"]
        self.assertEqual(candidates[1]["spatial_shadow_score"], 1_000_000.0)
        self.assertTrue(
            all(item["spatial_shadow_selection_weight"] == 0.0 for item in candidates)
        )
        shadow_events = [
            event
            for event in logger.events
            if event["event"] == "spatial_shadow_branch_evaluated"
        ]
        self.assertEqual(len(shadow_events), 2)
        self.assertTrue(
            all(event["spatial_shadow_selection_weight"] == 0.0 for event in shadow_events)
        )

    def test_spatial_weight_prioritizes_verification_not_verified_commit(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                action_frames=1,
                actual_novelty_weight=0.0,
                scene_novelty_weight=0.0,
                prediction_error_weight=0.0,
                actual_change_weight=0.0,
                action_effect_weight=0.0,
                causal_spatial_novelty_weight=0.0,
                frontier_score_weight=0.0,
                temporal_option_score_weight=0.0,
                action_coverage_weight=0.0,
                duration_coverage_weight=0.0,
                consecutive_repeat_weight=0.0,
                spatial_selection_weight=1.0,
            ),
            event_logger=logger,
            spatial_shadow=AdversarialSpatialShadow(),
        )
        agent.reset()
        agent.planner.plan = lambda _frame: [
            NeuralPlan((Action.LEFT,), (1,), 10.0, 0.0),
            NeuralPlan((Action.RIGHT,), (1,), 0.0, 0.0),
        ]

        decision = agent.decide()

        self.assertEqual(decision.action, Action.LEFT)
        candidates = next(
            event for event in logger.events if event["event"] == "planner_candidates"
        )["candidates"]
        self.assertTrue(
            all(
                item["spatial_shadow_mode"] == "verification_priority"
                for item in candidates
            )
        )
        self.assertTrue(
            all(item["spatial_shadow_selection_weight"] == 1.0 for item in candidates)
        )
        shadow_branches = [
            event
            for event in logger.events
            if event["event"] == "spatial_shadow_branch_evaluated"
        ]
        self.assertEqual(shadow_branches[0]["action"], Action.RIGHT)
        committed = next(
            event for event in logger.events if event["event"] == "decision_committed"
        )
        self.assertEqual(
            committed["spatial_selection_mode"], "verification_priority"
        )
        self.assertEqual(committed["spatial_selection_weight"], 1.0)
        self.assertEqual(committed["spatial_selection_bonus"], 0.0)
        self.assertFalse(committed["spatial_selection_applied_to_commit"])

    def test_checkpoint_round_trip_is_frozen(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.pt"
            digest = save_ensemble_checkpoint(model, path, planning_horizon=3)
            loaded, horizon = load_ensemble_checkpoint(path, frozen=True)
        self.assertEqual(horizon, 3)
        self.assertEqual(digest, loaded.checkpoint_digest)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))

    def test_temporary_action_coverage_breaks_repetition(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                action_frames=1,
                action_coverage_weight=10.0,
                consecutive_repeat_weight=10.0,
            ),
        )
        agent.reset()
        actions = [decision.action for decision in agent.run(2)]
        self.assertEqual(set(actions), {Action.LEFT, Action.RIGHT})

    def test_duration_coverage_is_scoped_to_the_controller_action(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.UP),
                planning_depth=1,
                duration_coverage_weight=1.0,
            ),
        )
        agent.reset()
        agent.action_duration_counts[(Action.NOOP, 16)] = 100

        self.assertEqual(agent._action_penalty(Action.UP, 16), 0.0)
        self.assertEqual(agent._action_penalty(Action.NOOP, 16), 10.0)

    def test_delayed_return_penalty_can_be_capped_without_disabling_coverage(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.UP,),
                planning_depth=1,
                action_coverage_weight=1.0,
                duration_coverage_weight=1.0,
                consecutive_repeat_weight=1.0,
                delayed_return_weight=1.0,
                delayed_return_penalty_cap=2.0,
            ),
        )
        agent.action_counts[Action.UP] = 9
        agent.action_duration_counts[(Action.UP, 4)] = 4
        agent.last_action = Action.UP
        agent.last_duration = 4
        agent.action_streak = 1
        agent.current_scene = "scene"
        agent.delayed_return_costs[("scene", Action.UP, 4)] = 100

        components = agent._action_penalty_components(Action.UP, 4)

        self.assertEqual(components["action_coverage_penalty"], 3.0)
        self.assertEqual(components["duration_coverage_penalty"], 2.0)
        self.assertEqual(components["consecutive_repeat_penalty"], 1.0)
        self.assertEqual(components["delayed_return_penalty_raw"], 10.0)
        self.assertEqual(components["delayed_return_penalty"], 2.0)
        self.assertEqual(components["action_penalty"], 8.0)

    def test_matched_noop_branch_prioritizes_discovered_control(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.A, Action.RIGHT),
                planning_depth=1,
                beam_width=3,
                verify_actions=3,
                action_frames=4,
                actual_novelty_weight=0.0,
                scene_novelty_weight=0.0,
                prediction_error_weight=0.0,
                actual_change_weight=0.0,
                action_effect_weight=1.0,
                action_coverage_weight=0.0,
                duration_coverage_weight=0.0,
                consecutive_repeat_weight=0.0,
            ),
        )
        agent.reset()
        plans = [
            NeuralPlan((action,), (4,), 0.0, 0.0)
            for action in (Action.NOOP, Action.A, Action.RIGHT)
        ]
        agent.planner.plan = lambda _frame: plans

        decision = agent.decide()

        self.assertEqual(decision.action, Action.RIGHT)
        sources = {source for source, _action in agent.action_effect_samples}
        self.assertEqual(len(sources), 1)
        source = sources.pop()
        self.assertEqual(agent._action_effect_estimate(source, Action.A)[0], 0.0)
        self.assertEqual(agent._action_effect_estimate(source, Action.RIGHT)[0], 1.0)
        self.assertEqual(sum(agent.causal_spatial_visits.values()), 1)

    def test_causal_option_commits_a_neutral_observation_before_intervening(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=9, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            AutonomousAnimationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.RIGHT, Action.SELECT),
                planning_depth=1,
                beam_width=3,
                verify_actions=3,
                action_frames=4,
                actual_novelty_weight=0.0,
                scene_novelty_weight=0.0,
                prediction_error_weight=0.0,
                actual_change_weight=0.0,
                action_effect_weight=0.0,
                causal_spatial_novelty_weight=1.0,
                action_coverage_weight=0.0,
                duration_coverage_weight=0.0,
                consecutive_repeat_weight=0.0,
            ),
        )
        agent.reset()
        agent.pending_option_choice = ("source", Action.RIGHT, 4)
        agent.pending_option_decision = 0
        agent.pending_option_causal_evidence = True
        plans = [
            NeuralPlan((Action.NOOP,), (4,), 0.0, 0.0),
            NeuralPlan((Action.RIGHT,), (4,), 2.0, 0.0),
            NeuralPlan((Action.SELECT,), (4,), 1.0, 0.0),
        ]
        agent.planner.plan = lambda _frame: plans

        decision = agent.decide()

        self.assertEqual(decision.action, Action.NOOP)

    def test_causal_spatial_signature_localizes_matched_pixel_change(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                causal_spatial_columns=1,
                causal_spatial_rows=1,
            ),
        )
        neutral_pixels = bytearray(64)
        neutral_pixels[0] = 255
        factual_pixels = bytearray(64)
        factual_pixels[4] = 255

        signature, changed_pixels, centroid = agent._causal_spatial_effect(
            Frame(8, 8, 1, bytes(factual_pixels)),
            Frame(8, 8, 1, bytes(neutral_pixels)),
        )

        self.assertIsNotNone(signature)
        self.assertEqual(changed_pixels, 2)
        self.assertEqual(centroid, (2.0, 0.0))

    def test_causal_spatial_signature_masks_pixels_and_requires_support(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                causal_spatial_columns=1,
                causal_spatial_rows=1,
            ),
        )
        factual_pixels = bytearray(64)
        factual_pixels[:4] = bytes([255]) * 4
        neutral = Frame(8, 8, 1, bytes(64))

        signature, changed_pixels, _ = agent._causal_spatial_effect(
            Frame(8, 8, 1, bytes(factual_pixels)),
            neutral,
            ignored_pixels={(0, 0)},
            minimum_cell_pixels=4,
        )

        self.assertIsNone(signature)
        self.assertEqual(changed_pixels, 3)

    def test_causal_cell_coverage_decays_across_the_attempt(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                causal_spatial_columns=4,
                causal_spatial_rows=4,
                causal_cell_coverage_weight=4.0,
            ),
        )
        agent.reset()
        occupied = bytes([0] * 5 + [1, 1] + [0] * 9).hex()

        self.assertEqual(agent._causal_cell_coverage(occupied), (1.0, 2, 2))
        agent.causal_spatial_cell_visits[(1, 1)] = 3
        self.assertEqual(agent._causal_cell_coverage(occupied), (0.75, 1, 2))

        branch = _ArchivedBranch(
            1,
            agent.frame,
            NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0),
            0.0,
            "scene",
            1,
            causal_spatial_signature=occupied,
        )
        self.assertEqual(agent._archive_causal_cell_coverage_bonus(branch), 3.0)

    def test_causal_cell_coverage_weight_must_be_non_negative(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        with self.assertRaisesRegex(ValueError, "coverage weight"):
            VerifiedNeuralAgent(
                ActionEffectEnv(),
                model,
                "cpu",
                NeuralPlanningConfig(
                    actions=(Action.RIGHT,),
                    planning_depth=1,
                    causal_cell_coverage_weight=-1.0,
                ),
            )

    def test_behavioral_edge_coverage_counts_only_committed_interventions(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                behavioral_edge_coverage_weight=4.0,
            ),
        )
        agent.reset()

        self.assertEqual(
            agent._behavioral_edge_coverage("behavior-1", Action.RIGHT, 4),
            (0, True, 4.0),
        )
        self.assertEqual(
            agent._behavioral_edge_coverage("behavior-1", Action.NOOP, 4),
            (0, False, 0.0),
        )
        self.assertEqual(
            agent._record_behavioral_edge("behavior-1", Action.RIGHT, 4),
            0,
        )
        visits, unexpanded, bonus = agent._behavioral_edge_coverage(
            "behavior-1", Action.RIGHT, 4
        )
        self.assertEqual((visits, unexpanded), (1, False))
        self.assertAlmostEqual(bonus, 4.0 / (2.0**0.5))

        agent._migrate_frontier_signature("behavior-1", "behavior-2")
        self.assertEqual(
            agent.behavioral_edge_visits[
                ("behavior-1", Action.RIGHT, 4)
            ],
            0,
        )
        self.assertEqual(
            agent.behavioral_edge_visits[
                ("behavior-2", Action.RIGHT, 4)
            ],
            1,
        )

    def test_behavioral_edge_coverage_weight_must_be_non_negative(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        with self.assertRaisesRegex(ValueError, "behavioral edge coverage"):
            VerifiedNeuralAgent(
                ActionEffectEnv(),
                model,
                "cpu",
                NeuralPlanningConfig(
                    actions=(Action.RIGHT,),
                    planning_depth=1,
                    behavioral_edge_coverage_weight=-1.0,
                ),
            )

    def test_causal_cell_recovery_grace_must_be_non_negative(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        with self.assertRaisesRegex(ValueError, "recovery grace"):
            VerifiedNeuralAgent(
                ActionEffectEnv(),
                model,
                "cpu",
                NeuralPlanningConfig(
                    actions=(Action.RIGHT,),
                    planning_depth=1,
                    causal_cell_recovery_grace_decisions=-1,
                ),
            )

    def test_causal_cell_progress_temporarily_suppresses_recovery(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                causal_cell_recovery_grace_decisions=4,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.delayed_return_recovery = True
        agent.delayed_return_loop_start = 0
        agent.last_causal_cell_progress_decision = 0

        restored = agent._restore_if_stagnant()

        self.assertIsNone(restored)
        self.assertFalse(agent.delayed_return_recovery)
        suppressions = [
            event
            for event in logger.events
            if event["event"] == "causal_cell_recovery_suppressed"
        ]
        self.assertEqual(len(suppressions), 1)
        self.assertEqual(suppressions[0]["grace_decisions"], 4)

    def test_causal_cell_progress_suppresses_visual_stagnation_recovery(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
                causal_cell_recovery_grace_decisions=4,
            ),
            event_logger=logger,
        )
        frame = agent.reset()
        agent.archive = [
            _ArchivedBranch(
                env.save_state(),
                frame,
                NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                1.0,
                "other-scene",
                0,
            )
        ]
        agent.visual_stagnation_streak = 1
        agent.last_causal_cell_progress_decision = 0

        restored = agent._restore_if_stagnant()

        self.assertIsNone(restored)
        self.assertEqual(len(agent.archive), 1)
        suppression = next(
            event
            for event in logger.events
            if event["event"] == "causal_cell_recovery_suppressed"
        )
        self.assertEqual(suppression["recovery_reason"], "visual_stagnation")

    def test_archive_recovery_never_restores_an_all_hazard_frontier(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                behavioral_best_first_archive=True,
            ),
            event_logger=logger,
        )
        agent.reset()
        root = env.save_state()
        target = env.step(Action.RIGHT)
        target_state = env.save_state()
        env.load_state(root)
        choice = ("source", Action.RIGHT, 1)
        agent.archive = [
            _ArchivedBranch(
                target_state,
                target,
                NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                1.0,
                "target-scene",
                0,
                origin_signature="source",
            )
        ]
        agent.temporal_option_values[choice] = -2.0
        agent.temporal_option_samples[choice] = 1
        agent.delayed_return_recovery = True
        agent.delayed_return_loop_start = 0

        restored = agent._restore_if_stagnant()

        self.assertIsNone(restored)
        self.assertEqual(len(agent.archive), 1)
        self.assertFalse(agent.delayed_return_recovery)
        exhausted = next(
            event
            for event in logger.events
            if event["event"]
            == "archive_recovery_exhausted_by_learned_hazards"
        )
        self.assertEqual(exhausted["filtered"], 1)

    def test_dark_transition_return_to_known_scene_restores_archive(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,), planning_depth=1
            ),
            event_logger=logger,
        )
        agent.reset()
        constant = lambda value: Frame(
            32, 32, 3, bytes([value]) * (32 * 32 * 3)
        )
        known = constant(128)
        returned = constant(129)
        dark = constant(0)
        safe = constant(200)
        agent.frame = returned
        agent.bright_scene_memory = [
            agent._persistent_cell_values(known)
        ]
        agent.archive = [
            _ArchivedBranch(
                env.save_state(),
                safe,
                NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                1.0,
                "safe-scene",
                0,
                origin_signature="safe",
            )
        ]
        agent.decision_index = 2

        agent._observe_dark_transition(dark)
        agent._observe_dark_transition(returned)
        restored = agent._restore_if_stagnant()

        self.assertTrue(restored.restored_archive)
        self.assertEqual(restored.frame.digest, safe.digest)
        resolved = next(
            event
            for event in logger.events
            if event["event"] == "generic_dark_transition_resolved"
        )
        self.assertTrue(resolved["returned_to_known_scene"])
        committed = [
            event
            for event in logger.events
            if event["event"] == "decision_committed"
        ][-1]
        self.assertEqual(
            committed["restore_reason"],
            "known_scene_return_after_dark_transition",
        )

    def test_dark_transition_to_novel_scene_does_not_restore(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,), planning_depth=1
            ),
        )
        agent.reset()
        constant = lambda value: Frame(
            32, 32, 3, bytes([value]) * (32 * 32 * 3)
        )
        known = constant(128)
        novel = constant(224)
        agent.bright_scene_memory = [
            agent._persistent_cell_values(known)
        ]

        agent._observe_dark_transition(constant(0))
        agent._observe_dark_transition(novel)

        self.assertFalse(agent.known_scene_return_recovery_pending)
        self.assertEqual(len(agent.bright_scene_memory), 2)
        self.assertEqual(agent.pending_novel_room_frame, novel)

    def test_novel_room_boundary_resets_coordinate_local_memory(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,), planning_depth=1
            ),
            event_logger=logger,
        )
        agent.reset()
        novel = Frame(32, 32, 3, bytes([224]) * (32 * 32 * 3))
        agent.frame = novel
        agent.causal_spatial_cell_visits[(2, 3)] = 4
        agent.last_causal_cell_progress_decision = 7
        agent.persistent_change_cells[0] = 0
        agent.persistent_change_candidates[1] = (2, 3)
        agent.persistent_change_mismatches[2] = 1
        agent.pending_novel_room_frame = novel

        agent._apply_pending_novel_room_reset()

        self.assertIsNone(agent.pending_novel_room_frame)
        self.assertEqual(agent.causal_spatial_cell_visits, {})
        self.assertIsNone(agent.last_causal_cell_progress_decision)
        self.assertEqual(agent.persistent_change_cells, {})
        self.assertEqual(agent.persistent_change_candidates, {})
        self.assertEqual(agent.persistent_change_mismatches, {})
        self.assertEqual(
            agent.persistent_change_baseline,
            list(agent._persistent_cell_values(novel)),
        )
        self.assertTrue(
            any(
                event["event"] == "pixel_novel_room_started"
                for event in logger.events
            )
        )

    def test_behavioral_best_first_restore_prefers_unexpanded_edge(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.LEFT),
                planning_depth=1,
                visual_stagnation_visits=1,
                behavioral_best_first_archive=True,
            ),
            event_logger=logger,
        )
        current = agent.reset()
        scene = agent._scene_signature(current)
        repeated_frame = Frame(8, 8, 1, bytes([10]) + bytes(63))
        unexpanded_frame = Frame(8, 8, 1, bytes([20]) + bytes(63))
        repeated = _ArchivedBranch(
            state=env.save_state(),
            frame=repeated_frame,
            plan=NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0),
            score=100.0,
            scene=scene,
            created=10,
            origin_signature="behavior-1",
            causal_spatial_signature="01",
            causal_context_signature="causal-context-root",
            causal_event_outcome=True,
        )
        unexpanded = _ArchivedBranch(
            state=env.save_state(),
            frame=unexpanded_frame,
            plan=NeuralPlan((Action.LEFT,), (4,), 0.0, 0.0),
            score=0.0,
            scene=scene,
            created=1,
            origin_signature="behavior-1",
            causal_spatial_signature="02",
            causal_context_signature="causal-context-root",
            parent_state_id="state-parent",
            parent_frame_digest="frame-parent",
            parent_decision=7,
            search_depth=3,
        )
        agent.archive = [repeated, unexpanded]
        agent.behavioral_edge_visits[
            ("behavior-1", Action.RIGHT, 4)
        ] = 3
        agent.delayed_return_recovery = True
        agent.delayed_return_loop_start = 10

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, unexpanded_frame.digest)
        self.assertEqual(
            agent.behavioral_edge_visits[
                ("behavior-1", Action.LEFT, 4)
            ],
            1,
        )
        filtered = [
            event
            for event in logger.events
            if event["event"]
            == "behavioral_best_first_archives_filtered"
        ]
        self.assertEqual(len(filtered), 1)
        global_frontier = [
            event
            for event in logger.events
            if event["event"] == "behavioral_best_first_global_archive"
        ]
        self.assertEqual(len(global_frontier), 1)
        committed = [
            event
            for event in logger.events
            if event["event"] == "decision_committed"
        ][-1]
        self.assertTrue(committed["behavioral_edge_unexpanded"])
        self.assertTrue(committed["behavioral_best_first_applied"])
        self.assertEqual(committed["parent_state_id"], "state-parent")
        self.assertEqual(committed["parent_frame"], "frame-parent")
        self.assertEqual(committed["parent_decision"], 7)
        self.assertEqual(committed["search_depth"], 3)

    def test_human_prior_best_first_uses_stable_goal_state_edges(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.LEFT),
                planning_depth=1,
                visual_stagnation_visits=1,
                behavioral_best_first_archive=True,
                human_prior_best_first_archive=True,
                human_prior_graph_stagnation_visits=3,
            ),
            event_logger=logger,
        )
        current = agent.reset()
        scene = agent._scene_signature(current)
        repeated_frame = Frame(8, 8, 1, bytes([10]) + bytes(63))
        unexpanded_frame = Frame(8, 8, 1, bytes([20]) + bytes(63))
        older_unexpanded_frame = Frame(8, 8, 1, bytes([30]) + bytes(63))
        repeated = _ArchivedBranch(
            state=env.save_state(),
            frame=repeated_frame,
            plan=NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0),
            score=100.0,
            scene=scene,
            created=1,
            origin_signature="animation-cluster-a",
            causal_spatial_signature="01",
            causal_context_signature="causal-context-root",
            goal_source_signature="stable-goal-state",
        )
        unexpanded = _ArchivedBranch(
            state=env.save_state(),
            frame=unexpanded_frame,
            plan=NeuralPlan((Action.LEFT,), (4,), 0.0, 0.0),
            score=10.0,
            scene=scene,
            created=10,
            origin_signature="animation-cluster-b",
            causal_spatial_signature="02",
            causal_context_signature="causal-context-root",
            goal_source_signature="stable-goal-state",
            goal_target_world_context="world-transformed",
        )
        older_unexpanded = _ArchivedBranch(
            state=env.save_state(),
            frame=older_unexpanded_frame,
            plan=NeuralPlan((Action.UP,), (4,), 0.0, 0.0),
            score=0.0,
            scene=scene,
            created=1,
            origin_signature="animation-cluster-c",
            causal_spatial_signature="03",
            causal_context_signature="causal-context-root",
            goal_source_signature="stable-goal-state",
        )
        agent.archive = [repeated, older_unexpanded, unexpanded]
        agent._archive_frontier_score = lambda branch: branch.score
        agent.human_prior_graph_edge_visits[
            ("stable-goal-state", Action.RIGHT, 4)
        ] = 2
        agent.human_prior_graph_recovery_pending = True

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, unexpanded_frame.digest)
        self.assertEqual(
            agent.human_prior_graph_edge_visits[
                ("stable-goal-state", Action.LEFT, 4)
            ],
            1,
        )
        filtered = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_best_first_archives_filtered"
        ]
        self.assertEqual(len(filtered), 1)
        committed = [
            event
            for event in logger.events
            if event["event"] == "decision_committed"
        ][-1]
        self.assertTrue(committed["human_prior_best_first_applied"])
        self.assertTrue(committed["human_prior_graph_edge_unexpanded"])
        self.assertEqual(
            agent.current_human_prior_world_context_signature,
            "world-transformed",
        )
        self.assertEqual(
            committed["restore_reason"],
            "human_prior_graph_stagnation",
        )

    def test_human_prior_restore_prefers_unseen_world_state_frontier(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.LEFT),
                planning_depth=1,
                behavioral_best_first_archive=True,
                human_prior_best_first_archive=True,
            ),
            event_logger=logger,
        )
        current = agent.reset()
        scene = agent._scene_signature(current)
        mundane_frame = Frame(8, 8, 1, bytes([10]) + bytes(63))
        effect_frame = Frame(8, 8, 1, bytes([20]) + bytes(63))
        mundane = _ArchivedBranch(
            state=env.save_state(),
            frame=mundane_frame,
            plan=NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0),
            score=100.0,
            scene=scene,
            created=1,
            goal_player_slot=(0, 0),
            goal_source_signature="stable-goal-state",
            goal_target_signature="seen-target",
        )
        effect = _ArchivedBranch(
            state=env.save_state(),
            frame=effect_frame,
            plan=NeuralPlan((Action.LEFT,), (4,), 0.0, 0.0),
            score=1.0,
            scene=scene,
            created=2,
            goal_player_slot=(0, 0),
            goal_source_signature="stable-goal-state",
            goal_target_signature="unseen-world-target",
            goal_world_effect_signature="01",
        )
        agent.archive = [mundane, effect]
        agent._archive_frontier_score = lambda branch: branch.score
        agent.human_prior_player_position_visits[(0, 0)] = 3
        agent.human_prior_graph_recovery_pending = True

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.frame.digest, effect_frame.digest)
        filtered = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_best_first_archives_filtered"
        ][-1]
        self.assertTrue(filtered["world_state_frontier_preferred"])
        self.assertEqual(filtered["unvisited_world_states"], 1)

    def test_human_prior_recovery_releases_fully_expanded_target(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.LEFT),
                planning_depth=1,
                behavioral_best_first_archive=True,
                human_prior_best_first_archive=True,
            ),
        )
        current = agent.reset()
        branch_state = env.save_state()
        target = Frame(8, 8, 1, bytes([30]) + bytes(63))
        agent.archive = [
            _ArchivedBranch(
                state=branch_state,
                frame=target,
                plan=NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0),
                score=1.0,
                scene=agent._scene_signature(current),
                created=1,
                goal_player_slot=(0, 0),
                goal_source_signature="source-state",
                goal_target_signature="fully-expanded-target",
            )
        ]
        agent.human_prior_player_position_visits[(0, 0)] = 1
        for action in (Action.RIGHT, Action.LEFT):
            agent._record_human_prior_graph_edge_verification(
                "fully-expanded-target", action, 4
            )
        agent.human_prior_graph_recovery_pending = True

        restored = agent._restore_if_stagnant()

        self.assertIsNone(restored)
        self.assertEqual(agent.archive, [])
        self.assertNotIn(branch_state, env.active_states)

    def test_new_semantic_target_overrides_coarse_frontier_deduplication(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.UP,),
                planning_depth=1,
                human_prior_best_first_archive=True,
            ),
        )
        agent.reset()

        self.assertFalse(
            agent._human_prior_semantic_frontier_novel(
                "same-state", "same-state", Action.UP, 16
            )
        )

        self.assertTrue(
            agent._human_prior_semantic_frontier_novel(
                "source-tile", "new-target-tile", Action.UP, 16
            )
        )
        agent.human_prior_graph_edge_visits[
            ("source-tile", Action.UP, 16)
        ] = 1
        agent.human_prior_graph_state_visits["new-target-tile"] = 1
        self.assertFalse(
            agent._human_prior_semantic_frontier_novel(
                "source-tile", "new-target-tile", Action.UP, 16
            )
        )

    def test_human_prior_option_search_verifies_and_restores_sequence(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=3,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                visual_stagnation_visits=99,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits.clear()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits.clear()
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(env.position, 0)
        self.assertEqual(len(agent.archive), 1)
        branch = agent.archive[0]
        self.assertTrue(branch.human_prior_verified_option)
        self.assertEqual(branch.plan.path, (Action.RIGHT,) * 3)
        self.assertEqual(branch.plan.durations, (1, 1, 1))
        self.assertEqual(branch.goal_player_slot, (3, 0))
        self.assertIn(branch.state, env.active_states)
        verified = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_branch_verified"
        ]
        self.assertEqual(len(verified), 3)
        self.assertTrue(all(event["agent_visible"] for event in verified))
        neutral = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_neutral_verified"
        ]
        self.assertEqual(len(neutral), 3)
        self.assertEqual([event["depth"] for event in neutral], [1, 2, 3])
        self.assertTrue(
            all(
                event["human_prior_option_world_effect_signature"] is None
                for event in verified
            )
        )

        agent.human_prior_graph_recovery_pending = True
        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertTrue(restored.restored_archive)
        self.assertEqual(restored.planned_path, (Action.RIGHT,) * 3)
        self.assertEqual(env.position, 3)
        option_key = agent._human_prior_option_key(
            source_signature,
            (Action.RIGHT,) * 3,
            (1, 1, 1),
        )
        self.assertEqual(agent.human_prior_option_visits[option_key], 1)
        committed = [
            event
            for event in logger.events
            if event["event"] == "decision_committed"
        ][-1]
        self.assertTrue(committed["human_prior_verified_option"])
        self.assertEqual(committed["human_prior_option_depth"], 3)

    def test_option_search_does_not_reward_autonomous_player_motion(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            AutonomousPositionEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=3,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                human_prior_episodic_graph_guidance=True,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 0)
        self.assertEqual(agent.archive, [])
        self.assertEqual(agent.human_prior_episodic_graph_edges, {})
        verified = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_branch_verified"
        ]
        self.assertEqual(len(verified), 3)
        self.assertTrue(
            all(
                not event[
                    "human_prior_option_action_dependent_endpoint"
                ]
                for event in verified
            )
        )
        self.assertTrue(
            all(
                not event[
                    "human_prior_option_local_action_dependent"
                ]
                for event in verified
            )
        )
        self.assertTrue(
            all(
                event["human_prior_option_player_matches_neutral"]
                for event in verified
            )
        )
        self.assertTrue(
            all(
                event["human_prior_option_causal_goal_reward"] == 0.0
                for event in verified
            )
        )

    def test_option_search_reserves_spatially_divergent_position(
        self,
    ) -> None:
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            DivergentPositionEnv(),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.LEFT, Action.DOWN),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_position_reserve=1,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(3, 0)] = 1

        agent._search_human_prior_options()

        first_depth = next(
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_search_depth_completed"
            and event["depth"] == 1
        )
        self.assertEqual(
            first_depth["human_prior_option_position_reserve"], 1
        )
        self.assertEqual(
            first_depth[
                "human_prior_option_position_parents_retained"
            ],
            1,
        )
        self.assertEqual(
            first_depth["human_prior_option_position_reserve_slots"],
            ((0, 0),),
        )

    def test_option_archive_preserves_spatially_distinct_endpoint(
        self,
    ) -> None:
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            DivergentPositionEnv(),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.LEFT, Action.DOWN),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=2,
                human_prior_option_search_position_reserve=1,
                human_prior_option_search_action_frames=1,
                human_prior_option_archive_representatives=2,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(3, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 2)
        archived_positions = {
            branch.goal_player_slot for branch in agent.archive
        }
        self.assertIn((5, 0), archived_positions)
        self.assertTrue(
            any(
                position is not None and position[0] < 3
                for position in archived_positions
            )
        )
        archived_events = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_archive_added"
        ]
        self.assertEqual(
            sum(
                event[
                    "human_prior_option_archive_position_representative"
                ]
                for event in archived_events
            ),
            1,
        )
        position_archive = next(
            event
            for event in archived_events
            if event[
                "human_prior_option_archive_position_representative"
            ]
        )
        self.assertGreater(
            position_archive[
                "human_prior_option_archive_position_divergence"
            ],
            0,
        )
        completed = next(
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_completed"
        )
        self.assertEqual(completed["position_representatives_archived"], 1)
        self.assertGreater(
            completed["position_representative_max_divergence"], 0
        )

    def test_human_prior_option_search_retains_distinct_semantic_states(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=3,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                human_prior_option_archive_representatives=3,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 2)
        self.assertEqual(
            {branch.goal_player_slot for branch in agent.archive},
            {(2, 0), (3, 0)},
        )
        completed = next(
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_completed"
        )
        self.assertEqual(
            completed["semantic_state_representatives_available"], 2
        )
        self.assertEqual(
            completed["semantic_state_representatives_archived"], 2
        )

    def test_option_search_does_not_expand_a_known_milestone(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = MovingMilestoneSettlesEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.A),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=3,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        source = agent.reset()
        agent.goal_prior = MovingMilestoneGoalPrior()
        agent.human_prior_exhausted_milestone_transitions.add(
            (((7, 0),), (), False)
        )

        added = agent._search_human_prior_options()

        depths = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_depth_completed"
        ]
        milestone_depth = next(
            event for event in depths if event["repeated_milestone_candidates"]
        )
        self.assertEqual(
            milestone_depth["repeated_milestone_parents_retained"], 0
        )
        self.assertEqual(added, 1)
        self.assertTrue(agent.archive)
        self.assertTrue(
            all(branch.goal_heart_slots == ((7, 0),) for branch in agent.archive)
        )
        ordering_rejections = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_ordering_endpoint_rejected"
        ]
        self.assertEqual(len(ordering_rejections), 1)
        self.assertEqual(
            ordering_rejections[0]["path"], (Action.RIGHT, Action.A)
        )
        self.assertTrue(ordering_rejections[0]["exhausted_transition"])
        self.assertFalse(ordering_rejections[0]["exhausted_precursor"])

    def test_exhausted_milestone_transition_uses_semantic_alternative(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(),
        )
        source = env.reset()
        target = env.step(Action.RIGHT, 1)
        base = PositionGoalPrior().analyze(source, target)
        source_hearts = ((1, 0), (7, 0))
        milestone = replace(
            base,
            known_slots=source_hearts,
            source_present=source_hearts,
            target_present=((7, 0),),
            collected=((1, 0),),
            heart_reward=25.0,
            navigation_reward=0.0,
            total_reward=25.0,
        )
        alternative = replace(
            base,
            known_slots=source_hearts,
            source_present=source_hearts,
            target_present=source_hearts,
            collected=(),
            heart_reward=0.0,
            target_player_slot=(2, 0),
            total_reward=base.navigation_reward,
        )
        precursor = replace(
            alternative,
            target_player_slot=(1, 0),
        )
        milestone_state = object()
        precursor_state = object()
        alternative_state = object()
        milestone_branch = (
            25.0,
            NeuralPlan((Action.DOWN,), (1,), 0.0, 0.0),
            milestone_state,
            target,
        )
        precursor_branch = (
            2.0,
            NeuralPlan((Action.NOOP,), (1,), 0.0, 0.0),
            precursor_state,
            target,
        )
        alternative_branch = (
            1.0,
            NeuralPlan((Action.UP,), (1,), 0.0, 0.0),
            alternative_state,
            target,
        )
        verified = [
            milestone_branch,
            precursor_branch,
            alternative_branch,
        ]
        analyses = {
            id(milestone_state): milestone,
            id(precursor_state): precursor,
            id(alternative_state): alternative,
        }
        signatures = {
            id(milestone_state): ("source", "milestone"),
            id(precursor_state): ("source", "precursor"),
            id(alternative_state): ("source", "preparation"),
        }
        agent.human_prior_exhausted_milestone_transitions.add(
            (source_hearts, ((7, 0),), False)
        )

        filtered, exhausted, precursors, alternatives, fail_open = (
            agent._filter_exhausted_milestone_transitions(
                verified, analyses, signatures
            )
        )

        self.assertEqual(filtered, [alternative_branch])
        self.assertEqual(exhausted, [milestone_branch])
        self.assertEqual(precursors, [precursor_branch])
        self.assertEqual(alternatives, 1)
        self.assertFalse(fail_open)

        stationary = replace(
            alternative,
            target_player_slot=alternative.source_player_slot,
            navigation_reward=0.0,
            total_reward=0.0,
        )
        analyses[id(alternative_state)] = stationary
        signatures[id(alternative_state)] = ("source", "source")

        unfiltered, exhausted, precursors, alternatives, fail_open = (
            agent._filter_exhausted_milestone_transitions(
                verified, analyses, signatures
            )
        )

        self.assertEqual(unfiltered, verified)
        self.assertEqual(exhausted, [milestone_branch])
        self.assertEqual(precursors, [precursor_branch])
        self.assertEqual(alternatives, 0)
        self.assertTrue(fail_open)

    def test_option_search_filters_fully_mapped_control_leaves(
        self,
    ) -> None:
        env = ActionEffectEnv()
        agent = VerifiedNeuralAgent(
            env,
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(actions=(Action.LEFT, Action.RIGHT)),
        )
        source = agent.reset()
        target = env.step(Action.RIGHT, 1)
        analysis = PositionGoalPrior().analyze(source, target)
        agent._record_human_prior_episodic_graph_transition(
            "neighbor", "closed-leaf", 1
        )
        agent._record_human_prior_episodic_graph_transition(
            "closed-leaf", "neighbor", 1
        )
        agent._record_human_prior_episodic_graph_transition(
            "remote-option-source", "closed-leaf", 8
        )
        for action in agent.config.actions:
            agent._record_human_prior_graph_edge_verification(
                "closed-leaf", action, 1
            )

        closed_leaf = SimpleNamespace(
            source_signature="source",
            target_signature="closed-leaf",
            target_state_visits=1,
            analysis=analysis,
            confirmed_world_effect_signature="",
            confirmed_entity_state_signature="",
            episodic_graph_bridge_reached=False,
            episodic_graph_progress=0.0,
        )
        open_frontier = SimpleNamespace(
            source_signature="source",
            target_signature="open-frontier",
            target_state_visits=0,
            analysis=analysis,
            confirmed_world_effect_signature="",
            confirmed_entity_state_signature="",
            episodic_graph_bridge_reached=False,
            episodic_graph_progress=0.0,
        )

        filtered, leaves = (
            agent._filter_closed_control_leaf_option_endpoints(
                (closed_leaf, open_frontier)
            )
        )

        self.assertEqual(filtered, [open_frontier])
        self.assertEqual(leaves, [closed_leaf])
        self.assertTrue(
            agent._human_prior_closed_control_leaf("closed-leaf")
        )
        self.assertFalse(
            agent._human_prior_closed_control_leaf("open-frontier")
        )

        filtered, leaves = (
            agent._filter_closed_control_leaf_option_endpoints(
                (closed_leaf,)
            )
        )
        self.assertEqual(filtered, [])
        self.assertEqual(leaves, [closed_leaf])

        milestone_leaf = SimpleNamespace(
            **{
                **closed_leaf.__dict__,
                "analysis": replace(
                    analysis,
                    heart_reward=1.0,
                    total_reward=analysis.total_reward + 1.0,
                ),
            }
        )
        filtered, leaves = (
            agent._filter_closed_control_leaf_option_endpoints(
                (milestone_leaf,)
            )
        )
        self.assertEqual(filtered, [milestone_leaf])
        self.assertEqual(leaves, [])

    def test_human_prior_option_search_rejects_globally_visited_states(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            UniqueStateEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=3,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                human_prior_option_archive_representatives=3,
            ),
            event_logger=logger,
        )
        source = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1
        for position in range(1, 4):
            pixels = bytearray(64)
            pixels[position] = 255
            target = Frame(8, 8, 1, bytes(pixels))
            analysis = agent.goal_prior.analyze(source, target)
            target_signature = agent._human_prior_graph_signatures(
                analysis
            )[1]
            agent.human_prior_graph_state_visits[target_signature] = 1
            agent.human_prior_player_position_visits[(position, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 0)
        self.assertEqual(agent.archive, [])
        completed = next(
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_completed"
        )
        self.assertEqual(completed["reason"], "no_globally_novel_endpoint")
        self.assertEqual(completed["globally_novel_endpoints"], 0)

    def test_episodic_graph_plan_targets_closest_missing_bridge(
        self,
    ) -> None:
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                human_prior_heart_reward=1.0,
                human_prior_episodic_graph_guidance=True,
            ),
        )
        agent.reset()
        signature = lambda player: agent._human_prior_graph_signature(
            ((7, 0),), player, None, "life"
        )
        source = signature((0, 0))
        route = signature((1, 0))
        frontier = signature((2, 0))
        bridge = signature((3, 0))
        milestone = signature((4, 0))
        agent._record_human_prior_episodic_graph_transition(
            source, route, 1
        )
        agent._record_human_prior_episodic_graph_transition(
            route, frontier, 1
        )
        agent._record_human_prior_episodic_graph_transition(
            bridge, milestone, 1
        )
        agent._record_human_prior_episodic_graph_transition(
            milestone, "", 1, milestone=True
        )

        plan = agent._human_prior_episodic_graph_plan(source)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertFalse(plan.known_route)
        self.assertEqual(plan.waypoint_signature, frontier)
        self.assertEqual(plan.bridge_target_signature, bridge)
        self.assertEqual(plan.gap_distance, 1)
        self.assertEqual(plan.source_remaining_cost, 2)
        progress, reached, remaining = (
            agent._human_prior_episodic_graph_progress(plan, source)
        )
        self.assertEqual(progress, 0.0)
        self.assertFalse(reached)
        self.assertEqual(remaining, 2)
        progress, reached, remaining = (
            agent._human_prior_episodic_graph_progress(plan, route)
        )
        self.assertEqual(progress, 0.5)
        self.assertFalse(reached)
        self.assertEqual(remaining, 1)
        progress, reached, remaining = (
            agent._human_prior_episodic_graph_progress(plan, bridge)
        )
        self.assertEqual(progress, 2.0)
        self.assertTrue(reached)
        self.assertIsNone(remaining)

    def test_episodic_graph_plan_excludes_exhausted_milestone_outcome(
        self,
    ) -> None:
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                human_prior_heart_reward=1.0,
                human_prior_episodic_graph_guidance=True,
            ),
        )
        agent.reset()
        hearts = ((16, 64), (144, 192))
        source = agent._human_prior_graph_signature(
            hearts, (0, 0), None, "life"
        )
        failed_source = agent._human_prior_graph_signature(
            hearts, (16, 0), None, "life"
        )
        viable_source = agent._human_prior_graph_signature(
            hearts, (32, 0), None, "life"
        )
        failed_target_hearts = ((144, 192),)
        viable_target_hearts = ((16, 64),)
        failed_target = agent._human_prior_graph_signature(
            failed_target_hearts, (16, 0), None, "life"
        )
        viable_target = agent._human_prior_graph_signature(
            viable_target_hearts, (32, 0), None, "life"
        )
        agent._record_human_prior_episodic_graph_transition(
            source, failed_source, 1
        )
        agent._record_human_prior_episodic_graph_transition(
            source, viable_source, 1
        )
        agent._record_human_prior_episodic_graph_transition(
            failed_source, failed_target, 1, milestone=True
        )
        agent._record_human_prior_episodic_graph_transition(
            viable_source, viable_target, 1, milestone=True
        )
        exhausted_transition = (
            hearts,
            failed_target_hearts,
            False,
        )
        agent.human_prior_exhausted_milestone_transitions.add(
            exhausted_transition
        )
        unqualified_source = agent._human_prior_graph_signature(
            hearts, (48, 0), None, "life"
        )
        agent.human_prior_episodic_milestone_sources.add(
            unqualified_source
        )
        agent._record_human_prior_episodic_graph_transition(
            unqualified_source,
            viable_target,
            1,
            milestone=False,
        )

        plan = agent._human_prior_episodic_graph_plan(source)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.kind, "milestone_route")
        self.assertEqual(
            plan.milestone_source_signature, viable_source
        )
        self.assertTrue(
            agent._human_prior_episodic_milestone_source_exhausted(
                failed_source
            )
        )
        self.assertFalse(
            agent._human_prior_episodic_milestone_source_exhausted(
                viable_source
            )
        )
        self.assertTrue(
            agent._human_prior_episodic_milestone_source_exhausted(
                unqualified_source
            )
        )
        self.assertEqual(
            agent._human_prior_episodic_milestone_transition_keys(
                unqualified_source
            ),
            (),
        )

        agent.human_prior_disproved_ordering_hypotheses.add(
            (hearts, ((16, 64),), False)
        )
        reconsidered = agent._human_prior_episodic_graph_plan(source)
        self.assertIsNotNone(reconsidered)
        assert reconsidered is not None
        self.assertEqual(
            reconsidered.milestone_source_signature, failed_source
        )

    def test_episodic_graph_plan_targets_unexpanded_control_frontier(
        self,
    ) -> None:
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.DOWN),
                human_prior_heart_reward=1.0,
                human_prior_episodic_graph_guidance=True,
            ),
        )
        agent.reset()
        signature = lambda player: agent._human_prior_graph_signature(
            ((7, 0),), player, None, "life"
        )
        source = signature((0, 0))
        route = signature((1, 0))
        frontier = signature((2, 0))
        agent._record_human_prior_episodic_graph_transition(
            source, route, 1
        )
        agent._record_human_prior_episodic_graph_transition(
            route, frontier, 1
        )
        agent._record_human_prior_graph_edge_verification(
            frontier, Action.RIGHT, 1
        )

        plan = agent._human_prior_episodic_graph_plan(source)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.kind, "control_frontier")
        self.assertTrue(plan.known_route)
        self.assertEqual(plan.waypoint_signature, frontier)
        self.assertEqual(plan.frontier_actions, (Action.DOWN,))
        self.assertEqual(plan.source_remaining_cost, 2)
        progress, reached, remaining = (
            agent._human_prior_episodic_graph_progress(plan, route)
        )
        self.assertEqual(progress, 0.5)
        self.assertFalse(reached)
        self.assertEqual(remaining, 1)
        progress, reached, remaining = (
            agent._human_prior_episodic_graph_progress(plan, frontier)
        )
        self.assertEqual(progress, 1.0)
        self.assertFalse(reached)
        self.assertEqual(remaining, 0)

    def test_episodic_graph_plan_avoids_zero_length_milestone_route(
        self,
    ) -> None:
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.UP, Action.DOWN),
                human_prior_heart_reward=1.0,
                human_prior_episodic_graph_guidance=True,
            ),
        )
        agent.reset()
        signature = lambda player: agent._human_prior_graph_signature(
            ((7, 0),), player, None, "life"
        )
        source = signature((0, 0))
        frontier = signature((0, 1))
        agent.human_prior_episodic_milestone_sources.add(source)
        agent._record_human_prior_episodic_graph_transition(
            source, frontier, 1
        )
        agent._record_human_prior_graph_edge_verification(
            frontier, Action.UP, 1
        )

        plan = agent._human_prior_episodic_graph_plan(source)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.kind, "control_frontier")
        self.assertEqual(plan.waypoint_signature, frontier)
        self.assertEqual(plan.frontier_actions, (Action.DOWN,))
        self.assertEqual(plan.source_remaining_cost, 1)

    def test_option_search_reuses_visited_control_frontier_route(
        self,
    ) -> None:
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                human_prior_episodic_graph_guidance=True,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        signatures = [
            agent._human_prior_graph_signature(
                ((7, 0),), (position, 0), None, "life"
            )
            for position in range(3)
        ]
        for source, target in zip(signatures, signatures[1:]):
            agent._record_human_prior_episodic_graph_transition(
                source, target, 1
            )
        for position, signature in enumerate(signatures):
            agent.human_prior_graph_state_visits[signature] = 1
            agent.human_prior_player_position_visits[(position, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(len(agent.archive), 1)
        self.assertEqual(agent.archive[0].goal_player_slot, (2, 0))
        self.assertEqual(
            agent.archive[0].human_prior_episodic_graph_plan_kind,
            "control_frontier",
        )
        self.assertGreater(
            agent.archive[0].human_prior_episodic_graph_progress, 0.0
        )
        selected = next(
            event
            for event in logger.events
            if event["event"]
            == "human_prior_episodic_graph_plan_selected"
        )
        self.assertEqual(selected["plan_kind"], "control_frontier")
        self.assertEqual(selected["frontier_actions"], ("right",))
        completed = next(
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_completed"
        )
        self.assertEqual(
            completed["human_prior_episodic_graph_plan_kind"],
            "control_frontier",
        )
        self.assertGreater(
            completed["human_prior_episodic_graph_progress"], 0.0
        )

    def test_option_search_reuses_visited_episodic_graph_progress(
        self,
    ) -> None:
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                human_prior_episodic_graph_guidance=True,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        signatures = [
            agent._human_prior_graph_signature(
                ((7, 0),), (position, 0), None, "life"
            )
            for position in range(4)
        ]
        for source, target in zip(signatures, signatures[1:]):
            agent._record_human_prior_episodic_graph_transition(
                source, target, 1
            )
        agent._record_human_prior_episodic_graph_transition(
            signatures[-1], "", 1, milestone=True
        )
        for position, signature in enumerate(signatures[:3]):
            agent.human_prior_graph_state_visits[signature] = 1
            agent.human_prior_player_position_visits[(position, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(len(agent.archive), 1)
        self.assertEqual(agent.archive[0].goal_player_slot, (2, 0))
        self.assertGreater(
            agent.archive[0].human_prior_episodic_graph_progress, 0.0
        )
        selected = next(
            event
            for event in logger.events
            if event["event"]
            == "human_prior_episodic_graph_plan_selected"
        )
        self.assertTrue(selected["known_route"])
        completed = next(
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_completed"
        )
        self.assertGreater(
            completed["human_prior_episodic_graph_progress"], 0.0
        )

    def test_exhausted_option_frontier_filter_avoids_reentry(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                planning_depth=1,
                human_prior_option_search_depth=2,
            ),
        )
        source = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        target = env.step(Action.RIGHT, 1)
        dead_analysis = agent.goal_prior.analyze(source, target)
        stationary_analysis = agent.goal_prior.analyze(source, source)
        dead_state = object()
        alternative_state = object()
        dead_branch = (
            2.0,
            NeuralPlan((Action.RIGHT,), (1,), 2.0, 0.0),
            dead_state,
            target,
        )
        alternative_branch = (
            1.0,
            NeuralPlan((Action.NOOP,), (1,), 1.0, 0.0),
            alternative_state,
            source,
        )
        verified = [dead_branch, alternative_branch]
        analyses = {
            id(dead_state): dead_analysis,
            id(alternative_state): stationary_analysis,
        }
        signatures = {
            id(dead_state): ("source", "exhausted"),
            id(alternative_state): ("source", "escape"),
        }
        agent._record_human_prior_exhausted_option_frontier("exhausted")

        filtered, blocked, fail_open = (
            agent._filter_exhausted_option_frontiers(
                verified, analyses, signatures
            )
        )

        self.assertEqual(filtered, [alternative_branch])
        self.assertEqual(blocked, [dead_branch])
        self.assertFalse(fail_open)
        only_blocked, blocked, fail_open = (
            agent._filter_exhausted_option_frontiers(
                [dead_branch], analyses, signatures
            )
        )
        self.assertEqual(only_blocked, [dead_branch])
        self.assertEqual(blocked, [dead_branch])
        self.assertTrue(fail_open)
        stationary_signatures = dict(signatures)
        stationary_signatures[id(alternative_state)] = (
            "source",
            "source",
        )
        stationary_fallback, blocked, fail_open = (
            agent._filter_exhausted_option_frontiers(
                verified, analyses, stationary_signatures
            )
        )
        self.assertEqual(stationary_fallback, verified)
        self.assertEqual(blocked, [dead_branch])
        self.assertTrue(fail_open)
        agent.human_prior_exhausted_option_frontiers["exhausted"] = 1
        reopened, blocked, fail_open = (
            agent._filter_exhausted_option_frontiers(
                verified, analyses, signatures
            )
        )
        self.assertEqual(reopened, verified)
        self.assertEqual(blocked, [])
        self.assertFalse(fail_open)

    def test_option_exhaustion_egress_filter_prefers_graph_transition(
        self,
    ) -> None:
        stationary_state = object()
        egress_state = object()
        stationary = (
            2.0,
            NeuralPlan((Action.LEFT,), (1,), 2.0, 0.0),
            stationary_state,
            Frame(1, 1, 1, b"\x00"),
        )
        egress = (
            1.0,
            NeuralPlan((Action.DOWN,), (1,), 1.0, 0.0),
            egress_state,
            Frame(1, 1, 1, b"\x01"),
        )
        signatures = {
            id(stationary_state): ("source", "source"),
            id(egress_state): ("source", "escape"),
        }

        filtered, non_egress, fail_open = (
            VerifiedNeuralAgent._filter_option_exhaustion_egress(
                [stationary, egress], signatures
            )
        )

        self.assertEqual(filtered, [egress])
        self.assertEqual(non_egress, [stationary])
        self.assertFalse(fail_open)
        only_stationary, non_egress, fail_open = (
            VerifiedNeuralAgent._filter_option_exhaustion_egress(
                [stationary], signatures
            )
        )
        self.assertEqual(only_stationary, [stationary])
        self.assertEqual(non_egress, [stationary])
        self.assertTrue(fail_open)

    def test_option_search_does_not_archive_exhausted_frontier(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=2,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        source = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        root = env.save_state()
        target = env.step(Action.RIGHT, 2)
        analysis = agent.goal_prior.analyze(source, target)
        exhausted_signature = agent._human_prior_graph_signatures(
            analysis
        )[1]
        env.load_state(root)
        agent._record_human_prior_exhausted_option_frontier(
            exhausted_signature
        )

        agent._search_human_prior_options()

        self.assertEqual(agent.archive, [])
        filtered = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_exhausted_frontiers_filtered"
        ]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["endpoints_filtered"], 1)
        completed = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_completed"
        ][-1]
        self.assertEqual(
            completed["reason"], "only_exhausted_frontier_endpoints"
        )
        source_signature = agent._current_human_prior_graph_signature()
        self.assertIn(
            source_signature,
            agent.human_prior_exhausted_option_frontiers,
        )
        self.assertEqual(
            agent.human_prior_exhausted_option_frontiers[source_signature],
            2,
        )

    def test_option_search_accepts_new_graph_state_at_seen_position(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits.update(
            {(0, 0): 1, (1, 0): 4, (2, 0): 9}
        )

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(len(agent.archive), 1)
        self.assertEqual(agent.archive[0].goal_player_slot, (2, 0))
        self.assertEqual(agent.archive[0].goal_remaining_hearts, 1)
        depth_two = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_branch_verified"
            and event["depth"] == 2
            and event["human_prior_target_player_slot"] == (2, 0)
        ]
        self.assertEqual(len(depth_two), 1)
        self.assertEqual(depth_two[0]["target_graph_state_visits"], 0)
        self.assertEqual(depth_two[0]["target_player_position_visits"], 9)
        self.assertTrue(depth_two[0]["endpoint_eligible"])

    def test_human_prior_option_archive_uses_whole_path_coverage(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        agent.reset()
        branch = _ArchivedBranch(
            state=0,
            frame=Frame(8, 8, 1, bytes([255]) + bytes(63)),
            plan=NeuralPlan(
                (Action.RIGHT, Action.RIGHT), (1, 1), 0.0, 0.0
            ),
            score=0.0,
            scene="scene",
            created=0,
            goal_source_signature="source",
            human_prior_verified_option=True,
        )
        agent.human_prior_graph_edge_visits[
            ("source", Action.RIGHT, 1)
        ] = 10

        self.assertEqual(
            agent._human_prior_archive_edge_coverage(branch), (0, True)
        )
        agent._record_human_prior_archive_edge(branch)
        self.assertEqual(
            agent._human_prior_archive_edge_coverage(branch), (1, False)
        )

    def test_human_prior_option_search_caches_exhausted_source(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        first = agent._search_human_prior_options()
        verified_after_first = len(
            [
                event
                for event in logger.events
                if event["event"]
                == "human_prior_option_branch_verified"
            ]
        )
        second = agent._search_human_prior_options()

        self.assertEqual(first, 0)
        self.assertEqual(second, 0)
        self.assertEqual(verified_after_first, 2)
        self.assertEqual(
            len(
                [
                    event
                    for event in logger.events
                    if event["event"]
                    == "human_prior_option_branch_verified"
                ]
            ),
            verified_after_first,
        )
        skipped = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_skipped"
        ]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "source_already_exhausted")
        self.assertTrue(skipped[0]["exact_search_budget_match"])
        depth_events = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_search_depth_completed"
        ]
        self.assertEqual(
            [event["retained_parents"] for event in depth_events],
            [1, 0],
        )
        self.assertEqual(depth_events[-1]["novel_candidates"], 0)

        agent.config = replace(
            agent.config,
            human_prior_option_search_depth=3,
        )
        third = agent._search_human_prior_options()

        self.assertEqual(third, 0)
        reopened = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_reopened"
        ]
        self.assertEqual(len(reopened), 1)
        self.assertEqual(reopened[0]["reason"], "search_budget_changed")
        self.assertEqual(reopened[0]["maximum_depth"], 3)
        self.assertFalse(reopened[0]["exact_search_budget_match"])
        completed = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_completed"
        ]
        self.assertEqual(len(completed), 2)
        self.assertNotEqual(
            completed[0]["search_budget_sha256"],
            completed[1]["search_budget_sha256"],
        )

    def test_option_search_does_not_reward_missing_player_as_novel(
        self,
    ) -> None:
        class MissingPlayerChoiceEnv:
            def __init__(self) -> None:
                self.mode = "source"

            def _frame(self) -> Frame:
                pixels = bytearray(64)
                pixels[7] = 128
                if self.mode == "source":
                    pixels[0] = 255
                elif self.mode == "detected":
                    pixels[1] = 255
                return Frame(8, 8, 1, bytes(pixels))

            def reset(self) -> Frame:
                self.mode = "source"
                return self._frame()

            def step(self, action: Action, frames: int = 1) -> Frame:
                del frames
                if action == Action.RIGHT:
                    self.mode = "detected"
                elif action == Action.LEFT:
                    self.mode = "missing"
                return self._frame()

            def save_state(self) -> str:
                return self.mode

            def load_state(self, state: str) -> Frame:
                self.mode = state
                return self._frame()

        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            MissingPlayerChoiceEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.LEFT),
                planning_depth=1,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = MovingMilestoneGoalPrior()

        agent._search_human_prior_options()

        depth_two = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_branch_verified"
            and event["depth"] == 2
        ]
        self.assertTrue(depth_two)
        self.assertTrue(
            all(event["path"][0] == Action.RIGHT for event in depth_two)
        )

        streak_logger = RecordingLogger()
        streak_agent = VerifiedNeuralAgent(
            MissingPlayerChoiceEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT,),
                planning_depth=1,
                human_prior_option_search_depth=3,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_missing_player_reserve=1,
                human_prior_option_search_missing_player_max_streak=1,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=streak_logger,
        )
        streak_agent.reset()
        streak_agent.goal_prior = MovingMilestoneGoalPrior()

        streak_agent._search_human_prior_options()

        streak_depths = [
            event
            for event in streak_logger.events
            if event["event"]
            == "human_prior_option_search_depth_completed"
        ]
        self.assertEqual(
            [event["retained_parents"] for event in streak_depths],
            [1, 0],
        )

    def test_option_search_archives_goal_progress_at_a_known_position(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=2,
                human_prior_option_search_action_frames=1,
            ),
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1
        agent.human_prior_player_position_visits[(2, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(agent.archive[0].plan.path, (Action.RIGHT, Action.RIGHT))
        self.assertEqual(agent.archive[0].goal_player_slot, (2, 0))
        self.assertEqual(agent.archive[0].goal_progress_reward, 2.0)

    def test_option_search_retains_ordering_progress_at_known_graph_state(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_navigation_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=2,
                human_prior_option_search_action_frames=1,
            ),
        )
        agent.reset()
        agent.goal_prior = OrderingPositionGoalPrior()
        hearts = ((-16, 0), (32, 0))
        agent.human_prior_exhausted_milestone_transitions.add(
            (hearts, ((32, 0),), False)
        )
        ordering_key = agent._human_prior_ordering_hypothesis_key(
            hearts, False
        )
        assert ordering_key is not None
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1
        agent.human_prior_player_position_visits[(2, 0)] = 1
        root = env.save_state()
        target = env.step(Action.RIGHT, 2)
        analysis = agent.goal_prior.analyze(agent.frame, target)
        target_signature = agent._human_prior_graph_signatures(analysis)[1]
        agent.human_prior_graph_state_visits[target_signature] = 1
        env.load_state(root)

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(agent.archive[0].goal_player_slot, (2, 0))
        self.assertAlmostEqual(
            agent.archive[0].goal_progress_reward, 0.125
        )
        self.assertIn(
            ordering_key,
            agent.human_prior_ordering_progress_hypotheses,
        )

    def test_option_search_retains_reconsidered_progress_at_known_state(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_navigation_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=2,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = OrderingPositionGoalPrior()
        hearts = ((-16, 0), (32, 0))
        agent.human_prior_exhausted_milestone_transitions.add(
            (hearts, ((-16, 0),), False)
        )
        ordering_key = agent._human_prior_ordering_hypothesis_key(
            hearts, False
        )
        assert ordering_key is not None
        agent.human_prior_disproved_ordering_hypotheses.add(ordering_key)
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1
        agent.human_prior_player_position_visits[(2, 0)] = 1
        root = env.save_state()
        target = env.step(Action.RIGHT, 2)
        analysis = agent.goal_prior.analyze(agent.frame, target)
        target_signature = agent._human_prior_graph_signatures(analysis)[1]
        agent.human_prior_graph_state_visits[target_signature] = 1
        env.load_state(root)

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(agent.archive[0].goal_player_slot, (2, 0))
        archived = next(
            event
            for event in logger.events
            if event["event"] == "human_prior_option_archive_added"
        )
        self.assertTrue(
            archived["human_prior_navigation_reconsidered"]
        )
        self.assertEqual(
            archived["human_prior_navigation_reconsidered_targets"],
            ((32, 0),),
        )
        self.assertAlmostEqual(
            archived["human_prior_navigation_reconsidered_reward"],
            0.125,
        )

    def test_option_search_disproves_exhausted_ordering_trial(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_navigation_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=2,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        agent.reset()
        env.position = 63
        agent.frame = env._frame()
        agent.goal_prior = OrderingPositionGoalPrior()
        agent.goal_prior.current_player_slot = (63, 0)
        hearts = ((-16, 0), (32, 0))
        agent.human_prior_exhausted_milestone_transitions.add(
            (hearts, ((32, 0),), False)
        )
        ordering_key = agent._human_prior_ordering_hypothesis_key(
            hearts, False
        )
        assert ordering_key is not None
        agent.human_prior_ordering_progress_hypotheses.add(ordering_key)

        added = agent._search_human_prior_options()

        self.assertEqual(added, 0)
        self.assertIn(
            ordering_key,
            agent.human_prior_disproved_ordering_hypotheses,
        )
        disproved = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_ordering_hypothesis_disproved"
        ]
        self.assertEqual(len(disproved), 1)
        self.assertEqual(
            disproved[0]["policy_effect"],
            "ordering_retarget_disabled",
        )
        self.assertEqual(
            disproved[0]["human_prior_option_search_position_reserve"],
            0,
        )
        self.assertEqual(len(disproved[0]["search_budget_sha256"]), 64)
        completed = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_completed"
        ][-1]
        self.assertTrue(completed["ordering_hypothesis_disproved"])

    def test_decide_immediately_restores_new_verified_option(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        env = ActionEffectEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
                human_prior_graph_stagnation_visits=1,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=2,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        agent._calibrate_goal_prior = lambda _frame: None
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        decision = agent.decide()

        self.assertTrue(decision.restored_archive)
        self.assertEqual(decision.planned_path, (Action.RIGHT, Action.RIGHT))
        self.assertEqual(agent.goal_prior.current_player_slot, (2, 0))
        self.assertEqual(env.position, 2)
        self.assertTrue(
            any(
                event["event"] == "human_prior_option_recovery_armed"
                for event in logger.events
            )
        )

    def test_decide_defers_regressive_verified_option(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        env = ActionEffectEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                verify_actions=1,
                human_prior_best_first_archive=True,
                human_prior_graph_stagnation_visits=1,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=2,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = RegressivePositionGoalPrior()
        agent._calibrate_goal_prior = lambda _frame: None
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        decision = agent.decide()

        self.assertFalse(decision.restored_archive)
        self.assertTrue(
            any(branch.goal_progress_reward < 0.0 for branch in agent.archive)
        )
        self.assertTrue(
            any(
                event["event"] == "human_prior_option_recovery_deferred"
                for event in logger.events
            )
        )

    def test_decide_restores_verified_episodic_graph_progress(self) -> None:
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                verify_actions=1,
                human_prior_best_first_archive=True,
                human_prior_graph_stagnation_visits=1,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                human_prior_episodic_graph_guidance=True,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = RegressivePositionGoalPrior()
        agent._calibrate_goal_prior = lambda _frame: None
        signatures = [
            agent._human_prior_graph_signature(
                ((7, 0),), (position, 0), None, "life"
            )
            for position in range(4)
        ]
        for source, target in zip(signatures, signatures[1:]):
            agent._record_human_prior_episodic_graph_transition(
                source, target, 1
            )
        agent._record_human_prior_episodic_graph_transition(
            signatures[-1], "", 1, milestone=True
        )
        for position, signature in enumerate(signatures[:3]):
            agent.human_prior_graph_state_visits[signature] = 1
            agent.human_prior_player_position_visits[(position, 0)] = 1

        decision = agent.decide()

        self.assertTrue(decision.restored_archive)
        self.assertEqual(decision.planned_path, (Action.RIGHT, Action.RIGHT))
        self.assertEqual(env.position, 2)
        armed = next(
            event
            for event in logger.events
            if event["event"] == "human_prior_option_recovery_armed"
        )
        self.assertEqual(
            armed["reason"],
            "positive_verified_episodic_graph_progress",
        )
        restored = next(
            event
            for event in logger.events
            if event["event"] == "decision_committed"
        )
        self.assertGreater(
            restored["human_prior_episodic_graph_progress"], 0.0
        )

    def test_archive_restore_prefers_milestone_over_graph_progress(
        self,
    ) -> None:
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                behavioral_best_first_archive=True,
                human_prior_best_first_archive=True,
            ),
            event_logger=logger,
        )
        source = agent.reset()
        agent.goal_prior = OrderingPositionGoalPrior()
        root = env.save_state()
        milestone_frame = env.step(Action.RIGHT, 1)
        milestone_state = env.save_state()
        env.load_state(root)
        graph_frame = env.step(Action.RIGHT, 2)
        graph_state = env.save_state()
        env.load_state(root)
        source_hearts = tuple(sorted(agent.goal_prior.current_slots()))
        milestone = _ArchivedBranch(
            state=milestone_state,
            frame=milestone_frame,
            plan=NeuralPlan((Action.RIGHT,), (1,), 25.0, 0.0),
            score=25.0,
            scene=agent._scene_signature(milestone_frame),
            created=0,
            origin_signature="source",
            goal_source_signature="graph-source",
            goal_target_signature="milestone-target",
            goal_heart_slots=(source_hearts[1],),
            goal_progress_reward=25.0,
            goal_remaining_hearts=1,
            goal_total_hearts=2,
            goal_player_slot=(1, 0),
        )
        graph_progress = _ArchivedBranch(
            state=graph_state,
            frame=graph_frame,
            plan=NeuralPlan((Action.RIGHT,), (2,), 40.0, 0.0),
            score=40.0,
            scene=agent._scene_signature(graph_frame),
            created=0,
            origin_signature="source",
            goal_source_signature="graph-source",
            goal_target_signature="graph-target",
            goal_heart_slots=source_hearts,
            goal_progress_reward=3.0,
            goal_remaining_hearts=2,
            goal_total_hearts=2,
            goal_player_slot=(2, 0),
            human_prior_episodic_graph_progress=2.0,
            human_prior_episodic_graph_bridge_reached=True,
        )
        agent.archive = [graph_progress, milestone]
        agent.human_prior_graph_recovery_pending = True

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertTrue(decision.restored_archive)
        self.assertEqual(env.position, 1)
        self.assertEqual(agent.goal_prior.current_slots(), (source_hearts[1],))
        filtered = next(
            event
            for event in logger.events
            if event["event"] == "human_prior_best_first_archives_filtered"
        )
        self.assertTrue(filtered["milestone_goal_frontier_preferred"])
        env.release_state(root)

    def test_archive_restore_prefers_completed_graph_route_over_shaping(
        self,
    ) -> None:
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                behavioral_best_first_archive=True,
                human_prior_best_first_archive=True,
            ),
            event_logger=logger,
        )
        source = agent.reset()
        agent.goal_prior = OrderingPositionGoalPrior()
        root = env.save_state()
        completed_frame = env.step(Action.RIGHT, 1)
        completed_state = env.save_state()
        env.load_state(root)
        shaping_frame = env.step(Action.RIGHT, 2)
        shaping_state = env.save_state()
        env.load_state(root)
        source_hearts = tuple(sorted(agent.goal_prior.current_slots()))
        completed = _ArchivedBranch(
            state=completed_state,
            frame=completed_frame,
            plan=NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
            score=1.0,
            scene=agent._scene_signature(completed_frame),
            created=0,
            origin_signature="source",
            goal_source_signature="graph-source",
            goal_target_signature="completed-target",
            goal_heart_slots=source_hearts,
            goal_progress_reward=1.0,
            goal_remaining_hearts=2,
            goal_total_hearts=2,
            goal_player_slot=(1, 0),
            human_prior_episodic_graph_plan_kind="control_frontier",
            human_prior_episodic_graph_progress=1.0,
            human_prior_episodic_graph_remaining_cost=0,
        )
        shaping = _ArchivedBranch(
            state=shaping_state,
            frame=shaping_frame,
            plan=NeuralPlan((Action.RIGHT,), (2,), 4.0, 0.0),
            score=4.0,
            scene=agent._scene_signature(shaping_frame),
            created=0,
            origin_signature="source",
            goal_source_signature="graph-source",
            goal_target_signature="shaping-target",
            goal_heart_slots=source_hearts,
            goal_progress_reward=4.0,
            goal_remaining_hearts=2,
            goal_total_hearts=2,
            goal_player_slot=(2, 0),
            human_prior_episodic_graph_plan_kind="control_frontier",
            human_prior_episodic_graph_progress=0.4,
            human_prior_episodic_graph_remaining_cost=8,
        )
        agent.archive = [shaping, completed]
        agent.human_prior_graph_recovery_pending = True

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertTrue(decision.restored_archive)
        self.assertEqual(env.position, 1)
        restored = next(
            event
            for event in logger.events
            if event["event"] == "archive_branch_restored"
        )
        self.assertEqual(
            restored["human_prior_episodic_graph_progress"], 1.0
        )
        filtered = next(
            event
            for event in logger.events
            if event["event"]
            == "human_prior_best_first_archives_filtered"
        )
        self.assertTrue(filtered["episodic_graph_frontier_preferred"])
        env.release_state(root)

    def test_archive_restore_invalidates_stale_episodic_graph_progress(
        self,
    ) -> None:
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
                human_prior_episodic_graph_guidance=True,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        root = env.save_state()
        target_frame = env.step(Action.RIGHT, 1)
        target_state = env.save_state()
        env.load_state(root)
        source_signature = agent._current_human_prior_graph_signature()
        target_signature = agent._human_prior_graph_signature(
            ((7, 0),), (1, 0), None, "life"
        )
        agent.human_prior_player_position_visits[(1, 0)] = 1
        agent.human_prior_graph_edge_verifications[
            (target_signature, Action.RIGHT, 1)
        ] = 1
        stale = _ArchivedBranch(
            state=target_state,
            frame=target_frame,
            plan=NeuralPlan((Action.RIGHT,), (1,), 0.7, 0.0),
            score=0.7,
            scene=agent._scene_signature(target_frame),
            created=0,
            origin_signature="source",
            goal_source_signature=source_signature,
            goal_target_signature=target_signature,
            goal_heart_slots=((7, 0),),
            goal_progress_reward=-1.0,
            goal_remaining_hearts=1,
            goal_total_hearts=1,
            goal_player_slot=(1, 0),
            human_prior_verified_option=True,
            human_prior_episodic_graph_plan_kind="milestone_route",
            human_prior_episodic_graph_progress=0.7,
            human_prior_episodic_graph_remaining_cost=3,
        )
        agent.frame = source_frame
        agent.archive = [stale]
        agent.human_prior_graph_recovery_pending = True

        decision = agent._restore_if_stagnant()

        self.assertIsNone(decision)
        self.assertEqual(agent.archive, [])
        revalidated = next(
            event
            for event in logger.events
            if event["event"]
            == "human_prior_archive_episodic_graph_revalidated"
        )
        self.assertEqual(
            revalidated["stale_positive_progress_invalidated"], 1
        )
        self.assertEqual(revalidated["live_positive_progress"], 0)
        self.assertIsNone(revalidated["live_plan_kind"])
        env.release_state(root)

    def test_option_effect_probe_prioritizes_closer_goal_state(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            GoalDirectedEffectPriorityEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.A, Action.RIGHT),
                planning_depth=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=4,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=1,
                human_prior_option_effect_probe_limit=1,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        agent._search_human_prior_options()

        probes = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_world_effect_stability"
        ]
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0]["path"], (Action.RIGHT, Action.RIGHT))
        self.assertEqual(probes[0]["depth"], 2)

    def test_human_prior_option_effect_stability_rejects_transient(self) -> None:
        cases = (
            (False, 63, False),
            (True, 63, True),
            (True, 1, False),
            (True, (27, 28, 29, 30, 31), False),
        )
        for persistent, world_index, expected_stable in cases:
            with self.subTest(
                persistent=persistent, world_index=world_index
            ):
                model = EnsembleVisualDynamicsModel(
                    latent_size=32, action_size=8, ensemble_size=2
                )
                logger = RecordingLogger()
                agent = VerifiedNeuralAgent(
                    WorldEffectEnv(persistent, world_index),
                    model,
                    "cpu",
                    NeuralPlanningConfig(
                        actions=(Action.RIGHT,),
                        planning_depth=1,
                        action_frames=1,
                        human_prior_heart_reward=1.0,
                        human_prior_best_first_archive=True,
                        human_prior_option_search_depth=2,
                        human_prior_option_search_beam_width=1,
                        human_prior_option_search_action_frames=1,
                        human_prior_option_effect_stability_steps=2,
                        human_prior_option_effect_probe_limit=1,
                        causal_spatial_columns=8,
                        causal_spatial_rows=8,
                    ),
                    event_logger=logger,
                )
                agent.reset()
                agent.goal_prior = PositionGoalPrior()
                source_signature = (
                    agent._current_human_prior_graph_signature()
                )
                agent.human_prior_graph_state_visits[
                    source_signature
                ] = 1
                agent.human_prior_player_position_visits[(0, 0)] = 1

                added = agent._search_human_prior_options()

                self.assertEqual(added, 0)
                probes = [
                    event
                    for event in logger.events
                    if event["event"]
                    == "human_prior_option_world_effect_stability"
                ]
                self.assertEqual(len(probes), 1)
                self.assertEqual(probes[0]["stable"], expected_stable)
                self.assertTrue(probes[0]["safe"])
                controls = [
                    event
                    for event in logger.events
                    if event["event"]
                    == "human_prior_option_world_effect_action_control"
                ]
                self.assertEqual(len(controls), int(expected_stable))
                if expected_stable:
                    self.assertEqual(probes[0]["persistence_ratio"], 1.0)
                    self.assertEqual(
                        probes[0]["stable_world_effect_cells"], 1
                    )
                    self.assertTrue(controls[0]["confirmed"])
                else:
                    if isinstance(world_index, tuple):
                        self.assertEqual(
                            probes[0]["stable_world_effect_cells"],
                            len(world_index),
                        )
                        self.assertFalse(probes[0]["localized"])
                    else:
                        self.assertEqual(
                            probes[0]["stable_world_effect_cells"], 0
                        )
                    if persistent and world_index == 1:
                        self.assertEqual(
                            probes[0][
                                "local_only_persistent_world_effect_cells"
                            ],
                            1,
                        )

    def test_human_prior_option_effect_frontier_rejects_effect_without_gain(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            WorldEffectEnv(True),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
                human_prior_option_effect_probe_limit=1,
                human_prior_option_effect_phase_offsets=2,
                human_prior_option_effect_frontier=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 0)
        self.assertEqual(len(agent.archive), 0)
        eligible = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_effect_frontier_eligible"
        ]
        self.assertEqual(eligible, [])
        controls = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_world_effect_action_control"
        ]
        self.assertEqual(len(controls), 1)
        self.assertTrue(controls[0]["confirmed"])
        self.assertEqual(
            controls[0]["controllability"][
                "reachable_player_position_gain"
            ],
            0,
        )

    def test_causal_effect_frontier_archives_effect_without_immediate_gain(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            WorldEffectEnv(True),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
                human_prior_option_effect_probe_limit=1,
                human_prior_option_effect_phase_offsets=2,
                human_prior_option_causal_effect_frontier=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(len(agent.archive), 1)
        self.assertTrue(agent.archive[0].goal_world_effect_signature)
        eligible = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_effect_frontier_eligible"
        ]
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0]["frontier_reason"], "delayed_causal_effect")
        self.assertEqual(eligible[0]["confirmed_action_indices"], (0,))
        self.assertEqual(
            eligible[0]["reachability_confirmed_action_indices"], ()
        )

    def test_causal_effect_frontier_is_retained_beside_primary_movement(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            WorldEffectAndMovementEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.A),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=4,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
                human_prior_option_effect_probe_limit=4,
                human_prior_option_effect_phase_offsets=2,
                human_prior_option_causal_effect_frontier=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 2)
        self.assertEqual(len(agent.archive), 2)
        archived = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_archive_added"
        ]
        self.assertEqual(sum(event["selected_primary"] for event in archived), 1)
        self.assertEqual(
            sum(bool(event["human_prior_option_effect_frontier"]) for event in archived),
            1,
        )
        completed = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_completed"
        ][0]
        self.assertEqual(completed["distinct_effect_contexts_archived"], 1)

        primary = next(
            branch
            for branch in agent.archive
            if not branch.goal_world_effect_signature
        )
        causal = next(
            branch
            for branch in agent.archive
            if branch.goal_world_effect_signature
        )
        causal.human_prior_option_effect_frontier_reason = (
            "immediate_reachability_gain"
        )
        agent.frame = agent.env.load_state(primary.state)
        agent.goal_prior.restore(
            primary.goal_heart_slots,
            primary.frame,
            primary.goal_player_slot,
        )
        # Reproduce a localized effect that is intentionally invisible to the
        # generic coarse scene signature used by archive recovery.  The
        # primary movement remains an unvisited physical frontier; the
        # experimentally confirmed immediate reachability gain takes priority.
        agent._signature = lambda frame: "same-coarse-scene"
        agent.human_prior_graph_recovery_pending = True

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(agent.frame, causal.frame)
        self.assertEqual(
            agent.current_human_prior_world_context_signature,
            causal.goal_target_world_context,
        )
        filtered = next(
            event
            for event in logger.events
            if event["event"]
            == "human_prior_best_first_archives_filtered"
        )
        self.assertTrue(
            filtered["immediate_option_effect_frontier_preferred"]
        )
        self.assertFalse(filtered["physical_frontier_preferred"])

    def test_option_effect_controllability_detects_new_reachable_slot(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ControllabilityGainEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_option_effect_stability_steps=0,
                human_prior_option_effect_frontier=True,
                human_prior_option_effect_phase_offsets=1,
            ),
            event_logger=logger,
        )
        source = agent.reset()
        agent.goal_prior = FootprintPositionGoalPrior()
        root = env.save_state()

        result = agent._probe_human_prior_option_controllability_gain(
            root,
            source,
            (Action.RIGHT,),
            (1,),
            (Action.NOOP,),
            1,
            1,
            0,
        )

        self.assertTrue(result["endpoint_matched"])
        self.assertTrue(result["player_footprint_matched"])
        self.assertEqual(
            result["factual_reachable_player_slots"], ((1, 0),)
        )
        self.assertEqual(
            result["control_reachable_player_slots"], ((0, 0),)
        )
        self.assertEqual(result["newly_reachable_player_slots"], ((1, 0),))
        self.assertEqual(result["reachable_player_position_gain"], 1)
        self.assertTrue(
            any(
                event["event"]
                == "human_prior_option_effect_controllability_probe"
                for event in logger.events
            )
        )

    def test_option_effect_controllability_rejects_pose_only_gain(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = PoseControllabilityGainEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_option_effect_stability_steps=0,
                human_prior_option_effect_frontier=True,
                human_prior_option_effect_phase_offsets=1,
            ),
        )
        source = agent.reset()
        agent.goal_prior = PosePositionGoalPrior()
        root = env.save_state()

        result = agent._probe_human_prior_option_controllability_gain(
            root,
            source,
            (Action.LEFT,),
            (1,),
            (Action.NOOP,),
            1,
            1,
            0,
        )

        self.assertTrue(result["endpoint_matched"])
        self.assertFalse(result["player_footprint_matched"])
        self.assertEqual(
            result["player_footprint_symmetric_difference_pixels"], 2
        )
        self.assertEqual(result["factual_reachable_player_slots"], ())
        self.assertEqual(result["control_reachable_player_slots"], ())
        self.assertEqual(result["reachable_player_position_gain"], 0)

    def test_option_effect_controllability_detects_two_step_gain(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = DelayedControllabilityGainEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.A),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_option_effect_stability_steps=0,
                human_prior_option_effect_frontier=True,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_controllability_depth=2,
            ),
        )
        source = agent.reset()
        agent.goal_prior = FootprintPositionGoalPrior()
        root = env.save_state()

        result = agent._probe_human_prior_option_controllability_gain(
            root,
            source,
            (Action.A,),
            (1,),
            (Action.NOOP,),
            1,
            1,
            0,
        )

        self.assertEqual(result["probe_depth"], 2)
        self.assertEqual(
            result["factual_reachable_player_slots"], ((0, 0), (1, 0))
        )
        self.assertEqual(
            result["control_reachable_player_slots"], ((0, 0),)
        )
        self.assertEqual(result["newly_reachable_player_slots"], ((1, 0),))
        self.assertEqual(result["reachable_player_position_gain"], 1)
        self.assertTrue(
            any(
                row["path"] == (Action.RIGHT, Action.RIGHT)
                and row["factual_player_slot"] == (1, 0)
                for row in result["actions"]
            )
        )

    def test_human_prior_option_local_control_uses_matched_endpoint(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            WorldEffectEnv(True, 1),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
                human_prior_option_effect_probe_limit=1,
                human_prior_option_effect_local_controls=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 0)
        stability = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_world_effect_stability"
        ]
        self.assertEqual(len(stability), 1)
        self.assertFalse(stability[0]["stable"])
        self.assertTrue(stability[0]["local_candidate"])
        controls = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_world_effect_action_control"
        ]
        self.assertEqual(len(controls), 1)
        self.assertEqual(
            controls[0]["control_mode"], "endpoint_matched_local"
        )
        self.assertTrue(controls[0]["endpoint_matched"])
        self.assertTrue(controls[0]["confirmed"])
        self.assertEqual(controls[0]["minimum_cell_pixels"], 12)
        self.assertEqual(
            controls[0]["observations"][0]["ignored_player_pixels"],
            0,
        )

    def test_unlabeled_entity_frontier_archives_local_transformation(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        env = UnlabeledEntityTransformEnv(remote_display=True)
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT, Action.A),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=4,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
                human_prior_option_effect_probe_limit=4,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(len(agent.archive), 1)
        branch = agent.archive[0]
        self.assertEqual(branch.plan.path, (Action.RIGHT, Action.A))
        self.assertTrue(branch.human_prior_option_entity_state_signature)
        eligible = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_entity_frontier_eligible"
        ]
        self.assertEqual(len(eligible), 1)
        self.assertTrue(eligible[0]["eligible"])
        self.assertEqual(eligible[0]["entity_effect_cells"], ((1, 0),))
        self.assertEqual(eligible[0]["confirmed_action_indices"], (0, 1))

    def test_unlabeled_entity_behavior_transfers_to_second_encounter(
        self,
    ) -> None:
        behavior_model = AnonymousEntityBehaviorModel(
            minimum_prediction_samples=1
        )

        def run(learning: bool) -> RecordingLogger:
            dynamics = EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            )
            logger = RecordingLogger()
            agent = VerifiedNeuralAgent(
                UnlabeledEntityTransformEnv(),
                dynamics,
                "cpu",
                NeuralPlanningConfig(
                    actions=(Action.LEFT, Action.RIGHT, Action.A),
                    planning_depth=1,
                    action_frames=1,
                    human_prior_heart_reward=1.0,
                    human_prior_best_first_archive=True,
                    human_prior_option_search_depth=2,
                    human_prior_option_search_beam_width=4,
                    human_prior_option_search_action_frames=1,
                    human_prior_option_effect_stability_steps=2,
                    human_prior_option_effect_probe_limit=4,
                    human_prior_option_effect_phase_offsets=1,
                    human_prior_option_effect_local_controls=True,
                    human_prior_option_entity_frontier=True,
                    anonymous_entity_behavior_learning=learning,
                    causal_spatial_columns=8,
                    causal_spatial_rows=8,
                ),
                event_logger=logger,
                entity_behavior_model=behavior_model,
            )
            agent.reset()
            agent.goal_prior = PositionGoalPrior()
            source_signature = agent._current_human_prior_graph_signature()
            agent.human_prior_graph_state_visits[source_signature] = 1
            agent.human_prior_player_position_visits[(0, 0)] = 1
            agent._search_human_prior_options()
            return logger

        learned = run(True)
        digest_after_learning = behavior_model.digest
        observed = run(False)

        learned_events = [
            event
            for event in learned.events
            if event["event"] == "anonymous_entity_behavior_observed"
            and event["evidence_accepted"]
        ]
        transfer_events = [
            event
            for event in observed.events
            if event["event"] == "anonymous_entity_behavior_observed"
            and event["behavior_known_before"]
        ]
        self.assertGreaterEqual(len(learned_events), 1)
        self.assertGreaterEqual(len(transfer_events), 1)
        self.assertTrue(
            any(event["outcome_matched_prediction"] for event in transfer_events)
        )
        self.assertEqual(behavior_model.digest, digest_after_learning)
        self.assertTrue(
            all(not event["learning_enabled"] for event in transfer_events)
        )

    def test_entity_curiosity_learns_static_interaction_and_reuses_it(
        self,
    ) -> None:
        behavior_model = AnonymousEntityBehaviorModel(
            minimum_prediction_samples=1
        )

        def run(learning: bool) -> RecordingLogger:
            logger = RecordingLogger()
            agent = VerifiedNeuralAgent(
                UnlabeledEntityTransformEnv(),
                EnsembleVisualDynamicsModel(
                    latent_size=32, action_size=8, ensemble_size=2
                ),
                "cpu",
                NeuralPlanningConfig(
                    actions=(Action.RIGHT,),
                    planning_depth=1,
                    action_frames=1,
                    human_prior_heart_reward=1.0,
                    human_prior_best_first_archive=True,
                    human_prior_option_search_depth=2,
                    human_prior_option_search_beam_width=1,
                    human_prior_option_search_action_frames=1,
                    human_prior_option_effect_stability_steps=2,
                    human_prior_option_effect_probe_limit=1,
                    human_prior_option_effect_phase_offsets=1,
                    human_prior_option_effect_local_controls=True,
                    human_prior_option_entity_frontier=True,
                    human_prior_option_entity_curiosity_reserve=1,
                    human_prior_option_entity_inert_penalty_weight=2.0,
                    anonymous_entity_behavior_learning=learning,
                    causal_spatial_columns=8,
                    causal_spatial_rows=8,
                ),
                event_logger=logger,
                entity_behavior_model=behavior_model,
            )
            agent.reset()
            agent.goal_prior = PositionGoalPrior()
            source_signature = agent._current_human_prior_graph_signature()
            agent.human_prior_graph_state_visits[source_signature] = 1
            agent.human_prior_player_position_visits[(0, 0)] = 1

            self.assertEqual(agent._search_human_prior_options(), 0)
            return logger

        learned = run(True)
        learned_digest = behavior_model.digest
        frozen = run(False)

        learned_probe = next(
            event
            for event in learned.events
            if event["event"]
            == "human_prior_option_entity_curiosity_probe"
        )
        frozen_probe = next(
            event
            for event in frozen.events
            if event["event"]
            == "human_prior_option_entity_curiosity_probe"
        )
        self.assertEqual(learned_probe["action"], Action.RIGHT)
        self.assertEqual(learned_probe["interaction_cell"], (1, 0))
        self.assertFalse(learned_probe["entity_effect_confirmed"])
        self.assertTrue(learned_probe["evidence_accepted"])
        self.assertTrue(learned_probe["evidence_eligible"])
        self.assertTrue(learned_probe["interaction_cell_matched"])
        self.assertFalse(learned_probe["behavior_known_before"])
        self.assertTrue(frozen_probe["behavior_known_before"])
        self.assertGreaterEqual(frozen_probe["semantic_samples_before"], 1)
        self.assertEqual(frozen_probe["semantic_coverage_before"], 1.0)
        self.assertEqual(frozen_probe["inert_probability_before"], 1.0)
        self.assertGreater(frozen_probe["inert_confidence_before"], 0.0)
        self.assertGreater(frozen_probe["inert_penalty"], 0.0)
        self.assertFalse(frozen_probe["evidence_accepted"])
        self.assertEqual(behavior_model.digest, learned_digest)
        self.assertTrue(
            any(
                event["event"]
                == "human_prior_option_search_depth_completed"
                and event[
                    "anonymous_entity_curiosity_parents_retained"
                ]
                == 1
                for event in learned.events
            )
        )
        learned_observation = next(
            event
            for event in learned.events
            if event["event"] == "anonymous_entity_behavior_observed"
            and event["evidence_accepted"]
        )
        self.assertTrue(learned_observation["observed_intervention_inert"])
        self.assertIsNotNone(
            learned_observation["observed_outcome_descriptor"]
        )

    def test_entity_curiosity_probe_ranking_diversifies_spatial_loci(
        self,
    ) -> None:
        def candidate(
            cell: tuple[int, int], novelty: float, score: float
        ) -> SimpleNamespace:
            return SimpleNamespace(
                entity_behavior_known=False,
                entity_interaction_type_id=1,
                entity_behavior_novelty=novelty,
                entity_spatial_rarity=1.0,
                entity_curiosity=novelty,
                score=score,
                depth=2,
                entity_interaction_cell=cell,
                entity_interaction_appearance_fingerprint="appearance",
                entity_interaction_context_signature="context",
            )

        same_cell_best = candidate((1, 1), 1.0, 4.0)
        same_cell_second = candidate((1, 1), 0.9, 3.0)
        second_cell = candidate((2, 1), 0.8, 2.0)
        third_cell = candidate((3, 1), 0.7, 1.0)

        ranked, groups = (
            VerifiedNeuralAgent._human_prior_diverse_entity_probe_candidates(
                (
                    same_cell_best,
                    same_cell_second,
                    second_cell,
                    third_cell,
                )
            )
        )

        self.assertEqual(groups, 3)
        self.assertEqual(
            [node.entity_interaction_cell for node in ranked[:3]],
            [(1, 1), (2, 1), (3, 1)],
        )
        self.assertIs(ranked[3], same_cell_second)

    def test_entity_curiosity_reserve_cannot_exceed_option_beam(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            VerifiedNeuralAgent(
                UnlabeledEntityTransformEnv(),
                EnsembleVisualDynamicsModel(
                    latent_size=32, action_size=8, ensemble_size=2
                ),
                "cpu",
                NeuralPlanningConfig(
                    actions=(Action.RIGHT,),
                    planning_depth=1,
                    human_prior_option_search_beam_width=1,
                    human_prior_option_effect_stability_steps=1,
                    human_prior_option_effect_phase_offsets=1,
                    human_prior_option_effect_local_controls=True,
                    human_prior_option_entity_frontier=True,
                    human_prior_option_entity_curiosity_reserve=2,
                ),
                entity_behavior_model=AnonymousEntityBehaviorModel(),
            )

    def test_learned_inert_prior_cannot_override_verified_movement(
        self,
    ) -> None:
        behavior_model = AnonymousEntityBehaviorModel(
            minimum_prediction_samples=1
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=1,
                human_prior_option_effect_probe_limit=1,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                human_prior_option_entity_inert_penalty_weight=2.0,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
            entity_behavior_model=behavior_model,
        )
        frame = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        feature_index = agent._human_prior_option_entity_feature_index(
            frame
        )
        prior = agent._human_prior_option_entity_curiosity(
            frame,
            (0, 0),
            None,
            Action.RIGHT,
            1,
            feature_index,
        )
        feature = feature_index[0][(1, 0)]
        descriptor = behavior_model.effect_descriptor(
            feature, feature, feature
        )
        behavior_model.observe(
            feature,
            Action.RIGHT,
            1,
            descriptor.signature,
            context_signature=prior["context_signature"],
            outcome_descriptor=descriptor,
        )
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        agent._search_human_prior_options()

        branch = next(
            event
            for event in logger.events
            if event["event"] == "human_prior_option_branch_verified"
            and event["path"] == (Action.RIGHT,)
        )
        self.assertGreater(
            branch["anonymous_entity_predicted_inert_penalty"], 0.0
        )
        self.assertTrue(
            branch["anonymous_entity_current_branch_measured_effect"]
        )
        self.assertTrue(
            branch["anonymous_entity_inert_penalty_suppressed"]
        )
        self.assertFalse(
            branch["anonymous_entity_inert_penalty_eligible"]
        )
        self.assertEqual(branch["anonymous_entity_inert_penalty"], 0.0)

    def test_entity_curiosity_prefers_transferable_appearance_type(
        self,
    ) -> None:
        behavior_model = AnonymousEntityBehaviorModel()
        agent = VerifiedNeuralAgent(
            UnlabeledEntityTransformEnv(),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.DOWN),
                planning_depth=1,
                human_prior_option_search_beam_width=2,
                human_prior_option_effect_stability_steps=1,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                human_prior_option_entity_curiosity_reserve=1,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            entity_behavior_model=behavior_model,
        )
        frame = agent.reset()
        feature_index = agent._human_prior_option_entity_feature_index(frame)
        entity_feature = feature_index[0][(1, 0)]
        behavior_model.observe(
            entity_feature,
            Action.LEFT,
            1,
            "prior-outcome",
        )

        transferable = agent._human_prior_option_entity_curiosity(
            frame,
            (0, 0),
            None,
            Action.RIGHT,
            1,
            feature_index,
        )
        unfamiliar = agent._human_prior_option_entity_curiosity(
            frame,
            (0, 0),
            None,
            Action.DOWN,
            1,
            feature_index,
        )

        self.assertIsNotNone(transferable["anonymous_type_id"])
        self.assertIsNone(unfamiliar["anonymous_type_id"])
        self.assertEqual(transferable["appearance_transferability"], 1.0)
        self.assertEqual(unfamiliar["appearance_transferability"], 0.25)
        self.assertGreater(
            transferable["curiosity"], unfamiliar["curiosity"]
        )

    def test_entity_interaction_replay_carries_player_reference(self) -> None:
        env = ActionEffectEnv()
        agent = VerifiedNeuralAgent(
            env,
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_option_effect_stability_steps=1,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
        )
        source = agent.reset()
        agent.goal_prior = ReferenceSensitivePositionGoalPrior()
        source_analysis = agent.goal_prior.analyze(source, source)
        root = env.save_state()
        node = _HumanPriorOptionNode(
            state=root,
            frame=source,
            path=(Action.RIGHT, Action.RIGHT),
            durations=(1, 1),
            analysis=source_analysis,
            source_signature="source",
            target_signature="target",
            score=0.0,
            depth=2,
            target_state_visits=0,
            target_position_visits=0,
        )

        before, direction, interaction_ray = (
            agent._human_prior_option_interaction_ray(
                root,
                source,
                node,
                1,
            )
        )

        self.assertEqual(before.pixels.index(255), 1)
        self.assertEqual(direction, Action.RIGHT)
        self.assertEqual(interaction_ray[0], (2, 0))

    def test_passive_scan_learns_anonymous_autonomous_motion(self) -> None:
        behavior_model = AnonymousEntityBehaviorModel(
            minimum_prediction_samples=1
        )

        def run(learning: bool) -> RecordingLogger:
            logger = RecordingLogger()
            agent = VerifiedNeuralAgent(
                PassiveRareEntityEnv(),
                EnsembleVisualDynamicsModel(
                    latent_size=32, action_size=8, ensemble_size=2
                ),
                "cpu",
                NeuralPlanningConfig(
                    actions=(Action.NOOP,),
                    planning_depth=1,
                    action_frames=1,
                    human_prior_option_effect_stability_steps=1,
                    human_prior_option_effect_phase_offsets=1,
                    human_prior_option_effect_local_controls=True,
                    human_prior_option_entity_frontier=True,
                    anonymous_entity_behavior_learning=learning,
                    causal_spatial_columns=4,
                    causal_spatial_rows=4,
                ),
                event_logger=logger,
                entity_behavior_model=behavior_model,
            )
            agent.reset()
            agent.decide()
            agent.clear_archive()
            return logger

        learned = run(True)
        digest = behavior_model.digest
        frozen = run(False)

        learned_motion = [
            event
            for event in learned.events
            if event["event"] == "anonymous_entity_behavior_observed"
            and event["autonomous"]
            and event["relative_effect_cells"] == ((1, 0),)
        ]
        predicted_motion = [
            event
            for event in frozen.events
            if event["event"] == "anonymous_entity_behavior_observed"
            and event["behavior_known_before"]
            and event["outcome_matched_prediction"]
            and event["relative_effect_cells"] == ((1, 0),)
        ]
        self.assertEqual(len(learned_motion), 1)
        self.assertEqual(len(predicted_motion), 1)
        self.assertEqual(behavior_model.digest, digest)
        self.assertTrue(
            any(
                event["event"]
                == "anonymous_entity_passive_scan_completed"
                and event["candidate_cells"] == 1
                for event in frozen.events
            )
        )

    def test_passive_motion_ignores_distant_duplicate_appearances(self) -> None:
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            PassiveRareEntityEnv(duplicate=True),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP,),
                planning_depth=1,
                action_frames=1,
                human_prior_option_effect_stability_steps=1,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                anonymous_entity_behavior_learning=True,
                causal_spatial_columns=4,
                causal_spatial_rows=4,
            ),
            event_logger=logger,
            entity_behavior_model=AnonymousEntityBehaviorModel(
                minimum_prediction_samples=1
            ),
        )
        agent.reset()
        agent.decide()
        agent.clear_archive()

        moving = [
            event
            for event in logger.events
            if event["event"] == "anonymous_entity_behavior_observed"
            and event["anchor_cell"] == (1, 1)
        ]
        stationary = [
            event
            for event in logger.events
            if event["event"] == "anonymous_entity_behavior_observed"
            and event["anchor_cell"] == (3, 3)
        ]
        self.assertEqual(moving[0]["relative_effect_cells"], ((1, 0),))
        self.assertEqual(stationary[0]["relative_effect_cells"], ((0, 0),))

    def test_passive_horizon_discovers_delayed_anonymous_motion(self) -> None:
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            PassiveRareEntityEnv(minimum_motion_frames=2),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP,),
                planning_depth=1,
                action_frames=1,
                human_prior_option_effect_stability_steps=1,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                anonymous_entity_behavior_learning=True,
                anonymous_entity_passive_horizons=(1, 2),
                causal_spatial_columns=4,
                causal_spatial_rows=4,
            ),
            event_logger=logger,
            entity_behavior_model=AnonymousEntityBehaviorModel(
                minimum_prediction_samples=1
            ),
        )
        agent.reset()
        agent.goal_prior = HazardPositionGoalPrior()
        agent.decide()
        agent.clear_archive()

        observations = [
            event
            for event in logger.events
            if event["event"] == "anonymous_entity_behavior_observed"
            and event["anchor_cell"] == (1, 1)
        ]
        horizons = [
            event
            for event in logger.events
            if event["event"]
            == "anonymous_entity_passive_horizon_verified"
        ]
        self.assertEqual(
            {
                event["action_frames"]: event["relative_effect_cells"]
                for event in observations
            },
            {1: ((0, 0),), 2: ((1, 0),)},
        )
        self.assertEqual(
            [event["action_frames"] for event in horizons],
            [2],
        )
        delayed = next(
            event
            for event in observations
            if event["action_frames"] == 2
        )
        self.assertTrue(delayed["observed_hazard"])
        self.assertTrue(
            delayed["differential_terminal_visual_change"]
        )
        self.assertFalse(delayed["evidence_eligible"])
        self.assertFalse(delayed["evidence_accepted"])
        self.assertEqual(delayed["hazard_probability_after"], 0.0)

    def test_causal_wait_contrast_localizes_delayed_hazard(self) -> None:
        logger = RecordingLogger()
        model = AnonymousEntityBehaviorModel(
            minimum_prediction_samples=1
        )
        agent = VerifiedNeuralAgent(
            CausalRareEntityEnv(),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.NOOP),
                planning_depth=1,
                verify_actions=2,
                action_frames=1,
                human_prior_option_effect_stability_steps=1,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                anonymous_entity_behavior_learning=True,
                anonymous_entity_causal_horizons=(2, 3),
                causal_spatial_columns=4,
                causal_spatial_rows=4,
            ),
            event_logger=logger,
            entity_behavior_model=model,
        )
        agent.reset()
        agent.goal_prior = CausalEntityGoalPrior()
        agent.decide()
        agent.clear_archive()

        contrasts = [
            event
            for event in logger.events
            if event["event"]
            == "anonymous_entity_causal_contrast_completed"
        ]
        terminal = [
            event
            for event in logger.events
            if event["event"] == "anonymous_entity_behavior_observed"
            and event.get("causal_attribution")
            and event["action_frames"] == 3
        ]

        self.assertEqual(len(contrasts), 2)
        self.assertGreaterEqual(
            contrasts[0]["newly_localized_candidates"], 1
        )
        self.assertTrue(contrasts[1]["hazard_contrast"])
        self.assertEqual(
            {event["anchor_cell"] for event in terminal}, {(2, 1)}
        )
        self.assertEqual(
            {
                event["causal_role"]: event["observed_hazard"]
                for event in terminal
            },
            {"intervention": True, "neutral_control": False},
        )
        self.assertTrue(
            all(
                event["causal_localization_horizon"] == 2
                for event in terminal
            )
        )

    def test_entity_shadow_predicts_hazard_without_changing_policy(
        self,
    ) -> None:
        behavior_model = AnonymousEntityBehaviorModel(
            minimum_prediction_samples=1
        )
        learning_agent = VerifiedNeuralAgent(
            CausalRareEntityEnv(),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.NOOP),
                planning_depth=1,
                verify_actions=2,
                action_frames=1,
                human_prior_option_effect_stability_steps=1,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                anonymous_entity_behavior_learning=True,
                anonymous_entity_causal_horizons=(2, 3),
                causal_spatial_columns=4,
                causal_spatial_rows=4,
            ),
            entity_behavior_model=behavior_model,
        )
        learning_agent.reset()
        learning_agent.goal_prior = CausalEntityGoalPrior()
        learning_agent.decide()
        learning_agent.clear_archive()
        learned_digest = behavior_model.digest

        base = dict(
            actions=(Action.RIGHT, Action.NOOP),
            planning_depth=1,
            verify_actions=2,
            action_frames=1,
            human_prior_option_effect_stability_steps=1,
            human_prior_option_effect_phase_offsets=1,
            human_prior_option_effect_local_controls=True,
            human_prior_option_entity_frontier=True,
            causal_spatial_columns=4,
            causal_spatial_rows=4,
        )
        dynamics = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        control = VerifiedNeuralAgent(
            CausalRareEntityEnv(),
            dynamics,
            "cpu",
            NeuralPlanningConfig(**base),
            entity_behavior_model=behavior_model,
        )
        logger = RecordingLogger()
        shadow = VerifiedNeuralAgent(
            CausalRareEntityEnv(),
            dynamics,
            "cpu",
            NeuralPlanningConfig(
                **base,
                anonymous_entity_shadow_horizons=(2, 3),
                anonymous_entity_shadow_hazard_threshold=0.9,
            ),
            event_logger=logger,
            entity_behavior_model=behavior_model,
        )
        control.reset()
        shadow.reset()
        control.goal_prior = CausalEntityGoalPrior()
        shadow.goal_prior = CausalEntityGoalPrior()

        control_decision = control.decide()
        shadow_decision = shadow.decide()
        control.clear_archive()
        shadow.clear_archive()

        self.assertEqual(shadow_decision.action, control_decision.action)
        self.assertEqual(
            shadow_decision.action_frames,
            control_decision.action_frames,
        )
        self.assertEqual(shadow_decision.frame, control_decision.frame)
        self.assertAlmostEqual(shadow_decision.score, control_decision.score)
        self.assertEqual(behavior_model.digest, learned_digest)

        branch_events = [
            event
            for event in logger.events
            if event["event"]
            == "anonymous_entity_behavior_shadow_branch_evaluated"
        ]
        right_branch = next(
            event
            for event in branch_events
            if event["action"] == Action.RIGHT
        )
        noop_branch = next(
            event
            for event in branch_events
            if event["action"] == Action.NOOP
        )
        self.assertTrue(right_branch["shadow_would_reject"])
        self.assertEqual(
            right_branch["shadow_max_hazard_probability"], 1.0
        )
        self.assertEqual(right_branch["shadow_implicated_horizon"], 3)
        self.assertFalse(noop_branch["shadow_would_reject"])
        self.assertTrue(
            all(
                not event["shadow_policy_authority"]
                and event["shadow_selection_weight"] == 0.0
                and event["model_parameters_unchanged"]
                for event in branch_events
            )
        )

        entity_prediction = next(
            event
            for event in logger.events
            if event["event"]
            == "anonymous_entity_behavior_shadow_prediction"
            and event["action"] == Action.RIGHT
            and event["anchor_cell"] == (2, 1)
            and event["horizon_frames"] == 3
        )
        self.assertTrue(entity_prediction["behavior_known"])
        self.assertTrue(entity_prediction["context_matched"])
        self.assertEqual(entity_prediction["hazard_probability"], 1.0)
        self.assertEqual(
            entity_prediction["unconditional_hazard_probability"],
            0.5,
        )
        self.assertTrue(entity_prediction["shadow_would_reject"])

        veto_logger = RecordingLogger()
        veto = VerifiedNeuralAgent(
            CausalRareEntityEnv(),
            dynamics,
            "cpu",
            NeuralPlanningConfig(
                **base,
                anonymous_entity_shadow_horizons=(2, 3),
                anonymous_entity_shadow_hazard_threshold=0.9,
                anonymous_entity_hazard_veto=True,
            ),
            event_logger=veto_logger,
            entity_behavior_model=behavior_model,
        )
        veto.reset()
        veto.goal_prior = CausalEntityGoalPrior()
        veto_decision = veto.decide()
        veto.clear_archive()

        self.assertEqual(control_decision.action, Action.RIGHT)
        self.assertEqual(veto_decision.action, Action.NOOP)
        veto_event = next(
            event
            for event in veto_logger.events
            if event["event"]
            == "anonymous_entity_hazard_veto_evaluated"
        )
        self.assertEqual(veto_event["hazards_detected"], 1)
        self.assertEqual(veto_event["hazards_filtered"], 1)
        self.assertEqual(veto_event["alternatives_remaining"], 1)
        self.assertFalse(veto_event["fail_open"])

        fail_open_logger = RecordingLogger()
        fail_open = VerifiedNeuralAgent(
            CausalRareEntityEnv(),
            dynamics,
            "cpu",
            NeuralPlanningConfig(
                **{
                    **base,
                    "actions": (Action.RIGHT,),
                    "verify_actions": 1,
                },
                anonymous_entity_shadow_horizons=(2, 3),
                anonymous_entity_hazard_veto=True,
            ),
            event_logger=fail_open_logger,
            entity_behavior_model=behavior_model,
        )
        fail_open.reset()
        fail_open.goal_prior = CausalEntityGoalPrior()
        fail_open_decision = fail_open.decide()
        fail_open.clear_archive()

        self.assertEqual(fail_open_decision.action, Action.RIGHT)
        fail_open_event = next(
            event
            for event in fail_open_logger.events
            if event["event"]
            == "anonymous_entity_hazard_veto_evaluated"
        )
        self.assertTrue(fail_open_event["fail_open"])
        self.assertEqual(fail_open_event["hazards_filtered"], 0)

    def test_causal_wait_withholds_unlocalized_global_hazard(self) -> None:
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            CausalRareEntityEnv(local_motion=False),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.NOOP),
                planning_depth=1,
                verify_actions=2,
                action_frames=1,
                human_prior_option_effect_stability_steps=1,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                anonymous_entity_behavior_learning=True,
                anonymous_entity_causal_horizons=(2, 3),
                causal_spatial_columns=4,
                causal_spatial_rows=4,
            ),
            event_logger=logger,
            entity_behavior_model=AnonymousEntityBehaviorModel(
                minimum_prediction_samples=1
            ),
        )
        agent.reset()
        agent.goal_prior = CausalEntityGoalPrior()
        agent.decide()
        agent.clear_archive()

        contrasts = [
            event
            for event in logger.events
            if event["event"]
            == "anonymous_entity_causal_contrast_completed"
        ]
        attributed = [
            event
            for event in logger.events
            if event["event"] == "anonymous_entity_behavior_observed"
            and event.get("causal_attribution")
        ]

        self.assertTrue(contrasts[-1]["hazard_contrast"])
        self.assertEqual(
            sum(
                event["newly_localized_candidates"]
                for event in contrasts
            ),
            0,
        )
        self.assertEqual(attributed, [])

    def test_option_search_can_add_long_direction_edges(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            LongPressMovementEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.A),
                planning_depth=1,
                action_frames=4,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=4,
                human_prior_option_search_action_frames=4,
                human_prior_option_search_long_direction_frames=16,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(len(agent.archive), 1)
        self.assertIn(16, agent.archive[0].plan.durations)
        started = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_started"
        ][0]
        self.assertEqual(
            started["action_duration_edges"],
            (
                (Action.RIGHT, 4),
                (Action.RIGHT, 16),
                (Action.A, 4),
            ),
        )
        neutral = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_neutral_verified"
        ]
        self.assertIn(4, {event["elapsed_frames"] for event in neutral})
        self.assertIn(16, {event["elapsed_frames"] for event in neutral})

    def test_unlabeled_entity_frontier_archives_settled_state(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        env = MovingEntitySettlesEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.A),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=4,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
                human_prior_option_effect_probe_limit=4,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(len(agent.archive), 1)
        branch = agent.archive[0]
        self.assertEqual(branch.plan.path, (Action.RIGHT, Action.A))
        self.assertEqual(branch.state, (True, True, 2))
        self.assertEqual(branch.frame, env.load_state(branch.state))
        eligible = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_entity_frontier_eligible"
        ]
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0]["settling_steps"], 2)
        self.assertEqual(eligible[0]["settling_frames"], 2)
        self.assertNotEqual(
            eligible[0]["immediate_frame"], eligible[0]["frame"]
        )
        archived = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_archive_added"
        ]
        self.assertEqual(archived[0]["human_prior_option_settling_steps"], 2)
        self.assertEqual(archived[0]["human_prior_option_settling_frames"], 2)
        self.assertEqual(
            archived[0]["human_prior_option_immediate_frame"],
            eligible[0]["immediate_frame"],
        )

    def test_milestone_option_archives_settled_semantic_outcome(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        env = MovingMilestoneSettlesEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.A),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_life_loss_penalty=100.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=3,
                human_prior_option_search_beam_width=4,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        agent.goal_prior = MovingMilestoneGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(len(agent.archive), 1)
        branch = agent.archive[0]
        self.assertEqual(branch.state, (True, True, 2))
        self.assertEqual(branch.frame, env.load_state(branch.state))
        self.assertEqual(branch.goal_player_slot, (2, 0))
        self.assertEqual(branch.goal_progress_reward, 27.0)
        settled = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_milestone_settled"
        ]
        self.assertEqual(len(settled), 1)
        self.assertTrue(all(event["settling_steps"] == 2 for event in settled))
        self.assertTrue(
            all(
                event["human_prior_target_player_slot"] == (2, 0)
                for event in settled
            )
        )
        collapsed = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_milestone_candidates_collapsed"
        ]
        self.assertEqual(len(collapsed), 1)
        self.assertGreater(
            collapsed[0]["candidates_before"],
            collapsed[0]["representatives_after"],
        )
        archived = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_archive_added"
        ]
        self.assertEqual(archived[0]["human_prior_option_settling_steps"], 2)
        self.assertEqual(
            archived[0]["human_prior_target_player_slot"], (2, 0)
        )

        # Exercise native-style recovery where stable pixels contain the
        # player but the archived analysis did not retain its slot.
        branch.goal_player_slot = None
        agent.human_prior_graph_recovery_pending = True
        decision = agent._restore_if_stagnant()
        self.assertIsNotNone(decision)
        outcome_key = (
            ((7, 0),),
            (),
            (2, 0),
            False,
            False,
        )
        self.assertIn(outcome_key, agent.human_prior_milestone_outcomes)
        recorded = [
            event
            for event in logger.events
            if event["event"] == "human_prior_milestone_outcome_recorded"
        ]
        self.assertEqual(
            recorded[-1]["target_player_slot_source"],
            "restored_goal_prior",
        )
        checkpoint = agent.pending_goal_milestone_checkpoint
        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint.goal_target_heart_slots, ())
        self.assertTrue(checkpoint.goal_target_heart_slots_known)
        env.load_state(checkpoint.state)
        agent.frame = source_frame
        agent.goal_prior.restore(((7, 0),), source_frame, (0, 0))

        agent._search_human_prior_options()

        duplicates = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_milestone_duplicate_rejected"
        ]
        self.assertGreaterEqual(len(duplicates), 1)
        # The already-recorded +25 milestone must not be archived again.
        # Ordinary endpoints may retain their signed navigation progress.
        self.assertTrue(
            all(branch.goal_progress_reward < 25.0 for branch in agent.archive)
        )

    def test_milestone_settlement_tracks_from_verified_endpoint(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = VisibleMovingMilestoneSettlesEnv()
        prior = RecordingMovingMilestoneGoalPrior()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.A),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=3,
                human_prior_option_search_beam_width=4,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
            ),
        )
        source = agent.reset()
        agent.goal_prior = prior
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        agent._search_human_prior_options()

        self.assertEqual(len(agent.archive), 1)
        settled = agent.archive[0]
        self.assertEqual(settled.goal_player_slot, (2, 0))
        self.assertIn(
            (source.digest, settled.frame.digest, (2, 0)),
            prior.analysis_calls,
        )

    def test_milestone_selection_preserves_closest_followthrough(self) -> None:
        source = HeartGoalAnalysis(
            reliable=True,
            known_slots=((1, 0), (7, 0)),
            source_present=((1, 0), (7, 0)),
            target_present=((7, 0),),
            collected=((1, 0),),
            target_similarities=(),
            heart_reward=25.0,
            all_hearts_reward=0.0,
            chest_reward=0.0,
            navigation_reward=0.0,
            life_loss_penalty=0.0,
            total_reward=25.0,
            global_visual_change=0.0,
            target_intensity=1.0,
            source_player_slot=(0, 0),
            target_player_slot=(1, 0),
            source_heart_distance=1.0,
            target_heart_distance=6.0,
            source_chest_slot=None,
            target_chest_slot=None,
            source_chest_distance=None,
            target_chest_distance=None,
            chest_completed=False,
            source_life_signature="life",
            target_life_signature="life",
            life_counter_changed=False,
            dark_transition_started=False,
            life_loss_confirmed=False,
        )
        shallow = SimpleNamespace(
            analysis=source,
            target_position_visits=0,
            target_state_visits=0,
            score=100.0,
            depth=2,
        )
        deep = SimpleNamespace(
            analysis=replace(
                source,
                target_player_slot=(5, 0),
                target_heart_distance=2.0,
            ),
            target_position_visits=0,
            target_state_visits=0,
            score=1.0,
            depth=8,
        )

        selected = max(
            (shallow, deep),
            key=VerifiedNeuralAgent._human_prior_option_selection_key,
        )

        self.assertIs(selected, deep)

    def test_entity_state_hash_masks_overlapping_player_pose(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            PlayerOverlapEntityEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.UP, Action.DOWN, Action.RIGHT, Action.A),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=3,
                human_prior_option_search_beam_width=16,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
                human_prior_option_effect_probe_limit=16,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = OverlappingPlayerGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        eligible = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_entity_frontier_eligible"
            and event["eligible"]
        ]
        self.assertGreaterEqual(len(eligible), 2)
        self.assertEqual(
            len(
                {
                    signature
                    for event in eligible
                    for signature in event["entity_state_signatures"]
                }
            ),
            1,
        )
        self.assertTrue(
            all(
                event["entity_entries"][0][
                    "factual_control_feature_distance"
                ]
                > 0.08
                for event in eligible
            )
        )
        self.assertTrue(
            all(event["settling_steps"] == 2 for event in eligible)
        )
        self.assertEqual(added, 1)
        self.assertEqual(len(agent.archive), 1)

    def test_unlabeled_entity_frontier_rejects_remote_display_change(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            UnlabeledEntityTransformEnv((7, 7)),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.A),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=4,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
                human_prior_option_effect_probe_limit=4,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 0)
        self.assertEqual(len(agent.archive), 0)
        self.assertFalse(
            any(
                event["event"]
                == "human_prior_option_entity_frontier_eligible"
                for event in logger.events
            )
        )

    def test_unlabeled_entity_frontier_preserves_waiting_option_state(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            TemporalUnlabeledEntityTransformEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.A, Action.NOOP),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=4,
                human_prior_option_search_beam_width=16,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
                human_prior_option_effect_probe_limit=16,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 1)
        self.assertEqual(len(agent.archive), 1)
        self.assertEqual(
            agent.archive[0].plan.path,
            (Action.RIGHT, Action.A, Action.NOOP, Action.A),
        )
        self.assertTrue(
            any(
                event["event"]
                == "human_prior_option_entity_frontier_eligible"
                and event["eligible"]
                for event in logger.events
            )
        )

    def test_unlabeled_entity_frontier_archives_distinct_entity_states(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            MultiStateUnlabeledEntityTransformEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.A),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=3,
                human_prior_option_search_beam_width=8,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
                human_prior_option_effect_probe_limit=8,
                human_prior_option_effect_phase_offsets=1,
                human_prior_option_effect_local_controls=True,
                human_prior_option_entity_frontier=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 2)
        self.assertEqual(len(agent.archive), 2)
        self.assertEqual(
            {branch.plan.path for branch in agent.archive},
            {
                (Action.RIGHT, Action.A),
                (Action.RIGHT, Action.A, Action.A),
            },
        )
        self.assertEqual(
            len(
                {
                    branch.human_prior_option_entity_state_signature
                    for branch in agent.archive
                }
            ),
            2,
        )
        archived = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_archive_added"
        ]
        self.assertEqual(len(archived), 2)
        self.assertEqual(sum(event["selected_primary"] for event in archived), 1)
        completed = [
            event
            for event in logger.events
            if event["event"] == "human_prior_option_search_completed"
        ][-1]
        self.assertEqual(completed["archive_branches_added"], 2)
        self.assertEqual(completed["distinct_entity_contexts_archived"], 2)

    def test_goal_milestone_exhaustion_rolls_back_exact_choice(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
                human_prior_graph_stagnation_visits=1,
                human_prior_goal_exhaustion_rollback=True,
                human_prior_goal_exhaustion_minimum_steps=0,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        source_state = env.save_state()
        source_signature = agent.current_frontier_signature
        choice = (source_signature, Action.RIGHT, 1)
        agent.pending_goal_milestone_checkpoint = _LifeHazardCheckpoint(
            state=source_state,
            frame=source_frame,
            choice=choice,
            decision=0,
            frontier_signature=source_signature,
            causal_context_signature=agent.current_causal_context_signature,
            scene=agent.current_scene,
            pose_action=None,
            last_action=None,
            last_duration=None,
            action_streak=0,
            goal_heart_slots=((7, 0),),
            goal_player_slot=(0, 0),
            kind="goal_milestone",
            goal_target_heart_slots=(),
            goal_target_heart_slots_known=True,
        )
        agent.frame = env.step(Action.RIGHT, 1)
        agent.goal_prior = PositionGoalPrior()
        agent._calibrate_goal_prior = lambda _frame: None
        agent._current_human_prior_graph_signature = (
            lambda: "exhausted-goal"
        )
        agent.human_prior_graph_state_visits["exhausted-goal"] = 1
        agent._search_human_prior_options = lambda: 0

        decision = agent.decide()

        self.assertTrue(decision.restored_archive)
        self.assertEqual(decision.frame.digest, source_frame.digest)
        self.assertEqual(decision.score, 0.0)
        self.assertIsNone(agent.pending_goal_milestone_checkpoint)
        self.assertNotIn(choice, agent.temporal_option_values)
        self.assertEqual(agent.temporal_option_samples[choice], 0)
        self.assertIn(
            (((7, 0),), (), False),
            agent.human_prior_exhausted_milestone_transitions,
        )
        learned = [
            event
            for event in logger.events
            if event["event"] == "goal_milestone_exhaustion_learned"
        ]
        self.assertEqual(len(learned), 1)
        self.assertEqual(learned[0]["explored_graph_states"], 1)
        self.assertFalse(learned[0]["hazard_evidence"])
        self.assertEqual(
            learned[0]["policy_effect"], "milestone_priority_only"
        )
        self.assertTrue(learned[0]["preparation_transition_learned"])
        restored = [
            event
            for event in logger.events
            if event["event"]
            == "goal_milestone_exhaustion_state_restored"
        ]
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]["recovery_cause"], "goal_exhaustion")

    def test_goal_milestone_frontier_budget_rolls_back_without_full_exhaustion(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
                human_prior_graph_stagnation_visits=1,
                human_prior_goal_exhaustion_rollback=True,
                human_prior_goal_exhaustion_minimum_steps=0,
                human_prior_goal_exhaustion_frontier_budget=2,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        source_state = env.save_state()
        source_signature = agent.current_frontier_signature
        choice = (source_signature, Action.RIGHT, 1)
        agent.pending_goal_milestone_checkpoint = _LifeHazardCheckpoint(
            state=source_state,
            frame=source_frame,
            choice=choice,
            decision=0,
            frontier_signature=source_signature,
            causal_context_signature=agent.current_causal_context_signature,
            scene=agent.current_scene,
            pose_action=None,
            last_action=None,
            last_duration=None,
            action_streak=0,
            goal_heart_slots=((7, 0),),
            goal_player_slot=(0, 0),
            kind="goal_milestone",
            exploration_steps=1,
            goal_target_heart_slots=((3, 0),),
            goal_target_heart_slots_known=True,
        )
        agent.frame = env.step(Action.RIGHT, 1)
        agent.goal_prior = PositionGoalPrior()
        agent._calibrate_goal_prior = lambda _frame: None
        agent._current_human_prior_graph_signature = (
            lambda: "stagnant-goal"
        )
        agent.human_prior_graph_state_visits["stagnant-goal"] = 1
        agent._search_human_prior_options = lambda: self.fail(
            "frontier budget should restore before another exact search"
        )

        decision = agent.decide()

        self.assertTrue(decision.restored_archive)
        self.assertEqual(decision.frame.digest, source_frame.digest)
        exhausted = [
            event
            for event in logger.events
            if event["event"] == "goal_milestone_frontier_budget_exhausted"
        ]
        self.assertEqual(len(exhausted), 1)
        self.assertEqual(exhausted[0]["exploration_steps"], 2)
        self.assertEqual(exhausted[0]["frontier_budget"], 2)
        self.assertIn(
            (((7, 0),), ((3, 0),), False),
            agent.human_prior_exhausted_milestone_transitions,
        )

    def test_goal_milestone_exhaustion_waits_for_committed_exploration(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
                human_prior_graph_stagnation_visits=1,
                human_prior_goal_exhaustion_rollback=True,
                human_prior_goal_exhaustion_minimum_steps=2,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        source_state = env.save_state()
        source_signature = agent.current_frontier_signature
        choice = (source_signature, Action.RIGHT, 1)
        checkpoint = _LifeHazardCheckpoint(
            state=source_state,
            frame=source_frame,
            choice=choice,
            decision=0,
            frontier_signature=source_signature,
            causal_context_signature=agent.current_causal_context_signature,
            scene=agent.current_scene,
            pose_action=None,
            last_action=None,
            last_duration=None,
            action_streak=0,
            goal_heart_slots=((7, 0),),
            goal_player_slot=(0, 0),
            kind="goal_milestone",
            goal_target_heart_slots=(),
            goal_target_heart_slots_known=True,
        )
        agent.pending_goal_milestone_checkpoint = checkpoint
        agent.frame = env.step(Action.RIGHT, 1)
        agent.goal_prior = PositionGoalPrior()
        agent._calibrate_goal_prior = lambda _frame: None
        agent._current_human_prior_graph_signature = (
            lambda: "bounded-search-only"
        )
        agent.human_prior_graph_state_visits["bounded-search-only"] = 1
        agent._search_human_prior_options = lambda: 0

        decision = agent.decide()

        self.assertFalse(decision.restored_archive)
        self.assertIs(agent.pending_goal_milestone_checkpoint, checkpoint)
        # The bounded search is deferred at step one; the subsequently
        # committed new reachable state then restarts the no-progress clock.
        self.assertEqual(checkpoint.exploration_steps, 0)
        self.assertNotIn(choice, agent.temporal_option_values)
        deferred = [
            event
            for event in logger.events
            if event["event"] == "goal_milestone_exhaustion_deferred"
        ]
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["remaining_exploration_steps"], 1)
        self.assertTrue(
            any(
                event["event"]
                == "goal_milestone_exhaustion_progress_reset"
                for event in logger.events
            )
        )
        self.assertEqual(
            agent.human_prior_exhausted_milestone_transitions, set()
        )

    def test_goal_milestone_exhaustion_requires_transition_metadata(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
                human_prior_graph_stagnation_visits=1,
                human_prior_goal_exhaustion_rollback=True,
                human_prior_goal_exhaustion_minimum_steps=0,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        choice = (agent.current_frontier_signature, Action.RIGHT, 1)
        checkpoint = _LifeHazardCheckpoint(
            state=env.save_state(),
            frame=source_frame,
            choice=choice,
            decision=0,
            frontier_signature=agent.current_frontier_signature,
            causal_context_signature=agent.current_causal_context_signature,
            scene=agent.current_scene,
            pose_action=None,
            last_action=None,
            last_duration=None,
            action_streak=0,
            goal_heart_slots=((7, 0),),
            goal_player_slot=(0, 0),
            kind="goal_milestone",
            goal_target_heart_slots=(),
            goal_target_heart_slots_known=False,
        )
        agent.pending_goal_milestone_checkpoint = checkpoint

        decision = agent._restore_goal_milestone_after_exhaustion(
            "bounded-search", 4
        )

        self.assertIsNone(decision)
        self.assertIs(agent.pending_goal_milestone_checkpoint, checkpoint)
        self.assertNotIn(choice, agent.temporal_option_values)
        self.assertEqual(
            agent.human_prior_exhausted_milestone_transitions, set()
        )
        deferred = [
            event
            for event in logger.events
            if event["event"] == "goal_milestone_exhaustion_deferred"
        ]
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["reason"], "unknown_milestone_transition")
        self.assertFalse(deferred[0]["transition_metadata_known"])

    def test_goal_milestone_exhaustion_progress_resets_clock(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_graph_stagnation_visits=99,
                human_prior_goal_exhaustion_rollback=True,
                human_prior_goal_exhaustion_minimum_steps=16,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent.current_frontier_signature
        checkpoint = _LifeHazardCheckpoint(
            state=env.save_state(),
            frame=source_frame,
            choice=(source_signature, Action.RIGHT, 1),
            decision=0,
            frontier_signature=source_signature,
            causal_context_signature=agent.current_causal_context_signature,
            scene=agent.current_scene,
            pose_action=None,
            last_action=None,
            last_duration=None,
            action_streak=0,
            goal_heart_slots=((7, 0),),
            goal_player_slot=(0, 0),
            kind="goal_milestone",
            exploration_steps=5,
            goal_target_heart_slots=(),
            goal_target_heart_slots_known=True,
        )
        agent.pending_goal_milestone_checkpoint = checkpoint

        decision = agent.decide()

        self.assertEqual(decision.action, Action.RIGHT)
        self.assertIs(agent.pending_goal_milestone_checkpoint, checkpoint)
        self.assertEqual(checkpoint.exploration_steps, 0)
        resets = [
            event
            for event in logger.events
            if event["event"] == "goal_milestone_exhaustion_progress_reset"
        ]
        self.assertEqual(len(resets), 1)
        self.assertEqual(resets[0]["previous_exploration_steps"], 6)
        self.assertIn(
            resets[0]["reason"],
            {"new_goal_graph_state", "new_player_position"},
        )

    def test_archive_restored_goal_milestone_preserves_source_checkpoint(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
                human_prior_life_loss_penalty=100.0,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        env.position = 2
        parent_frame = env._frame()
        parent_state = env.save_state()
        env.position = 3
        target_frame = env._frame()
        target_state = env.save_state()
        env.position = 0
        agent.frame = source_frame
        plan = NeuralPlan((Action.RIGHT,), (1,), 25.0, 0.0)
        source_checkpoint = _LifeHazardCheckpoint(
            state=parent_state,
            frame=parent_frame,
            choice=(agent.current_frontier_signature, Action.RIGHT, 1),
            decision=0,
            frontier_signature=agent.current_frontier_signature,
            causal_context_signature=agent.current_causal_context_signature,
            scene=agent.current_scene,
            pose_action=agent.current_pose_action,
            last_action=agent.last_action,
            last_duration=agent.last_duration,
            action_streak=agent.action_streak,
            goal_heart_slots=((7, 0),),
            goal_player_slot=(2, 0),
            goal_chest_obtained=False,
            human_prior_world_context_signature=(
                agent.current_human_prior_world_context_signature
            ),
            kind="goal_milestone",
            state_id=agent._state_id(parent_state),
        )
        agent.archive = [
            _ArchivedBranch(
                state=target_state,
                frame=target_frame,
                plan=plan,
                score=25.0,
                scene=agent._scene_signature(target_frame),
                created=0,
                origin_signature=agent.current_frontier_signature,
                frontier_signature="collected-heart",
                goal_heart_slots=(),
                goal_progress_reward=25.0,
                goal_remaining_hearts=0,
                goal_total_hearts=1,
                goal_player_slot=(3, 0),
                parent_state_id="source-state",
                parent_frame_digest=parent_frame.digest,
                goal_source_signature="source-goal",
                goal_target_signature="target-goal",
                goal_milestone_checkpoint=source_checkpoint,
            )
        ]
        agent.human_prior_graph_recovery_pending = True

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        self.assertEqual(decision.frame, target_frame)
        checkpoint = agent.pending_goal_milestone_checkpoint
        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertIs(checkpoint, source_checkpoint)
        self.assertEqual(checkpoint.frame, parent_frame)
        self.assertNotEqual(checkpoint.frame, source_frame)
        self.assertEqual(env.load_state(checkpoint.state), parent_frame)
        self.assertIn(checkpoint.state, env.active_states)
        created = [
            event
            for event in logger.events
            if event["event"] == "goal_milestone_checkpoint_created"
        ]
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["checkpoint_source"], "archive_parent")
        self.assertEqual(created[0]["frame"], parent_frame.digest)

    def test_seed_goal_milestone_checkpoint_restores_persisted_metadata(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_heart_reward=1.0,
            ),
            event_logger=logger,
        )
        frame = agent.reset()
        state = env.save_state()
        metadata = {
            "choice": ["frontier", "right", 16],
            "checkpoint_decision": 7,
            "frontier_signature": "frontier",
            "causal_context_signature": "causal",
            "checkpoint_scene": agent.current_scene,
            "pose_action": "down",
            "last_action": "left",
            "last_duration": 4,
            "action_streak": 2,
            "goal_heart_slots": [[1, 2], [3, 4]],
            "goal_target_heart_slots": [],
            "goal_target_heart_slots_known": True,
            "goal_player_slot": [5, 6],
            "goal_chest_obtained": False,
            "human_prior_world_context_signature": "world",
            "checkpoint_kind": "goal_milestone",
            "exploration_steps": 11,
        }

        agent.seed_goal_milestone_checkpoint(
            state, frame, metadata, "parent", "state-7"
        )

        checkpoint = agent.pending_goal_milestone_checkpoint
        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint.choice, ("frontier", Action.RIGHT, 16))
        self.assertEqual(checkpoint.decision, 7)
        self.assertEqual(checkpoint.exploration_steps, 11)
        self.assertEqual(checkpoint.goal_heart_slots, ((1, 2), (3, 4)))
        self.assertEqual(checkpoint.goal_target_heart_slots, ())
        self.assertTrue(checkpoint.goal_target_heart_slots_known)
        created = [
            event
            for event in logger.events
            if event["event"] == "goal_milestone_checkpoint_created"
        ]
        self.assertEqual(created[-1]["checkpoint_source"], "episodic_resume")

    def test_archive_restored_goal_milestone_rejects_unknown_parent(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
                human_prior_life_loss_penalty=100.0,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        env.position = 1
        target_frame = env._frame()
        target_state = env.save_state()
        env.position = 2
        unknown_parent_frame = env._frame()
        env.position = 0
        agent.frame = source_frame
        plan = NeuralPlan((Action.RIGHT,), (1,), 25.0, 0.0)
        agent.archive = [
            _ArchivedBranch(
                state=target_state,
                frame=target_frame,
                plan=plan,
                score=25.0,
                scene=agent._scene_signature(target_frame),
                created=0,
                origin_signature=agent.current_frontier_signature,
                frontier_signature="collected-heart",
                goal_heart_slots=(),
                goal_progress_reward=25.0,
                goal_remaining_hearts=0,
                goal_total_hearts=1,
                goal_player_slot=(1, 0),
                parent_state_id="unknown-parent",
                parent_frame_digest=unknown_parent_frame.digest,
                goal_source_signature="source-goal",
                goal_target_signature="target-goal",
            )
        ]
        agent.human_prior_graph_recovery_pending = True

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        self.assertIsNone(agent.pending_goal_milestone_checkpoint)
        unavailable = [
            event
            for event in logger.events
            if event["event"] == "goal_milestone_checkpoint_unavailable"
        ]
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(unavailable[0]["parent_frame"], unknown_parent_frame.digest)

    def test_human_prior_option_effect_frontier_rejects_phase_shift(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            PhaseShiftWorldEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=1,
                human_prior_option_search_action_frames=1,
                human_prior_option_effect_stability_steps=2,
                human_prior_option_effect_probe_limit=1,
                human_prior_option_effect_phase_offsets=3,
                human_prior_option_effect_frontier=True,
                causal_spatial_columns=8,
                causal_spatial_rows=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1

        added = agent._search_human_prior_options()

        self.assertEqual(added, 0)
        phase_audits = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_option_world_effect_phase_alignment"
        ]
        self.assertEqual(len(phase_audits), 1)
        self.assertTrue(phase_audits[0]["phase_equivalent"])
        self.assertEqual(phase_audits[0]["best_patch_l1"], 0.0)
        self.assertFalse(
            any(
                event["event"]
                == "human_prior_option_world_effect_action_control"
                for event in logger.events
            )
        )
        self.assertFalse(
            any(
                event["event"]
                == "human_prior_option_effect_frontier_eligible"
                for event in logger.events
            )
        )

    def test_seed_human_prior_episodic_memory_restores_frontier_counts(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            WorldEffectEnv(True),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
            ),
            event_logger=logger,
        )
        agent.reset()
        events = [
            {
                "event": "branch_verified",
                "decision": 1,
                "action": "up",
                "action_frames": 4,
                "human_prior_graph_source_signature": "state-0",
            },
            {
                "event": "human_prior_option_branch_verified",
                "decision": 1,
                "source_graph_signature": "state-0",
                "path": ["up", "right"],
                "durations": [4, 16],
            },
            {
                "event": "decision_committed",
                "decision": 1,
                "action": "right",
                "action_frames": 1,
                "path": ["right"],
                "durations": [1],
                "human_prior_graph_source_signature": "state-0",
                "human_prior_graph_target_signature": "state-1",
                "human_prior_target_player_slot": [16, 0],
                "human_prior_known_heart_slots": [[32, 32]],
                "human_prior_target_hearts": [[32, 32]],
                "human_prior_target_life_signature": "life-1",
                "human_prior_world_target_context": (
                    "human-prior-world-root"
                ),
            },
            {
                "event": "human_prior_milestone_outcome_recorded",
                "decision": 2,
                "source_heart_slots": [[32, 32]],
                "target_heart_slots": [],
                "target_player_slot": None,
            },
            {
                "event": "decision_committed",
                "decision": 2,
                "action": "left",
                "action_frames": 1,
                "path": ["left", "right"],
                "durations": [1, 1],
                "restored_archive": True,
                "human_prior_verified_option": True,
                "human_prior_graph_source_signature": "state-1",
                "human_prior_graph_target_signature": "state-2",
                "human_prior_target_player_slot": [32, 0],
                "human_prior_target_hearts": [],
                "human_prior_world_target_context": "context-2",
            },
            {
                "event": "goal_milestone_exhaustion_learned",
                "milestone_choice": ["state-2", "left", 1],
                "learned_hazard_value": -0.5,
                "learned_hazard_samples": 2,
                "hazard_evidence": True,
            },
            {
                "event": "goal_milestone_exhaustion_learned",
                "milestone_choice": ["legacy-state", "up", 4],
                "learned_hazard_value": -2.0,
                "learned_hazard_samples": 1,
            },
        ]

        agent.seed_human_prior_episodic_memory(events)

        self.assertEqual(agent.human_prior_graph_state_visits["state-1"], 1)
        self.assertEqual(agent.human_prior_graph_state_visits["state-2"], 1)
        self.assertEqual(
            agent.human_prior_player_position_visits[(16, 0)], 1
        )
        self.assertEqual(
            agent.human_prior_player_position_visits[(32, 0)], 1
        )
        self.assertEqual(
            agent.human_prior_graph_edge_visits[
                ("state-0", Action.RIGHT, 1)
            ],
            1,
        )
        self.assertEqual(
            agent.human_prior_graph_edge_verifications[
                ("state-0", Action.UP, 4)
            ],
            1,
        )
        self.assertEqual(
            agent.human_prior_option_visits[
                (
                    "state-1",
                    ((Action.LEFT, 1), (Action.RIGHT, 1)),
                )
            ],
            1,
        )
        self.assertEqual(
            agent.human_prior_option_visits[
                (
                    "state-0",
                    ((Action.UP, 4), (Action.RIGHT, 16)),
                )
            ],
            1,
        )
        self.assertEqual(
            agent.current_human_prior_world_context_signature,
            "context-2",
        )
        self.assertEqual(agent.current_pose_action, Action.RIGHT)
        self.assertEqual(
            agent.temporal_option_values[
                ("state-2", Action.LEFT, 1)
            ],
            -0.5,
        )
        self.assertEqual(
            agent.temporal_option_samples[
                ("state-2", Action.LEFT, 1)
            ],
            2,
        )
        self.assertNotIn(
            ("legacy-state", Action.UP, 4),
            agent.temporal_option_values,
        )
        self.assertIn(
            (((32, 32),), (), (32, 0), False, False),
            agent.human_prior_milestone_outcomes,
        )
        seeded = [
            event
            for event in logger.events
            if event["event"] == "episodic_human_prior_memory_seeded"
        ]
        self.assertEqual(seeded[-1]["milestone_outcomes"], 1)
        assert agent.goal_prior is not None
        self.assertEqual(agent.goal_prior.known_slots, {(32, 32)})
        self.assertEqual(agent.goal_prior.current_present, set())
        self.assertEqual(agent.goal_prior.current_life_signature, "life-1")
        self.assertEqual(agent.goal_prior.current_player_slot, (32, 0))
        self.assertEqual(len(seeded), 1)
        self.assertEqual(seeded[0]["player_positions"], 2)
        self.assertEqual(seeded[0]["verified_option_paths"], 2)
        self.assertEqual(seeded[0]["pose_action"], Action.RIGHT)
        self.assertEqual(
            seeded[0]["unqualified_exhaustion_hazards_ignored"], 1
        )

    def test_seed_human_prior_memory_reconstructs_option_prefix_edges(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            WorldEffectEnv(True),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.DOWN,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_episodic_graph_guidance=True,
            ),
        )
        agent.reset()

        def signature(y_value: int) -> str:
            return (
                "hearts=16,64|"
                f"player=0,{y_value}|"
                "chest=none|treasure=pending|life=stable|world=room"
            )

        source = signature(0)
        first = signature(16)
        second = signature(32)
        frontier = signature(48)
        events = [
            {
                "event": "human_prior_option_branch_verified",
                "run_id": "prefix-run",
                "attempt": 1,
                "decision": 1,
                "source_graph_signature": source,
                "target_graph_signature": target,
                "path": ["down"] * depth,
                "durations": [16] * depth,
            }
            for depth, target in enumerate(
                (first, second, frontier), start=1
            )
        ]

        agent.seed_human_prior_episodic_memory(events)

        self.assertEqual(
            agent.human_prior_episodic_graph_edges[source][first], 1
        )
        self.assertEqual(
            agent.human_prior_episodic_graph_edges[first][second], 1
        )
        self.assertEqual(
            agent.human_prior_episodic_graph_edges[second][frontier], 1
        )
        self.assertEqual(
            agent.human_prior_graph_edge_verifications[
                (source, Action.DOWN, 16)
            ],
            1,
        )
        self.assertEqual(
            agent.human_prior_graph_edge_verifications[
                (first, Action.DOWN, 16)
            ],
            1,
        )
        self.assertEqual(
            agent.human_prior_graph_edge_verifications[
                (second, Action.DOWN, 16)
            ],
            1,
        )
        plan = agent._human_prior_episodic_graph_plan(source)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.kind, "control_frontier")
        self.assertEqual(plan.waypoint_signature, frontier)
        self.assertEqual(plan.source_remaining_cost, 3)
        self.assertEqual(
            agent._human_prior_episodic_graph_progress(plan, first),
            (1.0 / 3.0, False, 2),
        )
        self.assertEqual(
            agent._human_prior_episodic_graph_progress(plan, second),
            (2.0 / 3.0, False, 1),
        )

    def test_seed_memory_excludes_action_independent_option_edges(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            WorldEffectEnv(True),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.DOWN,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_episodic_graph_guidance=True,
            ),
            event_logger=logger,
        )
        agent.reset()
        source = "hearts=16,64|player=0,0|life=stable|world=room"
        autonomous = "hearts=16,64|player=0,16|life=stable|world=room"
        controlled = "hearts=16,64|player=0,32|life=stable|world=room"
        events = [
            {
                "event": "human_prior_option_branch_verified",
                "run_id": "causal-prefix-run",
                "attempt": 1,
                "decision": 1,
                "source_graph_signature": source,
                "target_graph_signature": autonomous,
                "path": ["down"],
                "durations": [16],
                "human_prior_option_action_dependent_endpoint": False,
                "human_prior_option_local_action_dependent": False,
            },
            {
                "event": "human_prior_option_branch_verified",
                "run_id": "causal-prefix-run",
                "attempt": 1,
                "decision": 1,
                "source_graph_signature": source,
                "parent_graph_signature": autonomous,
                "target_graph_signature": controlled,
                "path": ["down", "down"],
                "durations": [16, 16],
                "human_prior_option_action_dependent_endpoint": True,
                "human_prior_option_local_action_dependent": True,
            },
        ]

        agent.seed_human_prior_episodic_memory(events)

        self.assertNotIn(
            autonomous,
            agent.human_prior_episodic_graph_edges.get(source, {}),
        )
        self.assertEqual(
            agent.human_prior_episodic_graph_edges[autonomous][controlled],
            1,
        )
        self.assertEqual(
            agent.human_prior_graph_edge_verifications[
                (source, Action.DOWN, 16)
            ],
            1,
        )
        seeded = [
            event
            for event in logger.events
            if event["event"] == "episodic_human_prior_memory_seeded"
        ][-1]
        self.assertEqual(
            seeded["autonomous_option_transitions_ignored"], 2
        )

    def test_seed_option_prefix_milestone_stops_after_first_gain(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            WorldEffectEnv(True),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.DOWN,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_episodic_graph_guidance=True,
            ),
        )
        agent.reset()
        source = (
            "hearts=16,64|player=0,0|chest=none|"
            "treasure=pending|life=stable|world=room"
        )
        first = (
            "hearts=|player=0,16|chest=none|"
            "treasure=pending|life=stable|world=room"
        )
        second = first.replace("player=0,16", "player=0,32")
        events = [
            {
                "event": "human_prior_option_branch_verified",
                "run_id": "milestone-prefix-run",
                "attempt": 1,
                "decision": 1,
                "source_graph_signature": source,
                "target_graph_signature": target,
                "path": ["down"] * depth,
                "durations": [16] * depth,
                "human_prior_collected_hearts": 1,
                "human_prior_milestone_reward": 25.0,
            }
            for depth, target in enumerate((first, second), start=1)
        ]

        agent.seed_human_prior_episodic_memory(events)

        self.assertIn(
            source, agent.human_prior_episodic_milestone_sources
        )
        self.assertNotIn(
            first, agent.human_prior_episodic_milestone_sources
        )
        self.assertNotIn(
            second, agent.human_prior_episodic_milestone_sources
        )

    def test_seed_human_prior_memory_restores_navigation_grace(self) -> None:
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            WorldEffectEnv(True),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                human_prior_heart_reward=1.0,
                human_prior_navigation_reward=1.0,
                human_prior_navigation_recovery_grace=2,
            ),
            event_logger=logger,
        )
        agent.reset()
        events = [
            {
                "event": "archive_branch_restored",
                "run_id": "source-run",
                "attempt": 1,
                "decision": 7,
                "human_prior_navigation_grace_armed": True,
            },
            {
                "event": "decision_committed",
                "run_id": "source-run",
                "attempt": 1,
                "decision": 7,
                "restored_archive": True,
                "human_prior_graph_source_signature": "state-0",
                "human_prior_graph_target_signature": "state-1",
                "human_prior_target_player_slot": [16, 0],
                "human_prior_known_heart_slots": [[32, 32]],
                "human_prior_target_hearts": [[32, 32]],
                "human_prior_world_target_context": (
                    "human-prior-world-root"
                ),
            },
        ]

        agent.seed_human_prior_episodic_memory(events)

        self.assertEqual(agent.last_navigation_change_decision, 0)
        self.assertTrue(
            agent._human_prior_navigation_recovery_grace_active()
        )
        seeded = next(
            event
            for event in logger.events
            if event["event"] == "episodic_human_prior_memory_seeded"
        )
        self.assertTrue(seeded["navigation_recovery_grace_restored"])
        self.assertEqual(
            seeded["navigation_recovery_grace_elapsed_decisions"], 0
        )

    def test_seed_human_prior_memory_restores_disproved_ordering(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(human_prior_navigation_reward=1.0),
        )
        agent.reset()
        source_hearts = ((32, 32), (64, 32))
        failed_targets = ((32, 32),)
        events = [
            {
                "event": "goal_milestone_exhaustion_learned",
                "exhausted_milestone_transition": [
                    [[32, 32], [64, 32]],
                    [[64, 32]],
                    False,
                ],
            },
            {
                "event": "human_prior_option_archive_added",
                "human_prior_source_hearts": [[32, 32], [64, 32]],
                "human_prior_navigation_failed_targets": [[32, 32]],
                "human_prior_navigation_retargeted": True,
                "human_prior_navigation_ordering_reward": 1.0,
                "human_prior_chest_obtained": False,
            },
            {
                "event": "human_prior_ordering_hypothesis_disproved",
                "source_heart_slots": [[32, 32], [64, 32]],
                "failed_ordering_targets": [[32, 32]],
                "chest_obtained": False,
            },
        ]

        agent.seed_human_prior_episodic_memory(events)

        ordering_key = (source_hearts, failed_targets, False)
        self.assertIn(
            (source_hearts, ((64, 32),), False),
            agent.human_prior_exhausted_milestone_transitions,
        )
        self.assertIn(
            ordering_key,
            agent.human_prior_disproved_ordering_hypotheses,
        )
        self.assertNotIn(
            ordering_key,
            agent.human_prior_ordering_progress_hypotheses,
        )
        self.assertEqual(
            agent._human_prior_failed_ordering_targets(
                source_hearts, False
            ),
            (),
        )

    def test_seed_memory_retries_ordering_after_stronger_search_budget(
        self,
    ) -> None:
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            EnsembleVisualDynamicsModel(
                latent_size=32, action_size=8, ensemble_size=2
            ),
            "cpu",
            NeuralPlanningConfig(
                human_prior_navigation_reward=1.0,
                human_prior_option_search_depth=8,
                human_prior_option_search_beam_width=32,
                human_prior_option_search_position_reserve=8,
            ),
            event_logger=logger,
        )
        agent.reset()
        source_hearts = ((32, 32), (64, 32))
        failed_targets = ((32, 32),)
        events = [
            {
                "event": "goal_milestone_exhaustion_learned",
                "exhausted_milestone_transition": [
                    [[32, 32], [64, 32]],
                    [[64, 32]],
                    False,
                ],
            },
            {
                "event": "human_prior_option_archive_added",
                "human_prior_source_hearts": [[32, 32], [64, 32]],
                "human_prior_navigation_failed_targets": [[32, 32]],
                "human_prior_navigation_retargeted": True,
                "human_prior_navigation_ordering_reward": 1.0,
                "human_prior_chest_obtained": False,
            },
            {
                "event": "human_prior_ordering_hypothesis_disproved",
                "source_heart_slots": [[32, 32], [64, 32]],
                "failed_ordering_targets": [[32, 32]],
                "chest_obtained": False,
                "maximum_depth": 8,
                "beam_width": 32,
            },
        ]

        agent.seed_human_prior_episodic_memory(events)

        ordering_key = (source_hearts, failed_targets, False)
        self.assertNotIn(
            ordering_key,
            agent.human_prior_disproved_ordering_hypotheses,
        )
        self.assertIn(
            ordering_key,
            agent.human_prior_ordering_progress_hypotheses,
        )
        self.assertEqual(
            agent._human_prior_failed_ordering_targets(
                source_hearts, False
            ),
            failed_targets,
        )
        seeded = next(
            event
            for event in logger.events
            if event["event"] == "episodic_human_prior_memory_seeded"
        )
        self.assertEqual(
            seeded["budget_invalidated_ordering_disproofs"], 1
        )

    def test_seed_human_prior_memory_restores_exhausted_option_frontier(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(human_prior_navigation_reward=1.0),
        )
        agent.reset()
        events = [
            {
                "event": "human_prior_option_search_started",
                "run_id": "source-run",
                "decision": 3,
                "source_graph_signature": "bounded-frontier",
                "maximum_depth": 5,
            },
            {
                "event": "human_prior_option_search_completed",
                "run_id": "source-run",
                "decision": 3,
                "reason": "no_unexpanded_endpoint",
                "archive_branches_added": 0,
            },
            {
                "event": "human_prior_option_search_completed",
                "run_id": "source-run",
                "decision": 4,
                "source_graph_signature": "parent-frontier",
                "maximum_depth": 5,
                "reason": "only_exhausted_frontier_endpoints",
                "archive_branches_added": 0,
            },
        ]

        agent.seed_human_prior_episodic_memory(events)

        self.assertEqual(
            agent.human_prior_exhausted_option_frontiers,
            {"bounded-frontier": 5, "parent-frontier": 5},
        )
        events.append(
            {
                "event": "human_prior_option_search_completed",
                "run_id": "later-run",
                "decision": 1,
                "source_graph_signature": "bounded-frontier",
                "maximum_depth": 8,
                "eligible_endpoints": 1,
                "archive_branches_added": 0,
            }
        )

        agent.seed_human_prior_episodic_memory(events)

        self.assertEqual(
            agent.human_prior_exhausted_option_frontiers,
            {"parent-frontier": 5},
        )

    def test_seed_human_prior_option_archive_restores_promoted_branch(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
            ),
            event_logger=logger,
        )
        frame = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        state = env.save_state()
        metadata = {
            "path": ["right"],
            "durations": [1],
            "score": 2.5,
            "search_depth": 3,
            "source_graph_signature": "graph-source",
            "target_graph_signature": "graph-target",
            "source_behavioral_signature": "behavior-source",
            "source_causal_context_signature": "causal-source",
            "target_causal_context_signature": "causal-target",
            "target_pose_action": "right",
            "human_prior_known_heart_slots": [[7, 0]],
            "human_prior_target_hearts": [[7, 0]],
            "human_prior_remaining_hearts": 1,
            "human_prior_target_player_slot": [0, 0],
            "human_prior_goal_reward": 0.0,
            "human_prior_milestone_reward": 0.0,
            "human_prior_world_source_context": "world-source",
            "human_prior_world_target_context": "world-target",
            "human_prior_option_world_effect_signature": "effect",
            "human_prior_option_effect_frontier": True,
            "human_prior_option_effect_frontier_reason": (
                "delayed_causal_effect"
            ),
        }

        agent.seed_human_prior_option_archives(
            ((state, frame, metadata, "parent", "state-7"),)
        )

        self.assertEqual(len(agent.archive), 1)
        branch = agent.archive[0]
        self.assertIs(branch.state, state)
        self.assertEqual(branch.plan.path, (Action.RIGHT,))
        self.assertEqual(branch.plan.durations, (1,))
        self.assertEqual(branch.goal_target_signature, "graph-target")
        self.assertEqual(branch.goal_target_world_context, "world-target")
        self.assertEqual(
            branch.human_prior_option_effect_frontier_reason,
            "delayed_causal_effect",
        )
        seeded = next(
            event
            for event in logger.events
            if event["event"] == "episodic_option_archives_seeded"
        )
        self.assertEqual(seeded["seeded_archives"], 1)
        added = next(
            event
            for event in logger.events
            if event["event"] == "human_prior_option_archive_added"
        )
        self.assertEqual(added["source"], "episodic_resume")
        agent.clear_archive()
        self.assertNotIn(state, env.active_states)

        milestone_state = env.save_state()
        milestone_metadata = dict(metadata)
        milestone_metadata["human_prior_milestone_reward"] = 25.0
        agent.seed_human_prior_option_archives(
            (
                (
                    milestone_state,
                    frame,
                    milestone_metadata,
                    "parent",
                    "state-8",
                ),
            )
        )
        self.assertEqual(agent.archive, [])
        self.assertNotIn(milestone_state, env.active_states)
        skipped = next(
            event
            for event in logger.events
            if event["event"] == "episodic_option_archive_skipped"
        )
        self.assertEqual(
            skipped["reason"],
            "milestone_parent_checkpoint_not_persisted",
        )

    def test_seed_milestone_outcomes_scopes_decisions_by_run(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            WorldEffectEnv(True),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_heart_reward=1.0,
            ),
        )
        agent.reset()
        events = [
            {
                "event": "human_prior_milestone_outcome_recorded",
                "run_id": "parent",
                "decision": 2,
                "source_heart_slots": [[32, 32]],
                "target_heart_slots": [],
                "target_player_slot": None,
            },
            {
                "event": "decision_committed",
                "run_id": "parent",
                "decision": 2,
                "human_prior_target_player_slot": [32, 0],
                "human_prior_target_hearts": [],
            },
            {
                "event": "decision_committed",
                "run_id": "child",
                "decision": 2,
                "human_prior_target_player_slot": [48, 0],
                "human_prior_target_hearts": [],
            },
        ]

        agent.seed_human_prior_episodic_memory(events)

        self.assertIn(
            (((32, 32),), (), (32, 0), False, False),
            agent.human_prior_milestone_outcomes,
        )
        self.assertNotIn(
            (((32, 32),), (), (48, 0), False, False),
            agent.human_prior_milestone_outcomes,
        )

    def test_seed_human_prior_memory_anchors_strict_resume_to_pixels(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            WorldEffectEnv(True),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
            ),
            event_logger=logger,
        )
        agent.reset()
        assert agent.goal_prior is not None
        agent.goal_prior.known_slots = {(48, 48)}
        agent.goal_prior.current_present = {(48, 48)}
        agent.goal_prior.initialized = True
        agent.goal_prior.current_life_signature = "current-life"
        agent.goal_prior.current_player_slot = (8, 8)
        events = [
            {
                "event": "decision_committed",
                "human_prior_known_heart_slots": [[32, 32]],
                "human_prior_target_hearts": [],
                "human_prior_target_player_slot": [32, 0],
                "human_prior_target_life_signature": "stale-life",
                "human_prior_world_target_context": "stale-context",
            },
            {
                "event": "decision_committed",
                "action": "right",
                "action_frames": 1,
            },
        ]

        agent.seed_human_prior_episodic_memory(events)

        self.assertEqual(
            agent.goal_prior.current_present, {(48, 48)}
        )
        self.assertEqual(
            agent.goal_prior.current_life_signature, "current-life"
        )
        self.assertEqual(agent.goal_prior.current_player_slot, (8, 8))
        self.assertEqual(
            agent.current_human_prior_world_context_signature,
            "human-prior-world-root",
        )
        seeded = [
            event
            for event in logger.events
            if event["event"] == "episodic_human_prior_memory_seeded"
        ][0]
        self.assertEqual(seeded["current_state_source"], "resume_frame")

    def test_seed_human_prior_memory_respects_novel_room_boundary(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            WorldEffectEnv(True),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
            ),
            event_logger=logger,
        )
        agent.reset()
        events = [
            {
                "event": "decision_committed",
                "action": "right",
                "action_frames": 1,
                "human_prior_graph_source_signature": "room-2-source",
                "human_prior_graph_target_signature": "room-2-target",
                "human_prior_target_player_slot": [176, 32],
                "human_prior_known_heart_slots": [[176, 48]],
                "human_prior_target_hearts": [],
                "human_prior_chest_obtained": True,
            },
            {
                "event": "pixel_novel_room_started",
                "discovered_heart_slots": [[96, 128], [128, 64]],
            },
            {
                "event": "decision_committed",
                "action": "left",
                "action_frames": 1,
                "human_prior_graph_source_signature": "room-3-source",
                "human_prior_graph_target_signature": "room-3-target",
                "human_prior_target_player_slot": [112, 160],
                "human_prior_known_heart_slots": [[96, 128], [128, 64]],
                "human_prior_target_hearts": [[96, 128], [128, 64]],
                "human_prior_chest_obtained": False,
            },
        ]

        agent.seed_human_prior_episodic_memory(events)

        self.assertEqual(
            agent.human_prior_graph_state_visits,
            {"room-3-target": 1},
        )
        self.assertEqual(
            agent.human_prior_player_position_visits,
            {(112, 160): 1},
        )
        self.assertEqual(
            agent.human_prior_graph_edge_visits,
            {("room-3-source", Action.LEFT, 1): 1},
        )
        assert agent.goal_prior is not None
        self.assertEqual(
            agent.goal_prior.known_slots,
            {(96, 128), (128, 64)},
        )
        self.assertFalse(agent.goal_prior.chest_obtained)
        seeded = [
            event
            for event in logger.events
            if event["event"] == "episodic_human_prior_memory_seeded"
        ][0]
        self.assertEqual(seeded["room_boundaries"], 1)

    def test_human_prior_restore_prefers_unvisited_player_position(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                action_frames=1,
                human_prior_heart_reward=1.0,
                human_prior_best_first_archive=True,
                visual_stagnation_visits=99,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_signature = agent._current_human_prior_graph_signature()
        root = env.save_state()
        seen_frame = env.step(Action.RIGHT, 1)
        seen_state = env.save_state()
        novel_frame = env.step(Action.RIGHT, 1)
        novel_state = env.save_state()
        env.load_state(root)
        agent.frame = source_frame
        agent.human_prior_player_position_visits[(0, 0)] = 1
        agent.human_prior_player_position_visits[(1, 0)] = 1
        seen = _ArchivedBranch(
            state=seen_state,
            frame=seen_frame,
            plan=NeuralPlan((Action.RIGHT,), (1,), 100.0, 0.0),
            score=100.0,
            scene=agent._scene_signature(seen_frame),
            created=1,
            goal_heart_slots=((7, 0),),
            goal_remaining_hearts=1,
            goal_total_hearts=1,
            goal_player_slot=(1, 0),
            goal_source_signature=source_signature,
            goal_target_signature="seen-target",
        )
        novel = _ArchivedBranch(
            state=novel_state,
            frame=novel_frame,
            plan=NeuralPlan(
                (Action.RIGHT, Action.RIGHT), (1, 1), 1.0, 0.0
            ),
            score=1.0,
            scene=agent._scene_signature(novel_frame),
            created=1,
            goal_heart_slots=((7, 0),),
            goal_remaining_hearts=1,
            goal_total_hearts=1,
            goal_player_slot=(2, 0),
            goal_source_signature=source_signature,
            goal_target_signature="novel-target",
            human_prior_verified_option=True,
        )
        agent.archive = [seen, novel]
        agent._archive_frontier_score = lambda branch: branch.score
        agent.human_prior_graph_recovery_pending = True
        self.assertEqual(
            agent._human_prior_unvisited_archive_endpoints(), 1
        )
        self.assertEqual(
            agent._human_prior_unvisited_archive_endpoints(
                "unrelated-source"
            ),
            0,
        )

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.planned_path, (Action.RIGHT, Action.RIGHT))
        self.assertEqual(env.position, 2)
        filtered = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_best_first_archives_filtered"
        ][-1]
        self.assertTrue(filtered["physical_frontier_preferred"])
        self.assertEqual(filtered["unvisited_player_positions"], 1)

    def test_human_prior_restore_rejects_only_regressive_archive(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        agent.goal_prior.best_remaining_hearts = 1
        state = env.save_state()
        agent.archive = [
            _ArchivedBranch(
                state=state,
                frame=source_frame,
                plan=NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                score=1.0,
                scene=agent._scene_signature(source_frame),
                created=0,
                goal_heart_slots=((1, 0), (7, 0)),
                goal_remaining_hearts=2,
                goal_total_hearts=2,
                goal_player_slot=(0, 0),
                goal_source_signature="source",
                goal_target_signature="target",
            )
        ]
        agent.human_prior_graph_recovery_pending = True

        restored = agent._restore_if_stagnant()

        self.assertIsNone(restored)
        filtered = [
            event
            for event in logger.events
            if event["event"] == "human_prior_regressive_archives_filtered"
        ]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["alternatives_remaining"], 0)

    def test_failed_ordering_preserves_pre_milestone_archive(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_hearts = ((7, 0), (9, 0))
        agent.goal_prior.known_slots = set(source_hearts)
        agent.goal_prior.current_present = set(source_hearts)
        agent.goal_prior.best_remaining_hearts = 1
        agent.human_prior_exhausted_milestone_transitions.add(
            (source_hearts, ((7, 0),), False)
        )
        root = env.save_state()
        target_frame = env.step(Action.RIGHT, 1)
        target_state = env.save_state()
        env.load_state(root)
        agent.frame = source_frame
        source_signature = agent._current_human_prior_graph_signature()
        agent.archive = [
            _ArchivedBranch(
                state=target_state,
                frame=target_frame,
                plan=NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                score=1.0,
                scene=agent._scene_signature(target_frame),
                created=0,
                goal_heart_slots=source_hearts,
                goal_progress_reward=1.0,
                goal_remaining_hearts=2,
                goal_total_hearts=2,
                goal_player_slot=(1, 0),
                goal_source_signature=source_signature,
                goal_target_signature="preparation-target",
                human_prior_verified_option=True,
            )
        ]
        agent.human_prior_graph_recovery_pending = True

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.planned_path, (Action.RIGHT,))
        self.assertEqual(env.position, 1)
        preserved = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_preparation_archives_preserved"
        ]
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0]["preserved_branches"], 1)
        self.assertFalse(preserved[0]["hazard_evidence"])

    def test_failed_ordering_filters_transition_and_precursor_archives(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
            ),
            event_logger=logger,
        )
        source_frame = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        source_hearts = ((1, 0), (7, 0))
        agent.goal_prior.known_slots = set(source_hearts)
        agent.goal_prior.current_present = set(source_hearts)
        agent.goal_prior.best_remaining_hearts = 2
        agent.human_prior_exhausted_milestone_transitions.add(
            (source_hearts, ((7, 0),), False)
        )
        root = env.save_state()
        exact_frame = env.step(Action.RIGHT, 1)
        exact_state = env.save_state()
        precursor_frame = env.step(Action.RIGHT, 1)
        precursor_state = env.save_state()
        env.load_state(root)
        agent.frame = source_frame
        source_signature = agent._current_human_prior_graph_signature()
        agent.archive = [
            _ArchivedBranch(
                state=exact_state,
                frame=exact_frame,
                plan=NeuralPlan((Action.RIGHT,), (1,), 25.0, 0.0),
                score=25.0,
                scene=agent._scene_signature(exact_frame),
                created=0,
                goal_heart_slots=((7, 0),),
                goal_progress_reward=25.0,
                goal_remaining_hearts=1,
                goal_total_hearts=2,
                goal_player_slot=(1, 0),
                goal_source_signature=source_signature,
                goal_target_signature="exhausted-target",
            ),
            _ArchivedBranch(
                state=precursor_state,
                frame=precursor_frame,
                plan=NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                score=1.0,
                scene=agent._scene_signature(precursor_frame),
                created=0,
                goal_heart_slots=source_hearts,
                goal_progress_reward=1.0,
                goal_remaining_hearts=2,
                goal_total_hearts=2,
                goal_player_slot=(1, 0),
                goal_source_signature=source_signature,
                goal_target_signature="precursor-target",
            ),
        ]
        agent.human_prior_graph_recovery_pending = True

        restored = agent._restore_if_stagnant()

        self.assertIsNone(restored)
        self.assertEqual(agent.archive, [])
        filtered = [
            event
            for event in logger.events
            if event["event"]
            == "human_prior_exhausted_milestone_archives_filtered"
        ]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["filtered_branches"], 2)
        self.assertEqual(filtered[0]["exhausted_transition_branches"], 1)
        self.assertEqual(filtered[0]["exhausted_precursor_branches"], 1)
        self.assertFalse(filtered[0]["hazard_evidence"])

    def test_regressive_archive_does_not_defer_option_search(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = ActionEffectEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                human_prior_best_first_archive=True,
                human_prior_graph_stagnation_visits=1,
                human_prior_option_search_depth=2,
                human_prior_option_search_beam_width=2,
                human_prior_option_search_action_frames=1,
            ),
            event_logger=logger,
        )
        source = agent.reset()
        agent.goal_prior = PositionGoalPrior()
        agent._calibrate_goal_prior = lambda frame: None
        agent.goal_prior.best_remaining_hearts = 1
        source_signature = agent._current_human_prior_graph_signature()
        agent.human_prior_graph_state_visits[source_signature] = 1
        agent.human_prior_player_position_visits[(0, 0)] = 1
        agent.archive = [
            _ArchivedBranch(
                state=env.save_state(),
                frame=source,
                plan=NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                score=1.0,
                scene=agent._scene_signature(source),
                created=0,
                goal_heart_slots=((1, 0), (7, 0)),
                goal_remaining_hearts=2,
                goal_total_hearts=2,
                goal_player_slot=(0, 0),
                goal_source_signature=source_signature,
                goal_target_signature="target",
            )
        ]

        agent.decide()

        self.assertFalse(
            any(
                event["event"] == "human_prior_option_search_deferred"
                for event in logger.events
            )
        )
        self.assertTrue(
            any(
                event["event"] == "human_prior_option_search_started"
                for event in logger.events
            )
        )

    def test_human_prior_position_novelty_can_be_phase_conditioned(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(human_prior_phase_position_novelty=True),
        )
        player = (176, 96)
        pending = (
            "hearts=|player=176,96|chest=none|treasure=pending|"
            "life=life|world=human-prior-world-root"
        )
        obtained = pending.replace("treasure=pending", "treasure=obtained")

        agent._record_human_prior_player_position(pending, player)

        self.assertEqual(agent._human_prior_position_visits(pending, player), 1)
        self.assertEqual(agent._human_prior_position_visits(obtained, player), 0)
        self.assertEqual(agent.human_prior_player_position_visits[player], 1)

    def test_matched_effect_arms_archived_option_follow_through(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(action_equivalence_threshold=0.01),
        )

        eligible, contrast, counterfactuals = (
            agent._include_matched_effect_option_evidence(
                Action.A,
                {"contrast": 0.025},
                False,
                0.0,
                0,
            )
        )

        self.assertTrue(eligible)
        self.assertEqual(contrast, 0.025)
        self.assertEqual(counterfactuals, 1)

    def test_human_prior_world_effect_masks_player_motion(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                causal_spatial_columns=2,
                causal_spatial_rows=2,
            ),
        )
        frame = Frame(32, 32, 3, bytes(32 * 32 * 3))
        analysis = HeartGoalAnalysis(
            reliable=True,
            known_slots=(),
            source_present=(),
            target_present=(),
            collected=(),
            target_similarities=(),
            heart_reward=0.0,
            all_hearts_reward=0.0,
            chest_reward=0.0,
            navigation_reward=0.0,
            life_loss_penalty=0.0,
            total_reward=0.0,
            global_visual_change=0.0,
            target_intensity=0.0,
            source_player_slot=(0, 0),
            target_player_slot=(16, 0),
            source_heart_distance=None,
            target_heart_distance=None,
            source_chest_slot=None,
            target_chest_slot=None,
            source_chest_distance=None,
            target_chest_distance=None,
            chest_completed=False,
            source_life_signature=None,
            target_life_signature=None,
            life_counter_changed=False,
            dark_transition_started=False,
            life_loss_confirmed=False,
        )

        player_only = bytes((1, 1, 0, 0)).hex()
        with_world_change = bytes((1, 1, 1, 0)).hex()
        self.assertEqual(
            agent._human_prior_world_effect_signature(
                player_only, analysis, frame
            ),
            "",
        )
        self.assertEqual(
            agent._human_prior_world_effect_signature(
                with_world_change, analysis, frame
            ),
            bytes((0, 0, 1, 0)).hex(),
        )
        self.assertEqual(
            agent._human_prior_world_effect_signature(
                with_world_change,
                analysis,
                frame,
                Action.RIGHT,
            ),
            "",
        )
        self.assertEqual(
            agent._human_prior_world_effect_signature(
                with_world_change,
                analysis,
                frame,
                Action.RIGHT,
                allow_nonlocal=True,
            ),
            bytes((0, 0, 1, 0)).hex(),
        )
        stationary = replace(
            analysis,
            target_player_slot=analysis.source_player_slot,
        )
        blocked_pose_spill = bytes((1, 1, 0, 0)).hex()
        self.assertEqual(
            agent._human_prior_world_effect_signature(
                blocked_pose_spill,
                stationary,
                frame,
                Action.RIGHT,
            ),
            "",
        )
        self.assertEqual(
            agent._human_prior_world_effect_signature(
                blocked_pose_spill,
                stationary,
                frame,
                Action.RIGHT,
                allow_nonlocal=True,
            ),
            bytes((0, 1, 0, 0)).hex(),
        )
        target_context = agent._next_human_prior_world_context(
            "human-prior-world-root",
            bytes((0, 0, 1, 0)).hex(),
        )
        source_signature, target_signature = (
            agent._human_prior_graph_signatures(
                analysis,
                "human-prior-world-root",
                target_context,
            )
        )
        self.assertNotEqual(source_signature, target_signature)
        self.assertIn("world=human-prior-world-root", source_signature)
        self.assertIn(f"world={target_context}", target_signature)
        self.assertEqual(
            agent._next_human_prior_world_context(
                target_context,
                bytes((0, 0, 1, 0)).hex(),
            ),
            "human-prior-world-root",
        )

        # A real native trace produced two visually identical room states
        # whose only differing pixels occupied the player sprite bounding box.
        # The snapped player anchors disagreed by one tile, leaving sprite
        # spill two coarse Manhattan cells away from either anchor.  Keep that
        # spill out of the learned world context while retaining a genuinely
        # remote cell.
        native_frame = Frame(256, 240, 3, bytes(256 * 240 * 3))
        native_analysis = replace(
            analysis,
            source_player_slot=(128, 48),
            target_player_slot=(112, 48),
        )
        native_effect = bytearray(16 * 15)
        native_effect[2 * 16 + 6] = 1
        native_effect[6 * 16 + 7] = 1
        native_agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                causal_spatial_columns=16,
                causal_spatial_rows=15,
            ),
        )
        self.assertEqual(
            native_agent._human_prior_nonlocal_world_effect_cells(
                bytes(native_effect).hex(),
                native_analysis,
                native_frame,
                extra_player_slots=((128, 32),),
            ),
            {(7, 6)},
        )

    def test_persistent_change_filters_regressive_archives_when_alternatives_exist(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
                causal_spatial_columns=2,
                causal_spatial_rows=2,
                persistent_change_stability_decisions=2,
                persistent_change_minimum_value_drop=4,
            ),
        )
        baseline = Frame(8, 8, 1, bytes([255]) * 64)
        agent.reset(baseline)
        changed_pixels = bytearray([255] * 64)
        for row in range(4):
            changed_pixels[row * 8 : row * 8 + 4] = bytes(4)
        changed = Frame(8, 8, 1, bytes(changed_pixels))
        changed_variant_pixels = bytearray(changed_pixels)
        changed_variant_pixels[-1] = 254
        changed_variant = Frame(8, 8, 1, bytes(changed_variant_pixels))

        agent._observe_persistent_changes(baseline)
        agent._observe_persistent_changes(baseline)
        agent._observe_persistent_changes(changed)
        agent._observe_persistent_changes(changed)
        self.assertEqual(agent.persistent_change_cells, {0: 0})
        self.assertTrue(agent._matches_persistent_changes(changed_variant))
        self.assertFalse(agent._matches_persistent_changes(baseline))

        agent.frame = changed_variant
        scene = agent._scene_signature(changed_variant)
        plan = NeuralPlan((Action.RIGHT,), (1,), 0.0, 0.0)
        agent.archive = [
            _ArchivedBranch(
                env.save_state(),
                baseline,
                plan,
                100.0,
                scene,
                1,
                causal_spatial_signature="regression",
                causal_context_signature="causal-context-root",
            ),
            _ArchivedBranch(
                env.save_state(),
                changed,
                plan,
                0.0,
                scene,
                2,
                causal_spatial_signature="preserved",
                causal_context_signature="causal-context-root",
            ),
        ]
        agent.visual_stagnation_streak = 1

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, changed.digest)

        overlay_pixels = bytearray([255] * 64)
        for row in range(4):
            overlay_pixels[row * 8 : row * 8 + 4] = bytes([128]) * 4
        overlay = Frame(8, 8, 1, bytes(overlay_pixels))
        agent._observe_persistent_changes(overlay)
        agent._observe_persistent_changes(overlay)
        self.assertEqual(agent.persistent_change_cells, {0: 0})

        agent._observe_persistent_changes(baseline)
        agent._observe_persistent_changes(baseline)
        self.assertEqual(agent.persistent_change_cells, {})

    def test_speculative_persistent_change_preserves_candidate_on_restore(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
                causal_spatial_columns=2,
                causal_spatial_rows=2,
                persistent_change_stability_decisions=3,
                persistent_change_minimum_value_drop=4,
                persistent_change_speculative_recovery=True,
            ),
            event_logger=logger,
        )
        baseline = Frame(8, 8, 1, bytes([255]) * 64)
        agent.reset(baseline)
        changed_pixels = bytearray([255] * 64)
        for row in range(4):
            changed_pixels[row * 8 : row * 8 + 4] = bytes(4)
        changed = Frame(8, 8, 1, bytes(changed_pixels))
        current_pixels = bytearray(changed_pixels)
        current_pixels[-1] = 254
        current = Frame(8, 8, 1, bytes(current_pixels))

        agent._observe_persistent_changes(changed)
        self.assertEqual(agent.persistent_change_cells, {})
        self.assertEqual(agent.persistent_change_candidates, {0: (0, 1)})
        scene = agent._scene_signature(current)
        plan = NeuralPlan((Action.RIGHT,), (1,), 0.0, 0.0)
        agent.frame = current
        agent.archive = [
            _ArchivedBranch(
                env.save_state(),
                baseline,
                plan,
                100.0,
                scene,
                1,
                causal_spatial_signature="01",
                causal_context_signature="causal-context-root",
            ),
            _ArchivedBranch(
                env.save_state(),
                changed,
                plan,
                0.0,
                scene,
                2,
                causal_spatial_signature="02",
                causal_context_signature="causal-context-root",
            ),
        ]
        agent.visual_stagnation_streak = 1

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, changed.digest)
        self.assertEqual(agent.persistent_change_candidates, {0: (0, 1)})
        committed = [
            event
            for event in logger.events
            if event["event"] == "decision_committed"
        ][-1]
        self.assertTrue(committed["speculative_persistence_applied"])

    def test_learned_hazard_is_verified_but_not_committed_when_safe(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=9, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.SELECT),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                action_frames=4,
                actual_novelty_weight=0.0,
                scene_novelty_weight=0.0,
                prediction_error_weight=0.0,
                actual_change_weight=0.0,
                action_effect_weight=10.0,
                action_coverage_weight=0.0,
                duration_coverage_weight=0.0,
                consecutive_repeat_weight=0.0,
            ),
        )
        agent.reset()
        agent._record_temporal_option_sample(
            ("prior-state", Action.SELECT, 1),
            -2.0,
            generalize_action_hazard=True,
        )
        plans = [
            NeuralPlan((action,), (4,), 0.0, 0.0)
            for action in (Action.NOOP, Action.SELECT)
        ]
        agent.planner.plan = lambda _frame: plans

        decision = agent.decide()

        self.assertEqual(decision.branches_examined, 2)
        self.assertEqual(decision.action, Action.NOOP)
        self.assertFalse(any(branch.plan.path[0] == Action.SELECT for branch in agent.archive))

    def test_same_decision_archive_pruning_releases_each_state_once(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=9, ensemble_size=2)
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.RIGHT, Action.SELECT),
                planning_depth=1,
                beam_width=3,
                verify_actions=3,
                action_frames=4,
                archive_capacity=1,
            ),
        )
        agent.reset()
        plans = [
            NeuralPlan((action,), (4,), 0.0, 0.0)
            for action in (Action.NOOP, Action.RIGHT, Action.SELECT)
        ]
        agent.planner.plan = lambda _frame: plans

        decision = agent.decide()

        self.assertEqual(decision.branches_examined, 3)
        self.assertLessEqual(len(agent.archive), 1)

    def test_stagnation_restores_an_alternative_branch(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                visual_stagnation_visits=1,
            ),
        )
        agent.reset()
        first = agent.decide()
        agent.visual_stagnation_streak = 1
        second = agent.decide()
        self.assertFalse(first.restored_archive)
        self.assertTrue(second.restored_archive)

    def test_temporal_observation_window_precedes_stagnation_recovery(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
                autonomous_grace_decisions=2,
            ),
        )
        frame = agent.reset()
        branch_state = env.save_state()
        agent.archive = [
            _ArchivedBranch(
                branch_state,
                frame,
                NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                1.0,
                "other-scene",
                0,
            )
        ]
        agent.visual_stagnation_streak = 1
        agent.active_temporal_option = _TemporalOptionTrace(
            choice=("source", Action.RIGHT, 1),
            initiation_decision=1,
            start_decision=1,
            entry_signature="source",
            entry_scene="scene",
            passive_decisions=3,
        )

        suppressed = agent._restore_if_stagnant()

        self.assertIsNone(suppressed)
        self.assertEqual(len(agent.archive), 1)

        agent.active_temporal_option.passive_decisions = 4
        agent.autonomous_grace_remaining = 1
        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertTrue(restored.restored_archive)

    def test_passive_transition_cannot_create_persistent_progress(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP,),
                planning_depth=1,
                causal_spatial_columns=2,
                causal_spatial_rows=2,
                persistent_change_stability_decisions=2,
                persistent_change_minimum_value_drop=4,
            ),
        )
        baseline = Frame(8, 8, 1, bytes([255]) * 64)
        agent.reset(baseline)
        changed_pixels = bytearray([255] * 64)
        for row in range(4):
            changed_pixels[row * 8 : row * 8 + 4] = bytes(4)
        changed = Frame(8, 8, 1, bytes(changed_pixels))

        agent._observe_persistent_changes(changed, action_dependent=False)
        agent._observe_persistent_changes(changed, action_dependent=False)

        self.assertEqual(agent.persistent_change_candidates, {})
        self.assertEqual(agent.persistent_change_cells, {})

    def test_autonomous_grace_reserves_an_intervention_before_recovery(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
            ),
        )
        frame = agent.reset()
        state = env.save_state()
        agent.archive = [
            _ArchivedBranch(
                state,
                frame,
                NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                1.0,
                "other-scene",
                0,
            )
        ]
        agent.visual_stagnation_streak = 1
        agent.autonomous_grace_remaining = 1

        self.assertIsNone(agent._restore_if_stagnant())

        agent.autonomous_grace_remaining = 0
        agent.autonomous_intervention_pending = True
        self.assertIsNone(agent._restore_if_stagnant())

        agent.autonomous_intervention_pending = False
        restored = agent._restore_if_stagnant()
        self.assertIsNotNone(restored)

    def test_reserved_autonomous_intervention_forces_a_non_noop_probe(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=4,
        )
        agent = VerifiedNeuralAgent(
            AutonomousAnimationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.A, Action.NOOP),
                action_durations=(1, 4),
                planning_depth=1,
                beam_width=4,
                verify_actions=4,
            ),
        )
        agent.reset()
        agent.autonomous_intervention_pending = True

        decision = agent.decide()

        self.assertEqual(decision.action, Action.A)
        self.assertFalse(agent.autonomous_intervention_pending)

    def test_active_autonomous_grace_is_not_reset_by_fresh_detection(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=4,
        )
        agent = VerifiedNeuralAgent(
            AutonomousAnimationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.A, Action.NOOP),
                action_durations=(1, 4),
                planning_depth=1,
                beam_width=4,
                verify_actions=4,
                autonomous_grace_decisions=4,
            ),
        )
        agent.reset()
        agent.autonomous_grace_remaining = 1

        decision = agent.decide()

        self.assertEqual(decision.action, Action.NOOP)
        self.assertEqual(agent.autonomous_grace_remaining, 0)
        self.assertTrue(agent.autonomous_intervention_pending)

    def test_autonomous_intervention_prefers_action_dependent_control(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.A, Action.RIGHT, Action.NOOP),
                planning_depth=1,
                beam_width=3,
                verify_actions=3,
            ),
        )
        agent.reset()
        agent.autonomous_intervention_pending = True

        decision = agent.decide()

        self.assertEqual(decision.action, Action.RIGHT)

    def test_dynamic_control_selects_a_future_viable_escape(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            DynamicActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.SELECT, Action.NOOP),
                planning_depth=1,
                beam_width=3,
                verify_actions=3,
                action_frames=1,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.planner.plan = lambda _frame: [
            NeuralPlan((Action.RIGHT,), (1,), 100.0, 0.0),
            NeuralPlan((Action.SELECT,), (1,), 0.0, 0.0),
            NeuralPlan((Action.NOOP,), (1,), -100.0, 0.0),
        ]
        agent.pending_option_choice = ("source", Action.RIGHT, 1)
        agent.pending_option_decision = 0
        agent.pending_option_causal_evidence = True

        decision = agent.decide()

        self.assertEqual(decision.action, Action.SELECT)
        selected = [
            event
            for event in logger.events
            if event["event"] == "dynamic_control_selected"
        ]
        self.assertEqual(len(selected), 1)
        probes = [
            event
            for event in logger.events
            if event["event"] == "counterfactual_control_probe"
        ]
        self.assertEqual(len(probes), 1)
        self.assertTrue(probes[0]["control_collapsed"])
        escape = next(
            event
            for event in logger.events
            if event["event"] == "counterfactual_control_escape_probe"
        )
        self.assertEqual(escape["viable_alternatives"], 1)
        self.assertEqual(escape["selected_action"], Action.SELECT)

    def test_causal_observation_matches_a_short_initiating_press(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.NOOP),
                action_durations=(1, 8),
                planning_depth=1,
                beam_width=4,
                verify_actions=3,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.planner.plan = lambda _frame: [
            NeuralPlan((Action.RIGHT,), (8,), 3.0, 0.0),
            NeuralPlan((Action.NOOP,), (8,), 2.0, 0.0),
            NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
            NeuralPlan((Action.NOOP,), (1,), 0.0, 0.0),
        ]
        agent.pending_option_choice = ("source", Action.RIGHT, 1)
        agent.pending_option_decision = 0
        agent.pending_option_causal_evidence = True

        decision = agent.decide()

        self.assertEqual(decision.action, Action.NOOP)
        self.assertEqual(decision.action_frames, 1)
        wait = next(
            event
            for event in logger.events
            if event["event"] == "causal_observation_wait"
        )
        self.assertEqual(wait["initiation_duration"], 1)
        self.assertTrue(wait["duration_matched"])
        probes = next(
            event["probes"]
            for event in logger.events
            if event["event"] == "behavior_probe_selected"
        )
        matched = [
            probe for probe in probes if probe["matched_causal_observation"]
        ]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["action"], Action.NOOP)
        self.assertEqual(matched[0]["action_frames"], 1)

    def test_causal_observation_gets_an_intervention_before_recovery(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.NOOP),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                action_frames=1,
                visual_stagnation_visits=1,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.planner.plan = lambda _frame: [
            NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
            NeuralPlan((Action.NOOP,), (1,), 0.0, 0.0),
        ]
        agent.pending_option_choice = ("source", Action.RIGHT, 1)
        agent.pending_option_decision = 0
        agent.pending_option_causal_evidence = True

        observation = agent.decide()
        agent.delayed_return_recovery = True
        agent.delayed_return_loop_start = 0
        agent.visual_stagnation_streak = 1
        intervention = agent.decide()

        self.assertEqual(observation.action, Action.NOOP)
        self.assertEqual(intervention.action, Action.RIGHT)
        self.assertFalse(intervention.restored_archive)
        self.assertFalse(agent.causal_observation_intervention_pending)
        self.assertFalse(agent.delayed_return_recovery)
        self.assertTrue(
            any(
                event["event"]
                == "causal_observation_recovery_suppressed"
                for event in logger.events
            )
        )
        selected = next(
            event
            for event in logger.events
            if event["event"]
            == "causal_observation_intervention_selected"
        )
        self.assertEqual(selected["action"], Action.RIGHT)

    def test_control_collapse_restores_the_causal_checkpoint(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        env = DynamicActionEffectEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.NOOP),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                action_frames=1,
            ),
            event_logger=logger,
        )
        root_frame = agent.reset()
        agent.planner.plan = lambda _frame: [
            NeuralPlan((Action.RIGHT,), (1,), 100.0, 0.0),
            NeuralPlan((Action.NOOP,), (1,), -100.0, 0.0),
        ]
        choice = ("source", Action.RIGHT, 1)
        agent.pending_option_choice = choice
        agent.pending_option_decision = 0
        agent.pending_option_causal_evidence = True
        agent.pending_option_recovery_checkpoint = _LifeHazardCheckpoint(
            state=env.save_state(),
            frame=root_frame,
            choice=choice,
            decision=0,
            frontier_signature="source",
            causal_context_signature="causal-context-root",
            scene=agent.current_scene,
            pose_action=None,
            last_action=None,
            last_duration=None,
            action_streak=0,
            goal_heart_slots=(),
            goal_player_slot=None,
        )

        first = agent.decide()
        agent.archive.extend(
            [
                _ArchivedBranch(
                    env.save_state(),
                    first.frame,
                    NeuralPlan((Action.SELECT,), (1,), 0.0, 0.0),
                    0.0,
                    "sibling",
                    0,
                ),
                _ArchivedBranch(
                    env.save_state(),
                    first.frame,
                    NeuralPlan((Action.SELECT,), (1,), 0.0, 0.0),
                    0.0,
                    "descendant",
                    1,
                ),
            ]
        )
        restored = agent.decide()

        self.assertEqual(first.action, Action.NOOP)
        self.assertTrue(restored.restored_archive)
        self.assertEqual(restored.frame.digest, root_frame.digest)
        self.assertLess(agent.temporal_option_values[choice], 0.0)
        self.assertTrue(agent.archive)
        self.assertTrue(all(branch.created <= 0 for branch in agent.archive))
        learned = next(
            event
            for event in logger.events
            if event["event"]
            == "counterfactual_control_collapse_learned"
        )
        self.assertTrue(learned["recovery_checkpoint_available"])
        restored_event = next(
            event
            for event in logger.events
            if event["event"] == "control_collapse_state_restored"
        )
        self.assertEqual(restored_event["recovery_cause"], "control_collapse")
        removed = [
            event
            for event in logger.events
            if event["event"] == "archive_branch_removed"
            and event["reason"] == "control_collapse_rollback_descendant"
        ]
        self.assertTrue(removed)

    def test_temporary_control_pause_is_not_learned_as_a_collapse(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        env = TemporaryControlPauseEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.NOOP),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                action_frames=1,
                control_collapse_confirmation_steps=4,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.planner.plan = lambda _frame: [
            NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
            NeuralPlan((Action.NOOP,), (1,), 0.0, 0.0),
        ]
        agent.pending_option_choice = ("source", Action.RIGHT, 1)
        agent.pending_option_decision = 0
        agent.pending_option_causal_evidence = True

        decision = agent.decide()

        self.assertEqual(decision.action, Action.NOOP)
        confirmations = [
            event
            for event in logger.events
            if event["event"] == "counterfactual_control_confirmation"
        ]
        self.assertEqual(len(confirmations), 1)
        self.assertFalse(confirmations[0]["control_collapsed"])
        self.assertTrue(confirmations[0]["control_returned"])
        self.assertEqual(confirmations[0]["control_returned_step"], 2)
        self.assertFalse(
            any(
                event["event"]
                == "counterfactual_control_collapse_learned"
                for event in logger.events
            )
        )

    def test_novel_scene_after_darkness_is_not_a_control_collapse(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            NovelSceneTransitionEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.NOOP),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                action_frames=1,
                control_collapse_confirmation_steps=4,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.planner.plan = lambda _frame: [
            NeuralPlan((Action.RIGHT,), (1,), 10.0, 0.0),
            NeuralPlan((Action.NOOP,), (1,), 0.0, 0.0),
        ]

        trigger = agent.decide()
        observation = agent.decide()

        self.assertEqual(trigger.action, Action.RIGHT)
        self.assertEqual(observation.action, Action.NOOP)
        confirmation = next(
            event
            for event in logger.events
            if event["event"] == "counterfactual_control_confirmation"
        )
        self.assertFalse(confirmation["control_collapsed"])
        self.assertTrue(confirmation["dark_encountered"])
        self.assertTrue(confirmation["novel_scene_observed"])
        self.assertFalse(confirmation["returned_to_known_scene"])
        self.assertFalse(
            any(
                event["event"]
                == "counterfactual_control_collapse_learned"
                for event in logger.events
            )
        )

    def test_delayed_transition_probe_selects_and_observes_novel_scene(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            NovelSceneTransitionEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.NOOP),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                action_frames=1,
                delayed_transition_probe_steps=3,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.planner.plan = lambda _frame: [
            NeuralPlan((Action.NOOP,), (1,), 100.0, 0.0),
            NeuralPlan((Action.RIGHT,), (1,), 0.0, 0.0),
        ]

        trigger = agent.decide()
        first_observation = agent.decide()
        resolved = agent.decide()

        self.assertEqual(trigger.action, Action.RIGHT)
        self.assertEqual(first_observation.action, Action.NOOP)
        self.assertEqual(resolved.action, Action.NOOP)
        self.assertEqual(
            agent.anticipated_transition_observations_remaining, 0
        )
        probe = next(
            event
            for event in logger.events
            if event["event"] == "delayed_transition_probe"
        )
        self.assertTrue(probe["novel_scene_observed"])
        self.assertEqual(probe["resolution_step"], 2)
        selected = next(
            event
            for event in logger.events
            if event["event"] == "delayed_transition_branch_selected"
        )
        self.assertEqual(selected["action"], Action.RIGHT)
        self.assertEqual(selected["observations_scheduled"], 2)
        observations = [
            event
            for event in logger.events
            if event["event"] == "anticipated_transition_observation"
        ]
        self.assertEqual(len(observations), 2)
        transition = next(
            event
            for event in logger.events
            if event["event"] == "generic_dark_transition_resolved"
        )
        self.assertFalse(transition["returned_to_known_scene"])

    def test_duration_conditioned_planner_selects_a_press_length(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                action_durations=(1, 3),
                planning_depth=1,
                beam_width=4,
                verify_actions=4,
            ),
        )
        agent.reset()
        decision = agent.decide()
        self.assertIn(decision.action_frames, (1, 3))
        self.assertEqual(decision.planned_durations, (decision.action_frames,))
        self.assertEqual(decision.branches_examined, 4)

    def test_duration_checkpoint_round_trip(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=16,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duration-ensemble.pt"
            digest = save_ensemble_checkpoint(model, path, planning_horizon=2)
            loaded, horizon = load_ensemble_checkpoint(path, frozen=True)
        self.assertEqual(horizon, 2)
        self.assertEqual(digest, loaded.checkpoint_digest)
        self.assertTrue(loaded.duration_conditioned)
        self.assertEqual(loaded.max_action_frames, 16)

    def test_loaded_fixed_duration_checkpoint_rejects_a_different_duration(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            fixed_action_frames=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed-ensemble.pt"
            save_ensemble_checkpoint(model, path, planning_horizon=1)
            loaded, _ = load_ensemble_checkpoint(path, frozen=True)
        with self.assertRaises(ValueError):
            VerifiedNeuralAgent(
                MockPuzzleEnv(),
                loaded,
                "cpu",
                NeuralPlanningConfig(
                    actions=(Action.RIGHT,), planning_depth=1, action_frames=8
                ),
            )

    def test_uncontrollable_animation_selects_long_noop_without_archiving(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        agent = VerifiedNeuralAgent(
            AutonomousAnimationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.A, Action.NOOP),
                action_durations=(1, 4),
                planning_depth=1,
                beam_width=4,
                verify_actions=4,
            ),
        )
        agent.reset()
        decision = agent.decide()
        self.assertEqual(decision.action, Action.NOOP)
        self.assertEqual(decision.action_frames, 4)
        self.assertEqual(agent.archive, [])

    def test_autonomous_grace_ends_when_action_dependent_control_returns(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        agent = VerifiedNeuralAgent(
            AnimationPauseEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.A, Action.NOOP),
                action_durations=(1, 4),
                planning_depth=1,
                beam_width=4,
                verify_actions=4,
                autonomous_grace_decisions=2,
            ),
        )
        agent.reset()
        moving = agent.decide()
        controlled = agent.decide()
        self.assertEqual(moving.action, Action.NOOP)
        self.assertEqual(controlled.action, Action.A)
        self.assertEqual(agent.frame.pixels[0], 255)
        self.assertEqual(agent.autonomous_grace_remaining, 0)

    def test_archive_pruning_preserves_a_minority_scene(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                archive_capacity=3,
            ),
        )
        frame = Frame(2, 2, 1, b"\x00" * 4)
        plan = NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0)
        agent.archive = [
            _ArchivedBranch(index, frame, plan, float(index), "crowded", index)
            for index in range(4)
        ] + [_ArchivedBranch(99, frame, plan, 1.0, "minority", 0)]
        agent._prune_archive()
        self.assertEqual(len(agent.archive), 3)
        self.assertIn("minority", {branch.scene for branch in agent.archive})

    def test_archive_score_prefers_a_rare_causal_spatial_frontier(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        frame = agent.reset()
        plan = NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0)
        ordinary = _ArchivedBranch(1, frame, plan, 0.0, "scene", 1)
        spatial = _ArchivedBranch(
            2,
            frame,
            plan,
            0.0,
            "scene",
            1,
            causal_spatial_signature="new-grid-cell",
            causal_context_signature="world-a",
        )

        self.assertGreater(
            agent._archive_frontier_score(spatial),
            agent._archive_frontier_score(ordinary),
        )
        agent.causal_spatial_visits[
            agent._causal_frontier_key("world-a", "new-grid-cell")
        ] = 3
        self.assertEqual(agent._archive_causal_spatial_bonus(spatial), 1.0)

        same_effect_new_world = _ArchivedBranch(
            3,
            frame,
            plan,
            0.0,
            "scene",
            1,
            causal_spatial_signature="new-grid-cell",
            causal_context_signature="world-b",
        )
        self.assertEqual(
            agent._archive_causal_spatial_bonus(same_effect_new_world),
            2.0,
        )

        capable = _ArchivedBranch(
            4,
            frame,
            plan,
            0.0,
            "scene",
            1,
            causal_affordance_actions=(Action.A, Action.RIGHT),
        )
        self.assertGreater(
            agent._archive_frontier_score(capable),
            agent._archive_frontier_score(ordinary),
        )

    def test_affordance_checkpoint_key_deduplicates_actions(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        frame = agent.reset()

        key = agent._affordance_checkpoint_key(
            frame, "world-a", (Action.B, Action.A, Action.A), Action.LEFT
        )

        self.assertEqual(
            key,
            (agent._signature(frame), Action.LEFT, (Action.A, Action.B)),
        )
        self.assertEqual(
            key,
            agent._affordance_checkpoint_key(
                frame, "world-b", (Action.A, Action.B), Action.LEFT
            ),
        )
        self.assertNotEqual(
            key,
            agent._affordance_checkpoint_key(
                frame, "world-b", (Action.A, Action.B), Action.RIGHT
            ),
        )
        first_pixels = bytearray(32 * 32)
        first_pixels[0] = 10
        first_pixels[1] = 20
        second_pixels = bytearray(first_pixels)
        second_pixels[0], second_pixels[1] = second_pixels[1], second_pixels[0]
        first_pose = Frame(32, 32, 1, bytes(first_pixels))
        second_pose = Frame(32, 32, 1, bytes(second_pixels))
        self.assertNotEqual(first_pose.digest, second_pose.digest)
        self.assertEqual(
            agent._affordance_checkpoint_key(
                first_pose, "world-a", (Action.A,), Action.UP
            ),
            agent._affordance_checkpoint_key(
                second_pose, "world-b", (Action.A,), Action.UP
            ),
        )
        agent.archive_branch_restores[key] += 1
        agent.reset()
        self.assertEqual(agent.archive_branch_restores[key], 0)

    def test_independent_state_owner_survives_source_release(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        agent.reset()
        source_state = env.save_state()
        env.step(Action.RIGHT, 1)
        restore_state = env.save_state()

        independent_state = agent._clone_state_for_independent_owner(
            source_state, restore_state
        )

        self.assertNotEqual(independent_state, source_state)
        self.assertEqual(env.position, 1)
        env.release_state(source_state)
        env.load_state(independent_state)
        self.assertEqual(env.position, 0)
        env.load_state(restore_state)
        self.assertEqual(env.position, 1)

    def test_disconnected_causal_effect_starts_a_new_frontier_context(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        connected = bytearray(16 * 15)
        connected[6 * 16 + 12] = 1
        connected[7 * 16 + 12] = 1
        disconnected = bytearray(connected)
        disconnected[5 * 16 + 14] = 1

        same, detected, components = agent._causal_target_context(
            "world-a", bytes(connected).hex()
        )
        self.assertEqual(same, "world-a")
        self.assertFalse(detected)
        self.assertEqual(components, 1)

        target, detected, components = agent._causal_target_context(
            "world-a", bytes(disconnected).hex()
        )
        self.assertTrue(detected)
        self.assertEqual(components, 2)
        self.assertNotEqual(target, "world-a")
        self.assertTrue(target.startswith("world-a>"))

    def test_stagnation_can_restore_a_causal_frontier_in_the_same_scene(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
            ),
        )
        current = agent.reset()
        state = env.save_state()
        alternative = Frame(8, 8, 1, bytes([0, 255]) + bytes(62))
        scene = agent._scene_signature(current)
        agent.archive.append(
            _ArchivedBranch(
                state=state,
                frame=alternative,
                plan=NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                score=1.0,
                scene=scene,
                created=0,
                origin_signature=agent.current_frontier_signature,
                frontier_signature="causal-frontier",
                causal_spatial_signature="changed-cell",
            )
        )
        agent.visual_stagnation_streak = 1

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, alternative.digest)

    def test_stagnation_prefers_a_branch_from_the_current_causal_context(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
            ),
        )
        current = agent.reset()
        scene = agent._scene_signature(current)
        plan = NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0)
        agent.current_causal_context_signature = "new-world"
        old_frame = Frame(8, 8, 1, bytes([0, 10]) + bytes(62))
        new_frame = Frame(8, 8, 1, bytes([0, 20]) + bytes(62))
        agent.archive = [
            _ArchivedBranch(
                state=env.save_state(),
                frame=old_frame,
                plan=plan,
                score=100.0,
                scene=scene,
                created=2,
                causal_spatial_signature="old-effect",
                causal_context_signature="old-world",
            ),
            _ArchivedBranch(
                state=env.save_state(),
                frame=new_frame,
                plan=plan,
                score=0.0,
                scene=scene,
                created=1,
                causal_spatial_signature="new-effect",
                causal_context_signature="new-world",
                target_causal_context_signature="new-world",
            ),
        ]
        agent.visual_stagnation_streak = 1

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, new_frame.digest)

    def test_stagnation_falls_back_to_an_ancestor_when_context_is_exhausted(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
            ),
        )
        current = agent.reset()
        agent.current_causal_context_signature = "root>persistent-event"
        agent.archive.append(
            _ArchivedBranch(
                state=env.save_state(),
                frame=Frame(8, 8, 1, bytes([0, 10]) + bytes(62)),
                plan=NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                score=100.0,
                scene=agent._scene_signature(current),
                created=0,
                causal_spatial_signature="old-effect",
                causal_context_signature="root",
                target_causal_context_signature="root",
            )
        )
        agent.visual_stagnation_streak = 1

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)

    def test_stagnation_can_recover_a_lost_affordance_from_an_ancestor(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
            ),
        )
        current = agent.reset()
        scene = agent._scene_signature(current)
        plan = NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0)
        agent.current_causal_context_signature = "new-world"
        old_capable_frame = Frame(8, 8, 1, bytes([0, 10]) + bytes(62))
        new_ordinary_frame = Frame(8, 8, 1, bytes([0, 20]) + bytes(62))
        agent.archive = [
            _ArchivedBranch(
                state=env.save_state(),
                frame=old_capable_frame,
                plan=plan,
                score=0.0,
                scene=scene,
                created=1,
                causal_spatial_signature="old-effect",
                causal_context_signature="old-world",
                causal_affordance_actions=(Action.A,),
            ),
            _ArchivedBranch(
                state=env.save_state(),
                frame=new_ordinary_frame,
                plan=plan,
                score=0.0,
                scene=scene,
                created=2,
                causal_spatial_signature="new-effect",
                causal_context_signature="new-world",
                target_causal_context_signature="new-world",
            ),
        ]
        agent.visual_stagnation_streak = 1

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, old_capable_frame.digest)

    def test_stagnation_exhausts_a_new_causal_context_before_its_ancestors(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
            ),
        )
        current = agent.reset()
        scene = agent._scene_signature(current)
        plan = NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0)
        agent.current_causal_context_signature = "new-causal-world"
        agent.causal_outcome_contexts.add("new-causal-world")
        old_capable_frame = Frame(8, 8, 1, bytes([0, 10]) + bytes(62))
        new_successor_frame = Frame(8, 8, 1, bytes([0, 20]) + bytes(62))
        agent.archive = [
            _ArchivedBranch(
                state=env.save_state(),
                frame=old_capable_frame,
                plan=plan,
                score=100.0,
                scene=scene,
                created=1,
                causal_spatial_signature="old-effect",
                causal_context_signature="old-world",
                causal_affordance_actions=(Action.A,),
            ),
            _ArchivedBranch(
                state=env.save_state(),
                frame=new_successor_frame,
                plan=plan,
                score=0.0,
                scene=scene,
                created=2,
                causal_spatial_signature="new-effect",
                causal_context_signature="new-causal-world",
                target_causal_context_signature="new-causal-world",
            ),
        ]
        agent.visual_stagnation_streak = 1

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, new_successor_frame.digest)

    def test_archive_score_prioritizes_a_causal_event_outcome(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        frame = agent.reset()
        plan = NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0)
        ordinary = _ArchivedBranch(1, frame, plan, 0.0, "scene", 1)
        outcome = _ArchivedBranch(
            2,
            frame,
            plan,
            0.0,
            "scene",
            1,
            causal_event_outcome=True,
        )

        self.assertGreater(
            agent._archive_frontier_score(outcome),
            agent._archive_frontier_score(ordinary),
        )

    def test_stagnation_explores_causal_outcomes_breadth_first(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
            ),
        )
        current = agent.reset()
        scene = agent._scene_signature(current)
        plan = NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0)
        agent.current_causal_context_signature = "current-world"
        older_frame = Frame(8, 8, 1, bytes([0, 10]) + bytes(62))
        newer_frame = Frame(8, 8, 1, bytes([0, 20]) + bytes(62))
        agent.archive = [
            _ArchivedBranch(
                env.save_state(),
                older_frame,
                plan,
                0.0,
                scene,
                1,
                causal_spatial_signature="older-event",
                causal_context_signature="older-world",
                causal_event_outcome=True,
            ),
            _ArchivedBranch(
                env.save_state(),
                newer_frame,
                plan,
                100.0,
                scene,
                2,
                causal_spatial_signature="newer-event",
                causal_context_signature="newer-world",
                causal_event_outcome=True,
            ),
        ]
        agent.visual_stagnation_streak = 1

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, older_frame.digest)
        self.assertEqual(
            agent.causal_outcome_restores[
                agent._causal_outcome_key(older_frame, None)
            ],
            1,
        )

    def test_stagnation_breaks_equal_affordance_ties_in_favor_of_older_branches(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
            ),
        )
        current = agent.reset()
        scene = agent._scene_signature(current)
        plan = NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0)
        older_frame = Frame(8, 8, 1, bytes([0, 10]) + bytes(62))
        newer_frame = Frame(8, 8, 1, bytes([0, 20]) + bytes(62))
        agent.archive = [
            _ArchivedBranch(
                state=env.save_state(),
                frame=older_frame,
                plan=plan,
                score=0.0,
                scene=scene,
                created=1,
                causal_spatial_signature="old-effect",
                causal_context_signature="causal-context-root",
                causal_affordance_actions=(Action.A,),
                pose_action=Action.RIGHT,
            ),
            _ArchivedBranch(
                state=env.save_state(),
                frame=newer_frame,
                plan=plan,
                score=0.0,
                scene=scene,
                created=2,
                causal_spatial_signature="new-effect",
                causal_context_signature="causal-context-root",
                causal_affordance_actions=(Action.A,),
                pose_action=Action.RIGHT,
            ),
        ]
        agent.causal_spatial_visits[
            agent._causal_frontier_key(
                "causal-context-root", "old-effect", (Action.A,)
            )
        ] = 100
        agent.visual_stagnation_streak = 1

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, older_frame.digest)
        restored_key = agent._affordance_checkpoint_key(
            older_frame, "causal-context-root", (Action.A,), Action.RIGHT
        )
        self.assertEqual(agent.archive_branch_restores[restored_key], 1)

    def test_verification_budget_covers_distinct_buttons_before_durations(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        actions = (Action.LEFT, Action.RIGHT, Action.A)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=actions,
                action_durations=(1, 4),
                planning_depth=1,
                verify_actions=3,
            ),
        )
        agent.reset()
        source_scene = agent.current_scene
        decision = agent.decide()
        self.assertEqual(decision.branches_examined, 3)
        self.assertEqual(
            {action for scene, action in agent.scene_action_probes if scene == source_scene},
            set(actions),
        )

    def test_control_probes_prefer_long_directional_presses(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT, Action.A),
                action_durations=(1, 8),
                planning_depth=1,
                verify_actions=5,
            ),
        )
        agent.reset()
        plans = {
            (action, duration): NeuralPlan(
                (action,), (duration,), 0.0, 0.0
            )
            for action in agent.config.actions
            for duration in agent.planner.duration_choices
        }
        ranked = [plans[(action, 1)] for action in agent.config.actions]

        probed = agent._add_control_probes(ranked, plans)

        self.assertIn((Action.LEFT, 8), {(p.path[0], p.durations[0]) for p in probed})
        self.assertIn((Action.RIGHT, 8), {(p.path[0], p.durations[0]) for p in probed})

    def test_control_collapse_reserves_a_shorter_duration_probe(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        logger = RecordingLogger()
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT, Action.NOOP),
                action_durations=(1, 8),
                planning_depth=1,
                verify_actions=3,
            ),
            event_logger=logger,
        )
        agent.reset()
        agent.current_frontier_signature = "source"
        agent.temporal_option_values[("source", Action.RIGHT, 8)] = -2.0
        agent.temporal_option_samples[("source", Action.RIGHT, 8)] = 1
        plans = {
            (action, duration): NeuralPlan(
                (action,), (duration,), 0.0, 0.0
            )
            for action in agent.config.actions
            for duration in agent.planner.duration_choices
        }
        ranked = [
            plans[(Action.NOOP, 8)],
            plans[(Action.RIGHT, 8)],
            plans[(Action.NOOP, 1)],
        ]

        probed = agent._add_control_probes(ranked, plans)

        keys = {(plan.path[0], plan.durations[0]) for plan in probed}
        self.assertIn((Action.RIGHT, 8), keys)
        self.assertIn((Action.RIGHT, 1), keys)
        event = next(
            event
            for event in logger.events
            if event["event"] == "behavior_probe_selected"
        )
        recovery = [
            probe
            for probe in event["probes"]
            if probe["control_collapse_recovery_probe"]
        ]
        self.assertEqual(
            recovery,
            [
                {
                    "action": Action.RIGHT,
                    "action_frames": 1,
                    "prior_observations": 0,
                    "causal_continuation": False,
                    "long_press_control": False,
                    "short_press_control": False,
                    "control_collapse_recovery_probe": True,
                    "matched_causal_observation": False,
                }
            ],
        )

    def test_matched_control_probes_reserve_canonical_behavior_slots(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=16,
        )
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                action_durations=(1, 16),
                planning_depth=1,
                verify_actions=6,
            ),
        )
        ranked_actions = (
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.A,
            Action.B,
        )
        ranked = [
            NeuralPlan((action,), (1,), float(index), 0.0)
            for index, action in enumerate(ranked_actions)
        ]
        best = {
            (plan.path[0], plan.durations[0]): plan for plan in ranked
        }
        noop = NeuralPlan((Action.NOOP,), (16,), 0.0, 0.0)
        up = NeuralPlan((Action.UP,), (16,), 0.0, 0.0)
        right_long = NeuralPlan((Action.RIGHT,), (16,), 0.0, 0.0)
        start = NeuralPlan((Action.START,), (16,), 100.0, 0.0)
        best[(Action.NOOP, 16)] = noop
        best[(Action.UP, 16)] = up
        best[(Action.RIGHT, 16)] = right_long
        best[(Action.START, 16)] = start

        agent.reset()
        agent.last_action = Action.RIGHT
        agent.last_action_was_causal_spatial = True
        result = agent._add_control_probes(ranked, best)

        self.assertEqual(len(result), 6)
        self.assertIn(noop, result)
        self.assertIn(up, result)
        self.assertIn(right_long, result)
        self.assertNotIn(start, result)

    def test_causal_continuation_survives_the_neutral_observation(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=16,
        )
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                action_durations=(1, 16),
                planning_depth=1,
                verify_actions=6,
            ),
        )
        agent.reset()
        agent.active_temporal_option = _TemporalOptionTrace(
            choice=("source", Action.RIGHT, 16),
            initiation_decision=1,
            start_decision=2,
            entry_signature="source",
            entry_scene="scene",
            causal_evidence=True,
        )
        ranked = [
            NeuralPlan((Action.UP,), (1,), 0.0, 0.0),
            NeuralPlan((Action.DOWN,), (1,), 0.0, 0.0),
        ]
        right_long = NeuralPlan((Action.RIGHT,), (16,), 0.0, 0.0)
        best = {
            (Action.UP, 1): ranked[0],
            (Action.DOWN, 1): ranked[1],
            (Action.NOOP, 16): NeuralPlan((Action.NOOP,), (16,), 0.0, 0.0),
            (Action.RIGHT, 16): right_long,
        }

        result = agent._add_control_probes(ranked, best)

        self.assertIn(right_long, result)

    def test_active_behavior_probe_rotates_then_separates_hypotheses(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.UP, Action.DOWN),
                planning_depth=1,
                abstraction_latent_rmse_threshold=100.0,
                behavioral_abstraction_rmse_threshold=1e-9,
            ),
        )
        source = agent.reset()
        outcomes = {
            (Action.NOOP, 4): self.frame(10),
            (Action.UP, 4): self.frame(11),
            (Action.DOWN, 4): self.frame(12),
        }

        first = agent._behavior_probe_selection(source)
        self.assertEqual(first.reason, "coverage_rotation")
        self.assertEqual(first.selected_control, Action.UP)
        first_cluster = agent._behavioral_signature(
            source, outcomes, agent.current_frontier_signature, first
        )

        second = agent._behavior_probe_selection(source)
        self.assertEqual(second.selected_control, Action.DOWN)
        self.assertEqual(
            agent._behavioral_signature(
                source,
                outcomes,
                agent._new_provisional_signature(),
                second,
            ),
            first_cluster,
        )
        self.assertIn(
            (Action.DOWN, 4),
            agent.behavior_clusters[0].probe_centroids,
        )

        third = agent._behavior_probe_selection(source)
        self.assertEqual(third.selected_control, Action.UP)
        split_cluster = agent._behavioral_signature(
            source,
            {
                (Action.NOOP, 4): outcomes[(Action.NOOP, 4)],
                (Action.UP, 4): self.frame(200),
            },
            agent._new_provisional_signature(),
            third,
        )
        self.assertNotEqual(split_cluster, first_cluster)

        discriminating = agent._behavior_probe_selection(source)
        self.assertEqual(discriminating.reason, "hypothesis_separation")
        self.assertEqual(discriminating.selected_control, Action.UP)
        self.assertGreater(discriminating.hypothesis_separation, 0.0)

        ambiguous = _BehaviorProbeSelection(
            ((Action.NOOP, 4), (Action.RIGHT, 4)),
            discriminating.visual_cluster,
            "coverage_rotation",
            Action.RIGHT,
        )
        provisional = agent._new_provisional_signature()
        unresolved = agent._behavioral_signature(
            source,
            {
                (Action.NOOP, 4): outcomes[(Action.NOOP, 4)],
                (Action.RIGHT, 4): self.frame(13),
            },
            provisional,
            ambiguous,
        )
        self.assertEqual(unresolved, provisional)
        self.assertEqual(len(agent.behavior_clusters), 2)

    def test_delayed_visual_return_credits_the_loop_and_requests_recovery(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                planning_depth=1,
                delayed_return_min_length=3,
            ),
        )
        agent.reset()
        frames = [
            Frame(
                8,
                8,
                1,
                bytes((((x + y + offset) % 8) * 30) for y in range(8) for x in range(8)),
            )
            for offset in range(3)
        ]
        agent.visual_last_visit = {}
        transitions = (
            (1, "scene-a", Action.RIGHT, frames[0]),
            (2, "scene-b", Action.DOWN, frames[1]),
            (3, "scene-c", Action.LEFT, frames[2]),
            (4, "scene-d", Action.UP, frames[0]),
        )
        for decision, source_scene, action, target in transitions:
            agent.decision_index = decision
            agent._update_persistent_frontier(agent._signature(target), 1.0)
            agent._record_delayed_return(
                source_scene,
                action,
                4,
                target,
                agent._scene_signature(target),
            )

        self.assertTrue(agent.delayed_return_recovery)
        self.assertEqual(agent.delayed_return_loop_start, 1)
        self.assertEqual(sum(agent.delayed_return_costs.values()), 3)
        self.assertEqual(agent.delayed_return_costs[("scene-d", Action.UP, 4)], 1)
        self.assertLess(agent.frontier_values[agent._signature(frames[0])], 0.0)

    def test_persistent_frontier_accumulates_discounted_successor_novelty(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                frontier_credit_horizon=3,
                frontier_discount=1.0,
            ),
        )
        initial = agent.reset()
        initial_signature = agent.current_frontier_signature
        for decision, signature in enumerate(("one", "two", "three"), 1):
            agent.decision_index = decision
            agent._update_persistent_frontier(signature, 1.0)

        self.assertEqual(agent.frontier_values[initial_signature], 3.0)
        self.assertEqual(agent._frontier_estimate("one"), 2.0)
        self.assertEqual(agent._frontier_estimate("two"), 1.0)

    def test_repeated_identical_pixels_do_not_create_frontier_reward(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP,),
                planning_depth=1,
                verify_actions=1,
                action_frames=1,
            ),
        )
        initial = agent.reset()
        agent.run(3)

        self.assertEqual(
            agent._frontier_estimate(agent._abstract_signature(initial)), 0.0
        )
        self.assertTrue(
            all(trace.discounted_return == 0.0 for trace in agent.frontier_traces)
        )

    def test_frozen_encoder_clusters_nearby_frames_and_shares_choice_value(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                abstraction_latent_rmse_threshold=100.0,
            ),
        )
        before = model.checkpoint_digest
        agent.reset()
        first = self.frame(5)
        changed = bytearray(first.pixels)
        changed[0] = (changed[0] + 1) % 256
        nearby = Frame(first.width, first.height, first.channels, bytes(changed))
        different = Frame(32, 32, 3, b"\xff" * (32 * 32 * 3))
        self.assertEqual(
            agent._scene_signature(first), agent._scene_signature(nearby)
        )
        self.assertNotEqual(
            agent._scene_signature(first), agent._scene_signature(different)
        )

        first_cluster = agent._abstract_signature(first)
        nearby_cluster = agent._abstract_signature(nearby)
        different_cluster = agent._abstract_signature(different)
        self.assertEqual(first_cluster, nearby_cluster)
        self.assertNotEqual(first_cluster, different_cluster)
        choice = (first_cluster, Action.RIGHT, 4)
        agent.frontier_choice_values[choice] = 3.0
        agent.frontier_choice_samples[choice] = 1
        shared, known = agent._choice_frontier_estimate(
            nearby_cluster, Action.RIGHT, 4
        )
        self.assertTrue(known)
        self.assertEqual(shared, 3.0)
        self.assertEqual(before, model.checkpoint_digest)

    def test_behavioral_abstraction_shares_only_matching_observed_futures(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                planning_depth=1,
                abstraction_latent_rmse_threshold=100.0,
                behavioral_abstraction_rmse_threshold=1e-9,
            ),
        )
        before = model.checkpoint_digest
        source = agent.reset()
        matching_outcomes = {
            (Action.LEFT, 4): self.frame(6),
            (Action.RIGHT, 4): self.frame(7),
        }
        first = agent._behavioral_signature(
            source, matching_outcomes, agent.current_frontier_signature
        )
        choice = (first, Action.LEFT, 4)
        agent.frontier_choice_values[choice] = 3.0
        agent.frontier_choice_samples[choice] = 1

        second_provisional = agent._new_provisional_signature()
        value, known = agent._choice_frontier_estimate(
            second_provisional, Action.LEFT, 4
        )
        self.assertFalse(known)
        self.assertEqual(value, 0.0)
        agent.frontier_values[second_provisional] = 2.0
        agent.frontier_samples[second_provisional] = 1
        second = agent._behavioral_signature(
            source, matching_outcomes, second_provisional
        )
        self.assertEqual(first, second)
        self.assertEqual(agent.frontier_values[first], 2.0)
        self.assertNotIn(second_provisional, agent.frontier_values)
        value, known = agent._choice_frontier_estimate(second, Action.LEFT, 4)
        self.assertTrue(known)
        self.assertEqual(value, 3.0)

        different = agent._behavioral_signature(
            source,
            {
                (Action.LEFT, 4): self.frame(200),
                (Action.RIGHT, 4): self.frame(201),
            },
            agent._new_provisional_signature(),
        )
        self.assertNotEqual(first, different)
        value, known = agent._choice_frontier_estimate(different, Action.LEFT, 4)
        self.assertFalse(known)
        self.assertEqual(value, 0.0)
        self.assertEqual(before, model.checkpoint_digest)

    def test_frontier_choice_value_learns_a_delayed_return_outcome(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        agent.reset()
        agent.decision_index = 1
        agent._update_persistent_frontier(
            "target", 1.0, "source", Action.RIGHT, 4
        )
        provisional, known = agent._choice_frontier_estimate(
            "source", Action.RIGHT, 4
        )
        self.assertTrue(known)
        self.assertEqual(provisional, 1.0)

        agent._penalize_frontier_loop(1)

        learned, known = agent._choice_frontier_estimate(
            "source", Action.RIGHT, 4
        )
        self.assertTrue(known)
        self.assertEqual(learned, -agent.config.frontier_return_penalty)

    def test_known_bad_archive_choice_overrides_inherited_origin_value(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        frame = agent.reset()
        plan = NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0)
        branch = _ArchivedBranch(
            b"state", frame, plan, 0.0, "scene", 1, "valuable-origin"
        )
        agent.frontier_values["valuable-origin"] = 10.0
        self.assertGreater(agent._archive_frontier_score(branch), 0.0)
        choice = ("valuable-origin", Action.RIGHT, 4)
        agent.frontier_choice_values[choice] = -2.0
        agent.frontier_choice_samples[choice] = 1

        self.assertEqual(agent._archive_frontier_score(branch), -2.0)
        agent.temporal_option_values[choice] = 2.0
        agent.temporal_option_samples[choice] = 1
        self.assertEqual(
            agent._archive_frontier_score(branch),
            -2.0 + 2.0 * agent.config.temporal_option_score_weight,
        )

    def test_temporal_option_credits_an_initiating_action_through_passive_dynamics(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.START,), planning_depth=1),
        )
        before = model.checkpoint_digest
        agent.reset()
        choice = ("source", Action.START, 4)
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_causal_evidence = True

        agent.decision_index = 1
        agent._advance_temporal_option("animation-a", "scene-a", passive=True)
        agent.decision_index = 2
        agent._advance_temporal_option(
            "animation-b", "scene-b", passive=True, grace_continuation=True
        )
        self.assertIsNotNone(agent.active_temporal_option)
        self.assertEqual(agent.active_temporal_option.passive_decisions, 2)

        agent.decision_index = 3
        agent._advance_temporal_option("endpoint", "scene-c", passive=False)
        learned, known = agent._temporal_option_estimate(*choice)
        self.assertTrue(known)
        self.assertGreater(learned, 1.0)
        self.assertEqual(agent.temporal_option_samples[choice], 1)
        self.assertIsNone(agent.active_temporal_option)
        self.assertEqual(before, model.checkpoint_digest)

    def test_temporal_option_penalizes_a_return_to_its_source(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.A,), planning_depth=1),
        )
        agent.reset()
        choice = ("source", Action.A, 1)
        agent.behavior_visits["source"] = 1
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_causal_evidence = True
        for decision in range(1, 5):
            agent.decision_index = decision
            agent._advance_temporal_option(
                f"animation-{decision}",
                f"scene-{decision}",
                passive=True,
            )
        agent.decision_index = 5
        agent._advance_temporal_option("source", "scene", passive=False)

        learned, known = agent._temporal_option_estimate(*choice)
        self.assertTrue(known)
        self.assertLess(learned, 0.0)

    def test_single_neutral_observation_is_not_a_delayed_return_hazard(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        agent.reset()
        choice = ("source", Action.RIGHT, 16)
        agent.behavior_visits["source"] = 1
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_causal_evidence = True
        agent.decision_index = 1
        agent._advance_temporal_option("source", "room", passive=True)
        agent.decision_index = 2
        agent._advance_temporal_option("source", "room", passive=False)

        learned, known = agent._temporal_option_estimate(*choice)
        self.assertTrue(known)
        self.assertGreaterEqual(learned, 0.0)

    def test_temporal_option_penalizes_a_return_to_an_earlier_known_state(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.SELECT,), planning_depth=1),
        )
        agent.reset()
        choice = ("moved-state", Action.SELECT, 1)
        agent.behavior_visits["earlier-state"] = 1
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_causal_evidence = True
        for decision in range(1, 5):
            agent.decision_index = decision
            agent._advance_temporal_option(
                f"animation-{decision}",
                f"fade-{decision}",
                passive=True,
            )
        agent.decision_index = 5
        agent._advance_temporal_option("earlier-state", "room", passive=False)

        learned, known = agent._temporal_option_estimate(*choice)
        self.assertTrue(known)
        self.assertLess(learned, 0.0)

    def test_robust_direct_causal_return_generalizes_action_hazard(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=9, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.SELECT,), planning_depth=1),
        )
        agent.reset()
        choice = ("source", Action.SELECT, 1)
        agent.behavior_visits["known-endpoint"] = 1
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_causal_evidence = True
        for decision in range(1, 5):
            agent.decision_index = decision
            agent._advance_temporal_option(
                f"animation-{decision}",
                f"scene-{decision}",
                passive=True,
            )
        agent.decision_index = 5
        agent._advance_temporal_option(
            "known-endpoint", "endpoint-scene", passive=False
        )

        self.assertEqual(agent.temporal_option_action_samples[Action.SELECT], 1)
        self.assertLess(agent.temporal_option_action_values[Action.SELECT], 0.0)

    def test_temporal_option_action_prior_generalizes_unseen_state_and_duration(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.SELECT,), planning_depth=1),
        )
        agent.reset()
        exact_choice = ("moved-state", Action.SELECT, 1)
        agent._record_temporal_option_sample(
            exact_choice, -2.0, generalize_action_hazard=True
        )

        exact, exact_known = agent._temporal_option_estimate(*exact_choice)
        inherited, inherited_known = agent._temporal_option_estimate(
            "earlier-state", Action.SELECT, 8
        )

        self.assertTrue(exact_known)
        self.assertEqual(exact, -2.0)
        self.assertTrue(inherited_known)
        self.assertEqual(
            inherited,
            -2.0 * agent.config.temporal_option_action_prior_weight,
        )

        ordinary_choice = ("another-state", Action.UP, 16)
        agent._record_temporal_option_sample(ordinary_choice, -3.0)
        self.assertEqual(
            agent._temporal_option_estimate("unseen-state", Action.UP, 4),
            (0.0, False),
        )

    def test_temporal_option_requires_action_dependent_counterfactual_evidence(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.START, Action.NOOP), planning_depth=1),
        )
        identical = self.frame(10)
        changed = self.frame(200)
        start = NeuralPlan((Action.START,), (4,), 0.0, 0.0)
        noop = NeuralPlan((Action.NOOP,), (4,), 0.0, 0.0)
        candidate = (0.0, start, b"start", identical)
        uncaused = (0.0, noop, b"noop", identical)
        caused = (0.0, noop, b"noop", changed)

        eligible, contrast, count = agent._option_initiation_evidence(
            candidate, [candidate, uncaused]
        )
        self.assertFalse(eligible)
        self.assertEqual(contrast, 0.0)
        self.assertEqual(count, 1)
        self.assertIs(
            agent._delayed_option_counterfactual(
                candidate, [candidate, uncaused]
            ),
            uncaused,
        )

        eligible, contrast, count = agent._option_initiation_evidence(
            candidate, [candidate, caused]
        )
        self.assertTrue(eligible)
        self.assertGreater(contrast, agent.config.action_equivalence_threshold)
        self.assertEqual(count, 1)
        self.assertIsNone(
            agent._delayed_option_counterfactual(candidate, [candidate, caused])
        )

    def test_delayed_counterfactual_divergence_supplies_causal_option_evidence(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = DelayedCausalityEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.START, Action.A, Action.NOOP), planning_depth=1
            ),
        )
        agent.reset()
        root = env.save_state()
        immediate = env.step(Action.START)
        factual_state = env.save_state()
        env.load_state(root)
        counterfactual_frame = env.step(Action.A)
        counterfactual_state = env.save_state()
        self.assertEqual(immediate, counterfactual_frame)
        env.load_state(factual_state)
        factual_target = env.step(Action.NOOP)

        choice = ("source", Action.START, 1)
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_counterfactual = _OptionCounterfactual(
            counterfactual_state,
            counterfactual_frame,
            Action.A,
            1,
        )
        agent.decision_index = 1
        agent._advance_temporal_option(
            "animation",
            "scene-a",
            passive=True,
            passive_action=Action.NOOP,
            passive_duration=1,
            factual_target=factual_target,
        )
        self.assertFalse(agent.active_temporal_option.causal_evidence)
        self.assertGreater(
            agent.active_temporal_option.counterfactual.maximum_contrast,
            agent.config.action_equivalence_threshold,
        )

        agent.frame = factual_target
        agent.decision_index = 2
        agent._advance_temporal_option("endpoint", "scene-b", passive=False)
        learned, known = agent._temporal_option_estimate(*choice)
        self.assertTrue(known)
        self.assertGreater(learned, 0.0)
        self.assertIsNone(agent.active_temporal_option)
        self.assertGreaterEqual(env.released, 2)

    def test_delayed_counterfactual_requires_endpoint_divergence(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = DelayedCausalityEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.START,), planning_depth=1),
        )
        endpoint = agent.reset()
        choice = ("source", Action.START, 1)
        agent.active_temporal_option = _TemporalOptionTrace(
            choice=choice,
            initiation_decision=1,
            start_decision=2,
            entry_signature="animation",
            entry_scene="scene",
            counterfactual=_OptionCounterfactual(
                env.save_state(), endpoint, Action.A, 1
            ),
            passive_decisions=3,
        )
        agent.frame = endpoint
        agent.decision_index = 4
        agent._advance_temporal_option("endpoint", "scene", passive=False)

        learned, known = agent._temporal_option_estimate(*choice)
        self.assertFalse(known)
        self.assertEqual(learned, 0.0)

    def test_new_delayed_intervention_supersedes_active_option_credit(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.UP, Action.SELECT), planning_depth=1),
        )
        agent.reset()
        prior_choice = ("source", Action.UP, 16)
        agent.active_temporal_option = _TemporalOptionTrace(
            choice=prior_choice,
            initiation_decision=1,
            start_decision=2,
            entry_signature="animation",
            entry_scene="scene",
            causal_evidence=True,
        )

        superseded = agent._supersede_temporal_option_for_intervention(
            Action.SELECT, has_causal_candidate=True
        )

        self.assertTrue(superseded)
        self.assertIsNone(agent.active_temporal_option)
        self.assertEqual(agent._temporal_option_estimate(*prior_choice), (0.0, False))

        agent.active_temporal_option = _TemporalOptionTrace(
            choice=prior_choice,
            initiation_decision=1,
            start_decision=2,
            entry_signature="animation",
            entry_scene="scene",
        )
        self.assertFalse(
            agent._supersede_temporal_option_for_intervention(
                Action.NOOP, has_causal_candidate=True
            )
        )
        self.assertIsNotNone(agent.active_temporal_option)

    def test_archive_recovery_prefers_learned_persistent_frontier_value(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
            ),
        )
        agent.reset()
        root = env.save_state()
        low_frame = env.step(Action.RIGHT)
        low_state = env.save_state()
        env.load_state(root)
        high_frame = env.step(Action.DOWN)
        high_state = env.save_state()
        low_plan = NeuralPlan((Action.RIGHT,), (1,), 100.0, 0.0)
        high_plan = NeuralPlan((Action.DOWN,), (1,), -100.0, 0.0)
        agent.archive = [
            _ArchivedBranch(
                low_state,
                low_frame,
                low_plan,
                100.0,
                "low-scene",
                1,
                "low-origin",
            ),
            _ArchivedBranch(
                high_state,
                high_frame,
                high_plan,
                -100.0,
                "high-scene",
                1,
                "high-origin",
            ),
        ]
        agent.frontier_values["low-origin"] = 1.0
        agent.frontier_values["high-origin"] = 10.0
        agent.visual_stagnation_streak = 1

        decision = agent._restore_if_stagnant()

        self.assertEqual(decision.frame.digest, high_frame.digest)

    def test_delayed_return_restores_a_distinct_branch_before_stagnation(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=99,
            ),
        )
        agent.reset()
        branch_frame = env.step(Action.RIGHT)
        branch_state = env.save_state()
        plan = NeuralPlan((Action.RIGHT,), (1,), 0.0, 0.0)
        agent.archive = [
            _ArchivedBranch(
                branch_state,
                branch_frame,
                plan,
                0.0,
                agent._scene_signature(branch_frame),
                2,
            )
        ]
        agent.delayed_return_recovery = True
        agent.delayed_return_loop_start = 1

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        self.assertTrue(decision.restored_archive)
        self.assertFalse(agent.delayed_return_recovery)
