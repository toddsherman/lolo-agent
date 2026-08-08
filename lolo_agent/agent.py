from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .environment import Action, PixelSaveStateEnv
from .memory import EpisodeEvent, EpisodicMemory, VisualNovelty
from .pixels import Frame
from .world_model import EmpiricalWorldModel


@dataclass(frozen=True)
class AgentConfig:
    actions: Tuple[Action, ...] = (
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
        Action.A,
        Action.B,
        Action.NOOP,
    )
    planning_depth: int = 2
    beam_width: int = 12
    action_frames: int = 1
    discount: float = 0.9
    novelty_weight: float = 1.0
    surprise_weight: float = 0.12
    uncertainty_weight: float = 0.35
    visual_change_weight: float = 0.3


@dataclass(frozen=True)
class Decision:
    action: Action
    frame: Frame
    planned_path: Tuple[Action, ...]
    score: float
    branches_examined: int
    restored_archive: bool = False
    action_frames: int = 1
    planned_durations: Tuple[int, ...] = ()


@dataclass
class _Node:
    path: Tuple[Action, ...]
    state: object
    frame: Frame
    score: float
    first_state: Optional[object]
    first_frame: Optional[Frame]


class BranchingAgent:
    """Curiosity-driven receding-horizon planner over emulator branches."""

    def __init__(
        self,
        env: PixelSaveStateEnv,
        model: Optional[EmpiricalWorldModel] = None,
        config: Optional[AgentConfig] = None,
        training: bool = True,
    ) -> None:
        self.env = env
        self.model = model or EmpiricalWorldModel()
        self.config = config or AgentConfig()
        self.novelty = VisualNovelty()
        self.memory = EpisodicMemory()
        self.training = training
        if training:
            self.model.unfreeze()
        else:
            self.model.freeze()
        self.frame: Optional[Frame] = None

    def reset(self) -> Frame:
        self.frame = self.env.reset()
        self.novelty = VisualNovelty()
        self.memory = EpisodicMemory()
        self.novelty.observe(self.model.signature(self.frame))
        return self.frame

    def set_training(self, training: bool) -> None:
        self.training = training
        if training:
            self.model.unfreeze()
        else:
            self.model.freeze()

    def decide(self) -> Decision:
        if self.frame is None:
            self.reset()
        assert self.frame is not None

        root_state = self.env.save_state()
        frontier: List[_Node] = [_Node((), root_state, self.frame, 0.0, None, None)]
        candidates: List[_Node] = []
        created_states: List[object] = [root_state]
        branches = 0

        for depth in range(self.config.planning_depth):
            expanded: List[_Node] = []
            observations: List[Tuple[Frame, Action, Frame, EpisodeEvent]] = []
            for node in frontier:
                for action in self.config.actions:
                    self.env.load_state(node.state)
                    target = self.env.step(action, self.config.action_frames)
                    target_state = self.env.save_state()
                    created_states.append(target_state)
                    branches += 1

                    source_key = self.model.signature(node.frame)
                    target_key = self.model.signature(target)
                    prediction = self.model.predict(node.frame, action, target)
                    visual_change = node.frame.mean_absolute_difference(target)
                    repetition = self.memory.repetition_penalty(source_key, action, target_key)
                    immediate = (
                        self.config.novelty_weight * self.novelty.score(target_key)
                        + self.config.surprise_weight * min(prediction.surprise, 8.0)
                        + self.config.uncertainty_weight * prediction.uncertainty
                        + self.config.visual_change_weight * visual_change
                        - repetition
                    )
                    event = EpisodeEvent(source_key, action, target_key, visual_change)
                    observations.append((node.frame, action, target, event))

                    first_state = node.first_state if node.first_state is not None else target_state
                    first_frame = node.first_frame if node.first_frame is not None else target
                    child = _Node(
                        node.path + (action,),
                        target_state,
                        target,
                        node.score + (self.config.discount ** depth) * immediate,
                        first_state,
                        first_frame,
                    )
                    expanded.append(child)
            # Learn only after scoring the whole search layer. This preserves
            # information gained from branches without making scores depend on
            # the arbitrary order in which sibling actions were enumerated.
            for source, action, target, event in observations:
                self.memory.record(event)
                if self.training:
                    self.model.observe(source, action, target)
            expanded.sort(key=lambda item: (-item.score, tuple(a.value for a in item.path)))
            frontier = expanded[: self.config.beam_width]
            candidates.extend(frontier)

        if not candidates:
            self.env.load_state(root_state)
            raise RuntimeError("planner produced no candidates")

        best = max(candidates, key=lambda item: (item.score, tuple(a.value for a in item.path)))
        assert best.first_state is not None and best.first_frame is not None
        self.env.load_state(best.first_state)
        self.frame = best.first_frame
        self.novelty.observe(self.model.signature(self.frame))
        release_state = getattr(self.env, "release_state", None)
        if release_state is not None:
            for state in created_states:
                release_state(state)
        return Decision(best.path[0], self.frame, best.path, best.score, branches)

    def run(self, decisions: int) -> List[Decision]:
        if decisions < 0:
            raise ValueError("decisions must be non-negative")
        if self.frame is None:
            self.reset()
        return [self.decide() for _ in range(decisions)]
