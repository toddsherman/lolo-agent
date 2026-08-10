from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from .environment import Action, PixelSaveStateEnv
from .pixels import Frame


@dataclass(frozen=True)
class BidirectionalProbeResult:
    paths_evaluated: int
    returning_paths: int
    return_observed: bool
    shortest_return_depth: Optional[int]
    best_matched_noop_l1: float


@dataclass
class _ProbeNode:
    state: object
    frame: Frame
    path: Tuple[Action, ...]
    matched_noop_l1: float
    owned: bool


class BidirectionalProbeCollector:
    """Explore short endpoint-to-source paths without exposing them to policy.

    Each candidate is compared with a NOOP trajectory restored from the original
    branch root and advanced for the same total number of emulator frames. This
    controls for animation and autonomous dynamics using pixels and outcomes only.
    """

    def __init__(
        self,
        env: PixelSaveStateEnv,
        actions: Sequence[Action],
        maximum_depth: int,
        beam_width: int,
        pixel_l1_threshold: float,
        emit: Optional[Callable[..., Any]] = None,
        frame_fields: Optional[Callable[[Frame], Dict[str, Any]]] = None,
        state_id: Optional[Callable[[object], Optional[str]]] = None,
    ) -> None:
        if maximum_depth <= 0 or beam_width <= 0:
            raise ValueError("probe depth and beam width must be positive")
        if pixel_l1_threshold < 0.0:
            raise ValueError("probe pixel threshold must be non-negative")
        if not actions:
            raise ValueError("probe collection requires at least one action")
        self.env = env
        self.actions = tuple(actions)
        self.maximum_depth = maximum_depth
        self.beam_width = beam_width
        self.pixel_l1_threshold = pixel_l1_threshold
        self.emit = emit
        self.frame_fields = frame_fields or (lambda frame: {"frame": frame.digest})
        self.state_id = state_id or (lambda _state: None)

    def _emit(self, event: str, **fields: Any) -> None:
        if self.emit is not None:
            self.emit(event, **fields)

    def collect(
        self,
        *,
        root_state: object,
        endpoint_state: object,
        source_frame: Frame,
        endpoint_frame: Frame,
        initial_action: Action,
        action_frames: int,
        decision: int,
        branch_id: str,
        candidate_rank: int,
    ) -> BidirectionalProbeResult:
        if action_frames <= 0:
            raise ValueError("probe action duration must be positive")
        release_state = getattr(self.env, "release_state", None)
        owned_states: Dict[int, object] = {}

        def release(state: object) -> None:
            if id(state) not in owned_states:
                return
            if release_state is not None:
                release_state(state)
            owned_states.pop(id(state), None)

        previous_phase = getattr(self.env, "phase", None)
        has_phase = hasattr(self.env, "phase")
        if has_phase:
            setattr(self.env, "phase", "returnability_probe")
        frontier = [
            _ProbeNode(endpoint_state, endpoint_frame, (), float("inf"), False)
        ]
        paths_evaluated = 0
        returning_paths = 0
        shortest_return_depth: Optional[int] = None
        best_distance = float("inf")
        try:
            self._emit(
                "bidirectional_probe_started",
                decision=decision,
                branch_id=branch_id,
                candidate_rank=candidate_rank,
                initial_action=initial_action,
                initial_action_frames=action_frames,
                maximum_depth=self.maximum_depth,
                beam_width=self.beam_width,
                pixel_l1_threshold=self.pixel_l1_threshold,
                actions=self.actions,
                root_state_id=self.state_id(root_state),
                endpoint_state_id=self.state_id(endpoint_state),
                source_frame=source_frame.digest,
                endpoint_frame=endpoint_frame.digest,
            )
            for depth in range(1, self.maximum_depth + 1):
                matched_noop_frames = action_frames * (depth + 1)
                self.env.load_state(root_state)
                matched_noop = self.env.step(Action.NOOP, matched_noop_frames)
                self._emit(
                    "bidirectional_probe_reference",
                    decision=decision,
                    branch_id=branch_id,
                    candidate_rank=candidate_rank,
                    probe_depth=depth,
                    matched_noop_frames=matched_noop_frames,
                    env_step_seq=getattr(self.env, "last_step_seq", None),
                    **self.frame_fields(matched_noop),
                )
                children = []
                depth_return_observed = False
                for node in frontier:
                    for action in self.actions:
                        self.env.load_state(node.state)
                        target = self.env.step(action, action_frames)
                        env_step_seq = getattr(self.env, "last_step_seq", None)
                        child_state = self.env.save_state()
                        owned_states[id(child_state)] = child_state
                        path = node.path + (action,)
                        distance = target.mean_absolute_difference(matched_noop)
                        exact_match = target.digest == matched_noop.digest
                        returned = exact_match or distance <= self.pixel_l1_threshold
                        paths_evaluated += 1
                        best_distance = min(best_distance, distance)
                        if returned:
                            returning_paths += 1
                            depth_return_observed = True
                            if shortest_return_depth is None:
                                shortest_return_depth = depth
                        self._emit(
                            "bidirectional_probe_step",
                            decision=decision,
                            branch_id=branch_id,
                            candidate_rank=candidate_rank,
                            initial_action=initial_action,
                            initial_action_frames=action_frames,
                            probe_depth=depth,
                            probe_path=path,
                            probe_action=action,
                            probe_action_frames=action_frames,
                            total_action_frames=matched_noop_frames,
                            matched_noop_frame=matched_noop.digest,
                            matched_noop_l1=distance,
                            exact_pixel_return=exact_match,
                            return_observed=returned,
                            source_frame=source_frame.digest,
                            endpoint_frame=endpoint_frame.digest,
                            parent_state_id=self.state_id(node.state),
                            child_state_id=self.state_id(child_state),
                            env_step_seq=env_step_seq,
                            state_save_seq=getattr(
                                self.env, "last_state_event_seq", None
                            ),
                            **self.frame_fields(target),
                        )
                        children.append(
                            _ProbeNode(child_state, target, path, distance, True)
                        )
                    if node.owned:
                        release(node.state)
                frontier = []
                if depth_return_observed:
                    for child in children:
                        release(child.state)
                    break
                children.sort(
                    key=lambda node: (
                        node.matched_noop_l1,
                        tuple(action.value for action in node.path),
                    )
                )
                frontier = children[: self.beam_width]
                for child in children[self.beam_width :]:
                    release(child.state)
            result = BidirectionalProbeResult(
                paths_evaluated=paths_evaluated,
                returning_paths=returning_paths,
                return_observed=returning_paths > 0,
                shortest_return_depth=shortest_return_depth,
                best_matched_noop_l1=best_distance,
            )
            self._emit(
                "bidirectional_probe_completed",
                decision=decision,
                branch_id=branch_id,
                candidate_rank=candidate_rank,
                initial_action=initial_action,
                initial_action_frames=action_frames,
                maximum_depth=self.maximum_depth,
                beam_width=self.beam_width,
                actions=self.actions,
                pixel_l1_threshold=self.pixel_l1_threshold,
                paths_evaluated=result.paths_evaluated,
                returning_paths=result.returning_paths,
                return_observed=result.return_observed,
                no_return_within_probe_budget=not result.return_observed,
                shortest_return_depth=result.shortest_return_depth,
                best_matched_noop_l1=result.best_matched_noop_l1,
                source_frame=source_frame.digest,
                endpoint_frame=endpoint_frame.digest,
            )
            return result
        finally:
            for state in list(owned_states.values()):
                release(state)
            try:
                self.env.load_state(endpoint_state)
            finally:
                if has_phase:
                    setattr(self.env, "phase", previous_phase)
