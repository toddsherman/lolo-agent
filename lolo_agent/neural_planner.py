from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Counter as CounterType, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from .agent import Decision
from .ensemble_world_model import EnsembleVisualDynamicsModel
from .environment import Action, PixelSaveStateEnv
from .memory import VisualNovelty
from .neural_world_model import ACTION_TO_INDEX, frame_tensor
from .pixels import Frame, signature_key


@dataclass(frozen=True)
class NeuralPlanningConfig:
    actions: Tuple[Action, ...] = (
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
        Action.A,
        Action.B,
        Action.START,
        Action.SELECT,
        Action.NOOP,
    )
    planning_depth: int = 3
    beam_width: int = 16
    verify_actions: int = 4
    action_frames: int = 4
    action_durations: Tuple[int, ...] = ()
    discount: float = 0.9
    uncertainty_weight: float = 1.0
    latent_change_weight: float = 0.35
    actual_novelty_weight: float = 1.0
    scene_novelty_weight: float = 0.75
    within_scene_novelty_floor: float = 0.25
    prediction_error_weight: float = 0.5
    actual_change_weight: float = 0.25
    action_coverage_weight: float = 0.35
    duration_coverage_weight: float = 0.2
    consecutive_repeat_weight: float = 0.5
    archive_capacity: int = 96
    visual_stagnation_visits: int = 8
    archive_max_age: int = 512
    autonomous_change_threshold: float = 0.00025
    action_equivalence_threshold: float = 0.0001
    autonomous_grace_decisions: int = 4
    delayed_return_min_length: int = 4
    delayed_return_credit_horizon: int = 48
    delayed_return_weight: float = 0.75
    informative_signature_bins: int = 4
    frontier_credit_horizon: int = 48
    frontier_discount: float = 0.94
    frontier_return_penalty: float = 2.0
    frontier_score_weight: float = 0.6
    frontier_origin_weight: float = 0.35
    abstraction_latent_rmse_threshold: float = 0.04
    behavioral_abstraction_rmse_threshold: float = 0.02
    behavioral_abstraction_min_shared_probes: int = 2
    behavioral_probe_count: int = 2


@dataclass(frozen=True)
class NeuralPlan:
    path: Tuple[Action, ...]
    durations: Tuple[int, ...]
    score: float
    uncertainty: float


@dataclass
class _LatentNode:
    path: Tuple[Action, ...]
    durations: Tuple[int, ...]
    latents: Tensor
    score: float
    uncertainty: float


@dataclass
class _ArchivedBranch:
    state: object
    frame: Frame
    plan: NeuralPlan
    score: float
    scene: str
    created: int
    origin_signature: str = ""
    frontier_signature: str = ""


@dataclass(frozen=True)
class _CommittedTransition:
    decision: int
    source_scene: str
    action: Action
    duration: int
    target_scene: str
    target_signature: str


@dataclass
class _FrontierTrace:
    start_decision: int
    signature: str
    discounted_return: float = 0.0
    next_discount: float = 1.0
    choice: Optional[Tuple[str, Action, int]] = None


@dataclass
class _VisualCluster:
    key: str
    scene: str
    centroid: Tensor
    count: int = 1


@dataclass
class _BehaviorCluster:
    key: str
    visual_cluster: str
    probe_centroids: Dict[Tuple[Action, int], Tensor]
    probe_counts: CounterType[Tuple[Action, int]]
    count: int = 1


class NeuralRolloutPlanner:
    def __init__(
        self,
        model: EnsembleVisualDynamicsModel,
        device: Union[torch.device, str],
        config: Optional[NeuralPlanningConfig] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.config = config or NeuralPlanningConfig()
        self.duration_choices = self.config.action_durations or (self.config.action_frames,)
        if any(duration <= 0 for duration in self.duration_choices):
            raise ValueError("planning action durations must be positive")
        if any(duration > model.max_action_frames for duration in self.duration_choices):
            raise ValueError("planning duration exceeds the model maximum")
        if len(self.duration_choices) > 1 and not model.duration_conditioned:
            raise ValueError("multiple action durations require a duration-conditioned model")
        if (
            not model.duration_conditioned
            and model.fixed_action_frames_locked
            and self.duration_choices != (model.fixed_action_frames,)
        ):
            raise ValueError(
                "fixed-duration checkpoint must use its recorded action duration"
            )
        self.model.to(self.device)

    @torch.no_grad()
    def plan(self, frame: Frame) -> List[NeuralPlan]:
        self.model.eval()
        source = frame_tensor(frame, self.device).unsqueeze(0)
        initial = self.model.initial_ensemble(source)[:, 0]
        frontier = [_LatentNode((), (), initial, 0.0, 0.0)]
        effective_beam_width = max(
            self.config.beam_width,
            len(self.config.actions) * len(self.duration_choices),
        )
        for depth in range(self.config.planning_depth):
            pairs = [
                (node, action, duration)
                for node in frontier
                for action in self.config.actions
                for duration in self.duration_choices
            ]
            latents = torch.stack([node.latents for node, _, _ in pairs], dim=1)
            actions = torch.tensor(
                [ACTION_TO_INDEX[action] for _, action, _ in pairs],
                dtype=torch.long,
                device=self.device,
            )
            durations = torch.tensor(
                [duration for _, _, duration in pairs],
                dtype=torch.long,
                device=self.device,
            )
            predicted = self.model.transition_ensemble(
                latents, actions, durations if self.model.duration_conditioned else None
            )
            means = predicted.mean(dim=0)
            parent_means = latents.mean(dim=0)
            uncertainty = predicted.var(dim=0, unbiased=False).mean(dim=1).sqrt()
            change = (means - parent_means).pow(2).mean(dim=1).sqrt()
            uncertainty_values = uncertainty.detach().cpu().tolist()
            change_values = change.detach().cpu().tolist()
            expanded = []
            for index, (node, action, duration) in enumerate(pairs):
                immediate = (
                    self.config.uncertainty_weight * uncertainty_values[index]
                    + self.config.latent_change_weight * change_values[index]
                    - 0.03
                    * sum(
                        prior_action == action and prior_duration == duration
                        for prior_action, prior_duration in zip(node.path, node.durations)
                    )
                )
                expanded.append(
                    _LatentNode(
                        node.path + (action,),
                        node.durations + (duration,),
                        predicted[:, index],
                        node.score + self.config.discount**depth * immediate,
                        node.uncertainty + uncertainty_values[index],
                    )
                )
            expanded.sort(
                key=lambda item: (
                    -item.score,
                    tuple((action.value, duration) for action, duration in zip(item.path, item.durations)),
                )
            )
            # Preserve at least one continuation per first action when the
            # beam allows it, so real verification compares alternatives
            # instead of receiving many variants of the same first move.
            selected = []
            first_actions = set()
            for item in expanded:
                first = (item.path[0], item.durations[0])
                if first not in first_actions:
                    selected.append(item)
                    first_actions.add(first)
                    if len(selected) == effective_beam_width:
                        break
            if len(selected) < effective_beam_width:
                selected_ids = {id(item) for item in selected}
                selected.extend(
                    item
                    for item in expanded
                    if id(item) not in selected_ids
                )
            frontier = selected[:effective_beam_width]
        return [
            NeuralPlan(node.path, node.durations, node.score, node.uncertainty)
            for node in frontier
        ]

    @torch.no_grad()
    def one_step_error(
        self, source: Frame, action: Action, duration: int, target: Frame
    ) -> float:
        source_tensor = frame_tensor(source, self.device).unsqueeze(0)
        target_tensor = frame_tensor(target, self.device).unsqueeze(0)
        latents = self.model.initial_ensemble(source_tensor)
        actions = torch.tensor([ACTION_TO_INDEX[action]], dtype=torch.long, device=self.device)
        durations = torch.tensor([duration], dtype=torch.long, device=self.device)
        predicted = self.model.transition_ensemble(
            latents, actions, durations if self.model.duration_conditioned else None
        ).mean(dim=0)
        predicted_pixels = self.model.decode(predicted)
        return float((predicted_pixels - target_tensor).abs().mean().cpu())


class VerifiedNeuralAgent:
    """Neural latent planning with real first-action branch verification."""

    def __init__(
        self,
        env: PixelSaveStateEnv,
        model: EnsembleVisualDynamicsModel,
        device: Union[torch.device, str],
        config: Optional[NeuralPlanningConfig] = None,
        event_logger: Optional[Any] = None,
    ) -> None:
        self.env = env
        self.model = model
        self.config = config or NeuralPlanningConfig()
        if self.config.frontier_credit_horizon <= 0:
            raise ValueError("frontier credit horizon must be positive")
        if not 0.0 < self.config.frontier_discount <= 1.0:
            raise ValueError("frontier discount must be in (0, 1]")
        if self.config.abstraction_latent_rmse_threshold < 0.0:
            raise ValueError("abstraction latent threshold must be non-negative")
        if self.config.behavioral_abstraction_rmse_threshold < 0.0:
            raise ValueError("behavioral abstraction threshold must be non-negative")
        if self.config.behavioral_abstraction_min_shared_probes <= 0:
            raise ValueError("behavioral abstraction requires a shared probe")
        if self.config.behavioral_probe_count <= 0:
            raise ValueError("behavioral probe count must be positive")
        self.model.freeze()
        self.planner = NeuralRolloutPlanner(model, device, self.config)
        self.novelty = VisualNovelty()
        self.frame: Optional[Frame] = None
        self.action_counts: CounterType[Action] = Counter()
        self.duration_counts: CounterType[int] = Counter()
        self.last_action: Optional[Action] = None
        self.last_duration: Optional[int] = None
        self.action_streak = 0
        self.scene_visits: CounterType[str] = Counter()
        self.scene_action_probes: CounterType[Tuple[str, Action]] = Counter()
        self.delayed_return_costs: CounterType[Tuple[str, Action, int]] = Counter()
        self.transition_history: List[_CommittedTransition] = []
        self.visual_last_visit: Dict[str, int] = {}
        self.delayed_return_recovery = False
        self.delayed_return_loop_start: Optional[int] = None
        self.autonomous_grace_remaining = 0
        self.frontier_values: Dict[str, float] = {}
        self.frontier_samples: CounterType[str] = Counter()
        self.frontier_traces: List[_FrontierTrace] = []
        self.frontier_choice_values: Dict[Tuple[str, Action, int], float] = {}
        self.frontier_choice_samples: CounterType[Tuple[str, Action, int]] = Counter()
        self.visual_clusters: List[_VisualCluster] = []
        self.frame_clusters: Dict[str, str] = {}
        self.frame_latents: Dict[str, Tensor] = {}
        self.cluster_serial = 0
        self.behavior_clusters: List[_BehaviorCluster] = []
        self.behavior_cluster_serial = 0
        self.provisional_state_serial = 0
        self.current_frontier_signature = ""
        self.current_scene: Optional[str] = None
        self.scene_streak = 0
        self.visual_stagnation_streak = 0
        self.archive: List[_ArchivedBranch] = []
        self.decision_index = 0
        self.event_logger = event_logger

    def _emit(self, event_type: str, **fields: Any) -> None:
        if self.event_logger is not None:
            self.event_logger.log(event_type, **fields)

    def _frame_fields(self, frame: Frame) -> Dict[str, Any]:
        if self.event_logger is not None and hasattr(self.event_logger, "frame_fields"):
            return self.event_logger.frame_fields(frame)
        return {"frame": frame.digest}

    def _state_id(self, state: object) -> Optional[str]:
        state_id = getattr(self.env, "state_id", None)
        return None if state_id is None else state_id(state)

    @staticmethod
    def _signature(frame: Frame) -> str:
        return signature_key(frame.coarse_signature())

    @staticmethod
    def _scene_signature(frame: Frame) -> str:
        return signature_key(frame.coarse_signature(columns=3, rows=3))

    def reset(self) -> Frame:
        self.clear_archive()
        self.frame = self.env.reset()
        self.novelty = VisualNovelty()
        self.novelty.observe(self._signature(self.frame))
        self.action_counts = Counter()
        self.duration_counts = Counter()
        self.last_action = None
        self.last_duration = None
        self.action_streak = 0
        self.scene_visits = Counter()
        self.scene_action_probes = Counter()
        self.delayed_return_costs = Counter()
        self.transition_history = []
        self.visual_last_visit = {self._signature(self.frame): 0}
        self.delayed_return_recovery = False
        self.delayed_return_loop_start = None
        self.autonomous_grace_remaining = 0
        self.visual_clusters = []
        self.frame_clusters = {}
        self.frame_latents = {}
        self.cluster_serial = 0
        self.behavior_clusters = []
        self.behavior_cluster_serial = 0
        self.provisional_state_serial = 0
        self._abstract_signature(self.frame)
        initial_signature = self._new_provisional_signature()
        self.current_frontier_signature = initial_signature
        self.frontier_values = {}
        self.frontier_samples = Counter()
        self.frontier_traces = [_FrontierTrace(0, initial_signature)]
        self.frontier_choice_values = {}
        self.frontier_choice_samples = Counter()
        self.current_scene = self._scene_signature(self.frame)
        self.scene_visits[self.current_scene] += 1
        self.scene_streak = 1
        self.visual_stagnation_streak = 0
        self.archive = []
        self.decision_index = 0
        self._emit(
            "agent_reset",
            decision=0,
            action_counts={},
            **self._frame_fields(self.frame),
        )
        return self.frame

    def clear_archive(self) -> None:
        release_state = getattr(self.env, "release_state", None)
        if release_state is not None:
            for branch in getattr(self, "archive", []):
                try:
                    self._emit(
                        "archive_branch_removed",
                        reason="agent_reset_or_close",
                        state_id=self._state_id(branch.state),
                        created_decision=branch.created,
                    )
                    release_state(branch.state)
                except Exception:
                    pass
        self.archive = []

    def _action_penalty(self, action: Action, duration: Optional[int] = None) -> float:
        duration = self.config.action_frames if duration is None else duration
        coverage = self.config.action_coverage_weight * math.sqrt(self.action_counts[action])
        duration_coverage = self.config.duration_coverage_weight * math.sqrt(
            self.duration_counts[duration]
        )
        consecutive = (
            self.config.consecutive_repeat_weight * self.action_streak
            if action == self.last_action and duration == self.last_duration
            else 0.0
        )
        return_penalty = 0.0
        if self.current_scene is not None:
            return_penalty = self.config.delayed_return_weight * math.sqrt(
                self.delayed_return_costs[(self.current_scene, action, duration)]
            )
        return coverage + duration_coverage + consecutive + return_penalty

    def _informative_signature(self, frame: Frame) -> bool:
        signature = frame.coarse_signature()
        return (
            len(set(signature)) >= self.config.informative_signature_bins
            and max(signature) - min(signature) >= 3
        )

    @torch.no_grad()
    def _frame_latent(self, frame: Frame) -> Tensor:
        cached = self.frame_latents.get(frame.digest)
        if cached is not None:
            return cached
        latent = (
            self.model.encode(
                frame_tensor(frame, self.planner.device).unsqueeze(0)
            )[0]
            .detach()
            .to(device="cpu")
        )
        self.frame_latents[frame.digest] = latent
        return latent

    @torch.no_grad()
    def _abstract_signature(self, frame: Frame) -> str:
        cached = self.frame_clusters.get(frame.digest)
        if cached is not None:
            return cached
        latent = self._frame_latent(frame)
        scene = self._scene_signature(frame)
        candidates = [
            cluster for cluster in self.visual_clusters if cluster.scene == scene
        ]
        nearest = None
        distance = math.inf
        for cluster in candidates:
            candidate_distance = float(
                (latent - cluster.centroid).pow(2).mean().sqrt()
            )
            if candidate_distance < distance:
                nearest = cluster
                distance = candidate_distance
        created = (
            nearest is None
            or distance > self.config.abstraction_latent_rmse_threshold
        )
        if created:
            self.cluster_serial += 1
            nearest = _VisualCluster(
                f"latent-cluster-{self.cluster_serial:06d}",
                scene,
                latent.clone(),
            )
            self.visual_clusters.append(nearest)
        else:
            assert nearest is not None
            nearest.count += 1
            nearest.centroid += (latent - nearest.centroid) / nearest.count
        assert nearest is not None
        self.frame_clusters[frame.digest] = nearest.key
        self._emit(
            "visual_abstraction_assigned",
            cluster=nearest.key,
            cluster_created=created,
            cluster_size=nearest.count,
            scene=scene,
            latent_rmse_to_centroid=None if created else distance,
            threshold=self.config.abstraction_latent_rmse_threshold,
            exact_signature=self._signature(frame),
            **self._frame_fields(frame),
        )
        return nearest.key

    def _new_provisional_signature(self) -> str:
        self.provisional_state_serial += 1
        return f"provisional-state-{self.provisional_state_serial:08d}"

    @staticmethod
    def _fallback_frontier_signature(frame: Frame) -> str:
        return f"unprofiled-frame-{frame.digest}"

    def _behavior_probe_keys(self) -> Tuple[Tuple[Action, int], ...]:
        actions = []
        if Action.NOOP in self.config.actions:
            actions.append(Action.NOOP)
        actions.extend(action for action in self.config.actions if action != Action.NOOP)
        duration = max(self.planner.duration_choices)
        return tuple(
            (action, duration)
            for action in actions[: self.config.behavioral_probe_count]
        )

    def _merge_frontier_state_value(self, source: str, target: str) -> None:
        source_count = self.frontier_samples.pop(source, 0)
        if not source_count:
            self.frontier_values.pop(source, None)
            return
        target_count = self.frontier_samples.get(target, 0)
        source_value = self.frontier_values.pop(source, 0.0)
        target_value = self.frontier_values.get(target, 0.0)
        combined_count = source_count + target_count
        self.frontier_samples[target] = combined_count
        self.frontier_values[target] = (
            source_count * source_value + target_count * target_value
        ) / combined_count

    def _migrate_frontier_signature(self, source: str, target: str) -> None:
        if source == target:
            return
        self._merge_frontier_state_value(source, target)
        migrated_choices = 0
        for choice in list(self.frontier_choice_samples):
            signature, action, duration = choice
            if signature != source:
                continue
            target_choice = (target, action, duration)
            source_count = self.frontier_choice_samples.pop(choice)
            target_count = self.frontier_choice_samples.get(target_choice, 0)
            source_value = self.frontier_choice_values.pop(choice, 0.0)
            target_value = self.frontier_choice_values.get(target_choice, 0.0)
            combined_count = source_count + target_count
            self.frontier_choice_samples[target_choice] = combined_count
            self.frontier_choice_values[target_choice] = (
                source_count * source_value + target_count * target_value
            ) / combined_count
            migrated_choices += 1
        migrated_traces = 0
        for trace in self.frontier_traces:
            if trace.signature == source:
                trace.signature = target
                migrated_traces += 1
            if trace.choice is not None and trace.choice[0] == source:
                trace.choice = (target, trace.choice[1], trace.choice[2])
        migrated_origins = 0
        for branch in self.archive:
            if branch.origin_signature == source:
                branch.origin_signature = target
                migrated_origins += 1
        self._emit(
            "frontier_signature_migrated",
            decision=self.decision_index + 1,
            source_signature=source,
            target_signature=target,
            migrated_choice_values=migrated_choices,
            migrated_traces=migrated_traces,
            migrated_archive_origins=migrated_origins,
        )

    @torch.no_grad()
    def _behavioral_signature(
        self,
        frame: Frame,
        outcomes: Dict[Tuple[Action, int], Frame],
        provisional_signature: str,
    ) -> str:
        source_latent = self._frame_latent(frame)
        requested = self._behavior_probe_keys()
        profile = {
            probe: self._frame_latent(outcomes[probe]) - source_latent
            for probe in requested
            if probe in outcomes
        }
        visual_cluster = self._abstract_signature(frame)
        if len(profile) < self.config.behavioral_abstraction_min_shared_probes:
            self._emit(
                "behavioral_abstraction_deferred",
                decision=self.decision_index + 1,
                provisional_signature=provisional_signature,
                visual_cluster=visual_cluster,
                observed_probes=[
                    {"action": action, "action_frames": duration}
                    for action, duration in profile
                ],
                required_shared_probes=(
                    self.config.behavioral_abstraction_min_shared_probes
                ),
                **self._frame_fields(frame),
            )
            return provisional_signature

        nearest = None
        nearest_distance = math.inf
        nearest_probe_distances: Dict[Tuple[Action, int], float] = {}
        for cluster in self.behavior_clusters:
            if cluster.visual_cluster != visual_cluster:
                continue
            shared = sorted(set(profile) & set(cluster.probe_centroids))
            if len(shared) < self.config.behavioral_abstraction_min_shared_probes:
                continue
            probe_distances = {
                probe: float(
                    (profile[probe] - cluster.probe_centroids[probe])
                    .pow(2)
                    .mean()
                    .sqrt()
                )
                for probe in shared
            }
            distance = sum(probe_distances.values()) / len(probe_distances)
            if distance < nearest_distance:
                nearest = cluster
                nearest_distance = distance
                nearest_probe_distances = probe_distances
        created = (
            nearest is None
            or nearest_distance
            > self.config.behavioral_abstraction_rmse_threshold
        )
        if created:
            self.behavior_cluster_serial += 1
            nearest = _BehaviorCluster(
                key=f"behavior-cluster-{self.behavior_cluster_serial:06d}",
                visual_cluster=visual_cluster,
                probe_centroids={probe: value.clone() for probe, value in profile.items()},
                probe_counts=Counter({probe: 1 for probe in profile}),
            )
            self.behavior_clusters.append(nearest)
        else:
            assert nearest is not None
            nearest.count += 1
            for probe, value in profile.items():
                count = nearest.probe_counts[probe] + 1
                centroid = nearest.probe_centroids.get(probe)
                nearest.probe_centroids[probe] = (
                    value.clone()
                    if centroid is None
                    else centroid + (value - centroid) / count
                )
                nearest.probe_counts[probe] = count
        assert nearest is not None
        self._migrate_frontier_signature(provisional_signature, nearest.key)
        self._emit(
            "behavioral_abstraction_assigned",
            decision=self.decision_index + 1,
            cluster=nearest.key,
            cluster_created=created,
            cluster_size=nearest.count,
            visual_cluster=visual_cluster,
            provisional_signature=provisional_signature,
            successor_delta_rmse=None if created else nearest_distance,
            maximum_probe_rmse=(
                None
                if created
                else max(nearest_probe_distances.values(), default=0.0)
            ),
            threshold=self.config.behavioral_abstraction_rmse_threshold,
            matched_probes=[
                {
                    "action": action,
                    "action_frames": duration,
                    "successor_delta_rmse": nearest_probe_distances.get(
                        (action, duration)
                    ),
                }
                for action, duration in profile
            ],
            exact_signature=self._signature(frame),
            **self._frame_fields(frame),
        )
        return nearest.key

    def _frontier_estimate(self, signature: str) -> float:
        completed = self.frontier_values.get(signature, 0.0)
        provisional = max(
            (
                trace.discounted_return
                for trace in self.frontier_traces
                if trace.signature == signature
            ),
            default=0.0,
        )
        return completed + provisional

    def _record_frontier_sample(self, signature: str, sample: float) -> float:
        count = self.frontier_samples[signature] + 1
        previous = self.frontier_values.get(signature, 0.0)
        value = previous + (sample - previous) / count
        self.frontier_samples[signature] = count
        self.frontier_values[signature] = value
        return value

    def _choice_frontier_estimate(
        self, signature: str, action: Action, duration: int
    ) -> Tuple[float, bool]:
        choice = (signature, action, duration)
        completed = self.frontier_choice_values.get(choice, 0.0)
        active = [
            trace.discounted_return
            for trace in self.frontier_traces
            if trace.choice == choice
        ]
        return completed + max(active, default=0.0), bool(
            self.frontier_choice_samples[choice] or active
        )

    def _record_frontier_trace_sample(
        self, trace: _FrontierTrace, sample: float
    ) -> float:
        if trace.choice is None:
            return self._record_frontier_sample(trace.signature, sample)
        count = self.frontier_choice_samples[trace.choice] + 1
        previous = self.frontier_choice_values.get(trace.choice, 0.0)
        value = previous + (sample - previous) / count
        self.frontier_choice_samples[trace.choice] = count
        self.frontier_choice_values[trace.choice] = value
        return value

    def _update_persistent_frontier(
        self,
        target_signature: str,
        reward: float,
        source_signature: Optional[str] = None,
        action: Optional[Action] = None,
        duration: Optional[int] = None,
    ) -> None:
        retained = []
        credited = []
        completed = []
        for trace in self.frontier_traces:
            trace.discounted_return += trace.next_discount * reward
            trace.next_discount *= self.config.frontier_discount
            credited.append(
                {
                    "start_decision": trace.start_decision,
                    "signature": trace.signature,
                    "choice": trace.choice,
                    "provisional_return": trace.discounted_return,
                }
            )
            if self.decision_index - trace.start_decision >= self.config.frontier_credit_horizon:
                value = self._record_frontier_trace_sample(
                    trace, trace.discounted_return
                )
                completed.append(
                    {
                        "start_decision": trace.start_decision,
                        "signature": trace.signature,
                        "choice": trace.choice,
                        "sample": trace.discounted_return,
                        "value": value,
                    }
                )
            else:
                retained.append(trace)
        choice = None
        if source_signature is not None and action is not None and duration is not None:
            choice = (source_signature, action, duration)
            retained.append(
                _FrontierTrace(
                    self.decision_index,
                    source_signature,
                    reward,
                    self.config.frontier_discount,
                    choice,
                )
            )
        retained.append(_FrontierTrace(self.decision_index, target_signature))
        self.frontier_traces = retained
        self._emit(
            "persistent_frontier_updated",
            decision=self.decision_index,
            target_signature=target_signature,
            successor_novelty_reward=reward,
            credited_traces=credited,
            completed_samples=completed,
            target_frontier_value=self._frontier_estimate(target_signature),
            committed_choice=choice,
            committed_choice_frontier_value=(
                self._choice_frontier_estimate(*choice)[0]
                if choice is not None
                else None
            ),
        )

    def _penalize_frontier_loop(self, previous_decision: int) -> None:
        retained = []
        penalized = []
        for trace in self.frontier_traces:
            if trace.start_decision < previous_decision:
                retained.append(trace)
                continue
            value = self._record_frontier_trace_sample(
                trace, -self.config.frontier_return_penalty
            )
            penalized.append(
                {
                    "start_decision": trace.start_decision,
                    "signature": trace.signature,
                    "choice": trace.choice,
                    "discarded_provisional_return": trace.discounted_return,
                    "sample": -self.config.frontier_return_penalty,
                    "value": value,
                }
            )
        self.frontier_traces = retained
        self._emit(
            "persistent_frontier_return_penalized",
            decision=self.decision_index,
            previous_decision=previous_decision,
            penalized_traces=penalized,
        )

    def _restart_frontier_trace(self, signature: str, reason: str) -> None:
        discarded = [
            {
                "start_decision": trace.start_decision,
                "signature": trace.signature,
                "choice": trace.choice,
                "provisional_return": trace.discounted_return,
            }
            for trace in self.frontier_traces
        ]
        self.frontier_traces = [_FrontierTrace(self.decision_index, signature)]
        self._emit(
            "persistent_frontier_trace_restarted",
            decision=self.decision_index,
            reason=reason,
            discarded_traces=discarded,
            signature=signature,
            frontier_value=self._frontier_estimate(signature),
        )

    def _archive_frontier_score(self, branch: _ArchivedBranch) -> float:
        own_value = self._frontier_estimate(
            branch.frontier_signature
            or self._fallback_frontier_signature(branch.frame)
        )
        origin_value = (
            self._frontier_estimate(branch.origin_signature)
            if branch.origin_signature
            else 0.0
        )
        choice_value, choice_is_known = self._choice_frontier_estimate(
            branch.origin_signature,
            branch.plan.path[0],
            branch.plan.durations[0],
        )
        if choice_is_known:
            return choice_value
        return max(own_value, self.config.frontier_origin_weight * origin_value)

    def _record_delayed_return(
        self,
        source_scene: str,
        action: Action,
        duration: int,
        target: Frame,
        target_scene: str,
        frontier_signature: Optional[str] = None,
    ) -> None:
        signature = self._signature(target)
        transition = _CommittedTransition(
            self.decision_index,
            source_scene,
            action,
            duration,
            target_scene,
            signature,
        )
        self.transition_history.append(transition)
        previous = self.visual_last_visit.get(signature)
        self.visual_last_visit[signature] = self.decision_index
        if (
            previous is None
            or self.decision_index - previous < self.config.delayed_return_min_length
            or not self._informative_signature(target)
        ):
            return
        start_decision = max(
            previous + 1,
            self.decision_index - self.config.delayed_return_credit_horizon + 1,
        )
        credited = [
            item for item in self.transition_history if item.decision >= start_decision
        ]
        for item in credited:
            self.delayed_return_costs[
                (item.source_scene, item.action, item.duration)
            ] += 1
        self.delayed_return_recovery = True
        self.delayed_return_loop_start = previous
        self._penalize_frontier_loop(previous)
        self.frontier_traces.append(
            _FrontierTrace(
                self.decision_index, frontier_signature or signature
            )
        )
        self._emit(
            "delayed_visual_return_detected",
            decision=self.decision_index,
            returned_signature=signature,
            previous_decision=previous,
            loop_length=self.decision_index - previous,
            credited_decisions=[item.decision for item in credited],
            credited_choices=[
                {
                    "source_scene": item.source_scene,
                    "action": item.action,
                    "duration": item.duration,
                }
                for item in credited
            ],
        )

    def _add_control_probes(
        self,
        ranked: List[NeuralPlan],
        best_by_action: Dict[Tuple[Action, int], NeuralPlan],
    ) -> List[NeuralPlan]:
        probe_keys = self._behavior_probe_keys()
        if self.config.verify_actions < len(probe_keys):
            return ranked
        probes = [best_by_action[key] for key in probe_keys if key in best_by_action]
        required_actions = {probe.path[0] for probe in probes}
        result = list(ranked)
        for probe in probes:
            matching_index = next(
                (
                    index
                    for index, item in enumerate(result)
                    if item.path[0] == probe.path[0]
                ),
                None,
            )
            if matching_index is not None:
                result[matching_index] = probe
                continue
            if len(result) >= self.config.verify_actions:
                removable_index = next(
                    (
                        index
                        for index in range(len(result) - 1, -1, -1)
                        if result[index].path[0] not in required_actions
                    ),
                    len(result) - 1,
                )
                result.pop(removable_index)
            result.append(probe)
        return result

    def _autonomous_choice(
        self, source: Frame, verified: List[Tuple[Any, ...]]
    ) -> Optional[Tuple[Tuple[Any, ...], float, float]]:
        qualified = []
        for duration in self.planner.duration_choices:
            group = [item for item in verified if item[1].durations[0] == duration]
            if len({item[1].path[0] for item in group}) < 2:
                continue
            spread = max(
                left[3].mean_absolute_difference(right[3])
                for index, left in enumerate(group)
                for right in group[index + 1 :]
            )
            change = sum(source.mean_absolute_difference(item[3]) for item in group) / len(
                group
            )
            if (
                spread <= self.config.action_equivalence_threshold
                and change >= self.config.autonomous_change_threshold
            ):
                preferred = next(
                    (item for item in group if item[1].path[0] == Action.NOOP),
                    max(group, key=lambda item: item[0]),
                )
                qualified.append((duration, preferred, spread, change))
        if not qualified:
            return None
        _duration, preferred, spread, change = max(qualified, key=lambda item: item[0])
        return preferred, spread, change

    def decide(self) -> Decision:
        if self.frame is None:
            self.reset()
        assert self.frame is not None
        self._emit(
            "decision_started",
            decision=self.decision_index + 1,
            action_counts=self.action_counts,
            duration_counts=self.duration_counts,
            last_action=self.last_action,
            last_duration=self.last_duration,
            action_streak=self.action_streak,
            scene_streak=self.scene_streak,
            visual_stagnation_streak=self.visual_stagnation_streak,
            visual_stagnation_limit=self.config.visual_stagnation_visits,
            frontier_signature=self.current_frontier_signature,
            scene_visits=self.scene_visits,
            archive_size=len(self.archive),
            **self._frame_fields(self.frame),
        )
        restored = self._restore_if_stagnant()
        if restored is not None:
            return restored
        plans = self.planner.plan(self.frame)
        self._emit(
            "planner_candidates",
            decision=self.decision_index + 1,
            candidates=[
                {
                    "rank": rank,
                    "path": plan.path,
                    "durations": plan.durations,
                    "model_score": plan.score,
                    "uncertainty": plan.uncertainty,
                    "first_action_penalty": self._action_penalty(
                        plan.path[0], plan.durations[0]
                    ),
                }
                for rank, plan in enumerate(plans, 1)
            ],
        )
        best_by_action: Dict[Tuple[Action, int], NeuralPlan] = {}
        for plan in plans:
            timed_action = (plan.path[0], plan.durations[0])
            if (
                timed_action not in best_by_action
                or plan.score > best_by_action[timed_action].score
            ):
                best_by_action[timed_action] = plan
        current_scene = self._scene_signature(self.frame)
        source_signature = self.current_frontier_signature
        best_by_button: Dict[Action, NeuralPlan] = {}
        for plan in best_by_action.values():
            action = plan.path[0]
            existing = best_by_button.get(action)
            adjusted = plan.score - self._action_penalty(action, plan.durations[0])
            if existing is None or adjusted > existing.score - self._action_penalty(
                existing.path[0], existing.durations[0]
            ):
                best_by_button[action] = plan
        ranked = sorted(
            best_by_button.values(),
            key=lambda plan: (
                self.scene_action_probes[(current_scene, plan.path[0])],
                -(plan.score - self._action_penalty(plan.path[0], plan.durations[0])),
                tuple(
                    (action.value, duration)
                    for action, duration in zip(plan.path, plan.durations)
                ),
            ),
        )[: self.config.verify_actions]
        if len(ranked) < self.config.verify_actions:
            selected = {(plan.path[0], plan.durations[0]) for plan in ranked}
            remaining = sorted(
                (
                    plan
                    for key, plan in best_by_action.items()
                    if key not in selected
                ),
                key=lambda plan: -(
                    plan.score
                    - self._action_penalty(plan.path[0], plan.durations[0])
                ),
            )
            ranked.extend(remaining[: self.config.verify_actions - len(ranked)])
        ranked = self._add_control_probes(ranked, best_by_action)
        if not ranked:
            raise RuntimeError("neural planner produced no action candidates")

        root = self.env.save_state()
        states = [root]
        raw_verified = []
        release_state = getattr(self.env, "release_state", None)
        try:
            for candidate_rank, plan in enumerate(ranked, 1):
                self.env.load_state(root)
                duration = plan.durations[0]
                target = self.env.step(plan.path[0], duration)
                self.scene_action_probes[(current_scene, plan.path[0])] += 1
                state = self.env.save_state()
                states.append(state)
                novelty = self.novelty.score(self._signature(target))
                target_scene = self._scene_signature(target)
                target_signature_is_new = (
                    self.novelty.count(self._signature(target)) == 0
                )
                target_scene_is_new = self.scene_visits[target_scene] == 0
                scene_novelty = 1.0 / math.sqrt(self.scene_visits[target_scene] + 1)
                effective_novelty = novelty * (
                    self.config.within_scene_novelty_floor
                    + (1.0 - self.config.within_scene_novelty_floor) * scene_novelty
                )
                error = self.planner.one_step_error(
                    self.frame, plan.path[0], duration, target
                )
                visual_change = self.frame.mean_absolute_difference(target)
                target_visual_cluster = self._abstract_signature(target)
                target_frontier_signature = self._new_provisional_signature()
                raw_verified.append(
                    (
                        plan,
                        state,
                        target,
                        novelty,
                        error,
                        visual_change,
                        target_scene,
                        target_signature_is_new,
                        target_scene_is_new,
                        scene_novelty,
                        effective_novelty,
                        candidate_rank,
                        self.scene_action_probes[(current_scene, plan.path[0])],
                        getattr(self.env, "last_step_seq", None),
                        getattr(self.env, "last_state_event_seq", None),
                        target_visual_cluster,
                        target_frontier_signature,
                    )
                )

            probe_outcomes = {
                (item[0].path[0], item[0].durations[0]): item[2]
                for item in raw_verified
            }
            source_signature = self._behavioral_signature(
                self.frame,
                probe_outcomes,
                source_signature,
            )
            self.current_frontier_signature = source_signature
            verified = []
            for (
                plan,
                state,
                target,
                novelty,
                error,
                visual_change,
                target_scene,
                target_signature_is_new,
                target_scene_is_new,
                scene_novelty,
                effective_novelty,
                candidate_rank,
                scene_action_probe_count,
                env_step_seq,
                state_save_seq,
                target_visual_cluster,
                target_frontier_signature,
            ) in raw_verified:
                duration = plan.durations[0]
                persistent_frontier_value = self._frontier_estimate(
                    target_frontier_signature
                )
                choice_frontier_value, choice_frontier_is_known = (
                    self._choice_frontier_estimate(
                        source_signature, plan.path[0], duration
                    )
                )
                if choice_frontier_is_known:
                    persistent_frontier_value = choice_frontier_value
                score = (
                    plan.score
                    + self.config.actual_novelty_weight * effective_novelty
                    + self.config.scene_novelty_weight * scene_novelty
                    + self.config.prediction_error_weight * error
                    + self.config.actual_change_weight * visual_change
                    + self.config.frontier_score_weight * persistent_frontier_value
                    - self._action_penalty(plan.path[0], duration)
                )
                verified.append(
                    (
                        score,
                        plan,
                        state,
                        target,
                        novelty,
                        error,
                        visual_change,
                        target_frontier_signature,
                    )
                )
                self._emit(
                    "branch_verified",
                    decision=self.decision_index + 1,
                    branch_id=f"decision-{self.decision_index + 1:08d}-branch-{candidate_rank:02d}",
                    candidate_rank=candidate_rank,
                    scene_action_probe_count=scene_action_probe_count,
                    env_step_seq=env_step_seq,
                    state_save_seq=state_save_seq,
                    action=plan.path[0],
                    action_frames=duration,
                    path=plan.path,
                    durations=plan.durations,
                    model_score=plan.score,
                    model_uncertainty=plan.uncertainty,
                    novelty=novelty,
                    effective_novelty=effective_novelty,
                    scene_novelty=scene_novelty,
                    target_signature_is_new=target_signature_is_new,
                    target_scene_is_new=target_scene_is_new,
                    target_scene=target_scene,
                    prediction_error=error,
                    visual_change=visual_change,
                    persistent_frontier_value=persistent_frontier_value,
                    choice_frontier_value=choice_frontier_value,
                    choice_frontier_is_known=choice_frontier_is_known,
                    abstract_signature=target_visual_cluster,
                    source_behavioral_signature=source_signature,
                    target_frontier_signature=target_frontier_signature,
                    action_penalty=self._action_penalty(plan.path[0], duration),
                    combined_score=score,
                    state_id=self._state_id(state),
                    **self._frame_fields(target),
                )
            autonomous = self._autonomous_choice(self.frame, verified)
            if autonomous is not None:
                chosen, outcome_spread, autonomous_change = autonomous
                self.autonomous_grace_remaining = self.config.autonomous_grace_decisions
                self._emit(
                    "autonomous_dynamics_detected",
                    decision=self.decision_index + 1,
                    selected_action=chosen[1].path[0],
                    selected_duration=chosen[1].durations[0],
                    outcome_spread=outcome_spread,
                    autonomous_change=autonomous_change,
                )
            elif self.autonomous_grace_remaining > 0:
                neutral = [
                    item
                    for item in verified
                    if item[1].path[0] == Action.NOOP
                ]
                if neutral:
                    chosen = max(
                        neutral,
                        key=lambda item: (item[1].durations[0], item[0]),
                    )
                    self.autonomous_grace_remaining -= 1
                    self._emit(
                        "autonomous_grace_wait",
                        decision=self.decision_index + 1,
                        selected_duration=chosen[1].durations[0],
                        grace_remaining=self.autonomous_grace_remaining,
                    )
                else:
                    self.autonomous_grace_remaining = 0
                    chosen = max(
                        verified,
                        key=lambda item: (
                            item[0],
                            tuple(
                                (action.value, duration)
                                for action, duration in zip(
                                    item[1].path, item[1].durations
                                )
                            ),
                        ),
                    )
            else:
                chosen = max(
                    verified,
                    key=lambda item: (
                        item[0],
                        tuple(
                            (action.value, duration)
                            for action, duration in zip(
                                item[1].path, item[1].durations
                            )
                        ),
                    ),
                )
            (
                score,
                plan,
                state,
                target,
                _chosen_novelty,
                _error,
                _visual_change,
                target_frontier_signature,
            ) = chosen
            self.env.load_state(state)
            self.frame = target
            target_signature = self._signature(target)
            target_visual_cluster = self._abstract_signature(target)
            self.current_frontier_signature = target_frontier_signature
            target_scene = self._scene_signature(target)
            target_signature_is_new = self.novelty.count(target_signature) == 0
            target_scene_is_new = self.scene_visits[target_scene] == 0
            self.novelty.observe(target_signature)
            action = plan.path[0]
            duration = plan.durations[0]
            self.action_counts[action] += 1
            self.duration_counts[duration] += 1
            self.action_streak = (
                self.action_streak + 1
                if action == self.last_action and duration == self.last_duration
                else 1
            )
            self.last_action = action
            self.last_duration = duration
            self.decision_index += 1
            frontier_reward = (
                float(target_signature_is_new)
                + self.config.scene_novelty_weight * float(target_scene_is_new)
            )
            self._update_persistent_frontier(
                target_frontier_signature,
                frontier_reward,
                source_signature,
                action,
                duration,
            )
            self._record_delayed_return(
                current_scene,
                action,
                duration,
                target,
                target_scene,
                target_frontier_signature,
            )
            self.scene_visits[target_scene] += 1
            self.visual_stagnation_streak = (
                0
                if target_signature_is_new
                else self.visual_stagnation_streak + 1
            )
            if target_scene == self.current_scene:
                self.scene_streak += 1
            else:
                self.current_scene = target_scene
                self.scene_streak = 1
            added = 0
            for (
                alternative_score,
                alternative_plan,
                alternative_state,
                alternative_frame,
                _alternative_novelty,
                _alternative_error,
                _alternative_change,
                alternative_frontier_signature,
            ) in verified:
                if alternative_state == state:
                    continue
                if autonomous is not None:
                    continue
                if any(
                    branch.frame.digest == alternative_frame.digest
                    for branch in self.archive
                ):
                    continue
                self.archive.append(
                    _ArchivedBranch(
                        alternative_state,
                        alternative_frame,
                        alternative_plan,
                        alternative_score,
                        self._scene_signature(alternative_frame),
                        self.decision_index,
                        source_signature,
                        alternative_frontier_signature,
                    )
                )
                added += 1
                archive_frontier_value = self._archive_frontier_score(self.archive[-1])
                self._emit(
                    "archive_branch_added",
                    decision=self.decision_index,
                    state_id=self._state_id(alternative_state),
                    action=alternative_plan.path[0],
                    action_frames=alternative_plan.durations[0],
                    path=alternative_plan.path,
                    durations=alternative_plan.durations,
                    score=alternative_score,
                    scene=self._scene_signature(alternative_frame),
                    origin_signature=source_signature,
                    frontier_signature=alternative_frontier_signature,
                    persistent_frontier_value=archive_frontier_value,
                    **self._frame_fields(alternative_frame),
                )
            self._prune_archive()
            self._emit(
                "decision_committed",
                decision=self.decision_index,
                action=plan.path[0],
                action_frames=duration,
                path=plan.path,
                durations=plan.durations,
                score=score,
                model_score=plan.score,
                model_uncertainty=plan.uncertainty,
                branches_examined=len(verified),
                restored_archive=False,
                committed_state_id=self._state_id(state),
                archive_branches_added=added,
                archive_size=len(self.archive),
                autonomous_dynamics=autonomous is not None,
                autonomous_grace_remaining=self.autonomous_grace_remaining,
                delayed_return_recovery_pending=self.delayed_return_recovery,
                persistent_frontier_reward=frontier_reward,
                target_signature_is_new=target_signature_is_new,
                target_scene_is_new=target_scene_is_new,
                persistent_frontier_value=self._frontier_estimate(
                    target_frontier_signature
                ),
                abstract_signature=target_visual_cluster,
                source_behavioral_signature=source_signature,
                target_frontier_signature=target_frontier_signature,
                committed_choice_frontier_value=self._choice_frontier_estimate(
                    source_signature, action, duration
                )[0],
                action_counts=self.action_counts,
                duration_counts=self.duration_counts,
                scene_streak=self.scene_streak,
                visual_stagnation_streak=self.visual_stagnation_streak,
                **self._frame_fields(target),
            )
            return Decision(
                plan.path[0],
                target,
                plan.path,
                score,
                len(verified),
                action_frames=duration,
                planned_durations=plan.durations,
            )
        finally:
            if release_state is not None:
                archived_states = {id(branch.state) for branch in self.archive}
                for candidate in states:
                    if id(candidate) not in archived_states:
                        release_state(candidate)

    def _restore_if_stagnant(self) -> Optional[Decision]:
        assert self.frame is not None
        current_scene = self._scene_signature(self.frame)
        delayed_return = self.delayed_return_recovery
        if (
            not delayed_return
            and self.visual_stagnation_streak
            < self.config.visual_stagnation_visits
        ):
            return None
        recovery_reason = (
            "delayed_visual_return" if delayed_return else "visual_stagnation"
        )
        if delayed_return:
            loop_start = self.delayed_return_loop_start or 0
            current_signature = self._signature(self.frame)
            eligible = [
                branch
                for branch in self.archive
                if branch.created >= loop_start
                and self._signature(branch.frame) != current_signature
            ]
            if not eligible:
                eligible = [
                    branch
                    for branch in self.archive
                    if self._signature(branch.frame) != current_signature
                ]
        else:
            minimum_created = max(0, self.decision_index - self.config.archive_max_age)
            eligible = [
                branch
                for branch in self.archive
                if branch.scene != current_scene and branch.created >= minimum_created
            ]
        if not eligible:
            if delayed_return:
                self._emit(
                    "delayed_return_recovery_unavailable",
                    decision=self.decision_index,
                    loop_start=self.delayed_return_loop_start,
                    archive_size=len(self.archive),
                    **self._frame_fields(self.frame),
                )
                self.delayed_return_recovery = False
                self.delayed_return_loop_start = None
            return None
        branch = max(
            eligible,
            key=lambda item: (
                self._archive_frontier_score(item),
                self.novelty.score(self._signature(item.frame)),
                item.created,
                item.score,
            ),
        )
        self.archive.remove(branch)
        restored_state_id = self._state_id(branch.state)
        self.env.load_state(branch.state)
        release_state = getattr(self.env, "release_state", None)
        if release_state is not None:
            release_state(branch.state)
        self.frame = branch.frame
        self.novelty.observe(self._signature(branch.frame))
        self.scene_visits[branch.scene] += 1
        self.current_scene = branch.scene
        self.scene_streak = 1
        self.visual_stagnation_streak = 0
        self.decision_index += 1
        self.visual_last_visit[self._signature(branch.frame)] = self.decision_index
        selected_frontier_value = self._archive_frontier_score(branch)
        restored_visual_cluster = self._abstract_signature(branch.frame)
        restored_frontier_signature = (
            branch.frontier_signature
            or self._fallback_frontier_signature(branch.frame)
        )
        self.current_frontier_signature = restored_frontier_signature
        self._restart_frontier_trace(
            restored_frontier_signature, recovery_reason
        )
        self.delayed_return_recovery = False
        self.delayed_return_loop_start = None
        self._emit(
            "archive_branch_restored",
            decision=self.decision_index,
            reason=recovery_reason,
            state_id=restored_state_id,
            created_decision=branch.created,
            age=self.decision_index - branch.created,
            action=branch.plan.path[0],
            action_frames=branch.plan.durations[0],
            path=branch.plan.path,
            durations=branch.plan.durations,
            score=branch.score,
            persistent_frontier_value=selected_frontier_value,
            abstract_signature=restored_visual_cluster,
            target_frontier_signature=restored_frontier_signature,
            archive_size=len(self.archive),
            **self._frame_fields(branch.frame),
        )
        self._emit(
            "decision_committed",
            decision=self.decision_index,
            action=branch.plan.path[0],
            action_frames=branch.plan.durations[0],
            path=branch.plan.path,
            durations=branch.plan.durations,
            score=branch.score,
            branches_examined=0,
            restored_archive=True,
            restore_reason=recovery_reason,
            committed_state_id=restored_state_id,
            archive_branches_added=0,
            archive_size=len(self.archive),
            persistent_frontier_value=selected_frontier_value,
            abstract_signature=restored_visual_cluster,
            target_frontier_signature=restored_frontier_signature,
            action_counts=self.action_counts,
            duration_counts=self.duration_counts,
            scene_streak=self.scene_streak,
            visual_stagnation_streak=self.visual_stagnation_streak,
            **self._frame_fields(branch.frame),
        )
        return Decision(
            branch.plan.path[0],
            branch.frame,
            branch.plan.path,
            branch.score,
            0,
            restored_archive=True,
            action_frames=branch.plan.durations[0],
            planned_durations=branch.plan.durations,
        )

    def _prune_archive(self) -> None:
        if len(self.archive) <= self.config.archive_capacity:
            return
        removed = []
        while len(self.archive) > self.config.archive_capacity:
            scene_counts = Counter(branch.scene for branch in self.archive)
            largest_count = max(scene_counts.values())
            crowded_scenes = {
                scene for scene, count in scene_counts.items() if count == largest_count
            }
            victim = min(
                (branch for branch in self.archive if branch.scene in crowded_scenes),
                key=lambda item: (
                    self._archive_frontier_score(item),
                    item.created,
                    item.score,
                ),
            )
            self.archive.remove(victim)
            removed.append(victim)
        release_state = getattr(self.env, "release_state", None)
        if release_state is not None:
            for branch in removed:
                self._emit(
                    "archive_branch_removed",
                    reason="capacity_prune",
                    state_id=self._state_id(branch.state),
                    created_decision=branch.created,
                    score=branch.score,
                )
                release_state(branch.state)

    def run(self, decisions: int) -> List[Decision]:
        if decisions < 0:
            raise ValueError("decisions must be non-negative")
        return [self.decide() for _ in range(decisions)]
