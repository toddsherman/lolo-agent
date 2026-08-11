from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Counter as CounterType, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .agent import Decision
from .bidirectional_probe import BidirectionalProbeCollector
from .ensemble_world_model import EnsembleVisualDynamicsModel
from .environment import Action, PixelSaveStateEnv
from .goal_prior import HeartGoalAnalysis, PixelHeartGoalPrior
from .memory import VisualNovelty
from .neural_world_model import ACTION_TO_INDEX, frame_tensor
from .pixels import Frame, signature_key
from .spatial_shadow import SpatialShadowEvaluator
from .unlabeled_entities import UnlabeledEntityMemory


@dataclass(frozen=True)
class NeuralPlanningConfig:
    actions: Tuple[Action, ...] = (
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
        Action.A,
        Action.B,
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
    action_effect_weight: float = 0.75
    causal_spatial_novelty_weight: float = 2.0
    causal_cell_coverage_weight: float = 0.0
    causal_cell_recovery_grace_decisions: int = 0
    behavioral_edge_coverage_weight: float = 0.0
    behavioral_best_first_archive: bool = False
    persistent_change_stability_decisions: int = 0
    persistent_change_minimum_value_drop: int = 0
    persistent_change_speculative_recovery: bool = False
    causal_affordance_weight: float = 3.0
    causal_event_archive_weight: float = 4.0
    causal_change_pixel_threshold: int = 12
    causal_spatial_columns: int = 16
    causal_spatial_rows: int = 15
    causal_event_min_component_gap: int = 2
    action_coverage_weight: float = 0.35
    duration_coverage_weight: float = 0.2
    consecutive_repeat_weight: float = 0.5
    consecutive_repeat_penalty_cap: Optional[float] = None
    archive_capacity: int = 256
    visual_stagnation_visits: int = 3
    archive_max_age: int = 512
    autonomous_change_threshold: float = 0.00025
    action_equivalence_threshold: float = 0.0001
    autonomous_grace_decisions: int = 4
    control_collapse_confirmation_steps: int = 4
    delayed_transition_probe_steps: int = 0
    dark_transition_intensity_threshold: float = 0.05
    known_scene_return_distance_threshold: float = 0.04
    delayed_return_min_length: int = 4
    delayed_return_credit_horizon: int = 48
    delayed_return_weight: float = 0.75
    delayed_return_penalty_cap: Optional[float] = None
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
    temporal_option_score_weight: float = 1.0
    temporal_option_novelty_weight: float = 1.0
    temporal_option_scene_span_weight: float = 0.5
    temporal_option_duration_weight: float = 0.25
    temporal_option_duration_scale: int = 16
    temporal_option_return_penalty: float = 2.0
    temporal_option_action_prior_weight: float = 1.0
    human_prior_heart_reward: float = 0.0
    human_prior_all_hearts_reward: float = 0.0
    human_prior_chest_reward: float = 0.0
    human_prior_navigation_reward: float = 0.0
    human_prior_life_loss_penalty: float = 0.0
    human_prior_navigation_recovery_grace: int = 0
    human_prior_best_first_archive: bool = False
    human_prior_phase_position_novelty: bool = False
    human_prior_graph_stagnation_visits: int = 0
    human_prior_goal_exhaustion_rollback: bool = False
    human_prior_option_search_depth: int = 0
    human_prior_option_search_beam_width: int = 8
    human_prior_option_search_action_frames: int = 0
    human_prior_option_effect_stability_steps: int = 0
    human_prior_option_effect_probe_limit: int = 8
    human_prior_option_effect_max_stable_cells: int = 4
    human_prior_option_effect_phase_offsets: int = 0
    human_prior_option_effect_phase_l1_threshold: float = 0.002
    human_prior_option_effect_local_controls: bool = False
    human_prior_option_effect_local_minimum_cell_pixels: int = 12
    human_prior_option_effect_frontier: bool = False
    human_prior_option_effect_controllability_depth: int = 1
    human_prior_option_entity_frontier: bool = False
    human_prior_intrinsic_clip: float = 10.0
    spatial_selection_weight: float = 0.0
    returnability_probe_depth: int = 0
    returnability_probe_beam_width: int = 4
    returnability_probe_pixel_l1_threshold: float = 0.002


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
    option_initiation_eligible: bool = False
    option_counterfactual_contrast: float = 0.0
    option_counterfactuals: int = 0
    causal_spatial_signature: str = ""
    causal_spatial_novelty: float = 0.0
    causal_changed_pixels: int = 0
    causal_change_centroid: Optional[Tuple[float, float]] = None
    causal_context_signature: str = ""
    target_causal_context_signature: str = ""
    causal_affordance_actions: Tuple[Action, ...] = ()
    pose_action: Optional[Action] = None
    causal_event_outcome: bool = False
    goal_heart_slots: Tuple[Tuple[int, int], ...] = ()
    goal_progress_reward: float = 0.0
    goal_remaining_hearts: int = 0
    goal_total_hearts: int = 0
    goal_chest_slot: Optional[Tuple[int, int]] = None
    goal_player_slot: Optional[Tuple[int, int]] = None
    parent_state_id: Optional[str] = None
    parent_frame_digest: Optional[str] = None
    parent_decision: int = 0
    search_depth: int = 0
    goal_source_signature: str = ""
    goal_target_signature: str = ""
    goal_source_world_context: str = "human-prior-world-root"
    goal_target_world_context: str = "human-prior-world-root"
    goal_world_effect_signature: str = ""
    human_prior_verified_option: bool = False
    human_prior_option_world_effect_signature: str = ""
    human_prior_option_entity_state_signature: str = ""
    goal_chest_obtained: bool = False


@dataclass
class _HumanPriorOptionNode:
    state: object
    frame: Frame
    path: Tuple[Action, ...]
    durations: Tuple[int, ...]
    analysis: HeartGoalAnalysis
    source_signature: str
    target_signature: str
    score: float
    depth: int
    target_state_visits: int
    target_position_visits: int
    pose_action: Optional[Action] = None
    world_effect_signature: str = ""
    world_effect_state_signature: str = ""
    world_effect_changed_pixels: int = 0
    confirmed_world_effect_signature: str = ""
    confirmed_world_context: str = ""
    confirmed_action_indices: Tuple[int, ...] = ()
    confirmed_entity_state_signature: str = ""
    settling_steps: int = 0
    settling_frames: int = 0
    immediate_frame_digest: str = ""


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


@dataclass(frozen=True)
class _BehaviorProbeSelection:
    keys: Tuple[Tuple[Action, int], ...]
    visual_cluster: str
    reason: str
    selected_control: Optional[Action]
    hypothesis_separation: Optional[float] = None


@dataclass
class _OptionCounterfactual:
    state: object
    frame: Frame
    action: Action
    duration: int
    state_id: Optional[str] = None
    passive_steps: int = 0
    maximum_contrast: float = 0.0


@dataclass
class _LifeHazardCheckpoint:
    state: object
    frame: Frame
    choice: Tuple[str, Action, int]
    decision: int
    frontier_signature: str
    causal_context_signature: str
    scene: str
    pose_action: Optional[Action]
    last_action: Optional[Action]
    last_duration: Optional[int]
    action_streak: int
    goal_heart_slots: Tuple[Tuple[int, int], ...]
    goal_player_slot: Optional[Tuple[int, int]]
    human_prior_world_context_signature: str = "human-prior-world-root"
    kind: str = "causal_option"
    state_id: Optional[str] = None
    goal_chest_obtained: bool = False


@dataclass
class _TemporalOptionTrace:
    choice: Optional[Tuple[str, Action, int]]
    initiation_decision: Optional[int]
    start_decision: int
    entry_signature: str
    entry_scene: str
    initiation_frame_digest: Optional[str] = None
    recovery_checkpoint: Optional[_LifeHazardCheckpoint] = None
    causal_evidence: bool = False
    counterfactual: Optional[_OptionCounterfactual] = None
    passive_decisions: int = 0
    signatures: set[str] = field(default_factory=set)
    scenes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.signatures = {self.entry_signature}
        self.scenes = {self.entry_scene}


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
        spatial_shadow: Optional[SpatialShadowEvaluator] = None,
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
        if self.config.temporal_option_duration_scale <= 0:
            raise ValueError("temporal option duration scale must be positive")
        if not 0.0 <= self.config.temporal_option_action_prior_weight <= 1.0:
            raise ValueError("temporal option action prior weight must be in [0, 1]")
        if self.config.causal_change_pixel_threshold < 0:
            raise ValueError("causal pixel threshold must be non-negative")
        if self.config.causal_cell_coverage_weight < 0.0:
            raise ValueError("causal cell coverage weight must be non-negative")
        if self.config.causal_cell_recovery_grace_decisions < 0:
            raise ValueError(
                "causal cell recovery grace decisions must be non-negative"
            )
        if self.config.behavioral_edge_coverage_weight < 0.0:
            raise ValueError("behavioral edge coverage weight must be non-negative")
        if self.config.persistent_change_stability_decisions < 0:
            raise ValueError(
                "persistent change stability decisions must be non-negative"
            )
        if not 0 <= self.config.persistent_change_minimum_value_drop <= 15:
            raise ValueError(
                "persistent change minimum value drop must be in [0, 15]"
            )
        if (
            self.config.causal_spatial_columns <= 0
            or self.config.causal_spatial_rows <= 0
        ):
            raise ValueError("causal spatial grid dimensions must be positive")
        if self.config.causal_event_min_component_gap <= 0:
            raise ValueError("causal event component gap must be positive")
        if self.config.human_prior_heart_reward < 0.0:
            raise ValueError("human-prior heart reward must be non-negative")
        if self.config.human_prior_all_hearts_reward < 0.0:
            raise ValueError("human-prior all-hearts reward must be non-negative")
        if self.config.human_prior_chest_reward < 0.0:
            raise ValueError("human-prior chest reward must be non-negative")
        if self.config.human_prior_navigation_reward < 0.0:
            raise ValueError("human-prior navigation reward must be non-negative")
        if self.config.human_prior_life_loss_penalty < 0.0:
            raise ValueError("human-prior life-loss penalty must be non-negative")
        if self.config.human_prior_navigation_recovery_grace < 0:
            raise ValueError(
                "human-prior navigation recovery grace must be non-negative"
            )
        if self.config.human_prior_graph_stagnation_visits < 0:
            raise ValueError(
                "human-prior graph stagnation visits must be non-negative"
            )
        if self.config.human_prior_goal_exhaustion_rollback and (
            not self.config.human_prior_best_first_archive
            or self.config.human_prior_graph_stagnation_visits <= 0
            or self.config.human_prior_option_search_depth < 2
        ):
            raise ValueError(
                "human-prior goal exhaustion rollback requires best-first "
                "archive, graph stagnation, and option search"
            )
        if self.config.human_prior_option_search_depth < 0:
            raise ValueError(
                "human-prior option search depth must be non-negative"
            )
        if self.config.human_prior_option_search_beam_width <= 0:
            raise ValueError(
                "human-prior option search beam width must be positive"
            )
        if self.config.human_prior_option_search_action_frames < 0:
            raise ValueError(
                "human-prior option search action frames must be non-negative"
            )
        if self.config.human_prior_option_effect_stability_steps < 0:
            raise ValueError(
                "human-prior option effect stability steps must be non-negative"
            )
        if self.config.human_prior_option_effect_probe_limit <= 0:
            raise ValueError(
                "human-prior option effect probe limit must be positive"
            )
        if self.config.human_prior_option_effect_max_stable_cells <= 0:
            raise ValueError(
                "human-prior option effect max stable cells must be positive"
            )
        if self.config.human_prior_option_effect_phase_offsets < 0:
            raise ValueError(
                "human-prior option effect phase offsets must be non-negative"
            )
        if self.config.human_prior_option_effect_phase_l1_threshold < 0.0:
            raise ValueError(
                "human-prior option effect phase L1 threshold must be non-negative"
            )
        if (
            self.config.human_prior_option_effect_local_minimum_cell_pixels
            <= 0
        ):
            raise ValueError(
                "human-prior option effect local minimum cell pixels must be positive"
            )
        if (
            self.config.human_prior_option_effect_frontier
            and self.config.human_prior_option_effect_phase_offsets <= 0
        ):
            raise ValueError(
                "human-prior option effect frontier requires phase offsets"
            )
        if not (
            1
            <= self.config.human_prior_option_effect_controllability_depth
            <= 4
        ):
            raise ValueError(
                "human-prior option effect controllability depth must be "
                "between one and four"
            )
        if self.config.human_prior_option_entity_frontier and (
            not self.config.human_prior_option_effect_local_controls
            or self.config.human_prior_option_effect_stability_steps <= 0
            or self.config.human_prior_option_effect_phase_offsets <= 0
        ):
            raise ValueError(
                "human-prior option entity frontier requires local controls, "
                "effect stability, and phase offsets"
            )
        if self.config.human_prior_intrinsic_clip <= 0.0:
            raise ValueError("human-prior intrinsic clip must be positive")
        if self.config.spatial_selection_weight < 0.0:
            raise ValueError("spatial selection weight must be non-negative")
        if self.config.spatial_selection_weight > 0.0 and spatial_shadow is None:
            raise ValueError("positive spatial selection weight requires a spatial model")
        if self.config.returnability_probe_depth < 0:
            raise ValueError("returnability probe depth must be non-negative")
        if self.config.returnability_probe_beam_width <= 0:
            raise ValueError("returnability probe beam width must be positive")
        if self.config.returnability_probe_pixel_l1_threshold < 0.0:
            raise ValueError("returnability probe threshold must be non-negative")
        if self.config.control_collapse_confirmation_steps <= 0:
            raise ValueError("control-collapse confirmation steps must be positive")
        if self.config.delayed_transition_probe_steps < 0:
            raise ValueError("delayed-transition probe steps must be non-negative")
        if not 0.0 <= self.config.dark_transition_intensity_threshold <= 1.0:
            raise ValueError("dark transition threshold must be between zero and one")
        if not 0.0 <= self.config.known_scene_return_distance_threshold <= 1.0:
            raise ValueError("known-scene return threshold must be between zero and one")
        self.model.freeze()
        self.planner = NeuralRolloutPlanner(model, device, self.config)
        self.novelty = VisualNovelty()
        self.frame: Optional[Frame] = None
        self.action_counts: CounterType[Action] = Counter()
        self.duration_counts: CounterType[int] = Counter()
        self.action_duration_counts: CounterType[Tuple[Action, int]] = Counter()
        self.action_effect_values: Dict[Tuple[str, Action], float] = {}
        self.action_effect_samples: CounterType[Tuple[str, Action]] = Counter()
        self.causal_spatial_visits: CounterType[str] = Counter()
        self.last_causal_cell_progress_decision: Optional[int] = None
        self.behavioral_edge_visits: CounterType[
            Tuple[str, Action, int]
        ] = Counter()
        self.persistent_change_baseline: List[int] = []
        self.persistent_change_value_counts: List[
            CounterType[int]
        ] = []
        self.persistent_change_candidates: Dict[int, Tuple[int, int]] = {}
        self.persistent_change_cells: Dict[int, int] = {}
        self.persistent_change_mismatches: CounterType[int] = Counter()
        self.last_action: Optional[Action] = None
        self.last_duration: Optional[int] = None
        self.last_action_was_causal_spatial = False
        self.action_streak = 0
        self.scene_visits: CounterType[str] = Counter()
        self.scene_action_probes: CounterType[Tuple[str, Action]] = Counter()
        self.delayed_return_costs: CounterType[Tuple[str, Action, int]] = Counter()
        self.transition_history: List[_CommittedTransition] = []
        self.visual_last_visit: Dict[str, int] = {}
        self.delayed_return_recovery = False
        self.delayed_return_loop_start: Optional[int] = None
        self.autonomous_grace_remaining = 0
        self.autonomous_intervention_pending = False
        self.causal_observation_intervention_pending = False
        self.anticipated_transition_observations_remaining = 0
        self.anticipated_transition_observation_duration = (
            self.config.action_frames
        )
        self.dark_transition_active = False
        self.dark_transition_start_decision: Optional[int] = None
        self.pending_novel_room_frame: Optional[Frame] = None
        self.known_scene_return_recovery_pending = False
        self.bright_scene_memory: List[Tuple[int, ...]] = []
        self.known_scene_recovery_checkpoint: Optional[
            _LifeHazardCheckpoint
        ] = None
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
        self.visual_probe_counts: CounterType[Tuple[str, Action, int]] = Counter()
        self.behavior_cluster_serial = 0
        self.provisional_state_serial = 0
        self.current_frontier_signature = ""
        self.behavior_visits: CounterType[str] = Counter()
        self.pending_option_choice: Optional[Tuple[str, Action, int]] = None
        self.pending_option_decision: Optional[int] = None
        self.pending_option_frame_digest: Optional[str] = None
        self.pending_option_recovery_checkpoint: Optional[
            _LifeHazardCheckpoint
        ] = None
        self.pending_option_causal_evidence = False
        self.pending_option_counterfactual: Optional[_OptionCounterfactual] = None
        self.active_temporal_option: Optional[_TemporalOptionTrace] = None
        self.temporal_option_values: Dict[Tuple[str, Action, int], float] = {}
        self.temporal_option_samples: CounterType[Tuple[str, Action, int]] = Counter()
        self.temporal_option_action_values: Dict[Action, float] = {}
        self.temporal_option_action_samples: CounterType[Action] = Counter()
        self.current_scene: Optional[str] = None
        self.scene_streak = 0
        self.visual_stagnation_streak = 0
        self.archive: List[_ArchivedBranch] = []
        self.decision_index = 0
        self.current_search_depth = 0
        self.event_logger = event_logger
        self.spatial_shadow = spatial_shadow
        self.returnability_probe = (
            None
            if self.config.returnability_probe_depth == 0
            else BidirectionalProbeCollector(
                self.env,
                self.config.actions,
                maximum_depth=self.config.returnability_probe_depth,
                beam_width=self.config.returnability_probe_beam_width,
                pixel_l1_threshold=(
                    self.config.returnability_probe_pixel_l1_threshold
                ),
                emit=self._emit,
                frame_fields=self._frame_fields,
                state_id=self._state_id,
            )
        )
        self.goal_prior: Optional[PixelHeartGoalPrior] = None
        self.unlabeled_entity_memory: Optional[
            UnlabeledEntityMemory
        ] = None
        self.human_prior_graph_edge_visits: CounterType[
            Tuple[str, Action, int]
        ] = Counter()
        self.human_prior_graph_state_visits: CounterType[str] = Counter()
        self.human_prior_player_position_visits: CounterType[
            Tuple[int, int]
        ] = Counter()
        self.human_prior_phase_player_position_visits: CounterType[
            Tuple[str, Tuple[int, int]]
        ] = Counter()
        self.human_prior_graph_recovery_pending = False
        self.current_human_prior_world_context_signature = (
            "human-prior-world-root"
        )
        self.last_navigation_change_decision: Optional[int] = None
        self.pending_life_hazard_choice: Optional[
            Tuple[int, str, Action, int, str]
        ] = None
        self.pending_life_recovery: Optional[_LifeHazardCheckpoint] = None
        self.pending_recovery_cause: Optional[str] = None
        self.pending_goal_milestone_checkpoint: Optional[
            _LifeHazardCheckpoint
        ] = None

    def _reset_goal_prior(self) -> None:
        enabled = bool(
            self.config.human_prior_heart_reward
            or self.config.human_prior_all_hearts_reward
            or self.config.human_prior_chest_reward
            or self.config.human_prior_navigation_reward
            or self.config.human_prior_life_loss_penalty
        )
        self.goal_prior = (
            PixelHeartGoalPrior(
                heart_reward=self.config.human_prior_heart_reward,
                all_hearts_reward=self.config.human_prior_all_hearts_reward,
                chest_reward=self.config.human_prior_chest_reward,
                navigation_reward=self.config.human_prior_navigation_reward,
                life_loss_penalty=self.config.human_prior_life_loss_penalty,
            )
            if enabled
            else None
        )

    def _reset_unlabeled_entity_memory(self, frame: Frame) -> None:
        if not self.config.human_prior_option_entity_frontier:
            self.unlabeled_entity_memory = None
            return
        self.unlabeled_entity_memory = UnlabeledEntityMemory(
            columns=self.config.causal_spatial_columns,
            rows=self.config.causal_spatial_rows,
        )
        observation = self.unlabeled_entity_memory.observe(frame)
        self._emit(
            "human_prior_unlabeled_entity_memory_reset",
            decision=self.decision_index,
            prototype_count=self.unlabeled_entity_memory.prototype_count,
            entity_grid_signature=observation.signature(),
            agent_visible=True,
            **self._frame_fields(frame),
        )

    def _calibrate_goal_prior(self, frame: Frame) -> None:
        if self.goal_prior is None:
            return
        before = tuple(sorted(self.goal_prior.known_slots))
        discovered = self.goal_prior.observe_room(frame)
        after = tuple(sorted(self.goal_prior.known_slots))
        if after != before:
            self._emit(
                "human_prior_calibrated",
                decision=self.decision_index,
                reward_track="human_prior_v2",
                discovered_heart_slots=discovered,
                known_heart_slots=after,
                current_heart_slots=self.goal_prior.current_slots(),
                prototypes=(
                    "lolo-heart-16x16-v1",
                    "lolo-open-chest-16x16-v2-animated",
                    "lolo-life-hud-8x8-v1",
                ),
                agent_visible=True,
                **self._frame_fields(frame),
            )

    def _human_prior_score(
        self, intrinsic_score: float, analysis: Optional[HeartGoalAnalysis]
    ) -> Tuple[float, float]:
        if analysis is None:
            return intrinsic_score, intrinsic_score
        if analysis.outcome_reward == 0.0:
            return (
                intrinsic_score + analysis.navigation_reward,
                intrinsic_score,
            )
        clip = self.config.human_prior_intrinsic_clip
        clipped = max(-clip, min(clip, intrinsic_score))
        return clipped + analysis.total_reward, clipped

    def _commit_goal_prior(
        self, analysis: HeartGoalAnalysis, frame: Frame
    ) -> HeartGoalAnalysis:
        if self.goal_prior is None:
            return analysis
        before = tuple(sorted(self.goal_prior.known_slots))
        self.goal_prior.commit(analysis, frame)
        after = tuple(sorted(self.goal_prior.known_slots))
        if after != before:
            self._emit(
                "human_prior_calibrated",
                decision=self.decision_index + 1,
                reward_track="human_prior_v2",
                discovered_heart_slots=after,
                known_heart_slots=after,
                current_heart_slots=self.goal_prior.current_slots(),
                prototypes=(
                    "lolo-heart-16x16-v1",
                    "lolo-open-chest-16x16-v2-animated",
                    "lolo-life-hud-8x8-v1",
                ),
                agent_visible=True,
                **self._frame_fields(frame),
            )
        return analysis

    def _restore_goal_prior(
        self,
        present: Sequence[Tuple[int, int]],
        frame: Frame,
        player_slot: Optional[Tuple[int, int]],
        chest_obtained: bool,
    ) -> None:
        if self.goal_prior is None:
            return
        # Test and research sidecars may implement the original three-argument
        # restore protocol without the assisted chest state.
        self.goal_prior.restore(present, frame, player_slot)
        if hasattr(self.goal_prior, "chest_obtained"):
            self.goal_prior.chest_obtained = bool(chest_obtained)

    def _human_prior_fields(
        self, analysis: Optional[HeartGoalAnalysis],
    ) -> Dict[str, Any]:
        if analysis is None:
            return {
                "human_prior_enabled": False,
                "human_prior_goal_reward": 0.0,
            }
        return {
            "human_prior_enabled": True,
            "human_prior_reward_track": "human_prior_v2",
            "human_prior_best_remaining_hearts": (
                None
                if self.goal_prior is None
                else self.goal_prior.best_remaining_hearts
            ),
            **analysis.telemetry(),
        }

    @staticmethod
    def _human_prior_graph_signature(
        present: Sequence[Tuple[int, int]],
        player: Optional[Tuple[int, int]],
        chest: Optional[Tuple[int, int]],
        life_signature: Optional[str],
        world_context: str = "human-prior-world-root",
        chest_obtained: bool = False,
    ) -> str:
        """Stable assisted-track state key derived only from labelled pixels.

        Animation frames should not turn one physical puzzle position into an
        unlimited number of apparent graph nodes.  The key is unavailable
        when the explicit player detector is unavailable, in which case the
        strict visual/behavioral frontier remains authoritative.
        """

        if player is None:
            return ""
        hearts = ";".join(
            f"{x},{y}" for x, y in sorted(set(present))
        )
        chest_key = "none" if chest is None else f"{chest[0]},{chest[1]}"
        life_key = life_signature or "unknown"
        treasure_key = "obtained" if chest_obtained else "pending"
        return (
            f"hearts={hearts}|player={player[0]},{player[1]}|"
            f"chest={chest_key}|treasure={treasure_key}|"
            f"life={life_key}|world={world_context}"
        )

    def _human_prior_world_effect_signature(
        self,
        spatial_signature: Optional[str],
        analysis: Optional[HeartGoalAnalysis],
        frame: Frame,
        action: Optional[Action] = None,
        allow_nonlocal: bool = False,
    ) -> str:
        """Remove detected player motion from a matched causal pixel effect.

        The remaining cells are a rule-free, action-conditioned indication
        that something in the room changed independently of the controlled
        sprite.  Comparing against a duration-matched NOOP has already
        removed autonomous animation; masking the source and target player
        tiles prevents ordinary movement from creating path-dependent world
        states. Multi-action matched-time probes may retain non-local cells
        because their controlled path can legitimately affect several parts
        of the screen before the endpoint is observed.
        """

        if not spatial_signature or analysis is None:
            return ""
        try:
            occupied = bytearray.fromhex(spatial_signature)
        except ValueError:
            return ""
        columns = min(self.config.causal_spatial_columns, frame.width)
        rows = min(self.config.causal_spatial_rows, frame.height)
        if len(occupied) != columns * rows:
            return ""
        player_cells = set()
        for slot in {
            analysis.source_player_slot,
            analysis.target_player_slot,
        }:
            if slot is None:
                continue
            gx = min(columns - 1, max(0, slot[0] * columns // frame.width))
            gy = min(rows - 1, max(0, slot[1] * rows // frame.height))
            player_cells.add((gx, gy))
            occupied[gy * columns + gx] = 0
        if (
            not allow_nonlocal
            and action not in (Action.A, Action.B)
            and player_cells
        ):
            for index in range(len(occupied)):
                gx = index % columns
                gy = index // columns
                if min(
                    abs(gx - px) + abs(gy - py)
                    for px, py in player_cells
                ) > 1:
                    occupied[index] = 0
        if not any(occupied):
            return ""
        return occupied.hex()

    def _human_prior_world_effect_state_signature(
        self, frame: Frame, world_effect_signature: str
    ) -> str:
        """Hash anonymous absolute appearances at action-changed cells."""

        memory = self.unlabeled_entity_memory
        cells = self._causal_spatial_cells(world_effect_signature)
        if memory is None or not cells:
            return ""
        payload = ";".join(
            f"{column},{row}="
            + ",".join(
                str(value)
                for value in memory.feature_at(frame, column, row)
            )
            for column, row in sorted(cells)
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]

    def _human_prior_nonlocal_world_effect_cells(
        self,
        world_effect_signature: Optional[str],
        analysis: HeartGoalAnalysis,
        frame: Frame,
        extra_player_slots: Sequence[Optional[Tuple[int, int]]] = (),
    ) -> set[Tuple[int, int]]:
        """Conservatively exclude the controlled sprite's neighborhood.

        The assisted player detector reports a snapped tile anchor, while a
        rendered pose can spill across that tile boundary.  A one-cell
        Manhattan guard around both detected anchors keeps those pose pixels
        observationally separate from more distant world effects.  Nearby
        manipulation remains deliberately unresolved rather than being
        mislabeled as a persistent object change.
        """

        cells = self._causal_spatial_cells(world_effect_signature)
        if not cells:
            return set()
        columns = self.config.causal_spatial_columns
        rows = self.config.causal_spatial_rows
        player_cells = set()
        for slot in {
            analysis.source_player_slot,
            analysis.target_player_slot,
            *extra_player_slots,
        }:
            if slot is None:
                continue
            player_cells.add(
                (
                    min(
                        columns - 1,
                        max(0, slot[0] * columns // frame.width),
                    ),
                    min(
                        rows - 1,
                        max(0, slot[1] * rows // frame.height),
                    ),
                )
            )
        if not player_cells:
            return cells
        return {
            cell
            for cell in cells
            if min(
                abs(cell[0] - player[0])
                + abs(cell[1] - player[1])
                for player in player_cells
            )
            > 1
        }

    def _human_prior_cell_patch_l1(
        self,
        left: Frame,
        right: Frame,
        cells: set[Tuple[int, int]],
    ) -> float:
        if (
            not cells
            or left.width != right.width
            or left.height != right.height
            or left.channels != right.channels
        ):
            return 1.0
        columns = self.config.causal_spatial_columns
        rows = self.config.causal_spatial_rows
        total = 0
        values = 0
        for column, row in cells:
            x_start = column * left.width // columns
            x_end = (column + 1) * left.width // columns
            y_start = row * left.height // rows
            y_end = (row + 1) * left.height // rows
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    offset = (y * left.width + x) * left.channels
                    for channel in range(left.channels):
                        total += abs(
                            left.pixels[offset + channel]
                            - right.pixels[offset + channel]
                        )
                        values += 1
        return total / (255.0 * max(1, values))

    @staticmethod
    def _next_human_prior_world_context(
        source_context: str, world_effect_signature: str
    ) -> str:
        if not world_effect_signature:
            return source_context
        effect = bytes.fromhex(world_effect_signature)
        width = max(1, (len(effect) + 3) // 4)
        try:
            active = (
                0
                if source_context == "human-prior-world-root"
                else int(source_context, 16)
            )
        except ValueError:
            active = 0
        for index, changed in enumerate(effect):
            if changed:
                active ^= 1 << index
        return (
            "human-prior-world-root"
            if active == 0
            else f"{active:0{width}x}"
        )

    def _human_prior_graph_signatures(
        self,
        analysis: Optional[HeartGoalAnalysis],
        source_world_context: Optional[str] = None,
        target_world_context: Optional[str] = None,
    ) -> Tuple[str, str]:
        if analysis is None:
            return "", ""
        source_context = (
            source_world_context
            or self.current_human_prior_world_context_signature
        )
        target_context = target_world_context or source_context
        return (
            self._human_prior_graph_signature(
                analysis.source_present,
                analysis.source_player_slot,
                analysis.source_chest_slot,
                analysis.source_life_signature,
                source_context,
                bool(
                    analysis.chest_obtained
                    and not analysis.chest_completed
                ),
            ),
            self._human_prior_graph_signature(
                analysis.target_present,
                analysis.target_player_slot,
                (
                    analysis.target_chest_slot
                    if analysis.chest_obtained
                    else (
                        analysis.target_chest_slot
                        or analysis.source_chest_slot
                    )
                ),
                analysis.target_life_signature,
                target_context,
                analysis.chest_obtained,
            ),
        )

    def _current_human_prior_graph_signature(self) -> str:
        if self.goal_prior is None or self.frame is None:
            return ""
        analysis = self.goal_prior.analyze(self.frame, self.frame)
        return self._human_prior_graph_signatures(analysis)[1]

    @staticmethod
    def _human_prior_position_phase(signature: str) -> str:
        fields = {
            item.split("=", 1)[0]: item.split("=", 1)[1]
            for item in signature.split("|")
            if "=" in item
        }
        return (
            f"hearts={fields.get('hearts', '')}|"
            f"treasure={fields.get('treasure', 'pending')}"
        )

    def _human_prior_position_visits(
        self, signature: str, player_slot: Tuple[int, int]
    ) -> int:
        if not self.config.human_prior_phase_position_novelty:
            return self.human_prior_player_position_visits[player_slot]
        return self.human_prior_phase_player_position_visits[
            (self._human_prior_position_phase(signature), player_slot)
        ]

    def _record_human_prior_player_position(
        self, signature: str, player_slot: Tuple[int, int]
    ) -> None:
        self.human_prior_player_position_visits[player_slot] += 1
        self.human_prior_phase_player_position_visits[
            (self._human_prior_position_phase(signature), player_slot)
        ] += 1

    def _human_prior_graph_edge_coverage(
        self, signature: str, action: Action, duration: int
    ) -> Tuple[int, bool]:
        if not signature:
            return 0, False
        visits = self.human_prior_graph_edge_visits[
            (signature, action, duration)
        ]
        return visits, visits == 0

    def _record_human_prior_graph_edge(
        self, signature: str, action: Action, duration: int
    ) -> None:
        if signature:
            self.human_prior_graph_edge_visits[
                (signature, action, duration)
            ] += 1

    @staticmethod
    def _human_prior_option_key(
        source_signature: str,
        path: Sequence[Action],
        durations: Sequence[int],
    ) -> Tuple[str, Tuple[Tuple[Action, int], ...]]:
        return (
            source_signature,
            tuple(zip(tuple(path), tuple(durations))),
        )

    def _human_prior_option_coverage(
        self,
        source_signature: str,
        path: Sequence[Action],
        durations: Sequence[int],
    ) -> Tuple[int, bool]:
        if not source_signature or not path:
            return 0, False
        visits = self.human_prior_option_visits[
            self._human_prior_option_key(
                source_signature, path, durations
            )
        ]
        return visits, visits == 0

    def _human_prior_archive_edge_coverage(
        self, branch: _ArchivedBranch
    ) -> Tuple[int, bool]:
        if branch.human_prior_verified_option:
            return self._human_prior_option_coverage(
                branch.goal_source_signature,
                branch.plan.path,
                branch.plan.durations,
            )
        return self._human_prior_graph_edge_coverage(
            branch.goal_source_signature,
            branch.plan.path[0],
            branch.plan.durations[0],
        )

    def _record_human_prior_archive_edge(
        self, branch: _ArchivedBranch
    ) -> None:
        if branch.human_prior_verified_option:
            self.human_prior_option_visits[
                self._human_prior_option_key(
                    branch.goal_source_signature,
                    branch.plan.path,
                    branch.plan.durations,
                )
            ] += 1
            return
        self._record_human_prior_graph_edge(
            branch.goal_source_signature,
            branch.plan.path[0],
            branch.plan.durations[0],
        )

    def _human_prior_unvisited_archive_endpoints(
        self, source_signature: str = ""
    ) -> int:
        return sum(
            branch.plan.path[0] != Action.NOOP
            and bool(branch.goal_source_signature)
            and (
                not source_signature
                or branch.goal_source_signature == source_signature
            )
            and branch.goal_player_slot is not None
            and self._human_prior_position_visits(
                branch.goal_target_signature,
                branch.goal_player_slot,
            ) == 0
            and self._human_prior_archive_edge_coverage(branch)[1]
            for branch in self.archive
        )

    def _probe_human_prior_option_world_effect(
        self,
        root: object,
        source_frame: Frame,
        node: _HumanPriorOptionNode,
        future_duration: int,
        candidate_rank: int,
    ) -> Dict[str, Any]:
        assert self.goal_prior is not None
        observations = []
        cell_sets: List[set[Tuple[int, int]]] = []
        nonlocal_cell_sets: List[set[Tuple[int, int]]] = []
        safe = True
        final_factual = node.frame
        for future_step in range(
            self.config.human_prior_option_effect_stability_steps + 1
        ):
            self.env.load_state(root)
            factual = source_frame
            for action, duration in zip(node.path, node.durations):
                factual = self.env.step(action, duration)
            for _ in range(future_step):
                factual = self.env.step(Action.NOOP, future_duration)

            self.env.load_state(root)
            neutral = source_frame
            for duration in node.durations:
                neutral = self.env.step(Action.NOOP, duration)
            for _ in range(future_step):
                neutral = self.env.step(Action.NOOP, future_duration)

            analysis = self.goal_prior.analyze(source_frame, factual)
            spatial_signature, changed_pixels, _centroid = (
                self._causal_spatial_effect(factual, neutral)
            )
            world_effect_signature = (
                self._human_prior_world_effect_signature(
                    spatial_signature,
                    analysis,
                    factual,
                    action=node.path[-1],
                    allow_nonlocal=True,
                )
            )
            cells = self._causal_spatial_cells(
                world_effect_signature
            )
            cell_sets.append(cells)
            nonlocal_cells = self._human_prior_nonlocal_world_effect_cells(
                world_effect_signature, analysis, factual
            )
            nonlocal_cell_sets.append(nonlocal_cells)
            safe = bool(
                safe
                and analysis.target_player_slot is not None
                and not analysis.life_counter_changed
                and not analysis.dark_transition_started
            )
            observations.append(
                {
                    "future_step": future_step,
                    "factual_frame": factual.digest,
                    "neutral_frame": neutral.digest,
                    "world_effect_signature": (
                        world_effect_signature or None
                    ),
                    "world_effect_cells": sorted(cells),
                    "world_effect_nonlocal_cells": sorted(
                        nonlocal_cells
                    ),
                    "changed_pixels": changed_pixels,
                    "target_player_slot": (
                        analysis.target_player_slot
                    ),
                    "life_counter_changed": (
                        analysis.life_counter_changed
                    ),
                    "dark_transition_started": (
                        analysis.dark_transition_started
                    ),
                }
            )
            final_factual = factual
        initial_cells = cell_sets[0] if cell_sets else set()
        common_cells = (
            set.intersection(*cell_sets) if cell_sets else set()
        )
        initial_nonlocal_cells = (
            nonlocal_cell_sets[0] if nonlocal_cell_sets else set()
        )
        common_nonlocal_cells = (
            set.intersection(*nonlocal_cell_sets)
            if nonlocal_cell_sets
            else set()
        )
        persistence_ratio = (
            len(common_nonlocal_cells) / len(initial_nonlocal_cells)
            if initial_nonlocal_cells
            else 0.0
        )
        localized = bool(
            common_nonlocal_cells
            and len(common_nonlocal_cells)
            <= self.config.human_prior_option_effect_max_stable_cells
        )
        stable = bool(
            safe
            and initial_nonlocal_cells
            and all(nonlocal_cell_sets)
            and common_nonlocal_cells
            and localized
            and persistence_ratio >= 0.5
        )
        raw_persistence_ratio = (
            len(common_cells) / len(initial_cells)
            if initial_cells
            else 0.0
        )
        local_candidate = bool(
            safe
            and initial_cells
            and all(cell_sets)
            and common_cells
            and len(common_cells)
            <= self.config.human_prior_option_effect_max_stable_cells
            and raw_persistence_ratio >= 0.5
            and bool(common_cells - common_nonlocal_cells)
        )
        columns = self.config.causal_spatial_columns
        rows = self.config.causal_spatial_rows
        common_signature = bytearray(columns * rows)
        for column, row in common_nonlocal_cells:
            if 0 <= column < columns and 0 <= row < rows:
                common_signature[row * columns + column] = 1
        raw_common_signature = bytearray(columns * rows)
        for column, row in common_cells:
            if 0 <= column < columns and 0 <= row < rows:
                raw_common_signature[row * columns + column] = 1
        result = {
            "candidate_rank": candidate_rank,
            "depth": node.depth,
            "path": node.path,
            "durations": node.durations,
            "source_graph_signature": node.source_signature,
            "target_graph_signature": node.target_signature or None,
            "initial_world_effect_signature": (
                node.world_effect_signature or None
            ),
            "stable_world_effect_signature": (
                bytes(common_signature).hex()
                if common_nonlocal_cells
                else None
            ),
            "initial_world_effect_cells": len(initial_cells),
            "persistent_world_effect_cells": len(common_cells),
            "persistent_world_effect_signature": (
                bytes(raw_common_signature).hex()
                if common_cells
                else None
            ),
            "initial_nonlocal_world_effect_cells": len(
                initial_nonlocal_cells
            ),
            "stable_world_effect_cells": len(common_nonlocal_cells),
            "local_only_persistent_world_effect_cells": len(
                common_cells - common_nonlocal_cells
            ),
            "persistence_ratio": persistence_ratio,
            "raw_persistence_ratio": raw_persistence_ratio,
            "localized": localized,
            "local_candidate": local_candidate,
            "maximum_stable_cells": (
                self.config.human_prior_option_effect_max_stable_cells
            ),
            "stability_steps": (
                self.config.human_prior_option_effect_stability_steps
            ),
            "safe": safe,
            "stable": stable,
            "observations": observations,
        }
        self._emit(
            "human_prior_option_world_effect_stability",
            decision=self.decision_index + 1,
            agent_visible=True,
            **result,
            **self._frame_fields(final_factual),
        )
        return result

    def _probe_human_prior_option_phase_alignment(
        self,
        root: object,
        source_frame: Frame,
        node: _HumanPriorOptionNode,
        stable_world_effect_signature: str,
        candidate_rank: int,
    ) -> Dict[str, Any]:
        """Test whether a localized effect is merely an animation phase shift."""

        cells = self._causal_spatial_cells(
            stable_world_effect_signature
        )
        self.env.load_state(root)
        neutral = source_frame
        for duration in node.durations:
            neutral = self.env.step(Action.NOOP, duration)
        observations = []
        best_offset = 0
        best_patch_l1 = float("inf")
        best_frame = neutral
        for offset in range(
            self.config.human_prior_option_effect_phase_offsets + 1
        ):
            if offset > 0:
                neutral = self.env.step(Action.NOOP, 1)
            patch_l1 = self._human_prior_cell_patch_l1(
                node.frame, neutral, cells
            )
            observations.append(
                {
                    "offset_frames": offset,
                    "neutral_frame": neutral.digest,
                    "patch_l1": patch_l1,
                }
            )
            if patch_l1 < best_patch_l1:
                best_patch_l1 = patch_l1
                best_offset = offset
                best_frame = neutral
        phase_equivalent = bool(
            cells
            and best_patch_l1
            <= self.config.human_prior_option_effect_phase_l1_threshold
        )
        result = {
            "candidate_rank": candidate_rank,
            "depth": node.depth,
            "path": node.path,
            "durations": node.durations,
            "source_graph_signature": node.source_signature,
            "target_graph_signature": node.target_signature or None,
            "stable_world_effect_signature": (
                stable_world_effect_signature
            ),
            "stable_world_effect_cells": len(cells),
            "phase_offsets": (
                self.config.human_prior_option_effect_phase_offsets
            ),
            "phase_l1_threshold": (
                self.config.human_prior_option_effect_phase_l1_threshold
            ),
            "best_offset_frames": best_offset,
            "best_patch_l1": best_patch_l1,
            "best_neutral_frame": best_frame.digest,
            "phase_equivalent": phase_equivalent,
            "observations": observations,
        }
        self._emit(
            "human_prior_option_world_effect_phase_alignment",
            decision=self.decision_index + 1,
            agent_visible=True,
            **result,
            **self._frame_fields(node.frame),
        )
        return result

    def _human_prior_option_interaction_ray(
        self,
        root: object,
        source_frame: Frame,
        node: _HumanPriorOptionNode,
        action_index: int,
    ) -> Tuple[Frame, Optional[Action], Tuple[Tuple[int, int], ...]]:
        """Locate the pixel-only action ray before one option intervention."""

        assert self.goal_prior is not None
        memory = self.unlabeled_entity_memory
        if memory is None:
            return source_frame, None, ()
        pose = self.current_pose_action
        before = source_frame
        self.env.load_state(root)
        for index, (action, duration) in enumerate(
            zip(node.path, node.durations)
        ):
            if index == action_index:
                break
            before = self.env.step(action, duration)
            pose = self._resulting_pose_action(pose, action)
        intervention = node.path[action_index]
        direction = (
            intervention
            if intervention
            in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
            else pose if intervention in (Action.A, Action.B) else None
        )
        if direction is None:
            return before, None, ()
        analysis = self.goal_prior.analyze(source_frame, before)
        player = analysis.target_player_slot
        if player is None:
            return before, direction, ()
        column = min(
            memory.columns - 1,
            max(0, player[0] * memory.columns // before.width),
        )
        row = min(
            memory.rows - 1,
            max(0, player[1] * memory.rows // before.height),
        )
        offsets = {
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0),
        }
        dx, dy = offsets[direction]
        cells = []
        column += dx
        row += dy
        while 0 <= column < memory.columns and 0 <= row < memory.rows:
            cells.append((column, row))
            column += dx
            row += dy
        return before, direction, tuple(cells)

    def _probe_human_prior_option_action_control(
        self,
        root: object,
        source_frame: Frame,
        node: _HumanPriorOptionNode,
        future_duration: int,
        candidate_rank: int,
        action_index: int,
        allow_endpoint_matched_local: bool = False,
    ) -> Dict[str, Any]:
        """Replace one option action with an equal-duration NOOP.

        This observational control asks whether a compact persistent effect is
        attributable to a particular intervention rather than merely to the
        other actions, elapsed animation, or endpoint pose. Both factual and
        control player neighborhoods are excluded from the nonlocal effect.
        """

        assert self.goal_prior is not None
        if not 0 <= action_index < len(node.path):
            raise IndexError("option action control index out of range")
        control_actions = list(node.path)
        replaced_action = control_actions[action_index]
        control_actions[action_index] = Action.NOOP
        control_path = tuple(control_actions)
        observations = []
        audited_cell_sets: List[set[Tuple[int, int]]] = []
        safe = True
        endpoint_matched_all = True
        final_factual = node.frame
        final_control = source_frame
        final_entity_player_pixels: set[Tuple[int, int]] = set()
        before_intervention, interaction_direction, interaction_ray = (
            self._human_prior_option_interaction_ray(
                root, source_frame, node, action_index
            )
        )
        for future_step in range(
            self.config.human_prior_option_effect_stability_steps + 1
        ):
            self.env.load_state(root)
            factual = source_frame
            for action, duration in zip(node.path, node.durations):
                factual = self.env.step(action, duration)
            for _ in range(future_step):
                factual = self.env.step(Action.NOOP, future_duration)

            self.env.load_state(root)
            control = source_frame
            for action, duration in zip(control_path, node.durations):
                control = self.env.step(action, duration)
            for _ in range(future_step):
                control = self.env.step(Action.NOOP, future_duration)

            factual_analysis = self.goal_prior.analyze(
                source_frame, factual
            )
            control_analysis = self.goal_prior.analyze(
                source_frame, control
            )
            endpoint_matched = bool(
                factual_analysis.target_player_slot is not None
                and factual_analysis.target_player_slot
                == control_analysis.target_player_slot
            )
            ignored_player_pixels: set[Tuple[int, int]] = set()
            entity_player_pixels: set[Tuple[int, int]] = set()
            player_pixel_mask = getattr(
                self.goal_prior, "player_pixel_mask", None
            )
            if (
                endpoint_matched
                and callable(player_pixel_mask)
            ):
                entity_player_pixels.update(
                    player_pixel_mask(
                        factual, factual_analysis.target_player_slot
                    )
                )
                entity_player_pixels.update(
                    player_pixel_mask(
                        control, control_analysis.target_player_slot
                    )
                )
                if allow_endpoint_matched_local:
                    ignored_player_pixels.update(entity_player_pixels)
            final_entity_player_pixels = entity_player_pixels
            spatial_signature, changed_pixels, _centroid = (
                self._causal_spatial_effect(
                    factual,
                    control,
                    ignored_pixels=(
                        ignored_player_pixels
                        if ignored_player_pixels
                        else None
                    ),
                    minimum_cell_pixels=(
                        self.config.human_prior_option_effect_local_minimum_cell_pixels
                        if ignored_player_pixels
                        else 1
                    ),
                )
            )
            world_effect_signature = (
                self._human_prior_world_effect_signature(
                    spatial_signature,
                    factual_analysis,
                    factual,
                    action=replaced_action,
                    allow_nonlocal=True,
                )
            )
            nonlocal_cells = (
                self._human_prior_nonlocal_world_effect_cells(
                    world_effect_signature,
                    factual_analysis,
                    factual,
                    extra_player_slots=(
                        control_analysis.source_player_slot,
                        control_analysis.target_player_slot,
                    ),
                )
            )
            endpoint_matched_all = bool(
                endpoint_matched_all and endpoint_matched
            )
            world_effect_cells = self._causal_spatial_cells(
                world_effect_signature
            )
            audited_cells = (
                world_effect_cells
                if allow_endpoint_matched_local and endpoint_matched
                else nonlocal_cells
            )
            audited_cell_sets.append(audited_cells)
            safe = bool(
                safe
                and factual_analysis.target_player_slot is not None
                and control_analysis.target_player_slot is not None
                and not factual_analysis.life_counter_changed
                and not control_analysis.life_counter_changed
                and not factual_analysis.dark_transition_started
                and not control_analysis.dark_transition_started
            )
            observations.append(
                {
                    "future_step": future_step,
                    "factual_frame": factual.digest,
                    "control_frame": control.digest,
                    "world_effect_signature": (
                        world_effect_signature or None
                    ),
                    "world_effect_nonlocal_cells": sorted(
                        nonlocal_cells
                    ),
                    "world_effect_audited_cells": sorted(audited_cells),
                    "endpoint_matched": endpoint_matched,
                    "ignored_player_pixels": len(
                        ignored_player_pixels
                    ),
                    "entity_player_masked_pixels": len(
                        entity_player_pixels
                    ),
                    "changed_pixels": changed_pixels,
                    "factual_player_slot": (
                        factual_analysis.target_player_slot
                    ),
                    "control_player_slot": (
                        control_analysis.target_player_slot
                    ),
                    "factual_life_counter_changed": (
                        factual_analysis.life_counter_changed
                    ),
                    "control_life_counter_changed": (
                        control_analysis.life_counter_changed
                    ),
                    "factual_dark_transition_started": (
                        factual_analysis.dark_transition_started
                    ),
                    "control_dark_transition_started": (
                        control_analysis.dark_transition_started
                    ),
                }
            )
            final_factual = factual
            final_control = control

        initial_cells = (
            audited_cell_sets[0] if audited_cell_sets else set()
        )
        common_cells = (
            set.intersection(*audited_cell_sets)
            if audited_cell_sets
            else set()
        )
        persistence_ratio = (
            len(common_cells) / len(initial_cells)
            if initial_cells
            else 0.0
        )
        localized = bool(
            common_cells
            and len(common_cells)
            <= self.config.human_prior_option_effect_max_stable_cells
        )
        confirmed = bool(
            safe
            and initial_cells
            and all(audited_cell_sets)
            and common_cells
            and localized
            and persistence_ratio >= 0.5
            and (
                not allow_endpoint_matched_local
                or endpoint_matched_all
            )
        )
        controllability = {
            "endpoint_matched": False,
            "probe_depth": (
                self.config.human_prior_option_effect_controllability_depth
            ),
            "factual_player_slot": None,
            "control_player_slot": None,
            "factual_reachable_player_slots": (),
            "control_reachable_player_slots": (),
            "newly_reachable_player_slots": (),
            "reachable_player_position_gain": 0,
            "factual_outcome_spread": 0.0,
            "control_outcome_spread": 0.0,
            "actions": (),
        }
        if confirmed and self.config.human_prior_option_effect_frontier:
            controllability = (
                self._probe_human_prior_option_controllability_gain(
                    root,
                    source_frame,
                    node.path,
                    node.durations,
                    control_path,
                    future_duration,
                    candidate_rank,
                    action_index,
                )
            )
        columns = self.config.causal_spatial_columns
        rows = self.config.causal_spatial_rows
        common_signature = bytearray(columns * rows)
        for column, row in common_cells:
            if 0 <= column < columns and 0 <= row < rows:
                common_signature[row * columns + column] = 1
        entity_entries = []
        entity_effect_cells: set[Tuple[int, int]] = set()
        entity_state_signature = ""
        memory = self.unlabeled_entity_memory
        if memory is not None and interaction_ray:
            before_observation = memory.observe(before_intervention)
            factual_observation = memory.observe(final_factual)
            control_observation = memory.observe(final_control)
            for column, row in sorted(
                common_cells.intersection(interaction_ray)
            ):
                factual_feature = memory.feature_at(
                    final_factual,
                    column,
                    row,
                    final_entity_player_pixels,
                )
                control_feature = memory.feature_at(
                    final_control,
                    column,
                    row,
                    final_entity_player_pixels,
                )
                appearance_distance = memory.feature_distance(
                    factual_feature, control_feature
                )
                changed = appearance_distance > memory.match_threshold
                if changed:
                    entity_effect_cells.add((column, row))
                entity_entries.append(
                    {
                        "cell": (column, row),
                        "before_prototype": before_observation.prototype_at(
                            column, row
                        ),
                        "factual_prototype": factual_observation.prototype_at(
                            column, row
                        ),
                        "control_prototype": control_observation.prototype_at(
                            column, row
                        ),
                        "factual_control_feature_distance": (
                            appearance_distance
                        ),
                        "appearance_changed": changed,
                    }
                )
            if entity_effect_cells:
                payload = ";".join(
                    f"{column},{row}="
                    + ",".join(
                        str(value)
                        for value in memory.feature_at(
                            final_factual,
                            column,
                            row,
                            final_entity_player_pixels,
                        )
                    )
                    for column, row in sorted(entity_effect_cells)
                )
                entity_state_signature = hashlib.sha256(
                    payload.encode("ascii")
                ).hexdigest()[:16]
        entity_effect_signature = bytearray(columns * rows)
        for column, row in entity_effect_cells:
            entity_effect_signature[row * columns + column] = 1
        entity_effect_confirmed = bool(
            confirmed and entity_effect_cells and entity_state_signature
        )
        result = {
            "candidate_rank": candidate_rank,
            "depth": node.depth,
            "path": node.path,
            "durations": node.durations,
            "control_path": control_path,
            "replaced_action_index": action_index,
            "replaced_action": replaced_action,
            "source_graph_signature": node.source_signature,
            "target_graph_signature": node.target_signature or None,
            "initial_nonlocal_world_effect_cells": len(initial_cells),
            "persistent_nonlocal_world_effect_cells": len(common_cells),
            "persistent_nonlocal_world_effect_signature": (
                bytes(common_signature).hex() if common_cells else None
            ),
            "persistence_ratio": persistence_ratio,
            "control_mode": (
                "endpoint_matched_local"
                if allow_endpoint_matched_local
                else "nonlocal"
            ),
            "endpoint_matched": endpoint_matched_all,
            "minimum_cell_pixels": (
                self.config.human_prior_option_effect_local_minimum_cell_pixels
                if allow_endpoint_matched_local
                else 1
            ),
            "localized": localized,
            "maximum_stable_cells": (
                self.config.human_prior_option_effect_max_stable_cells
            ),
            "stability_steps": (
                self.config.human_prior_option_effect_stability_steps
            ),
            "safe": safe,
            "confirmed": confirmed,
            "interaction_direction": interaction_direction,
            "interaction_ray_cells": interaction_ray,
            "entity_effect_cells": tuple(sorted(entity_effect_cells)),
            "entity_effect_signature": (
                bytes(entity_effect_signature).hex()
                if entity_effect_cells
                else None
            ),
            "entity_state_signature": entity_state_signature or None,
            "entity_effect_confirmed": entity_effect_confirmed,
            "entity_player_masked_pixels": len(
                final_entity_player_pixels
            ),
            "entity_entries": entity_entries,
            "controllability": controllability,
            "observations": observations,
        }
        self._emit(
            "human_prior_option_world_effect_action_control",
            decision=self.decision_index + 1,
            agent_visible=True,
            **result,
            **self._frame_fields(final_factual),
        )
        return result

    def _probe_human_prior_option_controllability_gain(
        self,
        root: object,
        source_frame: Frame,
        factual_path: Sequence[Action],
        durations: Sequence[int],
        control_path: Sequence[Action],
        future_duration: int,
        candidate_rank: int,
        action_index: int,
    ) -> Dict[str, Any]:
        """Compare bounded reachability after factual and ablated effects."""

        assert self.goal_prior is not None
        probe_actions = tuple(
            action
            for action in self.config.actions
            if action in (
                Action.UP,
                Action.DOWN,
                Action.LEFT,
                Action.RIGHT,
            )
        )
        release_state = getattr(self.env, "release_state", None)
        endpoint_states: List[object] = []

        def replay_endpoint(path: Sequence[Action]) -> Tuple[Frame, object]:
            self.env.load_state(root)
            endpoint = source_frame
            for action, duration in zip(path, durations):
                endpoint = self.env.step(action, duration)
            for _ in range(
                self.config.human_prior_option_effect_stability_steps
            ):
                endpoint = self.env.step(Action.NOOP, future_duration)
            state = self.env.save_state()
            endpoint_states.append(state)
            return endpoint, state

        factual_endpoint, factual_state = replay_endpoint(factual_path)
        control_endpoint, control_state = replay_endpoint(control_path)
        factual_analysis = self.goal_prior.analyze(
            source_frame, factual_endpoint
        )
        control_analysis = self.goal_prior.analyze(
            source_frame, control_endpoint
        )
        endpoints_matched = bool(
            factual_analysis.target_player_slot is not None
            and factual_analysis.target_player_slot
            == control_analysis.target_player_slot
        )
        factual_slots: set[Tuple[int, int]] = set()
        control_slots: set[Tuple[int, int]] = set()
        factual_frames: List[Frame] = []
        control_frames: List[Frame] = []
        action_rows = []
        try:
            if endpoints_matched:
                probe_depth = (
                    self.config.human_prior_option_effect_controllability_depth
                )
                for depth in range(1, probe_depth + 1):
                    for action_path in product(probe_actions, repeat=depth):
                        self.env.load_state(factual_state)
                        factual = factual_endpoint
                        factual_result = None
                        for action in action_path:
                            previous = factual
                            factual = self.env.step(action, future_duration)
                            factual_result = self.goal_prior.analyze(
                                previous, factual
                            )
                        self.env.load_state(control_state)
                        control = control_endpoint
                        control_result = None
                        for action in action_path:
                            previous = control
                            control = self.env.step(action, future_duration)
                            control_result = self.goal_prior.analyze(
                                previous, control
                            )
                        assert factual_result is not None
                        assert control_result is not None
                        if factual_result.target_player_slot is not None:
                            factual_slots.add(
                                factual_result.target_player_slot
                            )
                        if control_result.target_player_slot is not None:
                            control_slots.add(
                                control_result.target_player_slot
                            )
                        factual_frames.append(factual)
                        control_frames.append(control)
                        action_rows.append(
                            {
                                "action": (
                                    action_path[0]
                                    if len(action_path) == 1
                                    else None
                                ),
                                "path": action_path,
                                "depth": depth,
                                "action_frames": future_duration,
                                "factual_frame": factual.digest,
                                "control_frame": control.digest,
                                "factual_player_slot": (
                                    factual_result.target_player_slot
                                ),
                                "control_player_slot": (
                                    control_result.target_player_slot
                                ),
                                "factual_control_pixel_l1": (
                                    factual.mean_absolute_difference(control)
                                ),
                            }
                        )
        finally:
            if release_state is not None:
                for state in endpoint_states:
                    release_state(state)
            self.env.load_state(root)

        def maximum_spread(frames: Sequence[Frame]) -> float:
            return max(
                (
                    first.mean_absolute_difference(second)
                    for index, first in enumerate(frames)
                    for second in frames[index + 1 :]
                ),
                default=0.0,
            )

        newly_reachable = factual_slots - control_slots
        result = {
            "endpoint_matched": endpoints_matched,
            "probe_depth": (
                self.config.human_prior_option_effect_controllability_depth
            ),
            "factual_player_slot": factual_analysis.target_player_slot,
            "control_player_slot": control_analysis.target_player_slot,
            "factual_reachable_player_slots": tuple(
                sorted(factual_slots)
            ),
            "control_reachable_player_slots": tuple(
                sorted(control_slots)
            ),
            "newly_reachable_player_slots": tuple(
                sorted(newly_reachable)
            ),
            "reachable_player_position_gain": len(newly_reachable),
            "factual_outcome_spread": maximum_spread(factual_frames),
            "control_outcome_spread": maximum_spread(control_frames),
            "actions": tuple(action_rows),
        }
        self._emit(
            "human_prior_option_effect_controllability_probe",
            decision=self.decision_index + 1,
            candidate_rank=candidate_rank,
            action_index=action_index,
            path=tuple(factual_path),
            durations=tuple(durations),
            control_path=tuple(control_path),
            **result,
            **self._frame_fields(factual_endpoint),
        )
        return result

    def _promote_human_prior_option_entity_frontier(
        self,
        root: object,
        source_frame: Frame,
        node: _HumanPriorOptionNode,
        future_duration: int,
        candidate_rank: int,
        source_signature: str,
        action_controls: Sequence[Dict[str, Any]],
        endpoints: List[_HumanPriorOptionNode],
        saved_states: List[object],
    ) -> bool:
        """Promote a causal local appearance change as an entity state."""

        if not self.config.human_prior_option_entity_frontier:
            return False
        confirmed = [
            control
            for control in action_controls
            if bool(control.get("entity_effect_confirmed"))
            and control.get("entity_effect_signature")
            and control.get("entity_state_signature")
        ]
        if not confirmed:
            return False
        accepted = []
        phase_results = []
        for control in confirmed:
            phase = self._probe_human_prior_option_phase_alignment(
                root,
                source_frame,
                node,
                str(control["entity_effect_signature"]),
                candidate_rank,
            )
            phase_results.append(phase)
            if not phase["phase_equivalent"]:
                accepted.append(control)
        if not accepted:
            return False
        columns = self.config.causal_spatial_columns
        rows = self.config.causal_spatial_rows
        combined = bytearray(columns * rows)
        for control in accepted:
            for column, row in control["entity_effect_cells"]:
                combined[row * columns + column] = 1
        entity_effect_signature = bytes(combined).hex()
        entity_state_signatures = tuple(
            sorted(
                {
                    str(control["entity_state_signature"])
                    for control in accepted
                }
            )
        )
        immediate_frame = node.frame
        self.env.load_state(root)
        settled_frame = source_frame
        for action, duration in zip(node.path, node.durations):
            settled_frame = self.env.step(action, duration)
        for _ in range(
            self.config.human_prior_option_effect_stability_steps
        ):
            settled_frame = self.env.step(Action.NOOP, future_duration)
        settled_state = self.env.save_state()
        saved_states.append(settled_state)
        settled_analysis = self.goal_prior.analyze(
            source_frame, settled_frame
        )
        node = _HumanPriorOptionNode(
            state=settled_state,
            frame=settled_frame,
            path=node.path,
            durations=node.durations,
            analysis=settled_analysis,
            source_signature=node.source_signature,
            target_signature=node.target_signature,
            score=node.score,
            depth=node.depth,
            target_state_visits=node.target_state_visits,
            target_position_visits=node.target_position_visits,
            pose_action=node.pose_action,
            world_effect_signature=node.world_effect_signature,
            world_effect_state_signature=(
                self._human_prior_world_effect_state_signature(
                    settled_frame, entity_effect_signature
                )
            ),
            world_effect_changed_pixels=node.world_effect_changed_pixels,
            settling_steps=(
                self.config.human_prior_option_effect_stability_steps
            ),
            settling_frames=(
                self.config.human_prior_option_effect_stability_steps
                * future_duration
            ),
            immediate_frame_digest=immediate_frame.digest,
        )
        state_payload = (
            self.current_human_prior_world_context_signature
            + "|"
            + "|".join(entity_state_signatures)
        )
        target_world_context = hashlib.sha256(
            state_payload.encode("ascii")
        ).hexdigest()
        _, target_signature = self._human_prior_graph_signatures(
            node.analysis,
            self.current_human_prior_world_context_signature,
            target_world_context,
        )
        target_state_visits = (
            0
            if not target_signature
            else self.human_prior_graph_state_visits[target_signature]
        )
        _, option_unexpanded = self._human_prior_option_coverage(
            source_signature, node.path, node.durations
        )
        eligible = bool(
            target_signature
            and target_signature != source_signature
            and target_state_visits == 0
            and option_unexpanded
        )
        confirmed_indices = tuple(
            sorted(
                {
                    int(control["replaced_action_index"])
                    for control in accepted
                }
            )
        )
        node.confirmed_world_effect_signature = entity_effect_signature
        node.confirmed_world_context = target_world_context
        node.confirmed_action_indices = confirmed_indices
        node.confirmed_entity_state_signature = ":".join(
            entity_state_signatures
        )
        node.target_signature = target_signature
        node.target_state_visits = target_state_visits
        node.target_position_visits = (
            0
            if node.analysis.target_player_slot is None
            else self._human_prior_position_visits(
                target_signature, node.analysis.target_player_slot
            )
        )
        if eligible and not any(candidate is node for candidate in endpoints):
            endpoints.append(node)
        self._emit(
            "human_prior_option_entity_frontier_eligible",
            decision=self.decision_index + 1,
            candidate_rank=candidate_rank,
            depth=node.depth,
            path=node.path,
            durations=node.durations,
            confirmed_action_indices=confirmed_indices,
            entity_effect_signature=entity_effect_signature,
            entity_effect_cells=tuple(
                (index % columns, index // columns)
                for index, changed in enumerate(combined)
                if changed
            ),
            entity_state_signatures=entity_state_signatures,
            entity_entries=tuple(
                entry
                for control in accepted
                for entry in control["entity_entries"]
                if entry["appearance_changed"]
            ),
            phase_alignments=tuple(phase_results),
            source_world_context=(
                self.current_human_prior_world_context_signature
            ),
            target_world_context=target_world_context,
            source_graph_signature=source_signature,
            target_graph_signature=target_signature or None,
            target_graph_state_visits=target_state_visits,
            option_path_unexpanded=option_unexpanded,
            immediate_frame=immediate_frame.digest,
            settling_steps=(
                self.config.human_prior_option_effect_stability_steps
            ),
            settling_frames=(
                self.config.human_prior_option_effect_stability_steps
                * future_duration
            ),
            eligible=eligible,
            agent_visible=True,
            **self._frame_fields(node.frame),
        )
        return eligible

    def _search_human_prior_options(self) -> int:
        """Verify short action sequences from the current emulator state.

        This is an assisted-track escape hatch for a specific failure mode:
        every one-step edge is known, but a previously visited intermediate
        position must be traversed before a new endpoint becomes visible.
        Search nodes contain only opaque save-state capabilities and pixels.
        """

        assert self.frame is not None
        if (
            self.goal_prior is None
            or self.config.human_prior_option_search_depth < 2
        ):
            return 0
        actions = tuple(
            action
            for action in self.config.actions
            if action
            in (
                Action.UP,
                Action.DOWN,
                Action.LEFT,
                Action.RIGHT,
                Action.A,
                Action.B,
                *(
                    (Action.NOOP,)
                    if self.config.human_prior_option_entity_frontier
                    else ()
                ),
            )
        )
        if not actions:
            return 0
        duration = self.config.human_prior_option_search_action_frames
        if duration <= 0:
            duration = max(
                self.config.action_durations
                or (self.config.action_frames,)
            )
        source_frame = self.frame
        source_analysis = self.goal_prior.analyze(
            source_frame, source_frame
        )
        source_signature = self._human_prior_graph_signatures(
            source_analysis
        )[1]
        if not source_signature:
            return 0
        if source_signature in self.human_prior_option_exhausted_sources:
            self._emit(
                "human_prior_option_search_skipped",
                decision=self.decision_index + 1,
                reason="source_already_exhausted",
                source_graph_signature=source_signature,
                **self._frame_fields(source_frame),
            )
            return 0

        root = self.env.save_state()
        saved_states = [root]
        retained_state_ids: set[int] = set()
        release_state = getattr(self.env, "release_state", None)
        root_node = _HumanPriorOptionNode(
            state=root,
            frame=source_frame,
            path=(),
            durations=(),
            analysis=source_analysis,
            source_signature=source_signature,
            target_signature=source_signature,
            score=0.0,
            depth=0,
            target_state_visits=self.human_prior_graph_state_visits[
                source_signature
            ],
            target_position_visits=(
                0
                if source_analysis.target_player_slot is None
                else self._human_prior_position_visits(
                    source_signature,
                    source_analysis.target_player_slot,
                )
            ),
            pose_action=self.current_pose_action,
        )
        parents = [root_node]
        endpoints: List[_HumanPriorOptionNode] = []
        effect_nodes: List[_HumanPriorOptionNode] = []
        branches_verified = 0
        active_failure = False
        self._emit(
            "human_prior_option_search_started",
            decision=self.decision_index + 1,
            source_state_id=self._state_id(root),
            source_graph_signature=source_signature,
            maximum_depth=self.config.human_prior_option_search_depth,
            beam_width=self.config.human_prior_option_search_beam_width,
            actions=actions,
            action_frames=duration,
            **self._frame_fields(source_frame),
        )
        try:
            for depth in range(
                1, self.config.human_prior_option_search_depth + 1
            ):
                self.env.load_state(root)
                neutral_target = source_frame
                for _neutral_step in range(depth):
                    neutral_target = self.env.step(
                        Action.NOOP, duration
                    )
                self._emit(
                    "human_prior_option_neutral_verified",
                    decision=self.decision_index + 1,
                    depth=depth,
                    path=(Action.NOOP,) * depth,
                    durations=(duration,) * depth,
                    source_state_id=self._state_id(root),
                    env_step_seq=getattr(
                        self.env, "last_step_seq", None
                    ),
                    **self._frame_fields(neutral_target),
                )
                depth_candidates: List[_HumanPriorOptionNode] = []
                for parent in parents:
                    for action in actions:
                        self.env.load_state(parent.state)
                        target = self.env.step(action, duration)
                        state = self.env.save_state()
                        saved_states.append(state)
                        path = (*parent.path, action)
                        durations = (*parent.durations, duration)
                        analysis = self.goal_prior.analyze(
                            source_frame, target
                        )
                        _, target_signature = (
                            self._human_prior_graph_signatures(
                                analysis,
                                self.current_human_prior_world_context_signature,
                                self.current_human_prior_world_context_signature,
                            )
                        )
                        target_state_visits = (
                            0
                            if not target_signature
                            else self.human_prior_graph_state_visits[
                                target_signature
                            ]
                        )
                        target_position_visits = (
                            0
                            if analysis.target_player_slot is None
                            else self._human_prior_position_visits(
                                target_signature,
                                analysis.target_player_slot,
                            )
                        )
                        option_visits, option_unexpanded = (
                            self._human_prior_option_coverage(
                                source_signature, path, durations
                            )
                        )
                        (
                            option_spatial_signature,
                            option_changed_pixels,
                            _option_change_centroid,
                        ) = self._causal_spatial_effect(
                            target, neutral_target
                        )
                        option_world_effect_signature = (
                            self._human_prior_world_effect_signature(
                                option_spatial_signature,
                                analysis,
                                target,
                                action=action,
                                allow_nonlocal=True,
                            )
                        )
                        option_world_effect_state_signature = (
                            self._human_prior_world_effect_state_signature(
                                target, option_world_effect_signature
                            )
                        )
                        option_nonlocal_world_effect_cells = (
                            self._human_prior_nonlocal_world_effect_cells(
                                option_world_effect_signature,
                                analysis,
                                target,
                            )
                        )
                        state_novelty = 1.0 / math.sqrt(
                            target_state_visits + 1
                        )
                        position_novelty = 1.0 / math.sqrt(
                            target_position_visits + 1
                        )
                        score = (
                            analysis.total_reward
                            + 4.0 * position_novelty
                            + 2.0 * state_novelty
                            + 0.25
                            * self.novelty.score(self._signature(target))
                            - 0.05 * depth
                            - 2.0 * option_visits
                        )
                        node = _HumanPriorOptionNode(
                            state=state,
                            frame=target,
                            path=path,
                            durations=durations,
                            analysis=analysis,
                            source_signature=source_signature,
                            target_signature=target_signature,
                            score=score,
                            depth=depth,
                            target_state_visits=target_state_visits,
                            target_position_visits=target_position_visits,
                            pose_action=self._resulting_pose_action(
                                parent.pose_action, action
                            ),
                            world_effect_signature=(
                                option_world_effect_signature
                            ),
                            world_effect_state_signature=(
                                option_world_effect_state_signature
                            ),
                            world_effect_changed_pixels=(
                                option_changed_pixels
                            ),
                        )
                        depth_candidates.append(node)
                        branches_verified += 1
                        endpoint_eligible = bool(
                            depth >= 2
                            and option_unexpanded
                            and not analysis.life_counter_changed
                            and not analysis.dark_transition_started
                            and (
                                analysis.milestone_reward > 0.0
                                or (
                                    target_signature
                                    and target_signature != source_signature
                                    and analysis.target_player_slot is not None
                                    and target_position_visits == 0
                                )
                            )
                        )
                        if endpoint_eligible:
                            endpoints.append(node)
                        if option_world_effect_signature:
                            effect_nodes.append(node)
                        self._emit(
                            "human_prior_option_branch_verified",
                            decision=self.decision_index + 1,
                            branch_index=branches_verified,
                            depth=depth,
                            path=path,
                            durations=durations,
                            source_state_id=self._state_id(root),
                            parent_state_id=self._state_id(parent.state),
                            state_id=self._state_id(state),
                            source_graph_signature=source_signature,
                            target_graph_signature=(
                                target_signature or None
                            ),
                            target_graph_state_visits=(
                                target_state_visits
                            ),
                            target_player_position_visits=(
                                target_position_visits
                            ),
                            source_pose_action=parent.pose_action,
                            target_pose_action=node.pose_action,
                            option_path_visits_before=option_visits,
                            option_path_unexpanded=option_unexpanded,
                            human_prior_option_world_effect_signature=(
                                option_world_effect_signature or None
                            ),
                            human_prior_option_world_effect_state_signature=(
                                option_world_effect_state_signature or None
                            ),
                            human_prior_option_world_effect_changed_pixels=(
                                option_changed_pixels
                            ),
                            human_prior_option_nonlocal_world_effect_cells=(
                                sorted(option_nonlocal_world_effect_cells)
                            ),
                            human_prior_option_nonlocal_world_effect_cell_count=(
                                len(option_nonlocal_world_effect_cells)
                            ),
                            endpoint_eligible=endpoint_eligible,
                            score=score,
                            agent_visible=True,
                            **analysis.telemetry(),
                            **self._frame_fields(target),
                        )
                if not depth_candidates:
                    break
                deduplicated: Dict[
                    Tuple[str, Optional[Action], str],
                    _HumanPriorOptionNode,
                ] = {}
                for node in depth_candidates:
                    key = (
                        node.target_signature or node.frame.digest,
                        node.pose_action,
                        node.world_effect_state_signature,
                    )
                    previous = deduplicated.get(key)
                    if previous is None or node.score > previous.score:
                        deduplicated[key] = node
                parents = sorted(
                    deduplicated.values(),
                    key=lambda node: (
                        node.analysis.milestone_reward,
                        node.target_position_visits == 0,
                        node.target_state_visits == 0,
                        node.score,
                        -node.depth,
                    ),
                    reverse=True,
                )[: self.config.human_prior_option_search_beam_width]

            if (
                self.config.human_prior_option_effect_stability_steps > 0
                and effect_nodes
            ):
                distinct_effect_nodes: Dict[
                    Tuple[
                        str,
                        str,
                        Optional[Tuple[int, int]],
                        Optional[Action],
                    ],
                    _HumanPriorOptionNode,
                ] = {}
                for node in effect_nodes:
                    effect_key = (
                        node.world_effect_signature,
                        node.world_effect_state_signature,
                        node.analysis.target_player_slot,
                        node.pose_action,
                    )
                    previous = distinct_effect_nodes.get(
                        effect_key
                    )
                    if previous is None or node.score > previous.score:
                        distinct_effect_nodes[effect_key] = node
                effect_probe_candidates = sorted(
                    distinct_effect_nodes.values(),
                    key=lambda node: (
                        node.analysis.target_player_slot is not None
                        and not node.analysis.life_counter_changed
                        and not node.analysis.dark_transition_started,
                        bool(
                            self._human_prior_nonlocal_world_effect_cells(
                                node.world_effect_signature,
                                node.analysis,
                                node.frame,
                            )
                        ),
                        node.path[-1] in (Action.A, Action.B),
                        -len(
                            self._human_prior_nonlocal_world_effect_cells(
                                node.world_effect_signature,
                                node.analysis,
                                node.frame,
                            )
                        ),
                        node.score,
                    ),
                    reverse=True,
                )[: self.config.human_prior_option_effect_probe_limit]
                for candidate_rank, node in enumerate(
                    effect_probe_candidates, 1
                ):
                    stability = self._probe_human_prior_option_world_effect(
                        root,
                        source_frame,
                        node,
                        duration,
                        candidate_rank,
                    )
                    if stability["stable"]:
                        stable_effect_signature = str(
                            stability.get(
                                "stable_world_effect_signature"
                            )
                            or ""
                        )
                        phase_alignment = (
                            None
                            if not stable_effect_signature
                            or self.config.human_prior_option_effect_phase_offsets
                            <= 0
                            else self._probe_human_prior_option_phase_alignment(
                                root,
                                source_frame,
                                node,
                                stable_effect_signature,
                                candidate_rank,
                            )
                        )
                        if (
                            phase_alignment is not None
                            and phase_alignment["phase_equivalent"]
                        ):
                            continue
                        action_controls = []
                        for action_index in range(len(node.path)):
                            action_controls.append(
                                self._probe_human_prior_option_action_control(
                                    root,
                                    source_frame,
                                    node,
                                    duration,
                                    candidate_rank,
                                    action_index,
                                )
                            )
                        confirmed_indices = tuple(
                            int(control["replaced_action_index"])
                            for control in action_controls
                            if control["confirmed"]
                            and int(
                                control["controllability"].get(
                                    "reachable_player_position_gain", 0
                                )
                            )
                            > 0
                        )
                        if (
                            self.config.human_prior_option_effect_frontier
                            and phase_alignment is not None
                            and not phase_alignment["phase_equivalent"]
                            and confirmed_indices
                            and stable_effect_signature
                        ):
                            target_world_context = (
                                self._next_human_prior_world_context(
                                    self.current_human_prior_world_context_signature,
                                    stable_effect_signature,
                                )
                            )
                            _, target_signature = (
                                self._human_prior_graph_signatures(
                                    node.analysis,
                                    self.current_human_prior_world_context_signature,
                                    target_world_context,
                                )
                            )
                            target_state_visits = (
                                0
                                if not target_signature
                                else self.human_prior_graph_state_visits[
                                    target_signature
                                ]
                            )
                            _, option_unexpanded = (
                                self._human_prior_option_coverage(
                                    source_signature,
                                    node.path,
                                    node.durations,
                                )
                            )
                            eligible = bool(
                                target_signature
                                and target_signature != source_signature
                                and target_state_visits == 0
                                and option_unexpanded
                            )
                            node.confirmed_world_effect_signature = (
                                stable_effect_signature
                            )
                            node.confirmed_world_context = (
                                target_world_context
                            )
                            node.confirmed_action_indices = (
                                confirmed_indices
                            )
                            node.target_signature = target_signature
                            node.target_state_visits = target_state_visits
                            if eligible and not any(
                                candidate is node
                                for candidate in endpoints
                            ):
                                endpoints.append(node)
                            self._emit(
                                "human_prior_option_effect_frontier_eligible",
                                decision=self.decision_index + 1,
                                candidate_rank=candidate_rank,
                                depth=node.depth,
                                path=node.path,
                                durations=node.durations,
                                confirmed_action_indices=(
                                    confirmed_indices
                                ),
                                world_effect_signature=(
                                    stable_effect_signature
                                ),
                                source_world_context=(
                                    self.current_human_prior_world_context_signature
                                ),
                                target_world_context=(
                                    target_world_context
                                ),
                                source_graph_signature=(
                                    source_signature
                                ),
                                target_graph_signature=(
                                    target_signature or None
                                ),
                                target_graph_state_visits=(
                                    target_state_visits
                                ),
                                option_path_unexpanded=(
                                    option_unexpanded
                                ),
                                eligible=eligible,
                                agent_visible=True,
                                **self._frame_fields(node.frame),
                            )
                        if not stability["local_candidate"]:
                            self._promote_human_prior_option_entity_frontier(
                                root,
                                source_frame,
                                node,
                                duration,
                                candidate_rank,
                                source_signature,
                                action_controls,
                                endpoints,
                                saved_states,
                            )
                    if (
                        self.config.human_prior_option_effect_local_controls
                        and stability["local_candidate"]
                    ):
                        local_action_controls = []
                        for action_index in range(len(node.path)):
                            local_action_controls.append(
                                self._probe_human_prior_option_action_control(
                                    root,
                                    source_frame,
                                    node,
                                    duration,
                                    candidate_rank,
                                    action_index,
                                    allow_endpoint_matched_local=True,
                                )
                            )
                        self._promote_human_prior_option_entity_frontier(
                            root,
                            source_frame,
                            node,
                            duration,
                            candidate_rank,
                            source_signature,
                            local_action_controls,
                            endpoints,
                            saved_states,
                        )

            if not endpoints:
                self.human_prior_option_exhausted_sources.add(
                    source_signature
                )
                self._emit(
                    "human_prior_option_search_completed",
                    decision=self.decision_index + 1,
                    branches_verified=branches_verified,
                    eligible_endpoints=0,
                    archive_branches_added=0,
                    reason="no_unexpanded_endpoint",
                    **self._frame_fields(source_frame),
                )
                return 0
            ordinary_endpoints = [
                node
                for node in endpoints
                if not node.confirmed_world_effect_signature
            ]
            selection_endpoints = ordinary_endpoints or endpoints
            selection_key = lambda node: (
                node.analysis.milestone_reward,
                node.target_position_visits == 0,
                node.analysis.total_reward,
                node.target_state_visits == 0,
                node.score,
                -node.depth,
            )
            selected = max(
                selection_endpoints,
                key=selection_key,
            )
            entity_representatives: Dict[
                str, _HumanPriorOptionNode
            ] = {}
            for node in endpoints:
                if not (
                    node.confirmed_entity_state_signature
                    and node.confirmed_world_context
                ):
                    continue
                previous = entity_representatives.get(
                    node.confirmed_world_context
                )
                if previous is None or selection_key(node) > selection_key(
                    previous
                ):
                    entity_representatives[
                        node.confirmed_world_context
                    ] = node
            additional_entity_endpoints = sorted(
                (
                    node
                    for node in entity_representatives.values()
                    if node is not selected
                    and node.confirmed_world_context
                    != selected.confirmed_world_context
                ),
                key=selection_key,
                reverse=True,
            )
            available_slots = max(
                1, self.config.archive_capacity - len(self.archive)
            )
            archived_endpoints = [selected]
            archived_endpoints.extend(
                additional_entity_endpoints[: max(0, available_slots - 1)]
            )
            for archived in archived_endpoints:
                archived_frontier_signature = (
                    self._new_provisional_signature()
                )
                archived_target_world_context = (
                    archived.confirmed_world_context
                    or self.current_human_prior_world_context_signature
                )
                archived_world_effect_signature = (
                    archived.confirmed_world_effect_signature
                    or archived.world_effect_signature
                )
                branch = _ArchivedBranch(
                    state=archived.state,
                    frame=archived.frame,
                    plan=NeuralPlan(
                        archived.path,
                        archived.durations,
                        archived.score,
                        0.0,
                    ),
                    score=archived.score,
                    scene=self._scene_signature(archived.frame),
                    created=self.decision_index,
                    origin_signature=self.current_frontier_signature,
                    frontier_signature=archived_frontier_signature,
                    causal_context_signature=(
                        self.current_causal_context_signature
                    ),
                    target_causal_context_signature=(
                        self.current_causal_context_signature
                    ),
                    pose_action=archived.pose_action,
                    goal_heart_slots=archived.analysis.target_present,
                    goal_progress_reward=(
                        archived.analysis.milestone_reward
                    ),
                    goal_remaining_hearts=(
                        archived.analysis.remaining_hearts
                    ),
                    goal_total_hearts=len(archived.analysis.known_slots),
                    goal_chest_slot=(
                        archived.analysis.target_chest_slot
                        or archived.analysis.source_chest_slot
                    ),
                    goal_player_slot=archived.analysis.target_player_slot,
                    goal_chest_obtained=archived.analysis.chest_obtained,
                    parent_state_id=self._state_id(root),
                    parent_frame_digest=source_frame.digest,
                    parent_decision=self.decision_index,
                    search_depth=self.current_search_depth + archived.depth,
                    goal_source_signature=archived.source_signature,
                    goal_target_signature=archived.target_signature,
                    goal_source_world_context=(
                        self.current_human_prior_world_context_signature
                    ),
                    goal_target_world_context=(
                        archived_target_world_context
                    ),
                    goal_world_effect_signature=(
                        archived.confirmed_world_effect_signature
                    ),
                    human_prior_verified_option=True,
                    human_prior_option_world_effect_signature=(
                        archived_world_effect_signature
                    ),
                    human_prior_option_entity_state_signature=(
                        archived.confirmed_entity_state_signature
                    ),
                )
                self.archive.append(branch)
                retained_state_ids.add(id(archived.state))
                self._emit(
                    "human_prior_option_archive_added",
                    decision=self.decision_index + 1,
                    state_id=self._state_id(archived.state),
                    parent_state_id=self._state_id(root),
                    search_depth=branch.search_depth,
                    option_depth=archived.depth,
                    path=archived.path,
                    durations=archived.durations,
                    source_graph_signature=archived.source_signature,
                    target_graph_signature=(
                        archived.target_signature or None
                    ),
                    target_graph_state_visits=(
                        archived.target_state_visits
                    ),
                    target_player_position_visits=(
                        archived.target_position_visits
                    ),
                    human_prior_option_world_effect_signature=(
                        archived_world_effect_signature or None
                    ),
                    human_prior_option_effect_frontier=bool(
                        archived.confirmed_world_effect_signature
                    ),
                    human_prior_option_entity_frontier=bool(
                        archived.confirmed_entity_state_signature
                    ),
                    human_prior_option_entity_state_signature=(
                        archived.confirmed_entity_state_signature or None
                    ),
                    human_prior_option_effect_confirmed_action_indices=(
                        archived.confirmed_action_indices
                    ),
                    human_prior_option_settling_steps=(
                        archived.settling_steps
                    ),
                    human_prior_option_settling_frames=(
                        archived.settling_frames
                    ),
                    human_prior_option_immediate_frame=(
                        archived.immediate_frame_digest or None
                    ),
                    human_prior_world_source_context=(
                        self.current_human_prior_world_context_signature
                    ),
                    human_prior_world_target_context=(
                        archived_target_world_context
                    ),
                    human_prior_option_world_effect_changed_pixels=(
                        archived.world_effect_changed_pixels
                    ),
                    selected_primary=(archived is selected),
                    score=archived.score,
                    archive_size=len(self.archive),
                    agent_visible=True,
                    **archived.analysis.telemetry(),
                    **self._frame_fields(archived.frame),
                )
            self._emit(
                "human_prior_option_search_completed",
                decision=self.decision_index + 1,
                branches_verified=branches_verified,
                eligible_endpoints=len(endpoints),
                ordinary_eligible_endpoints=len(ordinary_endpoints),
                confirmed_effect_fallback_used=bool(
                    selected.confirmed_world_effect_signature
                ),
                archive_branches_added=len(archived_endpoints),
                distinct_entity_contexts_archived=sum(
                    bool(node.confirmed_entity_state_signature)
                    for node in archived_endpoints
                ),
                selected_depth=selected.depth,
                selected_path=selected.path,
                selected_durations=selected.durations,
                selected_score=selected.score,
                **self._frame_fields(selected.frame),
            )
            return len(archived_endpoints)
        except BaseException:
            active_failure = True
            raise
        finally:
            try:
                self.env.load_state(root)
                if release_state is not None:
                    released: set[int] = set()
                    for state in reversed(saved_states):
                        state_identity = id(state)
                        if state_identity in released:
                            continue
                        released.add(state_identity)
                        if state_identity in retained_state_ids:
                            continue
                        release_state(state)
            except Exception as cleanup_error:
                self._emit(
                    "human_prior_option_cleanup_failed",
                    decision=self.decision_index + 1,
                    error_type=type(cleanup_error).__name__,
                    error=str(cleanup_error),
                    active_failure=active_failure,
                )
                if not active_failure:
                    raise

    def _human_prior_semantic_frontier_novel(
        self,
        source_signature: str,
        target_signature: str,
        action: Action,
        duration: int,
    ) -> bool:
        if (
            not self.config.human_prior_best_first_archive
            or not source_signature
        ):
            return False
        _visits, edge_unexpanded = self._human_prior_graph_edge_coverage(
            source_signature, action, duration
        )
        state_changed = bool(
            target_signature and target_signature != source_signature
        )
        target_unvisited = bool(
            state_changed
            and not self.human_prior_graph_state_visits[target_signature]
        )
        return target_unvisited or (edge_unexpanded and state_changed)

    def _record_human_prior_outcome(
        self,
        analysis: HeartGoalAnalysis,
        source_signature: str,
        action: Action,
        duration: int,
        source_frame: Frame,
        target_frame: Frame,
    ) -> None:
        if analysis.chest_completed:
            self._emit(
                "human_prior_chest_completed",
                decision=self.decision_index + 1,
                action=action,
                action_frames=duration,
                agent_visible=True,
                **analysis.telemetry(),
                **self._frame_fields(target_frame),
            )
        if analysis.dark_transition_started:
            causal_decision = self.decision_index + 1
            causal_signature = source_signature
            causal_action = action
            causal_duration = duration
            causal_frame = source_frame.digest
            trace = self.active_temporal_option
            delayed_cause = bool(
                trace is not None
                and trace.choice is not None
                and trace.causal_evidence
            )
            if delayed_cause:
                assert trace is not None and trace.choice is not None
                causal_signature, causal_action, causal_duration = trace.choice
                causal_decision = (
                    trace.initiation_decision
                    if trace.initiation_decision is not None
                    else trace.start_decision
                )
                causal_frame = (
                    trace.initiation_frame_digest or source_frame.digest
                )
            if self.pending_life_hazard_choice is None:
                self.pending_life_hazard_choice = (
                    causal_decision,
                    causal_signature,
                    causal_action,
                    causal_duration,
                    causal_frame,
                )
            self._emit(
                "human_prior_dark_transition_observed",
                decision=self.decision_index + 1,
                action=action,
                action_frames=duration,
                source_behavioral_signature=source_signature,
                source_frame=source_frame.digest,
                target_frame=target_frame.digest,
                causal_decision=causal_decision,
                causal_action=causal_action,
                causal_action_frames=causal_duration,
                causal_behavioral_signature=causal_signature,
                causal_frame=causal_frame,
                causal_source=(
                    "active_temporal_option"
                    if delayed_cause
                    else "dark_transition_action"
                ),
                agent_visible=True,
            )
            return
        if analysis.life_loss_confirmed:
            pending = self.pending_life_hazard_choice
            causal_decision = self.decision_index + 1
            causal_signature = source_signature
            causal_action = action
            causal_duration = duration
            causal_frame = source_frame.digest
            if pending is not None:
                (
                    causal_decision,
                    causal_signature,
                    causal_action,
                    causal_duration,
                    causal_frame,
                ) = pending
            choice = (causal_signature, causal_action, causal_duration)
            learned_value = self._record_temporal_option_sample(
                choice, -self.config.human_prior_life_loss_penalty
            )
            recovery = None
            recovery_source = None
            milestone_hazard_value = None
            milestone_hazard_samples = 0
            milestone_checkpoint = self.pending_goal_milestone_checkpoint
            if milestone_checkpoint is not None:
                recovery = milestone_checkpoint
                recovery_source = "goal_milestone"
                self.pending_goal_milestone_checkpoint = None
                milestone_hazard_value = self._record_temporal_option_sample(
                    milestone_checkpoint.choice,
                    -self.config.human_prior_life_loss_penalty,
                )
                milestone_hazard_samples = self.temporal_option_samples[
                    milestone_checkpoint.choice
                ]
            current_trace = self.active_temporal_option
            if (
                recovery is None
                and current_trace is not None
                and current_trace.choice == choice
                and current_trace.recovery_checkpoint is not None
            ):
                recovery = current_trace.recovery_checkpoint
                recovery_source = "causal_option"
                current_trace.recovery_checkpoint = None
            if recovery is not None:
                if self.pending_life_recovery is not None:
                    self._release_life_hazard_checkpoint(
                        self.pending_life_recovery,
                        "superseded_by_new_life_loss",
                    )
                self.pending_life_recovery = recovery
                self.pending_recovery_cause = "life_loss"
            self._emit(
                "human_prior_life_loss_confirmed",
                decision=self.decision_index + 1,
                confirmation_action=action,
                confirmation_action_frames=duration,
                causal_decision=causal_decision,
                causal_action=causal_action,
                causal_action_frames=causal_duration,
                causal_behavioral_signature=causal_signature,
                causal_frame=causal_frame,
                target_frame=target_frame.digest,
                learned_hazard_value=learned_value,
                learned_hazard_samples=self.temporal_option_samples[choice],
                recovery_checkpoint_available=recovery is not None,
                recovery_checkpoint_source=recovery_source,
                recovery_state_id=(
                    None if recovery is None else recovery.state_id
                ),
                milestone_hazard_choice=(
                    None
                    if milestone_checkpoint is None
                    else milestone_checkpoint.choice
                ),
                milestone_hazard_value=milestone_hazard_value,
                milestone_hazard_samples=milestone_hazard_samples,
                agent_visible=True,
                **analysis.telemetry(),
            )
            self.pending_life_hazard_choice = None
            return
        if (
            analysis.target_life_signature is not None
            and self.pending_life_hazard_choice is not None
        ):
            self._emit(
                "human_prior_dark_transition_cleared",
                decision=self.decision_index + 1,
                reason="life_counter_unchanged",
                pending_causal_decision=self.pending_life_hazard_choice[0],
                agent_visible=True,
                **self._frame_fields(target_frame),
            )
            self.pending_life_hazard_choice = None

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

    def _clone_state_for_independent_owner(
        self, source_state: object, restore_state: object
    ) -> object:
        """Clone a save state so two long-lived owners never share a handle."""

        release_state = getattr(self.env, "release_state", None)
        self.env.load_state(source_state)
        cloned_state = self.env.save_state()
        try:
            self.env.load_state(restore_state)
        except BaseException:
            if release_state is not None:
                release_state(cloned_state)
            raise
        return cloned_state

    @staticmethod
    def _signature(frame: Frame) -> str:
        return signature_key(frame.coarse_signature())

    @staticmethod
    def _scene_signature(frame: Frame) -> str:
        return signature_key(frame.coarse_signature(columns=3, rows=3))

    @staticmethod
    def _affordance_checkpoint_key(
        frame: Frame,
        causal_context_signature: str,
        actions: Sequence[Action],
        pose_action: Optional[Action],
    ) -> Tuple[str, Optional[Action], Tuple[Action, ...]]:
        del causal_context_signature
        return (
            signature_key(frame.coarse_signature()),
            pose_action,
            tuple(sorted(set(actions), key=lambda action: action.value)),
        )

    @classmethod
    def _causal_outcome_key(
        cls, frame: Frame, pose_action: Optional[Action]
    ) -> Tuple[str, Optional[Action]]:
        return cls._signature(frame), pose_action

    @staticmethod
    def _resulting_pose_action(
        current_pose_action: Optional[Action], action: Action
    ) -> Optional[Action]:
        if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            return action
        return current_pose_action

    def reset(self, initial_frame: Optional[Frame] = None) -> Frame:
        self.clear_archive()
        self.frame = self.env.reset() if initial_frame is None else initial_frame
        self.novelty = VisualNovelty()
        self.novelty.observe(self._signature(self.frame))
        self.action_counts = Counter()
        self.duration_counts = Counter()
        self.action_duration_counts = Counter()
        self.action_effect_values = {}
        self.action_effect_samples = Counter()
        self.causal_spatial_visits = Counter()
        self.human_prior_graph_edge_visits = Counter()
        self.human_prior_graph_state_visits = Counter()
        self.human_prior_option_visits = Counter()
        self.human_prior_player_position_visits = Counter()
        self.human_prior_phase_player_position_visits = Counter()
        self.human_prior_option_exhausted_sources: set[str] = set()
        self.human_prior_graph_recovery_pending = False
        self.current_human_prior_world_context_signature = (
            "human-prior-world-root"
        )
        self.last_causal_cell_progress_decision = None
        self.behavioral_edge_visits = Counter()
        self.causal_spatial_cell_visits: CounterType[Tuple[int, int]] = Counter()
        persistent_values = self._persistent_cell_values(self.frame)
        self.persistent_change_baseline = list(persistent_values)
        self.persistent_change_value_counts = [
            Counter({value: 1}) for value in persistent_values
        ]
        self.persistent_change_candidates = {}
        self.persistent_change_cells = {}
        self.persistent_change_mismatches = Counter()
        self.discovered_interaction_actions: set[Action] = set()
        self.discovered_interaction_durations: Dict[Action, set[int]] = {}
        self.archive_branch_restores: CounterType[
            Tuple[str, Optional[Action], Tuple[Action, ...]]
        ] = Counter()
        self.causal_outcome_restores: CounterType[
            Tuple[str, Optional[Action]]
        ] = Counter()
        self.causal_outcome_contexts: set[str] = set()
        self.current_causal_context_signature = "causal-context-root"
        self.current_pose_action: Optional[Action] = None
        self.last_action = None
        self.last_duration = None
        self.last_action_was_causal_spatial = False
        self.action_streak = 0
        self.scene_visits = Counter()
        self.scene_action_probes = Counter()
        self.delayed_return_costs = Counter()
        self.transition_history = []
        self.visual_last_visit = {self._signature(self.frame): 0}
        self.delayed_return_recovery = False
        self.delayed_return_loop_start = None
        self.autonomous_grace_remaining = 0
        self.autonomous_intervention_pending = False
        self.causal_observation_intervention_pending = False
        self.anticipated_transition_observations_remaining = 0
        self.anticipated_transition_observation_duration = (
            self.config.action_frames
        )
        self.dark_transition_active = False
        self.dark_transition_start_decision = None
        self.pending_novel_room_frame: Optional[Frame] = None
        self.known_scene_return_recovery_pending = False
        self.bright_scene_memory = (
            [self._persistent_cell_values(self.frame)]
            if self._mean_frame_intensity(self.frame)
            > self.config.dark_transition_intensity_threshold
            else []
        )
        self.visual_clusters = []
        self.frame_clusters = {}
        self.frame_latents = {}
        self.cluster_serial = 0
        self.behavior_clusters = []
        self.visual_probe_counts = Counter()
        self.behavior_cluster_serial = 0
        self.provisional_state_serial = 0
        self._abstract_signature(self.frame)
        initial_signature = self._new_provisional_signature()
        self.current_frontier_signature = initial_signature
        self.behavior_visits = Counter()
        self.pending_option_choice = None
        self.pending_option_decision = None
        self.pending_option_frame_digest = None
        self.pending_option_recovery_checkpoint = None
        self.pending_option_causal_evidence = False
        self.pending_option_counterfactual = None
        self.active_temporal_option = None
        self.temporal_option_values = {}
        self.temporal_option_samples = Counter()
        self.temporal_option_action_values = {}
        self.temporal_option_action_samples = Counter()
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
        self.current_search_depth = 0
        self.last_navigation_change_decision = None
        self.pending_life_hazard_choice = None
        self.pending_life_recovery = None
        self.pending_recovery_cause = None
        self.pending_goal_milestone_checkpoint = None
        self._reset_goal_prior()
        self._reset_unlabeled_entity_memory(self.frame)
        self._calibrate_goal_prior(self.frame)
        initial_goal_signature = self._current_human_prior_graph_signature()
        if initial_goal_signature:
            self.human_prior_graph_state_visits[
                initial_goal_signature
            ] += 1
        if (
            self.goal_prior is not None
            and self.goal_prior.current_player_slot is not None
        ):
            self._record_human_prior_player_position(
                initial_goal_signature,
                self.goal_prior.current_player_slot,
            )
        known_scene_state = self.env.save_state()
        self.known_scene_recovery_checkpoint = _LifeHazardCheckpoint(
            state=known_scene_state,
            frame=self.frame,
            choice=(initial_signature, Action.NOOP, 0),
            decision=0,
            frontier_signature=initial_signature,
            causal_context_signature=self.current_causal_context_signature,
            scene=self.current_scene,
            pose_action=self.current_pose_action,
            last_action=None,
            last_duration=None,
            action_streak=0,
            goal_heart_slots=(
                ()
                if self.goal_prior is None
                else tuple(sorted(self.goal_prior.current_present))
            ),
            goal_player_slot=(
                None
                if self.goal_prior is None
                else self.goal_prior.current_player_slot
            ),
            goal_chest_obtained=(
                False
                if self.goal_prior is None
                else self.goal_prior.chest_obtained
            ),
            human_prior_world_context_signature=(
                self.current_human_prior_world_context_signature
            ),
            kind="known_scene_root",
            state_id=self._state_id(known_scene_state),
        )
        self._emit(
            "known_scene_recovery_checkpoint_created",
            decision=0,
            state_id=self._state_id(known_scene_state),
            **self._frame_fields(self.frame),
        )
        self._emit(
            "agent_reset",
            decision=0,
            action_counts={},
            **self._frame_fields(self.frame),
        )
        return self.frame

    def clear_archive(self) -> None:
        if getattr(self, "active_temporal_option", None) is not None:
            self._discard_temporal_option("agent_reset_or_close")
        self._discard_pending_temporal_option("agent_reset_or_close")
        life_recovery = getattr(self, "pending_life_recovery", None)
        if life_recovery is not None:
            self._release_life_hazard_checkpoint(
                life_recovery, "agent_reset_or_close"
            )
            self.pending_life_recovery = None
            self.pending_recovery_cause = None
        milestone_recovery = getattr(
            self, "pending_goal_milestone_checkpoint", None
        )
        if milestone_recovery is not None:
            self._release_life_hazard_checkpoint(
                milestone_recovery, "agent_reset_or_close"
            )
            self.pending_goal_milestone_checkpoint = None
        known_scene_recovery = getattr(
            self, "known_scene_recovery_checkpoint", None
        )
        if known_scene_recovery is not None:
            self._release_life_hazard_checkpoint(
                known_scene_recovery, "agent_reset_or_close"
            )
            self.known_scene_recovery_checkpoint = None
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

    def _action_penalty_components(
        self, action: Action, duration: Optional[int] = None
    ) -> Dict[str, float]:
        duration = self.config.action_frames if duration is None else duration
        coverage = self.config.action_coverage_weight * math.sqrt(self.action_counts[action])
        duration_coverage = self.config.duration_coverage_weight * math.sqrt(
            self.action_duration_counts[(action, duration)]
        )
        consecutive_raw = (
            self.config.consecutive_repeat_weight * self.action_streak
            if action == self.last_action and duration == self.last_duration
            else 0.0
        )
        consecutive = (
            consecutive_raw
            if self.config.consecutive_repeat_penalty_cap is None
            else min(consecutive_raw, self.config.consecutive_repeat_penalty_cap)
        )
        return_penalty_raw = 0.0
        if self.current_scene is not None:
            return_penalty_raw = self.config.delayed_return_weight * math.sqrt(
                self.delayed_return_costs[(self.current_scene, action, duration)]
            )
        return_penalty = (
            return_penalty_raw
            if self.config.delayed_return_penalty_cap is None
            else min(return_penalty_raw, self.config.delayed_return_penalty_cap)
        )
        return {
            "action_coverage_penalty": coverage,
            "duration_coverage_penalty": duration_coverage,
            "consecutive_repeat_penalty_raw": consecutive_raw,
            "consecutive_repeat_penalty": consecutive,
            "delayed_return_penalty_raw": return_penalty_raw,
            "delayed_return_penalty": return_penalty,
            "action_penalty": (
                coverage + duration_coverage + consecutive + return_penalty
            ),
        }

    def _action_penalty(self, action: Action, duration: Optional[int] = None) -> float:
        return self._action_penalty_components(action, duration)["action_penalty"]

    def _action_duration_count_rows(self) -> List[Dict[str, Any]]:
        return [
            {"action": action, "action_frames": duration, "count": count}
            for (action, duration), count in sorted(
                self.action_duration_counts.items(),
                key=lambda item: (item[0][0].value, item[0][1]),
            )
        ]

    def _record_action_effect(
        self, source_signature: str, action: Action, contrast: float
    ) -> Tuple[float, int]:
        key = (source_signature, action)
        sample = min(
            1.0,
            contrast / max(self.config.action_equivalence_threshold, 1e-12),
        )
        count = self.action_effect_samples[key] + 1
        previous = self.action_effect_values.get(key, 0.0)
        value = previous + (sample - previous) / count
        self.action_effect_samples[key] = count
        self.action_effect_values[key] = value
        return value, count

    def _action_effect_estimate(
        self, source_signature: str, action: Action
    ) -> Tuple[float, bool, int]:
        key = (source_signature, action)
        count = self.action_effect_samples[key]
        return self.action_effect_values.get(key, 0.0), count > 0, count

    def _causal_spatial_effect(
        self,
        factual: Frame,
        neutral: Frame,
        ignored_pixels: Optional[set[Tuple[int, int]]] = None,
        minimum_cell_pixels: int = 1,
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        if minimum_cell_pixels <= 0:
            raise ValueError("minimum cell pixels must be positive")
        if (
            factual.width != neutral.width
            or factual.height != neutral.height
            or factual.channels != neutral.channels
        ):
            return None, 0, None
        columns = min(self.config.causal_spatial_columns, factual.width)
        rows = min(self.config.causal_spatial_rows, factual.height)
        cells = [0] * (columns * rows)
        changed_pixels = 0
        x_total = 0
        y_total = 0
        for y in range(factual.height):
            for x in range(factual.width):
                if ignored_pixels is not None and (x, y) in ignored_pixels:
                    continue
                offset = (y * factual.width + x) * factual.channels
                difference = sum(
                    abs(
                        factual.pixels[offset + channel]
                        - neutral.pixels[offset + channel]
                    )
                    for channel in range(factual.channels)
                ) / factual.channels
                if difference < self.config.causal_change_pixel_threshold:
                    continue
                changed_pixels += 1
                x_total += x
                y_total += y
                gx = min(columns - 1, x * columns // factual.width)
                gy = min(rows - 1, y * rows // factual.height)
                cells[gy * columns + gx] += 1
        if not changed_pixels:
            return None, 0, None
        occupied = bytes(
            1 if count >= minimum_cell_pixels else 0 for count in cells
        )
        if not any(occupied):
            return None, changed_pixels, (
                x_total / changed_pixels,
                y_total / changed_pixels,
            )
        return (
            occupied.hex(),
            changed_pixels,
            (x_total / changed_pixels, y_total / changed_pixels),
        )

    @staticmethod
    def _causal_frontier_key(
        context_signature: str,
        spatial_signature: str,
        affordance_actions: Tuple[Action, ...] = (),
    ) -> str:
        affordances = ",".join(
            sorted({action.value for action in affordance_actions})
        )
        return f"{context_signature}|affordances={affordances}:{spatial_signature}"

    def _causal_spatial_cells(
        self, spatial_signature: Optional[str]
    ) -> set[Tuple[int, int]]:
        if not spatial_signature:
            return set()
        try:
            occupied = bytes.fromhex(spatial_signature)
        except ValueError:
            return set()
        columns = self.config.causal_spatial_columns
        return {
            (index % columns, index // columns)
            for index, value in enumerate(occupied)
            if value
        }

    def _causal_cell_coverage(
        self, spatial_signature: Optional[str]
    ) -> Tuple[float, int, int]:
        """Return global novelty for cells changed relative to matched NOOP."""
        cells = self._causal_spatial_cells(spatial_signature)
        if not cells:
            return 0.0, 0, 0
        values = [
            1.0 / math.sqrt(self.causal_spatial_cell_visits[cell] + 1)
            for cell in cells
        ]
        return (
            sum(values) / len(values),
            sum(self.causal_spatial_cell_visits[cell] == 0 for cell in cells),
            len(cells),
        )

    @staticmethod
    def _behavioral_edge_key(
        source_signature: str, action: Action, duration: int
    ) -> Tuple[str, Action, int]:
        return source_signature, action, duration

    def _behavioral_edge_coverage(
        self, source_signature: str, action: Action, duration: int
    ) -> Tuple[int, bool, float]:
        """Return visits, unseen status, and a decaying intervention bonus."""
        if action == Action.NOOP or not source_signature:
            return 0, False, 0.0
        visits = self.behavioral_edge_visits[
            self._behavioral_edge_key(source_signature, action, duration)
        ]
        return (
            visits,
            visits == 0,
            self.config.behavioral_edge_coverage_weight
            / math.sqrt(visits + 1),
        )

    def _record_behavioral_edge(
        self, source_signature: str, action: Action, duration: int
    ) -> int:
        if action == Action.NOOP or not source_signature:
            return 0
        key = self._behavioral_edge_key(source_signature, action, duration)
        visits_before = self.behavioral_edge_visits[key]
        self.behavioral_edge_visits[key] += 1
        return visits_before

    def _persistent_cell_values(self, frame: Frame) -> Tuple[int, ...]:
        return frame.coarse_signature(
            columns=self.config.causal_spatial_columns,
            rows=self.config.causal_spatial_rows,
        )

    @staticmethod
    def _mean_frame_intensity(frame: Frame) -> float:
        return sum(frame.pixels) / (255.0 * max(1, len(frame.pixels)))

    @staticmethod
    def _coarse_scene_distance(
        left: Sequence[int], right: Sequence[int]
    ) -> float:
        if len(left) != len(right) or not left:
            return 1.0
        return sum(abs(a - b) for a, b in zip(left, right)) / (
            15.0 * len(left)
        )

    def _observe_dark_transition(self, frame: Frame) -> None:
        intensity = self._mean_frame_intensity(frame)
        if intensity <= self.config.dark_transition_intensity_threshold:
            if not self.bright_scene_memory:
                return
            if not self.dark_transition_active:
                self.dark_transition_active = True
                self.dark_transition_start_decision = self.decision_index
                self._emit(
                    "generic_dark_transition_started",
                    decision=self.decision_index,
                    scene_intensity=intensity,
                    dark_transition_intensity_threshold=(
                        self.config.dark_transition_intensity_threshold
                    ),
                    remembered_bright_scenes=len(self.bright_scene_memory),
                    **self._frame_fields(frame),
                )
            return

        signature = self._persistent_cell_values(frame)
        if self.dark_transition_active:
            minimum_distance = min(
                (
                    self._coarse_scene_distance(signature, remembered)
                    for remembered in self.bright_scene_memory
                ),
                default=1.0,
            )
            returned_to_known_scene = (
                minimum_distance
                <= self.config.known_scene_return_distance_threshold
            )
            self.known_scene_return_recovery_pending = returned_to_known_scene
            self.dark_transition_active = False
            self.anticipated_transition_observations_remaining = 0
            if not returned_to_known_scene:
                self.dark_transition_start_decision = None
                self.pending_novel_room_frame = frame
            self._emit(
                "generic_dark_transition_resolved",
                decision=self.decision_index,
                scene_intensity=intensity,
                minimum_known_scene_distance=minimum_distance,
                known_scene_return_distance_threshold=(
                    self.config.known_scene_return_distance_threshold
                ),
                returned_to_known_scene=returned_to_known_scene,
                recovery_pending=self.known_scene_return_recovery_pending,
                remembered_bright_scenes=len(self.bright_scene_memory),
                **self._frame_fields(frame),
            )
            if returned_to_known_scene:
                return
        minimum_existing_distance = min(
            (
                self._coarse_scene_distance(signature, remembered)
                for remembered in self.bright_scene_memory
            ),
            default=1.0,
        )
        if minimum_existing_distance > (
            self.config.known_scene_return_distance_threshold / 2.0
        ):
            self.bright_scene_memory.append(signature)
            if len(self.bright_scene_memory) > self.config.archive_capacity:
                self.bright_scene_memory.pop()

    def _apply_pending_novel_room_reset(self) -> None:
        frame = self.pending_novel_room_frame
        if frame is None:
            return
        self.pending_novel_room_frame = None
        self.clear_archive()
        self.human_prior_graph_edge_visits = Counter()
        self.human_prior_graph_state_visits = Counter()
        self.human_prior_option_visits = Counter()
        self.human_prior_player_position_visits = Counter()
        self.human_prior_phase_player_position_visits = Counter()
        self.human_prior_option_exhausted_sources = set()
        self.human_prior_graph_recovery_pending = False
        self.current_human_prior_world_context_signature = (
            "human-prior-world-root"
        )
        self.last_causal_cell_progress_decision = None
        self.causal_spatial_cell_visits = Counter()
        persistent_values = self._persistent_cell_values(frame)
        self.persistent_change_baseline = list(persistent_values)
        self.persistent_change_value_counts = [
            Counter({value: 1}) for value in persistent_values
        ]
        self.persistent_change_candidates = {}
        self.persistent_change_cells = {}
        self.persistent_change_mismatches = Counter()
        discovered: Tuple[Tuple[int, int], ...] = ()
        if self.goal_prior is not None:
            discovered = self.goal_prior.reset_room(frame)
        self._reset_unlabeled_entity_memory(frame)
        graph_signature = self._current_human_prior_graph_signature()
        if graph_signature:
            self.human_prior_graph_state_visits[graph_signature] += 1
        if (
            self.goal_prior is not None
            and self.goal_prior.current_player_slot is not None
        ):
            self._record_human_prior_player_position(
                graph_signature,
                self.goal_prior.current_player_slot,
            )
        self.current_scene = self._scene_signature(frame)
        checkpoint_state = self.env.save_state()
        self.known_scene_recovery_checkpoint = _LifeHazardCheckpoint(
            state=checkpoint_state,
            frame=frame,
            choice=(self.current_frontier_signature, Action.NOOP, 0),
            decision=self.decision_index,
            frontier_signature=self.current_frontier_signature,
            causal_context_signature=self.current_causal_context_signature,
            scene=self.current_scene,
            pose_action=self.current_pose_action,
            last_action=self.last_action,
            last_duration=self.last_duration,
            action_streak=self.action_streak,
            goal_heart_slots=(
                ()
                if self.goal_prior is None
                else self.goal_prior.current_slots()
            ),
            goal_player_slot=(
                None
                if self.goal_prior is None
                else self.goal_prior.current_player_slot
            ),
            goal_chest_obtained=False,
            human_prior_world_context_signature=(
                self.current_human_prior_world_context_signature
            ),
            kind="known_scene_root",
            state_id=self._state_id(checkpoint_state),
        )
        self._emit(
            "pixel_novel_room_started",
            decision=self.decision_index,
            discovered_heart_slots=discovered,
            human_prior_graph_signature=graph_signature or None,
            checkpoint_state_id=self._state_id(checkpoint_state),
            **self._frame_fields(frame),
        )

    def seed_bright_scene_memory(self, frames: Sequence[Frame]) -> None:
        remembered_before = len(self.bright_scene_memory)
        accepted = 0
        skipped_dark = 0
        for frame in frames:
            if (
                self._mean_frame_intensity(frame)
                <= self.config.dark_transition_intensity_threshold
            ):
                skipped_dark += 1
                continue
            signature = self._persistent_cell_values(frame)
            minimum_distance = min(
                (
                    self._coarse_scene_distance(signature, remembered)
                    for remembered in self.bright_scene_memory
                ),
                default=1.0,
            )
            if minimum_distance <= (
                self.config.known_scene_return_distance_threshold / 2.0
            ):
                continue
            if len(self.bright_scene_memory) >= self.config.archive_capacity:
                break
            self.bright_scene_memory.append(signature)
            accepted += 1
        self._emit(
            "episodic_scene_memory_seeded",
            decision=self.decision_index,
            source_frames=len(frames),
            accepted_scene_signatures=accepted,
            skipped_dark_frames=skipped_dark,
            remembered_scenes_before=remembered_before,
            remembered_scenes_after=len(self.bright_scene_memory),
            known_scene_return_distance_threshold=(
                self.config.known_scene_return_distance_threshold
            ),
        )

    def seed_human_prior_episodic_memory(
        self, events: Sequence[Dict[str, Any]]
    ) -> None:
        """Reconstruct temporary assisted frontier memory from telemetry."""

        if self.goal_prior is None:
            return
        graph_states: CounterType[str] = Counter()
        player_positions: CounterType[Tuple[int, int]] = Counter()
        phase_player_positions: CounterType[
            Tuple[str, Tuple[int, int]]
        ] = Counter()
        graph_edges: CounterType[Tuple[str, Action, int]] = Counter()
        option_paths: CounterType[
            Tuple[str, Tuple[Tuple[Action, int], ...]]
        ] = Counter()
        temporal_option_values: Dict[Tuple[str, Action, int], float] = {}
        temporal_option_samples: CounterType[
            Tuple[str, Action, int]
        ] = Counter()
        known_slots: set[Tuple[int, int]] = set(
            self.goal_prior.known_slots
        )
        observed_present_slots: Tuple[Tuple[int, int], ...] = tuple(
            self.goal_prior.current_slots()
        )
        observed_life_signature = self.goal_prior.current_life_signature
        observed_player_slot = self.goal_prior.current_player_slot
        observed_world_context = (
            self.current_human_prior_world_context_signature
        )
        observed_pose_action = self.current_pose_action
        observed_chest_obtained = self.goal_prior.chest_obtained
        event_present_slots = observed_present_slots
        event_life_signature = observed_life_signature
        event_player_slot = observed_player_slot
        event_world_context = observed_world_context
        event_pose_action = observed_pose_action
        event_chest_obtained = observed_chest_obtained
        latest_decision_has_semantic_state = False
        decisions = 0
        room_boundaries = 0
        for event in events:
            if event.get("event") == "pixel_novel_room_started":
                room_boundaries += 1
                graph_states.clear()
                player_positions.clear()
                phase_player_positions.clear()
                graph_edges.clear()
                option_paths.clear()
                known_slots = {
                    (int(slot[0]), int(slot[1]))
                    for slot in (
                        event.get("discovered_heart_slots") or ()
                    )
                }
                event_present_slots = tuple(sorted(known_slots))
                event_life_signature = observed_life_signature
                event_player_slot = observed_player_slot
                event_world_context = "human-prior-world-root"
                event_pose_action = None
                event_chest_obtained = False
                latest_decision_has_semantic_state = False
                continue
            for slot in event.get("human_prior_known_heart_slots") or ():
                known_slots.add((int(slot[0]), int(slot[1])))
            if event.get("event") == "goal_milestone_exhaustion_learned":
                choice_values = tuple(event.get("milestone_choice") or ())
                if len(choice_values) == 3:
                    try:
                        choice = (
                            str(choice_values[0]),
                            Action(str(choice_values[1])),
                            int(choice_values[2]),
                        )
                        temporal_option_values[choice] = float(
                            event["learned_hazard_value"]
                        )
                        temporal_option_samples[choice] = int(
                            event["learned_hazard_samples"]
                        )
                    except (KeyError, TypeError, ValueError):
                        pass
            if event.get("event") != "decision_committed":
                continue
            decisions += 1
            latest_decision_has_semantic_state = bool(
                "human_prior_target_hearts" in event
                or event.get("human_prior_target_player_slot") is not None
                or event.get("human_prior_world_target_context")
            )
            source_signature = str(
                event.get("human_prior_graph_source_signature") or ""
            )
            target_signature = str(
                event.get("human_prior_graph_target_signature") or ""
            )
            if target_signature:
                graph_states[target_signature] += 1
            target_player = event.get("human_prior_target_player_slot")
            if target_player is not None:
                event_player_slot = (
                    int(target_player[0]),
                    int(target_player[1]),
                )
                player_positions[event_player_slot] += 1
                phase_player_positions[
                    (
                        self._human_prior_position_phase(target_signature),
                        event_player_slot,
                    )
                ] += 1
            if "human_prior_target_hearts" in event:
                event_present_slots = tuple(
                    (int(slot[0]), int(slot[1]))
                    for slot in (
                        event.get("human_prior_target_hearts") or ()
                    )
                )
            target_life = event.get("human_prior_target_life_signature")
            if target_life:
                event_life_signature = str(target_life)
            target_context = event.get(
                "human_prior_world_target_context"
            )
            if target_context:
                event_world_context = str(target_context)
            if "human_prior_chest_obtained" in event:
                event_chest_obtained = bool(
                    event.get("human_prior_chest_obtained")
                )
            path_values = tuple(event.get("path") or ())
            duration_values = tuple(event.get("durations") or ())
            target_pose_value = event.get("target_pose_action")
            if target_pose_value is not None:
                try:
                    event_pose_action = Action(str(target_pose_value))
                except ValueError:
                    pass
            else:
                pose_values = (
                    path_values
                    if event.get("restored_archive")
                    else (event.get("action"),)
                )
                for pose_value in pose_values:
                    try:
                        pose_action = Action(str(pose_value))
                    except (TypeError, ValueError):
                        continue
                    event_pose_action = self._resulting_pose_action(
                        event_pose_action, pose_action
                    )
            if event.get("human_prior_verified_option") and source_signature:
                if len(path_values) == len(duration_values) and path_values:
                    try:
                        option_path = tuple(
                            (
                                Action(str(action_value)),
                                int(duration_value),
                            )
                            for action_value, duration_value in zip(
                                path_values, duration_values
                            )
                        )
                    except (TypeError, ValueError):
                        option_path = ()
                    if option_path:
                        option_paths[(source_signature, option_path)] += 1
                continue
            action_value = event.get("action")
            duration_value = event.get("action_frames")
            if (
                source_signature
                and action_value is not None
                and duration_value is not None
            ):
                try:
                    graph_edges[
                        (
                            source_signature,
                            Action(str(action_value)),
                            int(duration_value),
                        )
                    ] += 1
                except (TypeError, ValueError):
                    pass

        if graph_states:
            self.human_prior_graph_state_visits = graph_states
        if player_positions:
            self.human_prior_player_position_visits = player_positions
        self.human_prior_phase_player_position_visits = (
            phase_player_positions
        )
        self.human_prior_graph_edge_visits = graph_edges
        self.human_prior_option_visits = option_paths
        self.temporal_option_values.update(temporal_option_values)
        self.temporal_option_samples.update(temporal_option_samples)
        if latest_decision_has_semantic_state:
            present_slots = event_present_slots
            life_signature = event_life_signature
            player_slot = event_player_slot
            world_context = event_world_context
            pose_action = event_pose_action
            chest_obtained = event_chest_obtained
            current_state_source = "latest_assisted_decision"
        else:
            present_slots = observed_present_slots
            life_signature = observed_life_signature
            player_slot = observed_player_slot
            world_context = observed_world_context
            pose_action = observed_pose_action
            chest_obtained = observed_chest_obtained
            current_state_source = "resume_frame"
        self.current_human_prior_world_context_signature = world_context
        self.current_pose_action = pose_action
        self.goal_prior.seed_episodic_memory(
            tuple(sorted(known_slots)),
            present_slots,
            life_signature,
            player_slot,
            chest_obtained,
        )
        self._emit(
            "episodic_human_prior_memory_seeded",
            decision=self.decision_index,
            source_events=len(events),
            committed_decisions=decisions,
            room_boundaries=room_boundaries,
            graph_states=len(graph_states),
            graph_state_visits=sum(graph_states.values()),
            player_positions=len(player_positions),
            player_position_visits=sum(player_positions.values()),
            phase_player_positions=len(phase_player_positions),
            phase_player_position_visits=sum(
                phase_player_positions.values()
            ),
            graph_edges=len(graph_edges),
            graph_edge_visits=sum(graph_edges.values()),
            verified_option_paths=len(option_paths),
            verified_option_path_visits=sum(option_paths.values()),
            temporal_option_values=len(temporal_option_values),
            temporal_option_samples=sum(temporal_option_samples.values()),
            known_heart_slots=tuple(sorted(known_slots)),
            present_heart_slots=present_slots,
            player_slot=player_slot,
            world_context=world_context,
            pose_action=pose_action,
            chest_obtained=chest_obtained,
            current_state_source=current_state_source,
        )

    def _persistent_change_fields(self) -> Dict[str, Any]:
        columns = self.config.causal_spatial_columns
        return {
            "persistent_change_enabled": (
                self.config.persistent_change_stability_decisions > 0
            ),
            "persistent_change_stability_decisions": (
                self.config.persistent_change_stability_decisions
            ),
            "persistent_change_minimum_value_drop": (
                self.config.persistent_change_minimum_value_drop
            ),
            "persistent_change_speculative_recovery": (
                self.config.persistent_change_speculative_recovery
            ),
            "persistent_change_candidate_count": len(
                self.persistent_change_candidates
            ),
            "persistent_change_active_count": len(
                self.persistent_change_cells
            ),
            "persistent_change_active_cells": [
                {
                    "column": index % columns,
                    "row": index // columns,
                    "value": value,
                }
                for index, value in sorted(
                    self.persistent_change_cells.items()
                )
            ],
        }

    def _matches_persistent_changes(self, frame: Frame) -> bool:
        if not self.persistent_change_cells:
            return True
        values = self._persistent_cell_values(frame)
        return all(
            index < len(values) and values[index] == value
            for index, value in self.persistent_change_cells.items()
        )

    def _matches_persistent_change_candidates(self, frame: Frame) -> bool:
        if not self.persistent_change_candidates:
            return True
        values = self._persistent_cell_values(frame)
        return all(
            index < len(values) and values[index] == candidate_value
            for index, (candidate_value, _count) in (
                self.persistent_change_candidates.items()
            )
        )

    def _observe_persistent_changes(
        self,
        frame: Frame,
        *,
        action_dependent: bool = True,
    ) -> None:
        stability = self.config.persistent_change_stability_decisions
        if stability <= 0:
            return
        if not action_dependent:
            self._emit(
                "persistent_change_observation_skipped",
                decision=self.decision_index,
                reason="action_equivalent_or_passive_transition",
                **self._persistent_change_fields(),
                **self._frame_fields(frame),
            )
            return
        values = self._persistent_cell_values(frame)
        activated = []
        retired = []
        baseline_adapted = []
        for index, (baseline, value) in enumerate(
            zip(self.persistent_change_baseline, values)
        ):
            active_value = self.persistent_change_cells.get(index)
            if active_value is None:
                value_counts = self.persistent_change_value_counts[index]
                value_counts[value] += 1
                learned_baseline = max(
                    value_counts,
                    key=lambda candidate: (
                        value_counts[candidate],
                        candidate == baseline,
                    ),
                )
                if learned_baseline != baseline:
                    self.persistent_change_baseline[index] = learned_baseline
                    self.persistent_change_candidates.pop(index, None)
                    baseline_adapted.append(
                        (index, baseline, learned_baseline)
                    )
                    baseline = learned_baseline
            if active_value is not None:
                if value == active_value:
                    self.persistent_change_mismatches[index] = 0
                elif value != baseline:
                    self.persistent_change_mismatches[index] = 0
                else:
                    mismatches = self.persistent_change_mismatches[index] + 1
                    self.persistent_change_mismatches[index] = mismatches
                    if mismatches >= stability:
                        retired.append((index, active_value))
                        self.persistent_change_cells.pop(index, None)
                        self.persistent_change_mismatches.pop(index, None)
                        self.persistent_change_baseline[index] = value
                        self.persistent_change_value_counts[index] = Counter(
                            {value: 1}
                        )
                        baseline = value
            if index in self.persistent_change_cells:
                self.persistent_change_candidates.pop(index, None)
                continue
            if value == baseline:
                self.persistent_change_candidates.pop(index, None)
                continue
            if (
                self.config.persistent_change_minimum_value_drop > 0
                and baseline - value
                < self.config.persistent_change_minimum_value_drop
            ):
                self.persistent_change_candidates.pop(index, None)
                continue
            candidate_value, candidate_count = (
                self.persistent_change_candidates.get(index, (value, 0))
            )
            candidate_count = (
                candidate_count + 1
                if candidate_value == value
                else 1
            )
            self.persistent_change_candidates[index] = (
                value,
                candidate_count,
            )
            if (
                candidate_count >= stability
                and self.persistent_change_cells.get(index) != value
            ):
                previous = self.persistent_change_cells.get(index)
                self.persistent_change_cells[index] = value
                self.persistent_change_mismatches[index] = 0
                activated.append((index, value, previous))
        if activated or retired or baseline_adapted:
            columns = self.config.causal_spatial_columns
            self._emit(
                "persistent_change_evidence_updated",
                decision=self.decision_index,
                activated=[
                    {
                        "column": index % columns,
                        "row": index // columns,
                        "value": value,
                        "previous_value": previous,
                    }
                    for index, value, previous in activated
                ],
                retired=[
                    {
                        "column": index % columns,
                        "row": index // columns,
                        "value": value,
                    }
                    for index, value in retired
                ],
                baseline_adapted=[
                    {
                        "column": index % columns,
                        "row": index // columns,
                        "previous_value": previous,
                        "value": value,
                    }
                    for index, previous, value in baseline_adapted
                ],
                **self._persistent_change_fields(),
            )

    def _reset_persistent_change_observation_window(
        self, reason: str, preserve_candidates: bool = False
    ) -> None:
        if self.config.persistent_change_stability_decisions <= 0:
            return
        candidate_count = len(self.persistent_change_candidates)
        if not preserve_candidates:
            self.persistent_change_candidates = {}
        self.persistent_change_mismatches = Counter()
        if not preserve_candidates:
            self.persistent_change_value_counts = [
                Counter({value: 1})
                for value in self.persistent_change_baseline
            ]
        self._emit(
            "persistent_change_observation_window_reset",
            decision=self.decision_index,
            reason=reason,
            discarded_candidates=(0 if preserve_candidates else candidate_count),
            preserved_candidates=(candidate_count if preserve_candidates else 0),
            **self._persistent_change_fields(),
        )

    def _causal_target_context(
        self, source_context: str, spatial_signature: Optional[str]
    ) -> Tuple[str, bool, int]:
        if not spatial_signature:
            return source_context, False, 0
        cells = self._causal_spatial_cells(spatial_signature)
        components: List[set[Tuple[int, int]]] = []
        while cells:
            pending = [cells.pop()]
            component = set(pending)
            while pending:
                x, y = pending.pop()
                for neighbor in (
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1),
                ):
                    if neighbor in cells:
                        cells.remove(neighbor)
                        component.add(neighbor)
                        pending.append(neighbor)
            components.append(component)
        if len(components) < 2:
            return source_context, False, len(components)
        minimum_gap = min(
            abs(ax - bx) + abs(ay - by)
            for index, first in enumerate(components)
            for second in components[index + 1 :]
            for ax, ay in first
            for bx, by in second
        )
        if minimum_gap < self.config.causal_event_min_component_gap:
            return source_context, False, len(components)
        return (
            f"{source_context}>{spatial_signature}",
            True,
            len(components),
        )

    def _verified_without_learned_hazards(
        self, source_signature: str, verified: List[Tuple[Any, ...]]
    ) -> Tuple[List[Tuple[Any, ...]], List[Dict[str, Any]]]:
        safe = []
        filtered = []
        for item in verified:
            plan = item[1]
            value, known = self._temporal_option_estimate(
                source_signature, plan.path[0], plan.durations[0]
            )
            if known and value < 0.0:
                filtered.append(
                    {
                        "action": plan.path[0],
                        "action_frames": plan.durations[0],
                        "temporal_option_value": value,
                        "temporal_option_value_source": (
                            self._temporal_option_estimate_source(
                                source_signature,
                                plan.path[0],
                                plan.durations[0],
                            )
                        ),
                    }
                )
            else:
                safe.append(item)
        return (safe or verified), filtered if safe else []

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

    def _behavior_probe_selection(self, frame: Frame) -> _BehaviorProbeSelection:
        duration = max(self.planner.duration_choices)
        ordered_actions = list(dict.fromkeys(self.config.actions))
        anchor = (
            Action.NOOP
            if Action.NOOP in ordered_actions
            else ordered_actions[0]
        )
        controls = [action for action in ordered_actions if action != anchor]
        visual_cluster = self._abstract_signature(frame)
        slots = max(0, self.config.behavioral_probe_count - 1)
        if not controls or not slots:
            return _BehaviorProbeSelection(
                ((anchor, duration),),
                visual_cluster,
                "anchor_only",
                None,
            )

        hypotheses = [
            cluster
            for cluster in self.behavior_clusters
            if cluster.visual_cluster == visual_cluster
        ]
        separations: Dict[Action, float] = {}
        if len(hypotheses) >= 2:
            for action in controls:
                probe = (action, duration)
                if any(probe not in cluster.probe_centroids for cluster in hypotheses):
                    continue
                separations[action] = max(
                    float(
                        (
                            left.probe_centroids[probe]
                            - right.probe_centroids[probe]
                        )
                        .pow(2)
                        .mean()
                        .sqrt()
                    )
                    for index, left in enumerate(hypotheses)
                    for right in hypotheses[index + 1 :]
                )

        selected = []
        reason = "coverage_rotation"
        separation = None
        if separations:
            first = max(
                controls,
                key=lambda action: (
                    separations.get(action, -math.inf),
                    -self.visual_probe_counts[(visual_cluster, action, duration)],
                    -ordered_actions.index(action),
                ),
            )
            selected.append(first)
            reason = "hypothesis_separation"
            separation = separations[first]
        remaining = [action for action in controls if action not in selected]
        remaining.sort(
            key=lambda action: (
                self.visual_probe_counts[(visual_cluster, action, duration)],
                ordered_actions.index(action),
            )
        )
        selected.extend(remaining[: slots - len(selected)])
        keys = ((anchor, duration),) + tuple(
            (action, duration) for action in selected[:slots]
        )
        return _BehaviorProbeSelection(
            keys,
            visual_cluster,
            reason,
            selected[0] if selected else None,
            separation,
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
        migrated_behavioral_edges = 0
        for edge in list(self.behavioral_edge_visits):
            signature, action, duration = edge
            if signature != source:
                continue
            target_edge = (target, action, duration)
            visits = self.behavioral_edge_visits.pop(edge)
            self.behavioral_edge_visits[target_edge] += visits
            migrated_behavioral_edges += 1
        migrated_temporal_choices = 0
        for choice in list(self.temporal_option_samples):
            signature, action, duration = choice
            if signature != source:
                continue
            target_choice = (target, action, duration)
            source_count = self.temporal_option_samples.pop(choice)
            target_count = self.temporal_option_samples.get(target_choice, 0)
            source_value = self.temporal_option_values.pop(choice, 0.0)
            target_value = self.temporal_option_values.get(target_choice, 0.0)
            combined_count = source_count + target_count
            self.temporal_option_samples[target_choice] = combined_count
            self.temporal_option_values[target_choice] = (
                source_count * source_value + target_count * target_value
            ) / combined_count
            migrated_temporal_choices += 1
        if self.pending_option_choice is not None:
            signature, action, duration = self.pending_option_choice
            if signature == source:
                self.pending_option_choice = (target, action, duration)
        active_trace = self.active_temporal_option
        if active_trace is not None:
            if active_trace.choice is not None and active_trace.choice[0] == source:
                active_trace.choice = (
                    target,
                    active_trace.choice[1],
                    active_trace.choice[2],
                )
            if active_trace.entry_signature == source:
                active_trace.entry_signature = target
            if source in active_trace.signatures:
                active_trace.signatures.discard(source)
                active_trace.signatures.add(target)
        behavior_visits = self.behavior_visits.pop(source, 0)
        if behavior_visits:
            self.behavior_visits[target] += behavior_visits
        self._emit(
            "frontier_signature_migrated",
            decision=self.decision_index + 1,
            source_signature=source,
            target_signature=target,
            migrated_choice_values=migrated_choices,
            migrated_traces=migrated_traces,
            migrated_archive_origins=migrated_origins,
            migrated_behavioral_edges=migrated_behavioral_edges,
            migrated_temporal_choice_values=migrated_temporal_choices,
        )

    @torch.no_grad()
    def _behavioral_signature(
        self,
        frame: Frame,
        outcomes: Dict[Tuple[Action, int], Frame],
        provisional_signature: str,
        selection: Optional[_BehaviorProbeSelection] = None,
    ) -> str:
        source_latent = self._frame_latent(frame)
        selection = selection or self._behavior_probe_selection(frame)
        requested = selection.keys
        profile = {
            probe: self._frame_latent(outcomes[probe]) - source_latent
            for probe in requested
            if probe in outcomes
        }
        visual_cluster = self._abstract_signature(frame)
        for action, duration in profile:
            self.visual_probe_counts[(visual_cluster, action, duration)] += 1
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
        nearest_expands_profile = False
        hypotheses = [
            cluster
            for cluster in self.behavior_clusters
            if cluster.visual_cluster == visual_cluster
        ]
        anchor_probe = requested[0]
        anchor_distances = {
            cluster.key: float(
                (profile[anchor_probe] - cluster.probe_centroids[anchor_probe])
                .pow(2)
                .mean()
                .sqrt()
            )
            for cluster in hypotheses
            if anchor_probe in cluster.probe_centroids
        }
        anchor_matches = {
            key
            for key, distance in anchor_distances.items()
            if distance <= self.config.behavioral_abstraction_rmse_threshold
        }
        unique_anchor_match = (
            next(iter(anchor_matches)) if len(anchor_matches) == 1 else None
        )
        novel_anchor = bool(hypotheses) and not anchor_matches
        for cluster in hypotheses:
            shared = sorted(set(profile) & set(cluster.probe_centroids))
            has_unseen_probe = any(
                probe not in cluster.probe_centroids for probe in profile
            )
            expands_profile = has_unseen_probe and (
                len(hypotheses) == 1
                or cluster.key == unique_anchor_match
                or novel_anchor
            )
            required_shared = (
                1
                if expands_profile
                else self.config.behavioral_abstraction_min_shared_probes
            )
            if anchor_probe not in shared or len(shared) < required_shared:
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
                nearest_expands_profile = expands_profile
        if hypotheses and nearest is None:
            self._emit(
                "behavioral_abstraction_deferred",
                decision=self.decision_index + 1,
                reason="ambiguous_partial_profile",
                provisional_signature=provisional_signature,
                visual_cluster=visual_cluster,
                observed_probes=[
                    {"action": action, "action_frames": duration}
                    for action, duration in profile
                ],
                candidate_hypotheses=len(hypotheses),
                anchor_matches=len(anchor_matches),
                required_shared_probes=(
                    self.config.behavioral_abstraction_min_shared_probes
                ),
                **self._frame_fields(frame),
            )
            return provisional_signature
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
            classification_mode=(
                "new_cluster"
                if created
                else "anchor_profile_expansion"
                if nearest_expands_profile
                else "full_profile_match"
            ),
            shared_probe_count=0 if created else len(nearest_probe_distances),
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

    def _temporal_option_estimate(
        self, signature: str, action: Action, duration: int
    ) -> Tuple[float, bool]:
        choice = (signature, action, duration)
        if self.temporal_option_samples[choice]:
            return self.temporal_option_values.get(choice, 0.0), True
        if self.temporal_option_action_samples[action]:
            return (
                self.config.temporal_option_action_prior_weight
                * self.temporal_option_action_values.get(action, 0.0),
                True,
            )
        return 0.0, False

    def _temporal_option_estimate_source(
        self, signature: str, action: Action, duration: int
    ) -> str:
        if self.temporal_option_samples[(signature, action, duration)]:
            return "exact_choice"
        if self.temporal_option_action_samples[action]:
            return "action_prior"
        return "unseen"

    def _record_temporal_option_sample(
        self,
        choice: Tuple[str, Action, int],
        sample: float,
        generalize_action_hazard: bool = False,
    ) -> float:
        count = self.temporal_option_samples[choice] + 1
        previous = self.temporal_option_values.get(choice, 0.0)
        value = previous + (sample - previous) / count
        self.temporal_option_samples[choice] = count
        self.temporal_option_values[choice] = value
        if generalize_action_hazard and sample < 0.0:
            action = choice[1]
            action_count = self.temporal_option_action_samples[action] + 1
            action_previous = self.temporal_option_action_values.get(action, 0.0)
            self.temporal_option_action_samples[action] = action_count
            self.temporal_option_action_values[action] = action_previous + (
                sample - action_previous
            ) / action_count
        return value

    def _release_option_counterfactual(
        self, counterfactual: _OptionCounterfactual, reason: str
    ) -> None:
        self._emit(
            "temporal_option_counterfactual_released",
            decision=self.decision_index,
            reason=reason,
            state_id=counterfactual.state_id,
            action=counterfactual.action,
            action_frames=counterfactual.duration,
            passive_steps=counterfactual.passive_steps,
            maximum_contrast=counterfactual.maximum_contrast,
        )
        release_state = getattr(self.env, "release_state", None)
        if release_state is not None:
            release_state(counterfactual.state)

    def _release_life_hazard_checkpoint(
        self, checkpoint: _LifeHazardCheckpoint, reason: str
    ) -> None:
        self._emit(
            (
                "goal_milestone_checkpoint_released"
                if checkpoint.kind == "goal_milestone"
                else "life_hazard_checkpoint_released"
            ),
            decision=self.decision_index,
            reason=reason,
            checkpoint_kind=checkpoint.kind,
            choice=checkpoint.choice,
            initiation_decision=checkpoint.decision,
            state_id=checkpoint.state_id,
            **self._frame_fields(checkpoint.frame),
        )
        release_state = getattr(self.env, "release_state", None)
        if release_state is not None:
            release_state(checkpoint.state)

    def _discard_pending_temporal_option(self, reason: str) -> None:
        counterfactual = getattr(self, "pending_option_counterfactual", None)
        if counterfactual is not None:
            self._release_option_counterfactual(counterfactual, reason)
        recovery = getattr(
            self, "pending_option_recovery_checkpoint", None
        )
        if recovery is not None:
            self._release_life_hazard_checkpoint(recovery, reason)
        self.pending_option_choice = None
        self.pending_option_decision = None
        self.pending_option_frame_digest = None
        self.pending_option_recovery_checkpoint = None
        self.pending_option_causal_evidence = False
        self.pending_option_counterfactual = None

    def _discard_temporal_option(self, reason: str) -> None:
        trace = self.active_temporal_option
        if trace is not None:
            self._emit(
                "temporal_option_discarded",
                decision=self.decision_index,
                reason=reason,
                choice=trace.choice,
                initiation_decision=trace.initiation_decision,
                initiation_frame=trace.initiation_frame_digest,
                start_decision=trace.start_decision,
                passive_decisions=trace.passive_decisions,
                unique_signatures=len(trace.signatures),
                unique_scenes=len(trace.scenes),
                causal_evidence=trace.causal_evidence,
                counterfactual_steps=(
                    0
                    if trace.counterfactual is None
                    else trace.counterfactual.passive_steps
                ),
            )
            if trace.counterfactual is not None:
                self._release_option_counterfactual(
                    trace.counterfactual, reason
                )
            if trace.recovery_checkpoint is not None:
                self._release_life_hazard_checkpoint(
                    trace.recovery_checkpoint, reason
                )
        self.active_temporal_option = None

    def _supersede_temporal_option_for_intervention(
        self,
        action: Action,
        has_causal_candidate: bool,
    ) -> bool:
        if (
            self.active_temporal_option is None
            or action == Action.NOOP
            or not has_causal_candidate
        ):
            return False
        prior_choice = self.active_temporal_option.choice
        self._discard_temporal_option("superseded_by_intervention")
        self._emit(
            "temporal_option_superseded",
            decision=self.decision_index + 1,
            prior_choice=prior_choice,
            action=action,
            reason="new_action_with_counterfactual",
        )
        return True

    def _advance_option_counterfactual(
        self,
        trace: _TemporalOptionTrace,
        action: Action,
        duration: int,
        factual_target: Frame,
    ) -> None:
        counterfactual = trace.counterfactual
        if counterfactual is None:
            return
        prior_state = counterfactual.state
        prior_state_id = counterfactual.state_id
        self.env.load_state(prior_state)
        target = self.env.step(action, duration)
        env_step_seq = getattr(self.env, "last_step_seq", None)
        next_state = self.env.save_state()
        state_save_seq = getattr(self.env, "last_state_event_seq", None)
        next_state_id = self._state_id(next_state)
        contrast = factual_target.mean_absolute_difference(target)
        counterfactual.passive_steps += 1
        counterfactual.maximum_contrast = max(
            counterfactual.maximum_contrast, contrast
        )
        counterfactual.state = next_state
        counterfactual.state_id = next_state_id
        counterfactual.frame = target
        self._emit(
            "temporal_option_counterfactual_advanced",
            decision=self.decision_index + 1,
            choice=trace.choice,
            action=action,
            action_frames=duration,
            passive_steps=counterfactual.passive_steps,
            source_state_id=prior_state_id,
            target_state_id=next_state_id,
            env_step_seq=env_step_seq,
            state_save_seq=state_save_seq,
            pixel_contrast=contrast,
            maximum_contrast=counterfactual.maximum_contrast,
            delayed_divergence_observed=(
                contrast > self.config.action_equivalence_threshold
            ),
            **self._frame_fields(target),
        )
        release_state = getattr(self.env, "release_state", None)
        if release_state is not None:
            release_state(prior_state)

    def _advance_temporal_option(
        self,
        signature: str,
        scene: str,
        passive: bool,
        grace_continuation: bool = False,
        passive_action: Optional[Action] = None,
        passive_duration: Optional[int] = None,
        factual_target: Optional[Frame] = None,
    ) -> None:
        endpoint_is_new = self.behavior_visits[signature] == 0
        self.behavior_visits[signature] += 1
        trace = self.active_temporal_option
        if passive:
            if trace is None:
                trace = _TemporalOptionTrace(
                    choice=self.pending_option_choice,
                    initiation_decision=self.pending_option_decision,
                    start_decision=self.decision_index + 1,
                    entry_signature=signature,
                    entry_scene=scene,
                    initiation_frame_digest=(
                        self.pending_option_frame_digest
                    ),
                    recovery_checkpoint=(
                        self.pending_option_recovery_checkpoint
                    ),
                    causal_evidence=self.pending_option_causal_evidence,
                    counterfactual=self.pending_option_counterfactual,
                )
                self.active_temporal_option = trace
                self.pending_option_choice = None
                self.pending_option_decision = None
                self.pending_option_frame_digest = None
                self.pending_option_recovery_checkpoint = None
                self.pending_option_causal_evidence = False
                self.pending_option_counterfactual = None
                self._emit(
                    "temporal_option_started",
                    decision=self.decision_index + 1,
                    choice=trace.choice,
                    initiation_decision=trace.initiation_decision,
                    initiation_frame=trace.initiation_frame_digest,
                    life_recovery_state_id=(
                        None
                        if trace.recovery_checkpoint is None
                        else trace.recovery_checkpoint.state_id
                    ),
                    credited=trace.choice is not None and trace.causal_evidence,
                    causal_evidence=trace.causal_evidence,
                    counterfactual_state_id=(
                        None
                        if trace.counterfactual is None
                        else trace.counterfactual.state_id
                    ),
                    entry_signature=signature,
                    entry_scene=scene,
                )
            trace.passive_decisions += 1
            trace.signatures.add(signature)
            trace.scenes.add(scene)
            if (
                passive_action is not None
                and passive_duration is not None
                and factual_target is not None
            ):
                self._advance_option_counterfactual(
                    trace,
                    passive_action,
                    passive_duration,
                    factual_target,
                )
            self._emit(
                "temporal_option_advanced",
                decision=self.decision_index + 1,
                choice=trace.choice,
                passive_decisions=trace.passive_decisions,
                unique_signatures=len(trace.signatures),
                unique_scenes=len(trace.scenes),
                grace_continuation=grace_continuation,
                causal_evidence=trace.causal_evidence,
                counterfactual_steps=(
                    0
                    if trace.counterfactual is None
                    else trace.counterfactual.passive_steps
                ),
            )
            return
        if trace is None:
            self._discard_pending_temporal_option("no_passive_sequence")
            return

        trace.signatures.add(signature)
        trace.scenes.add(scene)
        endpoint_counterfactual_contrast = 0.0
        if trace.counterfactual is not None:
            endpoint_counterfactual_contrast = self.frame.mean_absolute_difference(
                trace.counterfactual.frame
            )
            trace.causal_evidence = (
                endpoint_counterfactual_contrast
                > self.config.action_equivalence_threshold
            )
        scene_span = min(max(0, len(trace.scenes) - 1) / 3.0, 1.0)
        duration_progress = min(
            trace.passive_decisions / self.config.temporal_option_duration_scale,
            1.0,
        )
        returned_to_source = bool(
            trace.choice is not None and signature == trace.choice[0]
        )
        returned_to_known_state = not endpoint_is_new
        robust_delayed_return = bool(
            returned_to_known_state
            and trace.passive_decisions >= self.config.delayed_return_min_length
            and (len(trace.signatures) > 1 or len(trace.scenes) > 1)
        )
        sample = (
            self.config.temporal_option_novelty_weight * float(endpoint_is_new)
            + self.config.temporal_option_scene_span_weight * scene_span
            + self.config.temporal_option_duration_weight * duration_progress
            - self.config.temporal_option_return_penalty
            * float(robust_delayed_return)
        )
        learned_value = None
        sample_count = 0
        credited = trace.choice is not None and trace.causal_evidence
        action_hazard_generalized = bool(
            trace.causal_evidence
            and sample < 0.0
            and robust_delayed_return
        )
        if credited:
            learned_value = self._record_temporal_option_sample(
                trace.choice,
                sample,
                generalize_action_hazard=action_hazard_generalized,
            )
            sample_count = self.temporal_option_samples[trace.choice]
        self._emit(
            "temporal_option_completed",
            decision=self.decision_index + 1,
            choice=trace.choice,
            initiation_decision=trace.initiation_decision,
            initiation_frame=trace.initiation_frame_digest,
            start_decision=trace.start_decision,
            endpoint_signature=signature,
            endpoint_scene=scene,
            endpoint_is_new=endpoint_is_new,
            passive_decisions=trace.passive_decisions,
            unique_signatures=len(trace.signatures),
            unique_scenes=len(trace.scenes),
            scene_span=scene_span,
            duration_progress=duration_progress,
            returned_to_source=returned_to_source,
            returned_to_known_state=returned_to_known_state,
            robust_delayed_return=robust_delayed_return,
            sample=sample,
            learned_value=learned_value,
            sample_count=sample_count,
            action_prior_value=(
                None
                if trace.choice is None
                else self.temporal_option_action_values.get(trace.choice[1])
            ),
            action_prior_sample_count=(
                0
                if trace.choice is None
                else self.temporal_option_action_samples[trace.choice[1]]
            ),
            credited=credited,
            action_hazard_generalized=(
                action_hazard_generalized and credited
            ),
            causal_evidence=trace.causal_evidence,
            counterfactual_steps=(
                0
                if trace.counterfactual is None
                else trace.counterfactual.passive_steps
            ),
            counterfactual_maximum_contrast=(
                0.0
                if trace.counterfactual is None
                else trace.counterfactual.maximum_contrast
            ),
            counterfactual_endpoint_contrast=(
                endpoint_counterfactual_contrast
            ),
        )
        if trace.counterfactual is not None:
            self._release_option_counterfactual(
                trace.counterfactual, "temporal_option_completed"
            )
        if trace.recovery_checkpoint is not None:
            self._release_life_hazard_checkpoint(
                trace.recovery_checkpoint, "temporal_option_completed"
            )
        self.active_temporal_option = None

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

    def _archive_causal_spatial_bonus(self, branch: _ArchivedBranch) -> float:
        if not branch.causal_spatial_signature:
            return 0.0
        frontier_key = self._causal_frontier_key(
            branch.causal_context_signature,
            branch.causal_spatial_signature,
            branch.causal_affordance_actions,
        )
        return self.config.causal_spatial_novelty_weight / math.sqrt(
            self.causal_spatial_visits[frontier_key] + 1
        )

    def _archive_causal_cell_coverage_bonus(
        self, branch: _ArchivedBranch
    ) -> float:
        coverage, _unvisited, _count = self._causal_cell_coverage(
            branch.causal_spatial_signature
        )
        return self.config.causal_cell_coverage_weight * coverage

    def _archive_behavioral_edge_coverage_bonus(
        self, branch: _ArchivedBranch
    ) -> float:
        return self._behavioral_edge_coverage(
            branch.origin_signature,
            branch.plan.path[0],
            branch.plan.durations[0],
        )[2]

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
        option_value, _option_is_known = self._temporal_option_estimate(
            branch.origin_signature,
            branch.plan.path[0],
            branch.plan.durations[0],
        )
        option_bonus = self.config.temporal_option_score_weight * option_value
        causal_spatial_bonus = self._archive_causal_spatial_bonus(branch)
        causal_cell_coverage_bonus = (
            self._archive_causal_cell_coverage_bonus(branch)
        )
        behavioral_edge_coverage_bonus = (
            self._archive_behavioral_edge_coverage_bonus(branch)
        )
        affordance_bonus = (
            self.config.causal_affordance_weight
            * math.sqrt(len(branch.causal_affordance_actions))
        )
        causal_event_bonus = (
            self.config.causal_event_archive_weight
            if branch.causal_event_outcome
            else 0.0
        )
        goal_navigation_bonus = 0.0
        if (
            self.goal_prior is not None
            and self.goal_prior.navigation_reward > 0.0
            and branch.goal_heart_slots
        ):
            goal_distance = self.goal_prior.distance_to_hearts(
                branch.frame, branch.goal_heart_slots
            )
            if goal_distance is not None:
                goal_navigation_bonus = -(
                    self.goal_prior.navigation_reward * goal_distance
                )
        elif (
            self.goal_prior is not None
            and self.goal_prior.navigation_reward > 0.0
            and branch.goal_chest_slot is not None
        ):
            chest_distance = self.goal_prior._nearest_distance(
                branch.goal_player_slot,
                (branch.goal_chest_slot,),
            )
            if chest_distance is not None:
                goal_navigation_bonus = -(
                    self.goal_prior.navigation_reward * chest_distance
                )
        goal_progress_bonus = branch.goal_progress_reward
        if branch.goal_total_hearts > 0:
            collected_hearts = max(
                0, branch.goal_total_hearts - branch.goal_remaining_hearts
            )
            goal_progress_bonus = max(
                goal_progress_bonus,
                self.config.human_prior_heart_reward * collected_hearts
                + (
                    self.config.human_prior_all_hearts_reward
                    if branch.goal_remaining_hearts == 0
                    else 0.0
                ),
            )
        if choice_is_known:
            return (
                choice_value
                + option_bonus
                + causal_spatial_bonus
                + causal_cell_coverage_bonus
                + behavioral_edge_coverage_bonus
                + affordance_bonus
                + causal_event_bonus
                + goal_navigation_bonus
                + goal_progress_bonus
            )
        return (
            max(own_value, self.config.frontier_origin_weight * origin_value)
            + option_bonus
            + causal_spatial_bonus
            + causal_cell_coverage_bonus
            + behavioral_edge_coverage_bonus
            + affordance_bonus
            + causal_event_bonus
            + goal_navigation_bonus
            + goal_progress_bonus
        )

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
        selection: Optional[_BehaviorProbeSelection] = None,
    ) -> List[NeuralPlan]:
        if selection is None:
            if self.frame is None:
                raise RuntimeError("behavior probes require a current frame")
            selection = self._behavior_probe_selection(self.frame)
        probe_keys = list(selection.keys)
        long_press_keys = []
        short_press_keys = []
        maximum_duration = max(self.planner.duration_choices)
        for action in (
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ):
            candidate = (action, maximum_duration)
            if (
                action in self.config.actions
                and candidate in best_by_action
                and candidate not in probe_keys
            ):
                probe_keys.append(candidate)
                long_press_keys.append(candidate)
        for action in (Action.A, Action.B):
            effective_durations = self.discovered_interaction_durations.get(
                action, set()
            )
            if not effective_durations:
                continue
            minimum_duration = min(effective_durations)
            candidate = (action, minimum_duration)
            if (
                action in self.config.actions
                and action in self.discovered_interaction_actions
                and candidate in best_by_action
                and candidate not in probe_keys
            ):
                probe_keys.append(candidate)
                short_press_keys.append(candidate)
        continuation_key = None
        continuation_action = None
        if self.last_action_was_causal_spatial and self.last_action is not None:
            continuation_action = self.last_action
        elif (
            self.active_temporal_option is not None
            and self.active_temporal_option.causal_evidence
            and self.active_temporal_option.choice is not None
        ):
            continuation_action = self.active_temporal_option.choice[1]
        if continuation_action is not None:
            candidate = (
                continuation_action,
                max(self.planner.duration_choices),
            )
            if candidate in best_by_action and candidate not in probe_keys:
                continuation_key = candidate
                probe_keys.append(candidate)
        matched_observation_key = None
        if self.pending_option_choice is not None:
            pending_duration = self.pending_option_choice[2]
            candidate = (Action.NOOP, pending_duration)
            if (
                Action.NOOP in self.config.actions
                and pending_duration != maximum_duration
                and candidate in best_by_action
                and candidate not in probe_keys
                and len(probe_keys) < self.config.verify_actions
            ):
                matched_observation_key = candidate
                probe_keys.append(candidate)
        collapse_recovery_keys = []
        hazardous_durations: Dict[Action, set[int]] = {}
        for (signature, action, duration), value in (
            self.temporal_option_values.items()
        ):
            if signature != self.current_frontier_signature or value >= 0.0:
                continue
            hazardous_durations.setdefault(action, set()).add(duration)
        remaining_probe_slots = max(
            0, self.config.verify_actions - len(probe_keys)
        )
        for duration in sorted(self.planner.duration_choices, reverse=True):
            if remaining_probe_slots <= 0:
                break
            for action in self.config.actions:
                if remaining_probe_slots <= 0:
                    break
                if action not in hazardous_durations:
                    continue
                candidate = (action, duration)
                if (
                    duration in hazardous_durations[action]
                    or candidate not in best_by_action
                    or candidate in probe_keys
                ):
                    continue
                probe_keys.append(candidate)
                collapse_recovery_keys.append(candidate)
                remaining_probe_slots -= 1
        self._emit(
            "behavior_probe_selected",
            decision=self.decision_index + 1,
            visual_cluster=selection.visual_cluster,
            reason=selection.reason,
            selected_control=selection.selected_control,
            hypothesis_separation=selection.hypothesis_separation,
            probes=[
                {
                    "action": action,
                    "action_frames": duration,
                    "prior_observations": self.visual_probe_counts[
                        (selection.visual_cluster, action, duration)
                    ],
                    "causal_continuation": (
                        (action, duration) == continuation_key
                    ),
                    "long_press_control": (
                        (action, duration) in long_press_keys
                    ),
                    "short_press_control": (
                        (action, duration) in short_press_keys
                    ),
                    "control_collapse_recovery_probe": (
                        (action, duration) in collapse_recovery_keys
                    ),
                    "matched_causal_observation": (
                        (action, duration) == matched_observation_key
                    ),
                }
                for action, duration in probe_keys
            ],
            causal_continuation=(
                None
                if continuation_key is None
                else {
                    "action": continuation_key[0],
                    "action_frames": continuation_key[1],
                }
            ),
        )
        if self.config.verify_actions < len(probe_keys):
            return ranked
        probes = [best_by_action[key] for key in probe_keys if key in best_by_action]
        required_keys = {
            (probe.path[0], probe.durations[0]) for probe in probes
        }
        required_action_counts = Counter(
            probe.path[0] for probe in probes
        )
        result = list(ranked)
        for probe in probes:
            matching_index = next(
                (
                    index
                    for index, item in enumerate(result)
                    if (
                        item.path[0], item.durations[0]
                    ) == (probe.path[0], probe.durations[0])
                ),
                None,
            )
            if (
                matching_index is None
                and required_action_counts[probe.path[0]] == 1
            ):
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
                        if (
                            result[index].path[0],
                            result[index].durations[0],
                        )
                        not in required_keys
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
            neutral = next(
                (item for item in group if item[1].path[0] == Action.NOOP),
                None,
            )
            if neutral is None or len({item[1].path[0] for item in group}) < 2:
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
                qualified.append((duration, neutral, spread, change))
        if not qualified:
            return None
        _duration, preferred, spread, change = max(qualified, key=lambda item: item[0])
        return preferred, spread, change

    def _verified_outcome_spread(
        self, verified: List[Tuple[Any, ...]]
    ) -> Tuple[float, Optional[int]]:
        maximum = 0.0
        maximum_duration = None
        for duration in self.planner.duration_choices:
            group = [item for item in verified if item[1].durations[0] == duration]
            if len({item[1].path[0] for item in group}) < 2:
                continue
            spread = max(
                left[3].mean_absolute_difference(right[3])
                for index, left in enumerate(group)
                for right in group[index + 1 :]
            )
            if spread > maximum:
                maximum = spread
                maximum_duration = duration
        return maximum, maximum_duration

    def _confirm_future_control_collapse(
        self,
        state: object,
        frame: Frame,
        actions: Sequence[Action],
        duration: int,
    ) -> Dict[str, Any]:
        """Distinguish a terminal state from a temporary visual transition.

        The probe is entirely counterfactual: it advances disposable save-state
        branches and restores the caller's root afterward.  A collapse is not
        confirmed if action-dependent outcomes return or if a dark/bright
        sequence resolves to a visually novel scene.
        """

        temporary_states: List[object] = []
        confirmation_state = state
        confirmation_frame = frame
        dark_encountered = False
        returned_to_known_scene = False
        novel_scene_observed = False
        control_returned = False
        control_returned_step: Optional[int] = None
        maximum_spread = 0.0
        observations: List[Dict[str, Any]] = []
        direct_novelty_threshold = max(
            0.05,
            self.config.known_scene_return_distance_threshold * 5.0,
        )

        try:
            for step in range(1, self.config.control_collapse_confirmation_steps + 1):
                intensity = self._mean_frame_intensity(confirmation_frame)
                is_dark = (
                    intensity
                    <= self.config.dark_transition_intensity_threshold
                )
                dark_encountered = dark_encountered or is_dark
                minimum_known_distance: Optional[float] = None
                if not is_dark and self.bright_scene_memory:
                    signature = self._persistent_cell_values(
                        confirmation_frame
                    )
                    minimum_known_distance = min(
                        self._coarse_scene_distance(signature, remembered)
                        for remembered in self.bright_scene_memory
                    )

                endpoints = []
                for probe_action in actions:
                    self.env.load_state(confirmation_state)
                    endpoints.append(
                        self.env.step(probe_action, duration)
                    )
                spread = max(
                    (
                        left.mean_absolute_difference(right)
                        for index, left in enumerate(endpoints)
                        for right in endpoints[index + 1 :]
                    ),
                    default=0.0,
                )
                maximum_spread = max(maximum_spread, spread)
                observations.append(
                    {
                        "step": step,
                        "scene_intensity": intensity,
                        "dark": is_dark,
                        "minimum_known_scene_distance": (
                            minimum_known_distance
                        ),
                        "future_outcome_spread": spread,
                    }
                )
                if spread > self.config.action_equivalence_threshold:
                    control_returned = True
                    control_returned_step = step
                    break

                if (
                    not is_dark
                    and minimum_known_distance is not None
                    and dark_encountered
                ):
                    returned_to_known_scene = (
                        minimum_known_distance
                        <= self.config.known_scene_return_distance_threshold
                    )
                    novel_scene_observed = not returned_to_known_scene
                    break

                if (
                    not is_dark
                    and minimum_known_distance is not None
                    and minimum_known_distance > direct_novelty_threshold
                ):
                    novel_scene_observed = True
                    break

                if step >= self.config.control_collapse_confirmation_steps:
                    break
                self.env.load_state(confirmation_state)
                confirmation_frame = self.env.step(Action.NOOP, duration)
                confirmation_state = self.env.save_state()
                temporary_states.append(confirmation_state)
        finally:
            release_state = getattr(self.env, "release_state", None)
            if release_state is not None:
                for temporary_state in reversed(temporary_states):
                    try:
                        release_state(temporary_state)
                    except Exception:
                        pass

        return {
            "control_collapsed": not (
                control_returned or novel_scene_observed
            ),
            "control_returned": control_returned,
            "control_returned_step": control_returned_step,
            "dark_encountered": dark_encountered,
            "returned_to_known_scene": returned_to_known_scene,
            "novel_scene_observed": novel_scene_observed,
            "maximum_future_outcome_spread": maximum_spread,
            "confirmation_steps": len(observations),
            "observations": observations,
        }

    def _probe_delayed_scene_transition(
        self,
        state: object,
        frame: Frame,
        observation_duration: int,
    ) -> Dict[str, Any]:
        """Passively inspect a verified branch for a delayed scene outcome."""

        dark_encountered = False
        returned_to_known_scene = False
        novel_scene_observed = False
        resolution_step: Optional[int] = None
        maximum_visual_change = 0.0
        observations: List[Dict[str, Any]] = []
        self.env.load_state(state)
        for step in range(1, self.config.delayed_transition_probe_steps + 1):
            observed = self.env.step(Action.NOOP, observation_duration)
            intensity = self._mean_frame_intensity(observed)
            visual_change = frame.mean_absolute_difference(observed)
            maximum_visual_change = max(
                maximum_visual_change, visual_change
            )
            is_dark = (
                intensity
                <= self.config.dark_transition_intensity_threshold
            )
            dark_encountered = dark_encountered or is_dark
            minimum_known_distance: Optional[float] = None
            if not is_dark and self.bright_scene_memory:
                signature = self._persistent_cell_values(observed)
                minimum_known_distance = min(
                    self._coarse_scene_distance(signature, remembered)
                    for remembered in self.bright_scene_memory
                )
            observations.append(
                {
                    "step": step,
                    "scene_intensity": intensity,
                    "dark": is_dark,
                    "visual_change_from_endpoint": visual_change,
                    "minimum_known_scene_distance": (
                        minimum_known_distance
                    ),
                }
            )
            if (
                dark_encountered
                and not is_dark
                and minimum_known_distance is not None
            ):
                returned_to_known_scene = (
                    minimum_known_distance
                    <= self.config.known_scene_return_distance_threshold
                )
                novel_scene_observed = not returned_to_known_scene
                resolution_step = step
                break
        return {
            "dark_encountered": dark_encountered,
            "returned_to_known_scene": returned_to_known_scene,
            "novel_scene_observed": novel_scene_observed,
            "resolution_step": resolution_step,
            "observation_duration": observation_duration,
            "maximum_visual_change": maximum_visual_change,
            "observations": observations,
        }

    def _option_initiation_evidence(
        self,
        candidate: Tuple[Any, ...],
        verified: List[Tuple[Any, ...]],
    ) -> Tuple[bool, float, int]:
        plan = candidate[1]
        target = candidate[3]
        action = plan.path[0]
        duration = plan.durations[0]
        counterfactuals = [
            item
            for item in verified
            if item[1].durations[0] == duration
            and item[1].path[0] != action
        ]
        contrast = max(
            (
                target.mean_absolute_difference(item[3])
                for item in counterfactuals
            ),
            default=0.0,
        )
        return (
            bool(counterfactuals)
            and contrast > self.config.action_equivalence_threshold,
            contrast,
            len(counterfactuals),
        )

    def _include_matched_effect_option_evidence(
        self,
        action: Action,
        effect: Optional[Dict[str, Any]],
        eligible: bool,
        contrast: float,
        counterfactuals: int,
    ) -> Tuple[bool, float, int]:
        if (
            action == Action.NOOP
            or effect is None
            or effect["contrast"]
            <= self.config.action_equivalence_threshold
        ):
            return eligible, contrast, counterfactuals
        return True, max(contrast, effect["contrast"]), counterfactuals + 1

    def _delayed_option_counterfactual(
        self,
        candidate: Tuple[Any, ...],
        verified: List[Tuple[Any, ...]],
    ) -> Optional[Tuple[Any, ...]]:
        plan = candidate[1]
        target = candidate[3]
        action = plan.path[0]
        duration = plan.durations[0]
        matched = [
            item
            for item in verified
            if item[1].durations[0] == duration
            and item[1].path[0] != action
            and target.mean_absolute_difference(item[3])
            <= self.config.action_equivalence_threshold
        ]
        if not matched:
            return None
        return min(
            matched,
            key=lambda item: (
                item[1].path[0] != Action.NOOP,
                target.mean_absolute_difference(item[3]),
                item[1].path[0].value,
            ),
        )

    def decide(self) -> Decision:
        if self.frame is None:
            self.reset()
        assert self.frame is not None
        self._apply_pending_novel_room_reset()
        self._calibrate_goal_prior(self.frame)
        self._emit(
            "decision_started",
            decision=self.decision_index + 1,
            action_counts=self.action_counts,
            duration_counts=self.duration_counts,
            action_duration_counts=self._action_duration_count_rows(),
            last_action=self.last_action,
            last_duration=self.last_duration,
            action_streak=self.action_streak,
            scene_streak=self.scene_streak,
            visual_stagnation_streak=self.visual_stagnation_streak,
            visual_stagnation_limit=self.config.visual_stagnation_visits,
            frontier_signature=self.current_frontier_signature,
            scene_visits=self.scene_visits,
            archive_size=len(self.archive),
            search_depth=self.current_search_depth,
            **self._frame_fields(self.frame),
        )
        life_recovery = self._restore_after_life_loss()
        if life_recovery is not None:
            return life_recovery
        semantic_stagnation_limit = (
            self.config.human_prior_graph_stagnation_visits
        )
        current_goal_graph_signature = (
            self._current_human_prior_graph_signature()
        )
        current_goal_graph_visits = (
            0
            if not current_goal_graph_signature
            else self.human_prior_graph_state_visits[
                current_goal_graph_signature
            ]
        )
        human_prior_graph_stagnant = bool(
            self.config.human_prior_best_first_archive
            and semantic_stagnation_limit > 0
            and current_goal_graph_signature
            and current_goal_graph_visits >= semantic_stagnation_limit
        )
        self.human_prior_graph_recovery_pending = bool(
            human_prior_graph_stagnant and self.archive
        )
        option_search_exhausted = False
        if human_prior_graph_stagnant:
            self._emit(
                "human_prior_graph_stagnation_detected",
                decision=self.decision_index + 1,
                goal_signature=current_goal_graph_signature,
                state_visits=current_goal_graph_visits,
                stagnation_limit=semantic_stagnation_limit,
                archive_size=len(self.archive),
                **self._frame_fields(self.frame),
            )
            unvisited_archive_endpoints = (
                self._human_prior_unvisited_archive_endpoints(
                    current_goal_graph_signature
                )
            )
            if (
                self.config.human_prior_option_search_depth >= 2
                and unvisited_archive_endpoints > 0
            ):
                self._emit(
                    "human_prior_option_search_deferred",
                    decision=self.decision_index + 1,
                    reason="unvisited_local_archive_endpoint_available",
                    source_graph_signature=(
                        current_goal_graph_signature
                    ),
                    unvisited_archive_endpoints=(
                        unvisited_archive_endpoints
                    ),
                    archive_size=len(self.archive),
                    **self._frame_fields(self.frame),
                )
            else:
                option_search_exhausted = bool(
                    self.config.human_prior_option_search_depth >= 2
                    and self._search_human_prior_options() == 0
                )
        restored = self._restore_if_stagnant()
        if restored is not None:
            return restored
        if (
            self.config.human_prior_goal_exhaustion_rollback
            and option_search_exhausted
            and self.pending_goal_milestone_checkpoint is not None
        ):
            goal_exhaustion_recovery = (
                self._restore_goal_milestone_after_exhaustion(
                    current_goal_graph_signature,
                    current_goal_graph_visits,
                )
            )
            if goal_exhaustion_recovery is not None:
                return goal_exhaustion_recovery
        anticipated_transition_observation_due = (
            self.anticipated_transition_observations_remaining > 0
        )
        autonomous_intervention_due = self.autonomous_intervention_pending
        causal_observation_intervention_due = (
            self.causal_observation_intervention_pending
        )
        control_intervention_due = (
            autonomous_intervention_due
            or causal_observation_intervention_due
        )
        if control_intervention_due:
            self._emit(
                (
                    "autonomous_intervention_started"
                    if autonomous_intervention_due
                    else "causal_observation_intervention_started"
                ),
                decision=self.decision_index + 1,
                reason=(
                    "autonomous_dynamics"
                    if autonomous_intervention_due
                    else "matched_causal_observation_completed"
                ),
                visual_stagnation_streak=self.visual_stagnation_streak,
                archive_size=len(self.archive),
                **self._frame_fields(self.frame),
            )
            self.autonomous_intervention_pending = False
            self.causal_observation_intervention_pending = False
        plans = self.planner.plan(self.frame)
        shadow_candidates = (
            [{} for _ in plans]
            if self.spatial_shadow is None
            else self.spatial_shadow.score_plans(
                self.frame,
                [(plan.path, plan.durations) for plan in plans],
            )
        )
        spatial_mode = (
            "verification_priority"
            if self.config.spatial_selection_weight > 0.0
            else "observational"
        )
        shadow_by_plan_id = {
            id(plan): metrics
            for plan, metrics in zip(plans, shadow_candidates)
        }

        def spatial_bonus(plan: NeuralPlan) -> float:
            metrics = shadow_by_plan_id.get(id(plan), {})
            return self.config.spatial_selection_weight * float(
                metrics.get("spatial_shadow_score", 0.0)
            )

        candidate_rows = []
        for rank, (plan, shadow_metrics) in enumerate(
            zip(plans, shadow_candidates), 1
        ):
            candidate = {
                "rank": rank,
                "path": plan.path,
                "durations": plan.durations,
                "model_score": plan.score,
                "uncertainty": plan.uncertainty,
                "first_action_penalty": self._action_penalty(
                    plan.path[0], plan.durations[0]
                ),
                "first_action_penalty_components": self._action_penalty_components(
                    plan.path[0], plan.durations[0]
                ),
            }
            if shadow_metrics:
                candidate.update(
                    {
                        "spatial_shadow_mode": spatial_mode,
                        "spatial_shadow_selection_weight": (
                            self.config.spatial_selection_weight
                        ),
                        "spatial_shadow_selection_bonus": spatial_bonus(plan),
                        **shadow_metrics,
                    }
                )
            candidate_rows.append(candidate)
        self._emit(
            "planner_candidates",
            decision=self.decision_index + 1,
            candidates=candidate_rows,
        )
        best_by_action: Dict[Tuple[Action, int], NeuralPlan] = {}
        for plan in plans:
            timed_action = (plan.path[0], plan.durations[0])
            if (
                timed_action not in best_by_action
                or plan.score + spatial_bonus(plan)
                > best_by_action[timed_action].score
                + spatial_bonus(best_by_action[timed_action])
            ):
                best_by_action[timed_action] = plan
        current_scene = self._scene_signature(self.frame)
        source_frame = self.frame
        source_causal_context_signature = self.current_causal_context_signature
        source_human_prior_world_context_signature = (
            self.current_human_prior_world_context_signature
        )
        source_signature = self.current_frontier_signature
        source_pose_action = self.current_pose_action
        source_last_action = self.last_action
        source_last_duration = self.last_duration
        source_action_streak = self.action_streak
        source_search_depth = self.current_search_depth
        source_goal_heart_slots = (
            () if self.goal_prior is None else self.goal_prior.current_slots()
        )
        source_goal_player_slot = (
            None
            if self.goal_prior is None
            else self.goal_prior.current_player_slot
        )
        source_goal_chest_obtained = bool(
            self.goal_prior is not None
            and self.goal_prior.chest_obtained
        )
        best_by_button: Dict[Action, NeuralPlan] = {}
        for plan in best_by_action.values():
            action = plan.path[0]
            existing = best_by_button.get(action)
            adjusted = (
                plan.score
                + spatial_bonus(plan)
                - self._action_penalty(action, plan.durations[0])
            )
            if existing is None or adjusted > existing.score - self._action_penalty(
                existing.path[0], existing.durations[0]
            ) + spatial_bonus(existing):
                best_by_button[action] = plan
        ranked = sorted(
            best_by_button.values(),
            key=lambda plan: (
                self.scene_action_probes[(current_scene, plan.path[0])],
                -(
                    plan.score
                    + spatial_bonus(plan)
                    - self._action_penalty(plan.path[0], plan.durations[0])
                ),
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
                    + spatial_bonus(plan)
                    - self._action_penalty(plan.path[0], plan.durations[0])
                ),
            )
            ranked.extend(remaining[: self.config.verify_actions - len(ranked)])
        probe_selection = self._behavior_probe_selection(self.frame)
        ranked = self._add_control_probes(
            ranked, best_by_action, probe_selection
        )
        if not ranked:
            raise RuntimeError("neural planner produced no action candidates")

        root = self.env.save_state()
        states = [root]
        pruned_state_ids: set[int] = set()
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

            neutral_outcomes = {
                item[0].durations[0]: item[2]
                for item in raw_verified
                if item[0].path[0] == Action.NOOP
            }
            requested_neutral_durations = sorted(
                {
                    item[0].durations[0]
                    for item in raw_verified
                    if item[0].path[0] != Action.NOOP
                }
                - set(neutral_outcomes)
            )
            for duration in requested_neutral_durations:
                self.env.load_state(root)
                neutral_target = self.env.step(Action.NOOP, duration)
                neutral_outcomes[duration] = neutral_target
                self._emit(
                    "matched_neutral_verified",
                    decision=self.decision_index + 1,
                    action=Action.NOOP,
                    action_frames=duration,
                    env_step_seq=getattr(self.env, "last_step_seq", None),
                    source_state_id=self._state_id(root),
                    **self._frame_fields(neutral_target),
                )

            probe_outcomes = {
                (item[0].path[0], item[0].durations[0]): item[2]
                for item in raw_verified
            }
            source_signature = self._behavioral_signature(
                self.frame,
                probe_outcomes,
                source_signature,
                probe_selection,
            )
            self.current_frontier_signature = source_signature
            observed_action_effects: Dict[Tuple[Action, int], Dict[str, Any]] = {}
            for item in raw_verified:
                plan = item[0]
                action = plan.path[0]
                duration = plan.durations[0]
                neutral_target = neutral_outcomes.get(duration)
                if action == Action.NOOP or neutral_target is None:
                    continue
                contrast = item[2].mean_absolute_difference(neutral_target)
                effect_value, effect_samples = self._record_action_effect(
                    source_signature, action, contrast
                )
                (
                    causal_spatial_signature,
                    causal_changed_pixels,
                    causal_change_centroid,
                ) = self._causal_spatial_effect(item[2], neutral_target)
                causal_spatial_visits = (
                    0
                    if causal_spatial_signature is None
                    else self.causal_spatial_visits[
                        self._causal_frontier_key(
                            source_causal_context_signature,
                            causal_spatial_signature,
                        )
                    ]
                )
                causal_spatial_novelty = (
                    0.0
                    if causal_spatial_signature is None
                    else 1.0 / math.sqrt(causal_spatial_visits + 1)
                )
                (
                    causal_cell_coverage,
                    causal_cell_unvisited,
                    causal_cell_count,
                ) = self._causal_cell_coverage(causal_spatial_signature)
                (
                    target_causal_context_signature,
                    causal_event_detected,
                    causal_component_count,
                ) = self._causal_target_context(
                    source_causal_context_signature,
                    causal_spatial_signature,
                )
                if action in (Action.A, Action.B):
                    if causal_spatial_signature:
                        self.discovered_interaction_actions.add(action)
                        self.discovered_interaction_durations.setdefault(
                            action, set()
                        ).add(duration)
                    target_causal_context_signature = (
                        source_causal_context_signature
                    )
                    causal_event_detected = False
                observed_action_effects[(action, duration)] = {
                    "contrast": contrast,
                    "value": effect_value,
                    "samples": effect_samples,
                    "causal_spatial_signature": causal_spatial_signature,
                    "causal_context_signature": source_causal_context_signature,
                    "causal_frontier_key": (
                        None
                        if causal_spatial_signature is None
                        else self._causal_frontier_key(
                            source_causal_context_signature,
                            causal_spatial_signature,
                        )
                    ),
                    "causal_changed_pixels": causal_changed_pixels,
                    "causal_change_centroid": causal_change_centroid,
                    "causal_spatial_visits": causal_spatial_visits,
                    "causal_spatial_novelty": causal_spatial_novelty,
                    "causal_cell_coverage": causal_cell_coverage,
                    "causal_cell_unvisited": causal_cell_unvisited,
                    "causal_cell_count": causal_cell_count,
                    "target_causal_context_signature": (
                        target_causal_context_signature
                    ),
                    "causal_event_detected": causal_event_detected,
                    "causal_component_count": causal_component_count,
                }
            verified = []
            source_causal_affordance_actions = tuple(
                sorted(
                    {
                        action
                        for (action, _duration), effect in observed_action_effects.items()
                        if action in (Action.A, Action.B)
                        and effect["causal_spatial_signature"]
                    },
                    key=lambda action: action.value,
                )
            )
            for effect in observed_action_effects.values():
                spatial_signature = effect["causal_spatial_signature"]
                if not spatial_signature:
                    continue
                frontier_key = self._causal_frontier_key(
                    source_causal_context_signature,
                    spatial_signature,
                    source_causal_affordance_actions,
                )
                visits = self.causal_spatial_visits[frontier_key]
                effect["causal_frontier_key"] = frontier_key
                effect["causal_spatial_visits"] = visits
                effect["causal_spatial_novelty"] = 1.0 / math.sqrt(
                    visits + 1
                )
            branch_causal_contexts: Dict[int, Dict[str, Any]] = {}
            branch_action_penalties: Dict[int, Dict[str, float]] = {}
            branch_goal_analyses: Dict[int, Optional[HeartGoalAnalysis]] = {}
            branch_goal_signatures: Dict[int, Tuple[str, str]] = {}
            branch_goal_world_contexts: Dict[
                int, Tuple[str, str, str]
            ] = {}
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
                temporal_option_value, temporal_option_is_known = (
                    self._temporal_option_estimate(
                        source_signature, plan.path[0], duration
                    )
                )
                observed_effect = observed_action_effects.get(
                    (plan.path[0], duration)
                )
                if observed_effect is None:
                    (
                        action_effect_value,
                        action_effect_is_known,
                        action_effect_samples,
                    ) = self._action_effect_estimate(
                        source_signature, plan.path[0]
                    )
                    action_effect_contrast = None
                    causal_spatial_signature = None
                    causal_changed_pixels = 0
                    causal_change_centroid = None
                    causal_spatial_visits = 0
                    causal_spatial_novelty = 0.0
                    causal_cell_coverage = 0.0
                    causal_cell_unvisited = 0
                    causal_cell_count = 0
                else:
                    action_effect_contrast = observed_effect["contrast"]
                    action_effect_value = observed_effect["value"]
                    action_effect_samples = observed_effect["samples"]
                    action_effect_is_known = True
                    causal_spatial_signature = observed_effect[
                        "causal_spatial_signature"
                    ]
                    causal_changed_pixels = observed_effect[
                        "causal_changed_pixels"
                    ]
                    causal_change_centroid = observed_effect[
                        "causal_change_centroid"
                    ]
                    causal_spatial_visits = observed_effect[
                        "causal_spatial_visits"
                    ]
                    causal_spatial_novelty = observed_effect[
                        "causal_spatial_novelty"
                    ]
                    causal_cell_coverage = observed_effect[
                        "causal_cell_coverage"
                    ]
                    causal_cell_unvisited = observed_effect[
                        "causal_cell_unvisited"
                    ]
                    causal_cell_count = observed_effect[
                        "causal_cell_count"
                    ]
                transition_spatial_signature = self._causal_spatial_effect(
                    target, self.frame
                )[0]
                (
                    branch_target_causal_context_signature,
                    branch_causal_event_detected,
                    branch_causal_component_count,
                ) = self._causal_target_context(
                    source_causal_context_signature,
                    transition_spatial_signature,
                )
                causal_event_basis = (
                    "observed_transition"
                    if branch_causal_event_detected
                    else None
                )
                if plan.path[0] in (Action.A, Action.B):
                    branch_target_causal_context_signature = (
                        source_causal_context_signature
                    )
                    branch_causal_event_detected = False
                    causal_event_basis = None
                transition_cells = self._causal_spatial_cells(
                    transition_spatial_signature
                )
                causal_event_novel_cells = sum(
                    self.causal_spatial_cell_visits[cell] == 0
                    for cell in transition_cells
                )
                if (
                    branch_causal_event_detected
                    and causal_event_novel_cells == 0
                ):
                    branch_target_causal_context_signature = (
                        source_causal_context_signature
                    )
                    branch_causal_event_detected = False
                    causal_event_basis = None
                if (
                    observed_effect is not None
                    and observed_effect["causal_event_detected"]
                ):
                    branch_target_causal_context_signature = observed_effect[
                        "target_causal_context_signature"
                    ]
                    branch_causal_event_detected = True
                    branch_causal_component_count = observed_effect[
                        "causal_component_count"
                    ]
                    causal_event_basis = "matched_action_effect"
                branch_causal_contexts[id(state)] = {
                    "target": branch_target_causal_context_signature,
                    "detected": branch_causal_event_detected,
                    "components": branch_causal_component_count,
                    "basis": causal_event_basis,
                    "transition_spatial_signature": (
                        transition_spatial_signature
                    ),
                    "novel_cells": causal_event_novel_cells,
                }
                action_effect_bonus = (
                    self.config.action_effect_weight * action_effect_value
                    if temporal_option_value >= 0.0
                    else 0.0
                )
                causal_spatial_bonus = (
                    self.config.causal_spatial_novelty_weight
                    * causal_spatial_novelty
                    if temporal_option_value >= 0.0
                    else 0.0
                )
                causal_cell_coverage_bonus = (
                    self.config.causal_cell_coverage_weight
                    * causal_cell_coverage
                    if temporal_option_value >= 0.0
                    else 0.0
                )
                (
                    behavioral_edge_visits_before,
                    behavioral_edge_unexpanded,
                    behavioral_edge_coverage_bonus,
                ) = self._behavioral_edge_coverage(
                    source_signature, plan.path[0], duration
                )
                if temporal_option_value < 0.0:
                    behavioral_edge_coverage_bonus = 0.0
                if choice_frontier_is_known:
                    persistent_frontier_value = choice_frontier_value
                action_penalty_components = self._action_penalty_components(
                    plan.path[0], duration
                )
                branch_action_penalties[id(state)] = action_penalty_components
                intrinsic_score = (
                    plan.score
                    + self.config.actual_novelty_weight * effective_novelty
                    + self.config.scene_novelty_weight * scene_novelty
                    + self.config.prediction_error_weight * error
                    + self.config.actual_change_weight * visual_change
                    + action_effect_bonus
                    + causal_spatial_bonus
                    + causal_cell_coverage_bonus
                    + behavioral_edge_coverage_bonus
                    + self.config.frontier_score_weight * persistent_frontier_value
                    + self.config.temporal_option_score_weight
                    * temporal_option_value
                    - action_penalty_components["action_penalty"]
                )
                goal_analysis = (
                    None
                    if self.goal_prior is None
                    else self.goal_prior.analyze(source_frame, target)
                )
                branch_goal_analyses[id(state)] = goal_analysis
                goal_world_effect_signature = (
                    self._human_prior_world_effect_signature(
                        causal_spatial_signature,
                        goal_analysis,
                        target,
                        plan.path[0],
                    )
                )
                world_effect_confirmation: Optional[Dict[str, Any]] = None
                if (
                    goal_world_effect_signature
                    and self.config.human_prior_best_first_archive
                    and len(self.config.actions) > 1
                ):
                    control_probe_actions = tuple(
                        dict.fromkeys((*self.config.actions, Action.NOOP))
                    )
                    try:
                        world_effect_confirmation = (
                            self._confirm_future_control_collapse(
                                state,
                                target,
                                control_probe_actions,
                                duration,
                            )
                        )
                    finally:
                        self.env.load_state(root)
                    world_effect_confirmed = bool(
                        world_effect_confirmation["control_returned"]
                        and world_effect_confirmation[
                            "control_returned_step"
                        ]
                        == 1
                    )
                    self._emit(
                        "human_prior_world_effect_confirmation",
                        decision=self.decision_index + 1,
                        action=plan.path[0],
                        action_frames=duration,
                        accepted=world_effect_confirmed,
                        human_prior_world_effect_signature=(
                            goal_world_effect_signature
                        ),
                        **world_effect_confirmation,
                        **self._frame_fields(target),
                    )
                    if not world_effect_confirmed:
                        goal_world_effect_signature = ""
                goal_target_world_context = (
                    self._next_human_prior_world_context(
                        source_human_prior_world_context_signature,
                        goal_world_effect_signature,
                    )
                )
                branch_goal_world_contexts[id(state)] = (
                    source_human_prior_world_context_signature,
                    goal_target_world_context,
                    goal_world_effect_signature,
                )
                goal_source_signature, goal_target_signature = (
                    self._human_prior_graph_signatures(
                        goal_analysis,
                        source_human_prior_world_context_signature,
                        goal_target_world_context,
                    )
                )
                branch_goal_signatures[id(state)] = (
                    goal_source_signature,
                    goal_target_signature,
                )
                (
                    human_prior_graph_edge_visits_before,
                    human_prior_graph_edge_unexpanded,
                ) = self._human_prior_graph_edge_coverage(
                    goal_source_signature,
                    plan.path[0],
                    duration,
                )
                score, clipped_intrinsic_score = self._human_prior_score(
                    intrinsic_score, goal_analysis
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
                branch_id = (
                    f"decision-{self.decision_index + 1:08d}-"
                    f"branch-{candidate_rank:02d}"
                )
                if self.returnability_probe is not None:
                    self.returnability_probe.collect(
                        root_state=root,
                        endpoint_state=state,
                        source_frame=source_frame,
                        endpoint_frame=target,
                        initial_action=plan.path[0],
                        action_frames=duration,
                        decision=self.decision_index + 1,
                        branch_id=branch_id,
                        candidate_rank=candidate_rank,
                    )
                if self.spatial_shadow is not None:
                    self._emit(
                        "spatial_shadow_branch_evaluated",
                        decision=self.decision_index + 1,
                        branch_id=branch_id,
                        candidate_rank=candidate_rank,
                        action=plan.path[0],
                        action_frames=duration,
                        spatial_shadow_mode=spatial_mode,
                        spatial_shadow_selection_weight=(
                            self.config.spatial_selection_weight
                        ),
                        spatial_shadow_selection_bonus=spatial_bonus(plan),
                        spatial_shadow_actual_causal_contrast=(
                            action_effect_contrast
                        ),
                        **self.spatial_shadow.evaluate_transition(
                            source_frame,
                            plan.path[0],
                            duration,
                            target,
                        ),
                    )
                self._emit(
                    "branch_verified",
                    decision=self.decision_index + 1,
                    branch_id=branch_id,
                    candidate_rank=candidate_rank,
                    scene_action_probe_count=scene_action_probe_count,
                    env_step_seq=env_step_seq,
                    state_save_seq=state_save_seq,
                    source_state_id=self._state_id(root),
                    source_frame=source_frame.digest,
                    parent_decision=self.decision_index,
                    search_depth=source_search_depth + 1,
                    action=plan.path[0],
                    action_frames=duration,
                    path=plan.path,
                    durations=plan.durations,
                    model_score=plan.score,
                    model_uncertainty=plan.uncertainty,
                    spatial_selection_bonus=spatial_bonus(plan),
                    novelty=novelty,
                    effective_novelty=effective_novelty,
                    scene_novelty=scene_novelty,
                    target_signature_is_new=target_signature_is_new,
                    target_scene_is_new=target_scene_is_new,
                    target_scene=target_scene,
                    prediction_error=error,
                    visual_change=visual_change,
                    action_effect_contrast=action_effect_contrast,
                    action_effect_value=action_effect_value,
                    action_effect_is_known=action_effect_is_known,
                    action_effect_samples=action_effect_samples,
                    action_effect_bonus=action_effect_bonus,
                    causal_spatial_signature=causal_spatial_signature,
                    causal_context_signature=source_causal_context_signature,
                    target_causal_context_signature=(
                        branch_target_causal_context_signature
                    ),
                    causal_event_detected=branch_causal_event_detected,
                    causal_component_count=branch_causal_component_count,
                    causal_event_basis=causal_event_basis,
                    causal_event_novel_cells=causal_event_novel_cells,
                    transition_spatial_signature=transition_spatial_signature,
                    causal_changed_pixels=causal_changed_pixels,
                    causal_change_centroid=causal_change_centroid,
                    causal_spatial_visits=causal_spatial_visits,
                    causal_spatial_novelty=causal_spatial_novelty,
                    causal_spatial_bonus=causal_spatial_bonus,
                    causal_cell_coverage=causal_cell_coverage,
                    causal_cell_unvisited=causal_cell_unvisited,
                    causal_cell_count=causal_cell_count,
                    causal_cell_coverage_bonus=causal_cell_coverage_bonus,
                    behavioral_edge_visits_before=(
                        behavioral_edge_visits_before
                    ),
                    behavioral_edge_unexpanded=(
                        behavioral_edge_unexpanded
                    ),
                    behavioral_edge_coverage_bonus=(
                        behavioral_edge_coverage_bonus
                    ),
                    behavioral_best_first_archive_enabled=(
                        self.config.behavioral_best_first_archive
                    ),
                    persistent_frontier_value=persistent_frontier_value,
                    choice_frontier_value=choice_frontier_value,
                    choice_frontier_is_known=choice_frontier_is_known,
                    temporal_option_value=temporal_option_value,
                    temporal_option_is_known=temporal_option_is_known,
                    temporal_option_value_source=(
                        self._temporal_option_estimate_source(
                            source_signature, plan.path[0], duration
                        )
                    ),
                    temporal_option_bonus=(
                        self.config.temporal_option_score_weight
                        * temporal_option_value
                    ),
                    intrinsic_score=intrinsic_score,
                    human_prior_clipped_intrinsic_score=(
                        clipped_intrinsic_score
                    ),
                    human_prior_graph_source_signature=(
                        goal_source_signature or None
                    ),
                    human_prior_graph_target_signature=(
                        goal_target_signature or None
                    ),
                    human_prior_world_source_context=(
                        source_human_prior_world_context_signature
                    ),
                    human_prior_world_target_context=(
                        goal_target_world_context
                    ),
                    human_prior_world_effect_signature=(
                        goal_world_effect_signature or None
                    ),
                    human_prior_graph_edge_visits_before=(
                        human_prior_graph_edge_visits_before
                    ),
                    human_prior_graph_edge_unexpanded=(
                        human_prior_graph_edge_unexpanded
                    ),
                    **self._human_prior_fields(goal_analysis),
                    abstract_signature=target_visual_cluster,
                    source_behavioral_signature=source_signature,
                    target_frontier_signature=target_frontier_signature,
                    **action_penalty_components,
                    combined_score=score,
                    state_id=self._state_id(state),
                    **self._frame_fields(target),
                )
            selection_verified, filtered_hazards = (
                self._verified_without_learned_hazards(
                    source_signature, verified
                )
            )
            positive_goal_branches = [
                item
                for item in selection_verified
                if branch_goal_analyses[id(item[2])] is not None
                and branch_goal_analyses[id(item[2])].milestone_reward > 0.0
            ]
            if filtered_hazards:
                self._emit(
                    "learned_hazards_filtered",
                    decision=self.decision_index + 1,
                    phase="commit_selection",
                    filtered=filtered_hazards,
                    alternatives_remaining=len(selection_verified),
                )
            delayed_transition_candidates: List[
                Tuple[Tuple[Any, ...], Dict[str, Any]]
            ] = []
            if (
                self.config.delayed_transition_probe_steps > 0
                and not anticipated_transition_observation_due
            ):
                observation_duration = max(
                    self.planner.duration_choices
                )
                for candidate in selection_verified:
                    action = candidate[1].path[0]
                    duration = candidate[1].durations[0]
                    effect = observed_action_effects.get(
                        (action, duration)
                    )
                    if (
                        action == Action.NOOP
                        or effect is None
                        or effect["contrast"]
                        <= self.config.action_equivalence_threshold
                    ):
                        continue
                    try:
                        delayed_probe = (
                            self._probe_delayed_scene_transition(
                                candidate[2],
                                candidate[3],
                                observation_duration,
                            )
                        )
                    finally:
                        self.env.load_state(root)
                    self._emit(
                        "delayed_transition_probe",
                        decision=self.decision_index + 1,
                        action=action,
                        action_frames=duration,
                        probe_steps=(
                            self.config.delayed_transition_probe_steps
                        ),
                        state_id=self._state_id(candidate[2]),
                        **delayed_probe,
                        **self._frame_fields(candidate[3]),
                    )
                    if delayed_probe["novel_scene_observed"]:
                        delayed_transition_candidates.append(
                            (candidate, delayed_probe)
                        )
            delayed_transition_choice = (
                None
                if not delayed_transition_candidates
                else max(
                    delayed_transition_candidates,
                    key=lambda row: (
                        -int(row[1]["resolution_step"]),
                        row[0][0],
                    ),
                )
            )
            autonomous = self._autonomous_choice(
                self.frame, selection_verified
            )
            learned_control_actions = sorted(
                (
                    action
                    for (signature, action), value in self.action_effect_values.items()
                    if signature == source_signature and value > 0.5
                ),
                key=lambda action: action.value,
            )
            causal_counterfactual_active = (
                self.pending_option_counterfactual is not None
                or (
                    self.pending_option_choice is not None
                    and self.pending_option_causal_evidence
                )
                or (
                    self.active_temporal_option is not None
                    and (
                        self.active_temporal_option.counterfactual is not None
                        or self.active_temporal_option.causal_evidence
                    )
                )
            )
            causal_observation_wait = None
            if (
                self.pending_option_choice is not None
                and self.pending_option_causal_evidence
            ):
                neutral = [
                    item
                    for item in selection_verified
                    if item[1].path[0] == Action.NOOP
                ]
                if neutral:
                    pending_duration = self.pending_option_choice[2]
                    matched_neutral = [
                        item
                        for item in neutral
                        if item[1].durations[0] == pending_duration
                    ]
                    causal_observation_wait = max(
                        matched_neutral or neutral,
                        key=lambda item: (
                            -abs(
                                item[1].durations[0] - pending_duration
                            ),
                            item[0],
                        ),
                    )
            human_prior_goal_choice = (
                max(
                    positive_goal_branches,
                    key=lambda item: (
                        branch_goal_analyses[id(item[2])].milestone_reward,
                        item[0],
                    ),
                )
                if positive_goal_branches
                else None
            )
            dynamic_control_choice = None
            control_probe_spread = None
            control_probe_actions: Tuple[Action, ...] = ()
            if (
                causal_observation_wait is not None
                and not anticipated_transition_observation_due
            ):
                passive_visual_change = causal_observation_wait[6]
                action_dependent_controls = [
                    item
                    for item in selection_verified
                    if (
                        item[1].path[0] != Action.NOOP
                        and observed_action_effects.get(
                            (
                                item[1].path[0],
                                item[1].durations[0],
                            )
                        )
                        is not None
                        and observed_action_effects[
                            (
                                item[1].path[0],
                                item[1].durations[0],
                            )
                        ]["contrast"]
                        > self.config.action_equivalence_threshold
                    )
                ]
                control_probe_actions = tuple(
                    dict.fromkeys((*self.config.actions, Action.NOOP))
                )
                probe_duration = causal_observation_wait[1].durations[0]

                def future_control_spread(state: object) -> float:
                    endpoints = []
                    for probe_action in control_probe_actions:
                        self.env.load_state(state)
                        endpoints.append(
                            self.env.step(probe_action, probe_duration)
                        )
                    return max(
                        (
                            first.mean_absolute_difference(second)
                            for index, first in enumerate(endpoints)
                            for second in endpoints[index + 1 :]
                        ),
                        default=0.0,
                    )

                try:
                    control_confirmation = (
                        self._confirm_future_control_collapse(
                            causal_observation_wait[2],
                            causal_observation_wait[3],
                            control_probe_actions,
                            probe_duration,
                        )
                    )
                finally:
                    self.env.load_state(root)
                control_probe_spread = control_confirmation[
                    "maximum_future_outcome_spread"
                ]
                control_collapsed = control_confirmation[
                    "control_collapsed"
                ]
                self._emit(
                    "counterfactual_control_confirmation",
                    decision=self.decision_index + 1,
                    passive_action=causal_observation_wait[1].path[0],
                    passive_action_frames=probe_duration,
                    **control_confirmation,
                )
                self._emit(
                    "counterfactual_control_probe",
                    decision=self.decision_index + 1,
                    passive_action=causal_observation_wait[1].path[0],
                    passive_action_frames=probe_duration,
                    passive_visual_change=passive_visual_change,
                    probe_actions=control_probe_actions,
                    future_outcome_spread=control_probe_spread,
                    control_collapsed=control_collapsed,
                    confirmation_steps=control_confirmation[
                        "confirmation_steps"
                    ],
                    control_returned=control_confirmation[
                        "control_returned"
                    ],
                    control_returned_step=control_confirmation[
                        "control_returned_step"
                    ],
                    dark_encountered=control_confirmation[
                        "dark_encountered"
                    ],
                    returned_to_known_scene=control_confirmation[
                        "returned_to_known_scene"
                    ],
                    novel_scene_observed=control_confirmation[
                        "novel_scene_observed"
                    ],
                    action_dependent_controls=len(
                        action_dependent_controls
                    ),
                    **self._frame_fields(causal_observation_wait[3]),
                )
                if control_collapsed:
                    control_escape_rows = []
                    try:
                        for candidate in action_dependent_controls:
                            candidate_spread = future_control_spread(
                                candidate[2]
                            )
                            control_escape_rows.append(
                                (candidate, candidate_spread)
                            )
                    finally:
                        self.env.load_state(root)
                    viable_controls = [
                        row
                        for row in control_escape_rows
                        if row[1]
                        > self.config.action_equivalence_threshold
                    ]
                    ranked_controls = (
                        viable_controls
                        if viable_controls
                        else control_escape_rows
                    )
                    if viable_controls:
                        dynamic_control_choice = max(
                            ranked_controls,
                            key=lambda row: (
                                row[0][0],
                                row[1],
                                observed_action_effects[
                                    (
                                        row[0][1].path[0],
                                        row[0][1].durations[0],
                                    )
                                ]["contrast"],
                            ),
                        )[0]
                    self._emit(
                        "counterfactual_control_escape_probe",
                        decision=self.decision_index + 1,
                        passive_future_outcome_spread=(
                            control_probe_spread
                        ),
                        alternatives=[
                            {
                                "action": candidate[1].path[0],
                                "action_frames": candidate[1].durations[0],
                                "score": candidate[0],
                                "future_outcome_spread": candidate_spread,
                                "control_viable": (
                                    candidate_spread
                                    > self.config.action_equivalence_threshold
                                ),
                            }
                            for candidate, candidate_spread in (
                                control_escape_rows
                            )
                        ],
                        viable_alternatives=len(viable_controls),
                        selected_action=(
                            None
                            if dynamic_control_choice is None
                            else dynamic_control_choice[1].path[0]
                        ),
                        selected_action_frames=(
                            None
                            if dynamic_control_choice is None
                            else dynamic_control_choice[1].durations[0]
                        ),
                    )
                    if not viable_controls:
                        recovery = self.pending_option_recovery_checkpoint
                        learned_choice = self.pending_option_choice
                        learned_value = None
                        learned_samples = 0
                        if learned_choice is not None:
                            learned_value = self._record_temporal_option_sample(
                                learned_choice,
                                -self.config.temporal_option_return_penalty,
                            )
                            learned_samples = self.temporal_option_samples[
                                learned_choice
                            ]
                        if recovery is not None:
                            self.pending_option_recovery_checkpoint = None
                            if self.pending_life_recovery is not None:
                                self._release_life_hazard_checkpoint(
                                    self.pending_life_recovery,
                                    "superseded_by_control_collapse",
                                )
                            self.pending_life_recovery = recovery
                            self.pending_recovery_cause = "control_collapse"
                        self._penalize_frontier_loop(self.decision_index)
                        self._emit(
                            "counterfactual_control_collapse_learned",
                            decision=self.decision_index + 1,
                            choice=learned_choice,
                            learned_hazard_value=learned_value,
                            learned_hazard_samples=learned_samples,
                            recovery_checkpoint_available=recovery is not None,
                            recovery_state_id=(
                                None if recovery is None else recovery.state_id
                            ),
                            passive_future_outcome_spread=control_probe_spread,
                            viable_alternatives=0,
                        )
            control_intervention_choice = None
            if control_intervention_due:
                intervention_choices = [
                    item
                    for item in selection_verified
                    if item[1].path[0] != Action.NOOP
                ]
                if intervention_choices:
                    action_dependent_interventions = [
                        item
                        for item in intervention_choices
                        if (
                            observed_action_effects.get(
                                (
                                    item[1].path[0],
                                    item[1].durations[0],
                                )
                            )
                            is not None
                            and observed_action_effects[
                                (
                                    item[1].path[0],
                                    item[1].durations[0],
                                )
                            ]["contrast"]
                            > self.config.action_equivalence_threshold
                        )
                    ]
                    if action_dependent_interventions:
                        intervention_choices = (
                            action_dependent_interventions
                        )
                    control_intervention_choice = max(
                        intervention_choices,
                        key=lambda item: (
                            (
                                observed_action_effects.get(
                                    (
                                        item[1].path[0],
                                        item[1].durations[0],
                                    ),
                                    {},
                                ).get("contrast", 0.0)
                            ),
                            item[0],
                            tuple(
                                (action.value, duration)
                                for action, duration in zip(
                                    item[1].path, item[1].durations
                                )
                            ),
                        ),
                    )
            if (
                autonomous is not None
                and learned_control_actions
                and causal_observation_wait is None
            ):
                self._emit(
                    "autonomous_dynamics_rejected",
                    decision=self.decision_index + 1,
                    reason="state_has_learned_control",
                    learned_control_actions=learned_control_actions,
                    proposed_outcome_spread=autonomous[1],
                    proposed_autonomous_change=autonomous[2],
                )
                autonomous = None
            anticipated_observation_choice = None
            if anticipated_transition_observation_due:
                neutral = [
                    item
                    for item in selection_verified
                    if item[1].path[0] == Action.NOOP
                ]
                if neutral:
                    matched = [
                        item
                        for item in neutral
                        if item[1].durations[0]
                        == self.anticipated_transition_observation_duration
                    ]
                    anticipated_observation_choice = max(
                        matched or neutral,
                        key=lambda item: (
                            -abs(
                                item[1].durations[0]
                                - self.anticipated_transition_observation_duration
                            ),
                            item[0],
                        ),
                    )
            passive_transition = False
            grace_continuation = False
            if anticipated_observation_choice is not None:
                chosen = anticipated_observation_choice
                passive_transition = True
                self.anticipated_transition_observations_remaining -= 1
                self._emit(
                    "anticipated_transition_observation",
                    decision=self.decision_index + 1,
                    action=chosen[1].path[0],
                    action_frames=chosen[1].durations[0],
                    observations_remaining=(
                        self.anticipated_transition_observations_remaining
                    ),
                )
            elif delayed_transition_choice is not None:
                chosen, delayed_probe = delayed_transition_choice
                self.anticipated_transition_observations_remaining = int(
                    delayed_probe["resolution_step"]
                )
                self.anticipated_transition_observation_duration = int(
                    delayed_probe["observation_duration"]
                )
                self._emit(
                    "delayed_transition_branch_selected",
                    decision=self.decision_index + 1,
                    action=chosen[1].path[0],
                    action_frames=chosen[1].durations[0],
                    observations_scheduled=(
                        self.anticipated_transition_observations_remaining
                    ),
                    observation_duration=(
                        self.anticipated_transition_observation_duration
                    ),
                    state_id=self._state_id(chosen[2]),
                    **self._frame_fields(chosen[3]),
                )
            elif human_prior_goal_choice is not None:
                chosen = human_prior_goal_choice
                selected_analysis = branch_goal_analyses[id(chosen[2])]
                self._emit(
                    "human_prior_goal_choice",
                    decision=self.decision_index + 1,
                    action=chosen[1].path[0],
                    action_frames=chosen[1].durations[0],
                    **self._human_prior_fields(selected_analysis),
                )
            elif dynamic_control_choice is not None:
                chosen = dynamic_control_choice
                selected_effect = observed_action_effects[
                    (
                        chosen[1].path[0],
                        chosen[1].durations[0],
                    )
                ]
                self._emit(
                    "dynamic_control_selected",
                    decision=self.decision_index + 1,
                    action=chosen[1].path[0],
                    action_frames=chosen[1].durations[0],
                    action_effect_contrast=selected_effect["contrast"],
                    passive_visual_change=causal_observation_wait[6],
                    future_outcome_spread=control_probe_spread,
                    control_probe_actions=control_probe_actions,
                    reason="counterfactual_future_control_collapse",
                )
            elif control_intervention_choice is not None:
                chosen = control_intervention_choice
                self._emit(
                    (
                        "autonomous_intervention_selected"
                        if autonomous_intervention_due
                        else "causal_observation_intervention_selected"
                    ),
                    decision=self.decision_index + 1,
                    action=chosen[1].path[0],
                    action_frames=chosen[1].durations[0],
                    reason=(
                        "autonomous_dynamics"
                        if autonomous_intervention_due
                        else "matched_causal_observation_completed"
                    ),
                    autonomous_dynamics_detected=autonomous is not None,
                )
            elif causal_observation_wait is not None:
                chosen = causal_observation_wait
                passive_transition = True
                self.causal_observation_intervention_pending = True
                self._emit(
                    "causal_observation_wait",
                    decision=self.decision_index + 1,
                    choice=self.pending_option_choice,
                    selected_duration=chosen[1].durations[0],
                    initiation_duration=(
                        None
                        if self.pending_option_choice is None
                        else self.pending_option_choice[2]
                    ),
                    duration_matched=(
                        self.pending_option_choice is not None
                        and chosen[1].durations[0]
                        == self.pending_option_choice[2]
                    ),
                    counterfactual_active=causal_counterfactual_active,
                )
            elif (
                autonomous is not None
                and self.autonomous_grace_remaining <= 0
            ):
                chosen, outcome_spread, autonomous_change = autonomous
                self.autonomous_grace_remaining = self.config.autonomous_grace_decisions
                self.autonomous_intervention_pending = False
                passive_transition = True
                self._emit(
                    "autonomous_dynamics_detected",
                    decision=self.decision_index + 1,
                    selected_action=chosen[1].path[0],
                    selected_duration=chosen[1].durations[0],
                    outcome_spread=outcome_spread,
                    autonomous_change=autonomous_change,
                )
            elif self.autonomous_grace_remaining > 0:
                outcome_spread, spread_duration = self._verified_outcome_spread(
                    selection_verified
                )
                control_returned = (
                    outcome_spread > self.config.action_equivalence_threshold
                    and self._informative_signature(self.frame)
                )
                if control_returned:
                    self.autonomous_grace_remaining = 0
                    self.autonomous_intervention_pending = False
                    chosen = max(
                        selection_verified,
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
                    self._emit(
                        "autonomous_grace_ended",
                        decision=self.decision_index + 1,
                        reason="action_dependent_outcomes",
                        outcome_spread=outcome_spread,
                        action_frames=spread_duration,
                    )
                else:
                    neutral = [
                        item
                        for item in selection_verified
                        if item[1].path[0] == Action.NOOP
                    ]
                    if neutral:
                        chosen = max(
                            neutral,
                            key=lambda item: (item[1].durations[0], item[0]),
                        )
                        passive_transition = True
                        grace_continuation = True
                        self.autonomous_grace_remaining -= 1
                        if self.autonomous_grace_remaining == 0:
                            self.autonomous_intervention_pending = True
                        self._emit(
                            "autonomous_grace_wait",
                            decision=self.decision_index + 1,
                            selected_duration=chosen[1].durations[0],
                            grace_remaining=self.autonomous_grace_remaining,
                            maximum_outcome_spread=outcome_spread,
                            endpoint_informative=self._informative_signature(
                                self.frame
                            ),
                        )
                    else:
                        self.autonomous_grace_remaining = 0
                        self.autonomous_intervention_pending = False
                        chosen = max(
                            selection_verified,
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
                    selection_verified,
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
            committed_goal_analysis = branch_goal_analyses[id(state)]
            (
                committed_goal_source_signature,
                committed_goal_target_signature,
            ) = branch_goal_signatures[id(state)]
            (
                committed_goal_source_world_context,
                committed_goal_target_world_context,
                committed_goal_world_effect_signature,
            ) = branch_goal_world_contexts[id(state)]
            self._advance_temporal_option(
                source_signature,
                current_scene,
                passive=passive_transition,
                grace_continuation=grace_continuation,
                passive_action=(plan.path[0] if passive_transition else None),
                passive_duration=(
                    plan.durations[0] if passive_transition else None
                ),
                factual_target=(target if passive_transition else None),
            )
            (
                option_initiation_eligible,
                option_counterfactual_contrast,
                option_counterfactuals,
            ) = self._option_initiation_evidence(chosen, verified)
            chosen_matched_effect = observed_action_effects.get(
                (plan.path[0], plan.durations[0])
            )
            (
                option_initiation_eligible,
                option_counterfactual_contrast,
                option_counterfactuals,
            ) = self._include_matched_effect_option_evidence(
                plan.path[0],
                chosen_matched_effect,
                option_initiation_eligible,
                option_counterfactual_contrast,
                option_counterfactuals,
            )
            delayed_counterfactual_branch = (
                None
                if option_initiation_eligible
                else self._delayed_option_counterfactual(chosen, verified)
            )
            self._supersede_temporal_option_for_intervention(
                plan.path[0],
                option_initiation_eligible
                or delayed_counterfactual_branch is not None,
            )
            self.env.load_state(state)
            self.frame = target
            if self.unlabeled_entity_memory is not None:
                self.unlabeled_entity_memory.observe(target)
            self.current_search_depth = source_search_depth + 1
            if self.goal_prior is not None and committed_goal_analysis is not None:
                committed_goal_analysis = self._commit_goal_prior(
                    committed_goal_analysis, target
                )
                self._record_human_prior_outcome(
                    committed_goal_analysis,
                    source_signature,
                    plan.path[0],
                    plan.durations[0],
                    source_frame,
                    target,
                )
            target_signature = self._signature(target)
            target_visual_cluster = self._abstract_signature(target)
            self.current_frontier_signature = target_frontier_signature
            target_scene = self._scene_signature(target)
            target_signature_is_new = self.novelty.count(target_signature) == 0
            target_scene_is_new = self.scene_visits[target_scene] == 0
            self.novelty.observe(target_signature)
            action = plan.path[0]
            duration = plan.durations[0]
            committed_spatial_effect = observed_action_effects.get(
                (action, duration)
            )
            committed_context = branch_causal_contexts[id(state)]
            committed_causal_spatial_signature = (
                None
                if committed_spatial_effect is None
                else committed_spatial_effect["causal_spatial_signature"]
            )
            if committed_causal_spatial_signature is not None:
                self.causal_spatial_visits[
                    self._causal_frontier_key(
                        source_causal_context_signature,
                        committed_causal_spatial_signature,
                        source_causal_affordance_actions,
                    )
                ] += 1
                self.causal_spatial_cell_visits.update(
                    self._causal_spatial_cells(
                        committed_causal_spatial_signature
                    )
                )
            committed_target_causal_context_signature = (
                committed_context["target"]
            )
            self.current_causal_context_signature = (
                committed_target_causal_context_signature
            )
            self.current_human_prior_world_context_signature = (
                committed_goal_target_world_context
            )
            if committed_context["detected"]:
                self.causal_outcome_contexts.add(
                    committed_target_causal_context_signature
                )
            (
                committed_behavioral_edge_visits_before,
                committed_behavioral_edge_unexpanded,
                committed_behavioral_edge_coverage_bonus,
            ) = self._behavioral_edge_coverage(
                source_signature, action, duration
            )
            self._record_behavioral_edge(
                source_signature, action, duration
            )
            (
                committed_goal_graph_edge_visits_before,
                committed_goal_graph_edge_unexpanded,
            ) = self._human_prior_graph_edge_coverage(
                committed_goal_source_signature, action, duration
            )
            self._record_human_prior_graph_edge(
                committed_goal_source_signature, action, duration
            )
            committed_goal_graph_state_visits_before = (
                0
                if not committed_goal_target_signature
                else self.human_prior_graph_state_visits[
                    committed_goal_target_signature
                ]
            )
            if committed_goal_target_signature:
                self.human_prior_graph_state_visits[
                    committed_goal_target_signature
                ] += 1
            if (
                committed_goal_analysis is not None
                and committed_goal_analysis.target_player_slot is not None
            ):
                self._record_human_prior_player_position(
                    committed_goal_target_signature,
                    committed_goal_analysis.target_player_slot,
                )
            self.action_counts[action] += 1
            self.duration_counts[duration] += 1
            self.action_duration_counts[(action, duration)] += 1
            self.action_streak = (
                self.action_streak + 1
                if action == self.last_action and duration == self.last_duration
                else 1
            )
            self.last_action = action
            self.current_pose_action = self._resulting_pose_action(
                self.current_pose_action, action
            )
            self.last_duration = duration
            self.last_action_was_causal_spatial = (
                committed_causal_spatial_signature is not None
            )
            self.decision_index += 1
            self._observe_persistent_changes(
                target,
                action_dependent=(
                    action != Action.NOOP
                    and committed_spatial_effect is not None
                    and committed_spatial_effect["contrast"]
                    > self.config.action_equivalence_threshold
                ),
            )
            self._observe_dark_transition(target)
            if (
                committed_goal_analysis is not None
                and committed_goal_analysis.navigation_reward != 0.0
            ):
                self.last_navigation_change_decision = self.decision_index
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
            if self.active_temporal_option is None:
                delayed_counterfactual = None
                if delayed_counterfactual_branch is not None:
                    delayed_plan = delayed_counterfactual_branch[1]
                    delayed_state = delayed_counterfactual_branch[2]
                    delayed_frame = self.env.load_state(delayed_state)
                    cloned_state = self.env.save_state()
                    cloned_state_id = self._state_id(cloned_state)
                    try:
                        self.env.load_state(state)
                    except Exception:
                        if release_state is not None:
                            release_state(cloned_state)
                        raise
                    delayed_counterfactual = _OptionCounterfactual(
                        state=cloned_state,
                        frame=delayed_frame,
                        action=delayed_plan.path[0],
                        duration=delayed_plan.durations[0],
                        state_id=cloned_state_id,
                    )
                self.pending_option_choice = (
                    (source_signature, action, duration)
                    if option_initiation_eligible
                    or delayed_counterfactual is not None
                    else None
                )
                self.pending_option_decision = (
                    self.decision_index
                    if self.pending_option_choice is not None
                    else None
                )
                self.pending_option_frame_digest = (
                    source_frame.digest
                    if self.pending_option_choice is not None
                    else None
                )
                positive_goal_milestone = bool(
                    committed_goal_analysis is not None
                    and committed_goal_analysis.milestone_reward > 0.0
                    and not committed_goal_analysis.chest_completed
                )
                if (
                    self.pending_option_choice is not None
                    and option_initiation_eligible
                    and not positive_goal_milestone
                ):
                    self.pending_option_recovery_checkpoint = (
                        _LifeHazardCheckpoint(
                            state=root,
                            frame=source_frame,
                            choice=self.pending_option_choice,
                            decision=self.decision_index,
                            frontier_signature=source_signature,
                            causal_context_signature=(
                                source_causal_context_signature
                            ),
                            scene=current_scene,
                            pose_action=source_pose_action,
                            last_action=source_last_action,
                            last_duration=source_last_duration,
                            action_streak=source_action_streak,
                            goal_heart_slots=source_goal_heart_slots,
                            goal_player_slot=source_goal_player_slot,
                            goal_chest_obtained=(
                                source_goal_chest_obtained
                            ),
                            human_prior_world_context_signature=(
                                source_human_prior_world_context_signature
                            ),
                            state_id=self._state_id(root),
                        )
                    )
                    self._emit(
                        "life_hazard_checkpoint_created",
                        decision=self.decision_index,
                        choice=self.pending_option_choice,
                        state_id=self._state_id(root),
                        source_behavioral_signature=source_signature,
                        **self._frame_fields(source_frame),
                    )
                if (
                    positive_goal_milestone
                    and self.config.human_prior_life_loss_penalty > 0.0
                ):
                    if self.pending_goal_milestone_checkpoint is not None:
                        self._release_life_hazard_checkpoint(
                            self.pending_goal_milestone_checkpoint,
                            "superseded_by_new_goal_milestone",
                        )
                    milestone_choice = (source_signature, action, duration)
                    self.pending_goal_milestone_checkpoint = (
                        _LifeHazardCheckpoint(
                            state=root,
                            frame=source_frame,
                            choice=milestone_choice,
                            decision=self.decision_index,
                            frontier_signature=source_signature,
                            causal_context_signature=(
                                source_causal_context_signature
                            ),
                            scene=current_scene,
                            pose_action=source_pose_action,
                            last_action=source_last_action,
                            last_duration=source_last_duration,
                            action_streak=source_action_streak,
                            goal_heart_slots=source_goal_heart_slots,
                            goal_player_slot=source_goal_player_slot,
                            goal_chest_obtained=(
                                source_goal_chest_obtained
                            ),
                            human_prior_world_context_signature=(
                                source_human_prior_world_context_signature
                            ),
                            kind="goal_milestone",
                            state_id=self._state_id(root),
                        )
                    )
                    self._emit(
                        "goal_milestone_checkpoint_created",
                        decision=self.decision_index,
                        choice=milestone_choice,
                        state_id=self._state_id(root),
                        milestone_reward=(
                            committed_goal_analysis.milestone_reward
                            if committed_goal_analysis is not None
                            else 0.0
                        ),
                        remaining_hearts=(
                            committed_goal_analysis.remaining_hearts
                            if committed_goal_analysis is not None
                            else None
                        ),
                        source_behavioral_signature=source_signature,
                        **self._frame_fields(source_frame),
                    )
                elif (
                    committed_goal_analysis is not None
                    and committed_goal_analysis.chest_completed
                    and self.pending_goal_milestone_checkpoint is not None
                ):
                    self._release_life_hazard_checkpoint(
                        self.pending_goal_milestone_checkpoint,
                        "chest_completed",
                    )
                    self.pending_goal_milestone_checkpoint = None
                self.pending_option_causal_evidence = option_initiation_eligible
                self.pending_option_counterfactual = delayed_counterfactual
                if delayed_counterfactual is not None:
                    self._emit(
                        "temporal_option_counterfactual_armed",
                        decision=self.decision_index,
                        choice=self.pending_option_choice,
                        state_id=delayed_counterfactual.state_id,
                        counterfactual_action=delayed_counterfactual.action,
                        counterfactual_action_frames=(
                            delayed_counterfactual.duration
                        ),
                        immediate_pixel_contrast=target.mean_absolute_difference(
                            delayed_counterfactual.frame
                        ),
                        **self._frame_fields(delayed_counterfactual.frame),
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
            committed_causal_outcome_key = self._causal_outcome_key(
                target, self.current_pose_action
            )
            if (
                committed_context["detected"]
                and not self.causal_outcome_restores[
                    committed_causal_outcome_key
                ]
                and not any(
                    branch.frame.digest == target.digest
                    for branch in self.archive
                )
                and not any(
                    branch.causal_event_outcome
                    and self._causal_outcome_key(
                        branch.frame, branch.pose_action
                    )
                    == committed_causal_outcome_key
                    for branch in self.archive
                )
            ):
                committed_effect = observed_action_effects.get(
                    (action, duration)
                )
                self.archive.append(
                    _ArchivedBranch(
                        state,
                        target,
                        plan,
                        score,
                        target_scene,
                        self.decision_index,
                        source_signature,
                        target_frontier_signature,
                        option_initiation_eligible,
                        option_counterfactual_contrast,
                        option_counterfactuals,
                        committed_causal_spatial_signature or "",
                        (
                            0.0
                            if committed_effect is None
                            else committed_effect["causal_spatial_novelty"]
                        ),
                        (
                            0
                            if committed_effect is None
                            else committed_effect["causal_changed_pixels"]
                        ),
                        (
                            None
                            if committed_effect is None
                            else committed_effect["causal_change_centroid"]
                        ),
                        source_causal_context_signature,
                        committed_target_causal_context_signature,
                        source_causal_affordance_actions,
                        self.current_pose_action,
                        True,
                        (
                            ()
                            if committed_goal_analysis is None
                            else committed_goal_analysis.target_present
                        ),
                        (
                            0.0
                            if committed_goal_analysis is None
                            else committed_goal_analysis.milestone_reward
                        ),
                        (
                            0
                            if committed_goal_analysis is None
                            else committed_goal_analysis.remaining_hearts
                        ),
                        (
                            0
                            if committed_goal_analysis is None
                            else len(committed_goal_analysis.known_slots)
                        ),
                        (
                            None
                            if committed_goal_analysis is None
                            else (
                                committed_goal_analysis.target_chest_slot
                                or committed_goal_analysis.source_chest_slot
                            )
                        ),
                        (
                            None
                            if committed_goal_analysis is None
                            else committed_goal_analysis.target_player_slot
                        ),
                        parent_state_id=self._state_id(root),
                        parent_frame_digest=source_frame.digest,
                        parent_decision=self.decision_index - 1,
                        search_depth=source_search_depth + 1,
                        goal_source_signature=(
                            committed_goal_source_signature
                        ),
                        goal_target_signature=(
                            committed_goal_target_signature
                        ),
                        goal_source_world_context=(
                            committed_goal_source_world_context
                        ),
                        goal_target_world_context=(
                            committed_goal_target_world_context
                        ),
                        goal_world_effect_signature=(
                            committed_goal_world_effect_signature
                        ),
                        goal_chest_obtained=bool(
                            committed_goal_analysis is not None
                            and committed_goal_analysis.chest_obtained
                        ),
                    )
                )
                added += 1
                self._emit(
                    "archive_causal_outcome_added",
                    decision=self.decision_index,
                    state_id=self._state_id(state),
                    parent_state_id=self._state_id(root),
                    parent_frame=source_frame.digest,
                    parent_decision=self.decision_index - 1,
                    search_depth=source_search_depth + 1,
                    action=action,
                    action_frames=duration,
                    causal_context_signature=(
                        source_causal_context_signature
                    ),
                    target_causal_context_signature=(
                        committed_target_causal_context_signature
                    ),
                    causal_spatial_signature=(
                        committed_causal_spatial_signature
                    ),
                    causal_affordance_actions=(
                        source_causal_affordance_actions
                    ),
                    causal_affordance_count=len(
                        source_causal_affordance_actions
                    ),
                    persistent_frontier_value=(
                        self._archive_frontier_score(self.archive[-1])
                    ),
                    behavioral_edge_visits_before=(
                        self._behavioral_edge_coverage(
                            self.archive[-1].origin_signature,
                            self.archive[-1].plan.path[0],
                            self.archive[-1].plan.durations[0],
                        )[0]
                    ),
                    behavioral_edge_unexpanded=(
                        self._behavioral_edge_coverage(
                            self.archive[-1].origin_signature,
                            self.archive[-1].plan.path[0],
                            self.archive[-1].plan.durations[0],
                        )[1]
                    ),
                    behavioral_edge_coverage_bonus=(
                        self._archive_behavioral_edge_coverage_bonus(
                            self.archive[-1]
                        )
                    ),
                    **self._frame_fields(target),
                )
            elif (
                committed_context["detected"]
                and self.causal_outcome_restores[
                    committed_causal_outcome_key
                ]
            ):
                self._emit(
                    "archive_branch_rejected",
                    decision=self.decision_index,
                    reason="causal_outcome_exhausted",
                    action=action,
                    action_frames=duration,
                    previous_restores=self.causal_outcome_restores[
                        committed_causal_outcome_key
                    ],
                    **self._frame_fields(target),
                )
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
                (
                    alternative_option_value,
                    alternative_option_is_known,
                ) = self._temporal_option_estimate(
                    source_signature,
                    alternative_plan.path[0],
                    alternative_plan.durations[0],
                )
                if alternative_option_is_known and alternative_option_value < 0.0:
                    self._emit(
                        "archive_branch_rejected",
                        decision=self.decision_index,
                        reason="learned_hazard",
                        action=alternative_plan.path[0],
                        action_frames=alternative_plan.durations[0],
                        temporal_option_value=alternative_option_value,
                        temporal_option_value_source=(
                            self._temporal_option_estimate_source(
                                source_signature,
                                alternative_plan.path[0],
                                alternative_plan.durations[0],
                            )
                        ),
                        state_id=self._state_id(alternative_state),
                        **self._frame_fields(alternative_frame),
                    )
                    continue
                (
                    alternative_option_eligible,
                    alternative_option_contrast,
                    alternative_option_counterfactuals,
                ) = self._option_initiation_evidence(
                    (
                        alternative_score,
                        alternative_plan,
                        alternative_state,
                        alternative_frame,
                    ),
                    verified,
                )
                alternative_effect = observed_action_effects.get(
                    (
                        alternative_plan.path[0],
                        alternative_plan.durations[0],
                    )
                )
                (
                    alternative_option_eligible,
                    alternative_option_contrast,
                    alternative_option_counterfactuals,
                ) = self._include_matched_effect_option_evidence(
                    alternative_plan.path[0],
                    alternative_effect,
                    alternative_option_eligible,
                    alternative_option_contrast,
                    alternative_option_counterfactuals,
                )
                alternative_context = branch_causal_contexts[
                    id(alternative_state)
                ]
                alternative_goal_analysis = branch_goal_analyses[
                    id(alternative_state)
                ]
                (
                    alternative_goal_source_signature,
                    alternative_goal_target_signature,
                ) = branch_goal_signatures[id(alternative_state)]
                (
                    alternative_goal_source_world_context,
                    alternative_goal_target_world_context,
                    alternative_goal_world_effect_signature,
                ) = branch_goal_world_contexts[id(alternative_state)]
                alternative_semantic_frontier_novel = (
                    self._human_prior_semantic_frontier_novel(
                        alternative_goal_source_signature,
                        alternative_goal_target_signature,
                        alternative_plan.path[0],
                        alternative_plan.durations[0],
                    )
                )
                alternative_pose_action = self._resulting_pose_action(
                    source_pose_action,
                    alternative_plan.path[0],
                )
                alternative_causal_outcome_key = self._causal_outcome_key(
                    alternative_frame, alternative_pose_action
                )
                if (
                    alternative_context["detected"]
                    and not alternative_semantic_frontier_novel
                    and (
                        self.causal_outcome_restores[
                            alternative_causal_outcome_key
                        ]
                        or any(
                            branch.causal_event_outcome
                            and self._causal_outcome_key(
                                branch.frame, branch.pose_action
                            )
                            == alternative_causal_outcome_key
                            for branch in self.archive
                        )
                    )
                ):
                    self._emit(
                        "archive_branch_rejected",
                        decision=self.decision_index,
                        reason="causal_outcome_exhausted",
                        action=alternative_plan.path[0],
                        action_frames=alternative_plan.durations[0],
                        previous_restores=self.causal_outcome_restores[
                            alternative_causal_outcome_key
                        ],
                        state_id=self._state_id(alternative_state),
                        **self._frame_fields(alternative_frame),
                    )
                    continue
                alternative_restore_key = self._affordance_checkpoint_key(
                    alternative_frame,
                    alternative_context["target"],
                    source_causal_affordance_actions,
                    alternative_pose_action,
                )
                previous_restores = self.archive_branch_restores[
                    alternative_restore_key
                ]
                if source_causal_affordance_actions and previous_restores:
                    self._emit(
                        "archive_branch_rejected",
                        decision=self.decision_index,
                        reason="archive_branch_exhausted",
                        action=alternative_plan.path[0],
                        action_frames=alternative_plan.durations[0],
                        causal_context_signature=(
                            source_causal_context_signature
                        ),
                        target_causal_context_signature=(
                            alternative_context["target"]
                        ),
                        causal_affordance_actions=(
                            source_causal_affordance_actions
                        ),
                        causal_affordance_count=len(
                            source_causal_affordance_actions
                        ),
                        previous_restores=previous_restores,
                        state_id=self._state_id(alternative_state),
                        **self._frame_fields(alternative_frame),
                    )
                    continue
                alternative_causal_spatial_signature = (
                    ""
                    if alternative_effect is None
                    else alternative_effect["causal_spatial_signature"] or ""
                )
                alternative_causal_spatial_novelty = (
                    0.0
                    if alternative_effect is None
                    else alternative_effect["causal_spatial_novelty"]
                )
                alternative_causal_changed_pixels = (
                    0
                    if alternative_effect is None
                    else alternative_effect["causal_changed_pixels"]
                )
                alternative_causal_change_centroid = (
                    None
                    if alternative_effect is None
                    else alternative_effect["causal_change_centroid"]
                )
                causal_frontier_already_covered = bool(
                    alternative_causal_spatial_signature
                    and (
                        self.causal_spatial_visits[
                            self._causal_frontier_key(
                                source_causal_context_signature,
                                alternative_causal_spatial_signature,
                                source_causal_affordance_actions,
                            )
                        ]
                        > 0
                        or any(
                            branch.causal_context_signature
                            == source_causal_context_signature
                            and branch.causal_affordance_actions
                            == source_causal_affordance_actions
                            and branch.causal_spatial_signature
                            == alternative_causal_spatial_signature
                            for branch in self.archive
                        )
                    )
                )
                if causal_frontier_already_covered:
                    if (
                        alternative_goal_analysis is not None
                        and alternative_goal_analysis.milestone_reward > 0.0
                    ):
                        causal_frontier_already_covered = False
                    elif alternative_semantic_frontier_novel:
                        causal_frontier_already_covered = False
                if causal_frontier_already_covered:
                    self._emit(
                        "archive_branch_rejected",
                        decision=self.decision_index,
                        reason="causal_frontier_already_covered",
                        action=alternative_plan.path[0],
                        action_frames=alternative_plan.durations[0],
                        causal_spatial_signature=(
                            alternative_causal_spatial_signature
                        ),
                        causal_context_signature=(
                            source_causal_context_signature
                        ),
                        state_id=self._state_id(alternative_state),
                        **self._frame_fields(alternative_frame),
                    )
                    continue
                if (
                    not alternative_causal_spatial_signature
                    and not alternative_option_eligible
                    and not (
                        alternative_goal_analysis is not None
                        and alternative_goal_analysis.milestone_reward > 0.0
                    )
                ):
                    self._emit(
                        "archive_branch_rejected",
                        decision=self.decision_index,
                        reason="no_causal_frontier",
                        action=alternative_plan.path[0],
                        action_frames=alternative_plan.durations[0],
                        state_id=self._state_id(alternative_state),
                        **self._frame_fields(alternative_frame),
                    )
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
                        alternative_option_eligible,
                        alternative_option_contrast,
                        alternative_option_counterfactuals,
                        alternative_causal_spatial_signature,
                        alternative_causal_spatial_novelty,
                        alternative_causal_changed_pixels,
                        alternative_causal_change_centroid,
                        source_causal_context_signature,
                        alternative_context["target"],
                        source_causal_affordance_actions,
                        alternative_pose_action,
                        alternative_context["detected"],
                        (
                            ()
                            if alternative_goal_analysis is None
                            else alternative_goal_analysis.target_present
                        ),
                        (
                            0.0
                            if alternative_goal_analysis is None
                            else alternative_goal_analysis.milestone_reward
                        ),
                        (
                            0
                            if alternative_goal_analysis is None
                            else alternative_goal_analysis.remaining_hearts
                        ),
                        (
                            0
                            if alternative_goal_analysis is None
                            else len(alternative_goal_analysis.known_slots)
                        ),
                        (
                            None
                            if alternative_goal_analysis is None
                            else (
                                alternative_goal_analysis.target_chest_slot
                                or alternative_goal_analysis.source_chest_slot
                            )
                        ),
                        (
                            None
                            if alternative_goal_analysis is None
                            else alternative_goal_analysis.target_player_slot
                        ),
                        parent_state_id=self._state_id(root),
                        parent_frame_digest=source_frame.digest,
                        parent_decision=self.decision_index - 1,
                        search_depth=source_search_depth + 1,
                        goal_source_signature=(
                            alternative_goal_source_signature
                        ),
                        goal_target_signature=(
                            alternative_goal_target_signature
                        ),
                        goal_source_world_context=(
                            alternative_goal_source_world_context
                        ),
                        goal_target_world_context=(
                            alternative_goal_target_world_context
                        ),
                        goal_world_effect_signature=(
                            alternative_goal_world_effect_signature
                        ),
                        goal_chest_obtained=bool(
                            alternative_goal_analysis is not None
                            and alternative_goal_analysis.chest_obtained
                        ),
                    )
                )
                added += 1
                archive_frontier_value = self._archive_frontier_score(self.archive[-1])
                archive_causal_spatial_bonus = self._archive_causal_spatial_bonus(
                    self.archive[-1]
                )
                self._emit(
                    "archive_branch_added",
                    decision=self.decision_index,
                    state_id=self._state_id(alternative_state),
                    parent_state_id=self._state_id(root),
                    parent_frame=source_frame.digest,
                    parent_decision=self.decision_index - 1,
                    search_depth=source_search_depth + 1,
                    action=alternative_plan.path[0],
                    action_frames=alternative_plan.durations[0],
                    path=alternative_plan.path,
                    durations=alternative_plan.durations,
                    score=alternative_score,
                    scene=self._scene_signature(alternative_frame),
                    origin_signature=source_signature,
                    frontier_signature=alternative_frontier_signature,
                    human_prior_semantic_frontier_override=(
                        alternative_semantic_frontier_novel
                    ),
                    human_prior_graph_source_signature=(
                        alternative_goal_source_signature or None
                    ),
                    human_prior_graph_target_signature=(
                        alternative_goal_target_signature or None
                    ),
                    human_prior_world_source_context=(
                        alternative_goal_source_world_context
                    ),
                    human_prior_world_target_context=(
                        alternative_goal_target_world_context
                    ),
                    human_prior_world_effect_signature=(
                        alternative_goal_world_effect_signature or None
                    ),
                    persistent_frontier_value=archive_frontier_value,
                    causal_spatial_archive_bonus=(
                        archive_causal_spatial_bonus
                    ),
                    behavioral_edge_visits_before=(
                        self._behavioral_edge_coverage(
                            source_signature,
                            alternative_plan.path[0],
                            alternative_plan.durations[0],
                        )[0]
                    ),
                    behavioral_edge_unexpanded=(
                        self._behavioral_edge_coverage(
                            source_signature,
                            alternative_plan.path[0],
                            alternative_plan.durations[0],
                        )[1]
                    ),
                    behavioral_edge_coverage_bonus=(
                        self._archive_behavioral_edge_coverage_bonus(
                            self.archive[-1]
                        )
                    ),
                    temporal_option_initiation_eligible=(
                        alternative_option_eligible
                    ),
                    temporal_option_counterfactual_contrast=(
                        alternative_option_contrast
                    ),
                    temporal_option_counterfactuals=(
                        alternative_option_counterfactuals
                    ),
                    causal_spatial_signature=(
                        alternative_causal_spatial_signature or None
                    ),
                    causal_context_signature=source_causal_context_signature,
                    target_causal_context_signature=(
                        alternative_context["target"]
                    ),
                    causal_event_detected=alternative_context["detected"],
                    causal_component_count=alternative_context["components"],
                    causal_event_basis=alternative_context["basis"],
                    causal_event_novel_cells=alternative_context["novel_cells"],
                    causal_affordance_actions=(
                        source_causal_affordance_actions
                    ),
                    causal_affordance_count=len(
                        source_causal_affordance_actions
                    ),
                    transition_spatial_signature=alternative_context[
                        "transition_spatial_signature"
                    ],
                    causal_spatial_novelty=(
                        alternative_causal_spatial_novelty
                    ),
                    causal_changed_pixels=alternative_causal_changed_pixels,
                    causal_change_centroid=(
                        alternative_causal_change_centroid
                    ),
                    **self._human_prior_fields(alternative_goal_analysis),
                    **self._frame_fields(alternative_frame),
                )
            if source_causal_affordance_actions:
                checkpoint_key = self._affordance_checkpoint_key(
                    source_frame,
                    source_causal_context_signature,
                    source_causal_affordance_actions,
                    source_pose_action,
                )
                existing_checkpoint = next(
                    (
                        branch
                        for branch in self.archive
                        if branch.frame.digest == source_frame.digest
                    ),
                    None,
                )
                checkpoint_restores = self.archive_branch_restores[
                    checkpoint_key
                ]
                if checkpoint_restores:
                    self._emit(
                        "archive_affordance_checkpoint_exhausted",
                        decision=self.decision_index,
                        causal_context_signature=(
                            source_causal_context_signature
                        ),
                        causal_affordance_actions=(
                            source_causal_affordance_actions
                        ),
                        causal_affordance_count=len(
                            source_causal_affordance_actions
                        ),
                        previous_restores=checkpoint_restores,
                        **self._frame_fields(source_frame),
                    )
                elif existing_checkpoint is not None:
                    existing_checkpoint.causal_affordance_actions = tuple(
                        sorted(
                            set(existing_checkpoint.causal_affordance_actions)
                            | set(source_causal_affordance_actions),
                            key=lambda action: action.value,
                        )
                    )
                else:
                    affordance_effect = next(
                        effect
                        for (candidate_action, _duration), effect in (
                            observed_action_effects.items()
                        )
                        if candidate_action in source_causal_affordance_actions
                        and effect["causal_spatial_signature"]
                    )
                    checkpoint_plan = NeuralPlan(
                        (Action.NOOP,),
                        (min(self.planner.duration_choices),),
                        max(item[1].score for item in verified),
                        0.0,
                    )
                    affordance_checkpoint_state = (
                        self._clone_state_for_independent_owner(
                            root, state
                        )
                    )
                    self.archive.append(
                        _ArchivedBranch(
                            affordance_checkpoint_state,
                            source_frame,
                            checkpoint_plan,
                            max(item[0] for item in verified),
                            current_scene,
                            self.decision_index,
                            source_signature,
                            source_signature,
                            False,
                            0.0,
                            0,
                            affordance_effect["causal_spatial_signature"],
                            affordance_effect["causal_spatial_novelty"],
                            affordance_effect["causal_changed_pixels"],
                            affordance_effect["causal_change_centroid"],
                            source_causal_context_signature,
                            source_causal_context_signature,
                            source_causal_affordance_actions,
                            source_pose_action,
                            False,
                            (
                                ()
                                if self.goal_prior is None
                                else self.goal_prior.current_slots()
                            ),
                            0.0,
                            (
                                0
                                if self.goal_prior is None
                                else len(self.goal_prior.current_slots())
                            ),
                            (
                                0
                                if self.goal_prior is None
                                else len(self.goal_prior.known_slots)
                            ),
                            (
                                None
                                if self.goal_prior is None
                                else self.goal_prior.detect_open_chest(
                                    source_frame
                                )
                            ),
                            (
                                None
                                if self.goal_prior is None
                                else self.goal_prior.current_player_slot
                            ),
                            parent_state_id=self._state_id(root),
                            parent_frame_digest=source_frame.digest,
                            parent_decision=self.decision_index - 1,
                            search_depth=source_search_depth,
                            goal_source_world_context=(
                                source_human_prior_world_context_signature
                            ),
                            goal_target_world_context=(
                                source_human_prior_world_context_signature
                            ),
                            goal_chest_obtained=(
                                source_goal_chest_obtained
                            ),
                        )
                    )
                    added += 1
                    self._emit(
                        "archive_affordance_checkpoint_added",
                        decision=self.decision_index,
                        state_id=self._state_id(
                            affordance_checkpoint_state
                        ),
                        parent_state_id=self._state_id(root),
                        parent_frame=source_frame.digest,
                        parent_decision=self.decision_index - 1,
                        search_depth=source_search_depth,
                        causal_context_signature=(
                            source_causal_context_signature
                        ),
                        causal_affordance_actions=(
                            source_causal_affordance_actions
                        ),
                        causal_affordance_count=len(
                            source_causal_affordance_actions
                        ),
                        **self._frame_fields(source_frame),
                    )
            archive_state_ids_before_prune = {
                id(branch.state) for branch in self.archive
            }
            self._prune_archive()
            pruned_state_ids.update(
                archive_state_ids_before_prune
                - {id(branch.state) for branch in self.archive}
            )
            committed_effect = observed_action_effects.get((action, duration))
            if committed_effect is None:
                (
                    committed_action_effect_value,
                    committed_action_effect_is_known,
                    committed_action_effect_samples,
                ) = self._action_effect_estimate(source_signature, action)
                committed_action_effect_contrast = None
            else:
                committed_action_effect_contrast = committed_effect["contrast"]
                committed_action_effect_value = committed_effect["value"]
                committed_action_effect_samples = committed_effect["samples"]
                committed_action_effect_is_known = True
            committed_temporal_option_value = self._temporal_option_estimate(
                source_signature, action, duration
            )[0]
            committed_action_effect_bonus = (
                self.config.action_effect_weight * committed_action_effect_value
                if committed_temporal_option_value >= 0.0
                else 0.0
            )
            if committed_effect is None:
                committed_causal_spatial_novelty = 0.0
                committed_causal_changed_pixels = 0
                committed_causal_change_centroid = None
                committed_causal_spatial_visits_before = 0
                committed_causal_cell_coverage = 0.0
                committed_causal_cell_unvisited = 0
                committed_causal_cell_count = 0
            else:
                committed_causal_spatial_novelty = committed_effect[
                    "causal_spatial_novelty"
                ]
                committed_causal_changed_pixels = committed_effect[
                    "causal_changed_pixels"
                ]
                committed_causal_change_centroid = committed_effect[
                    "causal_change_centroid"
                ]
                committed_causal_spatial_visits_before = committed_effect[
                    "causal_spatial_visits"
                ]
                committed_causal_cell_coverage = committed_effect[
                    "causal_cell_coverage"
                ]
                committed_causal_cell_unvisited = committed_effect[
                    "causal_cell_unvisited"
                ]
                committed_causal_cell_count = committed_effect[
                    "causal_cell_count"
                ]
            committed_causal_spatial_bonus = (
                self.config.causal_spatial_novelty_weight
                * committed_causal_spatial_novelty
                if committed_temporal_option_value >= 0.0
                else 0.0
            )
            committed_causal_cell_coverage_bonus = (
                self.config.causal_cell_coverage_weight
                * committed_causal_cell_coverage
                if committed_temporal_option_value >= 0.0
                else 0.0
            )
            if committed_causal_cell_unvisited > 0:
                self.last_causal_cell_progress_decision = (
                    self.decision_index
                )
            committed_action_penalty_components = branch_action_penalties[id(state)]
            self._emit(
                "decision_committed",
                decision=self.decision_index,
                action=plan.path[0],
                action_frames=duration,
                path=plan.path,
                durations=plan.durations,
                target_pose_action=self.current_pose_action,
                score=score,
                model_score=plan.score,
                model_uncertainty=plan.uncertainty,
                spatial_selection_mode=spatial_mode,
                spatial_selection_weight=self.config.spatial_selection_weight,
                spatial_selection_bonus=spatial_bonus(plan),
                spatial_selection_applied_to_commit=False,
                branches_examined=len(verified),
                restored_archive=False,
                committed_state_id=self._state_id(state),
                parent_state_id=self._state_id(root),
                parent_frame=source_frame.digest,
                parent_decision=self.decision_index - 1,
                search_depth=self.current_search_depth,
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
                action_effect_contrast=committed_action_effect_contrast,
                action_effect_value=committed_action_effect_value,
                action_effect_is_known=committed_action_effect_is_known,
                action_effect_samples=committed_action_effect_samples,
                action_effect_bonus=committed_action_effect_bonus,
                **committed_action_penalty_components,
                causal_spatial_signature=committed_causal_spatial_signature,
                causal_context_signature=source_causal_context_signature,
                target_causal_context_signature=(
                    committed_target_causal_context_signature
                ),
                causal_event_detected=(
                    committed_context["detected"]
                ),
                causal_component_count=committed_context["components"],
                causal_event_basis=committed_context["basis"],
                causal_event_novel_cells=committed_context["novel_cells"],
                causal_affordance_actions=source_causal_affordance_actions,
                causal_affordance_count=len(
                    source_causal_affordance_actions
                ),
                transition_spatial_signature=committed_context[
                    "transition_spatial_signature"
                ],
                causal_spatial_novelty=committed_causal_spatial_novelty,
                causal_spatial_visits_before=(
                    committed_causal_spatial_visits_before
                ),
                causal_changed_pixels=committed_causal_changed_pixels,
                causal_change_centroid=committed_causal_change_centroid,
                causal_spatial_bonus=committed_causal_spatial_bonus,
                causal_cell_coverage=committed_causal_cell_coverage,
                causal_cell_unvisited=committed_causal_cell_unvisited,
                causal_cell_count=committed_causal_cell_count,
                causal_cell_coverage_bonus=(
                    committed_causal_cell_coverage_bonus
                ),
                causal_cell_recovery_grace_decisions=(
                    self.config.causal_cell_recovery_grace_decisions
                ),
                last_causal_cell_progress_decision=(
                    self.last_causal_cell_progress_decision
                ),
                behavioral_edge_visits_before=(
                    committed_behavioral_edge_visits_before
                ),
                behavioral_edge_unexpanded=(
                    committed_behavioral_edge_unexpanded
                ),
                behavioral_edge_coverage_bonus=(
                    committed_behavioral_edge_coverage_bonus
                    if committed_temporal_option_value >= 0.0
                    else 0.0
                ),
                behavioral_best_first_archive_enabled=(
                    self.config.behavioral_best_first_archive
                ),
                temporal_option_value=committed_temporal_option_value,
                temporal_option_is_known=self._temporal_option_estimate(
                    source_signature, action, duration
                )[1],
                temporal_option_value_source=(
                    self._temporal_option_estimate_source(
                        source_signature, action, duration
                    )
                ),
                active_temporal_option=(
                    self.active_temporal_option is not None
                ),
                temporal_option_initiation_eligible=(
                    option_initiation_eligible
                ),
                temporal_option_counterfactual_contrast=(
                    option_counterfactual_contrast
                ),
                temporal_option_counterfactuals=option_counterfactuals,
                temporal_option_delayed_counterfactual_armed=(
                    self.pending_option_counterfactual is not None
                ),
                human_prior_graph_source_signature=(
                    committed_goal_source_signature or None
                ),
                human_prior_graph_target_signature=(
                    committed_goal_target_signature or None
                ),
                human_prior_world_source_context=(
                    committed_goal_source_world_context
                ),
                human_prior_world_target_context=(
                    committed_goal_target_world_context
                ),
                human_prior_world_effect_signature=(
                    committed_goal_world_effect_signature or None
                ),
                human_prior_graph_edge_visits_before=(
                    committed_goal_graph_edge_visits_before
                ),
                human_prior_graph_edge_unexpanded=(
                    committed_goal_graph_edge_unexpanded
                ),
                human_prior_graph_state_visits_before=(
                    committed_goal_graph_state_visits_before
                ),
                human_prior_graph_stagnation_limit=(
                    self.config.human_prior_graph_stagnation_visits
                ),
                human_prior_best_first_archive_enabled=(
                    self.config.human_prior_best_first_archive
                ),
                **self._human_prior_fields(committed_goal_analysis),
                action_counts=self.action_counts,
                duration_counts=self.duration_counts,
                action_duration_counts=self._action_duration_count_rows(),
                scene_streak=self.scene_streak,
                visual_stagnation_streak=self.visual_stagnation_streak,
                **self._persistent_change_fields(),
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
                option_states = set()
                if self.pending_option_counterfactual is not None:
                    option_states.add(id(self.pending_option_counterfactual.state))
                if self.pending_option_recovery_checkpoint is not None:
                    option_states.add(
                        id(self.pending_option_recovery_checkpoint.state)
                    )
                if (
                    self.active_temporal_option is not None
                    and self.active_temporal_option.counterfactual is not None
                ):
                    option_states.add(
                        id(self.active_temporal_option.counterfactual.state)
                    )
                if (
                    self.active_temporal_option is not None
                    and self.active_temporal_option.recovery_checkpoint
                    is not None
                ):
                    option_states.add(
                        id(
                            self.active_temporal_option.recovery_checkpoint.state
                        )
                    )
                if self.pending_life_recovery is not None:
                    option_states.add(id(self.pending_life_recovery.state))
                if self.pending_goal_milestone_checkpoint is not None:
                    option_states.add(
                        id(self.pending_goal_milestone_checkpoint.state)
                    )
                for candidate in states:
                    if (
                        id(candidate) not in archived_states
                        and id(candidate) not in option_states
                        and id(candidate) not in pruned_state_ids
                    ):
                        release_state(candidate)

    def _restore_goal_milestone_after_exhaustion(
        self,
        exhausted_graph_signature: str,
        exhausted_graph_visits: int,
    ) -> Optional[Decision]:
        """Learn from a bounded post-milestone search and restore its source.

        This assisted, opt-in recovery does not assert why the goal state is
        blocked. It requires both semantic graph stagnation and an exhausted
        verified option search, then gives the exact milestone choice the same
        small negative sample used for an unrecoverable temporal option.
        """

        checkpoint = self.pending_goal_milestone_checkpoint
        if checkpoint is None:
            return None
        self.pending_goal_milestone_checkpoint = None
        penalty = -self.config.temporal_option_return_penalty
        learned_value = self._record_temporal_option_sample(
            checkpoint.choice, penalty
        )
        learned_samples = self.temporal_option_samples[checkpoint.choice]
        if self.pending_life_recovery is not None:
            self._release_life_hazard_checkpoint(
                self.pending_life_recovery,
                "superseded_by_goal_milestone_exhaustion",
            )
        self.pending_life_recovery = checkpoint
        self.pending_recovery_cause = "goal_exhaustion"
        self._emit(
            "goal_milestone_exhaustion_learned",
            decision=self.decision_index + 1,
            exhausted_graph_signature=exhausted_graph_signature,
            exhausted_graph_visits=exhausted_graph_visits,
            explored_graph_states=len(self.human_prior_graph_state_visits),
            explored_player_positions=len(
                self.human_prior_player_position_visits
            ),
            exhausted_option_sources=len(
                self.human_prior_option_exhausted_sources
            ),
            milestone_choice=checkpoint.choice,
            milestone_decision=checkpoint.decision,
            learned_hazard_value=learned_value,
            learned_hazard_samples=learned_samples,
            recovery_state_id=checkpoint.state_id,
            agent_visible=True,
            **self._frame_fields(self.frame),
        )
        return self._restore_after_life_loss()

    def _restore_after_life_loss(self) -> Optional[Decision]:
        checkpoint = self.pending_life_recovery
        if checkpoint is None:
            return None
        recovery_cause = self.pending_recovery_cause or "life_loss"
        state_id = checkpoint.state_id
        self.env.load_state(checkpoint.state)
        self.pending_life_recovery = None
        self.pending_recovery_cause = None
        release_state = getattr(self.env, "release_state", None)
        if release_state is not None:
            release_state(checkpoint.state)
        invalidated_archive_branches = []
        rollback_descendants = bool(
            checkpoint.kind == "goal_milestone"
            or recovery_cause == "control_collapse"
        )
        rollback_reason = (
            "control_collapse_rollback_descendant"
            if recovery_cause == "control_collapse"
            else "goal_milestone_rollback_descendant"
        )
        if rollback_descendants:
            invalidated_archive_branches = [
                branch
                for branch in self.archive
                if branch.created > checkpoint.decision
            ]
            invalidated_ids = {
                id(branch) for branch in invalidated_archive_branches
            }
            retained_state_ids = {
                self._state_id(branch.state)
                for branch in self.archive
                if id(branch) not in invalidated_ids
                and self._state_id(branch.state) is not None
            }
            released_state_keys = set()
            for branch in invalidated_archive_branches:
                self.archive.remove(branch)
                branch_state_id = self._state_id(branch.state)
                self._emit(
                    "archive_branch_removed",
                    decision=self.decision_index + 1,
                    reason=rollback_reason,
                    state_id=branch_state_id,
                    created_decision=branch.created,
                    rollback_decision=checkpoint.decision,
                    milestone_decision=(
                        checkpoint.decision
                        if checkpoint.kind == "goal_milestone"
                        else None
                    ),
                    **self._frame_fields(branch.frame),
                )
                release_key = (
                    branch_state_id
                    if branch_state_id is not None
                    else f"object-{id(branch.state)}"
                )
                if (
                    release_state is None
                    or branch_state_id in retained_state_ids
                    or release_key in released_state_keys
                ):
                    continue
                released_state_keys.add(release_key)
                try:
                    release_state(branch.state)
                except Exception as error:
                    self._emit(
                        "archive_branch_release_failed",
                        decision=self.decision_index + 1,
                        reason=rollback_reason,
                        state_id=branch_state_id,
                        created_decision=branch.created,
                        error_type=type(error).__name__,
                        error=str(error),
                    )
            if recovery_cause == "control_collapse":
                self.transition_history = [
                    transition
                    for transition in self.transition_history
                    if transition.decision < checkpoint.decision
                ]
                self.visual_last_visit = {
                    signature: decision
                    for signature, decision in self.visual_last_visit.items()
                    if decision < checkpoint.decision
                }
            else:
                self.transition_history = [
                    transition
                    for transition in self.transition_history
                    if transition.decision <= checkpoint.decision
                ]
        checkpoint_restore_cleanup_reason = (
            "control_collapse_checkpoint_restore"
            if recovery_cause == "control_collapse"
            else (
                "goal_milestone_exhaustion_checkpoint_restore"
                if recovery_cause == "goal_exhaustion"
                else "life_loss_checkpoint_restore"
            )
        )
        self._discard_temporal_option(checkpoint_restore_cleanup_reason)
        self._discard_pending_temporal_option(
            checkpoint_restore_cleanup_reason
        )
        self.frame = checkpoint.frame
        self.current_frontier_signature = checkpoint.frontier_signature
        self.current_causal_context_signature = (
            checkpoint.causal_context_signature
        )
        self.current_human_prior_world_context_signature = (
            checkpoint.human_prior_world_context_signature
        )
        self.current_pose_action = checkpoint.pose_action
        self.last_action = checkpoint.last_action
        self.last_duration = checkpoint.last_duration
        self.action_streak = checkpoint.action_streak
        self.last_action_was_causal_spatial = False
        self.current_scene = checkpoint.scene
        self.scene_visits[checkpoint.scene] += 1
        self.scene_streak = 1
        self.visual_stagnation_streak = 0
        self.delayed_return_recovery = False
        self.delayed_return_loop_start = None
        self.autonomous_grace_remaining = 0
        self.autonomous_intervention_pending = False
        self.causal_observation_intervention_pending = False
        self.anticipated_transition_observations_remaining = 0
        self.last_navigation_change_decision = None
        self.pending_life_hazard_choice = None
        self._restore_goal_prior(
            checkpoint.goal_heart_slots,
            checkpoint.frame,
            checkpoint.goal_player_slot,
            checkpoint.goal_chest_obtained,
        )
        self.novelty.observe(self._signature(checkpoint.frame))
        self.decision_index += 1
        self.visual_last_visit[
            self._signature(checkpoint.frame)
        ] = self.decision_index
        if (
            checkpoint.kind == "goal_milestone"
            or recovery_cause == "control_collapse"
        ):
            self._restart_frontier_trace(
                checkpoint.frontier_signature,
                (
                    "control_collapse_rollback"
                    if recovery_cause == "control_collapse"
                    else (
                        "goal_milestone_exhaustion_rollback"
                        if recovery_cause == "goal_exhaustion"
                        else "goal_milestone_rollback"
                    )
                ),
            )
        learned_value, learned = self._temporal_option_estimate(
            *checkpoint.choice
        )
        restore_reason = (
            "control_collapse_checkpoint"
            if recovery_cause == "control_collapse"
            else (
                "goal_milestone_exhaustion"
                if recovery_cause == "goal_exhaustion"
                else (
                    "life_loss_goal_milestone"
                    if checkpoint.kind == "goal_milestone"
                    else "life_loss_checkpoint"
                )
            )
        )
        goal_analysis = (
            None
            if self.goal_prior is None
            else self.goal_prior.analyze(checkpoint.frame, checkpoint.frame)
        )
        self._emit(
            (
                "control_collapse_state_restored"
                if recovery_cause == "control_collapse"
                else (
                    "goal_milestone_exhaustion_state_restored"
                    if recovery_cause == "goal_exhaustion"
                    else "life_hazard_state_restored"
                )
            ),
            decision=self.decision_index,
            recovery_cause=recovery_cause,
            causal_decision=checkpoint.decision,
            choice=checkpoint.choice,
            checkpoint_kind=checkpoint.kind,
            state_id=state_id,
            learned_hazard_value=learned_value,
            learned_hazard_known=learned,
            invalidated_archive_branches=len(
                invalidated_archive_branches
            ),
            source_behavioral_signature=checkpoint.frontier_signature,
            **self._human_prior_fields(goal_analysis),
            **self._frame_fields(checkpoint.frame),
        )
        self._emit(
            "decision_committed",
            decision=self.decision_index,
            action=checkpoint.choice[1],
            action_frames=checkpoint.choice[2],
            path=(checkpoint.choice[1],),
            durations=(checkpoint.choice[2],),
            target_pose_action=self.current_pose_action,
            score=learned_value,
            branches_examined=0,
            restored_archive=True,
            restore_reason=restore_reason,
            committed_state_id=state_id,
            archive_branches_added=0,
            archive_size=len(self.archive),
            temporal_option_value=learned_value,
            temporal_option_is_known=learned,
            temporal_option_value_source=(
                self._temporal_option_estimate_source(*checkpoint.choice)
            ),
            active_temporal_option=False,
            source_behavioral_signature=checkpoint.frontier_signature,
            target_frontier_signature=checkpoint.frontier_signature,
            action_counts=self.action_counts,
            duration_counts=self.duration_counts,
            action_duration_counts=self._action_duration_count_rows(),
            scene_streak=self.scene_streak,
            visual_stagnation_streak=self.visual_stagnation_streak,
            **self._human_prior_fields(goal_analysis),
            **self._frame_fields(checkpoint.frame),
        )
        return Decision(
            checkpoint.choice[1],
            checkpoint.frame,
            (checkpoint.choice[1],),
            learned_value,
            0,
            restored_archive=True,
            action_frames=checkpoint.choice[2],
            planned_durations=(checkpoint.choice[2],),
        )

    def _restore_known_scene_checkpoint(self) -> Optional[Decision]:
        checkpoint = self.known_scene_recovery_checkpoint
        if checkpoint is None:
            return None
        self.env.load_state(checkpoint.state)
        self._discard_temporal_option("known_scene_return_checkpoint_restore")
        self._discard_pending_temporal_option(
            "known_scene_return_checkpoint_restore"
        )
        self.frame = checkpoint.frame
        self.current_frontier_signature = checkpoint.frontier_signature
        self.current_causal_context_signature = (
            checkpoint.causal_context_signature
        )
        self.current_human_prior_world_context_signature = (
            checkpoint.human_prior_world_context_signature
        )
        self.current_pose_action = checkpoint.pose_action
        self.last_action = checkpoint.last_action
        self.last_duration = checkpoint.last_duration
        self.action_streak = checkpoint.action_streak
        self.last_action_was_causal_spatial = False
        self.current_scene = checkpoint.scene
        self.scene_visits[checkpoint.scene] += 1
        self.scene_streak = 1
        self.visual_stagnation_streak = 0
        self.delayed_return_recovery = False
        self.delayed_return_loop_start = None
        self.known_scene_return_recovery_pending = False
        self.dark_transition_active = False
        self.dark_transition_start_decision = None
        self.autonomous_grace_remaining = 0
        self.autonomous_intervention_pending = False
        self.causal_observation_intervention_pending = False
        self.anticipated_transition_observations_remaining = 0
        self.last_navigation_change_decision = None
        self.pending_life_hazard_choice = None
        self._restore_goal_prior(
            checkpoint.goal_heart_slots,
            checkpoint.frame,
            checkpoint.goal_player_slot,
            checkpoint.goal_chest_obtained,
        )
        self.novelty.observe(self._signature(checkpoint.frame))
        self.decision_index += 1
        self.visual_last_visit = {
            self._signature(checkpoint.frame): self.decision_index
        }
        self.transition_history = []
        self._restart_frontier_trace(
            checkpoint.frontier_signature,
            "known_scene_return_checkpoint_restore",
        )
        self._reset_persistent_change_observation_window(
            "known_scene_return_checkpoint_restore"
        )
        self._emit(
            "known_scene_recovery_checkpoint_restored",
            decision=self.decision_index,
            state_id=checkpoint.state_id,
            remembered_bright_scenes=len(self.bright_scene_memory),
            **self._frame_fields(checkpoint.frame),
        )
        self._emit(
            "decision_committed",
            decision=self.decision_index,
            action=Action.NOOP,
            action_frames=0,
            path=(Action.NOOP,),
            durations=(0,),
            target_pose_action=self.current_pose_action,
            score=0.0,
            branches_examined=0,
            restored_archive=True,
            restore_reason="known_scene_return_checkpoint",
            committed_state_id=checkpoint.state_id,
            archive_branches_added=0,
            archive_size=len(self.archive),
            active_temporal_option=False,
            source_behavioral_signature=checkpoint.frontier_signature,
            target_frontier_signature=checkpoint.frontier_signature,
            action_counts=self.action_counts,
            duration_counts=self.duration_counts,
            action_duration_counts=self._action_duration_count_rows(),
            scene_streak=self.scene_streak,
            visual_stagnation_streak=self.visual_stagnation_streak,
            **self._persistent_change_fields(),
            **self._frame_fields(checkpoint.frame),
        )
        return Decision(
            Action.NOOP,
            checkpoint.frame,
            (Action.NOOP,),
            0.0,
            0,
            restored_archive=True,
            action_frames=0,
            planned_durations=(0,),
        )

    def _restore_if_stagnant(self) -> Optional[Decision]:
        assert self.frame is not None
        current_scene = self._scene_signature(self.frame)
        delayed_return = self.delayed_return_recovery
        known_scene_return = self.known_scene_return_recovery_pending
        human_prior_graph_stagnation = (
            self.human_prior_graph_recovery_pending
        )
        if (
            not delayed_return
            and not known_scene_return
            and not human_prior_graph_stagnation
            and self.visual_stagnation_streak
            < self.config.visual_stagnation_visits
        ):
            return None
        if (
            self.anticipated_transition_observations_remaining > 0
            and not known_scene_return
        ):
            self._emit(
                "anticipated_transition_recovery_suppressed",
                decision=self.decision_index + 1,
                observations_remaining=(
                    self.anticipated_transition_observations_remaining
                ),
                observation_duration=(
                    self.anticipated_transition_observation_duration
                ),
                delayed_return_recovery=delayed_return,
                visual_stagnation_streak=self.visual_stagnation_streak,
                archive_size=len(self.archive),
                **self._frame_fields(self.frame),
            )
            if delayed_return:
                self.delayed_return_recovery = False
                self.delayed_return_loop_start = None
            return None
        if (
            self.causal_observation_intervention_pending
            and not known_scene_return
        ):
            self._emit(
                "causal_observation_recovery_suppressed",
                decision=self.decision_index + 1,
                delayed_return_recovery=delayed_return,
                visual_stagnation_streak=self.visual_stagnation_streak,
                archive_size=len(self.archive),
                **self._frame_fields(self.frame),
            )
            if delayed_return:
                self.delayed_return_recovery = False
                self.delayed_return_loop_start = None
            return None
        active_trace = self.active_temporal_option
        maximum_passive_observations = (
            self.config.autonomous_grace_decisions
            + self.config.visual_stagnation_visits
        )
        passive_window_open = bool(
            active_trace is not None
            and active_trace.passive_decisions
            <= maximum_passive_observations
        )
        autonomous_window_open = bool(
            active_trace is None
            and (
                self.autonomous_grace_remaining > 0
                or self.autonomous_intervention_pending
            )
        )
        if (
            not delayed_return
            and not known_scene_return
            and not human_prior_graph_stagnation
            and (passive_window_open or autonomous_window_open)
        ):
            self._emit(
                "temporal_option_recovery_suppressed",
                decision=self.decision_index + 1,
                passive_decisions=(
                    None
                    if active_trace is None
                    else active_trace.passive_decisions
                ),
                maximum_passive_observations=maximum_passive_observations,
                autonomous_grace_remaining=self.autonomous_grace_remaining,
                autonomous_intervention_pending=(
                    self.autonomous_intervention_pending
                ),
                visual_stagnation_streak=self.visual_stagnation_streak,
                archive_size=len(self.archive),
                **self._frame_fields(self.frame),
            )
            return None
        recovery_reason = (
            "known_scene_return_after_dark_transition"
            if known_scene_return
            else (
                "human_prior_graph_stagnation"
                if human_prior_graph_stagnation
                else (
                    "delayed_visual_return"
                    if delayed_return
                    else "visual_stagnation"
                )
            )
        )
        causal_cell_grace = self.config.causal_cell_recovery_grace_decisions
        if (
            causal_cell_grace > 0
            and not known_scene_return
            and not human_prior_graph_stagnation
            and self.last_causal_cell_progress_decision is not None
            and self.decision_index
            - self.last_causal_cell_progress_decision
            < causal_cell_grace
        ):
            self._emit(
                "causal_cell_recovery_suppressed",
                decision=self.decision_index + 1,
                causal_cell_progress_decision=(
                    self.last_causal_cell_progress_decision
                ),
                decisions_since_causal_cell_progress=(
                    self.decision_index
                    - self.last_causal_cell_progress_decision
                ),
                grace_decisions=causal_cell_grace,
                loop_start=self.delayed_return_loop_start,
                recovery_reason=recovery_reason,
            )
            if delayed_return:
                self.delayed_return_recovery = False
                self.delayed_return_loop_start = None
            return None
        if (
            delayed_return
            and self.last_navigation_change_decision is not None
            and self.decision_index - self.last_navigation_change_decision
            < self.config.human_prior_navigation_recovery_grace
        ):
            self._emit(
                "human_prior_navigation_recovery_suppressed",
                decision=self.decision_index + 1,
                navigation_change_decision=self.last_navigation_change_decision,
                decisions_since_navigation_change=(
                    self.decision_index
                    - self.last_navigation_change_decision
                ),
                grace_decisions=(
                    self.config.human_prior_navigation_recovery_grace
                ),
            )
            self.delayed_return_recovery = False
            self.delayed_return_loop_start = None
            return None
        if delayed_return or known_scene_return or human_prior_graph_stagnation:
            loop_start = self.delayed_return_loop_start or 0
            current_signature = self._signature(self.frame)
            if self.config.behavioral_best_first_archive:
                eligible = [
                    branch
                    for branch in self.archive
                    if self._signature(branch.frame) != current_signature
                ]
                self._emit(
                    "behavioral_best_first_global_archive",
                    decision=self.decision_index + 1,
                    loop_start=loop_start,
                    alternatives_examined=len(eligible),
                )
            else:
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
                if branch.created >= minimum_created
                and (
                    branch.scene != current_scene
                    or (
                        bool(branch.causal_spatial_signature)
                        and branch.frame.digest != self.frame.digest
                    )
                )
            ]
        if known_scene_return and self.dark_transition_start_decision is not None:
            alternatives_before_dark_filter = len(eligible)
            eligible = [
                branch
                for branch in eligible
                if branch.created < self.dark_transition_start_decision
            ]
            self._emit(
                "post_dark_archive_branches_filtered",
                decision=self.decision_index + 1,
                dark_transition_start_decision=(
                    self.dark_transition_start_decision
                ),
                alternatives_before=alternatives_before_dark_filter,
                filtered_branches=(
                    alternatives_before_dark_filter - len(eligible)
                ),
                alternatives_remaining=len(eligible),
            )
        if (
            self.goal_prior is not None
            and self.goal_prior.best_remaining_hearts is not None
        ):
            non_regressive_goal_eligible = [
                branch
                for branch in eligible
                if branch.goal_total_hearts > 0
                and branch.goal_remaining_hearts
                <= self.goal_prior.best_remaining_hearts
            ]
            if non_regressive_goal_eligible:
                removed = len(eligible) - len(non_regressive_goal_eligible)
                eligible = non_regressive_goal_eligible
                if removed:
                    self._emit(
                        "human_prior_regressive_archives_filtered",
                        decision=self.decision_index + 1,
                        best_remaining_hearts=(
                            self.goal_prior.best_remaining_hearts
                        ),
                        filtered_branches=removed,
                        alternatives_remaining=len(eligible),
                    )
        if self.persistent_change_cells:
            persistent_change_eligible = [
                branch
                for branch in eligible
                if self._matches_persistent_changes(branch.frame)
            ]
            if persistent_change_eligible:
                removed = len(eligible) - len(persistent_change_eligible)
                eligible = persistent_change_eligible
                if removed:
                    self._emit(
                        "persistent_change_archives_filtered",
                        decision=self.decision_index + 1,
                        filtered_branches=removed,
                        alternatives_remaining=len(eligible),
                        **self._persistent_change_fields(),
                    )
            else:
                self._emit(
                    "persistent_change_preservation_unavailable",
                    decision=self.decision_index + 1,
                    alternatives_examined=len(eligible),
                    **self._persistent_change_fields(),
                )
        speculative_persistence_applied = False
        if (
            self.config.persistent_change_speculative_recovery
            and self.persistent_change_candidates
        ):
            speculative_persistence_eligible = [
                branch
                for branch in eligible
                if self._matches_persistent_change_candidates(branch.frame)
            ]
            if speculative_persistence_eligible:
                removed = len(eligible) - len(
                    speculative_persistence_eligible
                )
                eligible = speculative_persistence_eligible
                speculative_persistence_applied = True
                self._emit(
                    "persistent_change_candidate_archives_filtered",
                    decision=self.decision_index + 1,
                    filtered_branches=removed,
                    alternatives_remaining=len(eligible),
                    **self._persistent_change_fields(),
                )
            else:
                self._emit(
                    "persistent_change_candidate_preservation_unavailable",
                    decision=self.decision_index + 1,
                    alternatives_examined=len(eligible),
                    **self._persistent_change_fields(),
                )
        behavioral_frontier_candidates = list(eligible)
        global_goal_eligible = [
            branch for branch in eligible if branch.goal_progress_reward > 0.0
        ]
        global_causal_event_eligible = [
            branch for branch in eligible if branch.causal_event_outcome
        ]
        same_context_eligible = [
            branch
            for branch in eligible
            if branch.causal_context_signature
            == self.current_causal_context_signature
        ]
        if global_goal_eligible:
            eligible = global_goal_eligible
        elif global_causal_event_eligible:
            eligible = global_causal_event_eligible
        elif same_context_eligible:
            if (
                self.current_causal_context_signature
                in self.causal_outcome_contexts
            ):
                eligible = same_context_eligible
            else:
                ancestor_affordance_eligible = [
                    branch
                    for branch in eligible
                    if branch not in same_context_eligible
                    and branch.causal_affordance_actions
                ]
                eligible = (
                    same_context_eligible
                    + ancestor_affordance_eligible
                )
        if not eligible:
            if known_scene_return:
                checkpoint_restored = self._restore_known_scene_checkpoint()
                if checkpoint_restored is not None:
                    return checkpoint_restored
            if (
                delayed_return
                or known_scene_return
                or human_prior_graph_stagnation
            ):
                self._emit(
                    (
                        "known_scene_return_recovery_unavailable"
                        if known_scene_return
                        else (
                            "human_prior_graph_recovery_unavailable"
                            if human_prior_graph_stagnation
                            else "delayed_return_recovery_unavailable"
                        )
                    ),
                    decision=self.decision_index,
                    loop_start=self.delayed_return_loop_start,
                    archive_size=len(self.archive),
                    **self._frame_fields(self.frame),
                )
                self.delayed_return_recovery = False
                self.delayed_return_loop_start = None
                self.known_scene_return_recovery_pending = False
                self.human_prior_graph_recovery_pending = False
            return None
        safe_eligible = []
        hazardous_eligible = []
        for candidate in eligible:
            if candidate.goal_progress_reward > 0.0:
                safe_eligible.append(candidate)
                continue
            value, known = self._temporal_option_estimate(
                candidate.origin_signature,
                candidate.plan.path[0],
                candidate.plan.durations[0],
            )
            if known and value < 0.0:
                hazardous_eligible.append(
                    {
                        "state_id": self._state_id(candidate.state),
                        "action": candidate.plan.path[0],
                        "action_frames": candidate.plan.durations[0],
                        "temporal_option_value": value,
                        "temporal_option_value_source": (
                            self._temporal_option_estimate_source(
                                candidate.origin_signature,
                                candidate.plan.path[0],
                                candidate.plan.durations[0],
                            )
                        ),
                    }
                )
            else:
                safe_eligible.append(candidate)
        if hazardous_eligible:
            eligible = safe_eligible
            self._emit(
                "learned_hazards_filtered",
                decision=self.decision_index + 1,
                phase="archive_restore",
                filtered=hazardous_eligible,
                alternatives_remaining=len(eligible),
            )
        behavioral_best_first_applied = False
        behavioral_frontier_safe = []
        for candidate in behavioral_frontier_candidates:
            if candidate.goal_progress_reward > 0.0:
                behavioral_frontier_safe.append(candidate)
                continue
            value, known = self._temporal_option_estimate(
                candidate.origin_signature,
                candidate.plan.path[0],
                candidate.plan.durations[0],
            )
            if not known or value >= 0.0:
                behavioral_frontier_safe.append(candidate)
        if behavioral_frontier_safe or hazardous_eligible:
            behavioral_frontier_candidates = behavioral_frontier_safe
        if not eligible:
            if known_scene_return:
                checkpoint_restored = self._restore_known_scene_checkpoint()
                if checkpoint_restored is not None:
                    return checkpoint_restored
            self._emit(
                "archive_recovery_exhausted_by_learned_hazards",
                decision=self.decision_index + 1,
                recovery_reason=recovery_reason,
                filtered=len(hazardous_eligible),
                archive_size=len(self.archive),
                **self._frame_fields(self.frame),
            )
            if delayed_return:
                self.delayed_return_recovery = False
                self.delayed_return_loop_start = None
            if known_scene_return:
                self.known_scene_return_recovery_pending = False
            if human_prior_graph_stagnation:
                self.human_prior_graph_recovery_pending = False
            return None
        human_prior_best_first_applied = False
        human_prior_intervention_eligible = [
            candidate
            for candidate in behavioral_frontier_candidates
            if candidate.plan.path[0] != Action.NOOP
            and candidate.goal_source_signature
        ]
        if (
            self.config.human_prior_best_first_archive
            and human_prior_intervention_eligible
        ):
            human_prior_unexpanded_eligible = [
                candidate
                for candidate in human_prior_intervention_eligible
                if self._human_prior_archive_edge_coverage(candidate)[1]
            ]
            if human_prior_unexpanded_eligible:
                alternatives_before = len(behavioral_frontier_candidates)
                physical_frontier_eligible = [
                    candidate
                    for candidate in human_prior_unexpanded_eligible
                    if candidate.goal_player_slot is not None
                    and not (
                        candidate.human_prior_verified_option
                        and candidate.goal_world_effect_signature
                    )
                    and self._human_prior_position_visits(
                        candidate.goal_target_signature,
                        candidate.goal_player_slot,
                    ) == 0
                ]
                option_effect_frontier_eligible = [
                    candidate
                    for candidate in human_prior_unexpanded_eligible
                    if candidate.human_prior_verified_option
                    and bool(candidate.goal_world_effect_signature)
                    and bool(candidate.goal_target_signature)
                    and self.human_prior_graph_state_visits[
                        candidate.goal_target_signature
                    ]
                    == 0
                ]
                eligible = (
                    physical_frontier_eligible
                    or option_effect_frontier_eligible
                    or human_prior_unexpanded_eligible
                )
                human_prior_best_first_applied = True
                self._emit(
                    "human_prior_best_first_archives_filtered",
                    decision=self.decision_index + 1,
                    alternatives_before=alternatives_before,
                    filtered_branches=(
                        alternatives_before - len(eligible)
                    ),
                    alternatives_remaining=len(eligible),
                    unexpanded_goal_edges=len(eligible),
                    physical_frontier_preferred=bool(
                        physical_frontier_eligible
                    ),
                    option_effect_frontier_preferred=bool(
                        option_effect_frontier_eligible
                        and not physical_frontier_eligible
                    ),
                    confirmed_option_effects=len(
                        option_effect_frontier_eligible
                    ),
                    unvisited_player_positions=len(
                        physical_frontier_eligible
                    ),
                    recovery_reason=recovery_reason,
                )
            else:
                self._emit(
                    "human_prior_best_first_frontier_exhausted",
                    decision=self.decision_index + 1,
                    intervention_alternatives=len(
                        human_prior_intervention_eligible
                    ),
                    recovery_reason=recovery_reason,
                )
        behavioral_intervention_eligible = [
            candidate
            for candidate in behavioral_frontier_candidates
            if candidate.plan.path[0] != Action.NOOP
            and candidate.origin_signature
        ]
        if (
            not human_prior_best_first_applied
            and
            self.config.behavioral_best_first_archive
            and behavioral_intervention_eligible
        ):
            behavioral_unexpanded_eligible = [
                candidate
                for candidate in behavioral_intervention_eligible
                if self._behavioral_edge_coverage(
                    candidate.origin_signature,
                    candidate.plan.path[0],
                    candidate.plan.durations[0],
                )[1]
            ]
            if behavioral_unexpanded_eligible:
                alternatives_before = len(behavioral_frontier_candidates)
                eligible = behavioral_unexpanded_eligible
                behavioral_best_first_applied = True
                self._emit(
                    "behavioral_best_first_archives_filtered",
                    decision=self.decision_index + 1,
                    alternatives_before=alternatives_before,
                    filtered_branches=(
                        alternatives_before - len(eligible)
                    ),
                    alternatives_remaining=len(eligible),
                    unexpanded_behavioral_edges=len(eligible),
                    recovery_reason=recovery_reason,
                )
            else:
                self._emit(
                    "behavioral_best_first_frontier_exhausted",
                    decision=self.decision_index + 1,
                    intervention_alternatives=len(
                        behavioral_intervention_eligible
                    ),
                    recovery_reason=recovery_reason,
                )
        goal_eligible = [
            candidate
            for candidate in eligible
            if candidate.goal_progress_reward > 0.0
        ]
        if goal_eligible:
            eligible = goal_eligible
            affordance_breadth_first = False
            causal_event_eligible = []
            restore_key = lambda item: (
                item.goal_progress_reward,
                self._archive_frontier_score(item),
                item.score,
            )
        else:
            causal_event_eligible = (
                []
                if behavioral_best_first_applied
                else [
                    candidate
                    for candidate in eligible
                    if candidate.causal_event_outcome
                ]
            )
        causal_event_outcome_preferred = bool(causal_event_eligible)
        semantic_navigation_enabled = bool(
            self.goal_prior is not None
            and self.goal_prior.navigation_reward > 0.0
        )
        if goal_eligible:
            pass
        elif human_prior_best_first_applied:
            affordance_breadth_first = False
            restore_key = lambda item: (
                self._archive_frontier_score(item),
                -item.created,
                self.novelty.score(self._signature(item.frame)),
                item.score,
            )
        elif behavioral_best_first_applied:
            affordance_breadth_first = False
            restore_key = lambda item: (
                -item.created,
                self._archive_frontier_score(item),
                self.novelty.score(self._signature(item.frame)),
                item.score,
            )
        elif causal_event_eligible:
            eligible = causal_event_eligible
            affordance_breadth_first = False
            if semantic_navigation_enabled:
                restore_key = lambda item: (
                    self._archive_frontier_score(item),
                    -item.created,
                    self.novelty.score(self._signature(item.frame)),
                    item.score,
                )
            else:
                restore_key = lambda item: (
                    -item.created,
                    self._archive_frontier_score(item),
                    self.novelty.score(self._signature(item.frame)),
                    item.score,
                )
        else:
            affordance_eligible = [
                candidate
                for candidate in eligible
                if candidate.causal_affordance_actions
            ]
            affordance_breadth_first = bool(affordance_eligible)
            if affordance_eligible:
                eligible = affordance_eligible
        if human_prior_best_first_applied or behavioral_best_first_applied:
            pass
        elif not goal_eligible and not causal_event_eligible and affordance_breadth_first:
            eligible = affordance_eligible
            if semantic_navigation_enabled:
                restore_key = lambda item: (
                    self._archive_frontier_score(item),
                    -item.created,
                    self.novelty.score(self._signature(item.frame)),
                    item.score,
                )
            else:
                restore_key = lambda item: (
                    -item.created,
                    self._archive_frontier_score(item),
                    self.novelty.score(self._signature(item.frame)),
                    item.score,
                )
        elif not goal_eligible and not causal_event_eligible:
            restore_key = lambda item: (
                self._archive_frontier_score(item),
                self.novelty.score(self._signature(item.frame)),
                item.created,
                item.score,
            )
        branch = max(
            eligible,
            key=restore_key,
        )
        causal_context_preferred = branch in same_context_eligible
        self.archive.remove(branch)
        self._discard_temporal_option("archive_restore")
        self._discard_pending_temporal_option("archive_restore")
        restored_state_id = self._state_id(branch.state)
        self.env.load_state(branch.state)
        release_state = getattr(self.env, "release_state", None)
        if release_state is not None:
            release_state(branch.state)
        self.frame = branch.frame
        if self.unlabeled_entity_memory is not None:
            self.unlabeled_entity_memory.observe(branch.frame)
        self.current_search_depth = branch.search_depth
        self.last_navigation_change_decision = None
        self.pending_life_hazard_choice = None
        self.autonomous_intervention_pending = False
        self.causal_observation_intervention_pending = False
        self.anticipated_transition_observations_remaining = 0
        self._restore_goal_prior(
            branch.goal_heart_slots,
            branch.frame,
            branch.goal_player_slot,
            branch.goal_chest_obtained,
        )
        self.novelty.observe(self._signature(branch.frame))
        self.scene_visits[branch.scene] += 1
        self.current_scene = branch.scene
        self.scene_streak = 1
        self.visual_stagnation_streak = 0
        self.autonomous_grace_remaining = 0
        self.decision_index += 1
        self._reset_persistent_change_observation_window(
            "archive_restore",
            preserve_candidates=speculative_persistence_applied,
        )
        self.visual_last_visit[self._signature(branch.frame)] = self.decision_index
        selected_frontier_value = self._archive_frontier_score(branch)
        selected_causal_spatial_archive_bonus = (
            self._archive_causal_spatial_bonus(branch)
        )
        selected_causal_cell_coverage_archive_bonus = (
            self._archive_causal_cell_coverage_bonus(branch)
        )
        (
            restored_behavioral_edge_visits_before,
            restored_behavioral_edge_unexpanded,
            restored_behavioral_edge_coverage_bonus,
        ) = self._behavioral_edge_coverage(
            branch.origin_signature,
            branch.plan.path[0],
            branch.plan.durations[0],
        )
        self._record_behavioral_edge(
            branch.origin_signature,
            branch.plan.path[0],
            branch.plan.durations[0],
        )
        (
            restored_goal_graph_edge_visits_before,
            restored_goal_graph_edge_unexpanded,
        ) = self._human_prior_archive_edge_coverage(
            branch
        )
        self._record_human_prior_archive_edge(branch)
        restored_goal_graph_state_visits_before = (
            0
            if not branch.goal_target_signature
            else self.human_prior_graph_state_visits[
                branch.goal_target_signature
            ]
        )
        if branch.goal_target_signature:
            self.human_prior_graph_state_visits[
                branch.goal_target_signature
            ] += 1
        if branch.goal_player_slot is not None:
            self._record_human_prior_player_position(
                branch.goal_target_signature,
                branch.goal_player_slot,
            )
        restored_visual_cluster = self._abstract_signature(branch.frame)
        restored_frontier_signature = (
            branch.frontier_signature
            or self._fallback_frontier_signature(branch.frame)
        )
        self.current_frontier_signature = restored_frontier_signature
        restored_causal_context_signature = (
            branch.target_causal_context_signature
            or branch.causal_context_signature
            or "causal-context-root"
        )
        if branch.causal_affordance_actions:
            self.archive_branch_restores[
                self._affordance_checkpoint_key(
                    branch.frame,
                    restored_causal_context_signature,
                    branch.causal_affordance_actions,
                    branch.pose_action,
                )
            ] += 1
        if branch.causal_event_outcome:
            self.causal_outcome_restores[
                self._causal_outcome_key(
                    branch.frame, branch.pose_action
                )
            ] += 1
        self.current_pose_action = branch.pose_action
        self.current_causal_context_signature = (
            restored_causal_context_signature
        )
        self.current_human_prior_world_context_signature = (
            branch.goal_target_world_context
        )
        if branch.causal_event_outcome:
            self.causal_outcome_contexts.add(
                restored_causal_context_signature
            )
        self._restart_frontier_trace(
            restored_frontier_signature, recovery_reason
        )
        self.delayed_return_recovery = False
        self.delayed_return_loop_start = None
        self.known_scene_return_recovery_pending = False
        self.human_prior_graph_recovery_pending = False
        self.dark_transition_start_decision = None
        self.pending_option_choice = (
            (
                branch.origin_signature,
                branch.plan.path[0],
                branch.plan.durations[0],
            )
            if branch.option_initiation_eligible
            else None
        )
        self.pending_option_decision = (
            self.decision_index if branch.option_initiation_eligible else None
        )
        self.pending_option_frame_digest = None
        self.pending_option_causal_evidence = branch.option_initiation_eligible
        self.pending_option_counterfactual = None
        restored_choice = (
            branch.origin_signature,
            branch.plan.path[0],
            branch.plan.durations[0],
        )
        restored_option_value, restored_option_known = self._temporal_option_estimate(
            *restored_choice
        )
        (
            restored_action_effect_value,
            restored_action_effect_known,
            restored_action_effect_samples,
        ) = self._action_effect_estimate(
            branch.origin_signature, branch.plan.path[0]
        )
        restored_action_effect_bonus = (
            self.config.action_effect_weight * restored_action_effect_value
            if restored_option_value >= 0.0
            else 0.0
        )
        restored_causal_spatial_visits_before = (
            0
            if not branch.causal_spatial_signature
            else self.causal_spatial_visits[
                self._causal_frontier_key(
                    branch.causal_context_signature,
                    branch.causal_spatial_signature,
                    branch.causal_affordance_actions,
                )
            ]
        )
        restored_causal_spatial_novelty = (
            0.0
            if not branch.causal_spatial_signature
            else 1.0 / math.sqrt(restored_causal_spatial_visits_before + 1)
        )
        (
            restored_causal_cell_coverage,
            restored_causal_cell_unvisited,
            restored_causal_cell_count,
        ) = self._causal_cell_coverage(branch.causal_spatial_signature)
        if restored_causal_cell_unvisited > 0:
            self.last_causal_cell_progress_decision = self.decision_index
        if branch.causal_spatial_signature:
            self.causal_spatial_visits[
                self._causal_frontier_key(
                    branch.causal_context_signature,
                    branch.causal_spatial_signature,
                    branch.causal_affordance_actions,
                )
            ] += 1
            self.causal_spatial_cell_visits.update(
                self._causal_spatial_cells(
                    branch.causal_spatial_signature
                )
            )
        self.last_action = branch.plan.path[0]
        self.last_duration = branch.plan.durations[0]
        self.last_action_was_causal_spatial = bool(
            branch.causal_spatial_signature
        )
        self.action_streak = 1
        restored_causal_spatial_bonus = (
            self.config.causal_spatial_novelty_weight
            * restored_causal_spatial_novelty
            if restored_option_value >= 0.0
            else 0.0
        )
        restored_causal_cell_coverage_bonus = (
            self.config.causal_cell_coverage_weight
            * restored_causal_cell_coverage
            if restored_option_value >= 0.0
            else 0.0
        )
        self._emit(
            "archive_branch_restored",
            decision=self.decision_index,
            reason=recovery_reason,
            causal_context_preferred=causal_context_preferred,
            causal_event_outcome_preferred=(
                causal_event_outcome_preferred
            ),
            affordance_breadth_first=affordance_breadth_first,
            state_id=restored_state_id,
            parent_state_id=branch.parent_state_id,
            parent_frame=branch.parent_frame_digest,
            parent_decision=branch.parent_decision,
            search_depth=branch.search_depth,
            created_decision=branch.created,
            age=self.decision_index - branch.created,
            action=branch.plan.path[0],
            action_frames=branch.plan.durations[0],
            path=branch.plan.path,
            durations=branch.plan.durations,
            target_pose_action=self.current_pose_action,
            score=branch.score,
            persistent_frontier_value=selected_frontier_value,
            causal_spatial_archive_bonus=(
                selected_causal_spatial_archive_bonus
            ),
            causal_cell_coverage_archive_bonus=(
                selected_causal_cell_coverage_archive_bonus
            ),
            behavioral_edge_visits_before=(
                restored_behavioral_edge_visits_before
            ),
            behavioral_edge_unexpanded=(
                restored_behavioral_edge_unexpanded
            ),
            behavioral_edge_coverage_bonus=(
                restored_behavioral_edge_coverage_bonus
            ),
            behavioral_best_first_archive_enabled=(
                self.config.behavioral_best_first_archive
            ),
            behavioral_best_first_applied=behavioral_best_first_applied,
            human_prior_best_first_archive_enabled=(
                self.config.human_prior_best_first_archive
            ),
            human_prior_best_first_applied=(
                human_prior_best_first_applied
            ),
            human_prior_verified_option=(
                branch.human_prior_verified_option
            ),
            human_prior_option_depth=(
                len(branch.plan.path)
                if branch.human_prior_verified_option
                else 0
            ),
            human_prior_option_path_visits_before=(
                restored_goal_graph_edge_visits_before
                if branch.human_prior_verified_option
                else 0
            ),
            human_prior_option_world_effect_signature=(
                branch.human_prior_option_world_effect_signature or None
            ),
            human_prior_option_entity_state_signature=(
                branch.human_prior_option_entity_state_signature or None
            ),
            human_prior_graph_source_signature=(
                branch.goal_source_signature or None
            ),
            human_prior_graph_target_signature=(
                branch.goal_target_signature or None
            ),
            human_prior_world_source_context=(
                branch.goal_source_world_context
            ),
            human_prior_world_target_context=(
                branch.goal_target_world_context
            ),
            human_prior_world_effect_signature=(
                branch.goal_world_effect_signature or None
            ),
            human_prior_graph_edge_visits_before=(
                restored_goal_graph_edge_visits_before
            ),
            human_prior_graph_edge_unexpanded=(
                restored_goal_graph_edge_unexpanded
            ),
            human_prior_graph_state_visits_before=(
                restored_goal_graph_state_visits_before
            ),
            human_prior_graph_stagnation_limit=(
                self.config.human_prior_graph_stagnation_visits
            ),
            speculative_persistence_applied=(
                speculative_persistence_applied
            ),
            action_effect_contrast=None,
            action_effect_value=restored_action_effect_value,
            action_effect_is_known=restored_action_effect_known,
            action_effect_samples=restored_action_effect_samples,
            action_effect_bonus=restored_action_effect_bonus,
            causal_spatial_signature=branch.causal_spatial_signature or None,
            causal_context_signature=branch.causal_context_signature,
            target_causal_context_signature=(
                restored_causal_context_signature
            ),
            causal_event_detected=(
                branch.causal_event_outcome
                or restored_causal_context_signature
                != branch.causal_context_signature
            ),
            causal_affordance_actions=branch.causal_affordance_actions,
            causal_affordance_count=len(branch.causal_affordance_actions),
            causal_spatial_novelty=restored_causal_spatial_novelty,
            causal_spatial_visits_before=(
                restored_causal_spatial_visits_before
            ),
            causal_changed_pixels=branch.causal_changed_pixels,
            causal_change_centroid=branch.causal_change_centroid,
            causal_spatial_bonus=restored_causal_spatial_bonus,
            causal_cell_coverage=restored_causal_cell_coverage,
            causal_cell_unvisited=restored_causal_cell_unvisited,
            causal_cell_count=restored_causal_cell_count,
            causal_cell_coverage_bonus=restored_causal_cell_coverage_bonus,
            causal_cell_recovery_grace_decisions=(
                self.config.causal_cell_recovery_grace_decisions
            ),
            last_causal_cell_progress_decision=(
                self.last_causal_cell_progress_decision
            ),
            human_prior_enabled=self.goal_prior is not None,
            human_prior_reward_track=(
                "human_prior_v2" if self.goal_prior is not None else None
            ),
            human_prior_goal_reward=branch.goal_progress_reward,
            human_prior_target_hearts=branch.goal_heart_slots,
            human_prior_remaining_hearts=branch.goal_remaining_hearts,
            human_prior_total_hearts=branch.goal_total_hearts,
            human_prior_target_chest_slot=branch.goal_chest_slot,
            human_prior_target_player_slot=branch.goal_player_slot,
            human_prior_chest_obtained=branch.goal_chest_obtained,
            human_prior_best_remaining_hearts=(
                None
                if self.goal_prior is None
                else self.goal_prior.best_remaining_hearts
            ),
            temporal_option_value=restored_option_value,
            temporal_option_is_known=restored_option_known,
            temporal_option_value_source=(
                self._temporal_option_estimate_source(*restored_choice)
            ),
            temporal_option_initiation_eligible=(
                branch.option_initiation_eligible
            ),
            temporal_option_counterfactual_contrast=(
                branch.option_counterfactual_contrast
            ),
            temporal_option_counterfactuals=branch.option_counterfactuals,
            abstract_signature=restored_visual_cluster,
            target_frontier_signature=restored_frontier_signature,
            archive_size=len(self.archive),
            **self._persistent_change_fields(),
            **self._frame_fields(branch.frame),
        )
        self._emit(
            "decision_committed",
            decision=self.decision_index,
            action=branch.plan.path[0],
            action_frames=branch.plan.durations[0],
            path=branch.plan.path,
            durations=branch.plan.durations,
            target_pose_action=self.current_pose_action,
            score=branch.score,
            branches_examined=0,
            restored_archive=True,
            restore_reason=recovery_reason,
            behavioral_edge_visits_before=(
                restored_behavioral_edge_visits_before
            ),
            behavioral_edge_unexpanded=(
                restored_behavioral_edge_unexpanded
            ),
            behavioral_edge_coverage_bonus=(
                restored_behavioral_edge_coverage_bonus
            ),
            behavioral_best_first_archive_enabled=(
                self.config.behavioral_best_first_archive
            ),
            behavioral_best_first_applied=behavioral_best_first_applied,
            human_prior_best_first_archive_enabled=(
                self.config.human_prior_best_first_archive
            ),
            human_prior_best_first_applied=(
                human_prior_best_first_applied
            ),
            human_prior_verified_option=(
                branch.human_prior_verified_option
            ),
            human_prior_option_depth=(
                len(branch.plan.path)
                if branch.human_prior_verified_option
                else 0
            ),
            human_prior_option_path_visits_before=(
                restored_goal_graph_edge_visits_before
                if branch.human_prior_verified_option
                else 0
            ),
            human_prior_option_world_effect_signature=(
                branch.human_prior_option_world_effect_signature or None
            ),
            human_prior_option_entity_state_signature=(
                branch.human_prior_option_entity_state_signature or None
            ),
            human_prior_graph_source_signature=(
                branch.goal_source_signature or None
            ),
            human_prior_graph_target_signature=(
                branch.goal_target_signature or None
            ),
            human_prior_world_source_context=(
                branch.goal_source_world_context
            ),
            human_prior_world_target_context=(
                branch.goal_target_world_context
            ),
            human_prior_world_effect_signature=(
                branch.goal_world_effect_signature or None
            ),
            human_prior_graph_edge_visits_before=(
                restored_goal_graph_edge_visits_before
            ),
            human_prior_graph_edge_unexpanded=(
                restored_goal_graph_edge_unexpanded
            ),
            human_prior_graph_state_visits_before=(
                restored_goal_graph_state_visits_before
            ),
            human_prior_graph_stagnation_limit=(
                self.config.human_prior_graph_stagnation_visits
            ),
            speculative_persistence_applied=(
                speculative_persistence_applied
            ),
            causal_event_outcome_preferred=(
                causal_event_outcome_preferred
            ),
            affordance_breadth_first=affordance_breadth_first,
            committed_state_id=restored_state_id,
            parent_state_id=branch.parent_state_id,
            parent_frame=branch.parent_frame_digest,
            parent_decision=branch.parent_decision,
            search_depth=branch.search_depth,
            archive_branches_added=0,
            archive_size=len(self.archive),
            persistent_frontier_value=selected_frontier_value,
            action_effect_contrast=None,
            action_effect_value=restored_action_effect_value,
            action_effect_is_known=restored_action_effect_known,
            action_effect_samples=restored_action_effect_samples,
            action_effect_bonus=restored_action_effect_bonus,
            causal_spatial_signature=branch.causal_spatial_signature or None,
            causal_context_signature=branch.causal_context_signature,
            target_causal_context_signature=(
                restored_causal_context_signature
            ),
            causal_event_detected=(
                branch.causal_event_outcome
                or restored_causal_context_signature
                != branch.causal_context_signature
            ),
            causal_affordance_actions=branch.causal_affordance_actions,
            causal_affordance_count=len(branch.causal_affordance_actions),
            causal_spatial_novelty=restored_causal_spatial_novelty,
            causal_spatial_visits_before=(
                restored_causal_spatial_visits_before
            ),
            causal_changed_pixels=branch.causal_changed_pixels,
            causal_change_centroid=branch.causal_change_centroid,
            causal_spatial_bonus=restored_causal_spatial_bonus,
            causal_cell_coverage=restored_causal_cell_coverage,
            causal_cell_unvisited=restored_causal_cell_unvisited,
            causal_cell_count=restored_causal_cell_count,
            causal_cell_coverage_bonus=restored_causal_cell_coverage_bonus,
            human_prior_enabled=self.goal_prior is not None,
            human_prior_reward_track=(
                "human_prior_v2" if self.goal_prior is not None else None
            ),
            human_prior_goal_reward=branch.goal_progress_reward,
            human_prior_target_hearts=branch.goal_heart_slots,
            human_prior_remaining_hearts=branch.goal_remaining_hearts,
            human_prior_total_hearts=branch.goal_total_hearts,
            human_prior_target_chest_slot=branch.goal_chest_slot,
            human_prior_target_player_slot=branch.goal_player_slot,
            human_prior_chest_obtained=branch.goal_chest_obtained,
            human_prior_best_remaining_hearts=(
                None
                if self.goal_prior is None
                else self.goal_prior.best_remaining_hearts
            ),
            temporal_option_value=restored_option_value,
            temporal_option_is_known=restored_option_known,
            temporal_option_value_source=(
                self._temporal_option_estimate_source(*restored_choice)
            ),
            active_temporal_option=False,
            temporal_option_initiation_eligible=(
                branch.option_initiation_eligible
            ),
            temporal_option_counterfactual_contrast=(
                branch.option_counterfactual_contrast
            ),
            temporal_option_counterfactuals=branch.option_counterfactuals,
            abstract_signature=restored_visual_cluster,
            source_behavioral_signature=branch.origin_signature,
            target_frontier_signature=restored_frontier_signature,
            action_counts=self.action_counts,
            duration_counts=self.duration_counts,
            action_duration_counts=self._action_duration_count_rows(),
            scene_streak=self.scene_streak,
            visual_stagnation_streak=self.visual_stagnation_streak,
            **self._persistent_change_fields(),
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
                    (
                        -item.created
                        if (
                            item.causal_affordance_actions
                            or item.causal_event_outcome
                        )
                        else item.created
                    ),
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
