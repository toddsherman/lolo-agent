from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from .environment import Action
from .neural_world_model import ACTION_TO_INDEX, frame_tensor
from .pixels import Frame
from .spatial_world_model import SpatialTokenDynamicsModel, spatial_effect_target


class SpatialShadowEvaluator:
    """Score frozen spatial predictions without participating in action choice."""

    def __init__(
        self,
        model: SpatialTokenDynamicsModel,
        device: Union[torch.device, str],
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.freeze()

    @property
    def checkpoint_digest(self) -> str:
        return self.model.checkpoint_digest

    @torch.no_grad()
    def score_plans(
        self,
        frame: Frame,
        plans: Sequence[Tuple[Sequence[Action], Sequence[int]]],
    ) -> List[Dict[str, float]]:
        if not plans:
            return []
        horizon = len(plans[0][0])
        if horizon <= 0 or any(len(actions) != horizon for actions, _ in plans):
            raise ValueError("shadow plans must have one shared positive horizon")
        if any(len(actions) != len(durations) for actions, durations in plans):
            raise ValueError("shadow actions and durations must align")
        source = frame_tensor(frame, self.device).unsqueeze(0)
        source_batch = source.expand(len(plans), -1, -1, -1).contiguous()
        actions = torch.tensor(
            [
                [ACTION_TO_INDEX[action] for action in plan_actions]
                for plan_actions, _ in plans
            ],
            dtype=torch.long,
            device=self.device,
        )
        durations = torch.tensor(
            [list(plan_durations) for _, plan_durations in plans],
            dtype=torch.long,
            device=self.device,
        )
        predicted, _tokens, uncertainty, effects = self.model.rollout(
            source_batch,
            actions,
            durations if self.model.duration_conditioned else None,
        )
        horizon_weights = torch.tensor(
            [0.9**step for step in range(horizon)],
            dtype=predicted.dtype,
            device=self.device,
        )
        effect_mass = (
            effects.mean(dim=(2, 3)) * horizon_weights.unsqueeze(0)
        ).sum(dim=1)
        uncertainty_mass = (uncertainty * horizon_weights.unsqueeze(0)).sum(dim=1)
        final_change = (predicted[:, -1] - source_batch).abs().mean(dim=(1, 2, 3))
        shadow_score = effect_mass + uncertainty_mass
        return [
            {
                "spatial_shadow_score": float(shadow_score[index].cpu()),
                "spatial_shadow_predicted_effect": float(effect_mass[index].cpu()),
                "spatial_shadow_predicted_change": float(final_change[index].cpu()),
                "spatial_shadow_uncertainty": float(uncertainty_mass[index].cpu()),
            }
            for index in range(len(plans))
        ]

    @torch.no_grad()
    def evaluate_transition(
        self,
        source_frame: Frame,
        action: Action,
        duration: int,
        target_frame: Frame,
        effect_threshold: float = 0.2,
    ) -> Dict[str, Union[bool, float]]:
        source = frame_tensor(source_frame, self.device).unsqueeze(0)
        target = frame_tensor(target_frame, self.device).unsqueeze(0)
        actions = torch.tensor(
            [[ACTION_TO_INDEX[action]]], dtype=torch.long, device=self.device
        )
        durations = torch.tensor([[duration]], dtype=torch.long, device=self.device)
        predicted, _tokens, uncertainty, effects = self.model.rollout(
            source,
            actions,
            durations if self.model.duration_conditioned else None,
        )
        predicted_frame = predicted[:, 0]
        predicted_effect = effects[:, 0]
        actual_effect = spatial_effect_target(
            source,
            target,
            self.model.grid_size,
            self.model.effect_scale,
        )
        pixel_weights = 1.0 + 4.0 * F.interpolate(
            actual_effect.unsqueeze(1),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        denominator = pixel_weights.sum() * target.shape[1]
        weighted_error = float(
            (((predicted_frame - target).abs() * pixel_weights).sum() / denominator)
            .cpu()
        )
        weighted_persistence = float(
            (((source - target).abs() * pixel_weights).sum() / denominator).cpu()
        )
        predicted_active = predicted_effect >= effect_threshold
        actual_active = actual_effect >= effect_threshold
        true_positive = int((predicted_active & actual_active).sum().cpu())
        false_positive = int((predicted_active & ~actual_active).sum().cpu())
        false_negative = int((~predicted_active & actual_active).sum().cpu())
        f1_denominator = 2 * true_positive + false_positive + false_negative
        return {
            "spatial_shadow_pixel_l1": float(
                (predicted_frame - target).abs().mean().cpu()
            ),
            "spatial_shadow_persistence_l1": float((source - target).abs().mean().cpu()),
            "spatial_shadow_effect_weighted_pixel_l1": weighted_error,
            "spatial_shadow_effect_weighted_persistence_l1": weighted_persistence,
            "spatial_shadow_beats_persistence": weighted_error < weighted_persistence,
            "spatial_shadow_effect_l1": float(
                (predicted_effect - actual_effect).abs().mean().cpu()
            ),
            "spatial_shadow_effect_f1": (
                2.0 * true_positive / f1_denominator
                if f1_denominator
                else 1.0
            ),
            "spatial_shadow_predicted_effect": float(predicted_effect.mean().cpu()),
            "spatial_shadow_actual_effect": float(actual_effect.mean().cpu()),
            "spatial_shadow_uncertainty": float(uncertainty[0, 0].cpu()),
        }
