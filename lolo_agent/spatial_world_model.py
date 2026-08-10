from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ensemble_world_model import VisualSequence, sequence_batch
from .environment import Action
from .neural_world_model import ACTION_ORDER


@dataclass(frozen=True)
class SpatialTrainingMetrics:
    loss: float
    reconstruction: float
    pixel_prediction: float
    token_prediction: float
    effect_prediction: float


@dataclass(frozen=True)
class SpatialValidationReport:
    """Held-out measurements that emphasize action-dependent spatial effects."""

    horizon_pixel_l1: Tuple[float, ...]
    horizon_persistence_pixel_l1: Tuple[float, ...]
    horizon_effect_weighted_pixel_l1: Tuple[float, ...]
    horizon_effect_weighted_persistence_l1: Tuple[float, ...]
    horizon_effect_l1: Tuple[float, ...]
    horizon_zero_effect_l1: Tuple[float, ...]
    horizon_balanced_effect_l1: Tuple[float, ...]
    horizon_zero_balanced_effect_l1: Tuple[float, ...]
    horizon_effect_f1: Tuple[float, ...]
    horizon_effect_prevalence: Tuple[float, ...]
    horizon_uncertainty: Tuple[float, ...]
    horizon_uncertainty_effect_error_correlation: Tuple[float, ...]
    uncertainty_effect_error_correlation: float


def causal_dataset_statistics(sequences: Sequence[VisualSequence]) -> Dict[str, int]:
    """Describe counterfactual coverage without interpreting any visual entity."""

    roots: Dict[Tuple[int, str], set[Tuple[Action, int]]] = {}
    one_step = 0
    for sequence in sequences:
        if len(sequence.actions) != 1:
            continue
        one_step += 1
        duration = sequence.durations[0] if sequence.durations else 4
        roots.setdefault((sequence.group, sequence.frames[0].digest), set()).add(
            (sequence.actions[0], duration)
        )
    counterfactual_roots = sum(len(options) >= 2 for options in roots.values())
    noop_control_roots = sum(
        any(action == Action.NOOP for action, _ in options)
        and any(action != Action.NOOP for action, _ in options)
        for options in roots.values()
    )
    return {
        "sequences": len(sequences),
        "groups": len({sequence.group for sequence in sequences}),
        "one_step_sequences": one_step,
        "multi_step_sequences": len(sequences) - one_step,
        "causal_roots": len(roots),
        "counterfactual_roots": counterfactual_roots,
        "noop_control_roots": noop_control_roots,
    }


class _SpatialDynamicsHead(nn.Module):
    def __init__(self, token_size: int, context_size: int, delta_scale: float) -> None:
        super().__init__()
        self.delta_scale = delta_scale
        hidden_size = token_size * 2
        self.trunk = nn.Sequential(
            nn.Conv2d(token_size + context_size, hidden_size, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.SiLU(),
        )
        self.token_delta = nn.Conv2d(hidden_size, token_size, 1)
        self.effect_logits = nn.Conv2d(hidden_size, 1, 1)

    def forward(self, tokens: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        height, width = tokens.shape[-2:]
        broadcast = context[:, :, None, None].expand(-1, -1, height, width)
        features = self.trunk(torch.cat((tokens, broadcast), dim=1))
        return (
            tokens + self.delta_scale * self.token_delta(features),
            self.effect_logits(features),
        )


class SpatialTokenDynamicsModel(nn.Module):
    """Translation-sharing visual dynamics over an unlabeled spatial token map.

    The model receives only RGB frames, controller actions, and action durations.
    Tokens are learned convolutional features: no tile size, sprite identity, room
    coordinate, object class, reward, or evaluator annotation enters the model.
    Each ensemble head predicts both successor tokens and where pixels will change.
    """

    def __init__(
        self,
        token_size: int = 64,
        action_size: int = 16,
        ensemble_size: int = 3,
        grid_size: int = 8,
        duration_conditioned: bool = True,
        duration_size: int = 8,
        max_action_frames: int = 32,
        fixed_action_frames: int = 4,
        effect_scale: float = 0.05,
        effect_mask_power: float = 4.0,
        token_delta_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if token_size <= 0 or action_size <= 0 or grid_size <= 0:
            raise ValueError("token, action, and grid sizes must be positive")
        if ensemble_size < 2:
            raise ValueError("ensemble size must be at least two")
        if max_action_frames <= 0 or fixed_action_frames <= 0:
            raise ValueError("action frame limits must be positive")
        if effect_scale <= 0:
            raise ValueError("effect scale must be positive")
        if effect_mask_power <= 0:
            raise ValueError("effect mask power must be positive")
        if token_delta_scale <= 0:
            raise ValueError("token delta scale must be positive")
        self.token_size = token_size
        self.action_size = action_size
        self.ensemble_size = ensemble_size
        self.grid_size = grid_size
        self.duration_conditioned = duration_conditioned
        self.duration_size = duration_size
        self.max_action_frames = max_action_frames
        self.fixed_action_frames = fixed_action_frames
        self.effect_scale = effect_scale
        self.effect_mask_power = effect_mask_power
        self.token_delta_scale = token_delta_scale
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, token_size, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(token_size, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.action_embedding = nn.Embedding(len(ACTION_ORDER), action_size)
        self.duration_embedding = (
            nn.Embedding(max_action_frames + 1, duration_size)
            if duration_conditioned
            else None
        )
        context_size = action_size + (duration_size if duration_conditioned else 0)
        self.dynamics_heads = nn.ModuleList(
            [
                _SpatialDynamicsHead(token_size, context_size, token_delta_scale)
                for _ in range(ensemble_size)
            ]
        )

    def encode(self, frames: Tensor) -> Tensor:
        return self.encoder(frames)

    def decode(self, tokens: Tensor) -> Tensor:
        pixels = self.decoder(tokens)
        if pixels.shape[-2:] != (128, 128):
            pixels = F.interpolate(
                pixels, size=(128, 128), mode="bilinear", align_corners=False
            )
        return pixels

    def initial_ensemble(self, frames: Tensor) -> Tensor:
        tokens = self.encode(frames)
        return tokens.unsqueeze(0).expand(self.ensemble_size, -1, -1, -1, -1).contiguous()

    def render_successor(
        self, previous_pixels: Tensor, tokens: Tensor, effect_logits: Tensor
    ) -> Tensor:
        """Copy stable pixels and render only the learned action-dependent region."""

        if effect_logits.ndim != 3:
            raise ValueError("effect logits must have shape [batch, row, column]")
        candidate = self.decode(tokens)
        effect_mask = F.interpolate(
            effect_logits.sigmoid().unsqueeze(1),
            size=previous_pixels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).pow(self.effect_mask_power)
        return previous_pixels * (1.0 - effect_mask) + candidate * effect_mask

    def _context(self, actions: Tensor, durations: Optional[Tensor]) -> Tensor:
        context = self.action_embedding(actions)
        if self.duration_conditioned:
            if durations is None:
                raise ValueError("duration-conditioned dynamics require durations")
            if torch.any(durations <= 0) or torch.any(durations > self.max_action_frames):
                raise ValueError(
                    f"durations must be between 1 and {self.max_action_frames} frames"
                )
            assert self.duration_embedding is not None
            context = torch.cat((context, self.duration_embedding(durations)), dim=1)
        return context

    def transition_ensemble(
        self, tokens: Tensor, actions: Tensor, durations: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if tokens.ndim != 5 or tokens.shape[0] != self.ensemble_size:
            raise ValueError("tokens must have shape [ensemble, batch, token, row, column]")
        context = self._context(actions, durations)
        predictions = []
        effect_logits = []
        for index, head in enumerate(self.dynamics_heads):
            predicted, effect = head(tokens[index], context)
            predictions.append(predicted)
            effect_logits.append(effect.squeeze(1))
        return torch.stack(predictions), torch.stack(effect_logits)

    def rollout(
        self, source: Tensor, actions: Tensor, durations: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if actions.ndim != 2:
            raise ValueError("actions must have shape [batch, horizon]")
        if durations is not None and durations.shape != actions.shape:
            raise ValueError("durations must align with actions")
        tokens = self.initial_ensemble(source)
        pixels = []
        means = []
        uncertainty = []
        effects = []
        current_pixels = source
        for step in range(actions.shape[1]):
            step_durations = None if durations is None else durations[:, step]
            tokens, effect_logits = self.transition_ensemble(
                tokens, actions[:, step], step_durations
            )
            mean = tokens.mean(dim=0)
            mean_effect_logits = effect_logits.mean(dim=0)
            current_pixels = self.render_successor(
                current_pixels, mean, mean_effect_logits
            )
            pixels.append(current_pixels)
            means.append(mean)
            normalized_tokens = F.normalize(tokens, dim=2)
            effect_probabilities = effect_logits.sigmoid()
            uncertainty.append(
                normalized_tokens.var(dim=0, unbiased=False).mean(dim=(1, 2, 3))
                + effect_probabilities.var(dim=0, unbiased=False).mean(dim=(1, 2))
            )
            effects.append(effect_probabilities.mean(dim=0))
        return (
            torch.stack(pixels, dim=1),
            torch.stack(means, dim=1),
            torch.stack(uncertainty, dim=1),
            torch.stack(effects, dim=1),
        )

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        self.train()
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    @property
    def checkpoint_digest(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode())
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


def spatial_effect_target(
    source: Tensor, target: Tensor, grid_size: int, effect_scale: float
) -> Tensor:
    """Derive soft, unlabeled spatial effects directly from pixel differences."""

    if source.shape != target.shape or source.ndim != 4:
        raise ValueError("source and target must have matching [batch, channel, row, column]")
    magnitude = (target - source).abs().mean(dim=1, keepdim=True)
    pooled = F.adaptive_avg_pool2d(magnitude, (grid_size, grid_size)).squeeze(1)
    return (pooled / effect_scale).clamp(0.0, 1.0)


def _masked_mean(values: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return values.mean()
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    denominator = expanded.sum().clamp_min(1.0)
    return (values * expanded).sum() / denominator


def spatial_sequence_loss(
    model: SpatialTokenDynamicsModel,
    frames: Tensor,
    actions: Tensor,
    durations: Optional[Tensor] = None,
    discount: float = 0.9,
    bootstrap_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, SpatialTrainingMetrics]:
    source = frames[:, 0]
    source_tokens = model.encode(source)
    reconstruction_loss = F.l1_loss(model.decode(source_tokens), source)
    tokens = source_tokens.unsqueeze(0).expand(model.ensemble_size, -1, -1, -1, -1).contiguous()
    pixel_loss = source.new_zeros(())
    token_loss = source.new_zeros(())
    effect_loss = source.new_zeros(())
    weight_total = 0.0
    previous_target = source
    previous_prediction = source
    for step in range(actions.shape[1]):
        step_durations = None if durations is None else durations[:, step]
        tokens, effect_logits = model.transition_ensemble(
            tokens, actions[:, step], step_durations
        )
        mean_tokens = tokens.mean(dim=0)
        predicted_target = model.render_successor(
            previous_prediction, mean_tokens, effect_logits.mean(dim=0)
        )
        target = frames[:, step + 1]
        with torch.no_grad():
            target_tokens = model.encode(target)
            target_effect = spatial_effect_target(
                previous_target, target, model.grid_size, model.effect_scale
            )
        pixel_weights = 1.0 + 4.0 * F.interpolate(
            target_effect.unsqueeze(1),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        step_pixel = (
            (predicted_target - target).abs() * pixel_weights
        ).sum() / (pixel_weights.sum() * target.shape[1])
        expanded_target_tokens = target_tokens.unsqueeze(0).expand_as(tokens)
        normalized_predictions = F.normalize(tokens, dim=2)
        normalized_targets = F.normalize(expanded_target_tokens, dim=2)
        token_errors = (
            1.0 - (normalized_predictions * normalized_targets).sum(dim=2)
        ).clamp_min(0.0).mean(dim=(2, 3))
        expanded_effect = target_effect.unsqueeze(0).expand_as(effect_logits)
        effect_errors = F.binary_cross_entropy_with_logits(
            effect_logits, expanded_effect, reduction="none"
        )
        effect_weights = 1.0 + 4.0 * expanded_effect
        step_token = _masked_mean(token_errors, bootstrap_mask)
        step_effect = _masked_mean(effect_errors * effect_weights, bootstrap_mask)
        weight = discount**step
        pixel_loss = pixel_loss + weight * step_pixel
        token_loss = token_loss + weight * step_token
        effect_loss = effect_loss + weight * step_effect
        weight_total += weight
        previous_target = target
        previous_prediction = predicted_target
    pixel_loss = pixel_loss / weight_total
    token_loss = token_loss / weight_total
    effect_loss = effect_loss / weight_total
    loss = 0.1 * reconstruction_loss + 0.5 * pixel_loss + 0.5 * token_loss + 0.5 * effect_loss
    return loss, SpatialTrainingMetrics(
        loss=float(loss.detach().cpu()),
        reconstruction=float(reconstruction_loss.detach().cpu()),
        pixel_prediction=float(pixel_loss.detach().cpu()),
        token_prediction=float(token_loss.detach().cpu()),
        effect_prediction=float(effect_loss.detach().cpu()),
    )


def train_spatial_model(
    model: SpatialTokenDynamicsModel,
    sequences: Sequence[VisualSequence],
    device: Union[torch.device, str],
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> List[SpatialTrainingMetrics]:
    if not sequences:
        raise ValueError("at least one sequence is required")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    model.to(device)
    model.unfreeze()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: List[SpatialTrainingMetrics] = []
    for _ in range(epochs):
        batches = []
        for horizon in sorted({len(sequence.actions) for sequence in sequences}):
            bucket = [item for item in sequences if len(item.actions) == horizon]
            order = torch.randperm(len(bucket), generator=generator).tolist()
            batches.extend(
                [bucket[index] for index in order[start : start + batch_size]]
                for start in range(0, len(order), batch_size)
            )
        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        for batch_index in batch_order:
            batch = batches[batch_index]
            frames, actions, durations = sequence_batch(batch, device)
            bootstrap_mask = (
                torch.rand(
                    (model.ensemble_size, len(batch)),
                    generator=generator,
                    device="cpu",
                )
                >= 0.25
            )
            empty_columns = ~bootstrap_mask.any(dim=0)
            bootstrap_mask[0, empty_columns] = True
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = spatial_sequence_loss(
                model,
                frames,
                actions,
                durations if model.duration_conditioned else None,
                bootstrap_mask=bootstrap_mask.to(device=device, dtype=frames.dtype),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            history.append(metrics)
    return history


@torch.no_grad()
def validate_spatial_model(
    model: SpatialTokenDynamicsModel,
    sequences: Sequence[VisualSequence],
    device: Union[torch.device, str],
    batch_size: int = 8,
    effect_threshold: float = 0.2,
) -> SpatialValidationReport:
    if not sequences:
        raise ValueError("at least one validation sequence is required")
    model.to(device)
    model.eval()
    maximum_horizon = max(len(sequence.actions) for sequence in sequences)
    pixel_error_sums = [0.0] * maximum_horizon
    persistence_error_sums = [0.0] * maximum_horizon
    weighted_pixel_error_sums = [0.0] * maximum_horizon
    weighted_persistence_error_sums = [0.0] * maximum_horizon
    effect_error_sums = [0.0] * maximum_horizon
    zero_effect_error_sums = [0.0] * maximum_horizon
    effect_prevalence_sums = [0.0] * maximum_horizon
    positive_effect_error_sums = [0.0] * maximum_horizon
    negative_effect_error_sums = [0.0] * maximum_horizon
    positive_zero_error_sums = [0.0] * maximum_horizon
    negative_zero_error_sums = [0.0] * maximum_horizon
    positive_effect_counts = [0] * maximum_horizon
    negative_effect_counts = [0] * maximum_horizon
    uncertainty_sums = [0.0] * maximum_horizon
    counts = [0] * maximum_horizon
    true_positive = [0] * maximum_horizon
    false_positive = [0] * maximum_horizon
    false_negative = [0] * maximum_horizon
    pairs: List[Tuple[float, float]] = []
    horizon_pairs: List[List[Tuple[float, float]]] = [
        [] for _ in range(maximum_horizon)
    ]
    for horizon in sorted({len(sequence.actions) for sequence in sequences}):
        bucket = [item for item in sequences if len(item.actions) == horizon]
        for start in range(0, len(bucket), batch_size):
            batch = bucket[start : start + batch_size]
            frames, actions, durations = sequence_batch(batch, device)
            predictions, _tokens, uncertainty, predicted_effects = model.rollout(
                frames[:, 0],
                actions,
                durations if model.duration_conditioned else None,
            )
            pixel_errors = (predictions - frames[:, 1:]).abs().mean(dim=(2, 3, 4))
            for step in range(horizon):
                persistence_errors = (frames[:, 0] - frames[:, step + 1]).abs().mean(
                    dim=(1, 2, 3)
                )
                actual_effect = spatial_effect_target(
                    frames[:, step],
                    frames[:, step + 1],
                    model.grid_size,
                    model.effect_scale,
                )
                pixel_weights = 1.0 + 4.0 * F.interpolate(
                    actual_effect.unsqueeze(1),
                    size=frames.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                weighted_denominator = pixel_weights.sum(dim=(1, 2, 3)) * frames.shape[2]
                weighted_pixel_errors = (
                    (predictions[:, step] - frames[:, step + 1]).abs() * pixel_weights
                ).sum(dim=(1, 2, 3)) / weighted_denominator
                weighted_persistence_errors = (
                    (frames[:, 0] - frames[:, step + 1]).abs() * pixel_weights
                ).sum(dim=(1, 2, 3)) / weighted_denominator
                effect_errors = (predicted_effects[:, step] - actual_effect).abs().mean(
                    dim=(1, 2)
                )
                predicted_active = predicted_effects[:, step] >= effect_threshold
                actual_active = actual_effect >= effect_threshold
                absolute_effect_error = (predicted_effects[:, step] - actual_effect).abs()
                positive_effect_error_sums[step] += float(
                    absolute_effect_error[actual_active].sum().cpu()
                )
                negative_effect_error_sums[step] += float(
                    absolute_effect_error[~actual_active].sum().cpu()
                )
                positive_zero_error_sums[step] += float(
                    actual_effect[actual_active].sum().cpu()
                )
                negative_zero_error_sums[step] += float(
                    actual_effect[~actual_active].sum().cpu()
                )
                positive_effect_counts[step] += int(actual_active.sum().cpu())
                negative_effect_counts[step] += int((~actual_active).sum().cpu())
                true_positive[step] += int((predicted_active & actual_active).sum().cpu())
                false_positive[step] += int((predicted_active & ~actual_active).sum().cpu())
                false_negative[step] += int((~predicted_active & actual_active).sum().cpu())
                step_pixels = pixel_errors[:, step].detach().cpu().tolist()
                step_persistence = persistence_errors.detach().cpu().tolist()
                step_weighted_pixels = weighted_pixel_errors.detach().cpu().tolist()
                step_weighted_persistence = (
                    weighted_persistence_errors.detach().cpu().tolist()
                )
                step_effects = effect_errors.detach().cpu().tolist()
                step_zero_effect = actual_effect.mean(dim=(1, 2)).detach().cpu().tolist()
                step_prevalence = actual_active.float().mean(dim=(1, 2)).detach().cpu().tolist()
                step_uncertainty = uncertainty[:, step].detach().cpu().tolist()
                pixel_error_sums[step] += sum(step_pixels)
                persistence_error_sums[step] += sum(step_persistence)
                weighted_pixel_error_sums[step] += sum(step_weighted_pixels)
                weighted_persistence_error_sums[step] += sum(step_weighted_persistence)
                effect_error_sums[step] += sum(step_effects)
                zero_effect_error_sums[step] += sum(step_zero_effect)
                effect_prevalence_sums[step] += sum(step_prevalence)
                uncertainty_sums[step] += sum(step_uncertainty)
                counts[step] += len(batch)
                pairs.extend(zip(step_uncertainty, step_effects))
                horizon_pairs[step].extend(zip(step_uncertainty, step_effects))
    effect_f1 = []
    for tp, fp, fn in zip(true_positive, false_positive, false_negative):
        denominator = 2 * tp + fp + fn
        effect_f1.append((2 * tp / denominator) if denominator else 1.0)

    def balanced_errors(
        positive_sums: Sequence[float], negative_sums: Sequence[float]
    ) -> Tuple[float, ...]:
        values = []
        for positive, negative, positive_count, negative_count in zip(
            positive_sums,
            negative_sums,
            positive_effect_counts,
            negative_effect_counts,
        ):
            components = []
            if positive_count:
                components.append(positive / positive_count)
            if negative_count:
                components.append(negative / negative_count)
            values.append(sum(components) / len(components))
        return tuple(values)

    return SpatialValidationReport(
        horizon_pixel_l1=tuple(value / count for value, count in zip(pixel_error_sums, counts)),
        horizon_persistence_pixel_l1=tuple(
            value / count for value, count in zip(persistence_error_sums, counts)
        ),
        horizon_effect_weighted_pixel_l1=tuple(
            value / count for value, count in zip(weighted_pixel_error_sums, counts)
        ),
        horizon_effect_weighted_persistence_l1=tuple(
            value / count
            for value, count in zip(weighted_persistence_error_sums, counts)
        ),
        horizon_effect_l1=tuple(value / count for value, count in zip(effect_error_sums, counts)),
        horizon_zero_effect_l1=tuple(
            value / count for value, count in zip(zero_effect_error_sums, counts)
        ),
        horizon_balanced_effect_l1=balanced_errors(
            positive_effect_error_sums, negative_effect_error_sums
        ),
        horizon_zero_balanced_effect_l1=balanced_errors(
            positive_zero_error_sums, negative_zero_error_sums
        ),
        horizon_effect_f1=tuple(effect_f1),
        horizon_effect_prevalence=tuple(
            value / count for value, count in zip(effect_prevalence_sums, counts)
        ),
        horizon_uncertainty=tuple(value / count for value, count in zip(uncertainty_sums, counts)),
        horizon_uncertainty_effect_error_correlation=tuple(
            _pearson(step_pairs) for step_pairs in horizon_pairs
        ),
        uncertainty_effect_error_correlation=_pearson(pairs),
    )


def _pearson(pairs: Sequence[Tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return 0.0
    mean_x = sum(value[0] for value in pairs) / len(pairs)
    mean_y = sum(value[1] for value in pairs) / len(pairs)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    denominator_y = sum((y - mean_y) ** 2 for _, y in pairs)
    denominator = math.sqrt(denominator_x * denominator_y)
    return numerator / denominator if denominator > 0 else 0.0


def save_spatial_checkpoint(
    model: SpatialTokenDynamicsModel, path: Path, planning_horizon: int
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = model.checkpoint_digest
    torch.save(
        {
            "version": 5,
            "architecture": "unlabeled-spatial-token-dynamics",
            "model": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "token_size": model.token_size,
            "action_size": model.action_size,
            "ensemble_size": model.ensemble_size,
            "grid_size": model.grid_size,
            "duration_conditioned": model.duration_conditioned,
            "duration_size": model.duration_size,
            "max_action_frames": model.max_action_frames,
            "fixed_action_frames": model.fixed_action_frames,
            "effect_scale": model.effect_scale,
            "effect_mask_power": model.effect_mask_power,
            "token_delta_scale": model.token_delta_scale,
            "token_objective": "cosine",
            "planning_horizon": planning_horizon,
            "persistent_inputs": ["pixels", "actions", "action_durations"],
            "excluded_inputs": [
                "RAM",
                "object_labels",
                "rewards",
                "level_annotations",
                "solutions",
            ],
            "actions": [action.value for action in ACTION_ORDER],
            "digest": digest,
        },
        path,
    )
    return digest


def load_spatial_checkpoint(
    path: Path,
    device: Union[torch.device, str] = "cpu",
    frozen: bool = True,
) -> Tuple[SpatialTokenDynamicsModel, int]:
    checkpoint: Dict[str, object] = torch.load(
        Path(path), map_location="cpu", weights_only=True
    )
    if checkpoint.get("version") not in (1, 2, 3, 4, 5):
        raise ValueError("unsupported spatial checkpoint version")
    if checkpoint.get("architecture") != "unlabeled-spatial-token-dynamics":
        raise ValueError("spatial checkpoint architecture does not match runtime")
    if checkpoint.get("actions") != [action.value for action in ACTION_ORDER]:
        raise ValueError("checkpoint controller action order does not match runtime")
    model = SpatialTokenDynamicsModel(
        token_size=int(checkpoint["token_size"]),
        action_size=int(checkpoint["action_size"]),
        ensemble_size=int(checkpoint["ensemble_size"]),
        grid_size=int(checkpoint["grid_size"]),
        duration_conditioned=bool(checkpoint["duration_conditioned"]),
        duration_size=int(checkpoint["duration_size"]),
        max_action_frames=int(checkpoint["max_action_frames"]),
        fixed_action_frames=int(checkpoint["fixed_action_frames"]),
        effect_scale=float(checkpoint["effect_scale"]),
        effect_mask_power=float(checkpoint.get("effect_mask_power", 4.0)),
        token_delta_scale=float(checkpoint.get("token_delta_scale", 1.0)),
    )
    model.load_state_dict(checkpoint["model"])  # type: ignore[arg-type]
    if model.checkpoint_digest != checkpoint.get("digest"):
        raise ValueError("spatial checkpoint parameter digest mismatch")
    model.to(device)
    if frozen:
        model.freeze()
    return model, int(checkpoint["planning_horizon"])
