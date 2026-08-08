from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .environment import Action, PixelSaveStateEnv
from .neural_world_model import ACTION_ORDER, ACTION_TO_INDEX, frame_tensor
from .pixels import Frame


@dataclass(frozen=True)
class VisualSequence:
    group: int
    frames: Tuple[Frame, ...]
    actions: Tuple[Action, ...]
    durations: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.actions or len(self.frames) != len(self.actions) + 1:
            raise ValueError("a visual sequence requires one more frame than actions")
        if self.durations and len(self.durations) != len(self.actions):
            raise ValueError("sequence durations must align with actions")
        if any(duration <= 0 for duration in self.durations):
            raise ValueError("action durations must be positive")


@dataclass(frozen=True)
class EnsembleTrainingMetrics:
    loss: float
    reconstruction: float
    pixel_prediction: float
    latent_prediction: float


@dataclass(frozen=True)
class ValidationReport:
    horizon_pixel_l1: Tuple[float, ...]
    horizon_uncertainty: Tuple[float, ...]
    uncertainty_error_correlation: float


class EnsembleVisualDynamicsModel(nn.Module):
    """Shared visual representation with independently initialized dynamics heads."""

    def __init__(
        self,
        latent_size: int = 256,
        action_size: int = 32,
        ensemble_size: int = 3,
        duration_conditioned: bool = False,
        duration_size: int = 16,
        max_action_frames: int = 32,
        fixed_action_frames: int = 4,
    ) -> None:
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("ensemble size must be at least two")
        self.latent_size = latent_size
        self.action_size = action_size
        self.ensemble_size = ensemble_size
        self.duration_conditioned = duration_conditioned
        self.duration_size = duration_size
        self.max_action_frames = max_action_frames
        self.fixed_action_frames = fixed_action_frames
        self.fixed_action_frames_locked = False
        self.encoder_convolution = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.encoder_projection = nn.Linear(128 * 4 * 4, latent_size)
        self.action_embedding = nn.Embedding(len(ACTION_ORDER), action_size)
        self.duration_embedding = (
            nn.Embedding(max_action_frames + 1, duration_size)
            if duration_conditioned
            else None
        )
        dynamics_input_size = latent_size + action_size + (
            duration_size if duration_conditioned else 0
        )
        self.dynamics_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dynamics_input_size, latent_size * 2),
                    nn.SiLU(),
                    nn.Linear(latent_size * 2, latent_size),
                )
                for _ in range(ensemble_size)
            ]
        )
        self.decoder_projection = nn.Linear(latent_size, 128 * 8 * 8)
        self.decoder_convolution = nn.Sequential(
            nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, frames: Tensor) -> Tensor:
        return self.encoder_projection(self.encoder_convolution(frames).flatten(1))

    def decode(self, latent: Tensor) -> Tensor:
        features = self.decoder_projection(latent).reshape(-1, 128, 8, 8)
        return self.decoder_convolution(features)

    def initial_ensemble(self, frames: Tensor) -> Tensor:
        latent = self.encode(frames)
        return latent.unsqueeze(0).expand(self.ensemble_size, -1, -1).contiguous()

    def transition_ensemble(
        self, latents: Tensor, actions: Tensor, durations: Optional[Tensor] = None
    ) -> Tensor:
        if latents.ndim != 3 or latents.shape[0] != self.ensemble_size:
            raise ValueError("latents must have shape [ensemble, batch, latent]")
        embedded = self.action_embedding(actions)
        if self.duration_conditioned:
            if durations is None:
                raise ValueError("duration-conditioned dynamics require durations")
            if torch.any(durations <= 0) or torch.any(durations > self.max_action_frames):
                raise ValueError(
                    f"durations must be between 1 and {self.max_action_frames} frames"
                )
            assert self.duration_embedding is not None
            embedded = torch.cat((embedded, self.duration_embedding(durations)), dim=1)
        predictions = []
        for index, head in enumerate(self.dynamics_heads):
            latent = latents[index]
            delta = head(torch.cat((latent, embedded), dim=1))
            predictions.append(latent + delta)
        return torch.stack(predictions)

    def rollout(
        self, source: Tensor, actions: Tensor, durations: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if actions.ndim != 2:
            raise ValueError("actions must have shape [batch, horizon]")
        latents = self.initial_ensemble(source)
        pixels = []
        means = []
        uncertainty = []
        for step in range(actions.shape[1]):
            step_durations = None if durations is None else durations[:, step]
            latents = self.transition_ensemble(latents, actions[:, step], step_durations)
            mean = latents.mean(dim=0)
            pixels.append(self.decode(mean))
            means.append(mean)
            uncertainty.append(latents.var(dim=0, unbiased=False).mean(dim=1))
        return (
            torch.stack(pixels, dim=1),
            torch.stack(means, dim=1),
            torch.stack(uncertainty, dim=1),
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


def sequence_batch(
    sequences: Sequence[VisualSequence], device: Union[torch.device, str]
) -> Tuple[Tensor, Tensor, Tensor]:
    horizon = len(sequences[0].actions)
    if any(len(sequence.actions) != horizon for sequence in sequences):
        raise ValueError("all sequences in a batch must have the same horizon")
    frames = torch.stack(
        [torch.stack([frame_tensor(frame) for frame in sequence.frames]) for sequence in sequences]
    ).to(device)
    actions = torch.tensor(
        [[ACTION_TO_INDEX[action] for action in sequence.actions] for sequence in sequences],
        dtype=torch.long,
        device=device,
    )
    durations = torch.tensor(
        [
            list(sequence.durations) if sequence.durations else [4] * len(sequence.actions)
            for sequence in sequences
        ],
        dtype=torch.long,
        device=device,
    )
    return frames, actions, durations


def ensemble_sequence_loss(
    model: EnsembleVisualDynamicsModel,
    frames: Tensor,
    actions: Tensor,
    durations: Optional[Tensor] = None,
    discount: float = 0.9,
    bootstrap_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, EnsembleTrainingMetrics]:
    source = frames[:, 0]
    source_latent = model.encode(source)
    reconstruction_loss = F.l1_loss(model.decode(source_latent), source)
    latents = source_latent.unsqueeze(0).expand(model.ensemble_size, -1, -1).contiguous()
    pixel_loss = source.new_zeros(())
    latent_loss = source.new_zeros(())
    weight_total = 0.0
    for step in range(actions.shape[1]):
        step_durations = None if durations is None else durations[:, step]
        latents = model.transition_ensemble(latents, actions[:, step], step_durations)
        mean = latents.mean(dim=0)
        target = frames[:, step + 1]
        with torch.no_grad():
            target_latent = model.encode(target)
        weight = discount**step
        pixel_loss = pixel_loss + weight * F.l1_loss(model.decode(mean), target)
        expanded_target = target_latent.unsqueeze(0).expand_as(latents).contiguous()
        latent_errors = F.smooth_l1_loss(
            latents, expanded_target, reduction="none"
        ).mean(dim=2)
        if bootstrap_mask is None:
            step_latent_loss = latent_errors.mean()
        else:
            step_latent_loss = (latent_errors * bootstrap_mask).sum() / bootstrap_mask.sum()
        latent_loss = latent_loss + weight * step_latent_loss
        weight_total += weight
    pixel_loss = pixel_loss / weight_total
    latent_loss = latent_loss / weight_total
    loss = 0.25 * reconstruction_loss + pixel_loss + 0.5 * latent_loss
    return loss, EnsembleTrainingMetrics(
        loss=float(loss.detach().cpu()),
        reconstruction=float(reconstruction_loss.detach().cpu()),
        pixel_prediction=float(pixel_loss.detach().cpu()),
        latent_prediction=float(latent_loss.detach().cpu()),
    )


def train_ensemble_model(
    model: EnsembleVisualDynamicsModel,
    sequences: Sequence[VisualSequence],
    device: Union[torch.device, str],
    epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> List[EnsembleTrainingMetrics]:
    if not sequences:
        raise ValueError("at least one sequence is required")
    model.to(device)
    model.unfreeze()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    for _ in range(epochs):
        order = torch.randperm(len(sequences), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            batch = [sequences[index] for index in order[start : start + batch_size]]
            frames, actions, durations = sequence_batch(batch, device)
            bootstrap_mask = (
                torch.rand(
                    (model.ensemble_size, len(batch)),
                    generator=generator,
                    device="cpu",
                )
                >= 0.25
            )
            # Every transition must supervise at least one head.
            empty_columns = ~bootstrap_mask.any(dim=0)
            bootstrap_mask[0, empty_columns] = True
            bootstrap_mask = bootstrap_mask.to(device=device, dtype=frames.dtype)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = ensemble_sequence_loss(
                model,
                frames,
                actions,
                durations if model.duration_conditioned else None,
                bootstrap_mask=bootstrap_mask,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            history.append(metrics)
    return history


@torch.no_grad()
def validate_ensemble_model(
    model: EnsembleVisualDynamicsModel,
    sequences: Sequence[VisualSequence],
    device: Union[torch.device, str],
    batch_size: int = 4,
) -> ValidationReport:
    if not sequences:
        raise ValueError("at least one validation sequence is required")
    model.to(device)
    model.eval()
    horizon = len(sequences[0].actions)
    error_sums = [0.0] * horizon
    uncertainty_sums = [0.0] * horizon
    pairs: List[Tuple[float, float]] = []
    count = 0
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        frames, actions, durations = sequence_batch(batch, device)
        predictions, _latents, uncertainty = model.rollout(
            frames[:, 0], actions, durations if model.duration_conditioned else None
        )
        errors = (predictions - frames[:, 1:]).abs().mean(dim=(2, 3, 4))
        for step in range(horizon):
            step_errors = errors[:, step].detach().cpu().tolist()
            step_uncertainty = uncertainty[:, step].detach().cpu().tolist()
            error_sums[step] += sum(step_errors)
            uncertainty_sums[step] += sum(step_uncertainty)
            pairs.extend(zip(step_uncertainty, step_errors))
        count += len(batch)
    return ValidationReport(
        horizon_pixel_l1=tuple(value / count for value in error_sums),
        horizon_uncertainty=tuple(value / count for value in uncertainty_sums),
        uncertainty_error_correlation=_pearson(pairs),
    )


def _pearson(pairs: Sequence[Tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return 0.0
    mean_x = sum(pair[0] for pair in pairs) / len(pairs)
    mean_y = sum(pair[1] for pair in pairs) / len(pairs)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    denominator_y = sum((y - mean_y) ** 2 for _, y in pairs)
    denominator = math.sqrt(denominator_x * denominator_y)
    return numerator / denominator if denominator > 0 else 0.0


def collect_branched_sequences(
    env: PixelSaveStateEnv,
    roots: int = 20,
    branches_per_root: int = 2,
    horizon: int = 3,
    action_frames: int = 4,
    action_durations: Optional[Sequence[int]] = None,
    seed: int = 0,
    reset_env: bool = True,
    group_offset: int = 0,
    event_logger: Optional[Any] = None,
) -> List[VisualSequence]:
    if min(roots, branches_per_root, horizon, action_frames) <= 0:
        raise ValueError("collector sizes must be positive")
    randomizer = random.Random(seed)
    duration_choices = tuple(action_durations or (action_frames,))
    if any(duration <= 0 for duration in duration_choices):
        raise ValueError("action durations must be positive")
    current = env.reset() if reset_env else env.observe()  # type: ignore[attr-defined]
    release_state = getattr(env, "release_state", None)
    sequences = []
    for local_group in range(roots):
        group = group_offset + local_group
        if event_logger is not None:
            event_logger.log("collection_root_started", group=group)
        root = env.save_state()
        branches = []
        for branch_index in range(branches_per_root):
            env.load_state(root)
            frames = [current]
            actions = []
            durations = []
            for _ in range(horizon):
                action = randomizer.choice(ACTION_ORDER)
                duration = randomizer.choice(duration_choices)
                actions.append(action)
                durations.append(duration)
                frames.append(env.step(action, duration))
            child = env.save_state()
            sequence = VisualSequence(group, tuple(frames), tuple(actions), tuple(durations))
            sequences.append(sequence)
            if event_logger is not None:
                event_logger.log(
                    "training_sequence_collected",
                    group=group,
                    branch=branch_index + 1,
                    actions=actions,
                    durations=durations,
                    frames=[frame.digest for frame in frames],
                )
            branches.append((child, frames[-1]))
        chosen, current = randomizer.choice(branches)
        env.load_state(chosen)
        if release_state is not None:
            release_state(root)
            for child, _ in branches:
                release_state(child)
    return sequences


def split_sequence_groups(
    sequences: Sequence[VisualSequence], validation_modulus: int = 5
) -> Tuple[List[VisualSequence], List[VisualSequence]]:
    if validation_modulus < 2:
        raise ValueError("validation modulus must be at least two")
    training = [item for item in sequences if item.group % validation_modulus != 0]
    validation = [item for item in sequences if item.group % validation_modulus == 0]
    if not training or not validation:
        raise ValueError("sequence split produced an empty partition")
    return training, validation


def save_ensemble_checkpoint(
    model: EnsembleVisualDynamicsModel, path: Path, planning_horizon: int
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = model.checkpoint_digest
    torch.save(
        {
            "version": 2,
            "model": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "latent_size": model.latent_size,
            "action_size": model.action_size,
            "ensemble_size": model.ensemble_size,
            "duration_conditioned": model.duration_conditioned,
            "duration_size": model.duration_size,
            "max_action_frames": model.max_action_frames,
            "fixed_action_frames": model.fixed_action_frames,
            "planning_horizon": planning_horizon,
            "actions": [action.value for action in ACTION_ORDER],
            "digest": digest,
        },
        path,
    )
    return digest


def load_ensemble_checkpoint(
    path: Path,
    device: Union[torch.device, str] = "cpu",
    frozen: bool = True,
) -> Tuple[EnsembleVisualDynamicsModel, int]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("version") not in (1, 2):
        raise ValueError("unsupported ensemble checkpoint version")
    if checkpoint.get("actions") != [action.value for action in ACTION_ORDER]:
        raise ValueError("checkpoint controller action order does not match runtime")
    model = EnsembleVisualDynamicsModel(
        latent_size=int(checkpoint["latent_size"]),
        action_size=int(checkpoint["action_size"]),
        ensemble_size=int(checkpoint["ensemble_size"]),
        duration_conditioned=bool(checkpoint.get("duration_conditioned", False)),
        duration_size=int(checkpoint.get("duration_size", 16)),
        max_action_frames=int(checkpoint.get("max_action_frames", 32)),
        fixed_action_frames=int(checkpoint.get("fixed_action_frames", 4)),
    )
    model.load_state_dict(checkpoint["model"])
    model.fixed_action_frames_locked = True
    if model.checkpoint_digest != checkpoint.get("digest"):
        raise ValueError("ensemble checkpoint parameter digest mismatch")
    model.to(device)
    if frozen:
        model.freeze()
    return model, int(checkpoint["planning_horizon"])
