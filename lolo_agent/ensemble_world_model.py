from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

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

    def __post_init__(self) -> None:
        if not self.actions or len(self.frames) != len(self.actions) + 1:
            raise ValueError("a visual sequence requires one more frame than actions")


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
    ) -> None:
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("ensemble size must be at least two")
        self.latent_size = latent_size
        self.action_size = action_size
        self.ensemble_size = ensemble_size
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
        self.dynamics_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_size + action_size, latent_size * 2),
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

    def transition_ensemble(self, latents: Tensor, actions: Tensor) -> Tensor:
        if latents.ndim != 3 or latents.shape[0] != self.ensemble_size:
            raise ValueError("latents must have shape [ensemble, batch, latent]")
        embedded = self.action_embedding(actions)
        predictions = []
        for index, head in enumerate(self.dynamics_heads):
            latent = latents[index]
            delta = head(torch.cat((latent, embedded), dim=1))
            predictions.append(latent + delta)
        return torch.stack(predictions)

    def rollout(self, source: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if actions.ndim != 2:
            raise ValueError("actions must have shape [batch, horizon]")
        latents = self.initial_ensemble(source)
        pixels = []
        means = []
        uncertainty = []
        for step in range(actions.shape[1]):
            latents = self.transition_ensemble(latents, actions[:, step])
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
) -> Tuple[Tensor, Tensor]:
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
    return frames, actions


def ensemble_sequence_loss(
    model: EnsembleVisualDynamicsModel,
    frames: Tensor,
    actions: Tensor,
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
        latents = model.transition_ensemble(latents, actions[:, step])
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
            frames, actions = sequence_batch(batch, device)
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
                model, frames, actions, bootstrap_mask=bootstrap_mask
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
        frames, actions = sequence_batch(batch, device)
        predictions, _latents, uncertainty = model.rollout(frames[:, 0], actions)
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
    seed: int = 0,
) -> List[VisualSequence]:
    if min(roots, branches_per_root, horizon, action_frames) <= 0:
        raise ValueError("collector sizes must be positive")
    randomizer = random.Random(seed)
    current = env.reset()
    release_state = getattr(env, "release_state", None)
    sequences = []
    for group in range(roots):
        root = env.save_state()
        branches = []
        for _ in range(branches_per_root):
            env.load_state(root)
            frames = [current]
            actions = []
            for _ in range(horizon):
                action = randomizer.choice(ACTION_ORDER)
                actions.append(action)
                frames.append(env.step(action, action_frames))
            child = env.save_state()
            sequences.append(VisualSequence(group, tuple(frames), tuple(actions)))
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
            "version": 1,
            "model": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "latent_size": model.latent_size,
            "action_size": model.action_size,
            "ensemble_size": model.ensemble_size,
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
    if checkpoint.get("version") != 1:
        raise ValueError("unsupported ensemble checkpoint version")
    if checkpoint.get("actions") != [action.value for action in ACTION_ORDER]:
        raise ValueError("checkpoint controller action order does not match runtime")
    model = EnsembleVisualDynamicsModel(
        latent_size=int(checkpoint["latent_size"]),
        action_size=int(checkpoint["action_size"]),
        ensemble_size=int(checkpoint["ensemble_size"]),
    )
    model.load_state_dict(checkpoint["model"])
    if model.checkpoint_digest != checkpoint.get("digest"):
        raise ValueError("ensemble checkpoint parameter digest mismatch")
    model.to(device)
    if frozen:
        model.freeze()
    return model, int(checkpoint["planning_horizon"])
