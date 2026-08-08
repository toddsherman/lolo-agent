from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .environment import Action, PixelSaveStateEnv
from .pixels import Frame


ACTION_ORDER: Tuple[Action, ...] = (
    Action.NOOP,
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
    Action.A,
    Action.B,
    Action.START,
    Action.SELECT,
)
ACTION_TO_INDEX: Dict[Action, int] = {action: index for index, action in enumerate(ACTION_ORDER)}


@dataclass(frozen=True)
class VisualTransition:
    source: Frame
    action: Action
    target: Frame


@dataclass(frozen=True)
class TrainingMetrics:
    loss: float
    current_reconstruction: float
    next_prediction: float
    latent_prediction: float


class VisualDynamicsModel(nn.Module):
    """Action-conditioned visual world model with no game-specific concepts."""

    def __init__(self, latent_size: int = 256, action_size: int = 32) -> None:
        super().__init__()
        self.latent_size = latent_size
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
        self.dynamics = nn.Sequential(
            nn.Linear(latent_size + action_size, latent_size * 2),
            nn.SiLU(),
            nn.Linear(latent_size * 2, latent_size),
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
        features = self.encoder_convolution(frames)
        return self.encoder_projection(features.flatten(1))

    def transition(self, latent: Tensor, actions: Tensor) -> Tensor:
        embedded = self.action_embedding(actions)
        delta = self.dynamics(torch.cat((latent, embedded), dim=1))
        return latent + delta

    def decode(self, latent: Tensor) -> Tensor:
        features = self.decoder_projection(latent).reshape(-1, 128, 8, 8)
        return self.decoder_convolution(features)

    def forward(self, source: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        source_latent = self.encode(source)
        predicted_latent = self.transition(source_latent, actions)
        reconstructed_source = self.decode(source_latent)
        predicted_target = self.decode(predicted_latent)
        return reconstructed_source, predicted_target, predicted_latent

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
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(bytes(str(tuple(tensor.shape)), "ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


def frame_tensor(frame: Frame, device: Union[torch.device, str] = "cpu") -> Tensor:
    if frame.channels == 3:
        tensor = torch.frombuffer(bytearray(frame.pixels), dtype=torch.uint8)
        tensor = tensor.reshape(frame.height, frame.width, 3).permute(2, 0, 1)
    elif frame.channels == 1:
        tensor = torch.frombuffer(bytearray(frame.pixels), dtype=torch.uint8)
        tensor = tensor.reshape(1, frame.height, frame.width).repeat(3, 1, 1)
    else:
        raise ValueError(f"unsupported frame channel count: {frame.channels}")
    tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
    return F.interpolate(
        tensor.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
    ).squeeze(0)


def transition_batch(
    transitions: Sequence[VisualTransition], device: Union[torch.device, str]
) -> Tuple[Tensor, Tensor, Tensor]:
    source = torch.stack([frame_tensor(item.source) for item in transitions]).to(device)
    target = torch.stack([frame_tensor(item.target) for item in transitions]).to(device)
    actions = torch.tensor(
        [ACTION_TO_INDEX[item.action] for item in transitions], dtype=torch.long, device=device
    )
    return source, actions, target


def world_model_loss(
    model: VisualDynamicsModel, source: Tensor, actions: Tensor, target: Tensor
) -> Tuple[Tensor, TrainingMetrics]:
    reconstructed_source, predicted_target, predicted_latent = model(source, actions)
    with torch.no_grad():
        target_latent = model.encode(target)
    current_loss = F.l1_loss(reconstructed_source, source)
    next_loss = F.l1_loss(predicted_target, target)
    latent_loss = F.smooth_l1_loss(predicted_latent, target_latent)
    loss = 0.25 * current_loss + next_loss + 0.5 * latent_loss
    metrics = TrainingMetrics(
        loss=float(loss.detach().cpu()),
        current_reconstruction=float(current_loss.detach().cpu()),
        next_prediction=float(next_loss.detach().cpu()),
        latent_prediction=float(latent_loss.detach().cpu()),
    )
    return loss, metrics


def train_world_model(
    model: VisualDynamicsModel,
    transitions: Sequence[VisualTransition],
    device: Union[torch.device, str],
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> List[TrainingMetrics]:
    if not transitions:
        raise ValueError("at least one transition is required")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    model.to(device)
    model.unfreeze()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    metrics: List[TrainingMetrics] = []
    for _ in range(epochs):
        order = torch.randperm(len(transitions), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            batch = [transitions[index] for index in order[start : start + batch_size]]
            source, actions, target = transition_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, batch_metrics = world_model_loss(model, source, actions, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            metrics.append(batch_metrics)
    return metrics


def collect_branched_transitions(
    env: PixelSaveStateEnv,
    decisions: int = 24,
    branches_per_decision: int = 3,
    action_frames: int = 4,
    seed: int = 0,
) -> List[VisualTransition]:
    """Collect rule-free interaction data using save-state alternatives."""

    if decisions <= 0 or branches_per_decision <= 0 or action_frames <= 0:
        raise ValueError("collector counts must be positive")
    randomizer = random.Random(seed)
    env.reset()
    transitions: List[VisualTransition] = []
    release_state = getattr(env, "release_state", None)
    for _ in range(decisions):
        source = env.observe() if hasattr(env, "observe") else env.step(Action.NOOP)
        root = env.save_state()
        actions = randomizer.sample(
            ACTION_ORDER, k=min(branches_per_decision, len(ACTION_ORDER))
        )
        branches = []
        for action in actions:
            env.load_state(root)
            target = env.step(action, action_frames)
            child = env.save_state()
            transitions.append(VisualTransition(source, action, target))
            branches.append((child, target))
        chosen, _ = randomizer.choice(branches)
        env.load_state(chosen)
        if release_state is not None:
            release_state(root)
            for child, _ in branches:
                release_state(child)
    return transitions


def choose_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
