from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .environment import Action
from .neural_world_model import ACTION_TO_INDEX, frame_tensor
from .pixels import Frame
from .sequence_store import SequenceStore, StoredTransition
from .spatial_world_model import SpatialTokenDynamicsModel


@dataclass(frozen=True)
class ReturnabilitySpec:
    source_digest: str
    target_digest: str
    action: Action
    duration: int
    source_run_id: str
    label: int


@dataclass(frozen=True)
class ReturnabilityExample:
    source: Frame
    target_digest: str
    action: Action
    duration: int
    source_run_id: str
    label: int


@dataclass(frozen=True)
class ReturnabilityTrainingMetrics:
    loss: float
    accuracy: float
    positive_probability: float
    negative_probability: float


@dataclass(frozen=True)
class ReturnabilityValidationReport:
    examples: int
    positives: int
    negatives: int
    loss: float
    accuracy: float
    majority_accuracy: float
    roc_auc: float
    brier: float
    constant_brier: float
    expected_calibration_error: float
    mean_positive_probability: float
    mean_negative_probability: float
    mean_uncertainty: float
    uncertainty_error_correlation: float


def _reachable(
    adjacency: Dict[str, set[str]], source: str, maximum_steps: int
) -> set[str]:
    seen = {source}
    frontier = {source}
    for _ in range(maximum_steps):
        frontier = {
            target
            for node in frontier
            for target in adjacency.get(node, ())
            if target not in seen
        }
        seen.update(frontier)
    seen.remove(source)
    return seen


def build_returnability_specs(
    transitions: Sequence[StoredTransition],
    maximum_return_steps: int = 3,
    minimum_endpoint_actions: int = 5,
) -> Tuple[List[ReturnabilitySpec], Dict[str, int]]:
    """Label only observed returns and well-probed observed non-returns.

    A positive edge has a real visual-state path from its endpoint back to its
    source. A negative edge has no such path within the horizon and its endpoint
    has real outcomes for at least ``minimum_endpoint_actions`` distinct controls.
    Everything else is censored rather than treated as evidence of irreversibility.
    """

    if maximum_return_steps <= 0 or minimum_endpoint_actions <= 0:
        raise ValueError("return steps and endpoint action coverage must be positive")
    adjacency_by_run: Dict[str, Dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    actions_by_run: Dict[str, Dict[str, set[Action]]] = defaultdict(
        lambda: defaultdict(set)
    )
    unique_edges = set()
    for transition in transitions:
        adjacency_by_run[transition.source_run_id][transition.source_digest].add(
            transition.target_digest
        )
        actions_by_run[transition.source_run_id][transition.source_digest].add(
            transition.action
        )
        unique_edges.add(
            (
                transition.source_run_id,
                transition.source_digest,
                transition.target_digest,
                transition.action,
                transition.duration,
            )
        )

    reachability_cache: Dict[Tuple[str, str], set[str]] = {}

    def reach(run_id: str, digest: str) -> set[str]:
        key = (run_id, digest)
        if key not in reachability_cache:
            reachability_cache[key] = _reachable(
                adjacency_by_run[run_id], digest, maximum_return_steps
            )
        return reachability_cache[key]

    specs = []
    self_edges = 0
    censored = 0
    for run_id, source, target, action, duration in sorted(
        unique_edges,
        key=lambda item: (item[0], item[1], item[2], item[3].value, item[4]),
    ):
        if source == target:
            self_edges += 1
            continue
        if source in reach(run_id, target):
            label = 1
        elif (
            len(actions_by_run[run_id].get(target, ()))
            >= minimum_endpoint_actions
        ):
            label = 0
        else:
            censored += 1
            continue
        specs.append(
            ReturnabilitySpec(source, target, action, duration, run_id, label)
        )
    labels = Counter(item.label for item in specs)
    return specs, {
        "transitions": len(transitions),
        "unique_edges": len(unique_edges),
        "source_runs": len(adjacency_by_run),
        "self_edges_excluded": self_edges,
        "censored_edges": censored,
        "labeled_edges": len(specs),
        "positive_edges": labels[1],
        "well_probed_negative_edges": labels[0],
        "maximum_return_steps": maximum_return_steps,
        "minimum_endpoint_actions": minimum_endpoint_actions,
    }


def split_returnability_runs(
    specs: Sequence[ReturnabilitySpec], validation_modulus: int = 5
) -> Tuple[List[ReturnabilitySpec], List[ReturnabilitySpec]]:
    """Hold out complete source runs with a size-balanced deterministic split."""

    if validation_modulus < 2:
        raise ValueError("validation modulus must be at least two")
    source_runs = sorted({item.source_run_id for item in specs})
    if len(source_runs) < 2:
        raise ValueError("run-held-out splitting requires at least two source runs")
    counts = Counter(item.source_run_id for item in specs)
    target = len(specs) / validation_modulus
    stable_rank = {
        run_id: int(hashlib.sha256(run_id.encode()).hexdigest(), 16)
        for run_id in source_runs
    }
    validation_runs = set()
    validation_count = 0
    remaining = set(source_runs)
    while remaining and len(validation_runs) < len(source_runs) - 1:
        candidate = min(
            remaining,
            key=lambda run_id: (
                abs(target - validation_count - counts[run_id]),
                stable_rank[run_id],
            ),
        )
        candidate_error = abs(target - validation_count - counts[candidate])
        if validation_runs and candidate_error >= abs(target - validation_count):
            break
        validation_runs.add(candidate)
        remaining.remove(candidate)
        validation_count += counts[candidate]
    training = [item for item in specs if item.source_run_id not in validation_runs]
    validation = [item for item in specs if item.source_run_id in validation_runs]
    if not training or not validation:
        raise ValueError("run-held-out split produced an empty partition")
    return training, validation


def balanced_returnability_sample(
    specs: Sequence[ReturnabilitySpec], maximum_examples: int, seed: int
) -> List[ReturnabilitySpec]:
    if maximum_examples < 2:
        raise ValueError("maximum examples must be at least two")
    randomizer = random.Random(seed)
    positive = [item for item in specs if item.label == 1]
    negative = [item for item in specs if item.label == 0]
    count = min(len(positive), len(negative), maximum_examples // 2)
    if count == 0:
        raise ValueError("balanced sampling requires both labels")
    sample = randomizer.sample(positive, count) + randomizer.sample(negative, count)
    randomizer.shuffle(sample)
    return sample


def decode_returnability_examples(
    store: SequenceStore, specs: Sequence[ReturnabilitySpec]
) -> List[ReturnabilityExample]:
    frames = store.load_frame_subset(item.source_digest for item in specs)
    return [
        ReturnabilityExample(
            frames[item.source_digest],
            item.target_digest,
            item.action,
            item.duration,
            item.source_run_id,
            item.label,
        )
        for item in specs
    ]


class _SpatialRelationHead(nn.Module):
    def __init__(
        self, token_size: int, hidden_size: int, spatial_bins: int
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(token_size * 4, hidden_size, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((spatial_bins, spatial_bins)),
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * spatial_bins * spatial_bins, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        source = F.normalize(source, dim=1)
        target = F.normalize(target, dim=1)
        relation = torch.cat(
            (source, target, target - source, (target - source).abs()), dim=1
        )
        return self.output(self.features(relation)).squeeze(1)


class SpatialReturnabilityModel(nn.Module):
    """Ensemble relation head over frozen, unlabeled spatial world-model tokens."""

    def __init__(
        self,
        token_size: int,
        hidden_size: int = 64,
        ensemble_size: int = 3,
        spatial_bins: int = 1,
    ) -> None:
        super().__init__()
        if token_size <= 0 or hidden_size <= 0 or spatial_bins <= 0:
            raise ValueError("token, hidden, and spatial-bin sizes must be positive")
        if ensemble_size < 2:
            raise ValueError("returnability ensemble must contain at least two heads")
        self.token_size = token_size
        self.hidden_size = hidden_size
        self.ensemble_size = ensemble_size
        self.spatial_bins = spatial_bins
        self.heads = nn.ModuleList(
            _SpatialRelationHead(token_size, hidden_size, spatial_bins)
            for _ in range(ensemble_size)
        )

    def forward(self, source_tokens: Tensor, target_tokens: Tensor) -> Tensor:
        if source_tokens.shape != target_tokens.shape or source_tokens.ndim != 4:
            raise ValueError("relation tokens must have matching [batch, token, row, column]")
        return torch.stack(
            [head(source_tokens, target_tokens) for head in self.heads]
        )

    def predict(self, source_tokens: Tensor, target_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        probabilities = self(source_tokens, target_tokens).sigmoid()
        return (
            probabilities.mean(dim=0),
            probabilities.var(dim=0, unbiased=False),
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


def _example_batch(
    examples: Sequence[ReturnabilityExample], device: Union[torch.device, str]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    frames = torch.stack([frame_tensor(item.source) for item in examples]).to(device)
    actions = torch.tensor(
        [ACTION_TO_INDEX[item.action] for item in examples],
        dtype=torch.long,
        device=device,
    )
    durations = torch.tensor(
        [item.duration for item in examples], dtype=torch.long, device=device
    )
    labels = torch.tensor(
        [item.label for item in examples], dtype=frames.dtype, device=device
    )
    return frames, actions, durations, labels


@torch.no_grad()
def _predicted_relation_tokens(
    spatial_model: SpatialTokenDynamicsModel,
    frames: Tensor,
    actions: Tensor,
    durations: Tensor,
) -> Tuple[Tensor, Tensor]:
    source_tokens = spatial_model.encode(frames)
    _pixels, predicted_tokens, _uncertainty, _effects = spatial_model.rollout(
        frames,
        actions.unsqueeze(1),
        durations.unsqueeze(1) if spatial_model.duration_conditioned else None,
    )
    return source_tokens, predicted_tokens[:, 0]


def train_returnability_model(
    model: SpatialReturnabilityModel,
    spatial_model: SpatialTokenDynamicsModel,
    examples: Sequence[ReturnabilityExample],
    device: Union[torch.device, str],
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> List[ReturnabilityTrainingMetrics]:
    if not examples:
        raise ValueError("at least one returnability example is required")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("training parameters must be positive")
    spatial_model.to(device)
    spatial_model.freeze()
    model.to(device)
    model.unfreeze()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            batch = [examples[index] for index in order[start : start + batch_size]]
            frames, actions, durations, labels = _example_batch(batch, device)
            source_tokens, target_tokens = _predicted_relation_tokens(
                spatial_model, frames, actions, durations
            )
            logits = model(source_tokens, target_tokens)
            mask = (
                torch.rand(
                    logits.shape,
                    generator=generator,
                    device="cpu",
                )
                >= 0.25
            ).to(device=device, dtype=logits.dtype)
            empty_heads = ~mask.bool().any(dim=1)
            mask[empty_heads, 0] = 1.0
            expanded_labels = labels.unsqueeze(0).expand_as(logits)
            losses = F.binary_cross_entropy_with_logits(
                logits, expanded_labels, reduction="none"
            )
            loss = (losses * mask).sum() / mask.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            probabilities = logits.detach().sigmoid().mean(dim=0)
            positive = probabilities[labels > 0.5]
            negative = probabilities[labels <= 0.5]
            history.append(
                ReturnabilityTrainingMetrics(
                    float(loss.detach().cpu()),
                    float(((probabilities >= 0.5) == labels.bool()).float().mean().cpu()),
                    float(positive.mean().cpu()) if positive.numel() else 0.0,
                    float(negative.mean().cpu()) if negative.numel() else 0.0,
                )
            )
    return history


def _pearson(pairs: Sequence[Tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return 0.0
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    denominator_y = sum((y - mean_y) ** 2 for _, y in pairs)
    denominator = math.sqrt(denominator_x * denominator_y)
    return numerator / denominator if denominator > 0 else 0.0


def _roc_auc(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ordered = sorted(zip(probabilities, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


@torch.no_grad()
def validate_returnability_model(
    model: SpatialReturnabilityModel,
    spatial_model: SpatialTokenDynamicsModel,
    examples: Sequence[ReturnabilityExample],
    device: Union[torch.device, str],
    batch_size: int = 64,
) -> ReturnabilityValidationReport:
    if not examples:
        raise ValueError("at least one validation example is required")
    model.to(device)
    model.eval()
    spatial_model.to(device)
    spatial_model.freeze()
    probabilities: List[float] = []
    uncertainties: List[float] = []
    labels: List[int] = []
    losses = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        frames, actions, durations, batch_labels = _example_batch(batch, device)
        source_tokens, target_tokens = _predicted_relation_tokens(
            spatial_model, frames, actions, durations
        )
        mean, uncertainty = model.predict(source_tokens, target_tokens)
        losses.extend(
            F.binary_cross_entropy(mean, batch_labels, reduction="none")
            .cpu()
            .tolist()
        )
        probabilities.extend(mean.cpu().tolist())
        uncertainties.extend(uncertainty.cpu().tolist())
        labels.extend(int(value) for value in batch_labels.cpu().tolist())
    prevalence = sum(labels) / len(labels)
    predictions = [value >= 0.5 for value in probabilities]
    errors = [abs(value - label) for value, label in zip(probabilities, labels)]
    brier = sum(error * error for error in errors) / len(errors)
    calibration = 0.0
    for lower_index in range(10):
        lower = lower_index / 10.0
        upper = (lower_index + 1) / 10.0
        members = [
            index
            for index, value in enumerate(probabilities)
            if lower <= value < upper or (upper == 1.0 and value == 1.0)
        ]
        if members:
            confidence = sum(probabilities[index] for index in members) / len(members)
            frequency = sum(labels[index] for index in members) / len(members)
            calibration += len(members) / len(labels) * abs(confidence - frequency)
    positive_values = [
        value for value, label in zip(probabilities, labels) if label == 1
    ]
    negative_values = [
        value for value, label in zip(probabilities, labels) if label == 0
    ]
    return ReturnabilityValidationReport(
        examples=len(labels),
        positives=sum(labels),
        negatives=len(labels) - sum(labels),
        loss=sum(losses) / len(losses),
        accuracy=sum(prediction == bool(label) for prediction, label in zip(predictions, labels))
        / len(labels),
        majority_accuracy=max(prevalence, 1.0 - prevalence),
        roc_auc=_roc_auc(probabilities, labels),
        brier=brier,
        constant_brier=prevalence * (1.0 - prevalence),
        expected_calibration_error=calibration,
        mean_positive_probability=sum(positive_values) / len(positive_values),
        mean_negative_probability=sum(negative_values) / len(negative_values),
        mean_uncertainty=sum(uncertainties) / len(uncertainties),
        uncertainty_error_correlation=_pearson(list(zip(uncertainties, errors))),
    )


def save_returnability_checkpoint(
    model: SpatialReturnabilityModel,
    path: Path,
    spatial_checkpoint_digest: str,
    maximum_return_steps: int,
    minimum_endpoint_actions: int,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = model.checkpoint_digest
    torch.save(
        {
            "version": 1,
            "architecture": "unlabeled-spatial-returnability",
            "model": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "token_size": model.token_size,
            "hidden_size": model.hidden_size,
            "ensemble_size": model.ensemble_size,
            "spatial_bins": model.spatial_bins,
            "spatial_checkpoint_digest": spatial_checkpoint_digest,
            "maximum_return_steps": maximum_return_steps,
            "minimum_endpoint_actions": minimum_endpoint_actions,
            "target": "observed visual-state return path",
            "persistent_inputs": [
                "pixels",
                "actions",
                "action_durations",
                "observed_transition_graph",
            ],
            "excluded_inputs": [
                "RAM",
                "object_labels",
                "rewards",
                "level_annotations",
                "solutions",
            ],
            "digest": digest,
        },
        path,
    )
    return digest


def load_returnability_checkpoint(
    path: Path,
    spatial_checkpoint_digest: str,
    device: Union[torch.device, str] = "cpu",
    frozen: bool = True,
) -> Tuple[SpatialReturnabilityModel, Dict[str, int]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1:
        raise ValueError("unsupported returnability checkpoint version")
    if checkpoint.get("architecture") != "unlabeled-spatial-returnability":
        raise ValueError("returnability checkpoint architecture does not match runtime")
    if checkpoint.get("spatial_checkpoint_digest") != spatial_checkpoint_digest:
        raise ValueError("returnability checkpoint was trained with another spatial model")
    model = SpatialReturnabilityModel(
        token_size=int(checkpoint["token_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        ensemble_size=int(checkpoint["ensemble_size"]),
        spatial_bins=int(checkpoint.get("spatial_bins", 1)),
    )
    model.load_state_dict(checkpoint["model"])
    if model.checkpoint_digest != checkpoint.get("digest"):
        raise ValueError("returnability checkpoint parameter digest mismatch")
    model.to(device)
    if frozen:
        model.freeze()
    return model, {
        "maximum_return_steps": int(checkpoint["maximum_return_steps"]),
        "minimum_endpoint_actions": int(checkpoint["minimum_endpoint_actions"]),
    }
