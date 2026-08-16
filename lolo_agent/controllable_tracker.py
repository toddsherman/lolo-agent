"""Learned controllable-region tracker head over the frozen spatial encoder.

WP5 spike (roadmap section 17 item 7; direction-review 2026-08-16 Amendment
B step 2): distill the counterfactual controllable-region pseudo-labels
produced by ``counterfactual_labels`` into a small per-cell mask head
warm-started on the frozen spatial world-model encoder.  No supplied goal
prior, player detector, template prototype, or any other semantic anchor
participates anywhere in this module or its derivation: inputs are pixels,
actions, action durations, and duration-matched counterfactual endpoints.

Training-pair construction
--------------------------

Each *labeled* counterfactual arm with at least one controllable cell
yields one training example: the input is the arm's factual endpoint
frame, and the target is the arm's per-cell controllable mask.  Two label
populations are deliberately excluded or downweighted:

- Labeled arms whose controllable mask is empty carry no localization
  evidence (a blocked or no-effect action does not reveal where the
  controllable region is), so they are excluded from training rather than
  treated as all-negative frames that would contradict positive examples
  of the same appearance.
- Residual cells (changed but not action-corroborated) are kept as
  explicitly weighted hard negatives so the head learns to separate
  action-correlated change from incidental change.

Known phase-1 caveat, accepted for the spike: the label mask is the union
of vacated and occupied cells across one action, so the supervision blurs
the region's position by up to one displacement step on the coarse grid.

Checkpoint provenance
---------------------

Saved checkpoints declare ``reward_track``, ``persistent_inputs``, and
``excluded_inputs`` following the ``strict_lineage`` auditor conventions
(allowlisted input names only), and pin both the backbone parameter digest
and the label-manifest content digest so lineage is mechanically
verifiable.  Loading refuses a backbone whose parameter digest differs
from the one recorded at save time.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# The generator's digest prefix is deliberately reused (like the label
# generator reuses the store's internal record iterator) so manifest
# verification cannot drift from the writer's own digest scheme.
from .counterfactual_labels import (
    LABEL_VERSION,
    STATUS_LABELED,
    _DIGEST_PREFIX as _LABEL_DIGEST_PREFIX,
    content_digest,
)
from .environment import Action
from .neural_world_model import frame_tensor
from .pixels import Frame
from .sequence_store import SequenceStore
from .spatial_world_model import SpatialTokenDynamicsModel

Cell = Tuple[int, int]

CHECKPOINT_VERSION = 1
CHECKPOINT_ARCHITECTURE = "counterfactual-controllable-tracker"
REWARD_TRACK = "strict"
PERSISTENT_INPUTS = (
    "pixels",
    "actions",
    "action_durations",
    "duration_matched_noop_pixels",
    "verified_endpoint_pixels",
)
EXCLUDED_INPUTS = (
    "RAM",
    "object_labels",
    "rewards",
    "level_annotations",
    "solutions",
)


@dataclass(frozen=True)
class ControllableArmExample:
    """One labeled counterfactual arm prepared for mask training."""

    source_run_id: str
    group: int
    root_digest: str
    action: str
    duration: int
    endpoint_digest: str
    columns: int
    rows: int
    controllable_cells: Tuple[Cell, ...]
    residual_cells: Tuple[Cell, ...]
    frame: Optional[Frame] = None

    @property
    def sort_key(self) -> Tuple[str, int, str, str, int]:
        return (
            self.source_run_id,
            self.group,
            self.root_digest,
            self.action,
            self.duration,
        )


@dataclass(frozen=True)
class ControllableTrainingMetrics:
    loss: float
    controllable_probability: float
    residual_probability: float
    background_probability: float


@dataclass(frozen=True)
class ControllableValidationReport:
    """Held-out per-cell mask quality against counterfactual pseudo-labels."""

    examples: int
    cells: int
    controllable_cells: int
    residual_cells: int
    prevalence: float
    loss: float
    roc_auc: float
    brier: float
    constant_brier: float
    precision: float
    recall: float
    iou: float
    mean_controllable_probability: float
    mean_residual_probability: float
    mean_background_probability: float
    mean_uncertainty: float
    uncertainty_error_correlation: float


@dataclass(frozen=True)
class ControllableRegionPrediction:
    """Per-cell controllable-region probabilities for one frame.

    ``probabilities`` and ``uncertainty`` are indexed ``[row][column]``.
    ``confidence`` is the probability-mass-weighted ensemble agreement:
    ``1 - 4 * sum(p * var) / sum(p)`` clamped to ``[0, 1]`` (a Bernoulli
    ensemble variance is at most ``0.25``), and ``0.0`` when the map
    carries no probability mass at all.
    """

    columns: int
    rows: int
    probabilities: Tuple[Tuple[float, ...], ...]
    uncertainty: Tuple[Tuple[float, ...], ...]
    confidence: float


def _parse_cells(
    values: Any, columns: int, rows: int, location: str
) -> Tuple[Cell, ...]:
    cells: List[Cell] = []
    for value in values:
        column, row = int(value[0]), int(value[1])
        if not (0 <= column < columns and 0 <= row < rows):
            raise ValueError(f"label cell out of grid bounds in {location}")
        cells.append((column, row))
    return tuple(cells)


def arm_examples_from_records(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[Tuple[ControllableArmExample, ...], Dict[str, int]]:
    """Build training examples from verified label records.

    Only ``labeled`` arms with a non-empty controllable mask become
    examples; empty-mask labeled arms are counted and excluded because
    they carry no localization evidence (see the module docstring).
    """

    examples: List[ControllableArmExample] = []
    labeled_arms = 0
    censored_arms = 0
    empty_mask_arms = 0
    for record in records:
        columns = int(record["columns"])
        rows = int(record["rows"])
        if columns <= 0 or rows <= 0:
            raise ValueError("label grid dimensions must be positive")
        location = f"{record['source_run_id']}:{record['group']}"
        for arm in record["arms"]:
            if arm["status"] != STATUS_LABELED:
                censored_arms += 1
                continue
            labeled_arms += 1
            action = Action(arm["action"])
            if action is Action.NOOP:
                raise ValueError("label records must not contain control arms")
            controllable = _parse_cells(
                arm["controllable_cells"], columns, rows, location
            )
            if not controllable:
                empty_mask_arms += 1
                continue
            examples.append(
                ControllableArmExample(
                    source_run_id=str(record["source_run_id"]),
                    group=int(record["group"]),
                    root_digest=str(record["root_digest"]),
                    action=action.value,
                    duration=int(arm["duration"]),
                    endpoint_digest=str(arm["endpoint_digests"][0]),
                    columns=columns,
                    rows=rows,
                    controllable_cells=controllable,
                    residual_cells=_parse_cells(
                        arm["residual_cells"], columns, rows, location
                    ),
                )
            )
    examples.sort(key=lambda example: example.sort_key)
    return tuple(examples), {
        "roots": len(records),
        "labeled_arms": labeled_arms,
        "censored_arms": censored_arms,
        "empty_mask_labeled_arms": empty_mask_arms,
        "examples": len(examples),
    }


def load_labeled_arm_examples(
    labels_path: Path,
) -> Tuple[Tuple[ControllableArmExample, ...], Dict[str, Any], Dict[str, int]]:
    """Load a label corpus, verifying every content digest before use.

    Returns ``(examples, manifest, statistics)``.  Both the per-record
    digests and the manifest's aggregate digest over all record digests
    are recomputed; any mismatch fails loudly rather than training on a
    corrupted corpus.
    """

    labels_path = Path(labels_path).expanduser().resolve()
    manifest_path = labels_path.with_name(labels_path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != LABEL_VERSION:
        raise ValueError("unsupported label manifest version")
    records: List[Dict[str, Any]] = []
    digests: List[str] = []
    with labels_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if record.get("version") != LABEL_VERSION:
                raise ValueError(
                    f"unsupported label record in {labels_path}:{line_number}"
                )
            if record["content_digest"] != content_digest(record):
                raise ValueError(
                    f"label record digest mismatch in {labels_path}:{line_number}"
                )
            digests.append(record["content_digest"])
            records.append(record)
    aggregate = sha256(
        _LABEL_DIGEST_PREFIX + "\n".join(digests).encode("utf-8")
    ).hexdigest()
    if aggregate != manifest.get("content_digest"):
        raise ValueError(f"label manifest digest mismatch for {labels_path}")
    if manifest.get("roots") != len(records):
        raise ValueError(f"label manifest root count mismatch for {labels_path}")
    examples, statistics = arm_examples_from_records(records)
    return examples, manifest, statistics


def sample_arm_examples(
    examples: Sequence[ControllableArmExample], maximum_examples: int, seed: int
) -> List[ControllableArmExample]:
    """Deterministically subsample arms, preserving canonical order."""

    if maximum_examples <= 0:
        raise ValueError("maximum example count must be positive")
    ordered = sorted(examples, key=lambda example: example.sort_key)
    if len(ordered) <= maximum_examples:
        return ordered
    selected = random.Random(seed).sample(range(len(ordered)), maximum_examples)
    return [ordered[index] for index in sorted(selected)]


def decode_arm_examples(
    store: SequenceStore, examples: Sequence[ControllableArmExample]
) -> List[ControllableArmExample]:
    """Attach endpoint frames decoded through the store's public subset API."""

    frames = store.load_frame_subset(
        example.endpoint_digest for example in examples
    )
    return [
        replace(example, frame=frames[example.endpoint_digest])
        for example in examples
    ]


class _ControllableCellHead(nn.Module):
    """One per-cell mask head over normalized spatial tokens."""

    def __init__(
        self, token_size: int, hidden_size: int, columns: int, rows: int
    ) -> None:
        super().__init__()
        self.columns = columns
        self.rows = rows
        self.features = nn.Sequential(
            nn.Conv2d(token_size, hidden_size, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(hidden_size, 1, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        features = self.features(F.normalize(tokens, dim=1))
        if features.shape[-2:] != (self.rows, self.columns):
            features = F.interpolate(
                features,
                size=(self.rows, self.columns),
                mode="bilinear",
                align_corners=False,
            )
        return self.output(features).squeeze(1)


class ControllableRegionTracker(nn.Module):
    """Frozen spatial encoder plus an ensemble per-cell controllable head.

    The tracker owns its backbone and freezes it on construction; only the
    head ensemble ever trains, and checkpoints persist head parameters
    alone with the backbone pinned by parameter digest.
    """

    def __init__(
        self,
        backbone: SpatialTokenDynamicsModel,
        hidden_size: int = 64,
        ensemble_size: int = 3,
        columns: int = 16,
        rows: int = 15,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or columns <= 0 or rows <= 0:
            raise ValueError("hidden size and grid dimensions must be positive")
        if ensemble_size < 2:
            raise ValueError("tracker ensemble must contain at least two heads")
        self.hidden_size = hidden_size
        self.ensemble_size = ensemble_size
        self.columns = columns
        self.rows = rows
        backbone.freeze()
        self.backbone = backbone
        self.heads = nn.ModuleList(
            _ControllableCellHead(
                backbone.token_size, hidden_size, columns, rows
            )
            for _ in range(ensemble_size)
        )

    @property
    def token_size(self) -> int:
        return self.backbone.token_size

    def train(self, mode: bool = True) -> "ControllableRegionTracker":
        super().train(mode)
        self.backbone.eval()
        return self

    def head_parameters(self) -> Iterator[nn.Parameter]:
        return self.heads.parameters()

    def head_state_dict(self) -> Dict[str, Tensor]:
        return {
            name: value
            for name, value in self.state_dict().items()
            if name.startswith("heads.")
        }

    def forward_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [batch, token, row, column]")
        return torch.stack([head(tokens) for head in self.heads])

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 4:
            raise ValueError("frames must have shape [batch, channel, row, column]")
        with torch.no_grad():
            tokens = self.backbone.encode(frames)
        return self.forward_tokens(tokens)

    @torch.no_grad()
    def predict_map(self, frames: Tensor) -> Tuple[Tensor, Tensor]:
        """Ensemble mean probability and variance maps, ``[batch, row, column]``."""

        probabilities = self(frames).sigmoid()
        return (
            probabilities.mean(dim=0),
            probabilities.var(dim=0, unbiased=False),
        )

    @torch.no_grad()
    def predict(self, frame: Frame) -> ControllableRegionPrediction:
        """Per-cell controllable probabilities with ensemble confidence."""

        device = next(self.heads.parameters()).device
        frames = frame_tensor(frame, device).unsqueeze(0)
        mean, variance = self.predict_map(frames)
        mean_rows = mean[0].cpu().tolist()
        variance_rows = variance[0].cpu().tolist()
        mass = float(mean[0].sum().cpu())
        if mass > 1e-6:
            weighted_variance = float((mean[0] * variance[0]).sum().cpu()) / mass
            confidence = min(1.0, max(0.0, 1.0 - 4.0 * weighted_variance))
        else:
            confidence = 0.0
        return ControllableRegionPrediction(
            columns=self.columns,
            rows=self.rows,
            probabilities=tuple(
                tuple(float(value) for value in row) for row in mean_rows
            ),
            uncertainty=tuple(
                tuple(float(value) for value in row) for row in variance_rows
            ),
            confidence=confidence,
        )

    def freeze(self) -> None:
        self.eval()
        for parameter in self.heads.parameters():
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        self.train()
        for parameter in self.heads.parameters():
            parameter.requires_grad_(True)

    @property
    def checkpoint_digest(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.head_state_dict().items()):
            digest.update(name.encode())
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


def _example_batch(
    examples: Sequence[ControllableArmExample],
    columns: int,
    rows: int,
    device: Union[torch.device, str],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Frames, controllable targets, and residual masks for one batch."""

    if any(example.frame is None for example in examples):
        raise ValueError("mask training requires decoded endpoint frames")
    if any(
        (example.columns, example.rows) != (columns, rows) for example in examples
    ):
        raise ValueError("label grid does not match the tracker grid")
    frames = torch.stack(
        [frame_tensor(example.frame) for example in examples]
    ).to(device)
    targets = torch.zeros((len(examples), rows, columns), dtype=frames.dtype)
    residual = torch.zeros_like(targets)
    for index, example in enumerate(examples):
        for column, row in example.controllable_cells:
            targets[index, row, column] = 1.0
        for column, row in example.residual_cells:
            residual[index, row, column] = 1.0
    return frames, targets.to(device), residual.to(device)


def train_controllable_tracker(
    tracker: ControllableRegionTracker,
    examples: Sequence[ControllableArmExample],
    device: Union[torch.device, str],
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    seed: int = 0,
    positive_weight: float = 8.0,
    residual_weight: float = 4.0,
) -> List[ControllableTrainingMetrics]:
    """Train only the head ensemble; the backbone stays frozen throughout."""

    if not examples:
        raise ValueError("at least one training example is required")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("training parameters must be positive")
    if positive_weight <= 0 or residual_weight <= 0:
        raise ValueError("cell loss weights must be positive")
    tracker.to(device)
    tracker.unfreeze()
    optimizer = torch.optim.AdamW(tracker.head_parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: List[ControllableTrainingMetrics] = []
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            batch = [examples[index] for index in order[start : start + batch_size]]
            frames, targets, residual = _example_batch(
                batch, tracker.columns, tracker.rows, device
            )
            logits = tracker(frames)
            expanded_targets = targets.unsqueeze(0).expand_as(logits)
            errors = F.binary_cross_entropy_with_logits(
                logits, expanded_targets, reduction="none"
            )
            weights = (
                1.0
                + (positive_weight - 1.0) * targets
                + (residual_weight - 1.0) * residual
            ).unsqueeze(0).expand_as(logits)
            mask = (
                torch.rand(
                    (tracker.ensemble_size, len(batch)),
                    generator=generator,
                    device="cpu",
                )
                >= 0.25
            )
            uncovered = ~mask.any(dim=0)
            mask[0, uncovered] = True
            mask = (
                mask.to(device=device, dtype=logits.dtype)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand_as(logits)
            )
            loss = (errors * weights * mask).sum() / (weights * mask).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(tracker.heads.parameters(), 5.0)
            optimizer.step()
            probabilities = logits.detach().sigmoid().mean(dim=0)
            controllable = probabilities[targets > 0.5]
            residual_cells = probabilities[residual > 0.5]
            background = probabilities[(targets <= 0.5) & (residual <= 0.5)]
            history.append(
                ControllableTrainingMetrics(
                    loss=float(loss.detach().cpu()),
                    controllable_probability=(
                        float(controllable.mean().cpu()) if controllable.numel() else 0.0
                    ),
                    residual_probability=(
                        float(residual_cells.mean().cpu())
                        if residual_cells.numel()
                        else 0.0
                    ),
                    background_probability=(
                        float(background.mean().cpu()) if background.numel() else 0.0
                    ),
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
def validate_controllable_tracker(
    tracker: ControllableRegionTracker,
    examples: Sequence[ControllableArmExample],
    device: Union[torch.device, str],
    batch_size: int = 32,
) -> ControllableValidationReport:
    if not examples:
        raise ValueError("at least one validation example is required")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    tracker.to(device)
    tracker.eval()
    probabilities: List[float] = []
    uncertainties: List[float] = []
    labels: List[int] = []
    residual_flags: List[bool] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        frames, targets, residual = _example_batch(
            batch, tracker.columns, tracker.rows, device
        )
        mean, variance = tracker.predict_map(frames)
        probabilities.extend(mean.flatten().cpu().tolist())
        uncertainties.extend(variance.flatten().cpu().tolist())
        labels.extend(int(value > 0.5) for value in targets.flatten().cpu().tolist())
        residual_flags.extend(
            value > 0.5 for value in residual.flatten().cpu().tolist()
        )
    epsilon = 1e-6
    losses = [
        -(
            label * math.log(max(value, epsilon))
            + (1 - label) * math.log(max(1.0 - value, epsilon))
        )
        for value, label in zip(probabilities, labels)
    ]
    errors = [abs(value - label) for value, label in zip(probabilities, labels)]
    prevalence = sum(labels) / len(labels)
    predicted = [value >= 0.5 for value in probabilities]
    true_positive = sum(
        1 for flag, label in zip(predicted, labels) if flag and label
    )
    predicted_positive = sum(predicted)
    actual_positive = sum(labels)
    union = predicted_positive + actual_positive - true_positive
    controllable_values = [
        value for value, label in zip(probabilities, labels) if label == 1
    ]
    residual_values = [
        value for value, flag in zip(probabilities, residual_flags) if flag
    ]
    background_values = [
        value
        for value, label, flag in zip(probabilities, labels, residual_flags)
        if label == 0 and not flag
    ]
    return ControllableValidationReport(
        examples=len(examples),
        cells=len(labels),
        controllable_cells=actual_positive,
        residual_cells=sum(residual_flags),
        prevalence=prevalence,
        loss=sum(losses) / len(losses),
        roc_auc=_roc_auc(probabilities, labels),
        brier=sum(error * error for error in errors) / len(errors),
        constant_brier=prevalence * (1.0 - prevalence),
        precision=(
            true_positive / predicted_positive if predicted_positive else 0.0
        ),
        recall=true_positive / actual_positive if actual_positive else 0.0,
        iou=true_positive / union if union else 0.0,
        mean_controllable_probability=(
            sum(controllable_values) / len(controllable_values)
            if controllable_values
            else 0.0
        ),
        mean_residual_probability=(
            sum(residual_values) / len(residual_values) if residual_values else 0.0
        ),
        mean_background_probability=(
            sum(background_values) / len(background_values)
            if background_values
            else 0.0
        ),
        mean_uncertainty=sum(uncertainties) / len(uncertainties),
        uncertainty_error_correlation=_pearson(list(zip(uncertainties, errors))),
    )


def save_controllable_tracker_checkpoint(
    tracker: ControllableRegionTracker,
    path: Path,
    *,
    label_manifest_digest: str,
) -> str:
    """Persist head parameters with pinned backbone and label provenance."""

    if not label_manifest_digest:
        raise ValueError("a label manifest digest is required for provenance")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = tracker.checkpoint_digest
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "architecture": CHECKPOINT_ARCHITECTURE,
            "model": {
                name: value.detach().cpu()
                for name, value in tracker.head_state_dict().items()
            },
            "token_size": tracker.token_size,
            "hidden_size": tracker.hidden_size,
            "ensemble_size": tracker.ensemble_size,
            "columns": tracker.columns,
            "rows": tracker.rows,
            "backbone_parameter_sha256": tracker.backbone.checkpoint_digest,
            "label_manifest_sha256": label_manifest_digest,
            "reward_track": REWARD_TRACK,
            "persistent_inputs": list(PERSISTENT_INPUTS),
            "excluded_inputs": list(EXCLUDED_INPUTS),
            "digest": digest,
        },
        path,
    )
    return digest


def load_controllable_tracker_checkpoint(
    path: Path,
    backbone: SpatialTokenDynamicsModel,
    device: Union[torch.device, str] = "cpu",
    frozen: bool = True,
) -> Tuple[ControllableRegionTracker, Dict[str, Any]]:
    """Rebuild a tracker over an already-loaded backbone, verifying digests."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported controllable tracker checkpoint version")
    if checkpoint.get("architecture") != CHECKPOINT_ARCHITECTURE:
        raise ValueError(
            "controllable tracker checkpoint architecture does not match runtime"
        )
    if checkpoint.get("backbone_parameter_sha256") != backbone.checkpoint_digest:
        raise ValueError(
            "controllable tracker checkpoint was trained with another backbone"
        )
    tracker = ControllableRegionTracker(
        backbone,
        hidden_size=int(checkpoint["hidden_size"]),
        ensemble_size=int(checkpoint["ensemble_size"]),
        columns=int(checkpoint["columns"]),
        rows=int(checkpoint["rows"]),
    )
    state = checkpoint["model"]
    if set(state) != set(tracker.head_state_dict()):
        raise ValueError(
            "controllable tracker checkpoint parameters do not match runtime"
        )
    tracker.load_state_dict(state, strict=False)
    if tracker.checkpoint_digest != checkpoint.get("digest"):
        raise ValueError("controllable tracker checkpoint parameter digest mismatch")
    tracker.to(device)
    if frozen:
        tracker.freeze()
    return tracker, {
        "backbone_parameter_sha256": str(checkpoint["backbone_parameter_sha256"]),
        "label_manifest_sha256": str(checkpoint["label_manifest_sha256"]),
        "reward_track": str(checkpoint["reward_track"]),
        "persistent_inputs": list(checkpoint["persistent_inputs"]),
        "excluded_inputs": list(checkpoint["excluded_inputs"]),
    }
