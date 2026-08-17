"""Pixel-resolution controllable-silhouette refinement head (WP5 spike).

Learnings section 4.34: tracker v4 localizes the controllable region at
cell resolution, but a cell-resolution mask cannot silently substitute
into the pixel-resolution masking convention used by the ``object_tracks``
quantities -- the erased extent differs by construction.  This module is
the recorded plan-change response: a small refinement head that, given the
frame and the FROZEN tracker v4 per-cell controllable probability map,
predicts the per-pixel controllable silhouette within and around the
anchored cells.

Supervision (detector-free, strict)
-----------------------------------

Pixel-resolution supervision already exists without any supplied detector:
the factual-versus-duration-matched-``NOOP`` endpoint difference is per
PIXEL before ``counterfactual_labels`` pools it to cells.  The pixel label
path here mirrors the cell path exactly, at pixel granularity:

1. The changed-pixel set of a factual arm is the byte-exact per-pixel
   difference between its endpoint frame and the duration-matched control
   endpoint.
2. Controllable silhouette candidates are the 4-connected components of
   that changed-pixel set that survive leave-one-action-out corroboration:
   a component qualifies only if it intersects the intersection of the
   changed-pixel sets of every corroborating sibling arm (a different
   primitive action with a non-empty changed-pixel set).  Arms sharing an
   action never corroborate each other; empty-change siblings abstain.
3. Residual pixels (changed but not corroborated) are kept as explicitly
   weighted hard negatives.

Censoring is inherited from the pinned cell-label records: the pixel path
consumes verified ``counterfactual_labels`` records and emits pixel labels
only for arms the cell path labeled, copying every censor status
unchanged.  Cross-checks fail loudly if the pixel-level structure
disagrees with the record (the coarse cells touched by the changed pixels
must equal the record's ``changed_cells``, and the corroborating-arm count
must match).  Known inherited caveat: the changed-pixel set is the union
of the vacated and occupied silhouettes across one action, so the target
blurs by up to one displacement step; the appearance-conditioned head is
expected to resolve that blur because vacated pixels look like background.

Reconstruction convention (fixed here, before any gate run)
-----------------------------------------------------------

``reconstruct_silhouette_pixels`` turns head probabilities into a mask:
pixels at or above ``PIXEL_MASK_PROBABILITY_THRESHOLD`` (0.5, the same
pinned operating point every WP5 instrument uses) INSIDE the anchor
region -- tracker-v4 cells at or above
``ANCHOR_CELL_PROBABILITY_THRESHOLD`` (0.5) dilated by
``ANCHOR_CELL_DILATION`` (1) cell -- then Chebyshev-dilated by
``SILHOUETTE_HALO_DILATION`` (3) pixels, mirroring the documented halo
radius of the recorded assisted masking convention (a fixed public
parameter of that convention; no assisted code participates here).  An
empty tracker anchor yields an empty mask.

Strict lineage: this module references no assisted perception symbol;
inputs are pixels, actions, action durations, and duration-matched
counterfactual endpoints.  Checkpoints declare strict provenance and pin
the label manifest, pixel-target corpus, tracker v4 parameters, and
spatial backbone parameters by digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .controllable_tracker import ControllableArmExample
from .counterfactual_labels import (
    LABEL_VERSION,
    STATUS_CENSORED,
    STATUS_LABELED,
    _DIGEST_PREFIX as _LABEL_DIGEST_PREFIX,
    connected_components,
    content_digest,
)
from .pixels import Frame
from .sequence_store import SequenceStore

Cell = Tuple[int, int]
Pixel = Tuple[int, int]

CHECKPOINT_VERSION = 1
CHECKPOINT_ARCHITECTURE = "wp5-pixel-mask-head"
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

# Reconstruction convention constants; see the module docstring.  The
# probability thresholds are the pinned 0.5 operating point shared by
# every WP5 instrument (the substitution replay fixed it from the tracker
# checkpoint's own validation decision rule); the anchor dilation and the
# halo radius are fixed here, before any gate execution, and pinned by the
# unit tests.
PIXEL_MASK_PROBABILITY_THRESHOLD = 0.5
ANCHOR_CELL_PROBABILITY_THRESHOLD = 0.5
ANCHOR_CELL_DILATION = 1
SILHOUETTE_HALO_DILATION = 3

_PIXEL_TARGET_DIGEST_PREFIX = b"lolo-pixel-mask-targets-v1:"


# ---------------------------------------------------------------------------
# Grid geometry (identical integer partition to the downstream feature grid)
# ---------------------------------------------------------------------------


def cell_pixel_block(
    cell: Cell, width: int, height: int, columns: int, rows: int
) -> FrozenSet[Pixel]:
    """Pixels of one coarse cell under the pooled-feature grid partition."""

    column, row = cell
    return frozenset(
        (x, y)
        for y in range(row * height // rows, (row + 1) * height // rows)
        for x in range(column * width // columns, (column + 1) * width // columns)
    )


def pixel_grid_cell(
    x: int, y: int, width: int, height: int, columns: int, rows: int
) -> Cell:
    """The coarse cell containing one pixel (inverse of the block partition)."""

    return (
        ((x + 1) * columns - 1) // width,
        ((y + 1) * rows - 1) // height,
    )


def dilate_cells(
    cells: Iterable[Cell], columns: int, rows: int, radius: int
) -> Tuple[Cell, ...]:
    """Chebyshev-dilate cells on the coarse grid, clipped to bounds."""

    if radius < 0:
        raise ValueError("cell dilation radius must be non-negative")
    dilated: Set[Cell] = set()
    for column, row in cells:
        for row_offset in range(-radius, radius + 1):
            for column_offset in range(-radius, radius + 1):
                candidate = (column + column_offset, row + row_offset)
                if 0 <= candidate[0] < columns and 0 <= candidate[1] < rows:
                    dilated.add(candidate)
    return tuple(sorted(dilated))


def dilate_pixels(
    pixels: Iterable[Pixel], width: int, height: int, radius: int
) -> FrozenSet[Pixel]:
    """Chebyshev-dilate pixels, clipped to the frame bounds."""

    if radius < 0:
        raise ValueError("pixel dilation radius must be non-negative")
    dilated: Set[Pixel] = set()
    for x, y in pixels:
        for y_offset in range(-radius, radius + 1):
            for x_offset in range(-radius, radius + 1):
                candidate_x = x + x_offset
                candidate_y = y + y_offset
                if 0 <= candidate_x < width and 0 <= candidate_y < height:
                    dilated.add((candidate_x, candidate_y))
    return frozenset(dilated)


# ---------------------------------------------------------------------------
# Pixel-level counterfactual silhouette labels
# ---------------------------------------------------------------------------


def pixel_difference(first: Frame, second: Frame) -> FrozenSet[Pixel]:
    """Pixels whose bytes differ exactly between two frames.

    Endpoint frames from the deterministic emulator are byte-exact, so no
    tolerance threshold is applied (same rationale as the cell path's
    ``cell_difference``).
    """

    if (first.width, first.height, first.channels) != (
        second.width,
        second.height,
        second.channels,
    ):
        raise ValueError("cannot difference frames with mismatched geometry")
    if first.pixels == second.pixels:
        return frozenset()
    row_bytes = first.width * first.channels
    channels = first.channels
    changed: Set[Pixel] = set()
    for y in range(first.height):
        offset = y * row_bytes
        line_first = first.pixels[offset : offset + row_bytes]
        line_second = second.pixels[offset : offset + row_bytes]
        if line_first == line_second:
            continue
        for x in range(first.width):
            start = x * channels
            if line_first[start : start + channels] != line_second[start : start + channels]:
                changed.add((x, y))
    return frozenset(changed)


@dataclass(frozen=True)
class PixelArmLabel:
    """Pixel-granularity pseudo-label (or inherited censoring) for one arm."""

    action: str
    duration: int
    endpoint_digest: Optional[str]
    control_digest: Optional[str]
    status: str
    censor_reason: Optional[str]
    corroborating_arms: int
    changed_pixels: Tuple[Pixel, ...]
    controllable_components: Tuple[Tuple[Pixel, ...], ...]
    controllable_pixels: Tuple[Pixel, ...]
    residual_pixels: Tuple[Pixel, ...]


def _eligible_record_arms(
    record: Mapping[str, Any],
) -> Tuple[Tuple[Mapping[str, Any], str], ...]:
    """Arms with an unambiguous endpoint and a matched unambiguous control.

    The cell-path generator records ``control_digest`` exactly for those
    arms, so record-driven eligibility reproduces the cell path's
    eligibility rule without recomputation.
    """

    pairs = []
    for arm in record["arms"]:
        control_digest = arm.get("control_digest")
        if control_digest is None:
            continue
        if len(arm["endpoint_digests"]) != 1:
            raise ValueError(
                "label record carries a control for an ambiguous endpoint"
            )
        pairs.append((arm, str(control_digest)))
    return tuple(pairs)


def label_pixel_root(
    record: Mapping[str, Any],
    frames: Mapping[str, Frame],
    *,
    selected: Optional[AbstractSet[Tuple[str, int]]] = None,
) -> Tuple[PixelArmLabel, ...]:
    """Pixel-granularity labels for one verified cell-label record.

    Statuses and censor reasons are copied from the record (the pixel path
    censors exactly as the cell path does); the changed-pixel structure is
    cross-checked against the record's coarse cells and corroborating-arm
    count, failing loudly on any mismatch.  ``selected`` optionally
    restricts the emitted arms (sibling changed-pixel sets are still
    computed for corroboration).
    """

    columns = int(record["columns"])
    rows = int(record["rows"])
    eligible = _eligible_record_arms(record)
    changed: Dict[Tuple[str, int], FrozenSet[Pixel]] = {}
    geometry: Optional[Tuple[int, int]] = None
    for arm, control_digest in eligible:
        endpoint = frames[str(arm["endpoint_digests"][0])]
        control = frames[control_digest]
        if geometry is None:
            geometry = (endpoint.width, endpoint.height)
        elif geometry != (endpoint.width, endpoint.height):
            raise ValueError("label root mixes frame geometries")
        changed[(str(arm["action"]), int(arm["duration"]))] = pixel_difference(
            endpoint, control
        )
    labels: List[PixelArmLabel] = []
    for arm in record["arms"]:
        key = (str(arm["action"]), int(arm["duration"]))
        if selected is not None and key not in selected:
            continue
        if arm["status"] != STATUS_LABELED:
            labels.append(
                PixelArmLabel(
                    action=key[0],
                    duration=key[1],
                    endpoint_digest=(
                        str(arm["endpoint_digests"][0])
                        if len(arm["endpoint_digests"]) == 1
                        else None
                    ),
                    control_digest=(
                        None
                        if arm.get("control_digest") is None
                        else str(arm["control_digest"])
                    ),
                    status=STATUS_CENSORED,
                    censor_reason=str(arm["censor_reason"]),
                    corroborating_arms=0,
                    changed_pixels=(),
                    controllable_components=(),
                    controllable_pixels=(),
                    residual_pixels=(),
                )
            )
            continue
        if geometry is None:
            raise ValueError("labeled arm without any eligible frames")
        width, height = geometry
        arm_changed = changed[key]
        touched_cells = {
            pixel_grid_cell(x, y, width, height, columns, rows)
            for x, y in arm_changed
        }
        recorded_cells = {
            (int(value[0]), int(value[1])) for value in arm["changed_cells"]
        }
        if touched_cells != recorded_cells:
            raise ValueError(
                "pixel-level changed cells disagree with the label record "
                f"for {record['source_run_id']}:{record['group']} arm {key}"
            )
        corroborating = [
            changed[(str(other["action"]), int(other["duration"]))]
            for other, _control in eligible
            if str(other["action"]) != key[0]
            and changed[(str(other["action"]), int(other["duration"]))]
        ]
        if len(corroborating) != int(arm["corroborating_arms"]):
            raise ValueError(
                "pixel-level corroborating-arm count disagrees with the "
                f"label record for {record['source_run_id']}:{record['group']} "
                f"arm {key}"
            )
        corroborated = (
            frozenset.intersection(*corroborating) if corroborating else frozenset()
        )
        components = connected_components(arm_changed)
        controllable_components = tuple(
            component
            for component in components
            if corroborated.intersection(component)
        )
        controllable_pixels = sorted(
            {pixel for component in controllable_components for pixel in component}
        )
        residual_pixels = sorted(arm_changed.difference(controllable_pixels))
        labels.append(
            PixelArmLabel(
                action=key[0],
                duration=key[1],
                endpoint_digest=str(arm["endpoint_digests"][0]),
                control_digest=str(arm["control_digest"]),
                status=STATUS_LABELED,
                censor_reason=None,
                corroborating_arms=len(corroborating),
                changed_pixels=tuple(sorted(arm_changed)),
                controllable_components=controllable_components,
                controllable_pixels=tuple(controllable_pixels),
                residual_pixels=tuple(residual_pixels),
            )
        )
    return tuple(labels)


# ---------------------------------------------------------------------------
# Label corpus loading and training-example construction
# ---------------------------------------------------------------------------


def load_label_records(
    labels_path: Path,
) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, Any]]:
    """Load cell-label records, verifying every content digest before use."""

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
    return tuple(records), manifest


@dataclass(frozen=True)
class PixelMaskExample:
    """One labeled arm prepared for pixel-silhouette training."""

    source_run_id: str
    group: int
    root_digest: str
    action: str
    duration: int
    endpoint_digest: str
    width: int
    height: int
    columns: int
    rows: int
    target_pixels: Tuple[Pixel, ...]
    residual_pixels: Tuple[Pixel, ...]
    frame: Optional[Frame] = None
    cell_probabilities: Optional[Tuple[Tuple[float, ...], ...]] = None

    @property
    def sort_key(self) -> Tuple[str, int, str, str, int]:
        return (
            self.source_run_id,
            self.group,
            self.root_digest,
            self.action,
            self.duration,
        )


def record_index(
    records: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, int, str], Mapping[str, Any]]:
    index: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for record in records:
        key = (
            str(record["source_run_id"]),
            int(record["group"]),
            str(record["root_digest"]),
        )
        if key in index:
            raise ValueError(f"duplicate label record for root {key}")
        index[key] = record
    return index


def pixel_examples_from_store(
    store: SequenceStore,
    records: Sequence[Mapping[str, Any]],
    selected: Sequence[ControllableArmExample],
    *,
    root_batch_size: int = 256,
) -> Tuple[List[PixelMaskExample], Dict[str, int]]:
    """Build pixel training examples for already-selected labeled arms.

    ``selected`` is the run-held-out, deterministically sampled arm list
    produced by the cell trainer's own selection conventions; this function
    derives the pixel silhouette target of each selected arm and attaches
    the decoded endpoint frame.  Arms whose pixel-level corroborated
    silhouette is empty are excluded (they carry no pixel localization
    evidence), mirroring the cell trainer's empty-mask exclusion.
    """

    if root_batch_size <= 0:
        raise ValueError("root batch size must be positive")
    index = record_index(records)
    by_root: Dict[Tuple[str, int, str], List[ControllableArmExample]] = {}
    for example in sorted(selected, key=lambda item: item.sort_key):
        key = (example.source_run_id, example.group, example.root_digest)
        by_root.setdefault(key, []).append(example)
    root_keys = sorted(by_root)
    examples: List[PixelMaskExample] = []
    empty_pixel_mask_arms = 0
    for start in range(0, len(root_keys), root_batch_size):
        batch_keys = root_keys[start : start + root_batch_size]
        digests: Set[str] = set()
        for key in batch_keys:
            record = index.get(key)
            if record is None:
                raise KeyError(f"label corpus has no record for root {key}")
            for arm, control_digest in _eligible_record_arms(record):
                digests.add(str(arm["endpoint_digests"][0]))
                digests.add(control_digest)
        frames = store.load_frame_subset(digests)
        for key in batch_keys:
            record = index[key]
            arm_keys = {
                (item.action, item.duration) for item in by_root[key]
            }
            labels = label_pixel_root(record, frames, selected=arm_keys)
            labelled = {
                (label.action, label.duration): label for label in labels
            }
            for item in by_root[key]:
                label = labelled[(item.action, item.duration)]
                if label.status != STATUS_LABELED:
                    raise ValueError(
                        "selected arm is censored in the label record: "
                        f"{key} {item.action}/{item.duration}"
                    )
                if not label.controllable_pixels:
                    empty_pixel_mask_arms += 1
                    continue
                frame = frames[item.endpoint_digest]
                examples.append(
                    PixelMaskExample(
                        source_run_id=item.source_run_id,
                        group=item.group,
                        root_digest=item.root_digest,
                        action=item.action,
                        duration=item.duration,
                        endpoint_digest=item.endpoint_digest,
                        width=frame.width,
                        height=frame.height,
                        columns=int(record["columns"]),
                        rows=int(record["rows"]),
                        target_pixels=label.controllable_pixels,
                        residual_pixels=label.residual_pixels,
                        frame=frame,
                    )
                )
    statistics = {
        "selected_arms": len(selected),
        "roots": len(root_keys),
        "examples": len(examples),
        "empty_pixel_mask_arms": empty_pixel_mask_arms,
    }
    return examples, statistics


def pixel_targets_digest(examples: Sequence[PixelMaskExample]) -> str:
    """Deterministic content digest over a pixel-target corpus."""

    digest = hashlib.sha256(_PIXEL_TARGET_DIGEST_PREFIX)
    for example in sorted(examples, key=lambda item: item.sort_key):
        line = "|".join(
            (
                example.source_run_id,
                str(example.group),
                example.root_digest,
                example.action,
                str(example.duration),
                example.endpoint_digest,
                f"{example.width}x{example.height}",
                ";".join(f"{x},{y}" for x, y in sorted(example.target_pixels)),
                ";".join(f"{x},{y}" for x, y in sorted(example.residual_pixels)),
            )
        )
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The refinement head
# ---------------------------------------------------------------------------


def native_frame_tensor(
    frame: Frame, device: Union[torch.device, str] = "cpu"
) -> Tensor:
    """Frame as a ``[3, height, width]`` float tensor at native resolution."""

    if frame.channels == 3:
        tensor = torch.frombuffer(bytearray(frame.pixels), dtype=torch.uint8)
        tensor = tensor.reshape(frame.height, frame.width, 3).permute(2, 0, 1)
    elif frame.channels == 1:
        tensor = torch.frombuffer(bytearray(frame.pixels), dtype=torch.uint8)
        tensor = tensor.reshape(1, frame.height, frame.width).repeat(3, 1, 1)
    else:
        raise ValueError(f"unsupported frame channel count: {frame.channels}")
    return tensor.to(device=device, dtype=torch.float32).div_(255.0)


class PixelMaskHead(nn.Module):
    """Small convolutional per-pixel silhouette head.

    Inputs are the native-resolution frame and the frozen tracker's
    per-cell controllable probability map (upsampled to pixel blocks);
    the output is one logit per pixel.  Three 3x3 convolutions give a
    7x7 receptive field, enough context to separate the controlled
    sprite's palette and outline from background within an anchored
    region.  The tracker itself is never part of this module: freezing
    and provenance for it are handled where the two are composed.
    """

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden size must be positive")
        self.hidden_size = hidden_size
        self.features = nn.Sequential(
            nn.Conv2d(4, hidden_size, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(hidden_size, 1, 1)

    def forward(self, frames: Tensor, cell_maps: Tensor) -> Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("frames must have shape [batch, 3, height, width]")
        if cell_maps.ndim != 3:
            raise ValueError("cell maps must have shape [batch, rows, columns]")
        if frames.shape[0] != cell_maps.shape[0]:
            raise ValueError("frames and cell maps must share a batch size")
        upsampled = F.interpolate(
            cell_maps.unsqueeze(1), size=frames.shape[-2:], mode="nearest"
        )
        features = self.features(torch.cat([frames, upsampled], dim=1))
        return self.output(features).squeeze(1)

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


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelTrainingMetrics:
    loss: float
    target_probability: float
    residual_probability: float
    background_probability: float


@dataclass(frozen=True)
class PixelMaskValidationReport:
    """Held-out per-pixel silhouette quality against pixel pseudo-labels."""

    examples: int
    pixels: int
    target_pixels: int
    residual_pixels: int
    prevalence: float
    loss: float
    roc_auc: float
    brier: float
    constant_brier: float
    precision: float
    recall: float
    iou: float
    mean_target_probability: float
    mean_residual_probability: float
    mean_background_probability: float


def _pixel_batch(
    examples: Sequence[PixelMaskExample],
    device: Union[torch.device, str],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Frames, cell maps, silhouette targets, and residual masks."""

    if not examples:
        raise ValueError("a pixel batch requires at least one example")
    if any(example.frame is None for example in examples):
        raise ValueError("pixel training requires decoded endpoint frames")
    if any(example.cell_probabilities is None for example in examples):
        raise ValueError("pixel training requires attached tracker cell maps")
    geometry = {(example.width, example.height) for example in examples}
    if len(geometry) != 1:
        raise ValueError("a pixel batch cannot mix frame geometries")
    width, height = next(iter(geometry))
    frames = torch.stack(
        [native_frame_tensor(example.frame) for example in examples]
    )
    cell_maps = torch.tensor(
        [example.cell_probabilities for example in examples],
        dtype=frames.dtype,
    )
    targets = torch.zeros((len(examples), height, width), dtype=frames.dtype)
    residual = torch.zeros_like(targets)
    for index, example in enumerate(examples):
        for x, y in example.target_pixels:
            targets[index, y, x] = 1.0
        for x, y in example.residual_pixels:
            residual[index, y, x] = 1.0
    return (
        frames.to(device),
        cell_maps.to(device),
        targets.to(device),
        residual.to(device),
    )


def train_pixel_mask_head(
    head: PixelMaskHead,
    examples: Sequence[PixelMaskExample],
    device: Union[torch.device, str],
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 0,
    positive_weight: float = 8.0,
    residual_weight: float = 4.0,
) -> List[PixelTrainingMetrics]:
    """Train the refinement head alone; nothing else receives gradients."""

    if not examples:
        raise ValueError("at least one training example is required")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("training parameters must be positive")
    if positive_weight <= 0 or residual_weight <= 0:
        raise ValueError("pixel loss weights must be positive")
    head.to(device)
    head.unfreeze()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: List[PixelTrainingMetrics] = []
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            batch = [examples[index] for index in order[start : start + batch_size]]
            frames, cell_maps, targets, residual = _pixel_batch(batch, device)
            logits = head(frames, cell_maps)
            errors = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            weights = (
                1.0
                + (positive_weight - 1.0) * targets
                + (residual_weight - 1.0) * residual
            )
            loss = (errors * weights).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            probabilities = logits.detach().sigmoid()
            target_values = probabilities[targets > 0.5]
            residual_values = probabilities[residual > 0.5]
            background = probabilities[(targets <= 0.5) & (residual <= 0.5)]
            history.append(
                PixelTrainingMetrics(
                    loss=float(loss.detach().cpu()),
                    target_probability=(
                        float(target_values.mean().cpu())
                        if target_values.numel()
                        else 0.0
                    ),
                    residual_probability=(
                        float(residual_values.mean().cpu())
                        if residual_values.numel()
                        else 0.0
                    ),
                    background_probability=(
                        float(background.mean().cpu()) if background.numel() else 0.0
                    ),
                )
            )
    return history


def _roc_auc_from_tensors(probabilities: Tensor, labels: Tensor) -> float:
    """Exact tie-aware ROC AUC (rank formula), matching the cell trainer's."""

    labels = labels.to(torch.float64)
    positives = float(labels.sum().item())
    negatives = float(labels.numel() - positives)
    if positives == 0 or negatives == 0:
        return 0.0
    order = torch.argsort(probabilities)
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    _values, counts = torch.unique_consecutive(
        sorted_probabilities, return_counts=True
    )
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts
    average_ranks = (starts + 1 + ends).to(torch.float64) / 2.0
    label_cumsum = torch.cumsum(sorted_labels, dim=0)
    boundaries = torch.cat(
        [torch.zeros(1, dtype=torch.float64), label_cumsum[ends[:-1] - 1]]
    )
    group_positives = label_cumsum[ends - 1] - boundaries
    positive_rank_sum = float((average_ranks * group_positives).sum().item())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


@torch.no_grad()
def validate_pixel_mask_head(
    head: PixelMaskHead,
    examples: Sequence[PixelMaskExample],
    device: Union[torch.device, str],
    batch_size: int = 16,
) -> PixelMaskValidationReport:
    if not examples:
        raise ValueError("at least one validation example is required")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    head.to(device)
    head.eval()
    probability_chunks: List[Tensor] = []
    label_chunks: List[Tensor] = []
    residual_chunks: List[Tensor] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        frames, cell_maps, targets, residual = _pixel_batch(batch, device)
        probabilities = head(frames, cell_maps).sigmoid()
        probability_chunks.append(probabilities.flatten().to("cpu", torch.float32))
        label_chunks.append(targets.flatten().to("cpu") > 0.5)
        residual_chunks.append(residual.flatten().to("cpu") > 0.5)
    probabilities = torch.cat(probability_chunks)
    labels = torch.cat(label_chunks)
    residual_flags = torch.cat(residual_chunks)
    label_values = labels.to(torch.float64)
    probability_values = probabilities.to(torch.float64)
    pixels = int(labels.numel())
    target_pixels = int(labels.sum().item())
    residual_pixels = int(residual_flags.sum().item())
    prevalence = target_pixels / pixels
    epsilon = 1e-6
    losses = -(
        label_values * torch.log(probability_values.clamp(min=epsilon))
        + (1.0 - label_values)
        * torch.log((1.0 - probability_values).clamp(min=epsilon))
    )
    errors = (probability_values - label_values).abs()
    predicted = probabilities >= 0.5
    true_positive = int((predicted & labels).sum().item())
    predicted_positive = int(predicted.sum().item())
    union = predicted_positive + target_pixels - true_positive
    background_mask = ~labels & ~residual_flags
    return PixelMaskValidationReport(
        examples=len(examples),
        pixels=pixels,
        target_pixels=target_pixels,
        residual_pixels=residual_pixels,
        prevalence=prevalence,
        loss=float(losses.mean().item()),
        roc_auc=_roc_auc_from_tensors(probabilities, labels),
        brier=float((errors * errors).mean().item()),
        constant_brier=prevalence * (1.0 - prevalence),
        precision=(true_positive / predicted_positive if predicted_positive else 0.0),
        recall=(true_positive / target_pixels if target_pixels else 0.0),
        iou=(true_positive / union if union else 0.0),
        mean_target_probability=(
            float(probabilities[labels].mean().item()) if target_pixels else 0.0
        ),
        mean_residual_probability=(
            float(probabilities[residual_flags].mean().item())
            if residual_pixels
            else 0.0
        ),
        mean_background_probability=(
            float(probabilities[background_mask].mean().item())
            if int(background_mask.sum().item())
            else 0.0
        ),
    )


# ---------------------------------------------------------------------------
# Checkpoint provenance
# ---------------------------------------------------------------------------


def save_pixel_mask_head_checkpoint(
    head: PixelMaskHead,
    path: Path,
    *,
    label_manifest_digest: str,
    pixel_targets_sha256: str,
    tracker_parameter_digest: str,
    backbone_parameter_digest: str,
    cell_columns: int,
    cell_rows: int,
) -> str:
    """Persist head parameters with every upstream artifact pinned."""

    required = {
        "label manifest digest": label_manifest_digest,
        "pixel target corpus digest": pixel_targets_sha256,
        "tracker parameter digest": tracker_parameter_digest,
        "backbone parameter digest": backbone_parameter_digest,
    }
    for name, value in required.items():
        if not value:
            raise ValueError(f"a {name} is required for provenance")
    if cell_columns <= 0 or cell_rows <= 0:
        raise ValueError("cell grid dimensions must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = head.checkpoint_digest
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "architecture": CHECKPOINT_ARCHITECTURE,
            "model": {
                name: value.detach().cpu()
                for name, value in head.state_dict().items()
            },
            "hidden_size": head.hidden_size,
            "cell_columns": cell_columns,
            "cell_rows": cell_rows,
            "label_manifest_sha256": label_manifest_digest,
            "pixel_targets_sha256": pixel_targets_sha256,
            "tracker_parameter_sha256": tracker_parameter_digest,
            "backbone_parameter_sha256": backbone_parameter_digest,
            "reward_track": REWARD_TRACK,
            "persistent_inputs": list(PERSISTENT_INPUTS),
            "excluded_inputs": list(EXCLUDED_INPUTS),
            "digest": digest,
        },
        path,
    )
    return digest


def load_pixel_mask_head_checkpoint(
    path: Path,
    device: Union[torch.device, str] = "cpu",
    frozen: bool = True,
) -> Tuple[PixelMaskHead, Dict[str, Any]]:
    """Rebuild the head, verifying the stored parameter digest."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported pixel mask head checkpoint version")
    if checkpoint.get("architecture") != CHECKPOINT_ARCHITECTURE:
        raise ValueError(
            "pixel mask head checkpoint architecture does not match runtime"
        )
    head = PixelMaskHead(hidden_size=int(checkpoint["hidden_size"]))
    state = checkpoint["model"]
    if set(state) != set(head.state_dict()):
        raise ValueError(
            "pixel mask head checkpoint parameters do not match runtime"
        )
    head.load_state_dict(state)
    if head.checkpoint_digest != checkpoint.get("digest"):
        raise ValueError("pixel mask head checkpoint parameter digest mismatch")
    head.to(device)
    if frozen:
        head.freeze()
    return head, {
        "label_manifest_sha256": str(checkpoint["label_manifest_sha256"]),
        "pixel_targets_sha256": str(checkpoint["pixel_targets_sha256"]),
        "tracker_parameter_sha256": str(checkpoint["tracker_parameter_sha256"]),
        "backbone_parameter_sha256": str(checkpoint["backbone_parameter_sha256"]),
        "cell_columns": int(checkpoint["cell_columns"]),
        "cell_rows": int(checkpoint["cell_rows"]),
        "reward_track": str(checkpoint["reward_track"]),
        "persistent_inputs": list(checkpoint["persistent_inputs"]),
        "excluded_inputs": list(checkpoint["excluded_inputs"]),
    }


# ---------------------------------------------------------------------------
# Pixel-mask reconstruction (the substituted mask source for the gate)
# ---------------------------------------------------------------------------


def anchored_cells(
    cell_probabilities: Sequence[Sequence[float]],
    columns: int,
    rows: int,
    threshold: float = ANCHOR_CELL_PROBABILITY_THRESHOLD,
) -> Tuple[Cell, ...]:
    """Tracker cells at or above the pinned operating point."""

    return tuple(
        sorted(
            (column, row)
            for row in range(rows)
            for column in range(columns)
            if cell_probabilities[row][column] >= threshold
        )
    )


def anchor_pixel_region(
    cell_probabilities: Sequence[Sequence[float]],
    columns: int,
    rows: int,
    width: int,
    height: int,
    *,
    cell_threshold: float = ANCHOR_CELL_PROBABILITY_THRESHOLD,
    cell_dilation: int = ANCHOR_CELL_DILATION,
) -> FrozenSet[Pixel]:
    """Pixel region the reconstruction may mark: anchored cells, dilated."""

    cells = anchored_cells(cell_probabilities, columns, rows, cell_threshold)
    if not cells:
        return frozenset()
    region: Set[Pixel] = set()
    for cell in dilate_cells(cells, columns, rows, cell_dilation):
        region |= cell_pixel_block(cell, width, height, columns, rows)
    return frozenset(region)


def reconstruct_silhouette_pixels(
    pixel_probabilities: Sequence[Sequence[float]],
    anchor: AbstractSet[Pixel],
    width: int,
    height: int,
    *,
    threshold: float = PIXEL_MASK_PROBABILITY_THRESHOLD,
    halo: int = SILHOUETTE_HALO_DILATION,
) -> FrozenSet[Pixel]:
    """Head-positive pixels inside the anchor, plus the convention halo."""

    positives = {
        (x, y)
        for x, y in anchor
        if pixel_probabilities[y][x] >= threshold
    }
    if not positives:
        return frozenset()
    return dilate_pixels(positives, width, height, halo)


@dataclass(frozen=True)
class PixelMaskPrediction:
    """Pixel-resolution mask prediction, duck-typed like a cell prediction.

    ``columns``/``rows`` equal the frame width/height, so each grid unit
    is exactly one pixel: the unchanged substitution-replay helpers
    (`learned_mask_cells`, `learned_pixel_mask`, `learned_reference_slot`)
    recover exactly ``mask`` from ``probabilities`` at the pinned 0.5
    threshold without any code change.  ``probabilities`` is the
    reconstructed mask as a 1.0/0.0 indicator.
    """

    columns: int
    rows: int
    probabilities: Tuple[Tuple[float, ...], ...]
    mask: FrozenSet[Pixel]


class PixelSilhouettePredictor:
    """Frozen tracker v4 plus the refinement head as one mask source.

    ``predict`` mirrors the tracker predictor protocol consumed by the
    mask-sensitive gate: the returned prediction carries a per-unit
    probability map -- here at pixel resolution -- and the gate's own
    pinned 0.5 threshold recovers the reconstructed silhouette mask.
    """

    def __init__(
        self,
        tracker: Any,
        head: PixelMaskHead,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        self.tracker = tracker
        self.head = head.to(device)
        self.head.eval()
        self.device = device

    def predict(self, frame: Frame) -> PixelMaskPrediction:
        cell_prediction = self.tracker.predict(frame)
        anchor = anchor_pixel_region(
            cell_prediction.probabilities,
            cell_prediction.columns,
            cell_prediction.rows,
            frame.width,
            frame.height,
        )
        mask: FrozenSet[Pixel] = frozenset()
        if anchor:
            frames = native_frame_tensor(frame, self.device).unsqueeze(0)
            cell_maps = torch.tensor(
                [cell_prediction.probabilities],
                dtype=frames.dtype,
                device=frames.device,
            )
            with torch.no_grad():
                pixel_probabilities = (
                    self.head(frames, cell_maps).sigmoid()[0].cpu().tolist()
                )
            mask = reconstruct_silhouette_pixels(
                pixel_probabilities, anchor, frame.width, frame.height
            )
        indicator = [[0.0] * frame.width for _ in range(frame.height)]
        for x, y in mask:
            indicator[y][x] = 1.0
        return PixelMaskPrediction(
            columns=frame.width,
            rows=frame.height,
            probabilities=tuple(tuple(row) for row in indicator),
            mask=mask,
        )


def attach_cell_probabilities(
    tracker: Any,
    examples: Sequence[PixelMaskExample],
    device: Union[torch.device, str],
    batch_size: int = 32,
) -> List[PixelMaskExample]:
    """Attach frozen-tracker cell probability maps to training examples."""

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    from .neural_world_model import frame_tensor

    attached: List[PixelMaskExample] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        if any(example.frame is None for example in batch):
            raise ValueError("attaching cell maps requires decoded frames")
        frames = torch.stack(
            [frame_tensor(example.frame) for example in batch]
        ).to(device)
        with torch.no_grad():
            mean, _variance = tracker.predict_map(frames)
        for example, cell_map in zip(batch, mean.cpu().tolist()):
            attached.append(
                replace(
                    example,
                    cell_probabilities=tuple(
                        tuple(float(value) for value in row) for row in cell_map
                    ),
                )
            )
    return attached
