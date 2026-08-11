from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import (
    AbstractSet,
    Counter as CounterType,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .environment import Action
from .pixels import Frame


PatchFeature = Tuple[int, ...]
Cell = Tuple[int, int]


@dataclass(frozen=True)
class PatchObservation:
    """One unlabeled visual-patch assignment at a spatial location."""

    column: int
    row: int
    prototype_id: int
    distance: float
    feature: PatchFeature


@dataclass(frozen=True)
class EntityGridObservation:
    """A frame represented only by learned local appearance prototypes."""

    frame_digest: str
    columns: int
    rows: int
    patches: Tuple[PatchObservation, ...]

    def prototype_at(self, column: int, row: int) -> Optional[int]:
        if not 0 <= column < self.columns or not 0 <= row < self.rows:
            return None
        return self.patches[row * self.columns + column].prototype_id

    def signature(self) -> Tuple[int, ...]:
        return tuple(patch.prototype_id for patch in self.patches)


@dataclass(frozen=True)
class PrototypeStats:
    prototype_id: int
    observations: int
    frames_observed: int
    unique_cells: int
    cells: Tuple[Cell, ...]
    feature: PatchFeature

    @property
    def spatial_rarity(self) -> float:
        return 1.0 / math.sqrt(max(1, self.unique_cells))


class UnlabeledEntityMemory:
    """Temporary room memory of recurring visual patches.

    The representation deliberately has no sprite names, object labels, ROM
    values, collision rules, or reward annotations.  Each screen cell is
    pooled into a small RGB appearance vector and assigned to the nearest
    prototype learned from pixels observed during the current episode.  The
    same mechanism represents floor, walls, HUD glyphs, the controlled sprite,
    and puzzle entities; later causal evidence must distinguish their roles.
    """

    _DIRECTION_OFFSETS = {
        Action.UP: (0, -1),
        Action.DOWN: (0, 1),
        Action.LEFT: (-1, 0),
        Action.RIGHT: (1, 0),
    }

    def __init__(
        self,
        columns: int = 16,
        rows: int = 15,
        pooled_columns: int = 4,
        pooled_rows: int = 4,
        quantization: int = 16,
        match_threshold: float = 0.08,
    ) -> None:
        if columns <= 0 or rows <= 0:
            raise ValueError("entity-grid dimensions must be positive")
        if pooled_columns <= 0 or pooled_rows <= 0:
            raise ValueError("pooled patch dimensions must be positive")
        if not 1 <= quantization <= 256:
            raise ValueError("patch quantization must be in [1, 256]")
        if match_threshold < 0.0:
            raise ValueError("prototype match threshold must be non-negative")
        self.columns = columns
        self.rows = rows
        self.pooled_columns = pooled_columns
        self.pooled_rows = pooled_rows
        self.quantization = quantization
        self.match_threshold = match_threshold
        self._prototype_means: List[List[float]] = []
        self._prototype_counts: CounterType[int] = Counter()
        self._prototype_cells: Dict[int, set[Cell]] = {}
        self._prototype_frames: Dict[int, set[int]] = {}
        self._frame_index = 0
        self.interaction_visits: CounterType[
            Tuple[int, Action, int]
        ] = Counter()

    @property
    def prototype_count(self) -> int:
        return len(self._prototype_means)

    @property
    def frame_count(self) -> int:
        return self._frame_index

    def feature_at(
        self,
        frame: Frame,
        column: int,
        row: int,
        ignored_pixels: Optional[AbstractSet[Cell]] = None,
    ) -> PatchFeature:
        """Return the unlabeled pooled appearance at one grid cell.

        Ignored pixels contribute no samples. A fully masked pool is encoded
        as zeroes so action-controlled entity state can remain independent of
        a detected player sprite overlapping the same coarse cell.
        """

        if not 0 <= column < self.columns or not 0 <= row < self.rows:
            raise IndexError("entity-grid cell out of range")
        x0 = column * frame.width // self.columns
        x1 = (column + 1) * frame.width // self.columns
        y0 = row * frame.height // self.rows
        y1 = (row + 1) * frame.height // self.rows
        values: List[int] = []
        for pooled_row in range(self.pooled_rows):
            py0 = y0 + pooled_row * (y1 - y0) // self.pooled_rows
            py1 = y0 + (pooled_row + 1) * (y1 - y0) // self.pooled_rows
            for pooled_column in range(self.pooled_columns):
                px0 = x0 + pooled_column * (x1 - x0) // self.pooled_columns
                px1 = x0 + (pooled_column + 1) * (x1 - x0) // self.pooled_columns
                totals = [0] * frame.channels
                samples = 0
                for y in range(py0, max(py0 + 1, py1)):
                    for x in range(px0, max(px0 + 1, px1)):
                        if ignored_pixels is not None and (
                            x, y
                        ) in ignored_pixels:
                            continue
                        offset = (y * frame.width + x) * frame.channels
                        for channel in range(frame.channels):
                            totals[channel] += frame.pixels[offset + channel]
                        samples += 1
                for total in totals:
                    mean = total // max(1, samples)
                    values.append(min(255, mean) // self.quantization)
        return tuple(values)

    @staticmethod
    def feature_distance(first: Sequence[float], second: Sequence[float]) -> float:
        if len(first) != len(second) or not first:
            return 1.0
        maximum = max(1.0, max((*first, *second)))
        return sum(abs(a - b) for a, b in zip(first, second)) / (
            maximum * len(first)
        )

    def _assign(self, feature: PatchFeature) -> Tuple[int, float]:
        if not self._prototype_means:
            self._prototype_means.append([float(value) for value in feature])
            return 0, 0.0
        distances = [
            self.feature_distance(feature, prototype)
            for prototype in self._prototype_means
        ]
        prototype_id = min(range(len(distances)), key=distances.__getitem__)
        distance = distances[prototype_id]
        if distance > self.match_threshold:
            prototype_id = len(self._prototype_means)
            self._prototype_means.append([float(value) for value in feature])
            return prototype_id, 0.0
        return prototype_id, distance

    def observe(self, frame: Frame) -> EntityGridObservation:
        self._frame_index += 1
        frame_index = self._frame_index
        patches: List[PatchObservation] = []
        assignments: List[Tuple[int, PatchFeature]] = []
        for row in range(self.rows):
            for column in range(self.columns):
                feature = self.feature_at(frame, column, row)
                prototype_id, distance = self._assign(feature)
                patches.append(
                    PatchObservation(
                        column=column,
                        row=row,
                        prototype_id=prototype_id,
                        distance=distance,
                        feature=feature,
                    )
                )
                assignments.append((prototype_id, feature))
                self._prototype_counts[prototype_id] += 1
                self._prototype_cells.setdefault(prototype_id, set()).add(
                    (column, row)
                )
                self._prototype_frames.setdefault(prototype_id, set()).add(
                    frame_index
                )

        # Update prototypes after assigning the complete frame. This avoids a
        # left-to-right scan within one observation changing later identities.
        frame_sums: Dict[int, List[float]] = {}
        frame_counts: CounterType[int] = Counter()
        for prototype_id, feature in assignments:
            if prototype_id not in frame_sums:
                frame_sums[prototype_id] = [0.0] * len(feature)
            for index, value in enumerate(feature):
                frame_sums[prototype_id][index] += value
            frame_counts[prototype_id] += 1
        for prototype_id, sums in frame_sums.items():
            current_count = self._prototype_counts[prototype_id]
            added = frame_counts[prototype_id]
            previous_count = current_count - added
            prototype = self._prototype_means[prototype_id]
            denominator = max(1, current_count)
            for index, value in enumerate(sums):
                prototype[index] = (
                    prototype[index] * previous_count + value
                ) / denominator

        return EntityGridObservation(
            frame_digest=frame.digest,
            columns=self.columns,
            rows=self.rows,
            patches=tuple(patches),
        )

    def stats(self) -> Tuple[PrototypeStats, ...]:
        result = []
        for prototype_id, feature in enumerate(self._prototype_means):
            result.append(
                PrototypeStats(
                    prototype_id=prototype_id,
                    observations=self._prototype_counts[prototype_id],
                    frames_observed=len(
                        self._prototype_frames.get(prototype_id, set())
                    ),
                    unique_cells=len(
                        self._prototype_cells.get(prototype_id, set())
                    ),
                    cells=tuple(
                        sorted(self._prototype_cells.get(prototype_id, set()))
                    ),
                    feature=tuple(round(value) for value in feature),
                )
            )
        return tuple(result)

    def target_prototype(
        self,
        observation: EntityGridObservation,
        anchor: Tuple[int, int],
        action: Action,
        frame_width: int,
        frame_height: int,
    ) -> Optional[int]:
        target = self.action_target_cell(
            anchor, action, frame_width, frame_height
        )
        if target is None:
            return None
        return observation.prototype_at(*target)

    def action_target_cell(
        self,
        anchor: Tuple[int, int],
        action: Action,
        frame_width: int,
        frame_height: int,
    ) -> Optional[Cell]:
        offset = self._DIRECTION_OFFSETS.get(action)
        if offset is None:
            return None
        column = min(
            self.columns - 1,
            max(0, anchor[0] * self.columns // frame_width),
        )
        row = min(
            self.rows - 1,
            max(0, anchor[1] * self.rows // frame_height),
        )
        target = (column + offset[0], row + offset[1])
        if not 0 <= target[0] < self.columns or not 0 <= target[1] < self.rows:
            return None
        return target

    def record_interaction(
        self, prototype_id: Optional[int], action: Action, duration: int
    ) -> int:
        if prototype_id is None or duration <= 0:
            return 0
        key = (prototype_id, action, duration)
        visits_before = self.interaction_visits[key]
        self.interaction_visits[key] += 1
        return visits_before
