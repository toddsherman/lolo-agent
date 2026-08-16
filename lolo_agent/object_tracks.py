from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .environment import Action
from .pixels import Frame


Cell = Tuple[int, int]

_DIRECTION_DELTAS: Dict[Action, Cell] = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}

_DIRECTIONAL_ACTIONS = (
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
)


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _optional_action(value: Any) -> Optional[Action]:
    if value is None or value == "":
        return None
    return value if isinstance(value, Action) else Action(str(value))


def _optional_cell(value: Any) -> Optional[Cell]:
    if value is None:
        return None
    return int(value[0]), int(value[1])


def direction_displacement(
    direction: Optional[Action],
) -> Optional[Cell]:
    """Return the unit cell displacement implied by a directional control."""

    return None if direction is None else _DIRECTION_DELTAS.get(direction)


@dataclass(frozen=True)
class AnonymousObjectTrack:
    """One persistent anonymous visual object learned purely from pixels.

    Every field is either copied from learned pixel evidence or derived from
    it; no ROM values, sprite names, or game rules appear here.  Fields with
    no current source of truth default to ``None``/empty until later work
    packages begin populating them.
    """

    appearance_fingerprint: str = ""
    anonymous_type_id: Optional[int] = None
    current_cell: Optional[Cell] = None
    source_cell: Optional[Cell] = None
    displacement: Optional[Cell] = None
    appearance_state_signature: str = ""
    local_context_signature: str = ""
    phase_signature: str = ""
    neighborhood_signature: str = ""
    persistence_steps: int = 0
    persisted_in_search: bool = False
    # No current source of truth; reserved for later work packages.
    track_id: Optional[str] = None
    previous_cell: Optional[Cell] = None
    destination_cell: Optional[Cell] = None
    previous_appearance_state_signature: Optional[str] = None
    first_observed_frame: Optional[str] = None
    latest_observed_frame: Optional[str] = None
    confidence: Optional[float] = None
    controlled_change_evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AnonymousObjectTransition:
    """One verified controlled interaction with an anonymous object."""

    action: Optional[Action] = None
    action_index: Optional[int] = None
    direction: Optional[Action] = None
    source_cell: Optional[Cell] = None
    interaction_signature: str = ""
    effect_target_distance: Optional[int] = None
    # No current source of truth; reserved for later work packages.
    duration: Optional[int] = None
    effect_cells: Tuple[Cell, ...] = ()
    persistence_horizons: Tuple[int, ...] = ()
    causal_confidence: Optional[float] = None
    reversible_status: Optional[str] = None
    source_appearance_state_signature: Optional[str] = None
    target_appearance_state_signature: Optional[str] = None
    source_phase_signature: Optional[str] = None
    target_phase_signature: Optional[str] = None


@dataclass(frozen=True)
class HumanPriorRootObjectState:
    """Pixel-derived object state imported with an exact save-state restore."""

    world_effect_signature: str = ""
    world_effect_state_signature: str = ""
    tracked_world_effect_cells: Tuple[Cell, ...] = ()
    tracked_world_state_signature: str = ""
    world_effect_changed_pixels: int = 0
    confirmed_world_effect_signature: str = ""
    confirmed_world_context: str = ""
    confirmed_action_indices: Tuple[int, ...] = ()
    confirmed_effect_frontier_reason: str = ""
    confirmed_entity_state_signature: str = ""
    entity_interaction_signature: str = ""
    entity_interaction_action: Optional[Action] = None
    entity_interaction_action_index: Optional[int] = None
    entity_interaction_direction: Optional[Action] = None
    entity_interaction_cell: Optional[Cell] = None
    entity_interaction_appearance_fingerprint: str = ""
    entity_interaction_type_id: Optional[int] = None
    entity_interaction_context_signature: str = ""
    entity_interaction_phase_signature: str = ""
    entity_interaction_neighborhood_signature: str = ""
    entity_effect_target_distance: Optional[int] = None
    entity_effect_persisted_in_search: bool = False
    entity_effect_persistence_steps: int = 0


def causal_spatial_cells(
    spatial_signature: Optional[str], columns: int
) -> set[Cell]:
    """Decode the coarse cells set in one hex effect bitmask."""

    if not spatial_signature:
        return set()
    try:
        occupied = bytes.fromhex(spatial_signature)
    except ValueError:
        return set()
    return {
        (index % columns, index // columns)
        for index, value in enumerate(occupied)
        if value
    }


def player_masked_world_effect_signature(
    spatial_signature: Optional[str],
    analysis: Any,
    frame: Frame,
    action: Optional[Action] = None,
    allow_nonlocal: bool = False,
    *,
    columns: int,
    rows: int,
) -> str:
    """Remove assisted sprites from a matched causal pixel effect.

    The remaining cells are a rule-free, action-conditioned indication
    that something in the room changed independently of the controlled
    sprite.  Comparing against a duration-matched NOOP has already
    removed autonomous animation; masking the source and target player
    tiles prevents ordinary movement from creating path-dependent world
    states.  Detected goals are masked for the same reason: their
    disappearance remains reward evidence, but is not an anonymous
    obstacle transformation. Multi-action matched-time probes may retain
    non-local cells because their controlled path can legitimately affect
    several parts of the screen before the endpoint is observed.
    """

    if not spatial_signature or analysis is None:
        return ""
    try:
        occupied = bytearray.fromhex(spatial_signature)
    except ValueError:
        return ""
    columns = min(columns, frame.width)
    rows = min(rows, frame.height)
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
    goal_slots = set(analysis.collected)
    if analysis.chest_completed or analysis.chest_obtained:
        for slot in (
            analysis.source_chest_slot,
            analysis.target_chest_slot,
        ):
            if slot is not None:
                goal_slots.add(slot)
    for x, y in goal_slots:
        gx = min(columns - 1, max(0, x * columns // frame.width))
        gy = min(rows - 1, max(0, y * rows // frame.height))
        occupied[gy * columns + gx] = 0
    if (
        not allow_nonlocal
        and action
        in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
    ):
        # The stable graph key already represents ordinary detected
        # movement. Coarse cells beside the anchors are dominated by
        # sprite outline/animation spill even when a blocked press leaves
        # the snapped player tile unchanged, so toggling them into the
        # world context invents a room state for every directional pose.
        # Exact option search uses allow_nonlocal=True and retains its
        # stricter persistence, phase, player-mask, and action-control
        # audits for real transformations that accompany movement.
        return ""
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


def world_effect_cells_state_signature(
    frame: Frame,
    cells: Sequence[Cell],
    player_slot: Optional[Cell] = None,
    memory: Any = None,
    player_pixel_mask: Optional[
        Callable[[Frame, Cell], AbstractSet[Cell]]
    ] = None,
) -> str:
    """Hash anonymous cell appearances without the controlled sprite.

    A genuinely manipulated cell can later be occupied by the player.
    Leaving those pixels in the cumulative fingerprint makes one world
    configuration look different for every player pose and wastes the
    configuration reserve on locomotion.  The visual player detector is
    used only to mask the controlled sprite; the remaining anonymous
    appearance still determines the state identity.
    """

    if memory is None or not cells:
        return ""
    ignored_pixels: Optional[set[Cell]] = None
    if player_slot is not None and callable(player_pixel_mask):
        ignored_pixels = player_pixel_mask(frame, player_slot)
    payload = ";".join(
        f"{column},{row}="
        + ",".join(
            str(value)
            for value in memory.feature_at(
                frame,
                column,
                row,
                ignored_pixels,
            )
        )
        for column, row in sorted(cells)
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def world_effect_state_signature(
    frame: Frame,
    effect_signature: str,
    memory: Any = None,
    *,
    columns: int,
    player_slot: Optional[Cell] = None,
    player_pixel_mask: Optional[
        Callable[[Frame, Cell], AbstractSet[Cell]]
    ] = None,
) -> str:
    """Hash anonymous absolute appearances at action-changed cells."""

    cells = causal_spatial_cells(effect_signature, columns)
    return world_effect_cells_state_signature(
        frame,
        sorted(cells),
        player_slot,
        memory,
        player_pixel_mask,
    )


def masked_cell_fingerprint(
    frame: Frame,
    cell: Cell,
    player_slot: Optional[Cell] = None,
    memory: Any = None,
    model: Any = None,
    player_pixel_mask: Optional[
        Callable[[Frame, Cell], AbstractSet[Cell]]
    ] = None,
) -> Tuple[str, Optional[int]]:
    """Fingerprint and classify one cell appearance without player pixels."""

    if memory is None or model is None:
        return "", None
    ignored_pixels = None
    if player_slot is not None and callable(player_pixel_mask):
        ignored_pixels = player_pixel_mask(frame, player_slot)
    feature = memory.feature_at(frame, *cell, ignored_pixels)
    fingerprint = model.appearance_fingerprint(feature)
    type_id, _distance = model.classify(feature)
    return fingerprint, type_id


def anonymous_entity_frame_features(
    frame: Frame,
    memory: Any = None,
    ignored_pixels: Optional[AbstractSet[Cell]] = None,
) -> Dict[Cell, Tuple[int, ...]]:
    """Index every coarse-cell appearance of one frame."""

    if memory is None:
        return {}
    return {
        (column, row): memory.feature_at(
            frame, column, row, ignored_pixels
        )
        for row in range(memory.rows)
        for column in range(memory.columns)
    }


def anonymous_entity_controlled_displacement(
    source_feature: Sequence[int],
    anchor: Cell,
    factual_features: Dict[Cell, Tuple[int, ...]],
    control_features: Dict[Cell, Tuple[int, ...]],
    source_features: Dict[Cell, Tuple[int, ...]],
    *,
    memory: Any = None,
    model: Any = None,
) -> Optional[Cell]:
    """Measure action-caused translation of a rare visual appearance."""

    if memory is None or model is None:
        return None
    source_fingerprint = model.appearance_fingerprint(source_feature)
    source_count = sum(
        model.appearance_fingerprint(feature) == source_fingerprint
        for feature in source_features.values()
    )
    if source_count > 4:
        return None

    def matched_offset(
        features: Dict[Cell, Tuple[int, ...]]
    ) -> Optional[Cell]:
        candidates = []
        for cell, feature in features.items():
            offset = (cell[0] - anchor[0], cell[1] - anchor[1])
            if abs(offset[0]) + abs(offset[1]) > 4:
                continue
            distance = memory.feature_distance(source_feature, feature)
            if distance <= memory.match_threshold:
                candidates.append(
                    (
                        distance,
                        abs(offset[0]) + abs(offset[1]),
                        offset,
                    )
                )
        return None if not candidates else min(candidates)[2]

    factual_offset = matched_offset(factual_features)
    control_offset = matched_offset(control_features)
    if factual_offset is None or control_offset is None:
        return None
    displacement = (
        factual_offset[0] - control_offset[0],
        factual_offset[1] - control_offset[1],
    )
    return displacement if displacement != (0, 0) else None


def anonymous_entity_context_factors(
    frame: Frame,
    anchor: Cell,
    controlled_cells: Iterable[Cell],
    *,
    memory: Any = None,
    model: Any = None,
    features: Optional[Dict[Cell, Tuple[int, ...]]] = None,
    phase_stable_observations: Optional[Mapping[Cell, int]] = None,
    phase_changed_observations: Optional[Mapping[Cell, int]] = None,
    persistent_world_context: str = "",
) -> Dict[str, Any]:
    """Build phase-conditioned relational context from pixels only."""

    if memory is None or model is None:
        return {}
    stable_counts: Mapping[Cell, int] = phase_stable_observations or {}
    changed_counts: Mapping[Cell, int] = phase_changed_observations or {}
    controlled = {
        (int(column), int(row)) for column, row in controlled_cells
    }
    if features is None:
        features = {
            (column, row): memory.feature_at(frame, column, row)
            for row in range(memory.rows)
            for column in range(memory.columns)
        }
    relation = model.relational_context_signature(
        anchor, controlled
    )
    neighborhood = model.neighborhood_signature(anchor, features)
    # Local manipulation and the controlled sprite do not define the
    # room-wide phase. Distant simultaneous changes—including HUD-like
    # visual counters and newly active entities—remain observable.
    stable_phase_cells = {
        cell
        for cell in features
        if stable_counts.get(cell, 0) >= 2
        and stable_counts.get(cell, 0)
        > 2 * changed_counts.get(cell, 0)
    }
    if stable_phase_cells:
        phase_cells = stable_phase_cells - controlled
        phase = model.phase_signature(
            feature
            for cell, feature in features.items()
            if cell in phase_cells
        )
    else:
        phase = "unresolved"
    if persistent_world_context not in (
        "",
        "human-prior-world-root",
    ):
        # A compact persistent pixel effect can represent a consumable
        # visual resource or another mode that is absent from the local
        # entity patch.  Conditioning the anonymous phase on that learned
        # context prevents an interaction observed while the resource was
        # unavailable from suppressing a fresh probe when it is present.
        # The value is itself learned from action-matched pixels; no HUD
        # region or resource meaning is supplied.
        phase = hashlib.sha256(
            (
                f"{phase}|persistent-world="
                f"{persistent_world_context}"
            ).encode("ascii")
        ).hexdigest()[:16]
    return {
        "context_signature": model.factored_context_signature(
            relation, neighborhood, phase
        ),
        "relation_signature": relation,
        "neighborhood_signature": neighborhood,
        "phase_signature": phase,
        "phase_stable_cells": len(stable_phase_cells),
        "persistent_world_context_signature": (
            persistent_world_context
        ),
    }


def legacy_interaction_from_effect_bitmask(
    bitmask_hex: Optional[str],
    path: Sequence[Any],
    columns: int,
    rows: int,
) -> Optional[AnonymousObjectTransition]:
    """Reconstruct a directional interaction from a legacy effect bitmask.

    Archives created before the detailed track was serialized carry only
    the world-effect hex bitmask and the exact controlled path.  When the
    bitmask marks exactly one changed cell and the final control was
    directional, the interaction source cell is the neighbor the control
    departed from, at unit distance.  Anything more ambiguous returns
    ``None`` rather than guessing.
    """

    if not bitmask_hex:
        return None
    try:
        occupied = bytes.fromhex(bitmask_hex)
    except ValueError:
        return None
    if len(occupied) != columns * rows:
        return None
    tracked_cells = tuple(
        sorted(causal_spatial_cells(bitmask_hex, columns))
    )
    if len(tracked_cells) != 1 or not path:
        return None
    action = _optional_action(path[-1])
    if action not in _DIRECTIONAL_ACTIONS:
        return None
    delta = _DIRECTION_DELTAS[action]
    destination = tracked_cells[0]
    return AnonymousObjectTransition(
        action=action,
        action_index=len(path) - 1,
        direction=action,
        source_cell=(
            destination[0] - delta[0],
            destination[1] - delta[1],
        ),
        effect_target_distance=1,
    )


def archived_track_fields(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Parse the persisted track-state subset of option-archive metadata.

    The returned keys match the archived-branch track fields exactly and
    deliberately perform no legacy reconstruction: promoted alternatives
    restore only what their run serialized.  Malformed values raise the
    same ``KeyError``/``TypeError``/``ValueError`` family the archive
    seeding path already treats as invalid metadata.
    """

    return {
        "world_effect_state_signature": str(
            metadata.get(
                "human_prior_option_world_effect_state_signature"
            )
            or ""
        ),
        "tracked_world_effect_cells": tuple(
            sorted(
                (int(value[0]), int(value[1]))
                for value in (
                    metadata.get(
                        "human_prior_option_tracked_world_effect_cells"
                    )
                    or ()
                )
            )
        ),
        "tracked_world_state_signature": str(
            metadata.get(
                "human_prior_option_tracked_world_state_signature"
            )
            or ""
        ),
        "world_effect_changed_pixels": int(
            metadata.get(
                "human_prior_option_world_effect_changed_pixels",
                0,
            )
            or 0
        ),
        "confirmed_action_indices": tuple(
            int(value)
            for value in (
                metadata.get(
                    "human_prior_option_effect_confirmed_action_indices"
                )
                or ()
            )
        ),
        "entity_interaction_signature": str(
            metadata.get(
                "human_prior_option_entity_interaction_signature"
            )
            or ""
        ),
        "entity_interaction_action": (
            None
            if metadata.get(
                "human_prior_option_entity_interaction_action"
            )
            is None
            else Action(
                str(
                    metadata[
                        "human_prior_option_entity_interaction_action"
                    ]
                )
            )
        ),
        "entity_interaction_action_index": (
            None
            if metadata.get(
                "human_prior_option_entity_interaction_action_index"
            )
            is None
            else int(
                metadata[
                    "human_prior_option_entity_interaction_action_index"
                ]
            )
        ),
        "entity_interaction_direction": (
            None
            if metadata.get(
                "human_prior_option_entity_interaction_direction"
            )
            is None
            else Action(
                str(
                    metadata[
                        "human_prior_option_entity_interaction_direction"
                    ]
                )
            )
        ),
        "entity_interaction_cell": _optional_cell(
            metadata.get(
                "human_prior_option_entity_interaction_cell"
            )
        ),
        "entity_interaction_appearance_fingerprint": str(
            metadata.get(
                "anonymous_entity_appearance_fingerprint"
            )
            or ""
        ),
        "entity_interaction_type_id": (
            None
            if metadata.get("anonymous_entity_type_id") is None
            else int(metadata["anonymous_entity_type_id"])
        ),
        "entity_interaction_context_signature": str(
            metadata.get("anonymous_entity_context_signature")
            or ""
        ),
        "entity_interaction_phase_signature": str(
            metadata.get("anonymous_entity_phase_signature")
            or ""
        ),
        "entity_interaction_neighborhood_signature": str(
            metadata.get(
                "anonymous_entity_neighborhood_signature"
            )
            or ""
        ),
        "entity_effect_target_distance": (
            None
            if metadata.get(
                "human_prior_option_entity_effect_target_distance"
            )
            is None
            else int(
                metadata[
                    "human_prior_option_entity_effect_target_distance"
                ]
            )
        ),
        "entity_effect_persisted_in_search": bool(
            metadata.get(
                "human_prior_option_entity_persistence_observed",
                False,
            )
        ),
        "entity_effect_persistence_steps": int(
            metadata.get(
                "human_prior_option_entity_persistence_steps", 0
            )
            or 0
        ),
    }


@dataclass(frozen=True)
class ObjectTrackSet:
    """The complete anonymous-object memory attached to one exact state.

    Work package 1 carries at most one confirmed track and one confirmed
    transition, mirroring the single-track root object state the planner
    already maintains; the tuple representation is the growth point for
    multi-object tracking.
    """

    tracks: Tuple[AnonymousObjectTrack, ...] = ()
    transitions: Tuple[AnonymousObjectTransition, ...] = ()
    world_effect_signature: str = ""
    world_effect_state_signature: str = ""
    tracked_world_effect_cells: Tuple[Cell, ...] = ()
    tracked_world_state_signature: str = ""
    world_effect_changed_pixels: int = 0
    confirmed_world_effect_signature: str = ""
    confirmed_world_context: str = ""
    confirmed_action_indices: Tuple[int, ...] = ()
    confirmed_effect_frontier_reason: str = ""

    @classmethod
    def empty(cls) -> "ObjectTrackSet":
        return cls()

    @property
    def signature(self) -> str:
        """Deterministic content digest of every stored field."""

        payload = json.dumps(
            self._canonical(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    def _canonical(self) -> Dict[str, Any]:
        def canonical_cell(cell: Optional[Cell]) -> Optional[List[int]]:
            return None if cell is None else [int(cell[0]), int(cell[1])]

        def canonical_action(action: Optional[Action]) -> Optional[str]:
            return None if action is None else str(action.value)

        def canonical_track(track: AnonymousObjectTrack) -> Dict[str, Any]:
            return {
                "appearance_fingerprint": track.appearance_fingerprint,
                "anonymous_type_id": track.anonymous_type_id,
                "current_cell": canonical_cell(track.current_cell),
                "source_cell": canonical_cell(track.source_cell),
                "displacement": canonical_cell(track.displacement),
                "appearance_state_signature": (
                    track.appearance_state_signature
                ),
                "local_context_signature": track.local_context_signature,
                "phase_signature": track.phase_signature,
                "neighborhood_signature": track.neighborhood_signature,
                "persistence_steps": track.persistence_steps,
                "persisted_in_search": track.persisted_in_search,
                "track_id": track.track_id,
                "previous_cell": canonical_cell(track.previous_cell),
                "destination_cell": canonical_cell(track.destination_cell),
                "previous_appearance_state_signature": (
                    track.previous_appearance_state_signature
                ),
                "first_observed_frame": track.first_observed_frame,
                "latest_observed_frame": track.latest_observed_frame,
                "confidence": track.confidence,
                "controlled_change_evidence": list(
                    track.controlled_change_evidence
                ),
            }

        def canonical_transition(
            transition: AnonymousObjectTransition,
        ) -> Dict[str, Any]:
            return {
                "action": canonical_action(transition.action),
                "action_index": transition.action_index,
                "direction": canonical_action(transition.direction),
                "source_cell": canonical_cell(transition.source_cell),
                "interaction_signature": transition.interaction_signature,
                "effect_target_distance": (
                    transition.effect_target_distance
                ),
                "duration": transition.duration,
                "effect_cells": [
                    canonical_cell(cell)
                    for cell in transition.effect_cells
                ],
                "persistence_horizons": list(
                    transition.persistence_horizons
                ),
                "causal_confidence": transition.causal_confidence,
                "reversible_status": transition.reversible_status,
                "source_appearance_state_signature": (
                    transition.source_appearance_state_signature
                ),
                "target_appearance_state_signature": (
                    transition.target_appearance_state_signature
                ),
                "source_phase_signature": (
                    transition.source_phase_signature
                ),
                "target_phase_signature": (
                    transition.target_phase_signature
                ),
            }

        return {
            "tracks": sorted(
                (canonical_track(track) for track in self.tracks),
                key=lambda value: json.dumps(value, sort_keys=True),
            ),
            "transitions": sorted(
                (
                    canonical_transition(transition)
                    for transition in self.transitions
                ),
                key=lambda value: json.dumps(value, sort_keys=True),
            ),
            "world_effect_signature": self.world_effect_signature,
            "world_effect_state_signature": (
                self.world_effect_state_signature
            ),
            "tracked_world_effect_cells": [
                [int(cell[0]), int(cell[1])]
                for cell in self.tracked_world_effect_cells
            ],
            "tracked_world_state_signature": (
                self.tracked_world_state_signature
            ),
            "world_effect_changed_pixels": (
                self.world_effect_changed_pixels
            ),
            "confirmed_world_effect_signature": (
                self.confirmed_world_effect_signature
            ),
            "confirmed_world_context": self.confirmed_world_context,
            "confirmed_action_indices": [
                int(value) for value in self.confirmed_action_indices
            ],
            "confirmed_effect_frontier_reason": (
                self.confirmed_effect_frontier_reason
            ),
        }

    def to_telemetry(self) -> Dict[str, Any]:
        """Serialize with the exact archive metadata keys used on disk."""

        track = self.tracks[0] if self.tracks else AnonymousObjectTrack()
        transition = (
            self.transitions[0]
            if self.transitions
            else AnonymousObjectTransition()
        )
        cell = (
            transition.source_cell
            if self.transitions
            else track.source_cell
        )
        return {
            "human_prior_option_world_effect_signature": (
                self.world_effect_signature or None
            ),
            "human_prior_option_world_effect_state_signature": (
                self.world_effect_state_signature or None
            ),
            "human_prior_option_tracked_world_effect_cells": [
                [int(column), int(row)]
                for column, row in self.tracked_world_effect_cells
            ],
            "human_prior_option_tracked_world_state_signature": (
                self.tracked_world_state_signature or None
            ),
            "human_prior_option_world_effect_changed_pixels": (
                self.world_effect_changed_pixels
            ),
            "human_prior_option_effect_confirmed_action_indices": [
                int(value) for value in self.confirmed_action_indices
            ],
            "human_prior_option_effect_frontier": bool(
                self.confirmed_world_effect_signature
            ),
            "human_prior_option_effect_frontier_reason": (
                self.confirmed_effect_frontier_reason or None
            ),
            "human_prior_world_target_context": (
                self.confirmed_world_context or None
            ),
            "human_prior_option_entity_frontier": bool(
                track.appearance_state_signature
            ),
            "human_prior_option_entity_state_signature": (
                track.appearance_state_signature or None
            ),
            "human_prior_option_entity_interaction_signature": (
                transition.interaction_signature or None
            ),
            "human_prior_option_entity_interaction_action": (
                None
                if transition.action is None
                else str(transition.action.value)
            ),
            "human_prior_option_entity_interaction_action_index": (
                transition.action_index
            ),
            "human_prior_option_entity_interaction_direction": (
                None
                if transition.direction is None
                else str(transition.direction.value)
            ),
            "human_prior_option_entity_interaction_cell": (
                None if cell is None else [int(cell[0]), int(cell[1])]
            ),
            "anonymous_entity_appearance_fingerprint": (
                track.appearance_fingerprint or None
            ),
            "anonymous_entity_type_id": track.anonymous_type_id,
            "anonymous_entity_context_signature": (
                track.local_context_signature or None
            ),
            "anonymous_entity_phase_signature": (
                track.phase_signature or None
            ),
            "anonymous_entity_neighborhood_signature": (
                track.neighborhood_signature or None
            ),
            "human_prior_option_entity_effect_target_distance": (
                transition.effect_target_distance
            ),
            "human_prior_option_entity_persistence_observed": (
                track.persisted_in_search
            ),
            "human_prior_option_entity_persistence_steps": (
                track.persistence_steps
            ),
        }

    @classmethod
    def from_root_object_state(
        cls, state: HumanPriorRootObjectState
    ) -> "ObjectTrackSet":
        """Lift the flat single-track planner state into typed tracks."""

        transition = AnonymousObjectTransition(
            action=state.entity_interaction_action,
            action_index=state.entity_interaction_action_index,
            direction=state.entity_interaction_direction,
            source_cell=state.entity_interaction_cell,
            interaction_signature=state.entity_interaction_signature,
            effect_target_distance=state.entity_effect_target_distance,
        )
        track = AnonymousObjectTrack(
            appearance_fingerprint=(
                state.entity_interaction_appearance_fingerprint
            ),
            anonymous_type_id=state.entity_interaction_type_id,
            current_cell=(
                state.tracked_world_effect_cells[0]
                if len(state.tracked_world_effect_cells) == 1
                else None
            ),
            source_cell=state.entity_interaction_cell,
            displacement=direction_displacement(
                state.entity_interaction_direction
            ),
            appearance_state_signature=(
                state.confirmed_entity_state_signature
            ),
            local_context_signature=(
                state.entity_interaction_context_signature
            ),
            phase_signature=state.entity_interaction_phase_signature,
            neighborhood_signature=(
                state.entity_interaction_neighborhood_signature
            ),
            persistence_steps=state.entity_effect_persistence_steps,
            persisted_in_search=state.entity_effect_persisted_in_search,
        )
        # The current cell and displacement are derived views; a track (or
        # transition) exists only when some learned identity field is set.
        track_baseline = AnonymousObjectTrack(
            current_cell=track.current_cell,
            displacement=track.displacement,
        )
        return cls(
            tracks=() if track == track_baseline else (track,),
            transitions=(
                ()
                if transition == AnonymousObjectTransition()
                else (transition,)
            ),
            world_effect_signature=state.world_effect_signature,
            world_effect_state_signature=(
                state.world_effect_state_signature
            ),
            tracked_world_effect_cells=state.tracked_world_effect_cells,
            tracked_world_state_signature=(
                state.tracked_world_state_signature
            ),
            world_effect_changed_pixels=state.world_effect_changed_pixels,
            confirmed_world_effect_signature=(
                state.confirmed_world_effect_signature
            ),
            confirmed_world_context=state.confirmed_world_context,
            confirmed_action_indices=state.confirmed_action_indices,
            confirmed_effect_frontier_reason=(
                state.confirmed_effect_frontier_reason
            ),
        )

    def to_root_object_state(self) -> HumanPriorRootObjectState:
        """Flatten back into the planner's single-track root state."""

        track = self.tracks[0] if self.tracks else AnonymousObjectTrack()
        transition = (
            self.transitions[0]
            if self.transitions
            else AnonymousObjectTransition()
        )
        return HumanPriorRootObjectState(
            world_effect_signature=self.world_effect_signature,
            world_effect_state_signature=(
                self.world_effect_state_signature
            ),
            tracked_world_effect_cells=self.tracked_world_effect_cells,
            tracked_world_state_signature=(
                self.tracked_world_state_signature
            ),
            world_effect_changed_pixels=self.world_effect_changed_pixels,
            confirmed_world_effect_signature=(
                self.confirmed_world_effect_signature
            ),
            confirmed_world_context=self.confirmed_world_context,
            confirmed_action_indices=self.confirmed_action_indices,
            confirmed_effect_frontier_reason=(
                self.confirmed_effect_frontier_reason
            ),
            confirmed_entity_state_signature=(
                track.appearance_state_signature
            ),
            entity_interaction_signature=(
                transition.interaction_signature
            ),
            entity_interaction_action=transition.action,
            entity_interaction_action_index=transition.action_index,
            entity_interaction_direction=transition.direction,
            entity_interaction_cell=(
                transition.source_cell
                if self.transitions
                else track.source_cell
            ),
            entity_interaction_appearance_fingerprint=(
                track.appearance_fingerprint
            ),
            entity_interaction_type_id=track.anonymous_type_id,
            entity_interaction_context_signature=(
                track.local_context_signature
            ),
            entity_interaction_phase_signature=track.phase_signature,
            entity_interaction_neighborhood_signature=(
                track.neighborhood_signature
            ),
            entity_effect_target_distance=(
                transition.effect_target_distance
            ),
            entity_effect_persisted_in_search=track.persisted_in_search,
            entity_effect_persistence_steps=track.persistence_steps,
        )

    @classmethod
    def from_archive_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        path: Optional[Sequence[Any]] = None,
        columns: int = 16,
        tracked_state_resolver: Optional[
            Callable[[Tuple[Cell, ...]], str]
        ] = None,
        fingerprint_resolver: Optional[
            Callable[[Cell], Tuple[str, Optional[int]]]
        ] = None,
    ) -> "ObjectTrackSet":
        """Rebuild the confirmed track from persisted archive metadata.

        Current metadata round-trips verbatim.  Older archives did not
        serialize the detailed track, so the changed cells and directional
        source cell are conservatively reconstructed from their already
        learned pixel-effect signature; the optional resolvers recover the
        frame-dependent appearance signatures the same way the planner did
        before this module existed.
        """

        effect_signature = str(
            metadata.get("human_prior_option_world_effect_signature") or ""
        )
        serialized_cells = metadata.get(
            "human_prior_option_tracked_world_effect_cells"
        )
        tracked_cells = tuple(
            sorted(
                (int(value[0]), int(value[1]))
                for value in (serialized_cells or ())
            )
        )
        if not tracked_cells:
            tracked_cells = tuple(
                sorted(causal_spatial_cells(effect_signature, columns))
            )
        interaction_signature = str(
            metadata.get("human_prior_option_entity_interaction_signature")
            or metadata.get("human_prior_option_entity_state_signature")
            or ""
        )
        if path is None:
            path = tuple(metadata.get("path") or ())
        else:
            path = tuple(path)
        inferred_action = _optional_action(path[-1]) if path else None
        interaction_action = _optional_action(
            metadata.get("human_prior_option_entity_interaction_action")
        ) or (inferred_action if interaction_signature else None)
        interaction_direction = _optional_action(
            metadata.get("human_prior_option_entity_interaction_direction")
        )
        if interaction_direction is None and interaction_action in (
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ):
            interaction_direction = interaction_action
        interaction_cell = _optional_cell(
            metadata.get("human_prior_option_entity_interaction_cell")
        )
        direction_delta = _DIRECTION_DELTAS.get(interaction_direction)
        if (
            interaction_cell is None
            and direction_delta is not None
            and len(tracked_cells) == 1
        ):
            destination = tracked_cells[0]
            interaction_cell = (
                destination[0] - direction_delta[0],
                destination[1] - direction_delta[1],
            )
        tracked_state_signature = str(
            metadata.get(
                "human_prior_option_tracked_world_state_signature"
            )
            or ""
        )
        if (
            not tracked_state_signature
            and tracked_cells
            and tracked_state_resolver is not None
        ):
            tracked_state_signature = tracked_state_resolver(tracked_cells)
        world_state_signature = str(
            metadata.get("human_prior_option_world_effect_state_signature")
            or tracked_state_signature
        )
        entity_state_signature = str(
            metadata.get("human_prior_option_entity_state_signature") or ""
        )
        confirmed_effect = bool(
            metadata.get("human_prior_option_effect_frontier")
            or entity_state_signature
        )
        effect_distance = _optional_int(
            metadata.get(
                "human_prior_option_entity_effect_target_distance"
            )
        )
        if effect_distance is None and interaction_direction is not None:
            effect_distance = 1
        persistence_steps = int(
            metadata.get("human_prior_option_entity_persistence_steps", 0)
            or 0
        )
        persisted = bool(
            metadata.get("human_prior_option_entity_persistence_observed")
            or (confirmed_effect and interaction_signature)
        )
        if persisted and persistence_steps == 0:
            persistence_steps = 1
        appearance_fingerprint = str(
            metadata.get("anonymous_entity_appearance_fingerprint") or ""
        )
        entity_type_id = _optional_int(
            metadata.get("anonymous_entity_type_id")
        )
        if (
            not appearance_fingerprint
            and len(tracked_cells) == 1
            and fingerprint_resolver is not None
        ):
            appearance_fingerprint, entity_type_id = fingerprint_resolver(
                tracked_cells[0]
            )
        state = HumanPriorRootObjectState(
            world_effect_signature=effect_signature,
            world_effect_state_signature=world_state_signature,
            tracked_world_effect_cells=tracked_cells,
            tracked_world_state_signature=tracked_state_signature,
            world_effect_changed_pixels=int(
                metadata.get(
                    "human_prior_option_world_effect_changed_pixels", 0
                )
                or 0
            ),
            confirmed_world_effect_signature=(
                effect_signature if confirmed_effect else ""
            ),
            confirmed_world_context=str(
                metadata.get("human_prior_world_target_context") or ""
            ),
            confirmed_action_indices=tuple(
                int(value)
                for value in (
                    metadata.get(
                        "human_prior_option_effect_confirmed_action_indices"
                    )
                    or ()
                )
            ),
            confirmed_effect_frontier_reason=str(
                metadata.get(
                    "human_prior_option_effect_frontier_reason"
                )
                or ""
            ),
            confirmed_entity_state_signature=entity_state_signature,
            entity_interaction_signature=interaction_signature,
            entity_interaction_action=interaction_action,
            entity_interaction_action_index=_optional_int(
                metadata.get(
                    "human_prior_option_entity_interaction_action_index"
                )
            )
            if metadata.get(
                "human_prior_option_entity_interaction_action_index"
            )
            is not None
            else (len(path) - 1 if interaction_signature and path else None),
            entity_interaction_direction=interaction_direction,
            entity_interaction_cell=interaction_cell,
            entity_interaction_appearance_fingerprint=str(
                appearance_fingerprint
            ),
            entity_interaction_type_id=entity_type_id,
            entity_interaction_context_signature=str(
                metadata.get("anonymous_entity_context_signature") or ""
            ),
            entity_interaction_phase_signature=str(
                metadata.get("anonymous_entity_phase_signature") or ""
            ),
            entity_interaction_neighborhood_signature=str(
                metadata.get("anonymous_entity_neighborhood_signature")
                or ""
            ),
            entity_effect_target_distance=effect_distance,
            entity_effect_persisted_in_search=persisted,
            entity_effect_persistence_steps=persistence_steps,
        )
        return cls.from_root_object_state(state)

    @classmethod
    def from_archived_branch(cls, branch: Any) -> "ObjectTrackSet":
        """Copy the track-state subset of one archived exact branch."""

        return cls.from_root_object_state(
            HumanPriorRootObjectState(
                world_effect_signature=(
                    branch.human_prior_option_world_effect_signature
                ),
                world_effect_state_signature=(
                    branch.world_effect_state_signature
                ),
                tracked_world_effect_cells=(
                    branch.tracked_world_effect_cells
                ),
                tracked_world_state_signature=(
                    branch.tracked_world_state_signature
                ),
                world_effect_changed_pixels=(
                    branch.world_effect_changed_pixels
                ),
                confirmed_world_effect_signature=(
                    branch.goal_world_effect_signature
                ),
                confirmed_world_context=branch.goal_target_world_context,
                confirmed_action_indices=branch.confirmed_action_indices,
                confirmed_effect_frontier_reason=(
                    branch.human_prior_option_effect_frontier_reason
                ),
                confirmed_entity_state_signature=(
                    branch.human_prior_option_entity_state_signature
                ),
                entity_interaction_signature=(
                    branch.entity_interaction_signature
                ),
                entity_interaction_action=(
                    branch.entity_interaction_action
                ),
                entity_interaction_action_index=(
                    branch.entity_interaction_action_index
                ),
                entity_interaction_direction=(
                    branch.entity_interaction_direction
                ),
                entity_interaction_cell=branch.entity_interaction_cell,
                entity_interaction_appearance_fingerprint=(
                    branch.entity_interaction_appearance_fingerprint
                ),
                entity_interaction_type_id=(
                    branch.entity_interaction_type_id
                ),
                entity_interaction_context_signature=(
                    branch.entity_interaction_context_signature
                ),
                entity_interaction_phase_signature=(
                    branch.entity_interaction_phase_signature
                ),
                entity_interaction_neighborhood_signature=(
                    branch.entity_interaction_neighborhood_signature
                ),
                entity_effect_target_distance=(
                    branch.entity_effect_target_distance
                ),
                entity_effect_persisted_in_search=(
                    branch.entity_effect_persisted_in_search
                ),
                entity_effect_persistence_steps=(
                    branch.entity_effect_persistence_steps
                ),
            )
        )


def observe_frame(
    track_set: ObjectTrackSet,
    frame: Frame,
    *,
    memory: Any = None,
    player_slot: Optional[Cell] = None,
    player_pixel_mask: Optional[
        Callable[[Frame, Cell], AbstractSet[Cell]]
    ] = None,
) -> str:
    """Return the current player-masked appearance at the tracked cells."""

    return world_effect_cells_state_signature(
        frame,
        track_set.tracked_world_effect_cells,
        player_slot,
        memory,
        player_pixel_mask,
    )


def match(track_set: ObjectTrackSet, observed_state_signature: str) -> bool:
    """Whether an observed tracked-cell appearance matches the stored one."""

    return bool(track_set.tracked_world_state_signature) and (
        observed_state_signature == track_set.tracked_world_state_signature
    )


def apply_verified_transition(
    track_set: ObjectTrackSet,
    transition: AnonymousObjectTransition,
    track: Optional[AnonymousObjectTrack] = None,
    *,
    confirmed_world_effect_signature: str,
    confirmed_world_context: str,
    confirmed_action_indices: Tuple[int, ...] = (),
    confirmed_effect_frontier_reason: str = "",
    entity_state_signature: str = "",
) -> ObjectTrackSet:
    """Promote a verified transient interaction to the confirmed track.

    Mirrors the planner's entity-frontier promotion: the transient
    candidate becomes the confirmed manipulation identity, and the world
    context advances to the confirmed post-effect configuration.  Pure
    value semantics; the caller supplies the emulator-verified evidence.
    """

    if track is None:
        track = AnonymousObjectTrack(
            source_cell=transition.source_cell,
            displacement=direction_displacement(transition.direction),
        )
    track = replace(
        track,
        appearance_state_signature=(
            entity_state_signature
            or track.appearance_state_signature
        ),
    )
    return replace(
        track_set,
        tracks=(track,),
        transitions=(transition,),
        confirmed_world_effect_signature=confirmed_world_effect_signature,
        confirmed_world_context=confirmed_world_context,
        confirmed_action_indices=tuple(
            int(value) for value in confirmed_action_indices
        ),
        confirmed_effect_frontier_reason=confirmed_effect_frontier_reason,
    )
