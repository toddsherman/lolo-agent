from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Counter as CounterType,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
)

from .environment import Action


AppearanceFeature = Tuple[int, ...]
RelativeCell = Tuple[int, int]


@dataclass(frozen=True)
class AnonymousTypeStats:
    """Summary of one appearance category discovered without a label."""

    type_id: int
    observations: int
    feature: Tuple[float, ...]


@dataclass(frozen=True)
class BehaviorTransition:
    """Explicit, label-free transition implied by one outcome descriptor.

    Every field is derived from measured pixel facts already present in the
    content-addressed outcome descriptor.  ``kind`` names only the measured
    outcome category (displacement, transformation, removal, expulsion,
    blocked/no-effect), never what an appearance is.  Cells are relative to
    the observed appearance's anchor, so a transition transfers under
    translation; the object-track layer is responsible for anchoring a
    transition to absolute room cells.  A removal or expulsion has no
    destination: the tracked appearance ceased to exist locally after the
    persistence-horizon machinery ruled out occlusion, with expulsion
    additionally carrying anchor-relative transit evidence.
    """

    kind: str
    outcome_signature: str
    source_cell: Optional[RelativeCell]
    destination_cell: Optional[RelativeCell]
    displacement: Optional[RelativeCell]
    target_appearance: str
    transit_cells: Tuple[RelativeCell, ...]

    KINDS = (
        "displacement",
        "transformation",
        "removal",
        "expulsion",
        "no_effect",
        "other",
    )

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError("invalid behavior transition kind")


@dataclass(frozen=True)
class BehaviorOutcomeDescriptor:
    """Auditable, label-free semantics for one pixel outcome signature.

    Every field is measured from factual and equal-duration control images.
    The descriptor deliberately says nothing about what an appearance *is*;
    it only preserves how the two pixel outcomes related to one another.

    Schema 9 adds explicit transition facts.  ``target_appearance`` is the
    requantized anonymous fingerprint of the appearance holding the same
    locus or continuing track after a transformation.  ``entity_removed``
    records that the tracked appearance ceased to exist locally after the
    persistence-horizon machinery ruled out occlusion, and
    ``removal_transit_cells`` preserves anchor-relative transit evidence
    for an expulsion.  All remain measured pixel outcomes, never object
    names or supplied mechanics.
    """

    factual_source_relation: str
    control_source_relation: str
    factual_control_relation: str
    relative_effect_cells: Tuple[RelativeCell, ...] = ()
    player_displacement: Optional[RelativeCell] = None
    terminal_visual_change: bool = False
    entity_displacement: Optional[RelativeCell] = None
    global_phase_change: bool = False
    target_appearance: str = ""
    entity_removed: bool = False
    removal_transit_cells: Tuple[RelativeCell, ...] = ()

    _APPEARANCE_RELATIONS = frozenset(("same", "near", "different"))
    _FINGERPRINT_ALPHABET = frozenset("0123456789abcdef")

    def __post_init__(self) -> None:
        for relation in (
            self.factual_source_relation,
            self.control_source_relation,
            self.factual_control_relation,
        ):
            if relation not in self._APPEARANCE_RELATIONS:
                raise ValueError("invalid behavior appearance relation")
        if self.target_appearance and (
            len(self.target_appearance) != 16
            or not self._FINGERPRINT_ALPHABET.issuperset(
                self.target_appearance
            )
        ):
            # Enforcing the content-addressed fingerprint format keeps the
            # field anonymous: a supplied object name cannot fit here.
            raise ValueError(
                "transformation target must be an anonymous fingerprint"
            )
        if (
            self.target_appearance
            and self.factual_source_relation != "different"
        ):
            raise ValueError(
                "a transformation target requires a changed appearance"
            )
        if self.removal_transit_cells and not self.entity_removed:
            raise ValueError(
                "transit evidence requires an entity removal outcome"
            )
        if self.entity_removed:
            if self.factual_source_relation != "different":
                raise ValueError(
                    "a removed appearance cannot still match its source"
                )
            if self.entity_displacement is not None:
                raise ValueError(
                    "a removed appearance cannot keep a displacement"
                )
            if self.target_appearance:
                raise ValueError(
                    "a removed appearance cannot record a transformation"
                )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "factual_source_relation": self.factual_source_relation,
            "control_source_relation": self.control_source_relation,
            "factual_control_relation": self.factual_control_relation,
            "relative_effect_cells": [
                [int(column), int(row)]
                for column, row in self.relative_effect_cells
            ],
            "player_displacement": (
                None
                if self.player_displacement is None
                else [
                    int(self.player_displacement[0]),
                    int(self.player_displacement[1]),
                ]
            ),
            "terminal_visual_change": bool(self.terminal_visual_change),
        }
        # Omit new default-valued fields so descriptors created by schema 6
        # retain their content-addressed signatures during migration.
        if self.entity_displacement is not None:
            payload["entity_displacement"] = [
                int(self.entity_displacement[0]),
                int(self.entity_displacement[1]),
            ]
        if self.global_phase_change:
            payload["global_phase_change"] = True
        # Schema 9 transition fields are likewise omitted at their defaults
        # so descriptors created by schema 8 retain their content-addressed
        # signatures during migration.
        if self.target_appearance:
            payload["target_appearance"] = self.target_appearance
        if self.entity_removed:
            payload["entity_removed"] = True
        if self.removal_transit_cells:
            payload["removal_transit_cells"] = [
                [int(column), int(row)]
                for column, row in self.removal_transit_cells
            ]
        return payload

    @classmethod
    def from_dict(
        cls, payload: Dict[str, Any]
    ) -> "BehaviorOutcomeDescriptor":
        displacement = payload.get("player_displacement")
        entity_displacement = payload.get("entity_displacement")
        return cls(
            factual_source_relation=str(
                payload["factual_source_relation"]
            ),
            control_source_relation=str(
                payload["control_source_relation"]
            ),
            factual_control_relation=str(
                payload["factual_control_relation"]
            ),
            relative_effect_cells=tuple(
                sorted(
                    (int(cell[0]), int(cell[1]))
                    for cell in payload.get("relative_effect_cells") or ()
                )
            ),
            player_displacement=(
                None
                if displacement is None
                else (int(displacement[0]), int(displacement[1]))
            ),
            terminal_visual_change=bool(
                payload.get("terminal_visual_change", False)
            ),
            entity_displacement=(
                None
                if entity_displacement is None
                else (
                    int(entity_displacement[0]),
                    int(entity_displacement[1]),
                )
            ),
            global_phase_change=bool(
                payload.get("global_phase_change", False)
            ),
            target_appearance=str(
                payload.get("target_appearance") or ""
            ),
            entity_removed=bool(payload.get("entity_removed", False)),
            removal_transit_cells=tuple(
                sorted(
                    (int(cell[0]), int(cell[1]))
                    for cell in payload.get("removal_transit_cells") or ()
                )
            ),
        )

    @property
    def signature(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    @property
    def intervention_inert(self) -> bool:
        """Whether an intervention produced no measured control contrast."""

        return bool(
            self.factual_control_relation == "same"
            and not self.relative_effect_cells
            and self.player_displacement in (None, (0, 0))
            and self.entity_displacement in (None, (0, 0))
            and not self.terminal_visual_change
            and not self.global_phase_change
        )

    @property
    def controlled_movement(self) -> bool:
        return bool(
            self.player_displacement is not None
            and self.player_displacement != (0, 0)
        )

    @property
    def local_visual_change(self) -> bool:
        return bool(
            self.factual_control_relation != "same"
            or self.relative_effect_cells
        )

    @property
    def controlled_appearance_transition(self) -> bool:
        """Whether the factual patch changed beyond its matched control.

        A persistence-verified removal also changes the factual patch, but
        it leaves no appearance at the locus, so it is excluded here and
        reported through :attr:`controlled_entity_removal` instead.
        """

        return bool(
            not self.entity_removed
            and self.factual_control_relation != "same"
            and self.control_source_relation in ("same", "near")
            and self.factual_source_relation == "different"
        )

    @property
    def controlled_entity_displacement(self) -> bool:
        return bool(
            self.entity_displacement is not None
            and self.entity_displacement != (0, 0)
        )

    @property
    def controlled_entity_removal(self) -> bool:
        """Whether a persistence-verified local removal beat its control.

        The tracked appearance ceased to exist around its anchor for the
        verified neutral horizon while the matched control retained it, so
        the disappearance is attributable to the intervention rather than
        to occlusion or an autonomous departure.
        """

        return bool(
            self.entity_removed
            and self.control_source_relation in ("same", "near")
            and self.factual_control_relation != "same"
        )

    @property
    def controlled_entity_expulsion(self) -> bool:
        """A controlled removal with measured transit evidence."""

        return bool(
            self.controlled_entity_removal and self.removal_transit_cells
        )

    @property
    def manipulation_effect(self) -> bool:
        """A label-free state change useful to manipulation planning."""

        return bool(
            self.controlled_appearance_transition
            or self.controlled_entity_displacement
            or self.controlled_entity_removal
            or self.global_phase_change
        )

    @property
    def predictive_class(self) -> Tuple[Any, ...]:
        """Coarse semantics used to merge animation variants safely."""

        return (
            self.intervention_inert,
            self.controlled_movement,
            self.local_visual_change,
            self.controlled_appearance_transition,
            self.entity_displacement,
            self.global_phase_change,
            self.terminal_visual_change,
            self.entity_removed,
            bool(self.removal_transit_cells),
            self.target_appearance,
        )

    @property
    def transition_kind(self) -> str:
        """Name the measured outcome category, never a game concept.

        The named kinds cover control-attributed outcomes plus the inert
        no-effect contrast.  Outcomes reproduced by the matched control,
        such as an autonomous transition denied controller credit, report
        ``other`` so they cannot masquerade as manipulations.
        """

        if self.controlled_entity_expulsion:
            return "expulsion"
        if self.controlled_entity_removal:
            return "removal"
        if self.controlled_appearance_transition:
            return "transformation"
        if self.controlled_entity_displacement:
            return "displacement"
        if self.entity_removed:
            # An autonomous departure reproduced by the matched control is
            # still a removal of the appearance, so it must not be filed as
            # a blocked or no-effect intervention.
            return "other"
        if self.intervention_inert:
            return "no_effect"
        return "other"

    @property
    def transition(self) -> BehaviorTransition:
        """Explicit anchor-relative transition implied by this outcome.

        Cells are relative to the observed appearance's anchor, so the
        transition transfers under translation.  A removal or expulsion has
        no destination cell or displacement because the appearance ceased
        to exist locally.
        """

        kind = self.transition_kind
        source_cell: Optional[RelativeCell] = (0, 0)
        destination_cell: Optional[RelativeCell] = None
        displacement: Optional[RelativeCell] = None
        if kind in ("displacement", "transformation", "no_effect"):
            displacement = (
                (0, 0)
                if self.entity_displacement is None
                else self.entity_displacement
            )
            destination_cell = displacement
        elif kind == "other":
            source_cell = None
        return BehaviorTransition(
            kind=kind,
            outcome_signature=self.signature,
            source_cell=source_cell,
            destination_cell=destination_cell,
            displacement=displacement,
            target_appearance=self.target_appearance,
            transit_cells=self.removal_transit_cells,
        )

    @property
    def autonomous_visual_change(self) -> bool:
        """Whether a passive outcome changed or translated the appearance."""

        return bool(
            self.factual_source_relation != "same"
            or any(
                cell != (0, 0) for cell in self.relative_effect_cells
            )
        )

    def local_visual_change_for(self, autonomous: bool) -> bool:
        return (
            self.autonomous_visual_change
            if autonomous
            else self.local_visual_change
        )

    def measured_effect_for(self, autonomous: bool) -> bool:
        return bool(
            self.local_visual_change_for(autonomous)
            or (not autonomous and self.controlled_movement)
            or (not autonomous and self.controlled_entity_displacement)
            or self.global_phase_change
            or self.terminal_visual_change
        )

    @property
    def measured_effect(self) -> bool:
        return bool(
            self.controlled_movement
            or self.controlled_entity_displacement
            or self.local_visual_change
            or self.global_phase_change
            or self.terminal_visual_change
        )


@dataclass(frozen=True)
class BehaviorPrediction:
    """Posterior prediction for an anonymous type under one intervention."""

    type_id: Optional[int]
    appearance_distance: float
    outcome_signature: Optional[str]
    outcome_probability: float
    hazardous_probability: float
    samples: int
    confidence: float
    entropy: float
    known: bool
    context_matched: bool
    contexts_observed: int
    causal_hazardous_probability: float
    causal_hazard_samples: int
    causal_hazard_known: bool
    outcome_descriptor: Optional[BehaviorOutcomeDescriptor]
    semantic_samples: int
    semantic_coverage: float
    inert_probability: float
    inert_confidence: float
    measured_effect_probability: float
    controlled_movement_probability: float
    local_visual_change_probability: float
    terminal_visual_change_probability: float
    entity_displacement_probability: float = 0.0
    appearance_transition_probability: float = 0.0
    entity_removal_probability: float = 0.0
    entity_expulsion_probability: float = 0.0
    global_phase_change_probability: float = 0.0
    manipulation_probability: float = 0.0
    predictive_family_id: Optional[int] = None
    predictive_family_size: int = 1
    predictive_family_pooled: bool = False

    @property
    def novelty(self) -> float:
        return 1.0 / math.sqrt(self.samples + 1)


@dataclass(frozen=True)
class BehaviorObservation:
    """Result of adding one controlled pixel observation to the model."""

    type_id: int
    created_type: bool
    accepted: bool
    outcome_signature: str
    surprise: float
    prediction_before: BehaviorPrediction
    prediction_after: BehaviorPrediction


@dataclass
class _RuleStats:
    outcomes: CounterType[str]
    hazardous: int = 0
    samples: int = 0
    contexts: set[str] = field(default_factory=set)
    causal_hazardous: int = 0
    causal_hazard_samples: int = 0


class AnonymousEntityBehaviorModel:
    """Persistent pixel-only appearance types and conditional behavior rules.

    Types are anonymous nearest-prototype clusters over pooled RGB patches.
    Rules are empirical outcome distributions conditioned on hardware action,
    duration, optional anonymous scene context, and whether the observation was
    passive.  No sprite name, object definition, collision rule, or level label
    is represented here.

    The model intentionally stores distributions rather than a single rule.
    Identical-looking entities can therefore exhibit context-dependent behavior,
    and contradictory evidence reduces confidence instead of being discarded.
    """

    SCHEMA_VERSION = 9
    _UNCONDITIONAL_CONTEXT = "*"

    def __init__(
        self,
        appearance_match_threshold: float = 0.08,
        minimum_prediction_samples: int = 2,
    ) -> None:
        if appearance_match_threshold < 0.0:
            raise ValueError("appearance match threshold must be non-negative")
        if minimum_prediction_samples <= 0:
            raise ValueError("minimum prediction samples must be positive")
        self.appearance_match_threshold = appearance_match_threshold
        self.minimum_prediction_samples = minimum_prediction_samples
        self._type_means: list[list[float]] = []
        self._type_counts: list[int] = []
        self._rules: Dict[Tuple[int, str, int, bool, str], _RuleStats] = {}
        self._outcome_descriptors: Dict[
            str, BehaviorOutcomeDescriptor
        ] = {}
        self._evidence_ids: set[str] = set()
        self._causal_hazard_evidence_ids: set[str] = set()
        self._unconditional_rule_keys_by_type: Dict[
            int, set[Tuple[int, str, int, bool, str]]
        ] = {}
        self._predictive_profile_cache: Dict[
            int, Optional[Tuple[Tuple[Any, ...], ...]]
        ] = {}
        self._predictive_family_cache: Dict[
            int, Tuple[Optional[int], Tuple[int, ...]]
        ] = {}

    @property
    def type_count(self) -> int:
        return len(self._type_means)

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def observation_count(self) -> int:
        return sum(
            rule.samples
            for key, rule in self._rules.items()
            if key[4] == self._UNCONDITIONAL_CONTEXT
        )

    @property
    def causal_hazard_observation_count(self) -> int:
        return sum(
            rule.causal_hazard_samples
            for key, rule in self._rules.items()
            if key[4] == self._UNCONDITIONAL_CONTEXT
        )

    @property
    def outcome_descriptor_count(self) -> int:
        return len(self._outcome_descriptors)

    def register_outcome_descriptor(
        self,
        outcome_signature: str,
        descriptor: BehaviorOutcomeDescriptor,
    ) -> bool:
        """Attach auditable semantics without changing empirical counts."""

        if descriptor.signature != outcome_signature:
            raise ValueError(
                "behavior outcome descriptor does not match its signature"
            )
        previous = self._outcome_descriptors.get(outcome_signature)
        if previous is not None and previous != descriptor:
            raise ValueError("conflicting behavior outcome descriptor")
        if previous is not None:
            return False
        self._outcome_descriptors[outcome_signature] = descriptor
        # Descriptors supply the semantics used by predictive families.  A
        # descriptor may be registered after empirical counts were loaded, so
        # conservatively invalidate every cached profile.
        self._predictive_profile_cache.clear()
        self._predictive_family_cache.clear()
        return True

    def transition_for(
        self, outcome_signature: str
    ) -> Optional[BehaviorTransition]:
        """Explicit transition for one outcome, or None when undescribed."""

        descriptor = self._outcome_descriptors.get(outcome_signature)
        return None if descriptor is None else descriptor.transition

    def transitions(self) -> Tuple[BehaviorTransition, ...]:
        """Deterministically ordered transitions for every known outcome.

        Purely derived from registered descriptors: enumerating transitions
        never creates a type, descriptor, or rule, so frozen evaluations can
        query explicit transitions without perturbing the checkpoint digest.
        Legacy descriptors that already imply a displacement or
        transformation surface it here explicitly after migration.
        """

        return tuple(
            self._outcome_descriptors[signature].transition
            for signature in sorted(self._outcome_descriptors)
        )

    @staticmethod
    def _feature_distance(
        first: Sequence[float], second: Sequence[float]
    ) -> float:
        if len(first) != len(second) or not first:
            return 1.0
        maximum = max(1.0, max((*first, *second)))
        return sum(abs(a - b) for a, b in zip(first, second)) / (
            maximum * len(first)
        )

    @staticmethod
    def appearance_fingerprint(feature: Sequence[int]) -> str:
        payload = ",".join(str(int(value)) for value in feature)
        return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]

    @classmethod
    def context_signature(
        cls, features: Iterable[Sequence[int]]
    ) -> str:
        """Hash an order-independent anonymous appearance multiset."""

        counts = Counter(cls.appearance_fingerprint(feature) for feature in features)
        payload = ";".join(
            f"{fingerprint}:{count}"
            for fingerprint, count in sorted(counts.items())
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]

    @staticmethod
    def relational_context_signature(
        anchor: RelativeCell,
        controlled_cells: Iterable[RelativeCell],
        fallback_signature: str = "",
    ) -> str:
        """Describe only the translation-invariant controlled-entity relation.

        The controlled entity is whichever visual patch has already been
        localized from action correlation.  No object or mechanic label is
        represented.  Coarse distance bins deliberately pool nearby layouts
        while retaining evidence that an appearance can behave differently
        when the controllable patch is near, aligned, or far away.
        """

        cells = tuple(
            sorted(
                (int(column), int(row))
                for column, row in controlled_cells
            )
        )
        if not cells:
            return (
                f"anonymous-scene:{fallback_signature}"
                if fallback_signature
                else "controlled-relative:unknown"
            )
        target = min(
            cells,
            key=lambda cell: (
                abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1]),
                cell[0],
                cell[1],
            ),
        )
        dx = target[0] - anchor[0]
        dy = target[1] - anchor[1]
        distance = abs(dx) + abs(dy)
        if distance <= 2:
            distance_bin = str(distance)
        elif distance <= 4:
            distance_bin = "3-4"
        elif distance <= 8:
            distance_bin = "5-8"
        else:
            distance_bin = "9+"

        def sign(value: int) -> str:
            if value < 0:
                return "negative"
            if value > 0:
                return "positive"
            return "zero"

        alignment = (
            "overlap"
            if dx == 0 and dy == 0
            else "column"
            if dx == 0
            else "row"
            if dy == 0
            else "diagonal"
        )
        return (
            "controlled-relative:v1:"
            f"distance={distance_bin}:alignment={alignment}:"
            f"dx={sign(dx)}:dy={sign(dy)}"
        )

    @classmethod
    def phase_signature(
        cls, features: Iterable[Sequence[int]]
    ) -> str:
        """Return a coarse translation-invariant global visual mode.

        Features are deliberately requantized before counting. This makes the
        signature insensitive to small sprite-animation changes while still
        allowing disappearance, transformation, and HUD-like global changes
        to define a new empirical phase. No region or object has a supplied
        meaning.
        """

        counts: CounterType[str] = Counter()
        for feature in features:
            coarse = tuple(int(value) // 4 for value in feature)
            counts[cls.appearance_fingerprint(coarse)] += 1
        payload = ";".join(
            f"{fingerprint}:{count}"
            for fingerprint, count in sorted(counts.items())
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]

    @classmethod
    def neighborhood_signature(
        cls,
        anchor: RelativeCell,
        cell_features: Dict[RelativeCell, Sequence[int]],
    ) -> str:
        """Describe local occupancy without assigning object identities.

        Each neighboring cell contributes its relative offset, coarsened
        appearance, and within-frame frequency. This exposes destination
        context for effects such as pushing while remaining pixel-derived.
        """

        fingerprints = {
            cell: cls.appearance_fingerprint(
                tuple(int(value) // 2 for value in feature)
            )
            for cell, feature in cell_features.items()
        }
        counts = Counter(fingerprints.values())
        tokens = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                cell = (anchor[0] + dx, anchor[1] + dy)
                fingerprint = fingerprints.get(cell)
                if fingerprint is None:
                    tokens.append(f"{dx},{dy}=edge")
                    continue
                frequency = counts[fingerprint]
                frequency_bin = (
                    "one"
                    if frequency == 1
                    else "few"
                    if frequency <= 4
                    else "many"
                )
                tokens.append(
                    f"{dx},{dy}={fingerprint}:{frequency_bin}"
                )
        return hashlib.sha256(
            ";".join(tokens).encode("ascii")
        ).hexdigest()[:16]

    @classmethod
    def factored_context_signature(
        cls,
        relation_signature: str,
        neighborhood_signature: str,
        phase_signature: str,
    ) -> str:
        """Compose context dimensions with deterministic fallback support."""

        relation = relation_signature or "controlled-relative:unknown"
        neighborhood = neighborhood_signature or "unknown"
        phase = phase_signature or "unknown"
        return (
            f"factored:v1|phase={phase}|neighborhood={neighborhood}|"
            f"relation={relation}"
        )

    @staticmethod
    def _context_hierarchy(context_signature: str) -> Tuple[str, ...]:
        """Order exact factored context before reusable partial contexts."""

        if not context_signature.startswith("factored:v1|"):
            return (context_signature,) if context_signature else ()
        values: Dict[str, str] = {}
        for component in context_signature.split("|")[1:]:
            key, separator, value = component.partition("=")
            if separator:
                values[key] = value
        relation = values.get("relation", "controlled-relative:unknown")
        phase = values.get("phase", "unknown")
        neighborhood = values.get("neighborhood", "unknown")
        return (
            context_signature,
            f"factored:v1|phase={phase}|relation={relation}",
            f"factored:v1|neighborhood={neighborhood}|relation={relation}",
            relation,
        )

    @classmethod
    def effect_descriptor(
        cls,
        source_feature: Sequence[int],
        factual_feature: Sequence[int],
        control_feature: Sequence[int],
        relative_effect_cells: Sequence[RelativeCell] = (),
        player_displacement: Optional[RelativeCell] = None,
        terminal_visual_change: bool = False,
        entity_displacement: Optional[RelativeCell] = None,
        global_phase_change: bool = False,
        transformation_target_feature: Optional[Sequence[int]] = None,
        entity_removed: bool = False,
        removal_persistence_verified: bool = False,
        removal_transit_cells: Sequence[RelativeCell] = (),
    ) -> BehaviorOutcomeDescriptor:
        """Describe a position-invariant outcome from pixel-derived facts.

        ``entity_removed`` may only be asserted after the persistence-horizon
        machinery confirmed the disappearance outlasted its neutral horizon;
        an unverified absence is indistinguishable from occlusion and is
        rejected here rather than stored as a removal outcome.
        """

        def appearance_relation(
            first: Sequence[int], second: Sequence[int]
        ) -> str:
            distance = cls._feature_distance(first, second)
            if distance <= 0.08:
                return "same"
            if distance <= 0.2:
                return "near"
            return "different"

        if entity_removed and not removal_persistence_verified:
            raise ValueError(
                "entity removal requires persistence-horizon verification"
            )
        factual_source_relation = appearance_relation(
            factual_feature, source_feature
        )
        target_appearance = ""
        if (
            transformation_target_feature is not None
            and factual_source_relation == "different"
        ):
            # Requantize the post-transition appearance so animation phases
            # of the same target do not fragment the outcome signature.  The
            # fingerprint stays anonymous: it identifies pixels, not a named
            # object.  Animation-tolerant relations drop the target instead
            # of recording a false transformation.
            target_appearance = cls.appearance_fingerprint(
                tuple(
                    int(value) // 4
                    for value in transformation_target_feature
                )
            )
        return BehaviorOutcomeDescriptor(
            # The source appearance identifies the anonymous type and does not
            # belong in the outcome.  Storing its exact fingerprint here made
            # harmless animation phases look like different mechanics.
            factual_source_relation=factual_source_relation,
            control_source_relation=appearance_relation(
                control_feature, source_feature
            ),
            factual_control_relation=appearance_relation(
                factual_feature, control_feature
            ),
            relative_effect_cells=tuple(
                sorted(
                    (int(column), int(row))
                    for column, row in relative_effect_cells
                )
            ),
            player_displacement=(
                None
                if player_displacement is None
                else (
                    int(player_displacement[0]),
                    int(player_displacement[1]),
                )
            ),
            terminal_visual_change=bool(terminal_visual_change),
            entity_displacement=(
                None
                if entity_displacement is None
                else (
                    int(entity_displacement[0]),
                    int(entity_displacement[1]),
                )
            ),
            global_phase_change=bool(global_phase_change),
            target_appearance=target_appearance,
            entity_removed=bool(entity_removed),
            removal_transit_cells=tuple(
                sorted(
                    (int(column), int(row))
                    for column, row in removal_transit_cells
                )
            ),
        )

    @classmethod
    def effect_signature(
        cls,
        source_feature: Sequence[int],
        factual_feature: Sequence[int],
        control_feature: Sequence[int],
        relative_effect_cells: Sequence[RelativeCell] = (),
        player_displacement: Optional[RelativeCell] = None,
        terminal_visual_change: bool = False,
        entity_displacement: Optional[RelativeCell] = None,
        global_phase_change: bool = False,
        transformation_target_feature: Optional[Sequence[int]] = None,
        entity_removed: bool = False,
        removal_persistence_verified: bool = False,
        removal_transit_cells: Sequence[RelativeCell] = (),
    ) -> str:
        """Create a stable key for a pixel-derived outcome descriptor."""

        return cls.effect_descriptor(
            source_feature,
            factual_feature,
            control_feature,
            relative_effect_cells=relative_effect_cells,
            player_displacement=player_displacement,
            terminal_visual_change=terminal_visual_change,
            entity_displacement=entity_displacement,
            global_phase_change=global_phase_change,
            transformation_target_feature=transformation_target_feature,
            entity_removed=entity_removed,
            removal_persistence_verified=removal_persistence_verified,
            removal_transit_cells=removal_transit_cells,
        ).signature

    def _nearest_type(
        self, feature: Sequence[int]
    ) -> Tuple[Optional[int], float]:
        if not self._type_means:
            return None, 1.0
        distances = [
            self._feature_distance(feature, prototype)
            for prototype in self._type_means
        ]
        type_id = min(range(len(distances)), key=distances.__getitem__)
        return type_id, distances[type_id]

    def classify(
        self, feature: Sequence[int]
    ) -> Tuple[Optional[int], float]:
        type_id, distance = self._nearest_type(feature)
        if type_id is None or distance > self.appearance_match_threshold:
            return None, distance
        return type_id, distance

    def _assign(self, feature: AppearanceFeature) -> Tuple[int, float, bool]:
        if self._type_means and len(feature) != len(self._type_means[0]):
            raise ValueError("anonymous type feature dimensions differ")
        type_id, distance = self.classify(feature)
        if type_id is None:
            type_id = len(self._type_means)
            self._type_means.append([float(value) for value in feature])
            self._type_counts.append(0)
            return type_id, distance, True
        return type_id, distance, False

    @staticmethod
    def _rule_key(
        type_id: int,
        action: Action,
        duration: int,
        autonomous: bool,
        context_signature: str,
    ) -> Tuple[int, str, int, bool, str]:
        if duration <= 0:
            raise ValueError("behavior duration must be positive")
        return (
            int(type_id),
            action.value,
            int(duration),
            bool(autonomous),
            context_signature,
        )

    def _select_rule(
        self,
        type_id: int,
        action: Action,
        duration: int,
        autonomous: bool,
        context_signature: str,
    ) -> Tuple[Optional[_RuleStats], bool]:
        for condition in self._context_hierarchy(context_signature):
            contextual = self._rules.get(
                self._rule_key(
                    type_id,
                    action,
                    duration,
                    autonomous,
                    condition,
                )
            )
            if (
                contextual is not None
                and contextual.samples >= self.minimum_prediction_samples
            ):
                return contextual, True
        return (
            self._rules.get(
                self._rule_key(
                    type_id,
                    action,
                    duration,
                    autonomous,
                    self._UNCONDITIONAL_CONTEXT,
                )
            ),
            False,
        )

    def _predictive_profile(
        self, type_id: int
    ) -> Optional[Tuple[Tuple[Any, ...], ...]]:
        """Return behavior semantics independent of animation appearance."""

        if type_id in self._predictive_profile_cache:
            return self._predictive_profile_cache[type_id]
        rows = []
        keys = self._unconditional_rule_keys_by_type.get(type_id, set())
        for key in sorted(keys):
            rule = self._rules[key]
            row_type, action, duration, autonomous, context = key
            if (
                row_type != type_id
                or context != self._UNCONDITIONAL_CONTEXT
                or rule.samples <= 0
            ):
                continue
            semantic_counts: CounterType[Tuple[Any, ...]] = Counter()
            for signature, count in rule.outcomes.items():
                descriptor = self._outcome_descriptors.get(signature)
                if descriptor is not None:
                    semantic_counts[descriptor.predictive_class] += count
            if not semantic_counts:
                continue
            dominant = max(
                semantic_counts.items(),
                key=lambda item: (item[1], repr(item[0])),
            )[0]
            rows.append((action, duration, autonomous, *dominant))
        profile = tuple(rows) if rows else None
        self._predictive_profile_cache[type_id] = profile
        return profile

    def predictive_family(
        self, type_id: Optional[int]
    ) -> Tuple[Optional[int], Tuple[int, ...]]:
        """Group types only after their measured semantics agree exactly."""

        if type_id is None or not 0 <= type_id < self.type_count:
            return None, ()
        cached = self._predictive_family_cache.get(type_id)
        if cached is not None:
            return cached
        profile = self._predictive_profile(type_id)
        if profile is None:
            result = (type_id, (type_id,))
            self._predictive_family_cache[type_id] = result
            return result
        members = tuple(
            candidate
            for candidate in range(self.type_count)
            if self._predictive_profile(candidate) == profile
        )
        result = (min(members), members)
        for member in members:
            self._predictive_family_cache[member] = result
        return result

    @staticmethod
    def _merge_rules(rules: Sequence[_RuleStats]) -> _RuleStats:
        merged = _RuleStats(Counter())
        for rule in rules:
            merged.outcomes.update(rule.outcomes)
            merged.hazardous += rule.hazardous
            merged.samples += rule.samples
            merged.contexts.update(rule.contexts)
            merged.causal_hazardous += rule.causal_hazardous
            merged.causal_hazard_samples += rule.causal_hazard_samples
        return merged

    def _select_rule_with_family(
        self,
        type_id: int,
        action: Action,
        duration: int,
        autonomous: bool,
        context_signature: str,
    ) -> Tuple[Optional[_RuleStats], bool, int, int, bool]:
        own_rule, own_contextual = self._select_rule(
            type_id,
            action,
            duration,
            autonomous,
            context_signature,
        )
        family_id, members = self.predictive_family(type_id)
        assert family_id is not None
        if len(members) <= 1 or (
            own_rule is not None
            and own_rule.samples >= self.minimum_prediction_samples
        ):
            return own_rule, own_contextual, family_id, len(members), False
        conditions = (
            *self._context_hierarchy(context_signature),
            self._UNCONDITIONAL_CONTEXT,
        )
        for condition in conditions:
            rules = [
                rule
                for member in members
                if (
                    rule := self._rules.get(
                        self._rule_key(
                            member,
                            action,
                            duration,
                            autonomous,
                            condition,
                        )
                    )
                )
                is not None
            ]
            if not rules:
                continue
            merged = self._merge_rules(rules)
            if merged.samples >= self.minimum_prediction_samples:
                return (
                    merged,
                    condition != self._UNCONDITIONAL_CONTEXT,
                    family_id,
                    len(members),
                    True,
                )
        return own_rule, own_contextual, family_id, len(members), False

    def _prediction_from_rule(
        self,
        type_id: Optional[int],
        distance: float,
        rule: Optional[_RuleStats],
        minimum_prediction_samples: int,
        context_matched: bool,
        autonomous: bool,
        predictive_family_id: Optional[int] = None,
        predictive_family_size: int = 1,
        predictive_family_pooled: bool = False,
    ) -> BehaviorPrediction:
        if rule is None or rule.samples <= 0:
            return BehaviorPrediction(
                type_id=type_id,
                appearance_distance=distance,
                outcome_signature=None,
                outcome_probability=0.0,
                hazardous_probability=0.0,
                samples=0,
                confidence=0.0,
                entropy=0.0,
                known=False,
                context_matched=context_matched,
                contexts_observed=0,
                causal_hazardous_probability=0.0,
                causal_hazard_samples=0,
                causal_hazard_known=False,
                outcome_descriptor=None,
                semantic_samples=0,
                semantic_coverage=0.0,
                inert_probability=0.0,
                inert_confidence=0.0,
                measured_effect_probability=0.0,
                controlled_movement_probability=0.0,
                local_visual_change_probability=0.0,
                terminal_visual_change_probability=0.0,
                predictive_family_id=predictive_family_id,
                predictive_family_size=predictive_family_size,
                predictive_family_pooled=predictive_family_pooled,
            )
        outcome, count = max(
            rule.outcomes.items(), key=lambda item: (item[1], item[0])
        )
        probability = count / rule.samples
        entropy = -sum(
            (value / rule.samples) * math.log(value / rule.samples)
            for value in rule.outcomes.values()
            if value > 0
        )
        evidence = 1.0 - math.exp(
            -rule.samples / minimum_prediction_samples
        )
        described_outcomes = {
            signature: self._outcome_descriptors[signature]
            for signature in rule.outcomes
            if signature in self._outcome_descriptors
        }
        semantic_samples = sum(
            rule.outcomes[signature] for signature in described_outcomes
        )

        def semantic_probability(
            predicate: Callable[[BehaviorOutcomeDescriptor], bool]
        ) -> float:
            return (
                sum(
                    rule.outcomes[signature]
                    for signature, descriptor in described_outcomes.items()
                    if bool(predicate(descriptor))
                )
                / rule.samples
            )

        inert_probability = (
            0.0
            if autonomous
            else semantic_probability(
                lambda descriptor: descriptor.intervention_inert
            )
        )
        return BehaviorPrediction(
            type_id=type_id,
            appearance_distance=distance,
            outcome_signature=outcome,
            outcome_probability=probability,
            hazardous_probability=rule.hazardous / rule.samples,
            samples=rule.samples,
            confidence=probability * evidence,
            entropy=entropy,
            known=rule.samples >= minimum_prediction_samples,
            context_matched=context_matched,
            contexts_observed=len(rule.contexts),
            causal_hazardous_probability=(
                rule.causal_hazardous / rule.causal_hazard_samples
                if rule.causal_hazard_samples > 0
                else 0.0
            ),
            causal_hazard_samples=rule.causal_hazard_samples,
            causal_hazard_known=(
                rule.causal_hazard_samples >= minimum_prediction_samples
            ),
            outcome_descriptor=self._outcome_descriptors.get(outcome),
            semantic_samples=semantic_samples,
            semantic_coverage=semantic_samples / rule.samples,
            inert_probability=inert_probability,
            inert_confidence=inert_probability * evidence,
            measured_effect_probability=semantic_probability(
                lambda descriptor: descriptor.measured_effect_for(
                    autonomous
                )
            ),
            controlled_movement_probability=semantic_probability(
                lambda descriptor: (
                    not autonomous and descriptor.controlled_movement
                )
            ),
            local_visual_change_probability=semantic_probability(
                lambda descriptor: descriptor.local_visual_change_for(
                    autonomous
                )
            ),
            terminal_visual_change_probability=semantic_probability(
                lambda descriptor: descriptor.terminal_visual_change
            ),
            entity_displacement_probability=semantic_probability(
                lambda descriptor: (
                    not autonomous
                    and descriptor.controlled_entity_displacement
                )
            ),
            appearance_transition_probability=semantic_probability(
                lambda descriptor: (
                    not autonomous
                    and descriptor.controlled_appearance_transition
                )
            ),
            entity_removal_probability=semantic_probability(
                lambda descriptor: (
                    not autonomous
                    and descriptor.controlled_entity_removal
                )
            ),
            entity_expulsion_probability=semantic_probability(
                lambda descriptor: (
                    not autonomous
                    and descriptor.controlled_entity_expulsion
                )
            ),
            global_phase_change_probability=semantic_probability(
                lambda descriptor: descriptor.global_phase_change
            ),
            manipulation_probability=semantic_probability(
                lambda descriptor: (
                    not autonomous and descriptor.manipulation_effect
                )
            ),
            predictive_family_id=predictive_family_id,
            predictive_family_size=predictive_family_size,
            predictive_family_pooled=predictive_family_pooled,
        )

    def predict(
        self,
        feature: Sequence[int],
        action: Action,
        duration: int,
        context_signature: str = "",
        autonomous: bool = False,
    ) -> BehaviorPrediction:
        type_id, distance = self.classify(feature)
        if type_id is None:
            return self._prediction_from_rule(
                None,
                distance,
                None,
                self.minimum_prediction_samples,
                False,
                autonomous,
            )
        (
            rule,
            contextual,
            family_id,
            family_size,
            family_pooled,
        ) = self._select_rule_with_family(
            type_id,
            action,
            duration,
            autonomous,
            context_signature,
        )
        return self._prediction_from_rule(
            type_id,
            distance,
            rule,
            self.minimum_prediction_samples,
            contextual,
            autonomous,
            predictive_family_id=family_id,
            predictive_family_size=family_size,
            predictive_family_pooled=family_pooled,
        )

    def exact_context_samples(
        self,
        feature: Sequence[int],
        action: Action,
        duration: int,
        context_signature: str,
        autonomous: bool = False,
    ) -> int:
        """Return evidence for this exact context, excluding fallbacks."""

        type_id, _distance = self.classify(feature)
        if type_id is None or not context_signature:
            return 0
        rule = self._rules.get(
            self._rule_key(
                type_id,
                action,
                duration,
                autonomous,
                context_signature,
            )
        )
        return 0 if rule is None else rule.samples

    def outcome_probability(
        self,
        feature: Sequence[int],
        action: Action,
        duration: int,
        outcome_signature: str,
        context_signature: str = "",
        autonomous: bool = False,
    ) -> float:
        type_id, _distance = self.classify(feature)
        if type_id is None:
            return 0.0
        rule, _contextual, _family_id, _family_size, _pooled = (
            self._select_rule_with_family(
                type_id,
                action,
                duration,
                autonomous,
                context_signature,
            )
        )
        if rule is None or rule.samples <= 0:
            return 0.0
        return rule.outcomes[outcome_signature] / rule.samples

    def observe(
        self,
        feature: Sequence[int],
        action: Action,
        duration: int,
        outcome_signature: str,
        context_signature: str = "",
        hazardous: bool = False,
        autonomous: bool = False,
        evidence_id: str = "",
        causal_hazard_evidence: bool = False,
        outcome_descriptor: Optional[BehaviorOutcomeDescriptor] = None,
    ) -> BehaviorObservation:
        if not feature:
            raise ValueError("behavior appearance feature must be non-empty")
        if not outcome_signature:
            raise ValueError("behavior outcome signature must be non-empty")
        if (
            outcome_descriptor is not None
            and outcome_descriptor.signature != outcome_signature
        ):
            raise ValueError(
                "behavior outcome descriptor does not match its signature"
            )
        appearance = tuple(int(value) for value in feature)
        prediction_before = self.predict(
            appearance,
            action,
            duration,
            context_signature,
            autonomous,
        )
        probability_before = self.outcome_probability(
            appearance,
            action,
            duration,
            outcome_signature,
            context_signature,
            autonomous,
        )
        if evidence_id and evidence_id in self._evidence_ids:
            if prediction_before.type_id is None:
                raise ValueError("duplicate behavior evidence has no known type")
            return BehaviorObservation(
                type_id=prediction_before.type_id,
                created_type=False,
                accepted=False,
                outcome_signature=outcome_signature,
                surprise=-math.log(
                    max(
                        1e-9,
                        self.outcome_probability(
                            appearance,
                            action,
                            duration,
                            outcome_signature,
                            context_signature,
                            autonomous,
                        ),
                    )
                ),
                prediction_before=prediction_before,
                prediction_after=prediction_before,
            )
        if outcome_descriptor is not None:
            self.register_outcome_descriptor(
                outcome_signature, outcome_descriptor
            )
        type_id, _distance, created = self._assign(appearance)
        previous_count = self._type_counts[type_id]
        next_count = previous_count + 1
        prototype = self._type_means[type_id]
        for index, value in enumerate(appearance):
            prototype[index] += (value - prototype[index]) / next_count
        self._type_counts[type_id] = next_count

        contexts = [self._UNCONDITIONAL_CONTEXT]
        contexts.extend(self._context_hierarchy(context_signature))
        contexts = list(dict.fromkeys(contexts))
        for condition in contexts:
            key = self._rule_key(
                type_id, action, duration, autonomous, condition
            )
            rule = self._rules.setdefault(key, _RuleStats(Counter()))
            rule.outcomes[outcome_signature] += 1
            rule.samples += 1
            rule.hazardous += int(hazardous)
            if causal_hazard_evidence:
                rule.causal_hazard_samples += 1
                rule.causal_hazardous += int(hazardous)
            if context_signature:
                rule.contexts.add(context_signature)
            if condition == self._UNCONDITIONAL_CONTEXT:
                self._unconditional_rule_keys_by_type.setdefault(
                    type_id, set()
                ).add(key)
        # Only this type's empirical semantics changed.  Cached profiles for
        # every other type remain valid, while family membership must be
        # regrouped around the updated profile.
        self._predictive_profile_cache.pop(type_id, None)
        self._predictive_family_cache.clear()
        if evidence_id:
            self._evidence_ids.add(evidence_id)
            if causal_hazard_evidence:
                self._causal_hazard_evidence_ids.add(evidence_id)

        prediction_after = self.predict(
            appearance,
            action,
            duration,
            context_signature,
            autonomous,
        )
        surprise = -math.log(max(1e-9, probability_before))
        return BehaviorObservation(
            type_id=type_id,
            created_type=created,
            accepted=True,
            outcome_signature=outcome_signature,
            surprise=surprise,
            prediction_before=prediction_before,
            prediction_after=prediction_after,
        )

    def backfill_causal_hazard_evidence(
        self,
        type_id: int,
        action: Action,
        duration: int,
        context_signature: str,
        hazardous: bool,
        evidence_id: str,
        autonomous: bool = False,
    ) -> bool:
        """Mark an already-recorded observation as causally attributed.

        Older checkpoints stored terminal correlations but not their evidence
        provenance.  This migration hook accepts only evidence already present
        in the checkpoint and updates a separate hazard posterior without
        changing appearance prototypes, outcome counts, or observation counts.
        """

        if not evidence_id:
            raise ValueError("causal hazard backfill requires an evidence id")
        if evidence_id not in self._evidence_ids:
            raise ValueError(
                "causal hazard backfill evidence is absent from checkpoint"
            )
        if evidence_id in self._causal_hazard_evidence_ids:
            return False
        if not 0 <= type_id < self.type_count:
            raise ValueError("causal hazard backfill references missing type")
        contexts = [self._UNCONDITIONAL_CONTEXT]
        if context_signature:
            contexts.append(context_signature)
        rules = []
        for condition in contexts:
            key = self._rule_key(
                type_id,
                action,
                duration,
                autonomous,
                condition,
            )
            rule = self._rules.get(key)
            if rule is None:
                raise ValueError(
                    "causal hazard backfill references missing rule"
                )
            rules.append(rule)
        for rule in rules:
            rule.causal_hazard_samples += 1
            rule.causal_hazardous += int(hazardous)
        self._causal_hazard_evidence_ids.add(evidence_id)
        return True

    def type_stats(self) -> Tuple[AnonymousTypeStats, ...]:
        return tuple(
            AnonymousTypeStats(
                type_id=type_id,
                observations=self._type_counts[type_id],
                feature=tuple(prototype),
            )
            for type_id, prototype in enumerate(self._type_means)
        )

    def to_dict(self) -> Dict[str, Any]:
        rules = []
        for key, stats in sorted(self._rules.items()):
            type_id, action, duration, autonomous, context = key
            rules.append(
                {
                    "type_id": type_id,
                    "action": action,
                    "duration": duration,
                    "autonomous": autonomous,
                    "context": context,
                    "outcomes": dict(sorted(stats.outcomes.items())),
                    "hazardous": stats.hazardous,
                    "samples": stats.samples,
                    "contexts": sorted(stats.contexts),
                    "causal_hazardous": stats.causal_hazardous,
                    "causal_hazard_samples": stats.causal_hazard_samples,
                }
            )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "appearance_match_threshold": self.appearance_match_threshold,
            "minimum_prediction_samples": self.minimum_prediction_samples,
            "types": [
                {
                    "type_id": stats.type_id,
                    "observations": stats.observations,
                    "feature": list(stats.feature),
                }
                for stats in self.type_stats()
            ],
            "rules": rules,
            "outcome_descriptors": {
                signature: descriptor.to_dict()
                for signature, descriptor in sorted(
                    self._outcome_descriptors.items()
                )
            },
            "evidence_ids": sorted(self._evidence_ids),
            "causal_hazard_evidence_ids": sorted(
                self._causal_hazard_evidence_ids
            ),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AnonymousEntityBehaviorModel":
        schema_version = int(payload.get("schema_version", 0))
        if schema_version not in (3, 4, 5, 6, 7, 8, cls.SCHEMA_VERSION):
            raise ValueError("unsupported anonymous behavior schema")
        model = cls(
            appearance_match_threshold=float(
                payload["appearance_match_threshold"]
            ),
            minimum_prediction_samples=int(
                payload["minimum_prediction_samples"]
            ),
        )
        for expected_id, row in enumerate(payload.get("types") or ()):
            type_id = int(row["type_id"])
            if type_id != expected_id:
                raise ValueError("anonymous type ids must be contiguous")
            feature = [float(value) for value in row["feature"]]
            if not feature:
                raise ValueError("anonymous type feature must be non-empty")
            if model._type_means and len(feature) != len(model._type_means[0]):
                raise ValueError("anonymous type feature dimensions differ")
            model._type_means.append(feature)
            model._type_counts.append(int(row["observations"]))
        outcome_signature_remap: Dict[str, str] = {}
        for signature, descriptor_payload in dict(
            payload.get("outcome_descriptors") or {}
        ).items():
            descriptor = BehaviorOutcomeDescriptor.from_dict(
                dict(descriptor_payload)
            )
            stored_signature = str(signature)
            # Schema 7 treated any one distant stable-cell difference as a
            # room-wide phase change. Schema 8 requires distributed evidence,
            # which cannot be reconstructed from the aggregate checkpoint.
            # Preserve every other measured semantic and empirical count, but
            # remove that unsupported flag and remap its content hash.
            # Schema 8 -> 9 needs no remap: the explicit transition fields
            # are omitted at their defaults, so every schema-8 descriptor
            # keeps its content-addressed signature and its evidence
            # provenance unchanged.
            if schema_version == 7 and descriptor.global_phase_change:
                descriptor = replace(
                    descriptor, global_phase_change=False
                )
            outcome_signature_remap[stored_signature] = descriptor.signature
            model.register_outcome_descriptor(
                descriptor.signature, descriptor
            )
        for row in payload.get("rules") or ():
            type_id = int(row["type_id"])
            if not 0 <= type_id < model.type_count:
                raise ValueError("anonymous behavior references missing type")
            action = Action(str(row["action"]))
            key = model._rule_key(
                type_id,
                action,
                int(row["duration"]),
                bool(row.get("autonomous", False)),
                str(row["context"]),
            )
            outcomes: CounterType[str] = Counter()
            for outcome, count in dict(row["outcomes"]).items():
                stored_outcome = str(outcome)
                outcomes[
                    outcome_signature_remap.get(
                        stored_outcome, stored_outcome
                    )
                ] += int(count)
            samples = int(row["samples"])
            if samples != sum(outcomes.values()):
                raise ValueError("anonymous behavior outcome counts do not sum")
            hazardous = int(row.get("hazardous", 0))
            if not 0 <= hazardous <= samples:
                raise ValueError("anonymous behavior hazard count is invalid")
            causal_hazard_samples = int(
                row.get("causal_hazard_samples", 0)
            )
            causal_hazardous = int(row.get("causal_hazardous", 0))
            if not 0 <= causal_hazardous <= causal_hazard_samples:
                raise ValueError(
                    "anonymous causal hazard count is invalid"
                )
            if causal_hazard_samples > samples:
                raise ValueError(
                    "anonymous causal hazard samples exceed rule samples"
                )
            model._rules[key] = _RuleStats(
                outcomes=outcomes,
                hazardous=hazardous,
                samples=samples,
                contexts={str(value) for value in row.get("contexts") or ()},
                causal_hazardous=causal_hazardous,
                causal_hazard_samples=causal_hazard_samples,
            )
            if key[4] == model._UNCONDITIONAL_CONTEXT:
                model._unconditional_rule_keys_by_type.setdefault(
                    type_id, set()
                ).add(key)
        model._evidence_ids = {
            str(value) for value in payload.get("evidence_ids") or ()
        }
        model._causal_hazard_evidence_ids = {
            str(value)
            for value in payload.get("causal_hazard_evidence_ids") or ()
        }
        if not model._causal_hazard_evidence_ids <= model._evidence_ids:
            raise ValueError(
                "causal hazard evidence is absent from checkpoint evidence"
            )
        return model

    @classmethod
    def load(cls, path: Path) -> "AnonymousEntityBehaviorModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("anonymous behavior checkpoint must be an object")
        return cls.from_dict(payload)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(self.to_json() + "\n", encoding="utf-8")
        temporary.replace(path)
