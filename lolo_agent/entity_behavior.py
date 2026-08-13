from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Counter as CounterType, Dict, Iterable, Optional, Sequence, Tuple

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

    SCHEMA_VERSION = 3
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
        self._evidence_ids: set[str] = set()

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

    @classmethod
    def effect_signature(
        cls,
        source_feature: Sequence[int],
        factual_feature: Sequence[int],
        control_feature: Sequence[int],
        relative_effect_cells: Sequence[RelativeCell] = (),
        player_displacement: Optional[RelativeCell] = None,
        terminal_visual_change: bool = False,
    ) -> str:
        """Create a position-invariant outcome key from pixel-derived facts."""

        def appearance_relation(
            first: Sequence[int], second: Sequence[int]
        ) -> str:
            distance = cls._feature_distance(first, second)
            if distance <= 0.08:
                return "same"
            if distance <= 0.2:
                return "near"
            return "different"

        payload = {
            # The source appearance identifies the anonymous type and does not
            # belong in the outcome.  Storing its exact fingerprint here made
            # harmless animation phases look like different mechanics.
            "factual_source_relation": appearance_relation(
                factual_feature, source_feature
            ),
            "control_source_relation": appearance_relation(
                control_feature, source_feature
            ),
            "factual_control_relation": appearance_relation(
                factual_feature, control_feature
            ),
            "relative_effect_cells": sorted(
                (int(column), int(row))
                for column, row in relative_effect_cells
            ),
            "player_displacement": (
                None
                if player_displacement is None
                else (
                    int(player_displacement[0]),
                    int(player_displacement[1]),
                )
            ),
            "terminal_visual_change": bool(terminal_visual_change),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

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
        if context_signature:
            contextual = self._rules.get(
                self._rule_key(
                    type_id,
                    action,
                    duration,
                    autonomous,
                    context_signature,
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

    @staticmethod
    def _prediction_from_rule(
        type_id: Optional[int],
        distance: float,
        rule: Optional[_RuleStats],
        minimum_prediction_samples: int,
        context_matched: bool,
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
            )
        rule, contextual = self._select_rule(
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
        )

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
        rule, _contextual = self._select_rule(
            type_id,
            action,
            duration,
            autonomous,
            context_signature,
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
    ) -> BehaviorObservation:
        if not feature:
            raise ValueError("behavior appearance feature must be non-empty")
        if not outcome_signature:
            raise ValueError("behavior outcome signature must be non-empty")
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
        type_id, _distance, created = self._assign(appearance)
        previous_count = self._type_counts[type_id]
        next_count = previous_count + 1
        prototype = self._type_means[type_id]
        for index, value in enumerate(appearance):
            prototype[index] += (value - prototype[index]) / next_count
        self._type_counts[type_id] = next_count

        contexts = [self._UNCONDITIONAL_CONTEXT]
        if context_signature:
            contexts.append(context_signature)
        for condition in contexts:
            key = self._rule_key(
                type_id, action, duration, autonomous, condition
            )
            rule = self._rules.setdefault(key, _RuleStats(Counter()))
            rule.outcomes[outcome_signature] += 1
            rule.samples += 1
            rule.hazardous += int(hazardous)
            if context_signature:
                rule.contexts.add(context_signature)
        if evidence_id:
            self._evidence_ids.add(evidence_id)

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
            "evidence_ids": sorted(self._evidence_ids),
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
        if int(payload.get("schema_version", 0)) != cls.SCHEMA_VERSION:
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
            outcomes = Counter(
                {
                    str(outcome): int(count)
                    for outcome, count in dict(row["outcomes"]).items()
                }
            )
            samples = int(row["samples"])
            if samples != sum(outcomes.values()):
                raise ValueError("anonymous behavior outcome counts do not sum")
            hazardous = int(row.get("hazardous", 0))
            if not 0 <= hazardous <= samples:
                raise ValueError("anonymous behavior hazard count is invalid")
            model._rules[key] = _RuleStats(
                outcomes=outcomes,
                hazardous=hazardous,
                samples=samples,
                contexts={str(value) for value in row.get("contexts") or ()},
            )
        model._evidence_ids = {
            str(value) for value in payload.get("evidence_ids") or ()
        }
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
