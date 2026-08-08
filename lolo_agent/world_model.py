from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Counter as CounterType, DefaultDict, Dict, Mapping, Tuple

from .environment import Action
from .pixels import Frame, signature_key


class FrozenModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    probability: float
    uncertainty: float
    known_outcomes: int

    @property
    def surprise(self) -> float:
        return -math.log(max(self.probability, 1e-9))


class EmpiricalWorldModel:
    """A dependency-free learned visual dynamics baseline.

    Exact visual-context transitions are backed off to action-only dynamics.
    Laplace smoothing keeps unseen consequences surprising and gives the
    planner an uncertainty signal. No game concepts are encoded here.
    """

    def __init__(self, signature_columns: int = 8, signature_rows: int = 8) -> None:
        self.signature_columns = signature_columns
        self.signature_rows = signature_rows
        self._context: DefaultDict[Tuple[str, str], CounterType[str]] = defaultdict(Counter)
        self._action: DefaultDict[str, CounterType[str]] = defaultdict(Counter)
        self._trainable = True

    @property
    def trainable(self) -> bool:
        return self._trainable

    def freeze(self) -> None:
        self._trainable = False

    def unfreeze(self) -> None:
        self._trainable = True

    def signature(self, frame: Frame) -> str:
        values = frame.coarse_signature(self.signature_columns, self.signature_rows)
        return signature_key(values)

    def predict(self, source: Frame, action: Action, target: Frame) -> Prediction:
        source_key = self.signature(source)
        target_key = self.signature(target)
        exact = self._context.get((source_key, action.value), Counter())
        fallback = self._action.get(action.value, Counter())
        counts = exact if sum(exact.values()) >= 2 else fallback
        total = sum(counts.values())
        vocabulary = max(2, len(counts) + 1)
        probability = (counts[target_key] + 1.0) / (total + vocabulary)
        uncertainty = 1.0 / math.sqrt(total + 1.0)
        return Prediction(probability, uncertainty, len(counts))

    def observe(self, source: Frame, action: Action, target: Frame) -> None:
        if not self._trainable:
            raise FrozenModelError("persistent world model update attempted while frozen")
        source_key = self.signature(source)
        target_key = self.signature(target)
        self._context[(source_key, action.value)][target_key] += 1
        self._action[action.value][target_key] += 1

    def to_dict(self) -> Dict[str, object]:
        contexts = {
            json.dumps([source, action]): dict(sorted(counts.items()))
            for (source, action), counts in sorted(self._context.items())
        }
        actions = {action: dict(sorted(counts.items())) for action, counts in sorted(self._action.items())}
        return {
            "version": 1,
            "signature_columns": self.signature_columns,
            "signature_rows": self.signature_rows,
            "trainable": self._trainable,
            "context": contexts,
            "action": actions,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EmpiricalWorldModel":
        model = cls(int(data["signature_columns"]), int(data["signature_rows"]))
        for encoded, values in dict(data.get("context", {})).items():
            source, action = json.loads(encoded)
            model._context[(source, action)].update(dict(values))
        for action, values in dict(data.get("action", {})).items():
            model._action[action].update(dict(values))
        model._trainable = bool(data.get("trainable", True))
        return model

    def checkpoint_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()

    @property
    def checkpoint_digest(self) -> str:
        return sha256(self.checkpoint_bytes()).hexdigest()

