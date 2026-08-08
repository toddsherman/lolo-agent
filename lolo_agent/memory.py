from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import sqrt
from typing import Counter as CounterType, Deque, Dict, Iterable, Tuple

from .environment import Action


class VisualNovelty:
    """Temporary per-attempt counts; deliberately absent from checkpoints."""

    def __init__(self) -> None:
        self._counts: CounterType[str] = Counter()

    def score(self, signature: str) -> float:
        return 1.0 / sqrt(self._counts[signature] + 1.0)

    def observe(self, signature: str) -> None:
        self._counts[signature] += 1

    def count(self, signature: str) -> int:
        return self._counts[signature]


@dataclass(frozen=True)
class EpisodeEvent:
    source: str
    action: Action
    target: str
    visual_change: float


class EpisodicMemory:
    """Bounded, temporary memory of branches and ineffective attempts."""

    def __init__(self, capacity: int = 20_000) -> None:
        self._events: Deque[EpisodeEvent] = deque(maxlen=capacity)
        self._attempts: CounterType[Tuple[str, Action, str]] = Counter()
        self._ineffective: CounterType[Tuple[str, Action]] = Counter()

    def record(self, event: EpisodeEvent) -> None:
        self._events.append(event)
        self._attempts[(event.source, event.action, event.target)] += 1
        if event.visual_change < 1e-6:
            self._ineffective[(event.source, event.action)] += 1

    def repetition_penalty(self, source: str, action: Action, target: str) -> float:
        repeats = self._attempts[(source, action, target)]
        ineffective = self._ineffective[(source, action)]
        return 0.12 * sqrt(repeats) + 0.25 * sqrt(ineffective)

    def recent(self, limit: int = 100) -> Iterable[EpisodeEvent]:
        events = list(self._events)
        return events[-limit:]

