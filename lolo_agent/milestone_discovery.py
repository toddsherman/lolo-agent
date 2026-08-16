"""Offline milestone-discovery scoring skeleton (WP9a spike, census stage).

ENGINEERING-ONLY. This module is a pure, telemetry-shaped scoring skeleton
adopted under direction-review Amendment D (docs/direction-review-2026-08-16.md
section 3.D, roadmap section 17 item 7). It carries the assisted-footprint
caveat from that review: every real corpus currently available to feed it was
produced with assisted-track instruments, so nothing this module computes is
evidence of strict-track milestone discovery until WP5 supplies a
strict-legitimate controllable footprint. Until then its outputs are
engineering artifacts for fixture tests and the preregistered offline spike
only.

Inputs are plain arrays (matched factual/NOOP endpoint observations); the
module never touches an emulator, files, or telemetry streams. Callers are
responsible for reducing frames or telemetry rows to pooled integer arrays,
optionally via :func:`pool_values`.

Preregistered score, computed per event signature ``sigma`` over a corpus of
extracted events::

    m(sigma) = log_rarity(sigma)
             * action_dependence_rate(sigma)
             * censored_non_return_factor(sigma)
             * successor_novelty_margin(sigma)

with the component definitions:

- ``log_rarity``: ``ln(total_events / occurrences(sigma))``; a signature
  present in every extracted event scores exactly zero.
- ``action_dependence_rate``: fraction of dependence-evaluable occurrences
  whose matched equal-duration neutral control did NOT reproduce the change.
  Occurrences without a control, or with a control that partially reproduces
  the change, are dependence-censored and excluded from the rate; a fully
  censored signature scores zero (censoring never supports a claim).
- ``censored_non_return_factor``: fraction of return-evaluable occurrences
  (at least one successor observation) whose changed cells never reverted to
  their pre-event values within the observed successor window. Occurrences
  with no successor observations are return-censored and excluded; a fully
  censored signature scores zero.
- ``successor_novelty_margin``: mean, over return-evaluable occurrences, of
  the fraction of successor content signatures absent from the pre-supplied
  seen-signature pool, minus the configured novelty baseline, floored at
  zero. No successor observations anywhere yields zero.

Valence classification (per signature, over return-evaluable occurrences):

- ``negative`` when the reversion-to-seen rate reaches the configured
  threshold. An occurrence reverts-to-seen when its changed cells return to
  their pre-event values or its first successor's content signature is
  already in the seen pool (immediate collapse onto a known configuration).
- ``positive`` when the persistence rate (no reversion-to-seen) reaches its
  threshold and mean successor novelty reaches the novelty threshold
  (novel-and-persistent).
- ``unresolved`` otherwise, including every fully return-censored signature.

The outcome vocabulary is deliberately anonymous: events are changed cells
between matched endpoints, and categories are measured pixel outcomes only.
No supplied game-semantic names appear in code, fields, or telemetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from hashlib import sha256
from typing import (
    AbstractSet,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
)


PooledArray = Tuple[int, ...]
ChangedCell = Tuple[int, int, int]

VALENCE_POSITIVE = "positive"
VALENCE_NEGATIVE = "negative"
VALENCE_UNRESOLVED = "unresolved"

VALENCE_BASIS_NOVEL_AND_PERSISTENT = "novel_and_persistent"
VALENCE_BASIS_REVERSION_TO_SEEN = "reversion_to_seen"
VALENCE_BASIS_RETURN_CENSORED = "return_censored"
VALENCE_BASIS_MIXED = "mixed"


def pool_values(
    values: Sequence[int],
    columns: int,
    rows: int,
    pooled_columns: int,
    pooled_rows: int,
    quantization: int = 1,
) -> PooledArray:
    """Pool a row-major integer grid down to a smaller row-major grid.

    Each pooled cell is the mean of its source cells, floor-divided by
    ``quantization``. The reduction is deterministic and shape-checked; it is
    the array-only counterpart of the existing pooled-feature reductions and
    introduces no object or tile assumptions.
    """

    if columns <= 0 or rows <= 0:
        raise ValueError("grid dimensions must be positive")
    if pooled_columns <= 0 or pooled_rows <= 0:
        raise ValueError("pooled dimensions must be positive")
    if pooled_columns > columns or pooled_rows > rows:
        raise ValueError("pooled dimensions must not exceed the source grid")
    if quantization <= 0:
        raise ValueError("quantization must be positive")
    if len(values) != columns * rows:
        raise ValueError(
            "value count does not match dimensions: "
            f"expected {columns * rows}, got {len(values)}"
        )
    pooled: List[int] = []
    for pooled_row in range(pooled_rows):
        y0 = pooled_row * rows // pooled_rows
        y1 = max(y0 + 1, (pooled_row + 1) * rows // pooled_rows)
        for pooled_column in range(pooled_columns):
            x0 = pooled_column * columns // pooled_columns
            x1 = max(x0 + 1, (pooled_column + 1) * columns // pooled_columns)
            total = 0
            samples = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    total += values[y * columns + x]
                    samples += 1
            pooled.append((total // max(1, samples)) // quantization)
    return tuple(pooled)


def content_signature(values: Sequence[int]) -> str:
    """Deterministic content signature of one integer array."""

    header = f"{len(values)}:".encode()
    body = ",".join(str(value) for value in values).encode()
    return sha256(header + body).hexdigest()


def _changed_cells_signature(cells: Sequence[ChangedCell]) -> str:
    header = f"event:{len(cells)}:".encode()
    body = ";".join(
        f"{index},{before},{after}" for index, before, after in cells
    ).encode()
    return sha256(header + body).hexdigest()


@dataclass(frozen=True)
class EventProvenance:
    """Where one matched endpoint pair came from.

    ``source`` is explicit so synthetic fixtures can never masquerade as
    telemetry-derived evidence.
    """

    run_id: str
    decision: int
    branch_id: str
    action: str
    duration: int
    source: str = "synthetic"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("provenance requires a run id")
        if self.decision < 0:
            raise ValueError("provenance decision must be non-negative")
        if not self.branch_id:
            raise ValueError("provenance requires a branch id")
        if not self.action:
            raise ValueError("provenance requires an action label")
        if self.duration <= 0:
            raise ValueError("provenance duration must be positive")
        if self.source not in ("synthetic", "telemetry"):
            raise ValueError("provenance source must be synthetic or telemetry")


@dataclass(frozen=True)
class MatchedEndpointPair:
    """One factual endpoint with its matched equal-duration neutral control.

    ``root`` is the shared pre-action array, ``factual`` the endpoint after
    the recorded action, and ``control`` the endpoint after the matched
    neutral input from the same root for the same total duration (``None``
    when the control arm is missing from telemetry; the pair is then
    dependence-censored, never assumed dependent). ``successors`` are later
    observed arrays on the factual arm's timeline, in observation order.
    """

    provenance: EventProvenance
    root: PooledArray
    factual: PooledArray
    control: Optional[PooledArray] = None
    successors: Tuple[PooledArray, ...] = ()

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("matched pair requires a non-empty root array")
        if len(self.factual) != len(self.root):
            raise ValueError("factual array length must match the root")
        if self.control is not None and len(self.control) != len(self.root):
            raise ValueError("control array length must match the root")
        for successor in self.successors:
            if len(successor) != len(self.root):
                raise ValueError("successor array length must match the root")


@dataclass(frozen=True)
class ExtractedEvent:
    """One factual-vs-root endpoint difference with censoring bookkeeping.

    ``action_dependent`` is ``True`` when the matched control kept every
    changed cell at its root value, ``False`` when the control reproduced the
    complete change autonomously, and ``None`` when the control is missing or
    reproduced the change only partially (dependence-censored).

    ``reverted`` is ``True`` when some successor restored every changed cell
    to its pre-event value, ``False`` when successors exist and never did,
    and ``None`` when no successor was observed (return-censored).
    """

    signature: str
    changed_cells: Tuple[ChangedCell, ...]
    action_dependent: Optional[bool]
    reverted: Optional[bool]
    root_signature: str
    endpoint_signature: str
    successor_signatures: Tuple[str, ...]
    first_successor_signature: Optional[str]
    provenance: EventProvenance


def extract_event(pair: MatchedEndpointPair) -> Optional[ExtractedEvent]:
    """Extract the matched endpoint difference from one pair, if any."""

    changed = tuple(
        (index, before, after)
        for index, (before, after) in enumerate(zip(pair.root, pair.factual))
        if before != after
    )
    if not changed:
        return None

    action_dependent: Optional[bool]
    if pair.control is None:
        action_dependent = None
    else:
        control_at_root = all(
            pair.control[index] == before for index, before, _after in changed
        )
        control_at_factual = all(
            pair.control[index] == after for index, _before, after in changed
        )
        if control_at_root:
            action_dependent = True
        elif control_at_factual:
            action_dependent = False
        else:
            action_dependent = None

    reverted: Optional[bool]
    if not pair.successors:
        reverted = None
    else:
        reverted = any(
            all(
                successor[index] == before
                for index, before, _after in changed
            )
            for successor in pair.successors
        )

    successor_signatures = tuple(
        content_signature(successor) for successor in pair.successors
    )
    return ExtractedEvent(
        signature=_changed_cells_signature(changed),
        changed_cells=changed,
        action_dependent=action_dependent,
        reverted=reverted,
        root_signature=content_signature(pair.root),
        endpoint_signature=content_signature(pair.factual),
        successor_signatures=successor_signatures,
        first_successor_signature=(
            successor_signatures[0] if successor_signatures else None
        ),
        provenance=pair.provenance,
    )


def extract_events(
    pairs: Sequence[MatchedEndpointPair],
) -> Tuple[ExtractedEvent, ...]:
    """Extract events from every pair, dropping pairs without a difference."""

    events = []
    for pair in pairs:
        event = extract_event(pair)
        if event is not None:
            events.append(event)
    return tuple(events)


@dataclass(frozen=True)
class MilestoneScoreConfig:
    """Preregistered scoring and valence constants.

    The defaults are the spike's preregistered values; the event census
    (docs/milestone-event-census-2026-08-16.md) decides whether the corpora
    can support revising them before the scoring run.
    """

    novelty_baseline: float = 0.0
    negative_reversion_threshold: float = 0.5
    positive_persistence_threshold: float = 0.5
    positive_novelty_threshold: float = 0.25

    def __post_init__(self) -> None:
        if self.novelty_baseline < 0.0 or self.novelty_baseline > 1.0:
            raise ValueError("novelty baseline must be within [0, 1]")
        for name in (
            "negative_reversion_threshold",
            "positive_persistence_threshold",
            "positive_novelty_threshold",
        ):
            value = getattr(self, name)
            if value <= 0.0 or value > 1.0:
                raise ValueError(f"{name} must be within (0, 1]")


@dataclass(frozen=True)
class SignatureScore:
    """The preregistered score and valence for one event signature."""

    signature: str
    occurrences: int
    log_rarity: float
    action_dependence_rate: float
    dependence_evaluable: int
    dependence_censored: int
    censored_non_return_factor: float
    return_evaluable: int
    return_censored: int
    successor_novelty_margin: float
    reversion_to_seen_rate: float
    persistence_rate: float
    score: float
    valence: str
    valence_basis: str
    provenance: Tuple[EventProvenance, ...]


@dataclass(frozen=True)
class MilestoneReport:
    """Deterministic scoring output over one extracted-event corpus."""

    total_pairs: int
    pairs_without_event: int
    total_events: int
    scores: Tuple[SignatureScore, ...]
    config: MilestoneScoreConfig = field(default_factory=MilestoneScoreConfig)


def _score_signature(
    signature: str,
    events: Sequence[ExtractedEvent],
    total_events: int,
    seen_signatures: AbstractSet[str],
    config: MilestoneScoreConfig,
) -> SignatureScore:
    occurrences = len(events)
    log_rarity = math.log(total_events / occurrences)

    dependent = sum(1 for event in events if event.action_dependent is True)
    independent = sum(1 for event in events if event.action_dependent is False)
    dependence_evaluable = dependent + independent
    dependence_censored = occurrences - dependence_evaluable
    if dependence_evaluable > 0:
        action_dependence_rate = dependent / dependence_evaluable
    else:
        action_dependence_rate = 0.0

    reverted_cells = sum(1 for event in events if event.reverted is True)
    persisted_cells = sum(1 for event in events if event.reverted is False)
    return_evaluable = reverted_cells + persisted_cells
    return_censored = occurrences - return_evaluable
    if return_evaluable > 0:
        censored_non_return_factor = persisted_cells / return_evaluable
    else:
        censored_non_return_factor = 0.0

    novelty_fractions: List[float] = []
    reversion_to_seen = 0
    persistent_not_seen = 0
    for event in events:
        if event.reverted is None:
            continue
        novel = sum(
            1
            for successor in event.successor_signatures
            if successor not in seen_signatures
        )
        novelty_fractions.append(novel / len(event.successor_signatures))
        collapsed = (
            event.first_successor_signature is not None
            and event.first_successor_signature in seen_signatures
        )
        if event.reverted or collapsed:
            reversion_to_seen += 1
        else:
            persistent_not_seen += 1
    if novelty_fractions:
        mean_novelty = sum(novelty_fractions) / len(novelty_fractions)
    else:
        mean_novelty = 0.0
    successor_novelty_margin = max(
        0.0, mean_novelty - config.novelty_baseline
    )

    if return_evaluable > 0:
        reversion_to_seen_rate = reversion_to_seen / return_evaluable
        persistence_rate = persistent_not_seen / return_evaluable
    else:
        reversion_to_seen_rate = 0.0
        persistence_rate = 0.0

    score = (
        log_rarity
        * action_dependence_rate
        * censored_non_return_factor
        * successor_novelty_margin
    )

    if return_evaluable == 0:
        valence = VALENCE_UNRESOLVED
        valence_basis = VALENCE_BASIS_RETURN_CENSORED
    elif reversion_to_seen_rate >= config.negative_reversion_threshold:
        valence = VALENCE_NEGATIVE
        valence_basis = VALENCE_BASIS_REVERSION_TO_SEEN
    elif (
        persistence_rate >= config.positive_persistence_threshold
        and mean_novelty >= config.positive_novelty_threshold
    ):
        valence = VALENCE_POSITIVE
        valence_basis = VALENCE_BASIS_NOVEL_AND_PERSISTENT
    else:
        valence = VALENCE_UNRESOLVED
        valence_basis = VALENCE_BASIS_MIXED

    return SignatureScore(
        signature=signature,
        occurrences=occurrences,
        log_rarity=log_rarity,
        action_dependence_rate=action_dependence_rate,
        dependence_evaluable=dependence_evaluable,
        dependence_censored=dependence_censored,
        censored_non_return_factor=censored_non_return_factor,
        return_evaluable=return_evaluable,
        return_censored=return_censored,
        successor_novelty_margin=successor_novelty_margin,
        reversion_to_seen_rate=reversion_to_seen_rate,
        persistence_rate=persistence_rate,
        score=score,
        valence=valence,
        valence_basis=valence_basis,
        provenance=tuple(event.provenance for event in events),
    )


def score_events(
    events: Sequence[ExtractedEvent],
    seen_signatures: AbstractSet[str] = frozenset(),
    config: Optional[MilestoneScoreConfig] = None,
) -> Tuple[SignatureScore, ...]:
    """Score every distinct event signature with the preregistered formula.

    ``seen_signatures`` is the pre-event pool of full-array content
    signatures the corpus had already observed; the caller fixes it before
    scoring so novelty is never computed against hindsight.
    """

    if config is None:
        config = MilestoneScoreConfig()
    grouped: Dict[str, List[ExtractedEvent]] = {}
    for event in events:
        grouped.setdefault(event.signature, []).append(event)
    total_events = len(events)
    scores = [
        _score_signature(
            signature, grouped[signature], total_events, seen_signatures, config
        )
        for signature in grouped
    ]
    scores.sort(key=lambda score: (-score.score, score.signature))
    return tuple(scores)


def discover_milestones(
    pairs: Sequence[MatchedEndpointPair],
    seen_signatures: AbstractSet[str] = frozenset(),
    config: Optional[MilestoneScoreConfig] = None,
) -> MilestoneReport:
    """Extract and score matched endpoint pairs in one deterministic pass."""

    if config is None:
        config = MilestoneScoreConfig()
    events = extract_events(pairs)
    return MilestoneReport(
        total_pairs=len(pairs),
        pairs_without_event=len(pairs) - len(events),
        total_events=len(events),
        scores=score_events(events, seen_signatures, config),
        config=config,
    )


def seen_pool_from_pairs(
    pairs: Sequence[MatchedEndpointPair],
) -> FrozenSet[str]:
    """Content signatures of every root and control array in a corpus.

    A convenience for fixtures: the pre-event pool built from the states the
    corpus started from, excluding factual endpoints and their successors so
    the pool never contains the outcomes being scored.
    """

    pool = set()
    for pair in pairs:
        pool.add(content_signature(pair.root))
        if pair.control is not None:
            pool.add(content_signature(pair.control))
    return frozenset(pool)
