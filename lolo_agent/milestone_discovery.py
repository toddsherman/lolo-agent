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

V2 redesign (docs/learnings.md section 4.33; preregistered in
docs/milestone-scoring-v2-2026-08-16.md). The v1 functions above remain
unchanged; the ``*_v2`` functions implement the three redesign requirements:

1. Per-component censoring (:func:`extract_component_event`): changed cells
   are partitioned against the matched control into an action-DEPENDENT
   component (control kept the cell at its root value), an AUTONOMOUS
   component (control reproduced the factual value), and an AMBIGUOUS
   component (control reached a third value). When a dependent component
   exists the event is that component — its signature, reversion, and
   dependence are computed over the attributable cells only — so a real
   milestone co-occurring with animation cells is no longer censored away.
   Events with no dependent component remain autonomous (fully reproduced)
   or dependence-censored (ambiguous cells present), exactly as before.
2. Restore-robust successor windows: callers supply successors from
   lineage-filtered committed windows (and branch-level follow-ups) instead
   of truncating at the first archive restore; the module is agnostic to how
   the window was assembled and continues to treat an empty window as
   return-censored non-evidence.
3. Delayed-divergence valence (:func:`score_events_v2`): the v1
   reversion-to-seen negative rule is replaced. An occurrence carries
   negative evidence when its successor window is REWOUND — some successor
   lies at least ``rewind_transient_floor`` cells from the event's own root
   while lying within ``rewind_proximity_ceiling`` cells of a configuration
   observed before the event (the pair's ``history``) — and the occurrence
   participates in measured factual-vs-control structure: its own matched
   contrast is dependence-evaluable, or an escape divergence (a verified
   alternative arm avoiding at least ``escape_cell_minimum`` cells of
   change the equal-duration control exhibits) was observed at a decision
   root within the preregistered lookback BEFORE the commit. At a terminal
   commit both arms show the change, so the differential evidence comes
   from the pre-terminal root contrast plus the structural rewind, never
   from reversion of the event's own cells. Positive valence is unchanged:
   novel-and-persistent successor structure.

V3 rethink (docs/learnings.md section 4.36; preregistered in
docs/milestone-scoring-v3-2026-08-16.md). The v1 and v2 functions remain
unchanged; the ``*_v3`` functions implement the two rethink requirements:

1. Component-anchored rewind (:func:`extract_component_event_v3`): the v2
   structural reset recognizer is kept unchanged, and a successor marks the
   occurrence rewound only when, against the SAME history array that
   satisfied the proximity test, the event's own component cells revert
   toward the pre-event configuration — strictly more component cells hold
   a pre-event-consistent value (the cell's root value, or the matched
   history array's value at that cell) than hold their post-event value. A
   terminal reset that merely falls inside the successor window no longer
   poisons an unrelated event whose own cells survived it.
2. Occurrence-scoped valence (:func:`occurrence_valence`,
   :func:`score_events_v3`): each occurrence carries its own valence from
   its own evidence (negative: component-anchored rewind plus the unchanged
   v2 divergence-evidence rule, checked first; positive: own persistence
   and own successor novelty fraction; unresolved otherwise, including
   return-censored occurrences). The signature score m(sigma) is unchanged
   and remains the only signature-level aggregate with decision power
   (ranking); the per-signature valence label is a reporting-only plurality
   of occurrence valences that overwrites no occurrence.
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
VALENCE_BASIS_DELAYED_DIVERGENCE = "delayed_divergence"


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

    V2 optional fields (defaults preserve v1 construction sites):

    ``history`` holds arrays observed BEFORE this pair's root (the corpus
    pre-intervention pool plus the attempt's committed arrays prior to the
    root), used only by the v2 structural-rewind test. ``escape_lookback``
    is the caller-computed flag for whether an escape divergence (a
    verified alternative arm avoiding a change the equal-duration control
    exhibits) was observed at a decision root within the preregistered
    lookback window before this pair's decision; ``None`` means no
    evaluable contrast existed (censored, never assumed).
    """

    provenance: EventProvenance
    root: PooledArray
    factual: PooledArray
    control: Optional[PooledArray] = None
    successors: Tuple[PooledArray, ...] = ()
    history: Tuple[PooledArray, ...] = ()
    escape_lookback: Optional[bool] = None

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
        for reference in self.history:
            if len(reference) != len(self.root):
                raise ValueError("history array length must match the root")


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

    V2 optional fields (populated by :func:`extract_component_event` only;
    v1 extraction leaves the defaults):

    ``autonomous_cells``/``ambiguous_cells`` record the changed cells
    excluded from a dependent component's signature. ``rewound`` is ``True``
    when some successor lies at least the configured transient floor from
    the event root while within the proximity ceiling of a pre-event
    history array, ``False`` when successors exist and none does, ``None``
    without successors. ``escape_lookback`` is carried from the pair.
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
    autonomous_cells: Tuple[ChangedCell, ...] = ()
    ambiguous_cells: Tuple[ChangedCell, ...] = ()
    rewound: Optional[bool] = None
    escape_lookback: Optional[bool] = None


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

    The v2 fields (docs/milestone-scoring-v2-2026-08-16.md) are used only by
    the ``*_v2`` functions; v1 scoring ignores them:

    - ``negative_divergence_threshold``: fraction of return-evaluable
      occurrences carrying delayed-divergence negative evidence required to
      classify a signature negative.
    - ``rewind_transient_floor``: minimum changed-cell distance between a
      successor and the event root for the successor to count as crossing a
      terminal-scale transient.
    - ``rewind_proximity_ceiling``: maximum changed-cell distance between
      that successor and some pre-event history array for the transient to
      count as a structural rewind (reset-to-known-configuration).
    - ``escape_cell_minimum``: minimum number of cells a verified
      alternative arm must keep at root values while the equal-duration
      control changes them, for an escape divergence.
    - ``divergence_lookback``: decision-window length (inclusive of the
      event's own decision) over which callers evaluate escape divergence.
    """

    novelty_baseline: float = 0.0
    negative_reversion_threshold: float = 0.5
    positive_persistence_threshold: float = 0.5
    positive_novelty_threshold: float = 0.25
    negative_divergence_threshold: float = 0.5
    rewind_transient_floor: int = 16
    rewind_proximity_ceiling: int = 8
    escape_cell_minimum: int = 8
    divergence_lookback: int = 8

    def __post_init__(self) -> None:
        if self.novelty_baseline < 0.0 or self.novelty_baseline > 1.0:
            raise ValueError("novelty baseline must be within [0, 1]")
        for name in (
            "negative_reversion_threshold",
            "positive_persistence_threshold",
            "positive_novelty_threshold",
            "negative_divergence_threshold",
        ):
            value = getattr(self, name)
            if value <= 0.0 or value > 1.0:
                raise ValueError(f"{name} must be within (0, 1]")
        for name in (
            "rewind_transient_floor",
            "escape_cell_minimum",
            "divergence_lookback",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.rewind_proximity_ceiling, int)
            or self.rewind_proximity_ceiling < 0
        ):
            raise ValueError(
                "rewind_proximity_ceiling must be a non-negative integer"
            )
        if self.rewind_proximity_ceiling >= self.rewind_transient_floor:
            raise ValueError(
                "rewind_proximity_ceiling must be below the transient floor"
            )


@dataclass(frozen=True)
class SignatureScore:
    """The preregistered score and valence for one event signature.

    The trailing defaulted fields are v2/v3-only diagnostics; v1 scoring
    leaves them at their defaults. The occurrence-valence tallies are
    populated by v3 scoring only: they count each occurrence's own valence
    (docs/milestone-scoring-v3-2026-08-16.md section 1.2.2), and for v3 the
    signature-level ``valence``/``valence_basis`` are a REPORTING-ONLY
    plurality over those tallies — ranking comes exclusively from
    ``score``, and no occurrence's valence is overwritten by its class.
    """

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
    negative_divergence_rate: float = 0.0
    rewound_occurrences: int = 0
    escape_lookback_occurrences: int = 0
    positive_occurrences: int = 0
    negative_occurrences: int = 0
    unresolved_occurrences: int = 0


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


# ---------------------------------------------------------------------------
# V2 redesign (docs/learnings.md section 4.33, preregistered in
# docs/milestone-scoring-v2-2026-08-16.md). Additive: nothing above changes.
# ---------------------------------------------------------------------------


def cell_distance(first: Sequence[int], second: Sequence[int]) -> int:
    """Number of cells at which two equal-length arrays differ."""

    if len(first) != len(second):
        raise ValueError("cell distance requires equal-length arrays")
    return sum(1 for a, b in zip(first, second) if a != b)


def _within_distance(
    array: Sequence[int], reference: Sequence[int], ceiling: int
) -> bool:
    """Early-exit test for ``cell_distance(array, reference) <= ceiling``."""

    budget = ceiling
    for a, b in zip(array, reference):
        if a != b:
            if budget == 0:
                return False
            budget -= 1
    return True


def escape_divergence_cells(
    root: Sequence[int],
    factual: Sequence[int],
    control: Sequence[int],
) -> int:
    """Cells the control changed away from root while the factual arm kept.

    This is the pre-terminal differential of the causal-hazard pattern: a
    change that arrives under the equal-duration neutral control but is
    absent from a verified alternative arm was avoidable at that root.
    """

    if len(factual) != len(root) or len(control) != len(root):
        raise ValueError("escape divergence requires equal-length arrays")
    return sum(
        1
        for index in range(len(root))
        if control[index] != root[index] and factual[index] == root[index]
    )


def _component_reverts(
    successor: Sequence[int],
    reference: Sequence[int],
    component: Sequence[ChangedCell],
) -> bool:
    """Whether the component cells revert toward the pre-event configuration.

    The v3 component anchor (docs/milestone-scoring-v3-2026-08-16.md
    section 1.2.1): at ``successor``, strictly more component cells must
    hold a pre-event-consistent value (the cell's root value ``before``, or
    the matched history ``reference`` value at that cell) than hold their
    post-event value ``after``. Precedence is fixed: a cell whose
    post-event value equals the reference value counts as
    pre-event-consistent (the change moved the cell onto a known
    configuration — reset-shaped by construction). Cells at third values
    support neither side.
    """

    reverted = 0
    retained = 0
    for index, before, after in component:
        value = successor[index]
        if value == before or value == reference[index]:
            reverted += 1
        elif value == after:
            retained += 1
    return reverted > retained


def _extract_component_event(
    pair: MatchedEndpointPair,
    config: MilestoneScoreConfig,
    anchored_rewind: bool,
) -> Optional[ExtractedEvent]:
    """Shared v2/v3 per-component extraction (see the public wrappers)."""

    changed = tuple(
        (index, before, after)
        for index, (before, after) in enumerate(zip(pair.root, pair.factual))
        if before != after
    )
    if not changed:
        return None

    autonomous: Tuple[ChangedCell, ...] = ()
    ambiguous: Tuple[ChangedCell, ...] = ()
    if pair.control is None:
        action_dependent: Optional[bool] = None
        component = changed
    else:
        control = pair.control
        dependent_cells = tuple(
            cell for cell in changed if control[cell[0]] == cell[1]
        )
        autonomous = tuple(
            cell for cell in changed if control[cell[0]] == cell[2]
        )
        ambiguous = tuple(
            cell
            for cell in changed
            if control[cell[0]] != cell[1] and control[cell[0]] != cell[2]
        )
        if dependent_cells:
            action_dependent = True
            component = dependent_cells
        elif ambiguous:
            action_dependent = None
            component = changed
        else:
            action_dependent = False
            component = changed

    reverted: Optional[bool]
    rewound: Optional[bool]
    if not pair.successors:
        reverted = None
        rewound = None
    else:
        reverted = any(
            all(
                successor[index] == before
                for index, before, _after in component
            )
            for successor in pair.successors
        )
        rewound = False
        for successor in pair.successors:
            if (
                cell_distance(successor, pair.root)
                < config.rewind_transient_floor
            ):
                continue
            for reference in pair.history:
                if not _within_distance(
                    successor, reference, config.rewind_proximity_ceiling
                ):
                    continue
                if anchored_rewind and not _component_reverts(
                    successor, reference, component
                ):
                    continue
                rewound = True
                break
            if rewound:
                break

    successor_signatures = tuple(
        content_signature(successor) for successor in pair.successors
    )
    return ExtractedEvent(
        signature=_changed_cells_signature(component),
        changed_cells=component,
        action_dependent=action_dependent,
        reverted=reverted,
        root_signature=content_signature(pair.root),
        endpoint_signature=content_signature(pair.factual),
        successor_signatures=successor_signatures,
        first_successor_signature=(
            successor_signatures[0] if successor_signatures else None
        ),
        provenance=pair.provenance,
        autonomous_cells=autonomous,
        ambiguous_cells=ambiguous,
        rewound=rewound,
        escape_lookback=pair.escape_lookback,
    )


def extract_component_event(
    pair: MatchedEndpointPair,
    config: Optional[MilestoneScoreConfig] = None,
) -> Optional[ExtractedEvent]:
    """Extract the per-component matched endpoint difference from one pair.

    Requirement 1 of the section-4.33 redesign. Changed cells are
    partitioned against the matched control:

    - control at the root value: action-DEPENDENT cell;
    - control at the factual value: AUTONOMOUS cell;
    - control at a third value: AMBIGUOUS cell (cell-level censored).

    When dependent cells exist, the event IS that component: its signature,
    reversion, and dependence cover the attributable cells only, and the
    autonomous/ambiguous cells are recorded but excluded. With no dependent
    cells the event keeps the full changed set and is autonomous when the
    control reproduced everything, dependence-censored when ambiguous cells
    exist or the control is missing (censoring never supports a claim).

    ``rewound`` implements the structural reset test for requirement 3: a
    successor at least ``rewind_transient_floor`` cells from the event root
    that lies within ``rewind_proximity_ceiling`` cells of a pre-event
    ``history`` array.
    """

    if config is None:
        config = MilestoneScoreConfig()
    return _extract_component_event(pair, config, anchored_rewind=False)


def extract_component_event_v3(
    pair: MatchedEndpointPair,
    config: Optional[MilestoneScoreConfig] = None,
) -> Optional[ExtractedEvent]:
    """Per-component extraction with the v3 component-anchored rewind.

    Identical to :func:`extract_component_event` — same component
    partition, same signature, same reversion — except that ``rewound``
    requires the component anchor of the section-4.36 rethink: the
    successor must pass the unchanged structural reset test AND, against
    the same matched history array, the event's own component cells must
    revert toward the pre-event configuration (:func:`_component_reverts`).
    A later terminal reset merely crossing the successor window no longer
    marks an occurrence whose own cells survived it.
    """

    if config is None:
        config = MilestoneScoreConfig()
    return _extract_component_event(pair, config, anchored_rewind=True)


def extract_component_events(
    pairs: Sequence[MatchedEndpointPair],
    config: Optional[MilestoneScoreConfig] = None,
) -> Tuple[ExtractedEvent, ...]:
    """Per-component extraction over every pair, dropping no-change pairs."""

    if config is None:
        config = MilestoneScoreConfig()
    events = []
    for pair in pairs:
        event = extract_component_event(pair, config)
        if event is not None:
            events.append(event)
    return tuple(events)


def extract_component_events_v3(
    pairs: Sequence[MatchedEndpointPair],
    config: Optional[MilestoneScoreConfig] = None,
) -> Tuple[ExtractedEvent, ...]:
    """V3 anchored-rewind extraction over every pair (see the singular)."""

    if config is None:
        config = MilestoneScoreConfig()
    events = []
    for pair in pairs:
        event = extract_component_event_v3(pair, config)
        if event is not None:
            events.append(event)
    return tuple(events)


def _score_signature_v2(
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
    rewound_occurrences = 0
    negative_evidence = 0
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
        if event.rewound is True:
            rewound_occurrences += 1
            if (
                event.action_dependent is not None
                or event.escape_lookback is True
            ):
                negative_evidence += 1
    escape_lookback_occurrences = sum(
        1 for event in events if event.escape_lookback is True
    )
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
        negative_divergence_rate = negative_evidence / return_evaluable
    else:
        reversion_to_seen_rate = 0.0
        persistence_rate = 0.0
        negative_divergence_rate = 0.0

    score = (
        log_rarity
        * action_dependence_rate
        * censored_non_return_factor
        * successor_novelty_margin
    )

    if return_evaluable == 0:
        valence = VALENCE_UNRESOLVED
        valence_basis = VALENCE_BASIS_RETURN_CENSORED
    elif negative_divergence_rate >= config.negative_divergence_threshold:
        valence = VALENCE_NEGATIVE
        valence_basis = VALENCE_BASIS_DELAYED_DIVERGENCE
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
        negative_divergence_rate=negative_divergence_rate,
        rewound_occurrences=rewound_occurrences,
        escape_lookback_occurrences=escape_lookback_occurrences,
    )


def score_events_v2(
    events: Sequence[ExtractedEvent],
    seen_signatures: AbstractSet[str] = frozenset(),
    config: Optional[MilestoneScoreConfig] = None,
) -> Tuple[SignatureScore, ...]:
    """Score v2 component events with the unchanged preregistered formula.

    The score product is identical to v1 (rarity, dependence, non-return,
    novelty) computed over component events. Valence replaces the v1
    reversion-to-seen negative rule with delayed-divergence negative
    evidence: rewound occurrences whose matched contrast is
    dependence-evaluable or whose lookback contains an escape divergence.
    Positive valence is the unchanged novel-and-persistent rule.
    """

    if config is None:
        config = MilestoneScoreConfig()
    grouped: Dict[str, List[ExtractedEvent]] = {}
    for event in events:
        grouped.setdefault(event.signature, []).append(event)
    total_events = len(events)
    scores = [
        _score_signature_v2(
            signature, grouped[signature], total_events, seen_signatures, config
        )
        for signature in grouped
    ]
    scores.sort(key=lambda score: (-score.score, score.signature))
    return tuple(scores)


def discover_milestones_v2(
    pairs: Sequence[MatchedEndpointPair],
    seen_signatures: AbstractSet[str] = frozenset(),
    config: Optional[MilestoneScoreConfig] = None,
) -> MilestoneReport:
    """Extract and score pairs with the v2 semantics in one pass."""

    if config is None:
        config = MilestoneScoreConfig()
    events = extract_component_events(pairs, config)
    return MilestoneReport(
        total_pairs=len(pairs),
        pairs_without_event=len(pairs) - len(events),
        total_events=len(events),
        scores=score_events_v2(events, seen_signatures, config),
        config=config,
    )


# ---------------------------------------------------------------------------
# V3 rethink (docs/learnings.md section 4.36, preregistered in
# docs/milestone-scoring-v3-2026-08-16.md). Additive: nothing above changes.
# ---------------------------------------------------------------------------


def occurrence_valence(
    event: ExtractedEvent,
    seen_signatures: AbstractSet[str] = frozenset(),
    config: Optional[MilestoneScoreConfig] = None,
) -> Tuple[str, str]:
    """One occurrence's own valence from its own evidence (v3 requirement 2).

    Returns ``(valence, basis)``. Rules, in order
    (docs/milestone-scoring-v3-2026-08-16.md section 1.2.2):

    - no successor observations: unresolved (return-censored);
    - NEGATIVE (delayed divergence, checked first): the occurrence is
      rewound — the caller supplies an event extracted with the v3
      component-anchored rewind — and participates in measured
      factual-vs-control divergence structure (its matched contrast is
      dependence-evaluable, or an escape divergence was observed within the
      preregistered lookback). Dependence-censored occurrences with no
      escape evidence can never be negative;
    - POSITIVE (novel-and-persistent): the component never reverts within
      the window, the first successor does not collapse onto the seen pool,
      and the occurrence's own successor novelty fraction reaches
      ``positive_novelty_threshold``;
    - unresolved (mixed) otherwise.

    ``negative_divergence_threshold`` does not appear: at occurrence scope
    the negative rule is binary, exactly as v2 retired the v1 reversion
    threshold without removing it from the config.
    """

    if config is None:
        config = MilestoneScoreConfig()
    if event.reverted is None:
        return VALENCE_UNRESOLVED, VALENCE_BASIS_RETURN_CENSORED
    if event.rewound is True and (
        event.action_dependent is not None or event.escape_lookback is True
    ):
        return VALENCE_NEGATIVE, VALENCE_BASIS_DELAYED_DIVERGENCE
    collapsed = (
        event.first_successor_signature is not None
        and event.first_successor_signature in seen_signatures
    )
    if not event.reverted and not collapsed:
        novel = sum(
            1
            for successor in event.successor_signatures
            if successor not in seen_signatures
        )
        novelty = novel / len(event.successor_signatures)
        if novelty >= config.positive_novelty_threshold:
            return VALENCE_POSITIVE, VALENCE_BASIS_NOVEL_AND_PERSISTENT
    return VALENCE_UNRESOLVED, VALENCE_BASIS_MIXED


def _score_signature_v3(
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
    rewound_occurrences = 0
    positive_occurrences = 0
    negative_occurrences = 0
    unresolved_occurrences = 0
    for event in events:
        valence, _basis = occurrence_valence(event, seen_signatures, config)
        if valence == VALENCE_POSITIVE:
            positive_occurrences += 1
        elif valence == VALENCE_NEGATIVE:
            negative_occurrences += 1
        else:
            unresolved_occurrences += 1
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
        if event.rewound is True:
            rewound_occurrences += 1
    escape_lookback_occurrences = sum(
        1 for event in events if event.escape_lookback is True
    )
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
        negative_divergence_rate = negative_occurrences / return_evaluable
    else:
        reversion_to_seen_rate = 0.0
        persistence_rate = 0.0
        negative_divergence_rate = 0.0

    score = (
        log_rarity
        * action_dependence_rate
        * censored_non_return_factor
        * successor_novelty_margin
    )

    # REPORTING-ONLY plurality label (never gates, overwrites no
    # occurrence): negative when negative occurrences strictly exceed
    # positive, positive when the reverse, unresolved otherwise.
    if return_evaluable == 0:
        valence = VALENCE_UNRESOLVED
        valence_basis = VALENCE_BASIS_RETURN_CENSORED
    elif negative_occurrences > positive_occurrences:
        valence = VALENCE_NEGATIVE
        valence_basis = VALENCE_BASIS_DELAYED_DIVERGENCE
    elif positive_occurrences > negative_occurrences:
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
        negative_divergence_rate=negative_divergence_rate,
        rewound_occurrences=rewound_occurrences,
        escape_lookback_occurrences=escape_lookback_occurrences,
        positive_occurrences=positive_occurrences,
        negative_occurrences=negative_occurrences,
        unresolved_occurrences=unresolved_occurrences,
    )


def score_events_v3(
    events: Sequence[ExtractedEvent],
    seen_signatures: AbstractSet[str] = frozenset(),
    config: Optional[MilestoneScoreConfig] = None,
) -> Tuple[SignatureScore, ...]:
    """Score v3 events: unchanged ranking, occurrence-scoped valence.

    ``events`` must come from the v3 anchored-rewind extraction. The score
    product is identical to v1/v2 (rarity, dependence, non-return, novelty)
    and remains the only signature-level aggregate with decision power;
    each occurrence's valence is computed by :func:`occurrence_valence` and
    tallied, and the signature's ``valence`` field is only the reporting
    plurality of those tallies.
    """

    if config is None:
        config = MilestoneScoreConfig()
    grouped: Dict[str, List[ExtractedEvent]] = {}
    for event in events:
        grouped.setdefault(event.signature, []).append(event)
    total_events = len(events)
    scores = [
        _score_signature_v3(
            signature, grouped[signature], total_events, seen_signatures, config
        )
        for signature in grouped
    ]
    scores.sort(key=lambda score: (-score.score, score.signature))
    return tuple(scores)


def discover_milestones_v3(
    pairs: Sequence[MatchedEndpointPair],
    seen_signatures: AbstractSet[str] = frozenset(),
    config: Optional[MilestoneScoreConfig] = None,
) -> MilestoneReport:
    """Extract and score pairs with the v3 semantics in one pass."""

    if config is None:
        config = MilestoneScoreConfig()
    events = extract_component_events_v3(pairs, config)
    return MilestoneReport(
        total_pairs=len(pairs),
        pairs_without_event=len(pairs) - len(events),
        total_events=len(events),
        scores=score_events_v3(events, seen_signatures, config),
        config=config,
    )
