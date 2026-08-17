"""WP5 mask-sensitive promotion gate (learnings section 4.31 plan-change item c).

The original substitution replay recorded a formal letter-PASS whose gated
bits turned out to be MASK-IRRELEVANT for the replayed archive shapes: the
identity fields derived from metadata/bitmask and the recorded destination
signature had been computed unclipped, so the bits could not distinguish a
learned mask that works from one that does not (``docs/learnings.md``
section 4.31).  This gate fixes that instrument lesson by scoring
learned-versus-assisted agreement ONLY on frames where masking
demonstrably matters.

Mattering-frame detector
------------------------

A frame is MASK-MATTERING when the assisted goal-prior player mask has
causal influence on the downstream tracking quantities: the
``world_effect_cells_state_signature`` computed over the coarse cells the
assisted mask touches DIFFERS between the assisted-masked computation and
the completely unmasked computation of the same frame.  Frames without an
assisted player detection have no assisted mask and therefore cannot be
mattering; frames where erasing the assisted mask leaves every quantized
pooled appearance unchanged are non-mattering.  Per-cell
``masked_cell_fingerprint`` values are recorded alongside so the changed
cells are visible, and the two views (signature inequality, fingerprint
inequality) are hashes of the same ``UnlabeledEntityMemory.feature_at``
features and can only disagree through hash collision.

Gate quantities (scored on mattering frames)
--------------------------------------------

On each mattering frame the same downstream quantities are recomputed with
the LEARNED mask -- tracker v4 thresholded at the pinned validation
operating point ``LEARNED_MASK_PROBABILITY_THRESHOLD`` (0.5) and expanded
to pixel blocks exactly as in ``tracker_substitution_replay`` (whose
helpers are reused, not reimplemented) -- substituted for the assisted
mask.  The comparison cell set is every coarse cell touched by EITHER
mask, so learned over-coverage that erases neighbouring anonymous
appearance is a genuine disagreement rather than an invisible one:

- ``signature_equal``: ``world_effect_cells_state_signature`` over the
  comparison cells is identical under the learned and assisted masks
  (exact equality is how the planner compares world states, so this is
  the silently-replaceable criterion);
- ``l1_within``: every comparison cell's ``feature_at`` vectors under the
  two masks are within the established normalized-L1 appearance threshold
  ``APPEARANCE_L1_THRESHOLD`` (0.08) -- the graded instrument that says
  HOW close the learned reconstruction is when exact equality fails.

Preregistered gate (fixed before execution; see the preregistration
section appended to ``docs/wp5-tracker-training-2026-08-16.md``): per
corpus, both agreement rates over mattering frames must reach
``AGREEMENT_RATE_THRESHOLD`` (0.95) and the corpus must contain at least
``MINIMUM_MATTERING_FRAMES`` (50) mattering frames for the instrument to
be non-vacuous; the gate passes only when all three probe corpora pass.
Agreement on non-mattering frames is reported separately (expected to be
trivially high), and per-frame mask IoU is reported for every mattering
frame.  A FAIL means the learned mask changes downstream tracking
quantities and cannot silently replace the assisted mask yet.

Everything here is replay-only telemetry: the ``object_tracks`` pure
functions take the mask as a parameter, no planner code participates, and
no archive is modified.

Usage::

    python -m lolo_agent.mask_sensitive_gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .entity_behavior import AnonymousEntityBehaviorModel
from .experience_import import decode_logged_png
from .goal_prior import PixelHeartGoalPrior
from .object_tracks import (
    masked_cell_fingerprint,
    world_effect_cells_state_signature,
)
from .pixels import Frame
from .tracker_substitution_replay import (
    APPEARANCE_L1_THRESHOLD,
    DEFAULT_BACKBONE,
    LEARNED_MASK_PROBABILITY_THRESHOLD,
    learned_mask_cells,
    learned_pixel_mask,
    learned_reference_slot,
    load_replay_tracker,
    mask_divergence,
)
from .unlabeled_entities import UnlabeledEntityMemory

Cell = Tuple[int, int]
Pixel = Tuple[int, int]

GATE_VERSION = 1
GATE_KIND = "wp5-mask-sensitive-gate"
_DIGEST_PREFIX = f"{GATE_KIND}:v{GATE_VERSION}:"

# Preregistered gate constants, fixed before execution and pinned by the
# unit tests.  The mask-probability and appearance thresholds are the
# established operating points reused (imported) from the substitution
# replay; the agreement-rate and minimum-mattering constants are this
# gate's own preregistration.
AGREEMENT_RATE_THRESHOLD = 0.95
MINIMUM_MATTERING_FRAMES = 50

DEFAULT_CORPORA = (
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v322-room3-paired-probe-arm-a-pushed-d12",
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v323-room3-paired-probe-arm-b-prepush-d12",
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v325-room3-object-removed-probe-d12",
)
DEFAULT_CHECKPOINT = "experiments/lolo1-wp5/controllable-tracker-v4.pt"
DEFAULT_REPORT = "experiments/lolo1-wp5/mask-sensitive-gate-report.json"

REASON_NO_DETECTION = "no_assisted_detection"
REASON_EMPTY_MASK = "empty_assisted_mask"
REASON_UNCHANGED = "quantities_unchanged"
REASON_CHANGED = "quantities_changed"


def pixel_cell(
    x: int, y: int, width: int, height: int, columns: int, rows: int
) -> Cell:
    """The coarse cell containing one pixel under the ``feature_at`` grid.

    ``feature_at`` partitions the frame with integer floors
    (``x0 = column * width // columns``), so the cell containing pixel
    ``x`` is the largest ``column`` with ``column * width // columns <= x``
    -- exactly ``((x + 1) * columns - 1) // width``.  The unit tests pin
    this inverse against ``grid_cell_pixel_block`` on non-divisible
    dimensions.
    """

    return (
        ((x + 1) * columns - 1) // width,
        ((y + 1) * rows - 1) // height,
    )


def cells_touched_by_pixels(
    pixels: FrozenSet[Pixel],
    width: int,
    height: int,
    columns: int,
    rows: int,
) -> Tuple[Cell, ...]:
    """Sorted coarse cells containing at least one in-bounds mask pixel."""

    return tuple(
        sorted(
            {
                pixel_cell(x, y, width, height, columns, rows)
                for x, y in pixels
                if 0 <= x < width and 0 <= y < height
            }
        )
    )


@dataclass(frozen=True)
class FrameQuantities:
    """The downstream tracking quantities of one frame under one mask."""

    signature: str
    features: Mapping[Cell, Tuple[int, ...]]
    fingerprints: Mapping[Cell, str]


def masked_frame_quantities(
    frame: Frame,
    cells: Sequence[Cell],
    slot: Optional[Pixel],
    mask_pixels: FrozenSet[Pixel],
    memory: UnlabeledEntityMemory,
) -> FrameQuantities:
    """Compute signature, per-cell features and fingerprints under one mask.

    Mirrors the substitution replay's convention exactly: the mask is
    applied only when an anchor ``slot`` is provided (``slot=None`` is the
    explicitly unmasked computation), and the per-cell fingerprints come
    from ``object_tracks.masked_cell_fingerprint`` itself with a fresh
    behaviour model so the stateful ``classify`` side channel cannot leak
    between computations.
    """

    model = AnonymousEntityBehaviorModel()

    def mask_callable(_frame: Frame, _slot: Pixel) -> FrozenSet[Pixel]:
        return mask_pixels

    signature = world_effect_cells_state_signature(
        frame, tuple(cells), slot, memory, mask_callable
    )
    ignored = mask_pixels if slot is not None else None
    features: Dict[Cell, Tuple[int, ...]] = {}
    fingerprints: Dict[Cell, str] = {}
    for cell in cells:
        features[cell] = tuple(memory.feature_at(frame, *cell, ignored))
        fingerprint, _type_id = masked_cell_fingerprint(
            frame, cell, slot, memory, model, mask_callable
        )
        fingerprints[cell] = fingerprint
    return FrameQuantities(
        signature=signature, features=features, fingerprints=fingerprints
    )


def frame_mask_sensitivity(
    frame: Frame,
    assisted_slot: Optional[Pixel],
    assisted_pixels: FrozenSet[Pixel],
    memory: UnlabeledEntityMemory,
) -> Dict[str, Any]:
    """The mattering-frame detector: does the assisted mask change anything?

    Compares the downstream quantities over the assisted-mask-touched
    cells between the assisted-masked and the unmasked computation.  The
    frame is MASK-MATTERING exactly when the state signature differs.
    """

    if assisted_slot is None:
        return {
            "mattering": False,
            "reason": REASON_NO_DETECTION,
            "assisted_cells": (),
            "changed_cells": (),
        }
    if not assisted_pixels:
        return {
            "mattering": False,
            "reason": REASON_EMPTY_MASK,
            "assisted_cells": (),
            "changed_cells": (),
        }
    assisted_cells = cells_touched_by_pixels(
        assisted_pixels, frame.width, frame.height, memory.columns, memory.rows
    )
    unmasked = masked_frame_quantities(
        frame, assisted_cells, None, frozenset(), memory
    )
    assisted = masked_frame_quantities(
        frame, assisted_cells, assisted_slot, assisted_pixels, memory
    )
    changed_cells = tuple(
        cell
        for cell in assisted_cells
        if assisted.fingerprints[cell] != unmasked.fingerprints[cell]
    )
    mattering = assisted.signature != unmasked.signature
    return {
        "mattering": mattering,
        "reason": REASON_CHANGED if mattering else REASON_UNCHANGED,
        "assisted_cells": assisted_cells,
        "changed_cells": changed_cells,
        "unmasked_signature": unmasked.signature,
        "assisted_signature": assisted.signature,
    }


def score_frame(
    frame: Frame,
    prediction: Any,
    memory: UnlabeledEntityMemory,
    assisted_slot: Optional[Pixel],
    assisted_pixels: FrozenSet[Pixel],
    *,
    threshold: float = LEARNED_MASK_PROBABILITY_THRESHOLD,
    appearance_threshold: float = APPEARANCE_L1_THRESHOLD,
) -> Dict[str, Any]:
    """Classify one frame and score learned-vs-assisted agreement on it.

    Pure given its inputs: the assisted detection and the tracker
    prediction are parameters so tests can drive synthetic fixtures.
    """

    sensitivity = frame_mask_sensitivity(
        frame, assisted_slot, assisted_pixels, memory
    )
    learned_cells = learned_mask_cells(prediction, threshold)
    learned_pixels = learned_pixel_mask(
        prediction, frame.width, frame.height, threshold
    )
    learned_slot = learned_reference_slot(
        prediction, frame.width, frame.height, threshold
    )
    comparison_cells = tuple(
        sorted(
            set(sensitivity["assisted_cells"])
            | set(
                cells_touched_by_pixels(
                    learned_pixels,
                    frame.width,
                    frame.height,
                    memory.columns,
                    memory.rows,
                )
            )
        )
    )
    assisted_quantities = masked_frame_quantities(
        frame, comparison_cells, assisted_slot, assisted_pixels, memory
    )
    learned_quantities = masked_frame_quantities(
        frame, comparison_cells, learned_slot, learned_pixels, memory
    )
    cell_l1 = {
        cell: UnlabeledEntityMemory.feature_distance(
            assisted_quantities.features[cell],
            learned_quantities.features[cell],
        )
        for cell in comparison_cells
    }
    max_cell_l1 = max(cell_l1.values()) if cell_l1 else 0.0
    signature_equal = bool(
        assisted_quantities.signature == learned_quantities.signature
    )
    l1_within = all(
        value <= appearance_threshold for value in cell_l1.values()
    )
    return {
        "frame": frame.digest,
        "mattering": sensitivity["mattering"],
        "reason": sensitivity["reason"],
        "assisted_slot": (
            None
            if assisted_slot is None
            else [assisted_slot[0], assisted_slot[1]]
        ),
        "assisted_cells": len(sensitivity["assisted_cells"]),
        "changed_cells": len(sensitivity["changed_cells"]),
        "learned_cells": len(learned_cells),
        "comparison_cells": len(comparison_cells),
        "signature_equal": signature_equal,
        "max_cell_l1": max_cell_l1,
        "l1_within": l1_within,
        "agrees": bool(signature_equal and l1_within),
        "mask_divergence": mask_divergence(learned_pixels, assisted_pixels),
    }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile of empty sequence")
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _distribution(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "p75": _quantile(ordered, 0.75),
        "max": ordered[-1],
    }


def _agreement_rates(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "frames": 0,
            "signature_agreement_rate": None,
            "fingerprint_l1_agreement_rate": None,
            "joint_agreement_rate": None,
        }
    return {
        "frames": len(rows),
        "signature_agreement_rate": (
            sum(row["signature_equal"] for row in rows) / len(rows)
        ),
        "fingerprint_l1_agreement_rate": (
            sum(row["l1_within"] for row in rows) / len(rows)
        ),
        "joint_agreement_rate": (
            sum(row["agrees"] for row in rows) / len(rows)
        ),
    }


def corpus_gate(
    mattering_rows: Sequence[Mapping[str, Any]],
    *,
    agreement_rate_threshold: float = AGREEMENT_RATE_THRESHOLD,
    minimum_mattering_frames: int = MINIMUM_MATTERING_FRAMES,
) -> Dict[str, Any]:
    """The preregistered per-corpus gate bits over mattering frames."""

    rates = _agreement_rates(mattering_rows)
    sufficient = len(mattering_rows) >= minimum_mattering_frames
    signature_bit = bool(
        sufficient
        and rates["signature_agreement_rate"] is not None
        and rates["signature_agreement_rate"] >= agreement_rate_threshold
    )
    fingerprint_bit = bool(
        sufficient
        and rates["fingerprint_l1_agreement_rate"] is not None
        and rates["fingerprint_l1_agreement_rate"] >= agreement_rate_threshold
    )
    return {
        "agreement_rate_threshold": agreement_rate_threshold,
        "minimum_mattering_frames": minimum_mattering_frames,
        "mattering_frames": len(mattering_rows),
        "mattering_frames_sufficient": sufficient,
        "signature_bit": signature_bit,
        "fingerprint_l1_bit": fingerprint_bit,
        "passed": bool(signature_bit and fingerprint_bit),
    }


def score_corpus(
    run_dir: Path,
    predictor: Any,
    *,
    threshold: float = LEARNED_MASK_PROBABILITY_THRESHOLD,
    appearance_threshold: float = APPEARANCE_L1_THRESHOLD,
) -> Dict[str, Any]:
    """Score every archived frame of one probe corpus.

    Frames are the corpus's content-addressed ``frames/*.png`` in sorted
    digest order (each unique frame counted once), digest-verified on
    decode.  The assisted detection runs fresh per frame with the
    goal-prior detector, exactly as the substitution replay's divergence
    sweep did.
    """

    run_dir = Path(run_dir)
    prior = PixelHeartGoalPrior()
    memory = UnlabeledEntityMemory()
    rows: List[Dict[str, Any]] = []
    for frame_path in sorted((run_dir / "frames").glob("*.png")):
        frame = decode_logged_png(frame_path)
        if frame.digest != frame_path.stem:
            raise ValueError(f"frame digest mismatch for {frame_path}")
        slot = prior.detect_player(frame)
        assisted_pixels: FrozenSet[Pixel] = frozenset(
            prior.player_pixel_mask(frame, slot) if slot is not None else ()
        )
        prediction = predictor.predict(frame)
        rows.append(
            score_frame(
                frame,
                prediction,
                memory,
                slot,
                assisted_pixels,
                threshold=threshold,
                appearance_threshold=appearance_threshold,
            )
        )
    mattering = [row for row in rows if row["mattering"]]
    non_mattering = [row for row in rows if not row["mattering"]]
    reasons: Dict[str, int] = {}
    for row in rows:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    mattering_ious = [
        row["mask_divergence"]["iou"]
        for row in mattering
        if row["mask_divergence"]["iou"] is not None
    ]
    return {
        "run_id": run_dir.name,
        "frames": len(rows),
        "frames_by_reason": {key: reasons[key] for key in sorted(reasons)},
        "mattering": {
            **_agreement_rates(mattering),
            "mask_iou": _distribution(mattering_ious),
            "max_cell_l1": _distribution(
                [row["max_cell_l1"] for row in mattering]
            ),
            "rows": [
                {
                    "frame": row["frame"],
                    "iou": row["mask_divergence"]["iou"],
                    "signature_equal": row["signature_equal"],
                    "max_cell_l1": row["max_cell_l1"],
                    "l1_within": row["l1_within"],
                    "agrees": row["agrees"],
                    "comparison_cells": row["comparison_cells"],
                    "changed_cells": row["changed_cells"],
                    "learned_cells": row["learned_cells"],
                }
                for row in mattering
            ],
        },
        "non_mattering": {
            **_agreement_rates(non_mattering),
            "disagreeing_frames": [
                row["frame"] for row in non_mattering if not row["agrees"]
            ],
        },
        "gate": corpus_gate(mattering),
    }


def _round_floats(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item, digits) for item in value]
    return value


def content_digest(payload: Mapping[str, Any]) -> str:
    """Deterministic digest over the report minus its own digest field."""

    body = {
        key: value
        for key, value in payload.items()
        if key != "content_digest"
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(
        (_DIGEST_PREFIX + canonical).encode("utf-8")
    ).hexdigest()


def build_report(
    corpus_results: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble the deterministic content-digested gate report."""

    per_corpus = {
        result["run_id"]: dict(result["gate"]) for result in corpus_results
    }
    passed = bool(corpus_results) and all(
        gate["passed"] for gate in per_corpus.values()
    )
    report = {
        "version": GATE_VERSION,
        "kind": GATE_KIND,
        "preregistration": "docs/wp5-tracker-training-2026-08-16.md",
        "basis": "docs/learnings.md section 4.31 plan-change item (c)",
        "design_principle": (
            "agreement is scored only on frames where the assisted mask "
            "demonstrably changes the downstream tracking quantities, "
            "fixing the section 4.31 mask-irrelevant-bits instrument lesson"
        ),
        "provenance": dict(provenance),
        "thresholds": {
            "learned_mask_probability": LEARNED_MASK_PROBABILITY_THRESHOLD,
            "learned_mask_probability_rationale": (
                "the checkpoint's validation operating point, as pinned by "
                "the substitution replay"
            ),
            "appearance_l1": APPEARANCE_L1_THRESHOLD,
            "appearance_l1_rationale": (
                "established appearance-match threshold: "
                "UnlabeledEntityMemory.match_threshold and "
                "AnonymousEntityBehaviorModel.appearance_match_threshold "
                "defaults"
            ),
            "agreement_rate": AGREEMENT_RATE_THRESHOLD,
            "minimum_mattering_frames": MINIMUM_MATTERING_FRAMES,
        },
        "corpora": [dict(result) for result in corpus_results],
        "result": {
            "per_corpus": per_corpus,
            "gate": "PASS" if passed else "FAIL",
            "verdict": (
                "PROMOTE-to-shadow (parity claim with divergence "
                "telemetry per Amendment B salvaged form)"
                if passed
                else "NO-PROMOTE (learned mask changes downstream "
                "tracking quantities on mask-mattering frames)"
            ),
        },
    }
    report = _round_floats(report)
    report["content_digest"] = content_digest(report)
    return report


def run_gate(
    corpus_dirs: Sequence[Path],
    checkpoint_path: Path,
    backbone_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    tracker, provenance = load_replay_tracker(checkpoint_path, backbone_path)
    results = [
        score_corpus(Path(run_dir), tracker) for run_dir in corpus_dirs
    ]
    report = build_report(results, provenance)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "WP5 mask-sensitive promotion gate: learned-vs-assisted mask "
            "agreement scored only on frames where masking demonstrably "
            "matters (learnings section 4.31 item c)"
        )
    )
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpora",
        default=None,
        help="probe corpus run directory (repeatable)",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    corpora = [
        Path(path) for path in (arguments.corpora or DEFAULT_CORPORA)
    ]
    report = run_gate(
        corpora,
        Path(arguments.checkpoint),
        Path(arguments.backbone),
        Path(arguments.report),
    )
    summary = {
        "gate": report["result"]["gate"],
        "verdict": report["result"]["verdict"],
        "per_corpus": report["result"]["per_corpus"],
        "content_digest": report["content_digest"],
        "report": str(arguments.report),
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
