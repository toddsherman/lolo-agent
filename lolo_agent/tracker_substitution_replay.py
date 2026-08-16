"""WP5 substitution-replay promotion gate (direction-review Amendment B).

Offline replay that reconstructs the archived v318/v321 object-track state
from ``human_prior_option_archive_added`` metadata TWICE -- once with the
assisted goal-prior player mask exactly as recorded, once with the LEARNED
controllable-tracker mask substituted for it -- and scores the
preregistered promotion bits from
``docs/wp5-tracker-training-2026-08-16.md``:

1. The confirmed manipulation identity (source ``(7, 6)``, destination
   ``(8, 6)``, direction, effect signature) reconstructs equivalently
   under the learned mask on every replayed archive.
2. The appearance fingerprint recovered at the destination matches the
   recorded one within the established L1 threshold (0.08).
3. Per-frame mask divergence (IoU / pixel counts) between the learned and
   assisted masks is reported, not gated.

Everything here is replay-only: the ``object_tracks`` pure functions take
``player_pixel_mask`` as a parameter, so both reconstructions call them
with different mask callables over the recorded frames.  No planner code
participates and no archive is modified.

Learned-mask thresholding rule (fixed ONCE, before scoring; not tuned on
the replayed archives)
----------------------------------------------------------------------

A coarse cell belongs to the learned player mask when the tracker
ensemble's mean per-cell probability is at least
``LEARNED_MASK_PROBABILITY_THRESHOLD`` = 0.5.  This is the checkpoint's
own validation operating point: ``validate_controllable_tracker``
computes its gated precision / recall / IoU statistics with
``predicted = probability >= 0.5``, so the uncapped checkpoint's recorded
validation numbers (precision 0.7587, recall 0.9789, IoU 0.7464) are
statements about exactly this decision rule and no other.  Cells at or
above the threshold expand to their pixel blocks at the recorded frame
resolution using the same integer grid partition as
``UnlabeledEntityMemory.feature_at`` (``x in [column * width // columns,
(column + 1) * width // columns)`` and likewise for rows), so masked
pixels and pooled appearance features always agree about cell boundaries.

Appearance comparator (bit 2)
-----------------------------

The archives record the destination appearance as a 16-hex state
signature (``human_prior_option_entity_state_signature`` /
``..._tracked_world_state_signature``), not as a raw feature vector, so
"matches the recorded one" is scored with a fixed precedence:

- If the destination state signature recovered under the learned mask
  reproduces the archived signature exactly, the recorded appearance has
  been recovered bit-for-bit (L1 distance 0) and the bit passes.
- Otherwise the learned-mask destination feature is compared against the
  assisted-replay destination feature -- the closest available
  reconstruction of the recorded convention -- with the established
  normalized-L1 appearance threshold 0.08
  (``UnlabeledEntityMemory.match_threshold`` and
  ``AnonymousEntityBehaviorModel.appearance_match_threshold`` defaults;
  the ``appearance_relation`` "same" bound).

Both comparisons are always reported, including whether the assisted
replay itself reproduces the archived signature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .entity_behavior import AnonymousEntityBehaviorModel
from .environment import Action
from .experience_import import decode_logged_png
from .goal_prior import PixelHeartGoalPrior
from .object_tracks import (
    ObjectTrackSet,
    direction_displacement,
    masked_cell_fingerprint,
    world_effect_cells_state_signature,
)
from .pixels import Frame
from .unlabeled_entities import UnlabeledEntityMemory

Cell = Tuple[int, int]
Pixel = Tuple[int, int]

REPLAY_VERSION = 1
REPLAY_KIND = "wp5-substitution-replay"
_DIGEST_PREFIX = f"{REPLAY_KIND}:v{REPLAY_VERSION}:"

# See the module docstring for the derivation of both constants.  They are
# fixed here so the replay cannot silently drift to a tuned operating
# point; the unit tests pin them.
LEARNED_MASK_PROBABILITY_THRESHOLD = 0.5
APPEARANCE_L1_THRESHOLD = 0.08

PREREGISTERED_SOURCE_CELL: Cell = (7, 6)
PREREGISTERED_DESTINATION_CELL: Cell = (8, 6)

ARCHIVE_EVENT = "human_prior_option_archive_added"

DEFAULT_ARCHIVES = (
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v318-room3-known-push-connected-mask-d2",
    "experiments/lolo1-entity-v10/evaluations/"
    "entity-v321-room3-confirmed-identity-d2",
)
DEFAULT_CHECKPOINT = "experiments/lolo1-wp5/controllable-tracker-v2-uncapped.pt"
DEFAULT_BACKBONE = (
    "experiments/lolo1-spatial-v10/checkpoints/spatial-v10-native-adapt-e5.pt"
)
DEFAULT_REPORT = "experiments/lolo1-wp5/substitution-replay-report.json"


@dataclass(frozen=True)
class ManipulationIdentity:
    """The preregistered confirmed-manipulation identity of one archive."""

    source_cell: Optional[Cell] = None
    destination_cell: Optional[Cell] = None
    direction: Optional[Action] = None
    effect_signature: str = ""

    @property
    def complete(self) -> bool:
        return (
            self.source_cell is not None
            and self.destination_cell is not None
            and self.direction is not None
            and bool(self.effect_signature)
        )

    def serialized(self) -> Dict[str, Any]:
        return {
            "source_cell": _serialized_cell(self.source_cell),
            "destination_cell": _serialized_cell(self.destination_cell),
            "direction": (
                None if self.direction is None else str(self.direction.value)
            ),
            "effect_signature": self.effect_signature or None,
        }


def _serialized_cell(cell: Optional[Cell]) -> Optional[List[int]]:
    return None if cell is None else [int(cell[0]), int(cell[1])]


def manipulation_identity(track_set: ObjectTrackSet) -> ManipulationIdentity:
    """Extract the scored identity subset of one reconstructed track set."""

    if not track_set.transitions:
        return ManipulationIdentity(
            effect_signature=track_set.confirmed_world_effect_signature
        )
    transition = track_set.transitions[0]
    delta = direction_displacement(transition.direction)
    destination: Optional[Cell] = None
    if transition.source_cell is not None and delta is not None:
        destination = (
            transition.source_cell[0] + delta[0],
            transition.source_cell[1] + delta[1],
        )
    return ManipulationIdentity(
        source_cell=transition.source_cell,
        destination_cell=destination,
        direction=transition.direction,
        effect_signature=track_set.confirmed_world_effect_signature,
    )


def identity_equivalent(
    first: ManipulationIdentity, second: ManipulationIdentity
) -> bool:
    """Bit-1 comparator: equal AND complete confirmed identities."""

    return first == second and first.complete


def grid_cell_pixel_block(
    cell: Cell, width: int, height: int, columns: int, rows: int
) -> FrozenSet[Pixel]:
    """Pixels of one coarse cell under the ``feature_at`` grid partition."""

    column, row = cell
    return frozenset(
        (x, y)
        for y in range(row * height // rows, (row + 1) * height // rows)
        for x in range(column * width // columns, (column + 1) * width // columns)
    )


def learned_mask_cells(
    prediction: Any,
    threshold: float = LEARNED_MASK_PROBABILITY_THRESHOLD,
) -> Tuple[Cell, ...]:
    """Cells whose ensemble mean probability reaches the fixed threshold."""

    return tuple(
        sorted(
            (column, row)
            for row in range(prediction.rows)
            for column in range(prediction.columns)
            if prediction.probabilities[row][column] >= threshold
        )
    )


def learned_pixel_mask(
    prediction: Any,
    width: int,
    height: int,
    threshold: float = LEARNED_MASK_PROBABILITY_THRESHOLD,
) -> FrozenSet[Pixel]:
    """Threshold a per-cell probability map to a recorded-resolution mask."""

    mask: set[Pixel] = set()
    for cell in learned_mask_cells(prediction, threshold):
        mask |= grid_cell_pixel_block(
            cell, width, height, prediction.columns, prediction.rows
        )
    return frozenset(mask)


def learned_reference_slot(
    prediction: Any,
    width: int,
    height: int,
    threshold: float = LEARNED_MASK_PROBABILITY_THRESHOLD,
) -> Optional[Pixel]:
    """A representative pixel anchor derived only from the learned map.

    ``world_effect_cells_state_signature`` and ``masked_cell_fingerprint``
    invoke their mask callable only when a player slot is provided.  The
    learned path derives that anchor from its own argmax cell so no
    assisted detection leaks into the substituted reconstruction; when no
    cell reaches the threshold the mask is empty and ``None`` keeps the
    reconstruction explicitly unmasked.
    """

    cells = learned_mask_cells(prediction, threshold)
    if not cells:
        return None
    best = max(
        cells,
        key=lambda cell: (
            prediction.probabilities[cell[1]][cell[0]],
            -cell[1],
            -cell[0],
        ),
    )
    return (
        best[0] * width // prediction.columns,
        best[1] * height // prediction.rows,
    )


def mask_divergence(
    learned: FrozenSet[Pixel], assisted: FrozenSet[Pixel]
) -> Dict[str, Any]:
    """Bit-3 telemetry: pixel counts and IoU between the two masks."""

    intersection = len(learned & assisted)
    union = len(learned | assisted)
    return {
        "learned_pixels": len(learned),
        "assisted_pixels": len(assisted),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": (intersection / union) if union else None,
    }


def appearance_comparison(
    recorded_signature: str,
    assisted_signature: str,
    learned_signature: str,
    assisted_feature: Sequence[int],
    learned_feature: Sequence[int],
    threshold: float = APPEARANCE_L1_THRESHOLD,
) -> Dict[str, Any]:
    """Bit-2 comparator with the fixed precedence from the module docstring."""

    l1 = UnlabeledEntityMemory.feature_distance(
        assisted_feature, learned_feature
    )
    learned_matches_recorded = bool(
        recorded_signature and learned_signature == recorded_signature
    )
    if learned_matches_recorded:
        basis = "exact-recorded-signature"
        passed = True
    else:
        basis = "l1-vs-assisted-reference"
        passed = l1 <= threshold
    return {
        "recorded_signature": recorded_signature or None,
        "assisted_signature": assisted_signature or None,
        "learned_signature": learned_signature or None,
        "assisted_matches_recorded": bool(
            recorded_signature and assisted_signature == recorded_signature
        ),
        "learned_matches_recorded": learned_matches_recorded,
        "l1_assisted_vs_learned": l1,
        "l1_threshold": threshold,
        "basis": basis,
        "passed": passed,
    }


def find_confirmed_archive_events(events_path: Path) -> List[Dict[str, Any]]:
    """Archive-added events carrying a confirmed world-effect signature."""

    events: List[Dict[str, Any]] = []
    with Path(events_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("event") != ARCHIVE_EVENT:
                continue
            if not event.get("human_prior_option_world_effect_signature"):
                continue
            if not event.get("human_prior_option_effect_frontier"):
                continue
            events.append(event)
    return events


def _recorded_destination_signature(
    event: Mapping[str, Any], destination: Optional[Cell]
) -> str:
    """The archived destination appearance signature, when one was recorded."""

    entity_signature = str(
        event.get("human_prior_option_entity_state_signature") or ""
    )
    if entity_signature:
        return entity_signature
    tracked_cells = tuple(
        sorted(
            (int(value[0]), int(value[1]))
            for value in (
                event.get("human_prior_option_tracked_world_effect_cells")
                or ()
            )
        )
    )
    if destination is not None and tracked_cells == (destination,):
        return str(
            event.get("human_prior_option_tracked_world_state_signature")
            or ""
        )
    return ""


def replay_archive_event(
    run_dir: Path,
    event: Mapping[str, Any],
    predictor: Any,
    *,
    threshold: float = LEARNED_MASK_PROBABILITY_THRESHOLD,
    appearance_threshold: float = APPEARANCE_L1_THRESHOLD,
) -> Dict[str, Any]:
    """Replay one confirmed archive event under both masking conventions.

    ``predictor`` only needs a ``predict(frame)`` returning per-cell
    ``probabilities`` with ``columns``/``rows``; tests substitute a stub.
    """

    run_dir = Path(run_dir)
    frame_digest = str(event["frame"])
    frame = decode_logged_png(run_dir / "frames" / f"{frame_digest}.png")
    if frame.digest != frame_digest:
        raise ValueError(
            f"archived frame digest mismatch for {run_dir.name}:{frame_digest}"
        )

    # Assisted reconstruction, exactly as recorded: the goal-prior pixel
    # mask around the event's recorded player detection.
    prior = PixelHeartGoalPrior()
    recorded_slot = event.get("human_prior_target_player_slot")
    assisted_slot: Optional[Pixel] = (
        None
        if recorded_slot is None
        else (int(recorded_slot[0]), int(recorded_slot[1]))
    )
    assisted_pixels: FrozenSet[Pixel] = frozenset(
        prior.player_pixel_mask(frame, assisted_slot)
        if assisted_slot is not None
        else ()
    )

    # Learned reconstruction: the thresholded tracker map substituted for
    # the assisted mask; the anchor slot comes from the map itself.
    prediction = predictor.predict(frame)
    learned_cells = learned_mask_cells(prediction, threshold)
    learned_pixels = learned_pixel_mask(
        prediction, frame.width, frame.height, threshold
    )
    learned_slot = learned_reference_slot(
        prediction, frame.width, frame.height, threshold
    )

    def reconstruct(
        slot: Optional[Pixel],
        mask_pixels: FrozenSet[Pixel],
    ) -> Tuple[ObjectTrackSet, UnlabeledEntityMemory, Any]:
        memory = UnlabeledEntityMemory()
        model = AnonymousEntityBehaviorModel()

        def mask_callable(
            _frame: Frame, _slot: Pixel
        ) -> FrozenSet[Pixel]:
            return mask_pixels

        tracks = ObjectTrackSet.from_archive_metadata(
            event,
            columns=memory.columns,
            tracked_state_resolver=lambda cells: (
                world_effect_cells_state_signature(
                    frame, cells, slot, memory, mask_callable
                )
            ),
            fingerprint_resolver=lambda cell: masked_cell_fingerprint(
                frame, cell, slot, memory, model, mask_callable
            ),
        )
        return tracks, memory, model

    assisted_tracks, memory, _model = reconstruct(
        assisted_slot, assisted_pixels
    )
    learned_tracks, _memory, _model = reconstruct(learned_slot, learned_pixels)

    assisted_identity = manipulation_identity(assisted_tracks)
    learned_identity = manipulation_identity(learned_tracks)
    destination = learned_identity.destination_cell

    def destination_appearance(
        slot: Optional[Pixel], mask_pixels: FrozenSet[Pixel]
    ) -> Tuple[Tuple[int, ...], str, str]:
        if destination is None:
            return (), "", ""
        ignored = mask_pixels if slot is not None else None
        feature = memory.feature_at(frame, *destination, ignored)
        signature = world_effect_cells_state_signature(
            frame,
            (destination,),
            slot,
            memory,
            lambda _frame, _slot: mask_pixels,
        )
        fingerprint = AnonymousEntityBehaviorModel.appearance_fingerprint(
            feature
        )
        return feature, signature, fingerprint

    assisted_feature, assisted_signature, assisted_fingerprint = (
        destination_appearance(assisted_slot, assisted_pixels)
    )
    learned_feature, learned_signature, learned_fingerprint = (
        destination_appearance(learned_slot, learned_pixels)
    )
    recorded_signature = _recorded_destination_signature(event, destination)

    identity_bit = {
        "assisted_identity": assisted_identity.serialized(),
        "learned_identity": learned_identity.serialized(),
        "equivalent": identity_equivalent(assisted_identity, learned_identity),
        "matches_preregistered_cells": bool(
            learned_identity.source_cell == PREREGISTERED_SOURCE_CELL
            and learned_identity.destination_cell
            == PREREGISTERED_DESTINATION_CELL
        ),
        "effect_signature_matches_recorded": bool(
            learned_identity.effect_signature
            == str(
                event.get("human_prior_option_world_effect_signature") or ""
            )
        ),
    }
    identity_bit["passed"] = bool(
        identity_bit["equivalent"]
        and identity_bit["matches_preregistered_cells"]
        and identity_bit["effect_signature_matches_recorded"]
    )

    appearance_bit = appearance_comparison(
        recorded_signature,
        assisted_signature,
        learned_signature,
        assisted_feature,
        learned_feature,
        appearance_threshold,
    )
    appearance_bit["assisted_fingerprint"] = assisted_fingerprint or None
    appearance_bit["learned_fingerprint"] = learned_fingerprint or None
    appearance_bit["fingerprints_equal"] = bool(
        assisted_fingerprint
        and assisted_fingerprint == learned_fingerprint
    )

    return {
        "run_id": run_dir.name,
        "seq": event.get("seq"),
        "frame": frame_digest,
        "path": list(event.get("path") or ()),
        "recorded": {
            "target_player_slot": (
                None
                if assisted_slot is None
                else [assisted_slot[0], assisted_slot[1]]
            ),
            "entity_state_signature": (
                event.get("human_prior_option_entity_state_signature")
            ),
            "tracked_world_state_signature": (
                event.get("human_prior_option_tracked_world_state_signature")
            ),
            "tracked_world_effect_cells": (
                event.get("human_prior_option_tracked_world_effect_cells")
            ),
        },
        "assisted": {
            "player_slot": (
                None
                if assisted_slot is None
                else [assisted_slot[0], assisted_slot[1]]
            ),
            "mask_pixels": len(assisted_pixels),
            "destination_state_signature": assisted_signature or None,
            "track_set_signature": assisted_tracks.signature,
        },
        "learned": {
            "threshold": threshold,
            "mask_cells": [_serialized_cell(cell) for cell in learned_cells],
            "mask_pixels": len(learned_pixels),
            "reference_slot": (
                None
                if learned_slot is None
                else [learned_slot[0], learned_slot[1]]
            ),
            "destination_state_signature": learned_signature or None,
            "track_set_signature": learned_tracks.signature,
        },
        "track_set_signatures_equal": bool(
            assisted_tracks.signature == learned_tracks.signature
        ),
        "bits": {
            "reconstruction_identity": identity_bit,
            "destination_appearance": appearance_bit,
        },
        "endpoint_mask_divergence": mask_divergence(
            learned_pixels, assisted_pixels
        ),
    }


def divergence_sweep(
    run_dir: Path,
    predictor: Any,
    *,
    threshold: float = LEARNED_MASK_PROBABILITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Bit-3 telemetry over every archived frame of one evaluation run."""

    run_dir = Path(run_dir)
    prior = PixelHeartGoalPrior()
    rows: List[Dict[str, Any]] = []
    for frame_path in sorted((run_dir / "frames").glob("*.png")):
        frame = decode_logged_png(frame_path)
        prediction = predictor.predict(frame)
        learned = learned_pixel_mask(
            prediction, frame.width, frame.height, threshold
        )
        slot = prior.detect_player(frame)
        assisted: FrozenSet[Pixel] = frozenset(
            prior.player_pixel_mask(frame, slot) if slot is not None else ()
        )
        row = {
            "frame": frame_path.stem,
            "player_detected": slot is not None,
            "player_slot": None if slot is None else [slot[0], slot[1]],
            "learned_cells": len(learned_mask_cells(prediction, threshold)),
        }
        row.update(mask_divergence(learned, assisted))
        rows.append(row)
    return rows


def _summarize_sweep(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scored = [row["iou"] for row in rows if row["iou"] is not None]
    detected = [row for row in rows if row["player_detected"]]
    return {
        "frames": len(rows),
        "frames_with_detected_player": len(detected),
        "iou_mean": (sum(scored) / len(scored)) if scored else None,
        "iou_min": min(scored) if scored else None,
        "iou_max": max(scored) if scored else None,
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
    archive_results: Sequence[Mapping[str, Any]],
    sweeps: Mapping[str, Sequence[Mapping[str, Any]]],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    bit1_all = all(
        result["bits"]["reconstruction_identity"]["passed"]
        for result in archive_results
    )
    bit2_all = all(
        result["bits"]["destination_appearance"]["passed"]
        for result in archive_results
    )
    report = {
        "version": REPLAY_VERSION,
        "kind": REPLAY_KIND,
        "preregistration": "docs/wp5-tracker-training-2026-08-16.md",
        "provenance": dict(provenance),
        "thresholds": {
            "learned_mask_probability": LEARNED_MASK_PROBABILITY_THRESHOLD,
            "learned_mask_probability_rationale": (
                "validation operating point: validate_controllable_tracker "
                "scores precision/recall/IoU at probability >= 0.5, so the "
                "checkpoint's gated validation statistics pin this rule"
            ),
            "appearance_l1": APPEARANCE_L1_THRESHOLD,
            "appearance_l1_rationale": (
                "established appearance-match threshold: "
                "UnlabeledEntityMemory.match_threshold and "
                "AnonymousEntityBehaviorModel.appearance_match_threshold "
                "defaults; appearance_relation 'same' bound"
            ),
        },
        "archives": [dict(result) for result in archive_results],
        "divergence_telemetry": {
            run_id: {
                "summary": _summarize_sweep(rows),
                "frames": [dict(row) for row in rows],
            }
            for run_id, rows in sorted(sweeps.items())
        },
        "result": {
            "bit1_reconstruction_identity_all": bit1_all,
            "bit2_destination_appearance_all": bit2_all,
            "bit3_divergence": "reported-not-gated",
            "promotion_gate": "PASS" if (bit1_all and bit2_all) else "FAIL",
        },
    }
    report = _round_floats(report)
    report["content_digest"] = content_digest(report)
    return report


def load_replay_tracker(
    checkpoint_path: Path, backbone_path: Path
) -> Tuple[Any, Dict[str, Any]]:
    """Load the pinned tracker on CPU; imported lazily to keep tests light."""

    from .controllable_tracker import load_controllable_tracker_checkpoint
    from .spatial_world_model import load_spatial_checkpoint

    backbone, _planning_horizon = load_spatial_checkpoint(
        Path(backbone_path), device="cpu", frozen=True
    )
    tracker, provenance = load_controllable_tracker_checkpoint(
        Path(checkpoint_path), backbone, device="cpu", frozen=True
    )
    provenance = dict(provenance)
    provenance["checkpoint"] = str(checkpoint_path)
    provenance["checkpoint_parameter_sha256"] = tracker.checkpoint_digest
    provenance["backbone_checkpoint"] = str(backbone_path)
    return tracker, provenance


def run_replay(
    archive_dirs: Sequence[Path],
    checkpoint_path: Path,
    backbone_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    tracker, provenance = load_replay_tracker(checkpoint_path, backbone_path)
    archive_results: List[Dict[str, Any]] = []
    sweeps: Dict[str, List[Dict[str, Any]]] = {}
    for run_dir in archive_dirs:
        run_dir = Path(run_dir)
        events = find_confirmed_archive_events(run_dir / "events.jsonl")
        if not events:
            raise ValueError(
                f"no confirmed archive-added event found in {run_dir}"
            )
        for event in events:
            archive_results.append(
                replay_archive_event(run_dir, event, tracker)
            )
        sweeps[run_dir.name] = divergence_sweep(run_dir, tracker)
    report = build_report(archive_results, sweeps, provenance)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "WP5 substitution-replay promotion gate (Amendment B): replay "
            "archived track reconstruction with the learned mask "
            "substituted for the assisted player mask"
        )
    )
    parser.add_argument(
        "--archive",
        action="append",
        dest="archives",
        default=None,
        help="evaluation archive directory (repeatable)",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    archives = [
        Path(path) for path in (arguments.archives or DEFAULT_ARCHIVES)
    ]
    report = run_replay(
        archives,
        Path(arguments.checkpoint),
        Path(arguments.backbone),
        Path(arguments.report),
    )
    result = report["result"]
    print(
        json.dumps(
            {
                "promotion_gate": result["promotion_gate"],
                "bit1_reconstruction_identity_all": result[
                    "bit1_reconstruction_identity_all"
                ],
                "bit2_destination_appearance_all": result[
                    "bit2_destination_appearance_all"
                ],
                "content_digest": report["content_digest"],
                "report": str(arguments.report),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
