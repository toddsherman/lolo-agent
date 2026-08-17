"""WP5 pixel-mask spike entry points: training and the gate rerun.

Two subcommands:

``train``
    Trains the pixel-silhouette refinement head (``pixel_mask_head``) on
    pixel-level counterfactual silhouette targets derived from the pinned
    cell-label corpus, over the FROZEN tracker v4 and FROZEN spatial
    backbone -- only the new head receives gradients.  Follows the shared
    trainer conventions: run-held-out hash-stable splits (the cell
    trainer's own split and sampling helpers, reused), an untrained
    same-architecture baseline gate, atomic JSON metrics, deterministic
    checkpoint digests with pinned provenance, and an internal wall-clock
    ceiling so an external watchdog never has to kill a run mid-epoch.

``gate``
    Reruns the preregistered mask-sensitive promotion gate
    (``lolo_agent.mask_sensitive_gate``) UNCHANGED -- same mattering-frame
    detector, same scored quantities, same thresholds, same corpora --
    with exactly one substitution: the learned mask source is the
    reconstructed pixel silhouette (frozen tracker v4 anchor + refinement
    head + fixed halo) instead of tracker v4's thresholded cell blocks.
    The substitution mechanism is the predictor protocol the gate already
    exposes: ``score_corpus`` consumes any ``predict(frame)`` whose
    prediction carries a per-unit probability grid, and a pixel-resolution
    grid (one grid unit per pixel) makes the gate's own unchanged helpers
    recover the reconstructed pixel mask at the pinned 0.5 threshold.

Lineage note: the ``gate`` subcommand imports the gate instrument, which
is assisted-coupled by design (its mattering-frame detector uses the
recorded assisted mask as evaluation ground truth), so THIS module is
assisted-coupled through those evaluation imports and nothing else.  The
entire training derivation -- labels, head, training loop, checkpoint --
lives in ``lolo_agent.pixel_mask_head``, which references no assisted
symbol (linted by the unit tests).  No assisted quantity flows into
training data or head parameters.

Smoke usage (read-only against the strict store)::

    python -m lolo_agent.pixel_mask_train train \
        --labels experiments/lolo1-wp5/wp5-labels-full-v4.jsonl \
        --dataset experiments/lolo1-medium/dataset \
        --tracker-checkpoint experiments/lolo1-wp5/controllable-tracker-v4.pt \
        --spatial-checkpoint \
            experiments/lolo1-spatial-v10/checkpoints/spatial-v10-native-adapt-e5.pt \
        --checkpoint /tmp/pixel-head-smoke.pt \
        --max-training-arms 64 --max-validation-arms 16 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .controllable_tracker import (
    arm_examples_from_records,
    load_controllable_tracker_checkpoint,
    sample_arm_examples,
)
from .counterfactual_labels import open_strict_store
from .ensemble_world_model import split_sequence_runs
# Evaluation-instrument imports (assisted-coupled BY DESIGN; used only by
# the ``gate`` subcommand, never by training): the unchanged
# mask-sensitive gate scorer and the substitution replay's pinned loader.
from .mask_sensitive_gate import (
    DEFAULT_CHECKPOINT as GATE_TRACKER_CHECKPOINT,
    DEFAULT_CORPORA as GATE_CORPORA,
    build_report,
    score_corpus,
)
from .neural_world_model import choose_torch_device
from .pixel_mask_head import (
    CHECKPOINT_ARCHITECTURE,
    EXCLUDED_INPUTS,
    PERSISTENT_INPUTS,
    REWARD_TRACK,
    PixelMaskHead,
    PixelSilhouettePredictor,
    attach_cell_probabilities,
    load_label_records,
    load_pixel_mask_head_checkpoint,
    pixel_examples_from_store,
    pixel_targets_digest,
    save_pixel_mask_head_checkpoint,
    train_pixel_mask_head,
    validate_pixel_mask_head,
)
from .run_logging import sha256_file
from .spatial_world_model import load_spatial_checkpoint
from .tracker_substitution_replay import (
    DEFAULT_BACKBONE,
    load_replay_tracker,
)

DEFAULT_HEAD_CHECKPOINT = "experiments/lolo1-wp5/pixel-mask-head-v1.pt"
DEFAULT_GATE_REPORT = "experiments/lolo1-wp5/mask-sensitive-gate-v2-report.json"

MASK_SOURCE = (
    "reconstructed-pixel-silhouette: frozen tracker v4 cell anchor "
    "(threshold 0.5, dilation 1 cell) + pixel_mask_head positives "
    "(threshold 0.5) + Chebyshev halo dilation 3"
)
SUBSTITUTION_MECHANISM = (
    "lolo_agent.mask_sensitive_gate.score_corpus and build_report run "
    "unchanged; the predictor returns a pixel-resolution prediction "
    "(columns=width, rows=height, one grid unit per pixel) whose "
    "probability grid is the reconstructed mask indicator, so the gate's "
    "unchanged helpers and pinned 0.5 threshold recover exactly the "
    "reconstructed pixel mask"
)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if min(
        args.max_training_arms,
        args.max_validation_arms,
        args.epochs,
        args.batch_size,
        args.hidden_size,
        args.root_batch_size,
    ) <= 0:
        parser.error("model, data, and training sizes must be positive")
    if args.validation_modulus < 2:
        parser.error("--validation-modulus must be at least two")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.positive_weight <= 0.0 or args.residual_weight <= 0.0:
        parser.error("pixel loss weights must be positive")
    if args.wall_clock_ceiling <= 0.0:
        parser.error("--wall-clock-ceiling must be positive")

    started = time.monotonic()
    torch.manual_seed(args.seed)
    device = choose_torch_device()
    store = open_strict_store(args.dataset)
    records, label_manifest = load_label_records(args.labels)
    if label_manifest.get("reward_track") != REWARD_TRACK:
        parser.error("label corpus is not bound to the strict reward track")
    cell_examples, cell_statistics = arm_examples_from_records(records)
    if not cell_examples:
        parser.error("label corpus contains no usable labeled arms")
    columns = cell_examples[0].columns
    rows = cell_examples[0].rows

    backbone, planning_horizon = load_spatial_checkpoint(
        args.spatial_checkpoint, device=device, frozen=True
    )
    backbone_digest = backbone.checkpoint_digest
    tracker, tracker_provenance = load_controllable_tracker_checkpoint(
        args.tracker_checkpoint, backbone, device=device, frozen=True
    )
    tracker_digest = tracker.checkpoint_digest
    if (tracker.columns, tracker.rows) != (columns, rows):
        parser.error("tracker cell grid does not match the label grid")

    training_cells, validation_cells = split_sequence_runs(
        cell_examples, validation_modulus=args.validation_modulus
    )
    training_cells = sample_arm_examples(
        training_cells, args.max_training_arms, args.seed
    )
    validation_cells = sample_arm_examples(
        validation_cells, args.max_validation_arms, args.seed + 1
    )
    training, training_statistics = pixel_examples_from_store(
        store, records, training_cells, root_batch_size=args.root_batch_size
    )
    validation, validation_statistics = pixel_examples_from_store(
        store, records, validation_cells, root_batch_size=args.root_batch_size
    )
    if not training or not validation:
        parser.error("pixel target derivation produced an empty split")
    training = attach_cell_probabilities(
        tracker, training, device, args.batch_size
    )
    validation = attach_cell_probabilities(
        tracker, validation, device, args.batch_size
    )
    if tracker.checkpoint_digest != tracker_digest:
        raise SystemExit("cell-map attachment changed frozen tracker parameters")
    targets_digest = pixel_targets_digest(list(training) + list(validation))

    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        baseline = PixelMaskHead(hidden_size=args.hidden_size)
    before = validate_pixel_mask_head(
        baseline, validation, device, args.batch_size
    )
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        head = PixelMaskHead(hidden_size=args.hidden_size)
    history: List[Any] = []
    completed_epochs = 0
    ceiling_hit = False
    for epoch in range(args.epochs):
        elapsed = time.monotonic() - started
        if elapsed >= args.wall_clock_ceiling:
            ceiling_hit = True
            break
        history.extend(
            train_pixel_mask_head(
                head,
                training,
                device,
                epochs=1,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed + epoch,
                positive_weight=args.positive_weight,
                residual_weight=args.residual_weight,
            )
        )
        completed_epochs += 1
    if not history:
        raise SystemExit(
            "wall-clock ceiling left no time for a single training epoch"
        )
    after = validate_pixel_mask_head(head, validation, device, args.batch_size)
    if backbone.checkpoint_digest != backbone_digest:
        raise SystemExit("training changed frozen backbone parameters")
    if tracker.checkpoint_digest != tracker_digest:
        raise SystemExit("training changed frozen tracker parameters")
    head.freeze()
    frozen_digest = head.checkpoint_digest
    checkpoint_digest = save_pixel_mask_head_checkpoint(
        head,
        args.checkpoint,
        label_manifest_digest=str(label_manifest["content_digest"]),
        pixel_targets_sha256=targets_digest,
        tracker_parameter_digest=tracker_digest,
        backbone_parameter_digest=backbone_digest,
        cell_columns=columns,
        cell_rows=rows,
    )
    if checkpoint_digest != frozen_digest:
        raise SystemExit("freezing or checkpointing changed head parameters")

    residual_separated = after.residual_pixels == 0 or (
        after.mean_target_probability > after.mean_residual_probability
    )
    gate_passed = (
        after.loss < before.loss
        and after.roc_auc > before.roc_auc
        and after.mean_target_probability > after.mean_background_probability
        and residual_separated
    )
    metrics = {
        "version": 1,
        "architecture": CHECKPOINT_ARCHITECTURE,
        "reward_track": REWARD_TRACK,
        "persistent_inputs": list(PERSISTENT_INPUTS),
        "excluded_inputs": list(EXCLUDED_INPUTS),
        "device": str(device),
        "seed": args.seed,
        "dataset": str(store.root),
        "labels": {
            "path": str(args.labels),
            "file_sha256": sha256_file(args.labels),
            "manifest": label_manifest,
            "cell_statistics": cell_statistics,
        },
        "pixel_targets": {
            "content_sha256": targets_digest,
            "training": training_statistics,
            "validation": validation_statistics,
        },
        "grid": {"columns": columns, "rows": rows},
        "frame_geometry": {
            "width": training[0].width,
            "height": training[0].height,
        },
        "spatial_checkpoint": {
            "path": str(args.spatial_checkpoint),
            "file_sha256": sha256_file(args.spatial_checkpoint),
            "parameter_sha256": backbone_digest,
            "planning_horizon": planning_horizon,
        },
        "tracker_checkpoint": {
            "path": str(args.tracker_checkpoint),
            "file_sha256": sha256_file(args.tracker_checkpoint),
            "parameter_sha256": tracker_digest,
            "provenance": tracker_provenance,
        },
        "training_configuration": {
            "epochs": args.epochs,
            "completed_epochs": completed_epochs,
            "wall_clock_ceiling_seconds": args.wall_clock_ceiling,
            "wall_clock_ceiling_hit": ceiling_hit,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "positive_weight": args.positive_weight,
            "residual_weight": args.residual_weight,
            "validation_modulus": args.validation_modulus,
            "max_training_arms": args.max_training_arms,
            "max_validation_arms": args.max_validation_arms,
            "root_batch_size": args.root_batch_size,
        },
        "training_source_runs": sorted(
            {item.source_run_id for item in training}
        ),
        "validation_source_runs": sorted(
            {item.source_run_id for item in validation}
        ),
        "training_examples": len(training),
        "validation_examples": len(validation),
        "updates": len(history),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "first_training_metrics": asdict(history[0]),
        "final_training_metrics": asdict(history[-1]),
        "validation_gate_baseline": asdict(before),
        "validation_after": asdict(after),
        "held_out_gate": {
            "passed": gate_passed,
            "baseline": "untrained_same_architecture",
            "requirements": [
                "lower held-out per-pixel loss than the untrained head",
                "higher pixel ROC AUC than the untrained head",
                "higher mean probability on silhouette pixels than background",
                (
                    "higher mean probability on silhouette pixels than "
                    "residual pixels when residual pixels exist"
                ),
            ],
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_parameter_sha256": checkpoint_digest,
    }
    metrics_path = args.metrics or args.checkpoint.with_suffix(".metrics.json")
    _atomic_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _gate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    tracker, replay_provenance = load_replay_tracker(
        Path(args.tracker_checkpoint), Path(args.backbone)
    )
    head, head_provenance = load_pixel_mask_head_checkpoint(
        Path(args.head_checkpoint), device="cpu", frozen=True
    )
    if head_provenance["tracker_parameter_sha256"] != tracker.checkpoint_digest:
        parser.error(
            "pixel head was trained against a different tracker checkpoint"
        )
    if (
        head_provenance["backbone_parameter_sha256"]
        != replay_provenance["backbone_parameter_sha256"]
    ):
        parser.error(
            "pixel head was trained against a different spatial backbone"
        )
    if (
        head_provenance["cell_columns"],
        head_provenance["cell_rows"],
    ) != (tracker.columns, tracker.rows):
        parser.error("pixel head cell grid does not match the tracker grid")
    predictor = PixelSilhouettePredictor(tracker, head, device="cpu")
    results = [
        score_corpus(Path(run_dir), predictor) for run_dir in args.corpora
    ]
    provenance = dict(replay_provenance)
    provenance.update(
        {
            "pixel_mask_head_checkpoint": str(args.head_checkpoint),
            "pixel_mask_head_parameter_sha256": head.checkpoint_digest,
            "pixel_mask_head_label_manifest_sha256": head_provenance[
                "label_manifest_sha256"
            ],
            "pixel_mask_head_pixel_targets_sha256": head_provenance[
                "pixel_targets_sha256"
            ],
            "mask_source": MASK_SOURCE,
            "substitution_mechanism": SUBSTITUTION_MECHANISM,
        }
    )
    report = build_report(results, provenance)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    summary = {
        "gate": report["result"]["gate"],
        "verdict": report["result"]["verdict"],
        "per_corpus": report["result"]["per_corpus"],
        "content_digest": report["content_digest"],
        "report": str(args.report),
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "WP5 pixel-mask spike: train the pixel-silhouette refinement "
            "head, or rerun the unchanged mask-sensitive gate with the "
            "reconstructed pixel mask as the learned mask source"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser(
        "train", help="train the pixel-silhouette refinement head"
    )
    train.add_argument("--labels", type=Path, required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--tracker-checkpoint", type=Path, required=True)
    train.add_argument("--spatial-checkpoint", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--metrics", type=Path)
    train.add_argument("--max-training-arms", type=int, default=6000)
    train.add_argument("--max-validation-arms", type=int, default=1500)
    train.add_argument("--validation-modulus", type=int, default=5)
    train.add_argument("--root-batch-size", type=int, default=256)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--hidden-size", type=int, default=32)
    train.add_argument("--positive-weight", type=float, default=8.0)
    train.add_argument("--residual-weight", type=float, default=4.0)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--wall-clock-ceiling", type=float, default=2100.0)

    gate = commands.add_parser(
        "gate",
        help=(
            "rerun the unchanged mask-sensitive gate with the "
            "reconstructed pixel mask substituted as the learned mask"
        ),
    )
    gate.add_argument(
        "--corpus",
        action="append",
        dest="corpora",
        default=None,
        help="probe corpus run directory (repeatable)",
    )
    gate.add_argument("--head-checkpoint", default=DEFAULT_HEAD_CHECKPOINT)
    gate.add_argument("--tracker-checkpoint", default=GATE_TRACKER_CHECKPOINT)
    gate.add_argument("--backbone", default=DEFAULT_BACKBONE)
    gate.add_argument("--report", default=DEFAULT_GATE_REPORT)

    arguments = parser.parse_args(argv)
    if arguments.command == "train":
        return _train(arguments, parser)
    if arguments.corpora is None:
        arguments.corpora = list(GATE_CORPORA)
    return _gate(arguments, parser)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
