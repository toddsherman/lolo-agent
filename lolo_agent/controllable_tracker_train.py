"""Training entry for the WP5 controllable-region tracker head.

Distills counterfactual controllable-region pseudo-labels (see
``counterfactual_labels``) into a per-cell mask head over the frozen
spatial world-model encoder.  Follows the shared trainer conventions:
run-held-out hash-stable splits, an untrained same-architecture baseline
gate, atomic JSON metrics, and deterministic checkpoint digests.  The
default learning rate follows the recorded durable-experiment finding
that ``1e-5`` was the only rate to improve every held-out horizon
(docs/runpod-platform-gate-2026-08-15.md).

Smoke usage (read-only against the strict store)::

    python -m lolo_agent.controllable_tracker_train \
        --labels experiments/lolo1-wp5/wp5-labels-full.jsonl \
        --dataset experiments/lolo1-medium/dataset \
        --spatial-checkpoint \
            experiments/lolo1-spatial-v10/checkpoints/spatial-v10-native-adapt-e5.pt \
        --checkpoint /tmp/tracker-smoke.pt --max-training-arms 400
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from .controllable_tracker import (
    CHECKPOINT_ARCHITECTURE,
    EXCLUDED_INPUTS,
    PERSISTENT_INPUTS,
    REWARD_TRACK,
    ControllableRegionTracker,
    decode_arm_examples,
    load_labeled_arm_examples,
    sample_arm_examples,
    save_controllable_tracker_checkpoint,
    train_controllable_tracker,
    validate_controllable_tracker,
)
from .counterfactual_labels import open_strict_store
from .ensemble_world_model import split_sequence_runs
from .neural_world_model import choose_torch_device
from .run_logging import sha256_file
from .spatial_world_model import load_spatial_checkpoint


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the controllable-region mask head over a frozen spatial "
            "encoder from counterfactual pseudo-labels"
        )
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spatial-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--max-training-arms", type=int, default=4000)
    parser.add_argument("--max-validation-arms", type=int, default=1000)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--residual-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(
        args.max_training_arms,
        args.max_validation_arms,
        args.epochs,
        args.batch_size,
        args.hidden_size,
        args.ensemble_size,
    ) <= 0:
        parser.error("model, data, and training sizes must be positive")
    if args.validation_modulus < 2:
        parser.error("--validation-modulus must be at least two")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.positive_weight <= 0.0 or args.residual_weight <= 0.0:
        parser.error("cell loss weights must be positive")

    torch.manual_seed(args.seed)
    device = choose_torch_device()
    store = open_strict_store(args.dataset)
    examples, label_manifest, label_statistics = load_labeled_arm_examples(
        args.labels
    )
    if label_manifest.get("reward_track") != REWARD_TRACK:
        parser.error("label corpus is not bound to the strict reward track")
    if not examples:
        parser.error("label corpus contains no usable labeled arms")
    columns = examples[0].columns
    rows = examples[0].rows
    backbone, planning_horizon = load_spatial_checkpoint(
        args.spatial_checkpoint, device=device, frozen=True
    )
    backbone_digest = backbone.checkpoint_digest

    training_examples, validation_examples = split_sequence_runs(
        examples, validation_modulus=args.validation_modulus
    )
    training_examples = sample_arm_examples(
        training_examples, args.max_training_arms, args.seed
    )
    validation_examples = sample_arm_examples(
        validation_examples, args.max_validation_arms, args.seed + 1
    )
    training = decode_arm_examples(store, training_examples)
    validation = decode_arm_examples(store, validation_examples)

    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        baseline = ControllableRegionTracker(
            backbone,
            hidden_size=args.hidden_size,
            ensemble_size=args.ensemble_size,
            columns=columns,
            rows=rows,
        )
    before = validate_controllable_tracker(
        baseline, validation, device, args.batch_size
    )
    tracker = ControllableRegionTracker(
        backbone,
        hidden_size=args.hidden_size,
        ensemble_size=args.ensemble_size,
        columns=columns,
        rows=rows,
    )
    history = train_controllable_tracker(
        tracker,
        training,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        positive_weight=args.positive_weight,
        residual_weight=args.residual_weight,
    )
    after = validate_controllable_tracker(
        tracker, validation, device, args.batch_size
    )
    if backbone.checkpoint_digest != backbone_digest:
        raise SystemExit("training changed frozen backbone parameters")
    tracker.freeze()
    frozen_digest = tracker.checkpoint_digest
    checkpoint_digest = save_controllable_tracker_checkpoint(
        tracker,
        args.checkpoint,
        label_manifest_digest=str(label_manifest["content_digest"]),
    )
    if checkpoint_digest != frozen_digest:
        raise SystemExit("freezing or checkpointing changed tracker parameters")

    residual_separated = after.residual_cells == 0 or (
        after.mean_controllable_probability > after.mean_residual_probability
    )
    gate_passed = (
        after.loss < before.loss
        and after.roc_auc > before.roc_auc
        and after.mean_controllable_probability
        > after.mean_background_probability
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
            "statistics": label_statistics,
        },
        "grid": {"columns": columns, "rows": rows},
        "spatial_checkpoint": {
            "path": str(args.spatial_checkpoint),
            "file_sha256": sha256_file(args.spatial_checkpoint),
            "parameter_sha256": backbone_digest,
            "planning_horizon": planning_horizon,
        },
        "training_configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "ensemble_size": args.ensemble_size,
            "positive_weight": args.positive_weight,
            "residual_weight": args.residual_weight,
            "validation_modulus": args.validation_modulus,
            "max_training_arms": args.max_training_arms,
            "max_validation_arms": args.max_validation_arms,
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
        "first_training_metrics": asdict(history[0]),
        "final_training_metrics": asdict(history[-1]),
        "validation_gate_baseline": asdict(before),
        "validation_after": asdict(after),
        "held_out_gate": {
            "passed": gate_passed,
            "baseline": "untrained_same_architecture",
            "requirements": [
                "lower held-out per-cell loss than the untrained head",
                "higher cell ROC AUC than the untrained head",
                "higher mean probability on controllable cells than background",
                (
                    "higher mean probability on controllable cells than "
                    "residual cells when residual cells exist"
                ),
            ],
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_parameter_sha256": checkpoint_digest,
    }
    metrics_path = args.metrics or args.checkpoint.with_suffix(".metrics.json")
    _atomic_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
