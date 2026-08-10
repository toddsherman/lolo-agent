from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from .neural_world_model import choose_torch_device
from .run_logging import sha256_file
from .sequence_store import SequenceStore
from .spatial_returnability import (
    SpatialReturnabilityModel,
    balanced_returnability_sample,
    build_returnability_specs,
    decode_returnability_examples,
    save_returnability_checkpoint,
    split_returnability_runs,
    train_returnability_model,
    validate_returnability_model,
)
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
            "Train an unlabeled observed-returnability head over a frozen spatial model"
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spatial-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--reward-track", choices=("strict", "assisted"), default="strict")
    parser.add_argument("--maximum-return-steps", type=int, default=3)
    parser.add_argument("--minimum-endpoint-actions", type=int, default=5)
    parser.add_argument("--max-training-examples", type=int, default=8000)
    parser.add_argument("--max-validation-examples", type=int, default=2000)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument(
        "--spatial-bins",
        type=int,
        default=4,
        help="coarse relation layout retained before binary prediction",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(
        args.maximum_return_steps,
        args.minimum_endpoint_actions,
        args.max_training_examples,
        args.max_validation_examples,
        args.epochs,
        args.batch_size,
        args.hidden_size,
        args.ensemble_size,
        args.spatial_bins,
    ) <= 0:
        parser.error("model, data, and training sizes must be positive")
    if args.max_training_examples < 2 or args.max_validation_examples < 2:
        parser.error("example limits must be at least two")
    if args.validation_modulus < 2:
        parser.error("--validation-modulus must be at least two")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")

    torch.manual_seed(args.seed)
    device = choose_torch_device()
    store = SequenceStore(args.dataset)
    store.bind_reward_track(args.reward_track)
    spatial_model, planning_horizon = load_spatial_checkpoint(
        args.spatial_checkpoint, device=device, frozen=True
    )
    spatial_digest = spatial_model.checkpoint_digest
    specs, graph_statistics = build_returnability_specs(
        store.transition_metadata(),
        maximum_return_steps=args.maximum_return_steps,
        minimum_endpoint_actions=args.minimum_endpoint_actions,
    )
    training_specs, validation_specs = split_returnability_runs(
        specs, validation_modulus=args.validation_modulus
    )
    training_specs = balanced_returnability_sample(
        training_specs, args.max_training_examples, args.seed
    )
    validation_specs = balanced_returnability_sample(
        validation_specs, args.max_validation_examples, args.seed + 1
    )
    training = decode_returnability_examples(store, training_specs)
    validation = decode_returnability_examples(store, validation_specs)

    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        baseline = SpatialReturnabilityModel(
            spatial_model.token_size,
            hidden_size=args.hidden_size,
            ensemble_size=args.ensemble_size,
            spatial_bins=args.spatial_bins,
        )
    before = validate_returnability_model(
        baseline, spatial_model, validation, device, args.batch_size
    )
    model = SpatialReturnabilityModel(
        spatial_model.token_size,
        hidden_size=args.hidden_size,
        ensemble_size=args.ensemble_size,
        spatial_bins=args.spatial_bins,
    )
    history = train_returnability_model(
        model,
        spatial_model,
        training,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    after = validate_returnability_model(
        model, spatial_model, validation, device, args.batch_size
    )
    model.freeze()
    frozen_digest = model.checkpoint_digest
    checkpoint_digest = save_returnability_checkpoint(
        model,
        args.checkpoint,
        spatial_checkpoint_digest=spatial_digest,
        maximum_return_steps=args.maximum_return_steps,
        minimum_endpoint_actions=args.minimum_endpoint_actions,
    )
    if checkpoint_digest != frozen_digest:
        raise SystemExit("freezing or checkpointing changed returnability parameters")

    gate_passed = (
        after.roc_auc > before.roc_auc
        and after.roc_auc >= 0.6
        and after.brier < after.constant_brier
        and after.accuracy > after.majority_accuracy
        and after.mean_positive_probability > after.mean_negative_probability
    )
    metrics = {
        "version": 1,
        "architecture": "unlabeled-spatial-returnability",
        "reward_track": args.reward_track,
        "persistent_inputs": [
            "pixels",
            "actions",
            "action_durations",
            "observed_transition_graph",
        ],
        "excluded_inputs": [
            "RAM",
            "object_labels",
            "rewards",
            "level_annotations",
            "solutions",
        ],
        "device": str(device),
        "seed": args.seed,
        "dataset": str(store.root),
        "dataset_statistics": store.statistics(),
        "graph_statistics": graph_statistics,
        "target_definition": {
            "positive": (
                "the transition endpoint has an observed pixel-state path back "
                "to its source within the configured action horizon"
            ),
            "negative": (
                "no return path was observed within the horizon after the endpoint "
                "was probed with the configured number of distinct controls"
            ),
            "unlabeled": "all less-conclusive transitions are censored",
            "maximum_return_steps": args.maximum_return_steps,
            "minimum_endpoint_actions": args.minimum_endpoint_actions,
        },
        "spatial_checkpoint": {
            "path": str(args.spatial_checkpoint),
            "file_sha256": sha256_file(args.spatial_checkpoint),
            "parameter_sha256": spatial_digest,
            "planning_horizon": planning_horizon,
        },
        "training_source_runs": sorted({item.source_run_id for item in training}),
        "validation_source_runs": sorted({item.source_run_id for item in validation}),
        "training_examples": len(training),
        "validation_examples": len(validation),
        "updates": len(history),
        "first_training_metrics": asdict(history[0]),
        "final_training_metrics": asdict(history[-1]),
        "validation_before": asdict(before),
        "validation_after": asdict(after),
        "held_out_gate": {
            "passed": gate_passed,
            "requirements": [
                "higher ROC AUC than an untrained same-architecture head",
                "ROC AUC of at least 0.6",
                "lower Brier score than the constant-prevalence baseline",
                "higher accuracy than the majority-label baseline",
                "higher mean probability on observed-return examples",
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
