from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from .ensemble_world_model import split_sequence_groups
from .neural_world_model import choose_torch_device
from .sequence_store import SequenceStore
from .spatial_world_model import (
    SpatialTokenDynamicsModel,
    causal_dataset_statistics,
    save_spatial_checkpoint,
    train_spatial_model,
    validate_spatial_model,
)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the unlabeled spatial-token causal dynamics model"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--reward-track", choices=("strict", "assisted"), default="strict")
    parser.add_argument("--max-groups", type=int, default=1000)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--token-size", type=int, default=64)
    parser.add_argument("--action-size", type=int, default=16)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--duration-size", type=int, default=8)
    parser.add_argument("--max-action-frames", type=int, default=32)
    parser.add_argument("--effect-mask-power", type=float, default=4.0)
    parser.add_argument("--token-delta-scale", type=float, default=0.25)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(
        args.max_groups,
        args.epochs,
        args.batch_size,
        args.token_size,
        args.action_size,
        args.ensemble_size,
        args.grid_size,
        args.duration_size,
        args.max_action_frames,
        args.planning_horizon,
        args.effect_mask_power,
        args.token_delta_scale,
    ) <= 0:
        parser.error("model and training sizes must be positive")
    if args.validation_modulus < 2:
        parser.error("--validation-modulus must be at least two")

    torch.manual_seed(args.seed)
    device = choose_torch_device()
    store = SequenceStore(args.dataset)
    store.bind_reward_track(args.reward_track)
    sequences = store.load_group_sample(args.max_groups, seed=args.seed)
    training, validation = split_sequence_groups(
        sequences, validation_modulus=args.validation_modulus
    )
    model = SpatialTokenDynamicsModel(
        token_size=args.token_size,
        action_size=args.action_size,
        ensemble_size=args.ensemble_size,
        grid_size=args.grid_size,
        duration_conditioned=True,
        duration_size=args.duration_size,
        max_action_frames=args.max_action_frames,
        effect_mask_power=args.effect_mask_power,
        token_delta_scale=args.token_delta_scale,
    )
    before = validate_spatial_model(model, validation, device, args.batch_size)
    history = train_spatial_model(
        model,
        training,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    after = validate_spatial_model(model, validation, device, args.batch_size)
    model.freeze()
    frozen_digest = model.checkpoint_digest
    checkpoint_digest = save_spatial_checkpoint(
        model, args.checkpoint, planning_horizon=args.planning_horizon
    )
    if checkpoint_digest != frozen_digest:
        raise SystemExit("freezing or checkpointing changed persistent parameters")

    mean_effect_before = sum(before.horizon_effect_l1) / len(before.horizon_effect_l1)
    mean_effect_after = sum(after.horizon_effect_l1) / len(after.horizon_effect_l1)
    mean_balanced_effect_after = sum(after.horizon_balanced_effect_l1) / len(
        after.horizon_balanced_effect_l1
    )
    mean_zero_balanced_effect = sum(after.horizon_zero_balanced_effect_l1) / len(
        after.horizon_zero_balanced_effect_l1
    )
    mean_f1_before = sum(before.horizon_effect_f1) / len(before.horizon_effect_f1)
    mean_f1_after = sum(after.horizon_effect_f1) / len(after.horizon_effect_f1)
    effect_gate_passed = (
        mean_effect_after < mean_effect_before
        and mean_balanced_effect_after < mean_zero_balanced_effect
        and mean_f1_after > mean_f1_before
    )
    mean_weighted_pixel = sum(after.horizon_effect_weighted_pixel_l1) / len(
        after.horizon_effect_weighted_pixel_l1
    )
    mean_weighted_persistence = sum(
        after.horizon_effect_weighted_persistence_l1
    ) / len(after.horizon_effect_weighted_persistence_l1)
    mean_uncertainty_correlation = sum(
        after.horizon_uncertainty_effect_error_correlation
    ) / len(after.horizon_uncertainty_effect_error_correlation)
    uncertainty_calibrated = mean_uncertainty_correlation > 0.0
    metrics = {
        "version": 1,
        "architecture": "unlabeled-spatial-token-dynamics",
        "reward_track": args.reward_track,
        "persistent_inputs": ["pixels", "actions", "action_durations"],
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
        "sample_statistics": causal_dataset_statistics(sequences),
        "training_sequences": len(training),
        "validation_sequences": len(validation),
        "updates": len(history),
        "first_training_metrics": asdict(history[0]),
        "final_training_metrics": asdict(history[-1]),
        "validation_before": asdict(before),
        "validation_after": asdict(after),
        "held_out_effect_gate": {
            "passed": effect_gate_passed,
            "mean_effect_l1_before": mean_effect_before,
            "mean_effect_l1_after": mean_effect_after,
            "mean_balanced_effect_l1_after": mean_balanced_effect_after,
            "mean_zero_balanced_effect_l1": mean_zero_balanced_effect,
            "mean_effect_f1_before": mean_f1_before,
            "mean_effect_f1_after": mean_f1_after,
            "requirements": [
                "lower effect error than the untrained model",
                "lower class-balanced effect error than always predicting no change",
                "higher spatial effect F1 than the untrained model",
            ],
        },
        "planner_integration_gate": {
            "passed": (
                effect_gate_passed
                and mean_weighted_pixel < mean_weighted_persistence
                and uncertainty_calibrated
            ),
            "mean_effect_weighted_pixel_l1": mean_weighted_pixel,
            "mean_effect_weighted_persistence_l1": mean_weighted_persistence,
            "uncertainty_effect_error_correlation": (
                after.uncertainty_effect_error_correlation
            ),
            "mean_within_horizon_uncertainty_effect_error_correlation": (
                mean_uncertainty_correlation
            ),
            "requirements": [
                "pass the held-out spatial-effect gate",
                "beat frame persistence on effect-weighted pixels",
                "positive correlation between ensemble uncertainty and effect error",
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
