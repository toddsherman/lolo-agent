from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from .ensemble_world_model import split_sequence_groups, split_sequence_runs
from .experience_import import ExperienceSource, extract_experience
from .neural_world_model import choose_torch_device
from .run_logging import sha256_file
from .sequence_store import SequenceStore
from .spatial_world_model import (
    SpatialTokenDynamicsModel,
    causal_dataset_statistics,
    load_spatial_checkpoint,
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
    parser.add_argument(
        "--initialize-from-checkpoint",
        type=Path,
        help="warm-start training from an existing spatial checkpoint",
    )
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--reward-track", choices=("strict", "assisted"), default="strict")
    parser.add_argument("--max-groups", type=int, default=1000)
    parser.add_argument(
        "--additional-experience-run",
        type=Path,
        action="append",
        default=[],
        help=(
            "strict telemetry run to add after sampling the base dataset; only "
            "verified pixels, actions, durations, and committed windows are read"
        ),
    )
    parser.add_argument("--additional-experience-horizon", type=int, default=3)
    parser.add_argument(
        "--minimum-multistep-groups",
        type=int,
        default=0,
        help=(
            "minimum sampled causal groups containing a trajectory longer than "
            "one action; all branches in each selected group remain together"
        ),
    )
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument(
        "--validation-split",
        choices=("run", "group"),
        default="run",
        help="hold out complete source runs by default; group is a weaker development split",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=0.1)
    parser.add_argument("--pixel-loss-weight", type=float, default=0.5)
    parser.add_argument("--changed-region-loss-weight", type=float, default=0.0)
    parser.add_argument("--token-loss-weight", type=float, default=0.5)
    parser.add_argument("--effect-loss-weight", type=float, default=0.5)
    parser.add_argument("--token-size", type=int, default=64)
    parser.add_argument("--action-size", type=int, default=16)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--duration-size", type=int, default=8)
    parser.add_argument("--max-action-frames", type=int, default=32)
    parser.add_argument("--effect-mask-power", type=float, default=4.0)
    parser.add_argument("--token-delta-scale", type=float, default=0.25)
    parser.add_argument(
        "--renderer",
        choices=("flow_residual", "changed_patch", "blend"),
        default="flow_residual",
    )
    parser.add_argument(
        "--renderer-rollout",
        choices=("recursive", "anchored"),
        default="recursive",
    )
    parser.add_argument("--max-flow-pixels", type=float, default=16.0)
    parser.add_argument("--residual-scale", type=float, default=0.25)
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
        args.additional_experience_horizon,
        args.effect_mask_power,
        args.token_delta_scale,
        args.max_flow_pixels,
        args.residual_scale,
    ) <= 0:
        parser.error("model and training sizes must be positive")
    if args.validation_modulus < 2:
        parser.error("--validation-modulus must be at least two")
    if args.minimum_multistep_groups < 0:
        parser.error("--minimum-multistep-groups must be non-negative")
    if args.minimum_multistep_groups > args.max_groups:
        parser.error("--minimum-multistep-groups cannot exceed --max-groups")
    loss_weights = {
        "reconstruction": args.reconstruction_loss_weight,
        "pixel_prediction": args.pixel_loss_weight,
        "changed_region_prediction": args.changed_region_loss_weight,
        "token_prediction": args.token_loss_weight,
        "effect_prediction": args.effect_loss_weight,
    }
    if any(weight < 0.0 for weight in loss_weights.values()):
        parser.error("loss weights must be non-negative")
    if not any(loss_weights.values()):
        parser.error("at least one loss weight must be positive")

    torch.manual_seed(args.seed)
    device = choose_torch_device()
    store = SequenceStore(args.dataset)
    store.bind_reward_track(args.reward_track)
    sequences = store.load_group_sample(
        args.max_groups,
        seed=args.seed,
        minimum_multistep_groups=args.minimum_multistep_groups,
    )
    additional_experience = []
    next_additional_group = max((item.group for item in sequences), default=-1) + 1
    for run_dir in args.additional_experience_run:
        extracted, source_metadata = extract_experience(
            ExperienceSource(run_dir),
            group_offset=next_additional_group,
            committed_horizon=args.additional_experience_horizon,
        )
        if source_metadata["reward_track"] != args.reward_track:
            parser.error(
                "additional experience reward track does not match the dataset: "
                f"{source_metadata['run_id']}"
            )
        sequences.extend(extracted)
        additional_experience.append(source_metadata)
        next_additional_group = int(source_metadata["next_group"])
    splitter = (
        split_sequence_runs
        if args.validation_split == "run"
        else split_sequence_groups
    )
    training, validation = splitter(
        sequences, validation_modulus=args.validation_modulus
    )
    initialization_checkpoint = None
    if args.initialize_from_checkpoint is None:
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
            renderer_kind=args.renderer,
            renderer_rollout=args.renderer_rollout,
            max_flow_pixels=args.max_flow_pixels,
            residual_scale=args.residual_scale,
        )
    else:
        model, initialization_horizon = load_spatial_checkpoint(
            args.initialize_from_checkpoint,
            device=device,
            frozen=False,
        )
        expected_configuration = {
            "token_size": args.token_size,
            "action_size": args.action_size,
            "ensemble_size": args.ensemble_size,
            "grid_size": args.grid_size,
            "duration_size": args.duration_size,
            "max_action_frames": args.max_action_frames,
            "effect_mask_power": args.effect_mask_power,
            "token_delta_scale": args.token_delta_scale,
            "renderer_kind": args.renderer,
            "renderer_rollout": args.renderer_rollout,
            "max_flow_pixels": args.max_flow_pixels,
            "residual_scale": args.residual_scale,
            "planning_horizon": args.planning_horizon,
        }
        actual_configuration = {
            "token_size": model.token_size,
            "action_size": model.action_size,
            "ensemble_size": model.ensemble_size,
            "grid_size": model.grid_size,
            "duration_size": model.duration_size,
            "max_action_frames": model.max_action_frames,
            "effect_mask_power": model.effect_mask_power,
            "token_delta_scale": model.token_delta_scale,
            "renderer_kind": model.renderer_kind,
            "renderer_rollout": model.renderer_rollout,
            "max_flow_pixels": model.max_flow_pixels,
            "residual_scale": model.residual_scale,
            "planning_horizon": initialization_horizon,
        }
        if actual_configuration != expected_configuration:
            parser.error(
                "initialization checkpoint architecture does not match the "
                "requested training configuration"
            )
        initialization_checkpoint = {
            "path": str(args.initialize_from_checkpoint),
            "file_sha256": sha256_file(args.initialize_from_checkpoint),
            "parameter_sha256": model.checkpoint_digest,
            "configuration": actual_configuration,
        }
    before = validate_spatial_model(model, validation, device, args.batch_size)
    gate_baseline = before
    if args.initialize_from_checkpoint is not None:
        with torch.random.fork_rng():
            torch.manual_seed(args.seed)
            baseline_model = SpatialTokenDynamicsModel(
                token_size=args.token_size,
                action_size=args.action_size,
                ensemble_size=args.ensemble_size,
                grid_size=args.grid_size,
                duration_conditioned=True,
                duration_size=args.duration_size,
                max_action_frames=args.max_action_frames,
                effect_mask_power=args.effect_mask_power,
                token_delta_scale=args.token_delta_scale,
                renderer_kind=args.renderer,
                renderer_rollout=args.renderer_rollout,
                max_flow_pixels=args.max_flow_pixels,
                residual_scale=args.residual_scale,
            )
        gate_baseline = validate_spatial_model(
            baseline_model, validation, device, args.batch_size
        )
    history = train_spatial_model(
        model,
        training,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        reconstruction_weight=args.reconstruction_loss_weight,
        pixel_weight=args.pixel_loss_weight,
        changed_region_weight=args.changed_region_loss_weight,
        token_weight=args.token_loss_weight,
        effect_weight=args.effect_loss_weight,
    )
    after = validate_spatial_model(model, validation, device, args.batch_size)
    model.freeze()
    frozen_digest = model.checkpoint_digest
    checkpoint_digest = save_spatial_checkpoint(
        model, args.checkpoint, planning_horizon=args.planning_horizon
    )
    if checkpoint_digest != frozen_digest:
        raise SystemExit("freezing or checkpointing changed persistent parameters")

    mean_effect_before = sum(gate_baseline.horizon_effect_l1) / len(
        gate_baseline.horizon_effect_l1
    )
    mean_effect_after = sum(after.horizon_effect_l1) / len(after.horizon_effect_l1)
    mean_balanced_effect_after = sum(after.horizon_balanced_effect_l1) / len(
        after.horizon_balanced_effect_l1
    )
    mean_zero_balanced_effect = sum(after.horizon_zero_balanced_effect_l1) / len(
        after.horizon_zero_balanced_effect_l1
    )
    mean_f1_before = sum(gate_baseline.horizon_effect_f1) / len(
        gate_baseline.horizon_effect_f1
    )
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
    mean_predicted_change = sum(after.horizon_predicted_pixel_change) / len(
        after.horizon_predicted_pixel_change
    )
    mean_actual_change = sum(after.horizon_actual_pixel_change) / len(
        after.horizon_actual_pixel_change
    )
    predicted_change_ratio = (
        mean_predicted_change / mean_actual_change if mean_actual_change > 0.0 else 1.0
    )
    renderer_not_collapsed = predicted_change_ratio >= 0.1
    mean_uncertainty_correlation = sum(
        after.horizon_uncertainty_effect_error_correlation
    ) / len(after.horizon_uncertainty_effect_error_correlation)
    uncertainty_calibrated = mean_uncertainty_correlation > 0.0
    metrics = {
        "version": 1,
        "architecture": "unlabeled-spatial-token-dynamics",
        "renderer": args.renderer,
        "renderer_rollout": args.renderer_rollout,
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
        "minimum_multistep_groups": args.minimum_multistep_groups,
        "additional_experience": additional_experience,
        "additional_experience_horizon": args.additional_experience_horizon,
        "initialization_checkpoint": initialization_checkpoint,
        "loss_weights": loss_weights,
        "validation_split": args.validation_split,
        "training_source_runs": sorted({item.source_run_id for item in training}),
        "validation_source_runs": sorted({item.source_run_id for item in validation}),
        "dataset": str(store.root),
        "dataset_statistics": store.statistics(),
        "sample_statistics": causal_dataset_statistics(sequences),
        "training_sequences": len(training),
        "validation_sequences": len(validation),
        "updates": len(history),
        "first_training_metrics": asdict(history[0]),
        "final_training_metrics": asdict(history[-1]),
        "validation_before": asdict(before),
        "validation_gate_baseline": asdict(gate_baseline),
        "validation_after": asdict(after),
        "held_out_effect_gate": {
            "passed": effect_gate_passed,
            "mean_effect_l1_before": mean_effect_before,
            "mean_effect_l1_after": mean_effect_after,
            "mean_balanced_effect_l1_after": mean_balanced_effect_after,
            "mean_zero_balanced_effect_l1": mean_zero_balanced_effect,
            "mean_effect_f1_before": mean_f1_before,
            "baseline": "untrained_same_architecture",
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
                and renderer_not_collapsed
            ),
            "mean_effect_weighted_pixel_l1": mean_weighted_pixel,
            "mean_effect_weighted_persistence_l1": mean_weighted_persistence,
            "mean_predicted_pixel_change": mean_predicted_change,
            "mean_actual_pixel_change": mean_actual_change,
            "predicted_to_actual_change_ratio": predicted_change_ratio,
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
                "predict at least 10% of the held-out visual-change magnitude",
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
