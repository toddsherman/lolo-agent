from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from .neural_world_model import choose_torch_device
from .probe_returnability_import import (
    balanced_probe_sample,
    load_probe_returnability_corpus,
    probe_validation_sample,
)
from .run_logging import sha256_file
from .spatial_returnability import (
    SpatialReturnabilityModel,
    save_returnability_checkpoint,
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
            "Train an observational returnability head from explicit matched-NOOP probes"
        )
    )
    parser.add_argument("--training-run", type=Path, action="append", required=True)
    parser.add_argument("--validation-run", type=Path, action="append", required=True)
    parser.add_argument("--spatial-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument(
        "--reward-track", choices=("strict", "assisted"), default="strict"
    )
    parser.add_argument("--max-training-examples", type=int, default=8000)
    parser.add_argument("--max-validation-examples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--spatial-bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(
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
    if args.ensemble_size < 2:
        parser.error("--ensemble-size must be at least two")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")

    torch.manual_seed(args.seed)
    device = choose_torch_device()
    corpus = load_probe_returnability_corpus(
        args.training_run,
        args.validation_run,
        required_reward_track=args.reward_track,
    )
    training = balanced_probe_sample(
        corpus.training, args.max_training_examples, args.seed
    )
    validation = probe_validation_sample(
        corpus.validation, args.max_validation_examples, args.seed + 1
    )
    spatial_model, planning_horizon = load_spatial_checkpoint(
        args.spatial_checkpoint, device=device, frozen=True
    )
    spatial_digest = spatial_model.checkpoint_digest

    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        baseline = SpatialReturnabilityModel(
            spatial_model.token_size,
            hidden_size=args.hidden_size,
            ensemble_size=args.ensemble_size,
            spatial_bins=args.spatial_bins,
        )
    before = validate_returnability_model(
        baseline,
        spatial_model,
        validation,
        device,
        args.batch_size,
        use_observed_targets=True,
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
        use_observed_targets=True,
    )
    after = validate_returnability_model(
        model,
        spatial_model,
        validation,
        device,
        args.batch_size,
        use_observed_targets=True,
    )
    model.freeze()
    frozen_digest = model.checkpoint_digest
    probe_configuration = corpus.metadata["probe_configuration"]
    target_metadata = {
        "kind": "matched_noop_probe",
        "maximum_depth": int(probe_configuration["maximum_depth"]),
        "beam_width": int(probe_configuration["beam_width"]),
        "pixel_l1_threshold": float(probe_configuration["pixel_l1_threshold"]),
        "actions": list(probe_configuration["actions"]),
        "negative_scope": "no return observed within the logged probe budget",
        "relation_tokens": "observed source and verified endpoint pixels",
    }
    checkpoint_digest = save_returnability_checkpoint(
        model,
        args.checkpoint,
        spatial_checkpoint_digest=spatial_digest,
        maximum_return_steps=target_metadata["maximum_depth"],
        minimum_endpoint_actions=len(target_metadata["actions"]),
        target="matched-NOOP budget-scoped visual return",
        persistent_inputs=(
            "pixels",
            "actions",
            "action_durations",
            "save_state_branch_outcomes",
            "duration_matched_noop_pixels",
            "verified_endpoint_pixels",
        ),
        target_metadata=target_metadata,
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
        "architecture": "matched-noop-spatial-returnability",
        "reward_track": args.reward_track,
        "persistent_inputs": [
            "pixels",
            "actions",
            "action_durations",
            "save_state_branch_outcomes",
            "duration_matched_noop_pixels",
            "verified_endpoint_pixels",
        ],
        "excluded_inputs": [
            "RAM",
            "object_labels",
            "rewards",
            "level_annotations",
            "solutions",
            "planner_scores",
        ],
        "device": str(device),
        "seed": args.seed,
        "corpus": corpus.metadata,
        "balanced_training_examples": len(training),
        "validation_examples": len(validation),
        "validation_prevalence": sum(item.label for item in validation)
        / len(validation),
        "training_source_runs": sorted({item.source_run_id for item in training}),
        "validation_source_runs": sorted({item.source_run_id for item in validation}),
        "updates": len(history),
        "first_training_metrics": asdict(history[0]),
        "final_training_metrics": asdict(history[-1]),
        "validation_before": asdict(before),
        "validation_after": asdict(after),
        "held_out_research_gate": {
            "passed": gate_passed,
            "requirements": [
                "higher ROC AUC than an untrained same-architecture head",
                "ROC AUC of at least 0.6",
                "lower Brier score than the constant-prevalence baseline",
                "higher accuracy than the majority-label baseline",
                "higher mean probability on explicit-return examples",
            ],
        },
        "planner_control_eligible": False,
        "planner_control_blocker": (
            "explicit native negatives remain too few and too early-room-specific; "
            "this checkpoint is observational regardless of the research gate"
        ),
        "spatial_checkpoint": {
            "path": str(args.spatial_checkpoint),
            "file_sha256": sha256_file(args.spatial_checkpoint),
            "parameter_sha256": spatial_digest,
            "planning_horizon": planning_horizon,
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_parameter_sha256": checkpoint_digest,
        "target_metadata": target_metadata,
    }
    metrics_path = args.metrics or args.checkpoint.with_suffix(".metrics.json")
    _atomic_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
