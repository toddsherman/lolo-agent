from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .ensemble_world_model import (
    load_ensemble_checkpoint,
    save_ensemble_checkpoint,
    split_sequence_runs,
    train_ensemble_model,
    validate_ensemble_model,
)
from .neural_world_model import choose_torch_device
from .run_logging import sha256_file, utc_now
from .sequence_store import SequenceStore


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _validation(report: Any) -> Dict[str, Any]:
    return {
        "horizon_pixel_l1": list(report.horizon_pixel_l1),
        "horizon_uncertainty": list(report.horizon_uncertainty),
        "uncertainty_error_correlation": report.uncertainty_error_correlation,
    }


def run_real_data_training(
    *,
    dataset: Path,
    input_checkpoint: Path,
    output_checkpoint: Path,
    maximum_groups: int,
    minimum_multistep_groups: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    hourly_rate_usd: float,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if maximum_groups <= 1 or epochs <= 0 or batch_size <= 0:
        raise ValueError("maximum_groups, epochs, and batch_size must be positive")
    if minimum_multistep_groups < 0 or minimum_multistep_groups > maximum_groups:
        raise ValueError("minimum_multistep_groups must fit within maximum_groups")
    if learning_rate <= 0 or hourly_rate_usd < 0:
        raise ValueError("learning rate must be positive and hourly rate non-negative")

    selected = choose_torch_device() if device is None else torch.device(device)
    torch.manual_seed(seed)
    total_started = time.perf_counter()

    load_started = time.perf_counter()
    store = SequenceStore(dataset)
    sequences = store.load_group_sample(
        maximum_groups,
        seed=seed,
        minimum_multistep_groups=minimum_multistep_groups,
    )
    training, validation = split_sequence_runs(sequences, validation_modulus=5)
    model, planning_horizon = load_ensemble_checkpoint(
        input_checkpoint, device=selected, frozen=False
    )
    load_seconds = time.perf_counter() - load_started

    validation_started = time.perf_counter()
    before = validate_ensemble_model(model, validation, selected, batch_size)
    _synchronize(selected)
    validation_before_seconds = time.perf_counter() - validation_started

    training_started = time.perf_counter()
    history = train_ensemble_model(
        model,
        training,
        selected,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    _synchronize(selected)
    training_seconds = time.perf_counter() - training_started

    validation_started = time.perf_counter()
    after = validate_ensemble_model(model, validation, selected, batch_size)
    _synchronize(selected)
    validation_after_seconds = time.perf_counter() - validation_started

    checkpoint_started = time.perf_counter()
    checkpoint_digest = save_ensemble_checkpoint(
        model, output_checkpoint, planning_horizon
    )
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    total_seconds = time.perf_counter() - total_started
    examples = len(training) * epochs

    return {
        "version": 1,
        "completed_at": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(selected),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "input": {
            "dataset": str(Path(dataset).resolve()),
            "dataset_statistics": store.statistics(),
            "checkpoint": str(Path(input_checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(input_checkpoint),
            "maximum_groups": maximum_groups,
            "minimum_multistep_groups": minimum_multistep_groups,
            "sampled_sequences": len(sequences),
            "training_sequences": len(training),
            "validation_sequences": len(validation),
            "training_source_runs": len({item.source_run_id for item in training}),
            "validation_source_runs": len({item.source_run_id for item in validation}),
        },
        "training": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "updates": len(history),
            "examples": examples,
            "first_loss": history[0].loss,
            "final_loss": history[-1].loss,
            "validation_before": _validation(before),
            "validation_after": _validation(after),
        },
        "timing": {
            "load_seconds": load_seconds,
            "validation_before_seconds": validation_before_seconds,
            "training_seconds": training_seconds,
            "validation_after_seconds": validation_after_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "total_seconds": total_seconds,
            "training_examples_per_second": examples / training_seconds,
            "end_to_end_examples_per_second": examples / total_seconds,
            "hourly_rate_usd": hourly_rate_usd,
            "estimated_end_to_end_cost_usd": total_seconds
            * hourly_rate_usd
            / 3600,
        },
        "output": {
            "checkpoint": str(Path(output_checkpoint).resolve()),
            "checkpoint_parameter_sha256": checkpoint_digest,
            "checkpoint_file_sha256": sha256_file(output_checkpoint),
            "checkpoint_bytes": Path(output_checkpoint).stat().st_size,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and validate the ensemble world model on persisted pixels"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--maximum-groups", type=int, default=64)
    parser.add_argument("--minimum-multistep-groups", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hourly-rate-usd", type=float, default=0.0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    result = run_real_data_training(
        dataset=args.dataset,
        input_checkpoint=args.input_checkpoint,
        output_checkpoint=args.output_checkpoint,
        maximum_groups=args.maximum_groups,
        minimum_multistep_groups=args.minimum_multistep_groups,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        hourly_rate_usd=args.hourly_rate_usd,
        device=None if args.device is None else torch.device(args.device),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.metrics.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.metrics.expanduser().resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
