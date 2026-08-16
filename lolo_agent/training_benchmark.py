from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .neural_world_model import ACTION_ORDER, VisualDynamicsModel, choose_torch_device
from .run_logging import utc_now


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _loss(
    model: VisualDynamicsModel,
    source: Tensor,
    actions: Tensor,
    target: Tensor,
) -> Tensor:
    reconstructed, predicted, predicted_latent = model(source, actions)
    with torch.no_grad():
        target_latent = model.encode(target)
    return (
        0.25 * F.l1_loss(reconstructed, source)
        + F.l1_loss(predicted, target)
        + 0.5 * F.smooth_l1_loss(predicted_latent, target_latent)
    )


def _synthetic_batch(
    batch_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    source = torch.rand(
        batch_size, 3, 128, 128, generator=generator, dtype=torch.float32
    )
    actions = torch.randint(
        len(ACTION_ORDER), (batch_size,), generator=generator
    )
    shifts = actions.remainder(5).sub(2)
    target = torch.empty_like(source)
    for index, shift in enumerate(shifts.tolist()):
        target[index] = torch.roll(source[index], shifts=shift, dims=2)
    return source, actions, target


def benchmark_training(
    *,
    steps: int,
    warmup_steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    hourly_rate_usd: float,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if steps <= 0 or warmup_steps < 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive; warmup may be zero")
    if learning_rate <= 0 or hourly_rate_usd < 0:
        raise ValueError("learning_rate must be positive and hourly rate non-negative")
    selected = choose_torch_device() if device is None else torch.device(device)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches = [
        _synthetic_batch(batch_size, generator)
        for _ in range(warmup_steps + steps + 1)
    ]
    model = VisualDynamicsModel().to(selected)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def update(batch: tuple[Tensor, Tensor, Tensor]) -> float:
        source, actions, target = (
            value.to(selected, non_blocking=True) for value in batch
        )
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, source, actions, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        return float(loss.detach().cpu())

    for index in range(warmup_steps):
        update(batches[index])
    validation = tuple(value.to(selected) for value in batches[-1])
    with torch.no_grad():
        loss_before = float(_loss(model, *validation).detach().cpu())
    _synchronize(selected)
    started = time.perf_counter()
    final_training_loss = 0.0
    for index in range(steps):
        final_training_loss = update(batches[warmup_steps + index])
    _synchronize(selected)
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        loss_after = float(_loss(model, *validation).detach().cpu())
    examples = steps * batch_size
    examples_per_second = examples / elapsed
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
        "model": {
            "name": "VisualDynamicsModel",
            "parameters": sum(value.numel() for value in model.parameters()),
            "input_shape": [3, 128, 128],
        },
        "benchmark": {
            "seed": seed,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "batch_size": batch_size,
            "examples": examples,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed,
            "updates_per_second": steps / elapsed,
            "examples_per_second": examples_per_second,
            "validation_loss_before": loss_before,
            "validation_loss_after": loss_after,
            "final_training_loss": final_training_loss,
            "hourly_rate_usd": hourly_rate_usd,
            "estimated_cost_per_million_examples_usd": (
                None
                if hourly_rate_usd == 0
                else hourly_rate_usd
                / (examples_per_second * 3600)
                * 1_000_000
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed visual-world-model training updates"
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hourly-rate-usd", type=float, default=0.0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark_training(
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        hourly_rate_usd=args.hourly_rate_usd,
        device=None if args.device is None else torch.device(args.device),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
