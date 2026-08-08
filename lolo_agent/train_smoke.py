from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .native_env import NativeLibretroEnv
from .neural_world_model import (
    ACTION_ORDER,
    VisualDynamicsModel,
    choose_torch_device,
    collect_branched_transitions,
    train_world_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the visual dynamics smoke model")
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=24)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--action-frames", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_torch_device()
    with NativeLibretroEnv(args.host, args.core, args.rom) as env:
        transitions = collect_branched_transitions(
            env,
            decisions=args.decisions,
            branches_per_decision=args.branches,
            action_frames=args.action_frames,
            seed=args.seed,
        )

    model = VisualDynamicsModel()
    before = model.checkpoint_digest
    history = train_world_model(
        model,
        transitions,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    after = model.checkpoint_digest
    model.freeze()
    frozen = model.checkpoint_digest
    if before == after:
        raise SystemExit("training did not update the model")
    if after != frozen:
        raise SystemExit("freezing changed persistent model parameters")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 1,
                "model": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
                "actions": [action.value for action in ACTION_ORDER],
                "digest": frozen,
            },
            args.checkpoint,
        )
    print(f"device={device}")
    print(f"transitions={len(transitions)} updates={len(history)}")
    print(f"initial_loss={history[0].loss:.6f}")
    print(f"final_loss={history[-1].loss:.6f}")
    print(f"checkpoint_sha256={frozen}")
    print("freeze_audit=pass")


if __name__ == "__main__":
    main()
