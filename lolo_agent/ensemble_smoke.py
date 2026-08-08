from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .ensemble_world_model import (
    EnsembleVisualDynamicsModel,
    collect_branched_sequences,
    save_ensemble_checkpoint,
    split_sequence_groups,
    train_ensemble_model,
    validate_ensemble_model,
)
from .native_env import NativeLibretroEnv
from .neural_planner import NeuralPlanningConfig, VerifiedNeuralAgent
from .neural_world_model import ACTION_ORDER, choose_torch_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and verify multi-step ensemble planning")
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=20)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--action-frames", type=int, default=4)
    parser.add_argument("--action-durations")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--agent-decisions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_torch_device()
    action_durations = (
        tuple(int(value) for value in args.action_durations.split(","))
        if args.action_durations
        else (args.action_frames,)
    )
    with NativeLibretroEnv(args.host, args.core, args.rom) as env:
        sequences = collect_branched_sequences(
            env,
            roots=args.roots,
            branches_per_root=args.branches,
            horizon=args.horizon,
            action_frames=args.action_frames,
            action_durations=action_durations,
            seed=args.seed,
        )
    training, validation = split_sequence_groups(sequences)
    model = EnsembleVisualDynamicsModel(
        duration_conditioned=len(action_durations) > 1,
        max_action_frames=max(32, max(action_durations)),
        fixed_action_frames=action_durations[0],
    )
    before_report = validate_ensemble_model(model, validation, device, args.batch_size)
    history = train_ensemble_model(
        model,
        training,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    after_report = validate_ensemble_model(model, validation, device, args.batch_size)
    model.freeze()
    frozen_digest = model.checkpoint_digest

    with NativeLibretroEnv(args.host, args.core, args.rom) as env:
        agent = VerifiedNeuralAgent(
            env,
            model,
            device,
            NeuralPlanningConfig(
                actions=ACTION_ORDER,
                planning_depth=args.horizon,
                action_frames=args.action_frames,
                action_durations=action_durations if len(action_durations) > 1 else (),
            ),
        )
        agent.reset()
        decisions = agent.run(args.agent_decisions)
    if model.checkpoint_digest != frozen_digest:
        raise SystemExit("frozen neural planning changed persistent parameters")

    if args.checkpoint:
        save_ensemble_checkpoint(model, args.checkpoint, args.horizon)

    print(f"device={device}")
    print(f"train_sequences={len(training)} validation_sequences={len(validation)}")
    print(f"updates={len(history)} initial_loss={history[0].loss:.6f} final_loss={history[-1].loss:.6f}")
    print("validation_before=" + ",".join(f"{value:.6f}" for value in before_report.horizon_pixel_l1))
    print("validation_after=" + ",".join(f"{value:.6f}" for value in after_report.horizon_pixel_l1))
    print("uncertainty_by_horizon=" + ",".join(f"{value:.8f}" for value in after_report.horizon_uncertainty))
    print(f"uncertainty_error_correlation={after_report.uncertainty_error_correlation:.6f}")
    for index, decision in enumerate(decisions, 1):
        path = ",".join(action.value for action in decision.planned_path)
        print(
            f"decision_{index}={decision.action.value}@{decision.action_frames} "
            f"plan={path} verified={decision.branches_examined}"
        )
    print(f"checkpoint_sha256={frozen_digest}")
    print("frozen_planner_audit=pass")


if __name__ == "__main__":
    main()
