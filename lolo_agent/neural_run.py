from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .ensemble_world_model import load_ensemble_checkpoint
from .log_summary import build_run_summary
from .native_env import NativeLibretroEnv
from .neural_planner import NeuralPlanningConfig, VerifiedNeuralAgent
from .neural_world_model import ACTION_ORDER, choose_torch_device
from .run_logging import LoggedEnvironment, RunLogger, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen neural rollout planner")
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--action-frames", type=int, default=4)
    parser.add_argument(
        "--action-durations",
        help="comma-separated press lengths; requires a duration-conditioned checkpoint",
    )
    parser.add_argument("--verify-actions", type=int, default=4)
    parser.add_argument("--log-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--no-frame-images", action="store_true")
    args = parser.parse_args()

    device = choose_torch_device()
    model, horizon = load_ensemble_checkpoint(args.checkpoint, device=device, frozen=True)
    before = model.checkpoint_digest
    action_durations = (
        tuple(int(value) for value in args.action_durations.split(","))
        if args.action_durations
        else ()
    )
    config = NeuralPlanningConfig(
        actions=ACTION_ORDER,
        planning_depth=horizon,
        action_frames=args.action_frames,
        action_durations=action_durations,
        verify_actions=args.verify_actions,
    )
    metadata = {
        "mode": "frozen_neural_evaluation",
        "requested_decisions": args.decisions,
        "device": str(device),
        "planning_config": asdict(config),
        "inputs": {
            "rom": {"name": args.rom.name, "sha256": sha256_file(args.rom)},
            "core": {"name": args.core.name, "sha256": sha256_file(args.core)},
            "host": {"name": args.host.name, "sha256": sha256_file(args.host)},
            "checkpoint": {
                "name": args.checkpoint.name,
                "file_sha256": sha256_file(args.checkpoint),
                "parameter_sha256": before,
            },
        },
    }
    logger = RunLogger(
        args.log_root,
        run_id=args.run_id,
        metadata=metadata,
        store_frames=not args.no_frame_images,
    )
    agent = None
    try:
        with NativeLibretroEnv(args.host, args.core, args.rom) as native_env:
            logger.log(
                "emulator_started",
                core_name=native_env.core_name,
                core_version=native_env.core_version,
                base_width=native_env.base_width,
                base_height=native_env.base_height,
                fps=native_env.fps,
            )
            env = LoggedEnvironment(native_env, logger)
            agent = VerifiedNeuralAgent(env, model, device, config, event_logger=logger)
            agent.reset()
            decisions = agent.run(args.decisions)
            agent.clear_archive()
        after = model.checkpoint_digest
        if before != after:
            raise RuntimeError("frozen evaluation changed persistent parameters")
        logger.log(
            "frozen_parameter_audit",
            status="pass",
            parameter_sha256_before=before,
            parameter_sha256_after=after,
        )
        logger.close("complete")
    except Exception as exc:
        if agent is not None:
            agent.clear_archive()
        logger.close("error", str(exc))
        build_run_summary(logger.run_dir)
        raise
    summary = build_run_summary(logger.run_dir)
    print(f"device={device} planning_horizon={horizon}")
    for index, decision in enumerate(decisions, 1):
        path = ",".join(action.value for action in decision.planned_path)
        print(
            f"{index:04d} action={decision.action.value:<6} "
            f"frames={decision.action_frames:<2} "
            f"score={decision.score:.6f} plan={path} verified={decision.branches_examined} "
            f"restored={decision.restored_archive}"
        )
    print(f"checkpoint_sha256={after}")
    print("frozen_evaluation_audit=pass")
    print(f"telemetry_run={logger.run_dir}")
    print(
        f"telemetry_events={summary['events']} unique_frames={summary['unique_frames']} "
        f"verified_branches={summary['verified_branches']}"
    )


if __name__ == "__main__":
    main()
