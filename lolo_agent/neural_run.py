from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .bootstrap import (
    BOOTSTRAP_FIXTURES,
    apply_bootstrap_fixture,
    bootstrap_metadata,
    get_bootstrap_fixture,
)
from .ensemble_world_model import load_ensemble_checkpoint
from .log_summary import build_run_summary
from .native_env import NativeLibretroEnv
from .neural_planner import NeuralPlanningConfig, VerifiedNeuralAgent
from .neural_world_model import ACTION_ORDER, choose_torch_device
from .replay import restore_logged_decision, validate_replay_inputs
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
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="telemetry run containing a previously self-discovered state",
    )
    parser.add_argument(
        "--resume-decision",
        type=int,
        help="committed decision to reconstruct from --resume-run",
    )
    parser.add_argument("--no-frame-images", action="store_true")
    parser.add_argument(
        "--bootstrap",
        choices=("none", *sorted(BOOTSTRAP_FIXTURES)),
        default="none",
        help="evaluator-owned initialization fixture; strict power-on remains the default",
    )
    args = parser.parse_args()
    if (args.resume_run is None) != (args.resume_decision is None):
        parser.error("--resume-run and --resume-decision must be supplied together")
    if args.resume_run is not None and args.bootstrap != "none":
        parser.error("--resume-run cannot be combined with --bootstrap")

    device = choose_torch_device()
    model, horizon = load_ensemble_checkpoint(args.checkpoint, device=device, frozen=True)
    before = model.checkpoint_digest
    action_durations = (
        tuple(int(value) for value in args.action_durations.split(","))
        if args.action_durations
        else ()
    )
    bootstrap_fixture = (
        None if args.bootstrap == "none" else get_bootstrap_fixture(args.bootstrap)
    )
    gameplay_actions = (
        ACTION_ORDER
        if bootstrap_fixture is None and args.resume_run is None
        else NeuralPlanningConfig().actions
    )
    config = NeuralPlanningConfig(
        actions=gameplay_actions,
        planning_depth=horizon,
        action_frames=args.action_frames,
        action_durations=action_durations,
        verify_actions=args.verify_actions,
    )
    rom_sha256 = sha256_file(args.rom)
    resume_metadata = None
    if args.resume_run is not None:
        source_manifest = validate_replay_inputs(
            args.resume_run, args.host, args.core, args.rom
        )
        source_events = args.resume_run.expanduser().resolve() / "events.jsonl"
        resume_metadata = {
            "source_run": str(args.resume_run.expanduser().resolve()),
            "source_run_id": source_manifest.get("run_id"),
            "source_decision": args.resume_decision,
            "source_events_sha256": sha256_file(source_events),
        }
    metadata = {
        "mode": "frozen_neural_evaluation",
        "requested_decisions": args.decisions,
        "device": str(device),
        "planning_config": asdict(config),
        "inputs": {
            "rom": {"name": args.rom.name, "sha256": rom_sha256},
            "core": {"name": args.core.name, "sha256": sha256_file(args.core)},
            "host": {"name": args.host.name, "sha256": sha256_file(args.host)},
            "checkpoint": {
                "name": args.checkpoint.name,
                "file_sha256": sha256_file(args.checkpoint),
                "parameter_sha256": before,
            },
        },
        "bootstrap": bootstrap_metadata(bootstrap_fixture),
        "episodic_resume": resume_metadata,
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
            restored = (
                None
                if args.resume_run is None
                else restore_logged_decision(
                    native_env, args.resume_run, args.resume_decision
                )
            )
            env = LoggedEnvironment(native_env, logger)
            agent = VerifiedNeuralAgent(env, model, device, config, event_logger=logger)
            if restored is not None:
                initial_frame = env.start_attempt_from_current(
                    restored.frame,
                    reason=(
                        f"episodic_resume:{restored.run_id}:"
                        f"decision-{restored.decision}"
                    ),
                )
                logger.log(
                    "episodic_resume_completed",
                    source_run_id=restored.run_id,
                    source_decision=restored.decision,
                    source_event_seq=restored.event_seq,
                    **logger.frame_fields(initial_frame),
                )
                agent.reset(initial_frame=initial_frame)
            elif bootstrap_fixture is None:
                agent.reset()
            else:
                initial_frame = apply_bootstrap_fixture(
                    env, bootstrap_fixture, rom_sha256
                )
                agent.reset(initial_frame=initial_frame)
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
