from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
from .pixels import Frame, signature_key
from .replay import restore_logged_decision, validate_replay_inputs
from .run_logging import LoggedEnvironment, RunLogger, sha256_file


@dataclass
class StableSceneChangeDetector:
    """Evaluator-only visual stop rule; its state is never exposed to the agent.

    A distinct in-room state is not enough to trigger the rule.  The detector first
    requires a near-black transition frame, then waits for a non-dark scene to
    remain stable.  This keeps ordinary puzzle-state changes from masquerading as
    room boundaries without introducing any game-specific visual definitions.
    """

    initial_frame: Frame
    stable_observations: int = 2
    warmup_decisions: int = 4
    minimum_difference: float = 0.05
    dark_frame_threshold: float = 0.02
    minimum_scene_intensity: float = 0.05

    def __post_init__(self) -> None:
        if self.stable_observations <= 0:
            raise ValueError("stable observations must be positive")
        if self.warmup_decisions < 0:
            raise ValueError("warmup decisions must be non-negative")
        if self.minimum_difference < 0.0:
            raise ValueError("minimum difference must be non-negative")
        if not 0.0 <= self.dark_frame_threshold <= 1.0:
            raise ValueError("dark frame threshold must be between zero and one")
        if not 0.0 <= self.minimum_scene_intensity <= 1.0:
            raise ValueError("minimum scene intensity must be between zero and one")
        if self.minimum_scene_intensity <= self.dark_frame_threshold:
            raise ValueError("minimum scene intensity must exceed dark frame threshold")
        self._baseline = {self._signature(self.initial_frame)}
        self._transition_observed = False
        self._candidate: Optional[str] = None
        self._candidate_count = 0

    @staticmethod
    def _signature(frame: Frame) -> str:
        return signature_key(frame.coarse_signature(columns=3, rows=3))

    @staticmethod
    def _mean_intensity(frame: Frame) -> float:
        return sum(frame.pixels) / (255.0 * len(frame.pixels))

    def observe(self, decision: int, frame: Frame) -> Optional[Dict[str, Any]]:
        scene = self._signature(frame)
        difference = self.initial_frame.mean_absolute_difference(frame)
        intensity = self._mean_intensity(frame)
        if decision <= self.warmup_decisions:
            self._baseline.add(scene)
            self._candidate = None
            self._candidate_count = 0
            return None
        if intensity <= self.dark_frame_threshold:
            self._transition_observed = True
            self._candidate = None
            self._candidate_count = 0
            return None
        if not self._transition_observed or intensity < self.minimum_scene_intensity:
            self._candidate = None
            self._candidate_count = 0
            return None
        if scene in self._baseline or difference < self.minimum_difference:
            self._candidate = None
            self._candidate_count = 0
            return None
        if scene == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = scene
            self._candidate_count = 1
        if self._candidate_count < self.stable_observations:
            return None
        return {
            "decision": decision,
            "scene_signature": scene,
            "stable_observations": self._candidate_count,
            "minimum_difference": self.minimum_difference,
            "difference_from_initial": difference,
            "dark_transition_observed": self._transition_observed,
            "dark_frame_threshold": self.dark_frame_threshold,
            "minimum_scene_intensity": self.minimum_scene_intensity,
            "scene_intensity": intensity,
            "baseline_scene_signatures": sorted(self._baseline),
            "frame": frame.digest,
        }


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
    parser.add_argument("--archive-capacity", type=int, default=256)
    parser.add_argument("--archive-max-age", type=int, default=512)
    parser.add_argument(
        "--consecutive-repeat-penalty-cap",
        type=float,
        help="optional cap on the weighted consecutive-repeat penalty",
    )
    parser.add_argument(
        "--delayed-return-penalty-cap",
        type=float,
        help="optional cap on the weighted delayed-return penalty",
    )
    parser.add_argument(
        "--human-prior-hearts",
        action="store_true",
        help="enable the explicitly labeled pixel-heart goal-reward track",
    )
    parser.add_argument("--human-prior-heart-reward", type=float, default=25.0)
    parser.add_argument(
        "--human-prior-all-hearts-reward", type=float, default=75.0
    )
    parser.add_argument("--human-prior-chest-reward", type=float, default=100.0)
    parser.add_argument(
        "--human-prior-navigation-reward",
        type=float,
        default=0.0,
        help="reward per tile of pixel-detected progress toward a remaining heart",
    )
    parser.add_argument(
        "--human-prior-navigation-recovery-grace",
        type=int,
        default=2,
        help="decisions before delayed-return recovery may abandon a closer heart frontier",
    )
    parser.add_argument(
        "--human-prior-life-loss-penalty",
        type=float,
        default=100.0,
        help="penalty for a pixel-confirmed HUD life change after a dark transition",
    )
    parser.add_argument("--human-prior-intrinsic-clip", type=float, default=10.0)
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
        "--stop-on-stable-scene-change",
        type=int,
        default=0,
        metavar="OBSERVATIONS",
        help="evaluator-only stop after a visually distinct scene remains stable; disabled by default",
    )
    parser.add_argument("--scene-change-warmup", type=int, default=4)
    parser.add_argument("--scene-change-min-difference", type=float, default=0.05)
    parser.add_argument("--scene-change-dark-threshold", type=float, default=0.02)
    parser.add_argument("--scene-change-min-intensity", type=float, default=0.05)
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
    if args.stop_on_stable_scene_change < 0:
        parser.error("--stop-on-stable-scene-change must be non-negative")
    if args.scene_change_warmup < 0:
        parser.error("--scene-change-warmup must be non-negative")
    if args.scene_change_min_difference < 0.0:
        parser.error("--scene-change-min-difference must be non-negative")
    if not 0.0 <= args.scene_change_dark_threshold <= 1.0:
        parser.error("--scene-change-dark-threshold must be between zero and one")
    if not 0.0 <= args.scene_change_min_intensity <= 1.0:
        parser.error("--scene-change-min-intensity must be between zero and one")
    if args.scene_change_min_intensity <= args.scene_change_dark_threshold:
        parser.error("--scene-change-min-intensity must exceed --scene-change-dark-threshold")
    if args.archive_capacity <= 0:
        parser.error("--archive-capacity must be positive")
    if args.archive_max_age <= 0:
        parser.error("--archive-max-age must be positive")
    if (
        args.consecutive_repeat_penalty_cap is not None
        and args.consecutive_repeat_penalty_cap < 0.0
    ):
        parser.error("--consecutive-repeat-penalty-cap must be non-negative")
    if (
        args.delayed_return_penalty_cap is not None
        and args.delayed_return_penalty_cap < 0.0
    ):
        parser.error("--delayed-return-penalty-cap must be non-negative")
    if args.human_prior_heart_reward < 0.0:
        parser.error("--human-prior-heart-reward must be non-negative")
    if args.human_prior_all_hearts_reward < 0.0:
        parser.error("--human-prior-all-hearts-reward must be non-negative")
    if args.human_prior_chest_reward < 0.0:
        parser.error("--human-prior-chest-reward must be non-negative")
    if args.human_prior_navigation_reward < 0.0:
        parser.error("--human-prior-navigation-reward must be non-negative")
    if args.human_prior_life_loss_penalty < 0.0:
        parser.error("--human-prior-life-loss-penalty must be non-negative")
    if args.human_prior_navigation_recovery_grace < 0:
        parser.error(
            "--human-prior-navigation-recovery-grace must be non-negative"
        )
    if args.human_prior_intrinsic_clip <= 0.0:
        parser.error("--human-prior-intrinsic-clip must be positive")

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
        archive_capacity=args.archive_capacity,
        archive_max_age=args.archive_max_age,
        consecutive_repeat_penalty_cap=args.consecutive_repeat_penalty_cap,
        delayed_return_penalty_cap=args.delayed_return_penalty_cap,
        human_prior_heart_reward=(
            args.human_prior_heart_reward if args.human_prior_hearts else 0.0
        ),
        human_prior_all_hearts_reward=(
            args.human_prior_all_hearts_reward
            if args.human_prior_hearts
            else 0.0
        ),
        human_prior_chest_reward=(
            args.human_prior_chest_reward if args.human_prior_hearts else 0.0
        ),
        human_prior_navigation_reward=(
            args.human_prior_navigation_reward
            if args.human_prior_hearts
            else 0.0
        ),
        human_prior_life_loss_penalty=(
            args.human_prior_life_loss_penalty if args.human_prior_hearts else 0.0
        ),
        human_prior_navigation_recovery_grace=(
            args.human_prior_navigation_recovery_grace
            if args.human_prior_hearts
            else 0
        ),
        human_prior_intrinsic_clip=args.human_prior_intrinsic_clip,
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
        "reward_track": (
            "human_prior_v2" if args.human_prior_hearts else "strict_rule_free"
        ),
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
        "evaluator_stop": {
            "kind": "stable_scene_change",
            "stable_observations": args.stop_on_stable_scene_change,
            "warmup_decisions": args.scene_change_warmup,
            "minimum_difference": args.scene_change_min_difference,
            "dark_frame_threshold": args.scene_change_dark_threshold,
            "minimum_scene_intensity": args.scene_change_min_intensity,
            "requires_dark_transition": True,
            "agent_visible": False,
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
                initial_frame = agent.reset()
            else:
                initial_frame = apply_bootstrap_fixture(
                    env, bootstrap_fixture, rom_sha256
                )
                agent.reset(initial_frame=initial_frame)
            detector = (
                StableSceneChangeDetector(
                    initial_frame,
                    stable_observations=args.stop_on_stable_scene_change,
                    warmup_decisions=args.scene_change_warmup,
                    minimum_difference=args.scene_change_min_difference,
                    dark_frame_threshold=args.scene_change_dark_threshold,
                    minimum_scene_intensity=args.scene_change_min_intensity,
                )
                if args.stop_on_stable_scene_change
                else None
            )
            decisions = []
            for decision_index in range(1, args.decisions + 1):
                decision = agent.decide()
                decisions.append(decision)
                stop = (
                    None
                    if detector is None
                    else detector.observe(decision_index, decision.frame)
                )
                if stop is not None:
                    logger.log(
                        "evaluator_stable_scene_change",
                        agent_visible=False,
                        **stop,
                    )
                    break
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
