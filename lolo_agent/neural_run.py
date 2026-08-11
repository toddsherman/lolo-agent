from __future__ import annotations

import argparse
import json
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
from .experience_import import classify_reward_track, decode_logged_png
from .log_summary import build_run_summary
from .native_env import NativeLibretroEnv
from .neural_planner import NeuralPlanningConfig, VerifiedNeuralAgent
from .neural_world_model import ACTION_ORDER, choose_torch_device
from .pixels import Frame, signature_key
from .replay import restore_logged_decision, validate_replay_inputs
from .run_logging import LoggedEnvironment, RunLogger, read_events, sha256_file
from .spatial_returnability import load_returnability_checkpoint
from .spatial_shadow import SpatialShadowEvaluator
from .spatial_world_model import load_spatial_checkpoint


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


def load_episodic_scene_frames(
    run_dir: Path,
    through_decision: int,
    visited: Optional[set[Path]] = None,
) -> list[Frame]:
    """Load prior pixel observations for temporary scene-return memory."""

    run_dir = Path(run_dir).expanduser().resolve()
    visited = set() if visited is None else visited
    if run_dir in visited:
        raise RuntimeError(f"episodic resume cycle detected at {run_dir}")
    visited.add(run_dir)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    frames: list[Frame] = []
    resume = manifest.get("metadata", {}).get("episodic_resume")
    if resume:
        frames.extend(
            load_episodic_scene_frames(
                Path(resume["source_run"]),
                int(resume["source_decision"]),
                visited,
            )
        )
    seen_digests = {frame.digest for frame in frames}
    for event in read_events(run_dir):
        if event.get("event") != "decision_committed":
            continue
        if int(event.get("decision", 0)) > through_decision:
            continue
        digest = event.get("frame")
        if not digest or digest in seen_digests:
            continue
        frame_path = run_dir / "frames" / f"{digest}.png"
        if not frame_path.exists():
            continue
        frame = decode_logged_png(frame_path)
        seen_digests.add(frame.digest)
        frames.append(frame)
    return frames


def load_episodic_decision_events(
    run_dir: Path,
    through_decision: int,
    visited: Optional[set[Path]] = None,
) -> list[Dict[str, Any]]:
    """Load temporary planner-memory telemetry across a resume chain."""

    run_dir = Path(run_dir).expanduser().resolve()
    visited = set() if visited is None else visited
    if run_dir in visited:
        raise RuntimeError(f"episodic resume cycle detected at {run_dir}")
    visited.add(run_dir)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    events: list[Dict[str, Any]] = []
    resume = manifest.get("metadata", {}).get("episodic_resume")
    if resume:
        events.extend(
            load_episodic_decision_events(
                Path(resume["source_run"]),
                int(resume["source_decision"]),
                visited,
            )
        )
    events.extend(
        event
        for event in read_events(run_dir)
        if event.get("event")
        in {
            "decision_committed",
            "goal_milestone_exhaustion_learned",
            "pixel_novel_room_started",
        }
        and int(event.get("decision", 0)) <= through_decision
    )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen neural rollout planner")
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--spatial-shadow-checkpoint",
        type=Path,
        help=(
            "frozen spatial checkpoint to score candidates and verified branches; "
            "it remains observational unless --spatial-selection-weight is positive"
        ),
    )
    parser.add_argument(
        "--spatial-selection-weight",
        type=float,
        default=0.0,
        help=(
            "optional frozen spatial score for prioritizing save-state branch "
            "verification; verified commit scoring remains outcome-based"
        ),
    )
    parser.add_argument(
        "--spatial-returnability-checkpoint",
        type=Path,
        help=(
            "optional frozen observed-returnability sidecar; requires the exact "
            "spatial checkpoint it was trained against and remains telemetry-only"
        ),
    )
    parser.add_argument(
        "--returnability-probe-depth",
        type=int,
        default=0,
        help=(
            "telemetry-only save-state search for a matched-NOOP pixel return; "
            "zero disables probing"
        ),
    )
    parser.add_argument(
        "--returnability-probe-beam-width",
        type=int,
        default=4,
        help="closest endpoint states retained at each return-probe depth",
    )
    parser.add_argument(
        "--returnability-probe-pixel-l1-threshold",
        type=float,
        default=0.002,
        help="maximum matched-NOOP mean pixel difference counted as a return",
    )
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
        "--causal-cell-coverage-weight",
        type=float,
        default=0.0,
        help=(
            "reward action-caused changes in globally under-visited coarse "
            "screen cells; zero disables the reward"
        ),
    )
    parser.add_argument(
        "--causal-cell-recovery-grace-decisions",
        type=int,
        default=0,
        help=(
            "local decisions allowed after reaching a globally unvisited "
            "controlled cell before delayed-return archive recovery"
        ),
    )
    parser.add_argument(
        "--autonomous-grace-decisions",
        type=int,
        default=NeuralPlanningConfig().autonomous_grace_decisions,
        help=(
            "passive counterfactual observations allowed after detecting "
            "action-independent dynamics before forcing an intervention"
        ),
    )
    parser.add_argument(
        "--control-collapse-confirmation-steps",
        type=int,
        default=NeuralPlanningConfig().control_collapse_confirmation_steps,
        help=(
            "save-state lookahead steps used to distinguish permanent loss "
            "of control from a temporary animation or novel scene transition"
        ),
    )
    parser.add_argument(
        "--delayed-transition-probe-steps",
        type=int,
        default=NeuralPlanningConfig().delayed_transition_probe_steps,
        help=(
            "counterfactual matched-NOOP steps used to discover delayed dark "
            "transitions into visually novel scenes; zero disables probing"
        ),
    )
    parser.add_argument(
        "--dark-transition-intensity-threshold",
        type=float,
        default=NeuralPlanningConfig().dark_transition_intensity_threshold,
        help="mean pixel intensity at or below which a scene is considered dark",
    )
    parser.add_argument(
        "--known-scene-return-distance-threshold",
        type=float,
        default=NeuralPlanningConfig().known_scene_return_distance_threshold,
        help=(
            "maximum coarse pixel distance for treating a post-dark layout "
            "as a return to temporary scene memory"
        ),
    )
    parser.add_argument(
        "--behavioral-edge-coverage-weight",
        type=float,
        default=0.0,
        help=(
            "reward controller action/duration edges that have been committed "
            "less often from a learned behavioral state; zero disables it"
        ),
    )
    parser.add_argument(
        "--behavioral-best-first-archive",
        action="store_true",
        help=(
            "when recovering from a loop, prefer uncommitted intervention "
            "edges within the already eligible archive frontier"
        ),
    )
    parser.add_argument(
        "--persistent-change-stability-decisions",
        type=int,
        default=0,
        help=(
            "consecutive observations required before a changed coarse cell "
            "temporarily constrains archive recovery; zero disables it"
        ),
    )
    parser.add_argument(
        "--persistent-change-minimum-value-drop",
        type=int,
        default=0,
        help=(
            "optional minimum 4-bit coarse-intensity decrease for persistent "
            "change evidence; zero accepts changes in either direction"
        ),
    )
    parser.add_argument(
        "--persistent-change-speculative-recovery",
        action="store_true",
        help=(
            "during archive recovery, provisionally preserve newly observed "
            "coarse intensity drops until normal stability evidence resolves them"
        ),
    )
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
        "--human-prior-best-first-archive",
        action="store_true",
        help=(
            "on the explicitly labelled reward track, expand archived "
            "controller edges by stable pixel-detected goal state before "
            "animation-sensitive behavioral clusters"
        ),
    )
    parser.add_argument(
        "--human-prior-graph-stagnation-visits",
        type=int,
        default=0,
        help=(
            "repeated stable pixel-detected goal states required before "
            "assisted archive backtracking; zero disables the trigger"
        ),
    )
    parser.add_argument(
        "--human-prior-phase-position-novelty",
        action="store_true",
        help=(
            "on the assisted track, count player-position novelty separately "
            "for each visible heart configuration and treasure phase"
        ),
    )
    parser.add_argument(
        "--human-prior-goal-exhaustion-rollback",
        action="store_true",
        help=(
            "on the assisted track, learn a small negative value for an exact "
            "goal milestone and restore its pre-action state after both the "
            "semantic graph and verified option frontier are exhausted"
        ),
    )
    parser.add_argument(
        "--human-prior-option-search-depth",
        type=int,
        default=0,
        help=(
            "exact save-state action-sequence depth searched when the "
            "assisted semantic graph stagnates; zero disables search"
        ),
    )
    parser.add_argument(
        "--human-prior-option-search-beam-width",
        type=int,
        default=8,
        help="pixel endpoints retained at each assisted option-search depth",
    )
    parser.add_argument(
        "--human-prior-option-search-action-frames",
        type=int,
        default=0,
        help=(
            "press length for assisted option search; zero uses the longest "
            "configured gameplay duration"
        ),
    )
    parser.add_argument(
        "--human-prior-option-effect-stability-steps",
        type=int,
        default=0,
        help=(
            "future matched-NOOP horizons used to audit persistence of "
            "player-masked option effects; zero disables the audit"
        ),
    )
    parser.add_argument(
        "--human-prior-option-effect-probe-limit",
        type=int,
        default=8,
        help="distinct option effects audited per sequence search",
    )
    parser.add_argument(
        "--human-prior-option-effect-max-stable-cells",
        type=int,
        default=4,
        help=(
            "maximum persistent nonlocal coarse cells treated as a localized "
            "option effect"
        ),
    )
    parser.add_argument(
        "--human-prior-option-effect-phase-offsets",
        type=int,
        default=0,
        help=(
            "nearby future NOOP frame offsets searched for an animation-phase "
            "match; zero disables this audit"
        ),
    )
    parser.add_argument(
        "--human-prior-option-effect-phase-l1-threshold",
        type=float,
        default=0.002,
        help=(
            "maximum normalized patch L1 treated as phase-equivalent to a "
            "future all-NOOP control"
        ),
    )
    parser.add_argument(
        "--human-prior-option-effect-frontier",
        action="store_true",
        help=(
            "on the assisted track, archive safe localized persistent option "
            "effects confirmed by leave-one-action-out controls"
        ),
    )
    parser.add_argument(
        "--human-prior-option-effect-controllability-depth",
        type=int,
        default=1,
        help=(
            "directional sequence depth used to compare reachability after "
            "factual and action-ablated persistent effects (1-4)"
        ),
    )
    parser.add_argument(
        "--human-prior-option-effect-local-controls",
        action="store_true",
        help=(
            "telemetry-only action ablations for compact persistent effects "
            "when factual and control player endpoints match"
        ),
    )
    parser.add_argument(
        "--human-prior-option-effect-local-minimum-cell-pixels",
        type=int,
        default=12,
        help=(
            "minimum unmasked changed pixels required per coarse cell in "
            "telemetry-only endpoint-matched local controls"
        ),
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
    if args.causal_cell_coverage_weight < 0.0:
        parser.error("--causal-cell-coverage-weight must be non-negative")
    if args.causal_cell_recovery_grace_decisions < 0:
        parser.error(
            "--causal-cell-recovery-grace-decisions must be non-negative"
        )
    if args.autonomous_grace_decisions < 0:
        parser.error("--autonomous-grace-decisions must be non-negative")
    if args.control_collapse_confirmation_steps <= 0:
        parser.error(
            "--control-collapse-confirmation-steps must be positive"
        )
    if args.delayed_transition_probe_steps < 0:
        parser.error("--delayed-transition-probe-steps must be non-negative")
    if not 0.0 <= args.dark_transition_intensity_threshold <= 1.0:
        parser.error(
            "--dark-transition-intensity-threshold must be between zero and one"
        )
    if not 0.0 <= args.known_scene_return_distance_threshold <= 1.0:
        parser.error(
            "--known-scene-return-distance-threshold must be between zero and one"
        )
    if args.behavioral_edge_coverage_weight < 0.0:
        parser.error(
            "--behavioral-edge-coverage-weight must be non-negative"
        )
    if args.persistent_change_stability_decisions < 0:
        parser.error(
            "--persistent-change-stability-decisions must be non-negative"
        )
    if not 0 <= args.persistent_change_minimum_value_drop <= 15:
        parser.error(
            "--persistent-change-minimum-value-drop must be between 0 and 15"
        )
    if args.spatial_selection_weight < 0.0:
        parser.error("--spatial-selection-weight must be non-negative")
    if args.returnability_probe_depth < 0:
        parser.error("--returnability-probe-depth must be non-negative")
    if args.returnability_probe_beam_width <= 0:
        parser.error("--returnability-probe-beam-width must be positive")
    if args.returnability_probe_pixel_l1_threshold < 0.0:
        parser.error(
            "--returnability-probe-pixel-l1-threshold must be non-negative"
        )
    if (
        args.spatial_selection_weight > 0.0
        and args.spatial_shadow_checkpoint is None
    ):
        parser.error(
            "--spatial-selection-weight requires --spatial-shadow-checkpoint"
        )
    if (
        args.spatial_returnability_checkpoint is not None
        and args.spatial_shadow_checkpoint is None
    ):
        parser.error(
            "--spatial-returnability-checkpoint requires --spatial-shadow-checkpoint"
        )
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
    if args.human_prior_graph_stagnation_visits < 0:
        parser.error(
            "--human-prior-graph-stagnation-visits must be non-negative"
        )
    if args.human_prior_goal_exhaustion_rollback and (
        not args.human_prior_best_first_archive
        or args.human_prior_graph_stagnation_visits <= 0
        or args.human_prior_option_search_depth < 2
    ):
        parser.error(
            "--human-prior-goal-exhaustion-rollback requires assisted "
            "best-first archive, positive graph stagnation visits, and "
            "option search depth of at least two"
        )
    if args.human_prior_option_search_depth < 0:
        parser.error(
            "--human-prior-option-search-depth must be non-negative"
        )
    if args.human_prior_option_search_beam_width <= 0:
        parser.error(
            "--human-prior-option-search-beam-width must be positive"
        )
    if args.human_prior_option_search_action_frames < 0:
        parser.error(
            "--human-prior-option-search-action-frames must be non-negative"
        )
    if args.human_prior_option_effect_stability_steps < 0:
        parser.error(
            "--human-prior-option-effect-stability-steps must be non-negative"
        )
    if args.human_prior_option_effect_probe_limit <= 0:
        parser.error(
            "--human-prior-option-effect-probe-limit must be positive"
        )
    if args.human_prior_option_effect_max_stable_cells <= 0:
        parser.error(
            "--human-prior-option-effect-max-stable-cells must be positive"
        )
    if args.human_prior_option_effect_phase_offsets < 0:
        parser.error(
            "--human-prior-option-effect-phase-offsets must be non-negative"
        )
    if args.human_prior_option_effect_phase_l1_threshold < 0.0:
        parser.error(
            "--human-prior-option-effect-phase-l1-threshold must be non-negative"
        )
    if args.human_prior_option_effect_local_minimum_cell_pixels <= 0:
        parser.error(
            "--human-prior-option-effect-local-minimum-cell-pixels must be positive"
        )
    if (
        args.human_prior_option_effect_frontier
        and args.human_prior_option_effect_phase_offsets <= 0
    ):
        parser.error(
            "--human-prior-option-effect-frontier requires "
            "--human-prior-option-effect-phase-offsets"
        )
    if not 1 <= args.human_prior_option_effect_controllability_depth <= 4:
        parser.error(
            "--human-prior-option-effect-controllability-depth must be "
            "between 1 and 4"
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
    selected_durations = action_durations or (args.action_frames,)
    spatial_shadow = None
    spatial_shadow_before = None
    spatial_shadow_horizon = None
    spatial_returnability_before = None
    spatial_returnability_configuration = None
    if args.spatial_shadow_checkpoint is not None:
        shadow_model, spatial_shadow_horizon = load_spatial_checkpoint(
            args.spatial_shadow_checkpoint,
            device=device,
            frozen=True,
        )
        if shadow_model.duration_conditioned:
            if max(selected_durations) > shadow_model.max_action_frames:
                parser.error(
                    "planner action duration exceeds spatial shadow checkpoint limit"
                )
        elif any(
            duration != shadow_model.fixed_action_frames
            for duration in selected_durations
        ):
            parser.error(
                "planner action durations do not match the fixed-duration spatial "
                "shadow checkpoint"
            )
        returnability_model = None
        if args.spatial_returnability_checkpoint is not None:
            (
                returnability_model,
                spatial_returnability_configuration,
            ) = load_returnability_checkpoint(
                args.spatial_returnability_checkpoint,
                shadow_model.checkpoint_digest,
                device=device,
                frozen=True,
            )
            spatial_returnability_before = returnability_model.checkpoint_digest
        target_metadata = (spatial_returnability_configuration or {}).get(
            "target_metadata", {}
        )
        spatial_shadow = SpatialShadowEvaluator(
            shadow_model,
            device,
            returnability_model=returnability_model,
            returnability_observed_endpoints=(
                target_metadata.get("relation_tokens")
                == "observed source and verified endpoint pixels"
            ),
        )
        spatial_shadow_before = spatial_shadow.checkpoint_digest
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
        causal_cell_coverage_weight=args.causal_cell_coverage_weight,
        causal_cell_recovery_grace_decisions=(
            args.causal_cell_recovery_grace_decisions
        ),
        autonomous_grace_decisions=args.autonomous_grace_decisions,
        control_collapse_confirmation_steps=(
            args.control_collapse_confirmation_steps
        ),
        delayed_transition_probe_steps=args.delayed_transition_probe_steps,
        dark_transition_intensity_threshold=(
            args.dark_transition_intensity_threshold
        ),
        known_scene_return_distance_threshold=(
            args.known_scene_return_distance_threshold
        ),
        behavioral_edge_coverage_weight=(
            args.behavioral_edge_coverage_weight
        ),
        behavioral_best_first_archive=(
            args.behavioral_best_first_archive
        ),
        persistent_change_stability_decisions=(
            args.persistent_change_stability_decisions
        ),
        persistent_change_minimum_value_drop=(
            args.persistent_change_minimum_value_drop
        ),
        persistent_change_speculative_recovery=(
            args.persistent_change_speculative_recovery
        ),
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
        human_prior_best_first_archive=(
            args.human_prior_best_first_archive
            if args.human_prior_hearts
            else False
        ),
        human_prior_phase_position_novelty=(
            args.human_prior_phase_position_novelty
            if args.human_prior_hearts
            else False
        ),
        human_prior_graph_stagnation_visits=(
            args.human_prior_graph_stagnation_visits
            if args.human_prior_hearts
            else 0
        ),
        human_prior_goal_exhaustion_rollback=(
            args.human_prior_goal_exhaustion_rollback
            if args.human_prior_hearts
            else False
        ),
        human_prior_option_search_depth=(
            args.human_prior_option_search_depth
            if args.human_prior_hearts
            else 0
        ),
        human_prior_option_search_beam_width=(
            args.human_prior_option_search_beam_width
        ),
        human_prior_option_search_action_frames=(
            args.human_prior_option_search_action_frames
        ),
        human_prior_option_effect_stability_steps=(
            args.human_prior_option_effect_stability_steps
        ),
        human_prior_option_effect_probe_limit=(
            args.human_prior_option_effect_probe_limit
        ),
        human_prior_option_effect_max_stable_cells=(
            args.human_prior_option_effect_max_stable_cells
        ),
        human_prior_option_effect_phase_offsets=(
            args.human_prior_option_effect_phase_offsets
        ),
        human_prior_option_effect_phase_l1_threshold=(
            args.human_prior_option_effect_phase_l1_threshold
        ),
        human_prior_option_effect_frontier=(
            args.human_prior_option_effect_frontier
            if args.human_prior_hearts
            else False
        ),
        human_prior_option_effect_controllability_depth=(
            args.human_prior_option_effect_controllability_depth
        ),
        human_prior_option_effect_local_controls=(
            args.human_prior_option_effect_local_controls
            if args.human_prior_hearts
            else False
        ),
        human_prior_option_effect_local_minimum_cell_pixels=(
            args.human_prior_option_effect_local_minimum_cell_pixels
        ),
        human_prior_intrinsic_clip=args.human_prior_intrinsic_clip,
        spatial_selection_weight=args.spatial_selection_weight,
        returnability_probe_depth=args.returnability_probe_depth,
        returnability_probe_beam_width=args.returnability_probe_beam_width,
        returnability_probe_pixel_l1_threshold=(
            args.returnability_probe_pixel_l1_threshold
        ),
    )
    rom_sha256 = sha256_file(args.rom)
    resume_metadata = None
    resume_reward_track = None
    if args.resume_run is not None:
        source_manifest = validate_replay_inputs(
            args.resume_run, args.host, args.core, args.rom
        )
        source_events = args.resume_run.expanduser().resolve() / "events.jsonl"
        resume_reward_track = classify_reward_track(source_manifest)
        resume_metadata = {
            "source_run": str(args.resume_run.expanduser().resolve()),
            "source_run_id": source_manifest.get("run_id"),
            "source_decision": args.resume_decision,
            "source_events_sha256": sha256_file(source_events),
            "source_reward_track": resume_reward_track,
        }
    inputs = {
        "rom": {"name": args.rom.name, "sha256": rom_sha256},
        "core": {"name": args.core.name, "sha256": sha256_file(args.core)},
        "host": {"name": args.host.name, "sha256": sha256_file(args.host)},
        "checkpoint": {
            "name": args.checkpoint.name,
            "file_sha256": sha256_file(args.checkpoint),
            "parameter_sha256": before,
        },
    }
    if args.spatial_shadow_checkpoint is not None:
        inputs["spatial_shadow_checkpoint"] = {
            "name": args.spatial_shadow_checkpoint.name,
            "file_sha256": sha256_file(args.spatial_shadow_checkpoint),
            "parameter_sha256": spatial_shadow_before,
            "planning_horizon": spatial_shadow_horizon,
            "mode": (
                "verification_priority"
                if args.spatial_selection_weight > 0.0
                else "observational"
            ),
            "selection_weight": args.spatial_selection_weight,
        }
    if args.spatial_returnability_checkpoint is not None:
        inputs["spatial_returnability_checkpoint"] = {
            "name": args.spatial_returnability_checkpoint.name,
            "file_sha256": sha256_file(args.spatial_returnability_checkpoint),
            "parameter_sha256": spatial_returnability_before,
            "mode": "observational",
            **(spatial_returnability_configuration or {}),
        }
    metadata = {
        "mode": "frozen_neural_evaluation",
        "reward_track": (
            "human_prior_v2"
            if args.human_prior_hearts
            else (
                "human_prior_resume_observational"
                if resume_reward_track == "assisted"
                else "strict_rule_free"
            )
        ),
        "requested_decisions": args.decisions,
        "device": str(device),
        "planning_config": asdict(config),
        "inputs": inputs,
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
            agent = VerifiedNeuralAgent(
                env,
                model,
                device,
                config,
                event_logger=logger,
                spatial_shadow=spatial_shadow,
            )
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
                agent.seed_bright_scene_memory(
                    load_episodic_scene_frames(
                        args.resume_run, args.resume_decision
                    )
                )
                agent.seed_human_prior_episodic_memory(
                    load_episodic_decision_events(
                        args.resume_run, args.resume_decision
                    )
                )
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
            try:
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
            finally:
                try:
                    agent.clear_archive()
                except Exception as cleanup_error:
                    logger.log(
                        "agent_cleanup_failed",
                        error_type=type(cleanup_error).__name__,
                        error=str(cleanup_error),
                    )
        after = model.checkpoint_digest
        if before != after:
            raise RuntimeError("frozen evaluation changed persistent parameters")
        logger.log(
            "frozen_parameter_audit",
            status="pass",
            parameter_sha256_before=before,
            parameter_sha256_after=after,
        )
        if spatial_shadow is not None:
            spatial_shadow_after = spatial_shadow.checkpoint_digest
            if spatial_shadow_before != spatial_shadow_after:
                raise RuntimeError(
                    "frozen spatial shadow evaluation changed persistent parameters"
                )
            logger.log(
                "spatial_shadow_parameter_audit",
                status="pass",
                spatial_shadow_mode=(
                    "verification_priority"
                    if args.spatial_selection_weight > 0.0
                    else "observational"
                ),
                spatial_shadow_selection_weight=args.spatial_selection_weight,
                parameter_sha256_before=spatial_shadow_before,
                parameter_sha256_after=spatial_shadow_after,
            )
            if spatial_returnability_before is not None:
                spatial_returnability_after = (
                    spatial_shadow.returnability_checkpoint_digest
                )
                if spatial_returnability_before != spatial_returnability_after:
                    raise RuntimeError(
                        "frozen returnability evaluation changed persistent parameters"
                    )
                logger.log(
                    "spatial_returnability_parameter_audit",
                    status="pass",
                    mode="observational",
                    parameter_sha256_before=spatial_returnability_before,
                    parameter_sha256_after=spatial_returnability_after,
                )
        logger.close("complete")
    except BaseException as exc:
        logger.close(
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
            str(exc),
        )
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
