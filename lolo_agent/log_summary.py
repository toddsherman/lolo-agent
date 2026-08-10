from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .run_logging import SCHEMA_VERSION, read_events, utc_now


def read_annotations(run_dir: Path) -> Iterable[Dict[str, Any]]:
    path = Path(run_dir) / "evaluator_annotations.jsonl"
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


def append_level_annotation(
    run_dir: Path,
    label: str,
    start_seq: Optional[int] = None,
    end_seq: Optional[int] = None,
    attempt: Optional[int] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Add evaluator-only metadata after a run; it is never visible to the agent."""

    run_dir = Path(run_dir).expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    annotation = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "time_utc": utc_now(),
        "kind": "level",
        "label": label,
        "start_seq": start_seq,
        "end_seq": end_seq,
        "attempt": attempt,
        "note": note,
        "source": "evaluator",
    }
    with (run_dir / "evaluator_annotations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(annotation, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()
    return annotation


def _level_for_event(event: Dict[str, Any], annotations: List[Dict[str, Any]]) -> Optional[str]:
    seq = int(event["seq"])
    attempt = int(event.get("attempt", 0))
    matches = []
    for annotation in annotations:
        if annotation.get("kind") != "level":
            continue
        annotated_attempt = annotation.get("attempt")
        if annotated_attempt is not None and int(annotated_attempt) != attempt:
            continue
        start = annotation.get("start_seq")
        end = annotation.get("end_seq")
        if start is not None and seq < int(start):
            continue
        if end is not None and seq > int(end):
            continue
        matches.append(annotation)
    return None if not matches else str(matches[-1]["label"])


def build_run_summary(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    events = list(read_events(run_dir))
    annotations = list(read_annotations(run_dir))
    event_counts = Counter(event["event"] for event in events)
    actions = Counter()
    durations = Counter()
    investigated_actions = Counter()
    investigated_durations = Counter()
    frames: set[str] = set()
    scenes: set[str] = set()
    attempts: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "decisions": 0,
            "restores": 0,
            "actions": Counter(),
            "durations": Counter(),
            "env_steps": 0,
        }
    )
    decision_rows: List[Dict[str, Any]] = []
    delayed_return_recoveries = 0
    frontier_penalized_traces = 0
    frontier_choice_samples = 0
    committed_frontier_values: List[float] = []
    abstraction_clusters: set[str] = set()
    behavior_clusters: set[str] = set()
    behavior_probe_reasons: Counter[str] = Counter()
    behavior_probe_controls: Counter[str] = Counter()
    temporal_option_samples = 0
    learned_temporal_option_values: List[float] = []
    learned_temporal_option_choices: set[Tuple[str, str, int]] = set()
    delayed_temporal_option_samples = 0
    temporal_option_endpoint_contrasts: List[float] = []
    bootstrap_actions = Counter()
    bootstrap_durations = Counter()
    bootstrap_frames = 0
    action_effect_observations = 0
    action_effect_observations_by_action = Counter()
    action_effect_known_branches = 0
    learned_hazard_filtered_choices = 0
    global_action_hazard_samples = 0
    causal_spatial_observations = 0
    causal_spatial_signatures: set[str] = set()
    committed_causal_spatial_signatures: set[str] = set()
    causal_events_detected = 0
    archive_rejections_by_reason: Counter[str] = Counter()

    edge_counts: Counter[Tuple[str, str, str, int]] = Counter()
    node_details: Dict[str, Dict[str, Any]] = {}
    for event in events:
        if event["event"] == "archive_branch_rejected":
            archive_rejections_by_reason[
                str(event.get("reason", "unspecified"))
            ] += 1
        if event["event"] == "branch_verified":
            if event.get("action_effect_contrast") is not None:
                action_effect_observations += 1
                action_effect_observations_by_action[str(event["action"])] += 1
            if event.get("action_effect_is_known"):
                action_effect_known_branches += 1
            causal_signature = event.get("causal_spatial_signature")
            if causal_signature:
                causal_spatial_observations += 1
                causal_spatial_signatures.add(str(causal_signature))
        elif event["event"] == "learned_hazards_filtered":
            learned_hazard_filtered_choices += len(event.get("filtered", ()))
        if (
            event["event"] == "temporal_option_completed"
            and event.get("action_hazard_generalized")
        ):
            global_action_hazard_samples += 1
        if (
            event["event"] == "archive_branch_restored"
            and event.get("reason") == "delayed_visual_return"
        ):
            delayed_return_recoveries += 1
        if event["event"] == "persistent_frontier_return_penalized":
            penalized = event.get("penalized_traces", ())
            frontier_penalized_traces += len(penalized)
            frontier_choice_samples += sum(
                item.get("choice") is not None for item in penalized
            )
        elif event["event"] == "persistent_frontier_updated":
            frontier_choice_samples += sum(
                item.get("choice") is not None
                for item in event.get("completed_samples", ())
            )
        if event["event"] == "visual_abstraction_assigned":
            abstraction_clusters.add(str(event["cluster"]))
        elif event["event"] == "behavioral_abstraction_assigned":
            behavior_clusters.add(str(event["cluster"]))
        elif event["event"] == "behavior_probe_selected":
            behavior_probe_reasons[str(event["reason"])] += 1
            selected_control = event.get("selected_control")
            if selected_control is not None:
                behavior_probe_controls[str(selected_control)] += 1
        elif event["event"] == "temporal_option_completed" and event.get(
            "credited"
        ):
            temporal_option_samples += 1
            learned_value = event.get("learned_value")
            if learned_value is not None:
                learned_temporal_option_values.append(float(learned_value))
            choice = event.get("choice")
            if choice is not None and len(choice) == 3:
                learned_temporal_option_choices.add(
                    (str(choice[0]), str(choice[1]), int(choice[2]))
                )
            if int(event.get("counterfactual_steps", 0)) > 0:
                delayed_temporal_option_samples += 1
        if event["event"] == "temporal_option_completed":
            endpoint_contrast = event.get("counterfactual_endpoint_contrast")
            if endpoint_contrast is not None:
                temporal_option_endpoint_contrasts.append(
                    float(endpoint_contrast)
                )
        frame = event.get("frame") or event.get("target_frame")
        if frame:
            frames.add(frame)
            node_details.setdefault(
                frame,
                {
                    "id": frame,
                    "image": f"frames/{frame}.png",
                    "scene_signature": event.get("scene_signature"),
                    "visual_signature": event.get("visual_signature"),
                },
            )
        if event.get("scene_signature"):
            scenes.add(event["scene_signature"])
        attempt = int(event.get("attempt", 0))
        if event["event"] == "env_step":
            action = str(event["action"])
            action_frames = int(event["action_frames"])
            if event.get("phase") == "bootstrap":
                bootstrap_actions[action] += 1
                bootstrap_durations[action_frames] += 1
                bootstrap_frames += action_frames
            else:
                investigated_actions[action] += 1
                investigated_durations[action_frames] += 1
                attempts[attempt]["env_steps"] += 1
            source = event.get("source_frame")
            target = event.get("target_frame")
            if source and target:
                edge_counts[(source, target, action, action_frames)] += 1
        elif event["event"] == "decision_committed":
            if event.get("causal_event_detected"):
                causal_events_detected += 1
            action = str(event["action"])
            actions[action] += 1
            action_frames = int(event.get("action_frames") or 1)
            durations[action_frames] += 1
            attempts[attempt]["decisions"] += 1
            attempts[attempt]["actions"][action] += 1
            attempts[attempt]["durations"][action_frames] += 1
            restored = bool(event.get("restored_archive"))
            if restored:
                attempts[attempt]["restores"] += 1
            frontier_value = event.get("persistent_frontier_value")
            if frontier_value is not None:
                committed_frontier_values.append(float(frontier_value))
            decision_rows.append(
                {
                    "seq": event["seq"],
                    "elapsed_ms": event["elapsed_ms"],
                    "attempt": attempt,
                    "level": _level_for_event(event, annotations) or "",
                    "decision": event["decision"],
                    "action": action,
                    "action_frames": event.get("action_frames"),
                    "path": ",".join(event.get("path", [])),
                    "durations": ",".join(str(value) for value in event.get("durations", [])),
                    "score": event.get("score"),
                    "branches_examined": event.get("branches_examined", 0),
                    "restored_archive": restored,
                    "restore_reason": event.get("restore_reason", ""),
                    "delayed_return_recovery_pending": event.get(
                        "delayed_return_recovery_pending", False
                    ),
                    "persistent_frontier_reward": event.get(
                        "persistent_frontier_reward"
                    ),
                    "persistent_frontier_value": frontier_value,
                    "committed_choice_frontier_value": event.get(
                        "committed_choice_frontier_value"
                    ),
                    "action_effect_contrast": event.get("action_effect_contrast"),
                    "action_effect_value": event.get("action_effect_value"),
                    "action_effect_is_known": event.get(
                        "action_effect_is_known", False
                    ),
                    "action_effect_samples": event.get("action_effect_samples", 0),
                    "action_effect_bonus": event.get("action_effect_bonus"),
                    "causal_spatial_signature": event.get(
                        "causal_spatial_signature"
                    ),
                    "causal_context_signature": event.get(
                        "causal_context_signature"
                    ),
                    "target_causal_context_signature": event.get(
                        "target_causal_context_signature",
                        event.get("causal_context_signature"),
                    ),
                    "causal_event_detected": event.get(
                        "causal_event_detected", False
                    ),
                    "causal_component_count": event.get(
                        "causal_component_count", 0
                    ),
                    "causal_event_basis": event.get("causal_event_basis"),
                    "causal_event_novel_cells": event.get(
                        "causal_event_novel_cells", 0
                    ),
                    "causal_affordance_count": event.get(
                        "causal_affordance_count", 0
                    ),
                    "transition_spatial_signature": event.get(
                        "transition_spatial_signature"
                    ),
                    "causal_spatial_novelty": event.get(
                        "causal_spatial_novelty"
                    ),
                    "causal_spatial_visits_before": event.get(
                        "causal_spatial_visits_before"
                    ),
                    "causal_changed_pixels": event.get(
                        "causal_changed_pixels"
                    ),
                    "causal_change_centroid": json.dumps(
                        event.get("causal_change_centroid")
                    ),
                    "causal_spatial_bonus": event.get(
                        "causal_spatial_bonus"
                    ),
                    "temporal_option_value": event.get("temporal_option_value"),
                    "temporal_option_is_known": event.get(
                        "temporal_option_is_known", False
                    ),
                    "temporal_option_value_source": event.get(
                        "temporal_option_value_source", "unseen"
                    ),
                    "active_temporal_option": event.get(
                        "active_temporal_option", False
                    ),
                    "temporal_option_initiation_eligible": event.get(
                        "temporal_option_initiation_eligible", False
                    ),
                    "temporal_option_counterfactual_contrast": event.get(
                        "temporal_option_counterfactual_contrast"
                    ),
                    "temporal_option_counterfactuals": event.get(
                        "temporal_option_counterfactuals", 0
                    ),
                    "temporal_option_delayed_counterfactual_armed": event.get(
                        "temporal_option_delayed_counterfactual_armed", False
                    ),
                    "human_prior_enabled": event.get(
                        "human_prior_enabled", False
                    ),
                    "human_prior_reward_track": event.get(
                        "human_prior_reward_track"
                    ),
                    "human_prior_goal_phase": event.get(
                        "human_prior_goal_phase"
                    ),
                    "human_prior_remaining_hearts": event.get(
                        "human_prior_remaining_hearts"
                    ),
                    "human_prior_collected_hearts": event.get(
                        "human_prior_collected_hearts", 0
                    ),
                    "human_prior_goal_reward": event.get(
                        "human_prior_goal_reward", 0.0
                    ),
                    "human_prior_milestone_reward": event.get(
                        "human_prior_milestone_reward", 0.0
                    ),
                    "human_prior_navigation_reward": event.get(
                        "human_prior_navigation_reward", 0.0
                    ),
                    "human_prior_life_loss_penalty": event.get(
                        "human_prior_life_loss_penalty", 0.0
                    ),
                    "human_prior_life_loss_confirmed": event.get(
                        "human_prior_life_loss_confirmed", False
                    ),
                    "human_prior_source_player_slot": json.dumps(
                        event.get("human_prior_source_player_slot")
                    ),
                    "human_prior_target_player_slot": json.dumps(
                        event.get("human_prior_target_player_slot")
                    ),
                    "human_prior_target_chest_slot": json.dumps(
                        event.get("human_prior_target_chest_slot")
                    ),
                    "human_prior_target_chest_distance": event.get(
                        "human_prior_target_chest_distance"
                    ),
                    "abstract_signature": event.get("abstract_signature"),
                    "source_behavioral_signature": event.get(
                        "source_behavioral_signature"
                    ),
                    "target_frontier_signature": event.get(
                        "target_frontier_signature"
                    ),
                    "frame": event.get("frame"),
                    "scene_signature": event.get("scene_signature"),
                    "scene_streak": event.get("scene_streak"),
                    "visual_stagnation_streak": event.get(
                        "visual_stagnation_streak"
                    ),
                    "archive_size": event.get("archive_size"),
                    "committed_state_id": event.get("committed_state_id"),
                    "action_counts": json.dumps(event.get("action_counts", {}), sort_keys=True),
                    "duration_counts": json.dumps(
                        event.get("duration_counts", {}), sort_keys=True
                    ),
                    "action_duration_counts": json.dumps(
                        event.get("action_duration_counts", []), sort_keys=True
                    ),
                }
            )
            committed_causal_signature = event.get("causal_spatial_signature")
            if committed_causal_signature:
                committed_causal_spatial_signatures.add(
                    str(committed_causal_signature)
                )

    decision_columns = [
        "seq",
        "elapsed_ms",
        "attempt",
        "level",
        "decision",
        "action",
        "action_frames",
        "path",
        "durations",
        "score",
        "branches_examined",
        "restored_archive",
        "restore_reason",
        "delayed_return_recovery_pending",
        "persistent_frontier_reward",
        "persistent_frontier_value",
        "committed_choice_frontier_value",
        "action_effect_contrast",
        "action_effect_value",
        "action_effect_is_known",
        "action_effect_samples",
        "action_effect_bonus",
        "causal_spatial_signature",
        "causal_context_signature",
        "target_causal_context_signature",
        "causal_event_detected",
        "causal_component_count",
        "causal_event_basis",
        "causal_event_novel_cells",
        "causal_affordance_count",
        "transition_spatial_signature",
        "causal_spatial_novelty",
        "causal_spatial_visits_before",
        "causal_changed_pixels",
        "causal_change_centroid",
        "causal_spatial_bonus",
        "temporal_option_value",
        "temporal_option_is_known",
        "temporal_option_value_source",
        "active_temporal_option",
        "temporal_option_initiation_eligible",
        "temporal_option_counterfactual_contrast",
        "temporal_option_counterfactuals",
        "temporal_option_delayed_counterfactual_armed",
        "human_prior_enabled",
        "human_prior_reward_track",
        "human_prior_goal_phase",
        "human_prior_remaining_hearts",
        "human_prior_collected_hearts",
        "human_prior_goal_reward",
        "human_prior_milestone_reward",
        "human_prior_navigation_reward",
        "human_prior_life_loss_penalty",
        "human_prior_life_loss_confirmed",
        "human_prior_source_player_slot",
        "human_prior_target_player_slot",
        "human_prior_target_chest_slot",
        "human_prior_target_chest_distance",
        "abstract_signature",
        "source_behavioral_signature",
        "target_frontier_signature",
        "frame",
        "scene_signature",
        "scene_streak",
        "visual_stagnation_streak",
        "archive_size",
        "committed_state_id",
        "action_counts",
        "duration_counts",
        "action_duration_counts",
    ]
    with (run_dir / "decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=decision_columns)
        writer.writeheader()
        writer.writerows(decision_rows)

    graph = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "description": "All emulator transitions investigated, including rejected planning branches.",
        "nodes": sorted(node_details.values(), key=lambda item: item["id"]),
        "edges": [
            {
                "source": source,
                "target": target,
                "action": action,
                "action_frames": action_frames,
                "count": count,
            }
            for (source, target, action, action_frames), count in sorted(edge_counts.items())
        ],
    }
    (run_dir / "transitions.json").write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    attempt_rows = {}
    for attempt, values in sorted(attempts.items()):
        attempt_rows[str(attempt)] = {
            "decisions": values["decisions"],
            "restores": values["restores"],
            "env_steps": values["env_steps"],
            "actions": dict(sorted(values["actions"].items())),
            "durations": {
                str(key): value for key, value in sorted(values["durations"].items())
            },
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "status": manifest.get("status"),
        "events": len(events),
        "event_counts": dict(sorted(event_counts.items())),
        "attempt_count": event_counts.get("attempt_started", 0),
        "attempts": attempt_rows,
        "committed_decisions": len(decision_rows),
        "committed_actions": dict(sorted(actions.items())),
        "committed_durations": {str(key): value for key, value in sorted(durations.items())},
        "investigated_actions": dict(sorted(investigated_actions.items())),
        "investigated_durations": {
            str(key): value for key, value in sorted(investigated_durations.items())
        },
        "bootstrap_fixture": (
            manifest.get("metadata", {}).get("bootstrap") or {}
        ).get("fixture"),
        "bootstrap_completed": event_counts.get("bootstrap_completed", 0) > 0,
        "bootstrap_actions": dict(sorted(bootstrap_actions.items())),
        "bootstrap_durations": {
            str(key): value for key, value in sorted(bootstrap_durations.items())
        },
        "bootstrap_frames": bootstrap_frames,
        "archive_restores": event_counts.get("archive_branch_restored", 0),
        "delayed_visual_returns": event_counts.get(
            "delayed_visual_return_detected", 0
        ),
        "delayed_return_recoveries": delayed_return_recoveries,
        "autonomous_dynamics_decisions": event_counts.get(
            "autonomous_dynamics_detected", 0
        ),
        "autonomous_grace_waits": event_counts.get("autonomous_grace_wait", 0),
        "human_prior_chest_completions": event_counts.get(
            "human_prior_chest_completed", 0
        ),
        "human_prior_life_losses": event_counts.get(
            "human_prior_life_loss_confirmed", 0
        ),
        "life_hazard_checkpoints_created": event_counts.get(
            "life_hazard_checkpoint_created", 0
        ),
        "life_hazard_checkpoint_restores": event_counts.get(
            "life_hazard_state_restored", 0
        ),
        "goal_milestone_checkpoints_created": event_counts.get(
            "goal_milestone_checkpoint_created", 0
        ),
        "goal_milestone_checkpoint_restores": sum(
            event["event"] == "life_hazard_state_restored"
            and event.get("checkpoint_kind") == "goal_milestone"
            for event in events
        ),
        "goal_milestone_descendant_invalidations": sum(
            event["event"] == "archive_branch_removed"
            and event.get("reason")
            == "goal_milestone_rollback_descendant"
            for event in events
        ),
        "goal_milestone_descendant_release_failures": sum(
            event["event"] == "archive_branch_release_failed"
            and event.get("reason")
            == "goal_milestone_rollback_descendant"
            for event in events
        ),
        "action_effect_observations": action_effect_observations,
        "action_effect_observations_by_action": dict(
            sorted(action_effect_observations_by_action.items())
        ),
        "action_effect_known_branches": action_effect_known_branches,
        "learned_hazard_filter_events": event_counts.get(
            "learned_hazards_filtered", 0
        ),
        "learned_hazard_filtered_choices": learned_hazard_filtered_choices,
        "archive_branch_rejections": event_counts.get(
            "archive_branch_rejected", 0
        ),
        "archive_rejections_by_reason": dict(
            sorted(archive_rejections_by_reason.items())
        ),
        "archive_hazard_rejections": archive_rejections_by_reason.get(
            "learned_hazard", 0
        ),
        "causal_outcome_archive_additions": event_counts.get(
            "archive_causal_outcome_added", 0
        ),
        "causal_outcome_exhaustions": archive_rejections_by_reason.get(
            "causal_outcome_exhausted", 0
        ),
        "global_action_hazard_samples": global_action_hazard_samples,
        "matched_neutral_verifications": event_counts.get(
            "matched_neutral_verified", 0
        ),
        "causal_spatial_observations": causal_spatial_observations,
        "unique_causal_spatial_signatures": len(causal_spatial_signatures),
        "committed_causal_spatial_signatures": len(
            committed_causal_spatial_signatures
        ),
        "causal_events_detected": causal_events_detected,
        "temporal_options_started": event_counts.get(
            "temporal_option_started", 0
        ),
        "temporal_options_completed": event_counts.get(
            "temporal_option_completed", 0
        ),
        "temporal_options_discarded": event_counts.get(
            "temporal_option_discarded", 0
        ),
        "temporal_option_samples": temporal_option_samples,
        "delayed_temporal_option_samples": delayed_temporal_option_samples,
        "temporal_option_eligible_initiations": sum(
            bool(row["temporal_option_initiation_eligible"])
            for row in decision_rows
        ),
        "learned_temporal_option_choices": len(learned_temporal_option_choices),
        "maximum_temporal_option_value": max(
            learned_temporal_option_values, default=0.0
        ),
        "maximum_temporal_option_endpoint_contrast": max(
            temporal_option_endpoint_contrasts, default=0.0
        ),
        "temporal_option_counterfactuals_armed": event_counts.get(
            "temporal_option_counterfactual_armed", 0
        ),
        "temporal_option_counterfactual_steps": event_counts.get(
            "temporal_option_counterfactual_advanced", 0
        ),
        "temporal_option_counterfactuals_released": event_counts.get(
            "temporal_option_counterfactual_released", 0
        ),
        "persistent_frontier_updates": event_counts.get(
            "persistent_frontier_updated", 0
        ),
        "persistent_frontier_return_events": event_counts.get(
            "persistent_frontier_return_penalized", 0
        ),
        "persistent_frontier_penalized_traces": frontier_penalized_traces,
        "persistent_frontier_choice_samples": frontier_choice_samples,
        "persistent_frontier_trace_restarts": event_counts.get(
            "persistent_frontier_trace_restarted", 0
        ),
        "maximum_committed_frontier_value": max(
            committed_frontier_values, default=0.0
        ),
        "visual_abstraction_assignments": event_counts.get(
            "visual_abstraction_assigned", 0
        ),
        "visual_abstraction_clusters": len(abstraction_clusters),
        "behavioral_abstraction_assignments": event_counts.get(
            "behavioral_abstraction_assigned", 0
        ),
        "behavioral_abstraction_clusters": len(behavior_clusters),
        "behavioral_abstraction_deferrals": event_counts.get(
            "behavioral_abstraction_deferred", 0
        ),
        "frontier_signature_migrations": event_counts.get(
            "frontier_signature_migrated", 0
        ),
        "behavior_probe_selections": event_counts.get(
            "behavior_probe_selected", 0
        ),
        "behavior_probe_selection_reasons": dict(
            sorted(behavior_probe_reasons.items())
        ),
        "behavior_probe_selected_controls": dict(
            sorted(behavior_probe_controls.items())
        ),
        "verified_branches": event_counts.get("branch_verified", 0),
        "unique_frames": len(frames),
        "unique_scenes": len(scenes),
        "states_saved": event_counts.get("state_saved", 0),
        "states_loaded": event_counts.get("state_loaded", 0),
        "states_released": event_counts.get("state_released", 0),
        "annotations": annotations,
        "artifacts": {
            "decisions": "decisions.csv",
            "events": "events.jsonl",
            "frames": "frames/",
            "transitions": "transitions.json",
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize or annotate a Lolo telemetry run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize", help="generate CSV and graph artifacts")
    summarize.add_argument("--run", type=Path, required=True)
    annotate = subparsers.add_parser("annotate-level", help="add an evaluator-only level label")
    annotate.add_argument("--run", type=Path, required=True)
    annotate.add_argument("--label", required=True)
    annotate.add_argument("--start-seq", type=int)
    annotate.add_argument("--end-seq", type=int)
    annotate.add_argument("--attempt", type=int)
    annotate.add_argument("--note")
    args = parser.parse_args()
    if args.command == "annotate-level":
        result = append_level_annotation(
            args.run, args.label, args.start_seq, args.end_seq, args.attempt, args.note
        )
    else:
        result = build_run_summary(args.run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
