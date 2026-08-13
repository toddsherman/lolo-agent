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
    committed_behavioral_edges: set[Tuple[str, str, int]] = set()
    human_prior_world_effect_signatures: set[str] = set()
    human_prior_world_contexts: set[str] = set()
    human_prior_graph_states: set[str] = set()
    human_prior_player_positions: set[Tuple[int, int]] = set()
    human_prior_option_world_effect_signatures: set[str] = set()
    anonymous_behavior_rows: List[Dict[str, Any]] = []
    anonymous_behavior_types: set[int] = set()
    anonymous_behavior_outcomes: set[str] = set()
    anonymous_shadow_prediction_rows: List[Dict[str, Any]] = []
    anonymous_shadow_branch_rows: List[Dict[str, Any]] = []
    archive_rejections_by_reason: Counter[str] = Counter()
    spatial_shadow_rows: List[Dict[str, Any]] = []
    returnability_probe_rows: List[Dict[str, Any]] = []
    returnability_probe_summaries: List[Dict[str, Any]] = []
    spatial_shadow_metric_fields = (
        "spatial_shadow_pixel_l1",
        "spatial_shadow_persistence_l1",
        "spatial_shadow_predicted_pixel_change",
        "spatial_shadow_score",
        "spatial_shadow_usefulness_score",
        "spatial_shadow_raw_activity_score",
        "spatial_shadow_predicted_causal_change",
        "spatial_shadow_predicted_causal_effect",
        "spatial_shadow_actual_causal_contrast",
        "spatial_shadow_effect_weighted_pixel_l1",
        "spatial_shadow_effect_weighted_persistence_l1",
        "spatial_shadow_effect_l1",
        "spatial_shadow_effect_f1",
        "spatial_shadow_predicted_effect",
        "spatial_shadow_actual_effect",
        "spatial_shadow_uncertainty",
        "spatial_shadow_predicted_returnability",
        "spatial_shadow_returnability_uncertainty",
    )

    edge_counts: Counter[Tuple[str, str, str, int]] = Counter()
    node_details: Dict[str, Dict[str, Any]] = {}
    for event in events:
        if (
            event["event"]
            == "anonymous_entity_behavior_shadow_prediction"
        ):
            anonymous_shadow_prediction_rows.append(
                {
                    field: event.get(field)
                    for field in (
                        "seq",
                        "elapsed_ms",
                        "attempt",
                        "decision",
                        "branch_id",
                        "candidate_rank",
                        "action",
                        "action_frames",
                        "endpoint_state_id",
                        "horizon_frames",
                        "anchor_cell",
                        "appearance_fingerprint",
                        "appearance_occurrences",
                        "anonymous_type_id",
                        "appearance_distance",
                        "context_signature",
                        "controlled_cell",
                        "context_matched",
                        "predicted_outcome",
                        "predicted_outcome_probability",
                        "hazard_probability",
                        "causal_hazard_probability",
                        "causal_hazard_samples",
                        "causal_hazard_known",
                        "behavior_samples",
                        "behavior_known",
                        "behavior_confidence",
                        "behavior_entropy",
                        "unconditional_predicted_outcome",
                        "unconditional_hazard_probability",
                        "unconditional_behavior_samples",
                        "unconditional_behavior_known",
                        "shadow_hazard_threshold",
                        "shadow_prediction_actionable",
                        "shadow_would_reject",
                        "shadow_policy_authority",
                        "frame",
                    )
                }
            )
        elif (
            event["event"]
            == "anonymous_entity_behavior_shadow_branch_evaluated"
        ):
            anonymous_shadow_branch_rows.append(
                {
                    field: event.get(field)
                    for field in (
                        "seq",
                        "elapsed_ms",
                        "attempt",
                        "decision",
                        "branch_id",
                        "candidate_rank",
                        "action",
                        "action_frames",
                        "endpoint_state_id",
                        "shadow_horizons",
                        "shadow_hazard_threshold",
                        "shadow_policy_authority",
                        "shadow_selection_weight",
                        "shadow_candidate_cells",
                        "shadow_predictions",
                        "shadow_known_predictions",
                        "shadow_contextual_known_predictions",
                        "shadow_causal_known_predictions",
                        "shadow_max_hazard_probability",
                        "shadow_max_empirical_hazard_probability",
                        "shadow_max_unconditional_hazard_probability",
                        "shadow_would_reject",
                        "shadow_implicated_type_id",
                        "shadow_implicated_appearance",
                        "shadow_implicated_anchor",
                        "shadow_implicated_context",
                        "shadow_implicated_horizon",
                        "model_parameter_sha256_before",
                        "model_parameter_sha256_after",
                        "model_parameters_unchanged",
                        "frame",
                    )
                }
            )
        if event["event"] == "anonymous_entity_behavior_observed":
            type_id = event.get("anonymous_type_id")
            if type_id is not None:
                anonymous_behavior_types.add(int(type_id))
            outcome = event.get("observed_outcome")
            if outcome:
                anonymous_behavior_outcomes.add(str(outcome))
            behavior_row = {
                field: event.get(field)
                for field in (
                    "seq",
                    "elapsed_ms",
                    "attempt",
                    "decision",
                    "evidence_id",
                    "learning_enabled",
                    "evidence_accepted",
                    "evidence_eligible",
                    "anonymous_type_id",
                    "anonymous_type_created",
                    "appearance_fingerprint",
                    "appearance_distance",
                    "action",
                    "action_frames",
                    "autonomous",
                    "context_signature",
                    "context_matched_before",
                    "predicted_outcome_before",
                    "predicted_outcome_probability_before",
                    "predicted_outcome_descriptor_before",
                    "semantic_samples_before",
                    "semantic_coverage_before",
                    "inert_probability_before",
                    "inert_confidence_before",
                    "measured_effect_probability_before",
                    "observed_outcome_probability_before",
                    "behavior_samples_before",
                    "behavior_known_before",
                    "behavior_confidence_before",
                    "behavior_entropy_before",
                    "hazard_probability_before",
                    "causal_hazard_probability_before",
                    "causal_hazard_samples_before",
                    "causal_hazard_known_before",
                    "observed_outcome",
                    "observed_outcome_descriptor",
                    "observed_intervention_inert",
                    "observed_controlled_movement",
                    "observed_local_visual_change",
                    "observed_hazard",
                    "surprise",
                    "outcome_matched_prediction",
                    "behavior_samples_after",
                    "behavior_confidence_after",
                    "semantic_samples_after",
                    "semantic_coverage_after",
                    "inert_probability_after",
                    "inert_confidence_after",
                    "measured_effect_probability_after",
                    "hazard_probability_after",
                    "causal_hazard_probability_after",
                    "causal_hazard_samples_after",
                    "causal_hazard_known_after",
                    "anchor_cell",
                    "relative_effect_cells",
                    "player_displacement",
                    "differential_terminal_visual_change",
                    "causal_attribution",
                    "causal_role",
                    "causal_intervention_action",
                    "causal_intervention_frames",
                    "causal_other_branch_hazard",
                    "causal_localization_horizon",
                    "model_type_count",
                    "model_rule_count",
                    "model_observations",
                    "model_causal_hazard_observations",
                    "frame",
                )
            }
            for descriptor_field in (
                "predicted_outcome_descriptor_before",
                "observed_outcome_descriptor",
            ):
                descriptor = behavior_row.get(descriptor_field)
                if descriptor is not None:
                    behavior_row[descriptor_field] = json.dumps(
                        descriptor, sort_keys=True, separators=(",", ":")
                    )
            anonymous_behavior_rows.append(behavior_row)
        if event["event"] == "bidirectional_probe_step":
            returnability_probe_rows.append(
                {
                    "seq": event["seq"],
                    "elapsed_ms": event["elapsed_ms"],
                    "attempt": event.get("attempt", 0),
                    "decision": event.get("decision"),
                    "branch_id": event.get("branch_id"),
                    "candidate_rank": event.get("candidate_rank"),
                    "initial_action": event.get("initial_action"),
                    "initial_action_frames": event.get("initial_action_frames"),
                    "probe_depth": event.get("probe_depth"),
                    "probe_path": json.dumps(event.get("probe_path", [])),
                    "probe_action": event.get("probe_action"),
                    "probe_action_frames": event.get("probe_action_frames"),
                    "total_action_frames": event.get("total_action_frames"),
                    "matched_noop_frame": event.get("matched_noop_frame"),
                    "matched_noop_l1": event.get("matched_noop_l1"),
                    "exact_pixel_return": event.get("exact_pixel_return", False),
                    "return_observed": event.get("return_observed", False),
                    "source_frame": event.get("source_frame"),
                    "endpoint_frame": event.get("endpoint_frame"),
                    "frame": event.get("frame"),
                    "visual_signature": event.get("visual_signature"),
                    "scene_signature": event.get("scene_signature"),
                    "parent_state_id": event.get("parent_state_id"),
                    "child_state_id": event.get("child_state_id"),
                    "env_step_seq": event.get("env_step_seq"),
                    "state_save_seq": event.get("state_save_seq"),
                }
            )
        elif event["event"] == "bidirectional_probe_completed":
            returnability_probe_summaries.append(event)
        if event["event"] == "spatial_shadow_branch_evaluated":
            spatial_shadow_rows.append(
                {
                    "seq": event["seq"],
                    "elapsed_ms": event["elapsed_ms"],
                    "attempt": event.get("attempt", 0),
                    "decision": event.get("decision"),
                    "branch_id": event.get("branch_id"),
                    "candidate_rank": event.get("candidate_rank"),
                    "action": event.get("action"),
                    "action_frames": event.get("action_frames"),
                    "spatial_shadow_mode": event.get("spatial_shadow_mode"),
                    "spatial_shadow_selection_weight": event.get(
                        "spatial_shadow_selection_weight"
                    ),
                    "spatial_shadow_selection_bonus": event.get(
                        "spatial_shadow_selection_bonus", 0.0
                    ),
                    "spatial_shadow_actual_causal_contrast": event.get(
                        "spatial_shadow_actual_causal_contrast"
                    ),
                    "spatial_shadow_beats_persistence": event.get(
                        "spatial_shadow_beats_persistence", False
                    ),
                    **{
                        field: event.get(field)
                        for field in spatial_shadow_metric_fields
                    },
                }
            )
        if event["event"] == "archive_branch_rejected":
            archive_rejections_by_reason[
                str(event.get("reason", "unspecified"))
            ] += 1
        if event["event"] == "human_prior_world_effect_confirmation":
            world_effect = event.get("human_prior_world_effect_signature")
            if world_effect:
                human_prior_world_effect_signatures.add(str(world_effect))
        if event["event"] == "human_prior_option_branch_verified":
            option_world_effect = event.get(
                "human_prior_option_world_effect_signature"
            )
            if option_world_effect:
                human_prior_option_world_effect_signatures.add(
                    str(option_world_effect)
                )
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
            behavioral_source = event.get("source_behavioral_signature")
            if action.lower() != "noop" and behavioral_source:
                committed_behavioral_edges.add(
                    (str(behavioral_source), action, action_frames)
                )
            if restored:
                attempts[attempt]["restores"] += 1
            world_context = event.get("human_prior_world_target_context")
            if world_context:
                human_prior_world_contexts.add(str(world_context))
            graph_state = event.get("human_prior_graph_target_signature")
            if graph_state:
                human_prior_graph_states.add(str(graph_state))
            player_position = event.get("human_prior_target_player_slot")
            if (
                isinstance(player_position, (list, tuple))
                and len(player_position) == 2
            ):
                human_prior_player_positions.add(
                    (int(player_position[0]), int(player_position[1]))
                )
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
                    "spatial_selection_mode": event.get(
                        "spatial_selection_mode"
                    ),
                    "spatial_selection_weight": event.get(
                        "spatial_selection_weight", 0.0
                    ),
                    "spatial_selection_bonus": event.get(
                        "spatial_selection_bonus", 0.0
                    ),
                    "spatial_selection_applied_to_commit": event.get(
                        "spatial_selection_applied_to_commit", False
                    ),
                    "anonymous_entity_hazard_veto_enabled": event.get(
                        "anonymous_entity_hazard_veto_enabled", False
                    ),
                    "anonymous_entity_hazards_detected": event.get(
                        "anonymous_entity_hazards_detected", 0
                    ),
                    "anonymous_entity_hazards_filtered": event.get(
                        "anonymous_entity_hazards_filtered", 0
                    ),
                    "anonymous_entity_hazard_fail_open": event.get(
                        "anonymous_entity_hazard_fail_open", False
                    ),
                    "anonymous_entity_committed_hazard_probability": event.get(
                        "anonymous_entity_committed_hazard_probability", 0.0
                    ),
                    "anonymous_entity_committed_would_reject": event.get(
                        "anonymous_entity_committed_would_reject", False
                    ),
                    "branches_examined": event.get("branches_examined", 0),
                    "parent_state_id": event.get("parent_state_id"),
                    "parent_frame": event.get("parent_frame"),
                    "parent_decision": event.get("parent_decision"),
                    "search_depth": event.get("search_depth", 0),
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
                    "causal_cell_coverage": event.get(
                        "causal_cell_coverage"
                    ),
                    "causal_cell_unvisited": event.get(
                        "causal_cell_unvisited", 0
                    ),
                    "causal_cell_count": event.get(
                        "causal_cell_count", 0
                    ),
                    "causal_cell_coverage_bonus": event.get(
                        "causal_cell_coverage_bonus"
                    ),
                    "causal_cell_recovery_grace_decisions": event.get(
                        "causal_cell_recovery_grace_decisions", 0
                    ),
                    "last_causal_cell_progress_decision": event.get(
                        "last_causal_cell_progress_decision"
                    ),
                    "behavioral_edge_visits_before": event.get(
                        "behavioral_edge_visits_before", 0
                    ),
                    "behavioral_edge_unexpanded": event.get(
                        "behavioral_edge_unexpanded", False
                    ),
                    "behavioral_edge_coverage_bonus": event.get(
                        "behavioral_edge_coverage_bonus", 0.0
                    ),
                    "behavioral_best_first_archive_enabled": event.get(
                        "behavioral_best_first_archive_enabled", False
                    ),
                    "behavioral_best_first_applied": event.get(
                        "behavioral_best_first_applied", False
                    ),
                    "persistent_change_enabled": event.get(
                        "persistent_change_enabled", False
                    ),
                    "persistent_change_stability_decisions": event.get(
                        "persistent_change_stability_decisions", 0
                    ),
                    "persistent_change_minimum_value_drop": event.get(
                        "persistent_change_minimum_value_drop", 0
                    ),
                    "persistent_change_speculative_recovery": event.get(
                        "persistent_change_speculative_recovery", False
                    ),
                    "persistent_change_candidate_count": event.get(
                        "persistent_change_candidate_count", 0
                    ),
                    "speculative_persistence_applied": event.get(
                        "speculative_persistence_applied", False
                    ),
                    "persistent_change_active_count": event.get(
                        "persistent_change_active_count", 0
                    ),
                    "persistent_change_active_cells": json.dumps(
                        event.get("persistent_change_active_cells", []),
                        sort_keys=True,
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
                    "human_prior_navigation_retargeted": event.get(
                        "human_prior_navigation_retargeted", False
                    ),
                    "human_prior_navigation_failed_targets": json.dumps(
                        event.get(
                            "human_prior_navigation_failed_targets", []
                        ),
                        sort_keys=True,
                    ),
                    "human_prior_navigation_active_targets": json.dumps(
                        event.get(
                            "human_prior_navigation_active_targets", []
                        ),
                        sort_keys=True,
                    ),
                    "human_prior_navigation_ordering_source_distance": (
                        event.get(
                            "human_prior_navigation_ordering_source_distance"
                        )
                    ),
                    "human_prior_navigation_ordering_target_distance": (
                        event.get(
                            "human_prior_navigation_ordering_target_distance"
                        )
                    ),
                    "human_prior_navigation_ordering_reward": event.get(
                        "human_prior_navigation_ordering_reward", 0.0
                    ),
                    "human_prior_navigation_reconsidered": event.get(
                        "human_prior_navigation_reconsidered", False
                    ),
                    "human_prior_navigation_reconsidered_targets": json.dumps(
                        event.get(
                            "human_prior_navigation_reconsidered_targets", []
                        ),
                        sort_keys=True,
                    ),
                    "human_prior_navigation_reconsidered_source_distance": (
                        event.get(
                            "human_prior_navigation_reconsidered_source_distance"
                        )
                    ),
                    "human_prior_navigation_reconsidered_target_distance": (
                        event.get(
                            "human_prior_navigation_reconsidered_target_distance"
                        )
                    ),
                    "human_prior_navigation_reconsidered_reward": event.get(
                        "human_prior_navigation_reconsidered_reward", 0.0
                    ),
                    "human_prior_life_loss_penalty": event.get(
                        "human_prior_life_loss_penalty", 0.0
                    ),
                    "human_prior_life_loss_confirmed": event.get(
                        "human_prior_life_loss_confirmed", False
                    ),
                    "human_prior_best_first_applied": event.get(
                        "human_prior_best_first_applied", False
                    ),
                    "human_prior_verified_option": event.get(
                        "human_prior_verified_option", False
                    ),
                    "human_prior_option_depth": event.get(
                        "human_prior_option_depth", 0
                    ),
                    "human_prior_option_path_visits_before": event.get(
                        "human_prior_option_path_visits_before", 0
                    ),
                    "human_prior_option_world_effect_signature": event.get(
                        "human_prior_option_world_effect_signature"
                    ),
                    "human_prior_graph_source_signature": event.get(
                        "human_prior_graph_source_signature"
                    ),
                    "human_prior_graph_target_signature": event.get(
                        "human_prior_graph_target_signature"
                    ),
                    "human_prior_world_source_context": event.get(
                        "human_prior_world_source_context"
                    ),
                    "human_prior_world_target_context": event.get(
                        "human_prior_world_target_context"
                    ),
                    "human_prior_world_effect_signature": event.get(
                        "human_prior_world_effect_signature"
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
        "spatial_selection_mode",
        "spatial_selection_weight",
        "spatial_selection_bonus",
        "spatial_selection_applied_to_commit",
        "anonymous_entity_hazard_veto_enabled",
        "anonymous_entity_hazards_detected",
        "anonymous_entity_hazards_filtered",
        "anonymous_entity_hazard_fail_open",
        "anonymous_entity_committed_hazard_probability",
        "anonymous_entity_committed_would_reject",
        "branches_examined",
        "parent_state_id",
        "parent_frame",
        "parent_decision",
        "search_depth",
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
        "causal_cell_coverage",
        "causal_cell_unvisited",
        "causal_cell_count",
        "causal_cell_coverage_bonus",
        "causal_cell_recovery_grace_decisions",
        "last_causal_cell_progress_decision",
        "behavioral_edge_visits_before",
        "behavioral_edge_unexpanded",
        "behavioral_edge_coverage_bonus",
        "behavioral_best_first_archive_enabled",
        "behavioral_best_first_applied",
        "persistent_change_enabled",
        "persistent_change_stability_decisions",
        "persistent_change_minimum_value_drop",
        "persistent_change_speculative_recovery",
        "persistent_change_candidate_count",
        "speculative_persistence_applied",
        "persistent_change_active_count",
        "persistent_change_active_cells",
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
        "human_prior_navigation_retargeted",
        "human_prior_navigation_failed_targets",
        "human_prior_navigation_active_targets",
        "human_prior_navigation_ordering_source_distance",
        "human_prior_navigation_ordering_target_distance",
        "human_prior_navigation_ordering_reward",
        "human_prior_navigation_reconsidered",
        "human_prior_navigation_reconsidered_targets",
        "human_prior_navigation_reconsidered_source_distance",
        "human_prior_navigation_reconsidered_target_distance",
        "human_prior_navigation_reconsidered_reward",
        "human_prior_life_loss_penalty",
        "human_prior_life_loss_confirmed",
        "human_prior_best_first_applied",
        "human_prior_verified_option",
        "human_prior_option_depth",
        "human_prior_option_path_visits_before",
        "human_prior_option_world_effect_signature",
        "human_prior_graph_source_signature",
        "human_prior_graph_target_signature",
        "human_prior_world_source_context",
        "human_prior_world_target_context",
        "human_prior_world_effect_signature",
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

    spatial_shadow_columns = [
        "seq",
        "elapsed_ms",
        "attempt",
        "decision",
        "branch_id",
        "candidate_rank",
        "action",
        "action_frames",
        "spatial_shadow_mode",
        "spatial_shadow_selection_weight",
        "spatial_shadow_selection_bonus",
        "spatial_shadow_beats_persistence",
        *spatial_shadow_metric_fields,
    ]
    with (run_dir / "spatial_shadow.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=spatial_shadow_columns)
        writer.writeheader()
        writer.writerows(spatial_shadow_rows)

    returnability_probe_columns = [
        "seq",
        "elapsed_ms",
        "attempt",
        "decision",
        "branch_id",
        "candidate_rank",
        "initial_action",
        "initial_action_frames",
        "probe_depth",
        "probe_path",
        "probe_action",
        "probe_action_frames",
        "total_action_frames",
        "matched_noop_frame",
        "matched_noop_l1",
        "exact_pixel_return",
        "return_observed",
        "source_frame",
        "endpoint_frame",
        "frame",
        "visual_signature",
        "scene_signature",
        "parent_state_id",
        "child_state_id",
        "env_step_seq",
        "state_save_seq",
    ]
    with (run_dir / "returnability_probes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=returnability_probe_columns)
        writer.writeheader()
        writer.writerows(returnability_probe_rows)

    anonymous_behavior_columns = [
        "seq",
        "elapsed_ms",
        "attempt",
        "decision",
        "evidence_id",
        "learning_enabled",
        "evidence_accepted",
        "evidence_eligible",
        "anonymous_type_id",
        "anonymous_type_created",
        "appearance_fingerprint",
        "appearance_distance",
        "action",
        "action_frames",
        "autonomous",
        "context_signature",
        "context_matched_before",
        "predicted_outcome_before",
        "predicted_outcome_probability_before",
        "predicted_outcome_descriptor_before",
        "semantic_samples_before",
        "semantic_coverage_before",
        "inert_probability_before",
        "inert_confidence_before",
        "measured_effect_probability_before",
        "observed_outcome_probability_before",
        "behavior_samples_before",
        "behavior_known_before",
        "behavior_confidence_before",
        "behavior_entropy_before",
        "hazard_probability_before",
        "causal_hazard_probability_before",
        "causal_hazard_samples_before",
        "causal_hazard_known_before",
        "observed_outcome",
        "observed_outcome_descriptor",
        "observed_intervention_inert",
        "observed_controlled_movement",
        "observed_local_visual_change",
        "observed_hazard",
        "surprise",
        "outcome_matched_prediction",
        "behavior_samples_after",
        "behavior_confidence_after",
        "semantic_samples_after",
        "semantic_coverage_after",
        "inert_probability_after",
        "inert_confidence_after",
        "measured_effect_probability_after",
        "hazard_probability_after",
        "causal_hazard_probability_after",
        "causal_hazard_samples_after",
        "causal_hazard_known_after",
        "anchor_cell",
        "relative_effect_cells",
        "player_displacement",
        "differential_terminal_visual_change",
        "causal_attribution",
        "causal_role",
        "causal_intervention_action",
        "causal_intervention_frames",
        "causal_other_branch_hazard",
        "causal_localization_horizon",
        "model_type_count",
        "model_rule_count",
        "model_observations",
        "model_causal_hazard_observations",
        "frame",
    ]
    with (run_dir / "entity_behaviors.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=anonymous_behavior_columns
        )
        writer.writeheader()
        writer.writerows(anonymous_behavior_rows)

    anonymous_shadow_prediction_columns = [
        "seq",
        "elapsed_ms",
        "attempt",
        "decision",
        "branch_id",
        "candidate_rank",
        "action",
        "action_frames",
        "endpoint_state_id",
        "horizon_frames",
        "anchor_cell",
        "appearance_fingerprint",
        "appearance_occurrences",
        "anonymous_type_id",
        "appearance_distance",
        "context_signature",
        "controlled_cell",
        "context_matched",
        "predicted_outcome",
        "predicted_outcome_probability",
        "hazard_probability",
        "causal_hazard_probability",
        "causal_hazard_samples",
        "causal_hazard_known",
        "behavior_samples",
        "behavior_known",
        "behavior_confidence",
        "behavior_entropy",
        "unconditional_predicted_outcome",
        "unconditional_hazard_probability",
        "unconditional_behavior_samples",
        "unconditional_behavior_known",
        "shadow_hazard_threshold",
        "shadow_prediction_actionable",
        "shadow_would_reject",
        "shadow_policy_authority",
        "frame",
    ]
    with (run_dir / "entity_behavior_shadow.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=anonymous_shadow_prediction_columns
        )
        writer.writeheader()
        writer.writerows(anonymous_shadow_prediction_rows)

    anonymous_shadow_branch_columns = [
        "seq",
        "elapsed_ms",
        "attempt",
        "decision",
        "branch_id",
        "candidate_rank",
        "action",
        "action_frames",
        "endpoint_state_id",
        "shadow_horizons",
        "shadow_hazard_threshold",
        "shadow_policy_authority",
        "shadow_selection_weight",
        "shadow_candidate_cells",
        "shadow_predictions",
        "shadow_known_predictions",
        "shadow_contextual_known_predictions",
        "shadow_causal_known_predictions",
        "shadow_max_hazard_probability",
        "shadow_max_empirical_hazard_probability",
        "shadow_max_unconditional_hazard_probability",
        "shadow_would_reject",
        "shadow_implicated_type_id",
        "shadow_implicated_appearance",
        "shadow_implicated_anchor",
        "shadow_implicated_context",
        "shadow_implicated_horizon",
        "model_parameter_sha256_before",
        "model_parameter_sha256_after",
        "model_parameters_unchanged",
        "frame",
    ]
    with (run_dir / "entity_behavior_shadow_branches.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=anonymous_shadow_branch_columns
        )
        writer.writeheader()
        writer.writerows(anonymous_shadow_branch_rows)

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

    def spatial_mean(field: str) -> float:
        values = [
            float(row[field])
            for row in spatial_shadow_rows
            if row.get(field) is not None
        ]
        return sum(values) / len(values) if values else 0.0

    shadow_predictions_by_endpoint: Dict[
        Tuple[int, str, int, int], List[Dict[str, Any]]
    ] = defaultdict(list)
    for row in anonymous_shadow_prediction_rows:
        shadow_predictions_by_endpoint[
            (
                int(row.get("decision") or 0),
                str(row.get("action") or ""),
                int(row.get("action_frames") or 0),
                int(row.get("horizon_frames") or 0),
            )
        ].append(row)
    shadow_causal_confusion = Counter()
    shadow_unconditional_matches = 0
    shadow_persistence_matches = 0
    shadow_causal_outcomes_evaluable = 0
    for event in events:
        if event["event"] != "anonymous_entity_causal_contrast_completed":
            continue
        rows = shadow_predictions_by_endpoint.get(
            (
                int(event.get("decision") or 0),
                str(event.get("intervention_action") or ""),
                int(event.get("intervention_frames") or 0),
                int(event.get("wait_frames") or 0),
            ),
            (),
        )
        actionable = [
            row
            for row in rows
            if bool(row.get("shadow_prediction_actionable"))
        ]
        if not actionable:
            continue
        shadow_causal_outcomes_evaluable += 1
        observed_hazard = bool(event.get("factual_hazard"))
        predicted_hazard = any(
            bool(row.get("shadow_would_reject"))
            for row in actionable
        )
        shadow_causal_confusion[
            (
                "true_positive"
                if predicted_hazard and observed_hazard
                else "false_positive"
                if predicted_hazard
                else "false_negative"
                if observed_hazard
                else "true_negative"
            )
        ] += 1
        unconditional_hazard = any(
            bool(row.get("unconditional_behavior_known"))
            and float(
                row.get("unconditional_hazard_probability") or 0.0
            )
            >= float(row.get("shadow_hazard_threshold") or 0.0)
            for row in rows
        )
        shadow_unconditional_matches += int(
            unconditional_hazard == observed_hazard
        )
        # A persistence-only baseline has no mechanism for anticipating a
        # delayed terminal event and therefore always predicts no hazard.
        shadow_persistence_matches += int(not observed_hazard)

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
        "counterfactual_control_probes": event_counts.get(
            "counterfactual_control_probe", 0
        ),
        "counterfactual_control_confirmations": event_counts.get(
            "counterfactual_control_confirmation", 0
        ),
        "counterfactual_control_confirmed_collapses": sum(
            event["event"] == "counterfactual_control_confirmation"
            and bool(event.get("control_collapsed"))
            for event in events
        ),
        "counterfactual_control_returns": sum(
            event["event"] == "counterfactual_control_confirmation"
            and bool(event.get("control_returned"))
            for event in events
        ),
        "counterfactual_novel_scene_transitions": sum(
            event["event"] == "counterfactual_control_confirmation"
            and bool(event.get("novel_scene_observed"))
            for event in events
        ),
        "counterfactual_known_scene_returns": sum(
            event["event"] == "counterfactual_control_confirmation"
            and bool(event.get("returned_to_known_scene"))
            for event in events
        ),
        "counterfactual_control_escape_probes": event_counts.get(
            "counterfactual_control_escape_probe", 0
        ),
        "counterfactual_control_collapses_learned": event_counts.get(
            "counterfactual_control_collapse_learned", 0
        ),
        "control_collapse_checkpoint_restores": event_counts.get(
            "control_collapse_state_restored", 0
        ),
        "control_collapse_descendant_invalidations": sum(
            event["event"] == "archive_branch_removed"
            and event.get("reason")
            == "control_collapse_rollback_descendant"
            for event in events
        ),
        "control_collapse_recovery_duration_probes": sum(
            bool(probe.get("control_collapse_recovery_probe"))
            for event in events
            if event["event"] == "behavior_probe_selected"
            for probe in event.get("probes", [])
        ),
        "matched_causal_observation_probes": sum(
            bool(probe.get("matched_causal_observation"))
            for event in events
            if event["event"] == "behavior_probe_selected"
            for probe in event.get("probes", [])
        ),
        "matched_causal_observation_waits": sum(
            event["event"] == "causal_observation_wait"
            and bool(event.get("duration_matched"))
            for event in events
        ),
        "causal_observation_recovery_suppressions": event_counts.get(
            "causal_observation_recovery_suppressed", 0
        ),
        "causal_observation_interventions_started": event_counts.get(
            "causal_observation_intervention_started", 0
        ),
        "causal_observation_interventions_selected": event_counts.get(
            "causal_observation_intervention_selected", 0
        ),
        "delayed_transition_probes": event_counts.get(
            "delayed_transition_probe", 0
        ),
        "delayed_transition_novel_scenes": sum(
            event["event"] == "delayed_transition_probe"
            and bool(event.get("novel_scene_observed"))
            for event in events
        ),
        "delayed_transition_branches_selected": event_counts.get(
            "delayed_transition_branch_selected", 0
        ),
        "anticipated_transition_observations": event_counts.get(
            "anticipated_transition_observation", 0
        ),
        "anticipated_transition_recovery_suppressions": event_counts.get(
            "anticipated_transition_recovery_suppressed", 0
        ),
        "generic_dark_transitions_started": event_counts.get(
            "generic_dark_transition_started", 0
        ),
        "generic_dark_transitions_resolved": event_counts.get(
            "generic_dark_transition_resolved", 0
        ),
        "pixel_novel_rooms_started": event_counts.get(
            "pixel_novel_room_started", 0
        ),
        "generic_dark_returns_to_known_scene": sum(
            event["event"] == "generic_dark_transition_resolved"
            and bool(event.get("returned_to_known_scene"))
            for event in events
        ),
        "known_scene_return_archive_restores": sum(
            event["event"] == "archive_branch_restored"
            and event.get("reason")
            == "known_scene_return_after_dark_transition"
            for event in events
        ),
        "known_scene_root_checkpoint_restores": event_counts.get(
            "known_scene_recovery_checkpoint_restored", 0
        ),
        "post_dark_archive_branches_filtered": sum(
            int(event.get("filtered_branches", 0))
            for event in events
            if event["event"] == "post_dark_archive_branches_filtered"
        ),
        "episodic_scene_memory_seeds": event_counts.get(
            "episodic_scene_memory_seeded", 0
        ),
        "episodic_human_prior_memory_seeds": event_counts.get(
            "episodic_human_prior_memory_seeded", 0
        ),
        "episodic_human_prior_seeded_graph_states": max(
            (
                int(event.get("graph_states", 0))
                for event in events
                if event["event"]
                == "episodic_human_prior_memory_seeded"
            ),
            default=0,
        ),
        "episodic_human_prior_seeded_player_positions": max(
            (
                int(event.get("player_positions", 0))
                for event in events
                if event["event"]
                == "episodic_human_prior_memory_seeded"
            ),
            default=0,
        ),
        "episodic_human_prior_seeded_option_paths": max(
            (
                int(event.get("verified_option_paths", 0))
                for event in events
                if event["event"]
                == "episodic_human_prior_memory_seeded"
            ),
            default=0,
        ),
        "episodic_human_prior_seeded_temporal_options": max(
            (
                int(event.get("temporal_option_values", 0))
                for event in events
                if event["event"]
                == "episodic_human_prior_memory_seeded"
            ),
            default=0,
        ),
        "option_archive_snapshots_stored": event_counts.get(
            "option_archive_snapshot_stored", 0
        ),
        "goal_milestone_checkpoint_snapshots_stored": event_counts.get(
            "goal_milestone_checkpoint_snapshot_stored", 0
        ),
        "episodic_goal_milestone_checkpoint_state_imports": event_counts.get(
            "episodic_goal_milestone_checkpoint_state_imported", 0
        ),
        "episodic_option_archive_state_imports": event_counts.get(
            "episodic_option_archive_state_imported", 0
        ),
        "episodic_option_archive_seed_events": event_counts.get(
            "episodic_option_archives_seeded", 0
        ),
        "episodic_option_archives_seeded": sum(
            int(event.get("seeded_archives", 0))
            for event in events
            if event["event"] == "episodic_option_archives_seeded"
        ),
        "episodic_option_archives_skipped": sum(
            int(event.get("skipped_milestone_archives", 0))
            for event in events
            if event["event"] == "episodic_option_archives_seeded"
        ),
        "human_prior_chest_completions": event_counts.get(
            "human_prior_chest_completed", 0
        ),
        "human_prior_life_losses": event_counts.get(
            "human_prior_life_loss_confirmed", 0
        ),
        "human_prior_world_effect_confirmations": event_counts.get(
            "human_prior_world_effect_confirmation", 0
        ),
        "human_prior_world_effects_accepted": sum(
            event["event"] == "human_prior_world_effect_confirmation"
            and bool(event.get("accepted"))
            for event in events
        ),
        "human_prior_world_effects_rejected": sum(
            event["event"] == "human_prior_world_effect_confirmation"
            and not bool(event.get("accepted"))
            for event in events
        ),
        "human_prior_unique_world_effect_signatures": len(
            human_prior_world_effect_signatures
        ),
        "human_prior_unique_committed_world_contexts": len(
            human_prior_world_contexts
        ),
        "human_prior_unique_committed_graph_states": len(
            human_prior_graph_states
        ),
        "human_prior_unique_committed_player_positions": len(
            human_prior_player_positions
        ),
        "human_prior_semantic_frontier_overrides": sum(
            event["event"] == "archive_branch_added"
            and bool(event.get("human_prior_semantic_frontier_override"))
            for event in events
        ),
        "human_prior_best_first_filter_events": event_counts.get(
            "human_prior_best_first_archives_filtered", 0
        ),
        "human_prior_best_first_frontier_exhaustions": event_counts.get(
            "human_prior_best_first_frontier_exhausted", 0
        ),
        "human_prior_graph_stagnation_events": event_counts.get(
            "human_prior_graph_stagnation_detected", 0
        ),
        "human_prior_navigation_retargeted_evaluations": sum(
            event["event"]
            in ("branch_verified", "human_prior_option_branch_verified")
            and bool(event.get("human_prior_navigation_retargeted"))
            for event in events
        ),
        "human_prior_navigation_retargeted_option_branches": sum(
            event["event"] == "human_prior_option_branch_verified"
            and bool(event.get("human_prior_navigation_retargeted"))
            for event in events
        ),
        "human_prior_navigation_retargeted_commits": sum(
            event["event"] == "decision_committed"
            and bool(event.get("human_prior_navigation_retargeted"))
            for event in events
        ),
        "human_prior_navigation_ordering_committed_reward_total": sum(
            float(
                event.get(
                    "human_prior_navigation_ordering_reward", 0.0
                )
                or 0.0
            )
            for event in events
            if event["event"] == "decision_committed"
            and bool(event.get("human_prior_navigation_retargeted"))
        ),
        "human_prior_navigation_reconsidered_option_branches": sum(
            event["event"] == "human_prior_option_branch_verified"
            and bool(event.get("human_prior_navigation_reconsidered"))
            for event in events
        ),
        "human_prior_navigation_reconsidered_commits": sum(
            event["event"] == "decision_committed"
            and bool(event.get("human_prior_navigation_reconsidered"))
            for event in events
        ),
        "human_prior_navigation_reconsidered_committed_reward_total": sum(
            float(
                event.get(
                    "human_prior_navigation_reconsidered_reward", 0.0
                )
                or 0.0
            )
            for event in events
            if event["event"] == "decision_committed"
            and bool(event.get("human_prior_navigation_reconsidered"))
        ),
        "human_prior_ordering_progress_hypotheses": event_counts.get(
            "human_prior_ordering_progress_recorded", 0
        ),
        "human_prior_ordering_hypotheses_disproved": event_counts.get(
            "human_prior_ordering_hypothesis_disproved", 0
        ),
        "human_prior_ordering_stale_archives_removed": sum(
            int(event.get("stale_ordering_archives_removed", 0))
            for event in events
            if event["event"]
            == "human_prior_ordering_hypothesis_disproved"
        ),
        "human_prior_ordering_hypotheses_reactivated": sum(
            event["event"] == "goal_milestone_exhaustion_learned"
            and bool(event.get("ordering_hypothesis_reactivated"))
            for event in events
        ),
        "human_prior_option_searches": event_counts.get(
            "human_prior_option_search_started", 0
        ),
        "human_prior_option_search_deferrals": event_counts.get(
            "human_prior_option_search_deferred", 0
        ),
        "human_prior_option_search_skips": event_counts.get(
            "human_prior_option_search_skipped", 0
        ),
        "human_prior_option_search_budget_reopens": event_counts.get(
            "human_prior_option_search_reopened", 0
        ),
        "human_prior_option_cleanup_failures": event_counts.get(
            "human_prior_option_cleanup_failed", 0
        ),
        "human_prior_option_branches_verified": event_counts.get(
            "human_prior_option_branch_verified", 0
        ),
        "human_prior_option_neutral_verifications": event_counts.get(
            "human_prior_option_neutral_verified", 0
        ),
        "human_prior_option_world_effect_observations": sum(
            event["event"] == "human_prior_option_branch_verified"
            and bool(
                event.get(
                    "human_prior_option_world_effect_signature"
                )
            )
            for event in events
        ),
        "human_prior_option_nonlocal_world_effect_observations": sum(
            event["event"] == "human_prior_option_branch_verified"
            and int(
                event.get(
                    "human_prior_option_nonlocal_world_effect_cell_count",
                    0,
                )
            )
            > 0
            for event in events
        ),
        "human_prior_unique_option_world_effect_signatures": len(
            human_prior_option_world_effect_signatures
        ),
        "human_prior_option_world_effect_stability_probes": event_counts.get(
            "human_prior_option_world_effect_stability", 0
        ),
        "human_prior_option_world_effect_stable": sum(
            event["event"]
            == "human_prior_option_world_effect_stability"
            and bool(event.get("stable"))
            for event in events
        ),
        "human_prior_option_world_effect_local_candidates": sum(
            event["event"]
            == "human_prior_option_world_effect_stability"
            and bool(event.get("local_candidate"))
            for event in events
        ),
        "human_prior_option_world_effect_phase_audits": event_counts.get(
            "human_prior_option_world_effect_phase_alignment", 0
        ),
        "human_prior_option_world_effect_phase_equivalent": sum(
            event["event"]
            == "human_prior_option_world_effect_phase_alignment"
            and bool(event.get("phase_equivalent"))
            for event in events
        ),
        "human_prior_option_world_effect_safe": sum(
            event["event"]
            == "human_prior_option_world_effect_stability"
            and bool(event.get("safe"))
            for event in events
        ),
        "human_prior_option_world_effect_action_controls": (
            event_counts.get(
                "human_prior_option_world_effect_action_control", 0
            )
        ),
        "human_prior_option_world_effect_action_controls_confirmed": sum(
            event["event"]
            == "human_prior_option_world_effect_action_control"
            and bool(event.get("confirmed"))
            for event in events
        ),
        "human_prior_option_world_effect_local_controls": sum(
            event["event"]
            == "human_prior_option_world_effect_action_control"
            and event.get("control_mode") == "endpoint_matched_local"
            for event in events
        ),
        "human_prior_option_world_effect_local_controls_confirmed": sum(
            event["event"]
            == "human_prior_option_world_effect_action_control"
            and event.get("control_mode") == "endpoint_matched_local"
            and bool(event.get("confirmed"))
            for event in events
        ),
        "human_prior_option_effect_controllability_probes": (
            event_counts.get(
                "human_prior_option_effect_controllability_probe", 0
            )
        ),
        "human_prior_option_effect_controllability_gains": sum(
            event["event"]
            == "human_prior_option_effect_controllability_probe"
            and bool(event.get("player_footprint_matched"))
            and int(event.get("reachable_player_position_gain", 0)) > 0
            for event in events
        ),
        "human_prior_option_effect_frontier_evaluations": event_counts.get(
            "human_prior_option_effect_frontier_eligible", 0
        ),
        "human_prior_option_effect_frontier_eligible": sum(
            event["event"]
            == "human_prior_option_effect_frontier_eligible"
            and bool(event.get("eligible"))
            for event in events
        ),
        "human_prior_option_effect_frontier_archives": sum(
            event["event"] == "human_prior_option_archive_added"
            and bool(event.get("human_prior_option_effect_frontier"))
            for event in events
        ),
        "human_prior_option_entity_frontier_evaluations": event_counts.get(
            "human_prior_option_entity_frontier_eligible", 0
        ),
        "human_prior_option_entity_frontier_eligible": sum(
            event["event"]
            == "human_prior_option_entity_frontier_eligible"
            and bool(event.get("eligible"))
            for event in events
        ),
        "human_prior_option_entity_frontier_archives": sum(
            event["event"] == "human_prior_option_archive_added"
            and bool(event.get("human_prior_option_entity_frontier"))
            for event in events
        ),
        "human_prior_option_entity_curiosity_branches": sum(
            event["event"] == "human_prior_option_branch_verified"
            and bool(
                event.get(
                    "human_prior_option_entity_interaction_signature"
                )
            )
            for event in events
        ),
        "human_prior_option_entity_curiosity_beam_retained": sum(
            int(
                event.get(
                    "anonymous_entity_curiosity_parents_retained", 0
                )
            )
            for event in events
            if event["event"]
            == "human_prior_option_search_depth_completed"
        ),
        "human_prior_option_entity_curiosity_probes": event_counts.get(
            "human_prior_option_entity_curiosity_probe", 0
        ),
        "human_prior_option_entity_curiosity_known_probes": sum(
            event["event"]
            == "human_prior_option_entity_curiosity_probe"
            and bool(event.get("behavior_known_before"))
            for event in events
        ),
        "human_prior_option_entity_curiosity_transferable_probes": sum(
            event["event"]
            == "human_prior_option_entity_curiosity_probe"
            and event.get("anonymous_type_id") is not None
            for event in events
        ),
        "human_prior_option_entity_curiosity_cell_matches": sum(
            event["event"]
            == "human_prior_option_entity_curiosity_probe"
            and bool(event.get("interaction_cell_matched"))
            for event in events
        ),
        "human_prior_option_entity_curiosity_evidence_withheld": sum(
            event["event"]
            == "human_prior_option_entity_curiosity_probe"
            and event.get("evidence_eligible") is False
            for event in events
        ),
        "human_prior_option_entity_curiosity_evidence_accepted": sum(
            event["event"]
            == "human_prior_option_entity_curiosity_probe"
            and bool(event.get("evidence_accepted"))
            for event in events
        ),
        "human_prior_option_entity_inert_penalized_branches": sum(
            event["event"] == "human_prior_option_branch_verified"
            and float(
                event.get("anonymous_entity_inert_penalty") or 0.0
            )
            > 0.0
            for event in events
        ),
        "human_prior_option_entity_inert_penalty_total": sum(
            float(event.get("anonymous_entity_inert_penalty") or 0.0)
            for event in events
            if event["event"] == "human_prior_option_branch_verified"
        ),
        "human_prior_option_entity_predicted_inert_penalty_total": sum(
            float(
                event.get("anonymous_entity_predicted_inert_penalty")
                or 0.0
            )
            for event in events
            if event["event"] == "human_prior_option_branch_verified"
        ),
        "human_prior_option_entity_inert_penalty_suppressions": sum(
            event["event"] == "human_prior_option_branch_verified"
            and bool(
                event.get("anonymous_entity_inert_penalty_suppressed")
            )
            for event in events
        ),
        "anonymous_entity_behavior_observations": len(
            anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_evidence_accepted": sum(
            bool(row.get("evidence_accepted"))
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_terminal_evidence_withheld": sum(
            row.get("evidence_eligible") is False
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_known_predictions": sum(
            bool(row.get("behavior_known_before"))
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_prediction_matches": sum(
            bool(row.get("behavior_known_before"))
            and bool(row.get("outcome_matched_prediction"))
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_semantic_observations": sum(
            row.get("observed_outcome_descriptor") is not None
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_inert_observations": sum(
            bool(row.get("observed_intervention_inert"))
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_known_semantic_predictions": sum(
            int(row.get("semantic_samples_before") or 0) > 0
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_hazard_observations": sum(
            bool(row.get("observed_hazard"))
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_known_hazard_predictions": sum(
            bool(row.get("behavior_known_before"))
            and float(row.get("hazard_probability_before") or 0.0)
            >= 0.5
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_hazard_classification_matches": sum(
            bool(row.get("behavior_known_before"))
            and (
                float(row.get("hazard_probability_before") or 0.0)
                >= 0.5
            )
            == bool(row.get("observed_hazard"))
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_terminal_observations": sum(
            bool(row.get("differential_terminal_visual_change"))
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_behavior_types_observed": len(
            anonymous_behavior_types
        ),
        "anonymous_entity_behavior_outcomes_observed": len(
            anonymous_behavior_outcomes
        ),
        "anonymous_entity_behavior_checkpoint_updates": event_counts.get(
            "anonymous_entity_behavior_checkpoint_updated", 0
        ),
        "anonymous_entity_behavior_parameter_audits": event_counts.get(
            "anonymous_entity_behavior_parameter_audit", 0
        ),
        "anonymous_entity_passive_scans": event_counts.get(
            "anonymous_entity_passive_scan_completed", 0
        ),
        "anonymous_entity_passive_horizon_branches": event_counts.get(
            "anonymous_entity_passive_horizon_verified", 0
        ),
        "anonymous_entity_causal_horizon_branches": event_counts.get(
            "anonymous_entity_causal_horizon_verified", 0
        ),
        "anonymous_entity_causal_contrasts": event_counts.get(
            "anonymous_entity_causal_contrast_completed", 0
        ),
        "anonymous_entity_causal_hazard_contrasts": sum(
            event["event"]
            == "anonymous_entity_causal_contrast_completed"
            and bool(event.get("hazard_contrast"))
            for event in events
        ),
        "anonymous_entity_causal_candidates_localized": sum(
            int(event.get("newly_localized_candidates") or 0)
            for event in events
            if event["event"]
            == "anonymous_entity_causal_contrast_completed"
        ),
        "anonymous_entity_causal_attributions": sum(
            bool(row.get("causal_attribution"))
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_causal_hazard_attributions": sum(
            bool(row.get("causal_attribution"))
            and bool(row.get("observed_hazard"))
            for row in anonymous_behavior_rows
        ),
        "anonymous_entity_shadow_branch_evaluations": len(
            anonymous_shadow_branch_rows
        ),
        "anonymous_entity_shadow_predictions": len(
            anonymous_shadow_prediction_rows
        ),
        "anonymous_entity_shadow_known_predictions": sum(
            bool(row.get("behavior_known"))
            for row in anonymous_shadow_prediction_rows
        ),
        "anonymous_entity_shadow_contextual_known_predictions": sum(
            bool(row.get("behavior_known"))
            and bool(row.get("context_matched"))
            for row in anonymous_shadow_prediction_rows
        ),
        "anonymous_entity_shadow_causal_known_predictions": sum(
            bool(row.get("shadow_prediction_actionable"))
            for row in anonymous_shadow_prediction_rows
        ),
        "anonymous_entity_shadow_would_reject_branches": sum(
            bool(row.get("shadow_would_reject"))
            for row in anonymous_shadow_branch_rows
        ),
        "anonymous_entity_shadow_policy_authority_branches": sum(
            bool(row.get("shadow_policy_authority"))
            for row in anonymous_shadow_branch_rows
        ),
        "anonymous_entity_shadow_parameter_audits_passed": sum(
            bool(row.get("model_parameters_unchanged"))
            for row in anonymous_shadow_branch_rows
        ),
        "anonymous_entity_shadow_causal_outcomes_evaluable": (
            shadow_causal_outcomes_evaluable
        ),
        "anonymous_entity_shadow_causal_true_positives": (
            shadow_causal_confusion["true_positive"]
        ),
        "anonymous_entity_shadow_causal_false_positives": (
            shadow_causal_confusion["false_positive"]
        ),
        "anonymous_entity_shadow_causal_true_negatives": (
            shadow_causal_confusion["true_negative"]
        ),
        "anonymous_entity_shadow_causal_false_negatives": (
            shadow_causal_confusion["false_negative"]
        ),
        "anonymous_entity_shadow_causal_classification_matches": (
            shadow_causal_confusion["true_positive"]
            + shadow_causal_confusion["true_negative"]
        ),
        "anonymous_entity_shadow_unconditional_causal_matches": (
            shadow_unconditional_matches
        ),
        "anonymous_entity_shadow_persistence_causal_matches": (
            shadow_persistence_matches
        ),
        "anonymous_entity_hazard_veto_evaluations": event_counts.get(
            "anonymous_entity_hazard_veto_evaluated", 0
        ),
        "anonymous_entity_hazard_veto_detections": sum(
            int(event.get("hazards_detected") or 0)
            for event in events
            if event["event"]
            == "anonymous_entity_hazard_veto_evaluated"
        ),
        "anonymous_entity_hazard_veto_filtered": sum(
            int(event.get("hazards_filtered") or 0)
            for event in events
            if event["event"]
            == "anonymous_entity_hazard_veto_evaluated"
        ),
        "anonymous_entity_hazard_veto_fail_opens": sum(
            bool(event.get("fail_open"))
            for event in events
            if event["event"]
            == "anonymous_entity_hazard_veto_evaluated"
        ),
        "human_prior_option_archives_added": event_counts.get(
            "human_prior_option_archive_added", 0
        ),
        "human_prior_options_committed": sum(
            bool(row.get("human_prior_verified_option"))
            for row in decision_rows
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
            event["event"]
            in {
                "life_hazard_state_restored",
                "goal_milestone_exhaustion_state_restored",
            }
            and event.get("checkpoint_kind") == "goal_milestone"
            for event in events
        ),
        "goal_milestone_exhaustions_learned": event_counts.get(
            "goal_milestone_exhaustion_learned", 0
        ),
        "goal_milestone_exhaustion_deferrals": event_counts.get(
            "goal_milestone_exhaustion_deferred", 0
        ),
        "goal_milestone_exhaustion_progress_resets": event_counts.get(
            "goal_milestone_exhaustion_progress_reset", 0
        ),
        "goal_milestone_preparation_transitions": sum(
            event["event"] == "goal_milestone_exhaustion_learned"
            and bool(event.get("preparation_transition_learned"))
            for event in events
        ),
        "goal_milestone_preparation_filter_evaluations": event_counts.get(
            "human_prior_exhausted_milestone_filter_evaluated", 0
        ),
        "goal_milestone_preparation_branches_filtered": sum(
            int(event.get("exhausted_branches_filtered") or 0)
            for event in events
            if event["event"]
            == "human_prior_exhausted_milestone_filter_evaluated"
        ),
        "goal_milestone_preparation_precursors_filtered": sum(
            int(event.get("exhausted_precursor_branches_filtered") or 0)
            for event in events
            if event["event"]
            == "human_prior_exhausted_milestone_filter_evaluated"
        ),
        "goal_milestone_preparation_filter_fail_opens": sum(
            bool(event.get("fail_open"))
            for event in events
            if event["event"]
            == "human_prior_exhausted_milestone_filter_evaluated"
        ),
        "goal_milestone_preparation_archives_preserved": sum(
            int(event.get("preserved_branches") or 0)
            for event in events
            if event["event"]
            == "human_prior_preparation_archives_preserved"
        ),
        "goal_milestone_preparation_archive_filter_events": (
            event_counts.get(
                "human_prior_exhausted_milestone_archives_filtered", 0
            )
        ),
        "goal_milestone_preparation_archives_filtered": sum(
            int(event.get("filtered_branches") or 0)
            for event in events
            if event["event"]
            == "human_prior_exhausted_milestone_archives_filtered"
        ),
        "goal_milestone_preparation_option_endpoints_rejected": (
            event_counts.get(
                "human_prior_option_ordering_endpoint_rejected", 0
            )
        ),
        "goal_milestone_exhaustion_hazard_samples": sum(
            event["event"] == "goal_milestone_exhaustion_learned"
            and bool(event.get("hazard_evidence", True))
            for event in events
        ),
        "goal_milestone_frontier_budget_exhaustions": event_counts.get(
            "goal_milestone_frontier_budget_exhausted", 0
        ),
        "goal_milestone_exhaustion_restores": event_counts.get(
            "goal_milestone_exhaustion_state_restored", 0
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
        "causal_cells_first_visited": sum(
            int(row.get("causal_cell_unvisited") or 0)
            for row in decision_rows
        ),
        "causal_cell_coverage_bonus_total": sum(
            float(row.get("causal_cell_coverage_bonus") or 0.0)
            for row in decision_rows
        ),
        "causal_cell_coverage_mean": (
            sum(
                float(row["causal_cell_coverage"])
                for row in decision_rows
                if row.get("causal_cell_coverage") is not None
            )
            / sum(
                row.get("causal_cell_coverage") is not None
                for row in decision_rows
            )
            if any(
                row.get("causal_cell_coverage") is not None
                for row in decision_rows
            )
            else 0.0
        ),
        "persistent_change_updates": event_counts.get(
            "persistent_change_evidence_updated", 0
        ),
        "persistent_change_activations": sum(
            len(event.get("activated", []))
            for event in events
            if event["event"] == "persistent_change_evidence_updated"
        ),
        "persistent_change_retirements": sum(
            len(event.get("retired", []))
            for event in events
            if event["event"] == "persistent_change_evidence_updated"
        ),
        "persistent_change_baseline_adaptations": sum(
            len(event.get("baseline_adapted", []))
            for event in events
            if event["event"] == "persistent_change_evidence_updated"
        ),
        "persistent_change_archive_filter_events": event_counts.get(
            "persistent_change_archives_filtered", 0
        ),
        "persistent_change_archive_branches_filtered": sum(
            int(event.get("filtered_branches", 0))
            for event in events
            if event["event"] == "persistent_change_archives_filtered"
        ),
        "persistent_change_preservation_unavailable": event_counts.get(
            "persistent_change_preservation_unavailable", 0
        ),
        "persistent_change_max_active_cells": max(
            (
                int(event.get("persistent_change_active_count", 0))
                for event in events
            ),
            default=0,
        ),
        "causal_events_detected": causal_events_detected,
        "returnability_probe_branches": len(returnability_probe_summaries),
        "returnability_probe_paths": len(returnability_probe_rows),
        "returnability_probe_returning_paths": sum(
            bool(row.get("return_observed")) for row in returnability_probe_rows
        ),
        "returnability_probe_branches_with_return": sum(
            bool(event.get("return_observed"))
            for event in returnability_probe_summaries
        ),
        "returnability_probe_no_return_within_budget": sum(
            bool(event.get("no_return_within_probe_budget"))
            for event in returnability_probe_summaries
        ),
        "returnability_probe_mean_best_matched_noop_l1": (
            sum(
                float(event.get("best_matched_noop_l1", 0.0))
                for event in returnability_probe_summaries
            )
            / len(returnability_probe_summaries)
            if returnability_probe_summaries
            else 0.0
        ),
        "spatial_shadow_evaluations": len(spatial_shadow_rows),
        "spatial_selection_enabled": any(
            float(row.get("spatial_shadow_selection_weight") or 0.0) > 0.0
            for row in spatial_shadow_rows
        ),
        "spatial_selection_weight": max(
            (
                float(row.get("spatial_shadow_selection_weight") or 0.0)
                for row in spatial_shadow_rows
            ),
            default=0.0,
        ),
        "spatial_mean_selection_bonus": (
            sum(
                float(row.get("spatial_shadow_selection_bonus") or 0.0)
                for row in spatial_shadow_rows
            )
            / len(spatial_shadow_rows)
            if spatial_shadow_rows
            else 0.0
        ),
        "spatial_shadow_beats_persistence": sum(
            bool(row["spatial_shadow_beats_persistence"])
            for row in spatial_shadow_rows
        ),
        "spatial_shadow_parameter_audit_passed": any(
            event["event"] == "spatial_shadow_parameter_audit"
            and event.get("status") == "pass"
            and event.get("parameter_sha256_before")
            == event.get("parameter_sha256_after")
            for event in events
        ),
        "spatial_returnability_parameter_audit_passed": any(
            event["event"] == "spatial_returnability_parameter_audit"
            and event.get("status") == "pass"
            and event.get("parameter_sha256_before")
            == event.get("parameter_sha256_after")
            for event in events
        ),
        "spatial_shadow_mean_metrics": {
            field: spatial_mean(field)
            for field in spatial_shadow_metric_fields
        },
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
        "unique_committed_behavioral_edges": len(
            committed_behavioral_edges
        ),
        "behavioral_best_first_filter_events": event_counts.get(
            "behavioral_best_first_archives_filtered", 0
        ),
        "behavioral_best_first_frontier_exhaustions": event_counts.get(
            "behavioral_best_first_frontier_exhausted", 0
        ),
        "persistent_change_candidate_filter_events": event_counts.get(
            "persistent_change_candidate_archives_filtered", 0
        ),
        "persistent_change_candidate_preservation_unavailable": (
            event_counts.get(
                "persistent_change_candidate_preservation_unavailable", 0
            )
        ),
        "causal_cell_recovery_suppressions": event_counts.get(
            "causal_cell_recovery_suppressed", 0
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
            "spatial_shadow": "spatial_shadow.csv",
            "returnability_probes": "returnability_probes.csv",
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
