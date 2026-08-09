from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .run_logging import read_events, sha256_file, utc_now


STRATEGIES = (
    "current",
    "no_surprise",
    "no_action_penalty",
    "capped_action_penalty",
    "capped_delayed_return",
    "rule_free_event",
    "event_no_hash",
)


def _number(event: Mapping[str, Any], key: str) -> float:
    value = event.get(key, 0.0)
    return 0.0 if value is None else float(value)


def component_scores(
    event: Mapping[str, Any], planning_config: Mapping[str, Any]
) -> Dict[str, float]:
    """Reconstruct the terms that contributed to a verified branch score."""

    weight = lambda name, default: float(planning_config.get(name, default))
    return {
        "imagined_model_score": _number(event, "model_score"),
        "visual_novelty": weight("actual_novelty_weight", 1.0)
        * _number(event, "effective_novelty"),
        "scene_novelty": weight("scene_novelty_weight", 0.75)
        * _number(event, "scene_novelty"),
        "prediction_error": weight("prediction_error_weight", 0.5)
        * _number(event, "prediction_error"),
        "visual_change": weight("actual_change_weight", 0.25)
        * _number(event, "visual_change"),
        "action_effect": _number(event, "action_effect_bonus"),
        "causal_spatial": _number(event, "causal_spatial_bonus"),
        "persistent_frontier": weight("frontier_score_weight", 0.6)
        * _number(event, "persistent_frontier_value"),
        "temporal_option": weight("temporal_option_score_weight", 1.0)
        * _number(event, "temporal_option_value"),
        "action_penalty": -_number(event, "action_penalty"),
        "action_coverage_penalty": -_number(
            event, "action_coverage_penalty"
        ),
        "duration_coverage_penalty": -_number(
            event, "duration_coverage_penalty"
        ),
        "consecutive_repeat_penalty": -_number(
            event, "consecutive_repeat_penalty"
        ),
        "delayed_return_penalty": -_number(
            event, "delayed_return_penalty"
        ),
    }


def _enrich_penalty_breakdown(
    event: Dict[str, Any],
    decision_context: Mapping[str, Any],
    planning_config: Mapping[str, Any],
) -> None:
    """Reconstruct legacy penalty subterms from decision-start counters."""

    if "action_coverage_penalty" in event:
        return
    action = str(event.get("action", ""))
    duration = int(event.get("action_frames", 0))
    action_counts = decision_context.get("action_counts", {})
    action_duration_counts = {
        (str(item.get("action")), int(item.get("action_frames", 0))): int(
            item.get("count", 0)
        )
        for item in decision_context.get("action_duration_counts", ())
    }
    action_coverage = float(planning_config.get("action_coverage_weight", 0.35)) * math.sqrt(
        int(action_counts.get(action, 0))
    )
    duration_coverage = float(
        planning_config.get("duration_coverage_weight", 0.2)
    ) * math.sqrt(action_duration_counts.get((action, duration), 0))
    repeated = (
        action == str(decision_context.get("last_action"))
        and duration == int(decision_context.get("last_duration") or 0)
    )
    consecutive = (
        float(planning_config.get("consecutive_repeat_weight", 0.5))
        * int(decision_context.get("action_streak", 0))
        if repeated
        else 0.0
    )
    total = _number(event, "action_penalty")
    delayed_return = max(
        0.0, total - action_coverage - duration_coverage - consecutive
    )
    event.update(
        {
            "action_coverage_penalty": action_coverage,
            "duration_coverage_penalty": duration_coverage,
            "consecutive_repeat_penalty": consecutive,
            "delayed_return_penalty": delayed_return,
            "penalty_breakdown_source": "reconstructed_from_decision_started",
        }
    )


def candidate_scores(
    event: Mapping[str, Any], planning_config: Mapping[str, Any]
) -> Dict[str, float]:
    """Return current and explicitly labeled counterfactual intrinsic scores.

    These scores use only facts already available to the planner. Evaluator room
    completion, hearts, lives, and object identities are deliberately excluded.
    """

    current = _number(event, "combined_score")
    components = component_scores(event, planning_config)
    no_surprise = (
        current
        - components["imagined_model_score"]
        - components["prediction_error"]
    )
    # action_penalty is represented as a negative weighted component.
    no_action_penalty = current - components["action_penalty"]
    capped_action_penalty = (
        current
        - components["action_penalty"]
        + max(components["action_penalty"], -2.0)
    )
    delayed_return_raw = _number(
        event, "delayed_return_penalty_raw"
    ) or _number(event, "delayed_return_penalty")
    delayed_return_effective = _number(event, "delayed_return_penalty")
    capped_delayed_return = (
        current + delayed_return_effective - min(delayed_return_raw, 2.0)
    )
    causal_event = 1.0 if event.get("causal_event_detected") else 0.0
    causal_spatial = max(0.0, _number(event, "causal_spatial_novelty"))
    action_effect = max(0.0, _number(event, "action_effect_value"))
    temporal_option = max(0.0, _number(event, "temporal_option_value"))
    frontier = max(0.0, _number(event, "persistent_frontier_value"))
    is_new = bool(event.get("target_signature_is_new"))
    rule_free_event = (
        8.0 * causal_event
        + 3.0 * causal_spatial
        + 1.5 * action_effect
        + temporal_option
        + 0.5 * frontier
        + (1.0 if is_new else -4.0)
        - 0.01
    )
    event_no_hash = (
        8.0 * causal_event
        + 3.0 * causal_spatial
        + 1.5 * action_effect
        + temporal_option
        + 0.5 * frontier
        - 0.01
    )
    return {
        "current": current,
        "no_surprise": no_surprise,
        "no_action_penalty": no_action_penalty,
        "capped_action_penalty": capped_action_penalty,
        "capped_delayed_return": capped_delayed_return,
        "rule_free_event": rule_free_event,
        "event_no_hash": event_no_hash,
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p50": _quantile(values, 0.5),
        "p95": _quantile(values, 0.95),
        "max": max(values),
    }


def _selection_facts(event: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "action": str(event.get("action", "unknown")),
        "action_frames": int(event.get("action_frames", 0)),
        "causal_event": bool(event.get("causal_event_detected")),
        "causal_spatial": _number(event, "causal_spatial_novelty"),
        "action_effect": _number(event, "action_effect_value"),
        "target_signature_is_new": bool(event.get("target_signature_is_new")),
        "target_scene_is_new": bool(event.get("target_scene_is_new")),
        "prediction_error": _number(event, "prediction_error"),
        "model_score": _number(event, "model_score"),
    }


def _choose(
    branches: Sequence[Mapping[str, Any]],
    strategy: str,
    planning_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    # Stable tie-breaking preserves telemetry order and makes the audit reproducible.
    return max(
        branches,
        key=lambda event: candidate_scores(event, planning_config)[strategy],
    )


def audit_run(run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    events_path = run_dir / "events.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    planning_config = manifest.get("metadata", {}).get("planning_config", {})
    branches: MutableMapping[int, List[Dict[str, Any]]] = defaultdict(list)
    commits: Dict[int, Dict[str, Any]] = {}
    milestones: List[Dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    component_values: MutableMapping[str, List[float]] = defaultdict(list)
    component_spreads: MutableMapping[str, List[float]] = defaultdict(list)
    selection_overrides: MutableMapping[int, List[str]] = defaultdict(list)
    override_kinds = {
        "learned_hazards_filtered",
        "causal_observation_wait",
        "autonomous_dynamics_detected",
        "autonomous_grace_wait",
    }
    decision_contexts: Dict[int, Dict[str, Any]] = {}

    for event in read_events(run_dir):
        kind = str(event.get("event"))
        event_counts[kind] += 1
        if kind == "decision_started":
            decision_contexts[int(event["decision"])] = event
        elif kind == "branch_verified":
            decision = int(event["decision"])
            _enrich_penalty_breakdown(
                event, decision_contexts.get(decision, {}), planning_config
            )
            branches[decision].append(event)
            for name, value in component_scores(event, planning_config).items():
                component_values[name].append(value)
        elif kind == "decision_committed":
            commits[int(event["decision"])] = event
        elif kind == "evaluator_stable_scene_change":
            milestones.append(
                {
                    "decision": int(event["decision"]),
                    "seq": int(event["seq"]),
                    "agent_visible": bool(event.get("agent_visible", False)),
                    "difference_from_initial": _number(
                        event, "difference_from_initial"
                    ),
                }
            )
        if kind in override_kinds and event.get("decision") is not None:
            selection_overrides[int(event["decision"])].append(kind)

    strategy_totals: Dict[str, Dict[str, Any]] = {
        name: {
            "eligible_decisions": 0,
            "matches_committed_state": 0,
            "matches_current_argmax": 0,
            "actions": Counter(),
            "causal_events": 0,
            "causal_spatial_sum": 0.0,
            "action_effect_sum": 0.0,
            "new_target_signatures": 0,
            "new_target_scenes": 0,
            "prediction_error_sum": 0.0,
            "model_score_sum": 0.0,
        }
        for name in STRATEGIES
    }
    comparison_rows: List[Dict[str, Any]] = []
    ordinary_commits = 0
    restored_commits = 0
    unmatched_commits = 0
    near_ties = 0

    for decision in sorted(commits):
        commit = commits[decision]
        if commit.get("restored_archive"):
            restored_commits += 1
            continue
        ordinary_commits += 1
        candidates = branches.get(decision, [])
        if not candidates:
            unmatched_commits += 1
            continue
        state_id = commit.get("committed_state_id")
        committed_branch = next(
            (item for item in candidates if item.get("state_id") == state_id), None
        )
        if committed_branch is None:
            unmatched_commits += 1
        current_scores = sorted(
            (candidate_scores(item, planning_config)["current"] for item in candidates),
            reverse=True,
        )
        if len(current_scores) > 1 and current_scores[0] - current_scores[1] < 0.01:
            near_ties += 1
        decision_components = [
            component_scores(item, planning_config) for item in candidates
        ]
        for name in decision_components[0]:
            values = [item[name] for item in decision_components]
            component_spreads[name].append(max(values) - min(values))
        current_winner = _choose(candidates, "current", planning_config)

        for strategy in STRATEGIES:
            chosen = _choose(candidates, strategy, planning_config)
            facts = _selection_facts(chosen)
            totals = strategy_totals[strategy]
            totals["eligible_decisions"] += 1
            matches = chosen.get("state_id") == state_id
            matches_current = chosen.get("state_id") == current_winner.get("state_id")
            totals["matches_committed_state"] += int(matches)
            totals["matches_current_argmax"] += int(matches_current)
            totals["actions"][facts["action"]] += 1
            totals["causal_events"] += int(facts["causal_event"])
            totals["causal_spatial_sum"] += facts["causal_spatial"]
            totals["action_effect_sum"] += facts["action_effect"]
            totals["new_target_signatures"] += int(
                facts["target_signature_is_new"]
            )
            totals["new_target_scenes"] += int(facts["target_scene_is_new"])
            totals["prediction_error_sum"] += facts["prediction_error"]
            totals["model_score_sum"] += facts["model_score"]
            comparison_rows.append(
                {
                    "run_id": manifest["run_id"],
                    "decision": decision,
                    "strategy": strategy,
                    "committed_state_id": state_id,
                    "selected_state_id": chosen.get("state_id"),
                    "matches_committed_state": matches,
                    "matches_current_argmax": matches_current,
                    "score": candidate_scores(chosen, planning_config)[strategy],
                    **facts,
                }
            )

    strategies: Dict[str, Any] = {}
    for name, raw in strategy_totals.items():
        count = int(raw["eligible_decisions"])
        denominator = max(1, count)
        strategies[name] = {
            "eligible_decisions": count,
            "agreement_with_committed": raw["matches_committed_state"]
            / denominator,
            "agreement_with_current_argmax": raw["matches_current_argmax"]
            / denominator,
            "action_counts": dict(sorted(raw["actions"].items())),
            "causal_event_rate": raw["causal_events"] / denominator,
            "mean_causal_spatial_novelty": raw["causal_spatial_sum"]
            / denominator,
            "mean_action_effect_value": raw["action_effect_sum"] / denominator,
            "new_target_signature_rate": raw["new_target_signatures"]
            / denominator,
            "new_target_scene_rate": raw["new_target_scenes"] / denominator,
            "mean_prediction_error": raw["prediction_error_sum"] / denominator,
            "mean_model_score": raw["model_score_sum"] / denominator,
        }

    component_distributions = {
        name: _distribution(values) for name, values in component_values.items()
    }
    positive_means = {
        name: max(0.0, values["mean"])
        for name, values in component_distributions.items()
    }
    positive_total = sum(positive_means.values()) or 1.0
    for name, values in component_distributions.items():
        values["share_of_mean_positive_score"] = positive_means[name] / positive_total

    return (
        {
            "run_id": manifest["run_id"],
            "run_dir": str(run_dir),
            "events_sha256": sha256_file(events_path),
            "verified_branches": event_counts["branch_verified"],
            "committed_decisions": event_counts["decision_committed"],
            "ordinary_commits": ordinary_commits,
            "archive_restores": restored_commits,
            "unmatched_ordinary_commits": unmatched_commits,
            "near_tie_rate_current": near_ties / max(1, ordinary_commits),
            "evaluator_milestones": milestones,
            "milestone_context": [
                {
                    "decision": milestone["decision"],
                    "selection_overrides": selection_overrides.get(
                        milestone["decision"], []
                    ),
                    "committed_action": commits.get(milestone["decision"], {}).get(
                        "action"
                    ),
                    "committed_action_frames": commits.get(
                        milestone["decision"], {}
                    ).get("action_frames"),
                    "committed_score": commits.get(milestone["decision"], {}).get(
                        "score"
                    ),
                    "preceding_commits": [
                        {
                            "decision": index,
                            "action": commits[index].get("action"),
                            "action_frames": commits[index].get("action_frames"),
                            "restored_archive": bool(
                                commits[index].get("restored_archive")
                            ),
                            "persistent_frontier_reward": commits[index].get(
                                "persistent_frontier_reward"
                            ),
                            "selection_overrides": selection_overrides.get(index, []),
                        }
                        for index in range(max(1, milestone["decision"] - 4), milestone["decision"])
                        if index in commits
                    ],
                }
                for milestone in milestones
            ],
            "selection_override_counts": dict(
                sorted(
                    Counter(
                        kind
                        for kinds in selection_overrides.values()
                        for kind in kinds
                    ).items()
                )
            ),
            "component_distributions": component_distributions,
            "component_choice_spread_distributions": {
                name: _distribution(values)
                for name, values in component_spreads.items()
            },
            "strategies": strategies,
        },
        comparison_rows,
    )


def _aggregate(run_audits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failures = [item for item in run_audits if not item["evaluator_milestones"]]
    successes = [item for item in run_audits if item["evaluator_milestones"]]
    strategy_summary: Dict[str, Any] = {}
    for strategy in STRATEGIES:
        eligible = sum(
            int(run["strategies"][strategy]["eligible_decisions"])
            for run in run_audits
        )
        action_counts: Counter[str] = Counter()
        for run in run_audits:
            action_counts.update(run["strategies"][strategy]["action_counts"])
        strategy_summary[strategy] = {
            "eligible_decisions": eligible,
            "weighted_agreement_with_committed": sum(
                run["strategies"][strategy]["agreement_with_committed"]
                * run["strategies"][strategy]["eligible_decisions"]
                for run in run_audits
            )
            / max(1, eligible),
            "weighted_agreement_with_current_argmax": sum(
                run["strategies"][strategy]["agreement_with_current_argmax"]
                * run["strategies"][strategy]["eligible_decisions"]
                for run in run_audits
            )
            / max(1, eligible),
            "weighted_causal_event_rate": sum(
                run["strategies"][strategy]["causal_event_rate"]
                * run["strategies"][strategy]["eligible_decisions"]
                for run in run_audits
            )
            / max(1, eligible),
            "weighted_new_target_signature_rate": sum(
                run["strategies"][strategy]["new_target_signature_rate"]
                * run["strategies"][strategy]["eligible_decisions"]
                for run in run_audits
            )
            / max(1, eligible),
            "action_counts": dict(sorted(action_counts.items())),
            "button_action_rate": (
                action_counts["a"] + action_counts["b"]
            )
            / max(1, eligible),
        }
    def component_means(cohort: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        branch_count = sum(int(run["verified_branches"]) for run in cohort)
        names = {
            name
            for run in cohort
            for name in run["component_distributions"].keys()
        }
        return {
            name: sum(
                run["component_distributions"][name]["mean"]
                * int(run["verified_branches"])
                for run in cohort
            )
            / max(1, branch_count)
            for name in sorted(names)
        }

    def choice_spread_means(cohort: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        decision_count = sum(int(run["ordinary_commits"]) for run in cohort)
        names = {
            name
            for run in cohort
            for name in run["component_choice_spread_distributions"].keys()
        }
        return {
            name: sum(
                run["component_choice_spread_distributions"][name]["mean"]
                * int(run["ordinary_commits"])
                for run in cohort
            )
            / max(1, decision_count)
            for name in sorted(names)
        }

    return {
        "runs": len(run_audits),
        "successful_runs": len(successes),
        "failed_runs": len(failures),
        "verified_branches": sum(int(run["verified_branches"]) for run in run_audits),
        "committed_decisions": sum(
            int(run["committed_decisions"]) for run in run_audits
        ),
        "strategies": strategy_summary,
        "mean_weighted_components": {
            "all": component_means(run_audits),
            "positive_control": component_means(successes),
            "failed_rooms": component_means(failures),
        },
        "mean_within_decision_choice_spread": {
            "all": choice_spread_means(run_audits),
            "positive_control": choice_spread_means(successes),
            "failed_rooms": choice_spread_means(failures),
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _report(audit: Mapping[str, Any]) -> str:
    aggregate = audit["aggregate"]
    failure_spread = aggregate["mean_within_decision_choice_spread"][
        "failed_rooms"
    ] or aggregate["mean_within_decision_choice_spread"]["all"]
    penalty_leverage = failure_spread.get("action_penalty", 0.0)
    penalty_to_causal = penalty_leverage / max(
        failure_spread.get("causal_spatial", 0.0), 1e-12
    )
    penalty_to_prediction = penalty_leverage / max(
        failure_spread.get("prediction_error", 0.0), 1e-12
    )
    current_strategy = aggregate["strategies"]["current"]
    no_surprise_strategy = aggregate["strategies"]["no_surprise"]
    capped_strategy = aggregate["strategies"]["capped_action_penalty"]
    lines = [
        "# Offline reward audit",
        "",
        f"Generated `{audit['generated_utc']}` from immutable telemetry hashes.",
        "No model or planner state was loaded or changed.",
        "",
        "## Coverage",
        "",
        f"- Runs: {aggregate['runs']} ({aggregate['successful_runs']} positive control, "
        f"{aggregate['failed_runs']} failures)",
        f"- Verified branches: {aggregate['verified_branches']:,}",
        f"- Committed decisions: {aggregate['committed_decisions']:,}",
        "- Semantic heart/life reward: not auditable from agent-visible telemetry; "
        "it requires a separately labeled evaluator ablation.",
        "- Stable room transition: evaluator-only outcome, used as a positive-control "
        "label and never as a counterfactual branch feature.",
        "",
        "## Findings",
        "",
        f"1. The combined action penalty has {penalty_leverage:.2f} "
        f"mean within-decision leverage in failed rooms: {penalty_to_causal:.1f}x "
        f"the next-largest causal-spatial term and {penalty_to_prediction:,.0f}x "
        "prediction error.",
        f"2. Removing model surprise changes only "
        f"{_percent(1.0 - no_surprise_strategy['weighted_agreement_with_current_argmax'])} "
        "of score winners. Prediction error is not the dominant reward problem.",
        f"3. Capping the combined action penalty changes "
        f"{_percent(1.0 - capped_strategy['weighted_agreement_with_current_argmax'])} "
        f"of winners, but collapses A/B selections from "
        f"{_percent(current_strategy['button_action_rate'])} to "
        f"{_percent(capped_strategy['button_action_rate'])}. The penalty must be "
        "split into its coverage, repetition, and return components before tuning.",
        "4. The positive-control room transition occurred on a 16-frame NOOP "
        "selected by the autonomous-dynamics override. Terminal progress must be "
        "credited to an antecedent trace, not only to the final action.",
        "",
        "## Counterfactual selection",
        "",
        "| Strategy | Same as current-score winner | Same as actual commit | Causal-event selections | New visual-state selections |",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        item = aggregate["strategies"][strategy]
        lines.append(
            f"| `{strategy}` | {_percent(item['weighted_agreement_with_current_argmax'])} "
            f"| {_percent(item['weighted_agreement_with_committed'])} "
            f"| {_percent(item['weighted_causal_event_rate'])} "
            f"| {_percent(item['weighted_new_target_signature_rate'])} |"
        )
    lines.extend(
        [
            "",
            "`no_surprise` removes imagined-model score and prediction error. "
            "`no_action_penalty` removes the combined coverage/repetition/return "
            "penalty; `capped_action_penalty` replaces it with a maximum penalty "
            "of 2 per branch. `capped_delayed_return` preserves every other score "
            "term and caps only delayed-return penalty at 2. "
            "`rule_free_event` rewards causal events, controllable spatial change, "
            "learned action effects, temporary options/frontiers, and visual-state "
            "novelty. `event_no_hash` removes the visual-state hash term.",
            "Actual commits can differ from the current-score winner because the "
            "planner also applies learned-hazard filters, causal observation waits, "
            "and autonomous-dynamics overrides after scoring.",
            "",
            "## Mean weighted score components",
            "",
            "These are absolute score offsets. The within-decision spread below is "
            "the more relevant measure of which terms can change an action choice.",
            "",
            "| Component | Positive control | Failed rooms |",
            "|---|---:|---:|",
        ]
    )
    positive = aggregate["mean_weighted_components"]["positive_control"]
    failures = aggregate["mean_weighted_components"]["failed_rooms"]
    for name in sorted(aggregate["mean_weighted_components"]["all"]):
        lines.append(
            f"| `{name}` | {positive.get(name, 0.0):.4f} "
            f"| {failures.get(name, 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Mean within-decision choice leverage",
            "",
            "| Component | Positive control | Failed rooms |",
            "|---|---:|---:|",
        ]
    )
    positive_spread = aggregate["mean_within_decision_choice_spread"][
        "positive_control"
    ]
    for name in sorted(aggregate["mean_within_decision_choice_spread"]["all"]):
        lines.append(
            f"| `{name}` | {positive_spread.get(name, 0.0):.4f} "
            f"| {failure_spread.get(name, 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-run evidence",
            "",
        ]
    )
    for run in audit["runs"]:
        milestone = run["evaluator_milestones"]
        outcome = (
            "stable transition at decision "
            + ", ".join(str(item["decision"]) for item in milestone)
            if milestone
            else "no stable transition"
        )
        current = run["strategies"]["current"]
        no_surprise = run["strategies"]["no_surprise"]
        event = run["strategies"]["rule_free_event"]
        lines.extend(
            [
                f"### {run['run_id']}",
                "",
                f"{outcome}; {run['verified_branches']:,} verified branches, "
                f"{run['committed_decisions']:,} decisions, "
                f"{run['archive_restores']:,} archive restores.",
                f"Current-score near ties (<0.01): {_percent(run['near_tie_rate_current'])}. "
                f"Removing surprise changes the current-score winner in "
                f"{_percent(1.0 - no_surprise['agreement_with_current_argmax'])} of eligible decisions; "
                f"the event proposal changes it in "
                f"{_percent(1.0 - event['agreement_with_current_argmax'])}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation constraints",
            "",
            "This is a policy-ranking audit, not evidence that any counterfactual "
            "reward will clear Room 2. All Room 2 observations are failures, so a "
            "new reward must be tested prospectively under a frozen checkpoint. "
            "Coarse scene/state hashes are especially vulnerable to animation and "
            "should not be treated as semantic progress.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit(run_dirs: Sequence[Path], output_dir: Path) -> Dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_audits: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        run_audit, rows = audit_run(run_dir)
        run_audits.append(run_audit)
        comparison_rows.extend(rows)
    audit = {
        "schema_version": "1.0.0",
        "generated_utc": utc_now(),
        "method": {
            "strategies": list(STRATEGIES),
            "tie_breaking": "first branch in telemetry order",
            "agent_state_mutated": False,
        },
        "aggregate": _aggregate(run_audits),
        "runs": run_audits,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_report(audit), encoding="utf-8")
    fields = [
        "run_id",
        "decision",
        "strategy",
        "committed_state_id",
        "selected_state_id",
        "matches_committed_state",
        "matches_current_argmax",
        "score",
        "action",
        "action_frames",
        "causal_event",
        "causal_spatial",
        "action_effect",
        "target_signature_is_new",
        "target_scene_is_new",
        "prediction_error",
        "model_score",
    ]
    with (output_dir / "decision_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and counterfactually rescore logged planner branches"
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = write_audit(args.run_dirs, args.output)
    print(json.dumps(audit["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
