from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from .partitions import (
    CyclePartitionBinding,
    CyclePartitionContext,
    PartitionUpdateError,
    audit_persistent_artifacts,
    digest_audit_event,
    prepare_cycle_partition,
    verify_frozen_digests,
)
from .run_logging import utc_now


_CYCLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
_DECISIONS = {"continue", "revise", "stop"}


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _required_text_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return tuple(_required_text(item, field) for item in value)


@dataclass(frozen=True)
class CycleBudget:
    max_wall_seconds: float
    max_events: Optional[int]
    hourly_rate_usd: float
    max_cycle_cost_usd: float
    max_campaign_cost_usd: float

    @classmethod
    def from_dict(cls, value: Any) -> "CycleBudget":
        if not isinstance(value, dict):
            raise ValueError("budgets must be an object")
        budget = cls(
            max_wall_seconds=float(value.get("max_wall_seconds", 0)),
            max_events=(
                None
                if value.get("max_events") is None
                else int(value["max_events"])
            ),
            hourly_rate_usd=float(value.get("hourly_rate_usd", 0)),
            max_cycle_cost_usd=float(value.get("max_cycle_cost_usd", 0)),
            max_campaign_cost_usd=float(
                value.get("max_campaign_cost_usd", 0)
            ),
        )
        if budget.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")
        if budget.max_events is not None and budget.max_events <= 0:
            raise ValueError("max_events must be positive when present")
        if budget.hourly_rate_usd < 0:
            raise ValueError("hourly_rate_usd must be non-negative")
        if budget.max_cycle_cost_usd < 0:
            raise ValueError("max_cycle_cost_usd must be non-negative")
        if budget.max_campaign_cost_usd < 0:
            raise ValueError("max_campaign_cost_usd must be non-negative")
        ceiling = budget.max_wall_seconds * budget.hourly_rate_usd / 3600
        if (
            budget.hourly_rate_usd > 0
            and budget.max_cycle_cost_usd <= 0
        ):
            raise ValueError(
                "paid cycles require a positive max_cycle_cost_usd"
            )
        if (
            budget.hourly_rate_usd > 0
            and budget.max_campaign_cost_usd <= 0
        ):
            raise ValueError(
                "paid cycles require a positive max_campaign_cost_usd"
            )
        if ceiling > budget.max_cycle_cost_usd + 1e-9:
            raise ValueError(
                "max_wall_seconds could exceed max_cycle_cost_usd at the "
                "declared hourly rate"
            )
        if (
            budget.max_campaign_cost_usd > 0
            and budget.max_cycle_cost_usd
            > budget.max_campaign_cost_usd + 1e-9
        ):
            raise ValueError(
                "max_cycle_cost_usd cannot exceed max_campaign_cost_usd"
            )
        return budget


@dataclass(frozen=True)
class ResearchPlan:
    cycle_id: str
    hypothesis: str
    decision_question: str
    expected_evidence: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    command: tuple[str, ...]
    working_directory: Path
    telemetry_path: Optional[Path]
    prior_cycle_id: Optional[str]
    budgets: CycleBudget
    evaluation_partition: Optional[CyclePartitionBinding] = None

    @classmethod
    def load(cls, path: Path) -> "ResearchPlan":
        source = Path(path).expanduser().resolve()
        value = json.loads(source.read_text(encoding="utf-8"))
        if value.get("version") != 1:
            raise ValueError("research plan version must be 1")
        cycle_id = _required_text(value.get("cycle_id"), "cycle_id")
        if not _CYCLE_ID.fullmatch(cycle_id):
            raise ValueError("cycle_id contains unsupported characters")
        command = value.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise ValueError("command must be a non-empty string list")
        working_directory = Path(
            value.get("working_directory", source.parent)
        ).expanduser()
        if not working_directory.is_absolute():
            working_directory = source.parent / working_directory
        working_directory = working_directory.resolve()
        telemetry_value = value.get("telemetry_path")
        telemetry_path = None
        if telemetry_value:
            telemetry_path = Path(str(telemetry_value)).expanduser()
            if not telemetry_path.is_absolute():
                telemetry_path = working_directory / telemetry_path
            telemetry_path = telemetry_path.resolve()
        prior = value.get("prior_cycle_id")
        prior_cycle_id = None if prior is None else _required_text(
            prior, "prior_cycle_id"
        )
        partition_value = value.get("evaluation_partition")
        evaluation_partition = (
            None
            if partition_value is None
            else CyclePartitionBinding.from_dict(
                partition_value, source.parent
            )
        )
        return cls(
            cycle_id=cycle_id,
            hypothesis=_required_text(value.get("hypothesis"), "hypothesis"),
            decision_question=_required_text(
                value.get("decision_question"), "decision_question"
            ),
            expected_evidence=_required_text_list(
                value.get("expected_evidence"), "expected_evidence"
            ),
            stop_conditions=_required_text_list(
                value.get("stop_conditions"), "stop_conditions"
            ),
            command=tuple(command),
            working_directory=working_directory,
            telemetry_path=telemetry_path,
            prior_cycle_id=prior_cycle_id,
            budgets=CycleBudget.from_dict(value.get("budgets")),
            evaluation_partition=evaluation_partition,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "cycle_id": self.cycle_id,
            "hypothesis": self.hypothesis,
            "decision_question": self.decision_question,
            "expected_evidence": list(self.expected_evidence),
            "stop_conditions": list(self.stop_conditions),
            "command": list(self.command),
            "working_directory": str(self.working_directory),
            "telemetry_path": (
                None
                if self.telemetry_path is None
                else str(self.telemetry_path)
            ),
            "prior_cycle_id": self.prior_cycle_id,
            "budgets": {
                "max_wall_seconds": self.budgets.max_wall_seconds,
                "max_events": self.budgets.max_events,
                "hourly_rate_usd": self.budgets.hourly_rate_usd,
                "max_cycle_cost_usd": self.budgets.max_cycle_cost_usd,
                "max_campaign_cost_usd": (
                    self.budgets.max_campaign_cost_usd
                ),
            },
            "evaluation_partition": (
                None
                if self.evaluation_partition is None
                else self.evaluation_partition.to_dict()
            ),
        }


class _EventCounter:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self.offset = 0
        self.events = 0

    def update(self) -> int:
        if self.path is None or not self.path.exists():
            return self.events
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.events = 0
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            while chunk := handle.read(1024 * 1024):
                self.events += chunk.count(b"\n")
            self.offset = handle.tell()
        return self.events


def _terminate(process: subprocess.Popen[Any], grace_seconds: float = 5) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def audit_events(path: Optional[Path]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "available": bool(path is not None and path.is_file()),
        "events": 0,
        "invalid_lines": 0,
        "max_seq": None,
        "max_depth": None,
        "verified_branches": 0,
        "unique_frames": 0,
        "collected_heart_transitions": 0,
        "all_hearts_endpoints": 0,
        "chest_successes": 0,
        "room_transitions": 0,
        "directional_world_effects": 0,
        "tracked_world_state_endpoints": 0,
        "life_changes": 0,
        "minimum_heart_distance": None,
        "minimum_chest_distance": None,
        "tracked_loci": [],
        "terminal_event": None,
    }
    if path is None or not path.is_file():
        return metrics
    frames: set[str] = set()
    loci: Counter[tuple[tuple[int, int], ...]] = Counter()
    minimum_heart: Optional[float] = None
    minimum_chest: Optional[float] = None
    max_depth: Optional[int] = None
    max_seq: Optional[int] = None
    terminal_event: Optional[str] = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                metrics["invalid_lines"] += 1
                continue
            if not isinstance(event, dict):
                metrics["invalid_lines"] += 1
                continue
            metrics["events"] += 1
            event_name = str(event.get("event") or "")
            if event_name:
                terminal_event = event_name
            seq = event.get("seq")
            if isinstance(seq, int):
                max_seq = seq if max_seq is None else max(max_seq, seq)
            depth = event.get("depth")
            if isinstance(depth, int):
                max_depth = (
                    depth if max_depth is None else max(max_depth, depth)
                )
            frame = event.get("frame")
            if isinstance(frame, str) and frame:
                frames.add(frame)
            if event_name == "human_prior_option_branch_verified":
                metrics["verified_branches"] += 1
                if int(event.get("human_prior_collected_hearts") or 0) > 0:
                    metrics["collected_heart_transitions"] += 1
                target_hearts = event.get("human_prior_target_hearts")
                if isinstance(target_hearts, list) and not target_hearts:
                    metrics["all_hearts_endpoints"] += 1
                if event.get("human_prior_chest_completed") or event.get(
                    "human_prior_chest_obtained"
                ):
                    metrics["chest_successes"] += 1
                directional = event.get(
                    "human_prior_option_directional_interaction_effect_cells"
                )
                if isinstance(directional, list) and directional:
                    metrics["directional_world_effects"] += 1
                tracked = event.get(
                    "human_prior_option_tracked_world_effect_cells"
                )
                if isinstance(tracked, list) and tracked:
                    normalized = tuple(
                        sorted(
                            (int(cell[0]), int(cell[1]))
                            for cell in tracked
                            if isinstance(cell, list) and len(cell) == 2
                        )
                    )
                    if normalized:
                        metrics["tracked_world_state_endpoints"] += 1
                        loci[normalized] += 1
                if event.get("human_prior_life_counter_changed"):
                    metrics["life_changes"] += 1
                heart_distance = _safe_float(
                    event.get("human_prior_target_heart_distance")
                )
                if heart_distance is not None:
                    minimum_heart = (
                        heart_distance
                        if minimum_heart is None
                        else min(minimum_heart, heart_distance)
                    )
                chest_distance = _safe_float(
                    event.get("human_prior_target_chest_distance")
                )
                if chest_distance is not None:
                    minimum_chest = (
                        chest_distance
                        if minimum_chest is None
                        else min(minimum_chest, chest_distance)
                    )
            if event_name in {
                "evaluator_stable_scene_change",
                "stable_scene_change_detected",
            }:
                metrics["room_transitions"] += 1
    metrics.update(
        {
            "max_seq": max_seq,
            "max_depth": max_depth,
            "unique_frames": len(frames),
            "minimum_heart_distance": minimum_heart,
            "minimum_chest_distance": minimum_chest,
            "tracked_loci": [
                {
                    "cells": [list(cell) for cell in cells],
                    "endpoints": count,
                }
                for cells, count in loci.most_common(12)
            ],
            "terminal_event": terminal_event,
        }
    )
    return metrics


def _report_markdown(report: Dict[str, Any]) -> str:
    telemetry = report["telemetry"]
    budget = report["budget"]
    lines = [
        f"# Research cycle {report['cycle_id']}",
        "",
        f"- Outcome: `{report['outcome']}`",
        f"- Stop reason: `{report['stop_reason']}`",
        f"- Runtime: {report['elapsed_seconds']:.1f} seconds",
        f"- Estimated compute cost: ${budget['actual_cost_usd']:.4f}",
        f"- Campaign spend: ${budget['campaign_spend_usd']:.4f}",
        f"- Events: {telemetry['events']}",
        f"- Verified branches: {telemetry['verified_branches']}",
        f"- Maximum search depth: {telemetry['max_depth']}",
        f"- All-hearts endpoints: {telemetry['all_hearts_endpoints']}",
        f"- Chest successes: {telemetry['chest_successes']}",
        f"- Directional world effects: {telemetry['directional_world_effects']}",
        "",
        "## Reflection gate",
        "",
        "A subsequent cycle is disabled until `reflection.json` records the "
        "evidence, decision, plan changes, and next hypothesis.",
        "",
    ]
    return "\n".join(lines)


def _load_campaign_state(campaign_dir: Path) -> Dict[str, Any]:
    path = campaign_dir / "campaign.json"
    if not path.exists():
        return {
            "version": 1,
            "created_at": utc_now(),
            "last_cycle_id": None,
            "spent_usd": 0.0,
            "max_campaign_cost_usd": None,
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("version") != 1:
        raise ValueError("campaign state version must be 1")
    return value


def _verify_reflection_chain(
    campaign_dir: Path,
    campaign: Dict[str, Any],
    plan: ResearchPlan,
) -> None:
    last_cycle = campaign.get("last_cycle_id")
    if last_cycle is None:
        if plan.prior_cycle_id is not None:
            raise ValueError("the first campaign cycle cannot name a prior cycle")
        return
    if plan.prior_cycle_id != last_cycle:
        raise ValueError(
            f"prior_cycle_id must reference the last cycle: {last_cycle}"
        )
    reflection_path = (
        campaign_dir / "cycles" / str(last_cycle) / "reflection.json"
    )
    if not reflection_path.is_file():
        raise ValueError(
            f"cycle {last_cycle} must be reflected on before another run"
        )
    reflection = json.loads(reflection_path.read_text(encoding="utf-8"))
    if reflection.get("decision") == "stop":
        raise ValueError("the prior reflection stopped this campaign")
    if reflection.get("next_hypothesis") != plan.hypothesis:
        raise ValueError(
            "plan hypothesis must exactly match the prior reflection's "
            "next_hypothesis"
        )


def run_cycle(plan_path: Path, campaign_dir: Path) -> Dict[str, Any]:
    plan = ResearchPlan.load(plan_path)
    campaign_dir = Path(campaign_dir).expanduser().resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    campaign = _load_campaign_state(campaign_dir)
    _verify_reflection_chain(campaign_dir, campaign, plan)
    campaign_ceiling = campaign.get("max_campaign_cost_usd")
    if campaign_ceiling is not None and not math.isclose(
        float(campaign_ceiling),
        plan.budgets.max_campaign_cost_usd,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "max_campaign_cost_usd is immutable after the first cycle"
        )
    cycle_dir = campaign_dir / "cycles" / plan.cycle_id
    if cycle_dir.exists():
        raise FileExistsError(f"cycle already exists: {cycle_dir}")
    if not plan.working_directory.is_dir():
        raise FileNotFoundError(plan.working_directory)
    partition_context: Optional[CyclePartitionContext] = None
    if plan.evaluation_partition is not None:
        partition_context = prepare_cycle_partition(
            plan.evaluation_partition
        )
    projected_cost = (
        plan.budgets.max_wall_seconds
        * plan.budgets.hourly_rate_usd
        / 3600
    )
    spent = float(campaign.get("spent_usd", 0.0))
    if (
        plan.budgets.max_campaign_cost_usd > 0
        and spent + projected_cost
        > plan.budgets.max_campaign_cost_usd + 1e-9
    ):
        raise ValueError("projected cycle cost exceeds campaign budget")
    cycle_dir.mkdir(parents=True)
    _atomic_json(cycle_dir / "plan.json", plan.to_dict())
    _atomic_json(
        cycle_dir / "state.json",
        {
            "version": 1,
            "status": "running",
            "started_at": utc_now(),
            "events_observed": 0,
        },
    )
    started = time.monotonic()
    event_counter = _EventCounter(plan.telemetry_path)
    stop_reason = "process_exit"
    with (cycle_dir / "stdout.log").open("wb") as stdout, (
        cycle_dir / "stderr.log"
    ).open("wb") as stderr:
        process = subprocess.Popen(
            list(plan.command),
            cwd=plan.working_directory,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                events = event_counter.update()
                if elapsed >= plan.budgets.max_wall_seconds:
                    stop_reason = "wall_time_budget"
                    _terminate(process)
                    break
                if (
                    plan.budgets.max_events is not None
                    and events >= plan.budgets.max_events
                ):
                    stop_reason = "event_budget"
                    _terminate(process)
                    break
                time.sleep(0.25)
        except BaseException:
            _terminate(process)
            raise
        return_code = process.wait()
    elapsed = time.monotonic() - started
    events_observed = event_counter.update()
    actual_cost = elapsed * plan.budgets.hourly_rate_usd / 3600
    telemetry = audit_events(plan.telemetry_path)
    success = bool(
        telemetry["chest_successes"] or telemetry["room_transitions"]
    )
    outcome = "success" if success else "awaiting_reflection"
    campaign_spend = spent + actual_cost
    partition_report: Optional[Dict[str, Any]] = None
    partition_violation: Optional[PartitionUpdateError] = None
    if partition_context is not None:
        binding = partition_context.binding
        closing_audit = audit_persistent_artifacts(
            binding.artifact_inventory()
        )
        partition_report = {
            "loaded": partition_context.loaded_event,
            "intent": binding.intent,
            "opening_audit": digest_audit_event(
                partition_context.opening_audit, phase="cycle_start"
            ),
            "closing_audit": digest_audit_event(
                closing_audit, phase="cycle_end"
            ),
        }
        if binding.intent == "frozen_evaluation":
            try:
                verify_frozen_digests(
                    partition_context.opening_audit,
                    closing_audit,
                    partition_context.partition.category,
                )
                partition_report["frozen_digests_verified"] = True
            except PartitionUpdateError as error:
                partition_violation = error
                partition_report["frozen_digests_verified"] = False
                partition_report["violation"] = error.event
    report = {
        "version": 1,
        "cycle_id": plan.cycle_id,
        "hypothesis": plan.hypothesis,
        "decision_question": plan.decision_question,
        "started_at": json.loads(
            (cycle_dir / "state.json").read_text(encoding="utf-8")
        )["started_at"],
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "stop_reason": stop_reason,
        "outcome": outcome,
        "events_observed_by_guard": events_observed,
        "budget": {
            "hourly_rate_usd": plan.budgets.hourly_rate_usd,
            "projected_cycle_cost_usd": projected_cost,
            "actual_cost_usd": actual_cost,
            "max_cycle_cost_usd": plan.budgets.max_cycle_cost_usd,
            "max_campaign_cost_usd": (
                plan.budgets.max_campaign_cost_usd
            ),
            "campaign_spend_usd": campaign_spend,
        },
        "telemetry_path": (
            None
            if plan.telemetry_path is None
            else str(plan.telemetry_path)
        ),
        "telemetry": telemetry,
    }
    if partition_report is not None:
        report["evaluation_partition"] = partition_report
    _atomic_json(cycle_dir / "report.json", report)
    (cycle_dir / "report.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )
    _atomic_json(
        cycle_dir / "state.json",
        {
            "version": 1,
            "status": outcome,
            "completed_at": report["completed_at"],
            "stop_reason": stop_reason,
            "events_observed": events_observed,
        },
    )
    campaign.update(
        {
            "last_cycle_id": plan.cycle_id,
            "spent_usd": campaign_spend,
            "max_campaign_cost_usd": (
                plan.budgets.max_campaign_cost_usd
            ),
            "updated_at": utc_now(),
        }
    )
    _atomic_json(campaign_dir / "campaign.json", campaign)
    if partition_violation is not None:
        raise partition_violation
    return report


def record_reflection(
    campaign_dir: Path,
    cycle_id: str,
    reflection_path: Path,
) -> Dict[str, Any]:
    campaign_dir = Path(campaign_dir).expanduser().resolve()
    if not _CYCLE_ID.fullmatch(cycle_id):
        raise ValueError("cycle_id contains unsupported characters")
    cycle_dir = campaign_dir / "cycles" / cycle_id
    if not (cycle_dir / "report.json").is_file():
        raise FileNotFoundError("cycle report does not exist")
    destination = cycle_dir / "reflection.json"
    if destination.exists():
        raise FileExistsError("reflection is immutable once recorded")
    value = json.loads(
        Path(reflection_path).expanduser().read_text(encoding="utf-8")
    )
    decision = str(value.get("decision") or "").strip()
    if decision not in _DECISIONS:
        raise ValueError(
            "reflection decision must be continue, revise, or stop"
        )
    reflection: Dict[str, Any] = {
        "version": 1,
        "cycle_id": cycle_id,
        "finding_summary": _required_text(
            value.get("finding_summary"), "finding_summary"
        ),
        "evidence": list(
            _required_text_list(value.get("evidence"), "evidence")
        ),
        "decision": decision,
        "plan_changes": list(
            _required_text_list(value.get("plan_changes"), "plan_changes")
        ),
        "next_hypothesis": (
            None
            if decision == "stop"
            else _required_text(
                value.get("next_hypothesis"), "next_hypothesis"
            )
        ),
        "recorded_at": utc_now(),
    }
    _atomic_json(destination, reflection)
    return reflection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cost-gated research cycles with mandatory reflection"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--campaign-dir", type=Path, required=True)
    reflect_parser = subparsers.add_parser("reflect")
    reflect_parser.add_argument("--campaign-dir", type=Path, required=True)
    reflect_parser.add_argument("--cycle-id", required=True)
    reflect_parser.add_argument("--reflection", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "run":
        result = run_cycle(args.plan, args.campaign_dir)
    else:
        result = record_reflection(
            args.campaign_dir, args.cycle_id, args.reflection
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
