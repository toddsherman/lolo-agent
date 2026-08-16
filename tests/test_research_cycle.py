from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

from lolo_agent.research_cycle import (
    CycleBudget,
    record_reflection,
    run_cycle,
)


class ResearchCycleTests(unittest.TestCase):
    def _write_plan(
        self,
        root: Path,
        cycle_id: str,
        hypothesis: str,
        telemetry: Path,
        prior: Optional[str] = None,
    ) -> Path:
        script = "\n".join(
            [
                "import json, time",
                f"f = open({str(telemetry)!r}, 'a', buffering=1)",
                "for i in range(20):",
                "    f.write(json.dumps({'event': 'probe', 'seq': i}) + '\\n')",
                "    time.sleep(.05)",
                "time.sleep(10)",
            ]
        )
        value = {
            "version": 1,
            "cycle_id": cycle_id,
            "hypothesis": hypothesis,
            "decision_question": "Does the bounded probe produce evidence?",
            "expected_evidence": ["At least two telemetry events"],
            "stop_conditions": ["Two events", "Ten seconds"],
            "command": [sys.executable, "-u", "-c", script],
            "working_directory": str(root),
            "telemetry_path": str(telemetry),
            "prior_cycle_id": prior,
            "budgets": {
                "max_wall_seconds": 10,
                "max_events": 2,
                "hourly_rate_usd": 0,
                "max_cycle_cost_usd": 0,
                "max_campaign_cost_usd": 0,
            },
        }
        path = root / f"{cycle_id}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_paid_budget_must_fit_wall_clock_ceiling(self) -> None:
        with self.assertRaisesRegex(ValueError, "could exceed"):
            CycleBudget.from_dict(
                {
                    "max_wall_seconds": 3600,
                    "max_events": 1,
                    "hourly_rate_usd": 1,
                    "max_cycle_cost_usd": 0.50,
                    "max_campaign_cost_usd": 5,
                }
            )

    def test_paid_budget_requires_campaign_ceiling(self) -> None:
        with self.assertRaisesRegex(ValueError, "campaign"):
            CycleBudget.from_dict(
                {
                    "max_wall_seconds": 60,
                    "max_events": 1,
                    "hourly_rate_usd": 1,
                    "max_cycle_cost_usd": 0.02,
                    "max_campaign_cost_usd": 0,
                }
            )

    def test_cycle_stops_and_requires_immutable_reflection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign"
            telemetry = root / "events.jsonl"
            first = self._write_plan(
                root, "cycle-001", "The probe will emit events", telemetry
            )
            report = run_cycle(first, campaign)
            self.assertEqual(report["stop_reason"], "event_budget")
            self.assertGreaterEqual(report["events_observed_by_guard"], 2)
            self.assertEqual(report["outcome"], "awaiting_reflection")

            second = self._write_plan(
                root,
                "cycle-002",
                "A revised probe will emit events",
                root / "events-2.jsonl",
                "cycle-001",
            )
            with self.assertRaisesRegex(ValueError, "reflected"):
                run_cycle(second, campaign)

            reflection_source = root / "reflection-source.json"
            reflection_source.write_text(
                json.dumps(
                    {
                        "finding_summary": "The event guard stopped the run.",
                        "evidence": ["The report contains at least two events."],
                        "decision": "revise",
                        "plan_changes": ["Use a new telemetry file."],
                        "next_hypothesis": "A revised probe will emit events",
                    }
                ),
                encoding="utf-8",
            )
            recorded = record_reflection(
                campaign, "cycle-001", reflection_source
            )
            self.assertEqual(recorded["decision"], "revise")
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                record_reflection(campaign, "cycle-001", reflection_source)

            second_report = run_cycle(second, campaign)
            self.assertEqual(second_report["stop_reason"], "event_budget")

    def test_initial_room_event_is_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            telemetry = root / "initial-room.jsonl"
            telemetry.write_text(
                json.dumps({"event": "pixel_novel_room_started"}) + "\n",
                encoding="utf-8",
            )
            plan = {
                "version": 1,
                "cycle_id": "initial-room",
                "hypothesis": "An initial event is not a solved room",
                "decision_question": "Was the room solved?",
                "expected_evidence": ["No success"],
                "stop_conditions": ["Command exits"],
                "command": [sys.executable, "-c", "pass"],
                "working_directory": str(root),
                "telemetry_path": str(telemetry),
                "prior_cycle_id": None,
                "budgets": {
                    "max_wall_seconds": 2,
                    "max_events": None,
                    "hourly_rate_usd": 0,
                    "max_cycle_cost_usd": 0,
                    "max_campaign_cost_usd": 0,
                },
            }
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            report = run_cycle(plan_path, root / "campaign")
            self.assertEqual(report["outcome"], "awaiting_reflection")


if __name__ == "__main__":
    unittest.main()
