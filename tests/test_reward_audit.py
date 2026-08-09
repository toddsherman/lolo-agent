import json
import tempfile
import unittest
from pathlib import Path

from lolo_agent.reward_audit import candidate_scores, write_audit


class RewardAuditTests(unittest.TestCase):
    def test_counterfactual_scores_exclude_evaluator_semantics(self) -> None:
        event = {
            "combined_score": 10.0,
            "model_score": 2.0,
            "prediction_error": 4.0,
            "causal_event_detected": True,
            "causal_spatial_novelty": 1.0,
            "action_effect_value": 1.0,
            "target_signature_is_new": True,
            "heart_collected": 1000,
        }
        scores = candidate_scores(event, {"prediction_error_weight": 0.5})
        self.assertEqual(scores["no_surprise"], 6.0)
        self.assertAlmostEqual(scores["rule_free_event"], 13.49)

    def test_writes_reproducible_audit_and_matches_committed_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            (run / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "test-run",
                        "metadata": {
                            "planning_config": {"prediction_error_weight": 0.5}
                        },
                    }
                ),
                encoding="utf-8",
            )
            events = [
                {
                    "event": "branch_verified",
                    "seq": 1,
                    "decision": 1,
                    "state_id": "state-a",
                    "action": "a",
                    "action_frames": 1,
                    "combined_score": 2.0,
                    "model_score": 1.5,
                    "prediction_error": 0.8,
                },
                {
                    "event": "branch_verified",
                    "seq": 2,
                    "decision": 1,
                    "state_id": "state-right",
                    "action": "right",
                    "action_frames": 8,
                    "combined_score": 3.0,
                    "causal_event_detected": True,
                    "causal_spatial_novelty": 1.0,
                    "action_effect_value": 1.0,
                    "target_signature_is_new": True,
                },
                {
                    "event": "decision_committed",
                    "seq": 3,
                    "decision": 1,
                    "committed_state_id": "state-right",
                    "restored_archive": False,
                },
                {
                    "event": "evaluator_stable_scene_change",
                    "seq": 4,
                    "decision": 1,
                    "agent_visible": False,
                    "difference_from_initial": 0.2,
                },
            ]
            (run / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            output = root / "audit"
            audit = write_audit([run], output)

            self.assertEqual(audit["aggregate"]["successful_runs"], 1)
            self.assertEqual(
                audit["runs"][0]["strategies"]["current"][
                    "agreement_with_committed"
                ],
                1.0,
            )
            self.assertEqual(
                audit["runs"][0]["strategies"]["current"][
                    "agreement_with_current_argmax"
                ],
                1.0,
            )
            self.assertTrue((output / "report.md").exists())
            self.assertTrue((output / "decision_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
