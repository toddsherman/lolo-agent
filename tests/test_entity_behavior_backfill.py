from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lolo_agent.entity_behavior import AnonymousEntityBehaviorModel
from lolo_agent.entity_behavior_backfill import (
    backfill_causal_hazard_provenance,
)
from lolo_agent.environment import Action
from lolo_agent.run_logging import RunLogger


class EntityBehaviorBackfillTests(unittest.TestCase):
    def test_backfill_marks_only_existing_checkpoint_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            output = root / "output.json"
            appearance = (1, 4, 1, 5)
            context = "controlled-relative:v1:distance=1"
            model = AnonymousEntityBehaviorModel(
                minimum_prediction_samples=1
            )
            model.observe(
                appearance,
                Action.NOOP,
                224,
                "terminal",
                context_signature=context,
                hazardous=True,
                autonomous=True,
                evidence_id="existing-causal-row",
            )
            model.save(source)
            ordinary_observations = model.observation_count

            logger = RunLogger(root, run_id="causal-learning")
            logger.log(
                "anonymous_entity_behavior_observed",
                causal_attribution=True,
                learning_enabled=True,
                evidence_accepted=True,
                evidence_id="existing-causal-row",
                anonymous_type_id=0,
                action=Action.NOOP,
                action_frames=224,
                context_signature=context,
                observed_hazard=True,
                autonomous=True,
            )
            logger.close()

            report = backfill_causal_hazard_provenance(
                source, (logger.run_dir,), output
            )
            restored = AnonymousEntityBehaviorModel.load(output)
            prediction = restored.predict(
                appearance,
                Action.NOOP,
                224,
                context_signature=context,
                autonomous=True,
            )

        self.assertEqual(report["causal_records_backfilled"], 1)
        self.assertEqual(report["hazardous_records"], 1)
        self.assertEqual(restored.observation_count, ordinary_observations)
        self.assertEqual(restored.type_count, 1)
        self.assertEqual(prediction.causal_hazard_samples, 1)
        self.assertEqual(prediction.causal_hazardous_probability, 1.0)
        self.assertTrue(prediction.causal_hazard_known)


if __name__ == "__main__":
    unittest.main()
