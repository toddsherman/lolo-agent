from __future__ import annotations

import unittest

from lolo_agent.training_benchmark import benchmark_training


class TrainingBenchmarkTests(unittest.TestCase):
    def test_rejects_invalid_budget_before_allocating_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "steps"):
            benchmark_training(
                steps=0,
                warmup_steps=0,
                batch_size=1,
                seed=0,
                learning_rate=3e-4,
                hourly_rate_usd=0,
            )

    def test_rejects_negative_hourly_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "hourly"):
            benchmark_training(
                steps=1,
                warmup_steps=0,
                batch_size=1,
                seed=0,
                learning_rate=3e-4,
                hourly_rate_usd=-1,
            )


if __name__ == "__main__":
    unittest.main()
