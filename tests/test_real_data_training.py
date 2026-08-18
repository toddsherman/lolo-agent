from __future__ import annotations

import unittest

try:  # optional ML extra (imported transitively by the modules below)
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise unittest.SkipTest(
        "requires the optional 'ml' extra: pip install -e '.[ml]'"
    ) from exc
from pathlib import Path

from lolo_agent.real_data_training import run_real_data_training


class RealDataTrainingTests(unittest.TestCase):
    def test_rejects_invalid_group_budget_before_loading_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum_groups"):
            run_real_data_training(
                dataset=Path("missing"),
                input_checkpoint=Path("missing.pt"),
                output_checkpoint=Path("output.pt"),
                maximum_groups=1,
                minimum_multistep_groups=0,
                epochs=1,
                batch_size=1,
                learning_rate=3e-4,
                seed=0,
                hourly_rate_usd=0,
            )

    def test_rejects_negative_hourly_rate_before_loading_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "hourly"):
            run_real_data_training(
                dataset=Path("missing"),
                input_checkpoint=Path("missing.pt"),
                output_checkpoint=Path("output.pt"),
                maximum_groups=2,
                minimum_multistep_groups=0,
                epochs=1,
                batch_size=1,
                learning_rate=3e-4,
                seed=0,
                hourly_rate_usd=-1,
            )

    def test_rejects_nonpositive_learning_rate_before_loading_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning rate"):
            run_real_data_training(
                dataset=Path("missing"),
                input_checkpoint=Path("missing.pt"),
                output_checkpoint=Path("output.pt"),
                maximum_groups=2,
                minimum_multistep_groups=0,
                epochs=1,
                batch_size=1,
                learning_rate=0,
                seed=0,
                hourly_rate_usd=0,
            )


if __name__ == "__main__":
    unittest.main()
