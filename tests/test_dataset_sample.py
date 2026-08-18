from __future__ import annotations

import tempfile
import unittest

try:  # optional ML extra (imported transitively by the modules below)
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise unittest.SkipTest(
        "requires the optional 'ml' extra: pip install -e '.[ml]'"
    ) from exc
from pathlib import Path

from lolo_agent.dataset_sample import export_group_sample


class DatasetSampleTests(unittest.TestCase):
    def test_refuses_to_overwrite_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "existing"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                export_group_sample(
                    root / "source",
                    destination,
                    maximum_groups=2,
                    minimum_multistep_groups=0,
                    seed=0,
                )


if __name__ == "__main__":
    unittest.main()
