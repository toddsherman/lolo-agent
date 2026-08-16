from __future__ import annotations

import tempfile
import unittest
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
