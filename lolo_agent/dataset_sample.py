from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_logging import utc_now
from .sequence_store import SequenceStore


def export_group_sample(
    source: Path,
    destination: Path,
    *,
    maximum_groups: int,
    minimum_multistep_groups: int,
    seed: int,
) -> dict:
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    source_store = SequenceStore(source)
    sequences = source_store.load_group_sample(
        maximum_groups,
        seed=seed,
        minimum_multistep_groups=minimum_multistep_groups,
    )
    destination_store = SequenceStore(destination)
    if source_store.reward_track is not None:
        destination_store.bind_reward_track(source_store.reward_track)
    destination_store.append_segment("sample", sequences)
    manifest = {
        "version": 1,
        "created_at": utc_now(),
        "seed": seed,
        "maximum_groups": maximum_groups,
        "minimum_multistep_groups": minimum_multistep_groups,
        "source_statistics": source_store.statistics(),
        "sample_statistics": destination_store.statistics(),
        "source_runs": sorted({item.source_run_id for item in sequences}),
        "groups": len({(item.source_run_id, item.group) for item in sequences}),
    }
    (destination / "sample-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a deterministic, ROM-free sample of pixel sequences"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--maximum-groups", type=int, default=64)
    parser.add_argument("--minimum-multistep-groups", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = export_group_sample(
        args.source,
        args.destination,
        maximum_groups=args.maximum_groups,
        minimum_multistep_groups=args.minimum_multistep_groups,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
