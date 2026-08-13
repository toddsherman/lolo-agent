from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence

from .entity_behavior import AnonymousEntityBehaviorModel
from .environment import Action
from .run_logging import read_events, sha256_file


def backfill_causal_hazard_provenance(
    checkpoint: Path,
    run_dirs: Sequence[Path],
    output: Path,
) -> Dict[str, Any]:
    """Backfill causal-hazard labels from immutable learning telemetry.

    Only observations already represented by ``evidence_id`` in the input
    checkpoint are accepted.  The operation cannot add types, rules, outcomes,
    or ordinary samples; it only marks which existing terminal labels came
    from localized intervention/control evidence.
    """

    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    resolved_runs = tuple(
        Path(run_dir).expanduser().resolve() for run_dir in run_dirs
    )
    if not resolved_runs:
        raise ValueError("at least one causal evidence run is required")
    model = AnonymousEntityBehaviorModel.load(checkpoint)
    digest_before = model.digest
    causal_before = model.causal_hazard_observation_count
    examined = 0
    accepted = 0
    duplicates = 0
    labels = Counter()
    source_runs = []
    for run_dir in resolved_runs:
        events_path = run_dir / "events.jsonl"
        if not events_path.is_file():
            raise ValueError(f"missing run events: {events_path}")
        source_runs.append(
            {
                "run_dir": str(run_dir),
                "events_sha256": sha256_file(events_path),
            }
        )
        for event in read_events(run_dir):
            if (
                event.get("event")
                != "anonymous_entity_behavior_observed"
                or not event.get("causal_attribution")
                or not event.get("learning_enabled")
                or not event.get("evidence_accepted")
            ):
                continue
            examined += 1
            type_id = event.get("anonymous_type_id")
            evidence_id = str(event.get("evidence_id") or "")
            context_signature = str(
                event.get("context_signature") or ""
            )
            if type_id is None:
                raise ValueError(
                    f"causal evidence has no anonymous type: {evidence_id}"
                )
            added = model.backfill_causal_hazard_evidence(
                int(type_id),
                Action(str(event["action"])),
                int(event["action_frames"]),
                context_signature,
                bool(event.get("observed_hazard")),
                evidence_id,
                autonomous=bool(event.get("autonomous", False)),
            )
            if added:
                accepted += 1
                labels[
                    "hazardous"
                    if event.get("observed_hazard")
                    else "safe"
                ] += 1
            else:
                duplicates += 1
    if accepted == 0 and causal_before == 0:
        raise ValueError("no causal hazard evidence was backfilled")
    model.save(output)
    restored = AnonymousEntityBehaviorModel.load(output)
    if restored.digest != model.digest:
        raise RuntimeError("causal provenance checkpoint round trip failed")
    return {
        "input_checkpoint": str(checkpoint),
        "input_file_sha256": sha256_file(checkpoint),
        "output_checkpoint": str(output),
        "output_file_sha256": sha256_file(output),
        "parameter_sha256_before": digest_before,
        "parameter_sha256_after": model.digest,
        "source_runs": source_runs,
        "eligible_records_examined": examined,
        "causal_records_backfilled": accepted,
        "duplicate_causal_records": duplicates,
        "hazardous_records": labels["hazardous"],
        "safe_records": labels["safe"],
        "causal_hazard_observations_before": causal_before,
        "causal_hazard_observations_after": (
            model.causal_hazard_observation_count
        ),
        "type_count": model.type_count,
        "rule_count": model.rule_count,
        "observations": model.observation_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill causal hazard provenance into an anonymous behavior "
            "checkpoint from immutable learning telemetry"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--run",
        dest="run_dirs",
        type=Path,
        action="append",
        required=True,
        help="learning run containing accepted causal-attribution events",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing output checkpoint",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if output == checkpoint:
        parser.error("--output must differ from --checkpoint")
    if output.exists() and not args.overwrite:
        parser.error("--output already exists; pass --overwrite to replace it")
    report = backfill_causal_hazard_provenance(
        checkpoint,
        args.run_dirs,
        output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
