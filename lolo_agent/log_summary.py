from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .run_logging import SCHEMA_VERSION, read_events, utc_now


def read_annotations(run_dir: Path) -> Iterable[Dict[str, Any]]:
    path = Path(run_dir) / "evaluator_annotations.jsonl"
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


def append_level_annotation(
    run_dir: Path,
    label: str,
    start_seq: Optional[int] = None,
    end_seq: Optional[int] = None,
    attempt: Optional[int] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Add evaluator-only metadata after a run; it is never visible to the agent."""

    run_dir = Path(run_dir).expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    annotation = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "time_utc": utc_now(),
        "kind": "level",
        "label": label,
        "start_seq": start_seq,
        "end_seq": end_seq,
        "attempt": attempt,
        "note": note,
        "source": "evaluator",
    }
    with (run_dir / "evaluator_annotations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(annotation, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()
    return annotation


def _level_for_event(event: Dict[str, Any], annotations: List[Dict[str, Any]]) -> Optional[str]:
    seq = int(event["seq"])
    attempt = int(event.get("attempt", 0))
    matches = []
    for annotation in annotations:
        if annotation.get("kind") != "level":
            continue
        annotated_attempt = annotation.get("attempt")
        if annotated_attempt is not None and int(annotated_attempt) != attempt:
            continue
        start = annotation.get("start_seq")
        end = annotation.get("end_seq")
        if start is not None and seq < int(start):
            continue
        if end is not None and seq > int(end):
            continue
        matches.append(annotation)
    return None if not matches else str(matches[-1]["label"])


def build_run_summary(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    events = list(read_events(run_dir))
    annotations = list(read_annotations(run_dir))
    event_counts = Counter(event["event"] for event in events)
    actions = Counter()
    durations = Counter()
    investigated_actions = Counter()
    investigated_durations = Counter()
    frames: set[str] = set()
    scenes: set[str] = set()
    attempts: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "decisions": 0,
            "restores": 0,
            "actions": Counter(),
            "durations": Counter(),
            "env_steps": 0,
        }
    )
    decision_rows: List[Dict[str, Any]] = []
    delayed_return_recoveries = 0
    frontier_penalized_traces = 0
    frontier_choice_samples = 0
    committed_frontier_values: List[float] = []
    abstraction_clusters: set[str] = set()

    edge_counts: Counter[Tuple[str, str, str, int]] = Counter()
    node_details: Dict[str, Dict[str, Any]] = {}
    for event in events:
        if (
            event["event"] == "archive_branch_restored"
            and event.get("reason") == "delayed_visual_return"
        ):
            delayed_return_recoveries += 1
        if event["event"] == "persistent_frontier_return_penalized":
            penalized = event.get("penalized_traces", ())
            frontier_penalized_traces += len(penalized)
            frontier_choice_samples += sum(
                item.get("choice") is not None for item in penalized
            )
        elif event["event"] == "persistent_frontier_updated":
            frontier_choice_samples += sum(
                item.get("choice") is not None
                for item in event.get("completed_samples", ())
            )
        if event["event"] == "visual_abstraction_assigned":
            abstraction_clusters.add(str(event["cluster"]))
        frame = event.get("frame") or event.get("target_frame")
        if frame:
            frames.add(frame)
            node_details.setdefault(
                frame,
                {
                    "id": frame,
                    "image": f"frames/{frame}.png",
                    "scene_signature": event.get("scene_signature"),
                    "visual_signature": event.get("visual_signature"),
                },
            )
        if event.get("scene_signature"):
            scenes.add(event["scene_signature"])
        attempt = int(event.get("attempt", 0))
        if event["event"] == "env_step":
            action = str(event["action"])
            investigated_actions[action] += 1
            investigated_durations[int(event["action_frames"])] += 1
            attempts[attempt]["env_steps"] += 1
            source = event.get("source_frame")
            target = event.get("target_frame")
            if source and target:
                edge_counts[(source, target, action, int(event["action_frames"]))] += 1
        elif event["event"] == "decision_committed":
            action = str(event["action"])
            actions[action] += 1
            action_frames = int(event.get("action_frames") or 1)
            durations[action_frames] += 1
            attempts[attempt]["decisions"] += 1
            attempts[attempt]["actions"][action] += 1
            attempts[attempt]["durations"][action_frames] += 1
            restored = bool(event.get("restored_archive"))
            if restored:
                attempts[attempt]["restores"] += 1
            frontier_value = event.get("persistent_frontier_value")
            if frontier_value is not None:
                committed_frontier_values.append(float(frontier_value))
            decision_rows.append(
                {
                    "seq": event["seq"],
                    "elapsed_ms": event["elapsed_ms"],
                    "attempt": attempt,
                    "level": _level_for_event(event, annotations) or "",
                    "decision": event["decision"],
                    "action": action,
                    "action_frames": event.get("action_frames"),
                    "path": ",".join(event.get("path", [])),
                    "durations": ",".join(str(value) for value in event.get("durations", [])),
                    "score": event.get("score"),
                    "branches_examined": event.get("branches_examined", 0),
                    "restored_archive": restored,
                    "restore_reason": event.get("restore_reason", ""),
                    "delayed_return_recovery_pending": event.get(
                        "delayed_return_recovery_pending", False
                    ),
                    "persistent_frontier_reward": event.get(
                        "persistent_frontier_reward"
                    ),
                    "persistent_frontier_value": frontier_value,
                    "committed_choice_frontier_value": event.get(
                        "committed_choice_frontier_value"
                    ),
                    "abstract_signature": event.get("abstract_signature"),
                    "frame": event.get("frame"),
                    "scene_signature": event.get("scene_signature"),
                    "scene_streak": event.get("scene_streak"),
                    "visual_stagnation_streak": event.get(
                        "visual_stagnation_streak"
                    ),
                    "archive_size": event.get("archive_size"),
                    "committed_state_id": event.get("committed_state_id"),
                    "action_counts": json.dumps(event.get("action_counts", {}), sort_keys=True),
                }
            )

    decision_columns = [
        "seq",
        "elapsed_ms",
        "attempt",
        "level",
        "decision",
        "action",
        "action_frames",
        "path",
        "durations",
        "score",
        "branches_examined",
        "restored_archive",
        "restore_reason",
        "delayed_return_recovery_pending",
        "persistent_frontier_reward",
        "persistent_frontier_value",
        "committed_choice_frontier_value",
        "abstract_signature",
        "frame",
        "scene_signature",
        "scene_streak",
        "visual_stagnation_streak",
        "archive_size",
        "committed_state_id",
        "action_counts",
    ]
    with (run_dir / "decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=decision_columns)
        writer.writeheader()
        writer.writerows(decision_rows)

    graph = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "description": "All emulator transitions investigated, including rejected planning branches.",
        "nodes": sorted(node_details.values(), key=lambda item: item["id"]),
        "edges": [
            {
                "source": source,
                "target": target,
                "action": action,
                "action_frames": action_frames,
                "count": count,
            }
            for (source, target, action, action_frames), count in sorted(edge_counts.items())
        ],
    }
    (run_dir / "transitions.json").write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    attempt_rows = {}
    for attempt, values in sorted(attempts.items()):
        attempt_rows[str(attempt)] = {
            "decisions": values["decisions"],
            "restores": values["restores"],
            "env_steps": values["env_steps"],
            "actions": dict(sorted(values["actions"].items())),
            "durations": {
                str(key): value for key, value in sorted(values["durations"].items())
            },
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "status": manifest.get("status"),
        "events": len(events),
        "event_counts": dict(sorted(event_counts.items())),
        "attempt_count": event_counts.get("attempt_started", 0),
        "attempts": attempt_rows,
        "committed_decisions": len(decision_rows),
        "committed_actions": dict(sorted(actions.items())),
        "committed_durations": {str(key): value for key, value in sorted(durations.items())},
        "investigated_actions": dict(sorted(investigated_actions.items())),
        "investigated_durations": {
            str(key): value for key, value in sorted(investigated_durations.items())
        },
        "archive_restores": event_counts.get("archive_branch_restored", 0),
        "delayed_visual_returns": event_counts.get(
            "delayed_visual_return_detected", 0
        ),
        "delayed_return_recoveries": delayed_return_recoveries,
        "autonomous_dynamics_decisions": event_counts.get(
            "autonomous_dynamics_detected", 0
        ),
        "autonomous_grace_waits": event_counts.get("autonomous_grace_wait", 0),
        "persistent_frontier_updates": event_counts.get(
            "persistent_frontier_updated", 0
        ),
        "persistent_frontier_return_events": event_counts.get(
            "persistent_frontier_return_penalized", 0
        ),
        "persistent_frontier_penalized_traces": frontier_penalized_traces,
        "persistent_frontier_choice_samples": frontier_choice_samples,
        "persistent_frontier_trace_restarts": event_counts.get(
            "persistent_frontier_trace_restarted", 0
        ),
        "maximum_committed_frontier_value": max(
            committed_frontier_values, default=0.0
        ),
        "visual_abstraction_assignments": event_counts.get(
            "visual_abstraction_assigned", 0
        ),
        "visual_abstraction_clusters": len(abstraction_clusters),
        "verified_branches": event_counts.get("branch_verified", 0),
        "unique_frames": len(frames),
        "unique_scenes": len(scenes),
        "states_saved": event_counts.get("state_saved", 0),
        "states_loaded": event_counts.get("state_loaded", 0),
        "states_released": event_counts.get("state_released", 0),
        "annotations": annotations,
        "artifacts": {
            "decisions": "decisions.csv",
            "events": "events.jsonl",
            "frames": "frames/",
            "transitions": "transitions.json",
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize or annotate a Lolo telemetry run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize", help="generate CSV and graph artifacts")
    summarize.add_argument("--run", type=Path, required=True)
    annotate = subparsers.add_parser("annotate-level", help="add an evaluator-only level label")
    annotate.add_argument("--run", type=Path, required=True)
    annotate.add_argument("--label", required=True)
    annotate.add_argument("--start-seq", type=int)
    annotate.add_argument("--end-seq", type=int)
    annotate.add_argument("--attempt", type=int)
    annotate.add_argument("--note")
    args = parser.parse_args()
    if args.command == "annotate-level":
        result = append_level_annotation(
            args.run, args.label, args.start_seq, args.end_seq, args.attempt, args.note
        )
    else:
        result = build_run_summary(args.run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
