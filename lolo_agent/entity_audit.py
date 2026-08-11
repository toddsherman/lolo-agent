from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .environment import Action
from .experience_import import decode_logged_png
from .run_logging import read_events
from .unlabeled_entities import UnlabeledEntityMemory


def audit_run(
    run_dir: Path,
    maximum_frames: int = 0,
    match_threshold: float = 0.08,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    memory = UnlabeledEntityMemory(match_threshold=match_threshold)
    committed = [
        event
        for event in read_events(run_dir)
        if event.get("event") == "decision_committed" and event.get("frame")
    ]
    if maximum_frames > 0:
        committed = committed[:maximum_frames]
    frame_cache: Dict[str, Any] = {}
    grid_observations = []
    interactions = []
    for event in committed:
        digest = str(event["frame"])
        if digest not in frame_cache:
            frame = decode_logged_png(run_dir / "frames" / f"{digest}.png")
            if frame.digest != digest:
                raise ValueError(f"frame digest mismatch: {digest}")
            frame_cache[digest] = frame
        frame = frame_cache[digest]
        observation = memory.observe(frame)
        grid_observations.append(
            {
                "decision": int(event["decision"]),
                "frame": digest,
                "prototype_grid": observation.signature(),
                "source_player_slot": event.get(
                    "human_prior_source_player_slot"
                ),
                "target_player_slot": event.get(
                    "human_prior_target_player_slot"
                ),
                "remaining_hearts": event.get(
                    "human_prior_remaining_hearts"
                ),
            }
        )
        player = event.get("human_prior_source_player_slot")
        if player is None:
            player = event.get("human_prior_target_player_slot")
        try:
            action = Action(str(event.get("action")))
        except ValueError:
            continue
        duration = int(event.get("action_frames", 0))
        target = None
        target_cell = None
        if player is not None:
            target_cell = memory.action_target_cell(
                (int(player[0]), int(player[1])),
                action,
                frame.width,
                frame.height,
            )
            target = memory.target_prototype(
                observation,
                (int(player[0]), int(player[1])),
                action,
                frame.width,
                frame.height,
            )
        visits_before = memory.record_interaction(target, action, duration)
        interactions.append(
            {
                "decision": int(event["decision"]),
                "action": action.value,
                "duration": duration,
                "target_cell": target_cell,
                "target_prototype": target,
                "visits_before": visits_before,
            }
        )

    stats = memory.stats()
    persistent_rare = [
        item
        for item in stats
        if item.unique_cells <= 4
        and item.frames_observed >= max(2, memory.frame_count // 4)
    ]
    persistent_rare.sort(
        key=lambda item: (
            item.unique_cells,
            -item.frames_observed,
            item.prototype_id,
        )
    )
    persistent_ids = {item.prototype_id for item in persistent_rare}
    entity_state_visits: Dict[str, int] = {}
    previous_entity_state = None
    entity_state_transitions = 0
    for item in grid_observations:
        player = item.get("target_player_slot") or item.get(
            "source_player_slot"
        )
        ignored_cell = None
        if player is not None:
            ignored_cell = (
                min(
                    memory.columns - 1,
                    max(
                        0,
                        int(player[0]) * memory.columns // frame.width,
                    ),
                ),
                min(
                    memory.rows - 1,
                    max(0, int(player[1]) * memory.rows // frame.height),
                ),
            )
        entries = tuple(
            (index % memory.columns, index // memory.columns, prototype_id)
            for index, prototype_id in enumerate(item["prototype_grid"])
            if prototype_id in persistent_ids
            and (index % memory.columns, index // memory.columns) != ignored_cell
        )
        payload = ";".join(
            f"{column},{row}={prototype_id}"
            for column, row, prototype_id in entries
        )
        entity_state = hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]
        item["persistent_entity_state"] = entity_state
        item["persistent_entity_entries"] = entries
        entity_state_visits[entity_state] = entity_state_visits.get(
            entity_state, 0
        ) + 1
        if (
            previous_entity_state is not None
            and previous_entity_state != entity_state
        ):
            entity_state_transitions += 1
        previous_entity_state = entity_state
    return {
        "run_dir": str(run_dir),
        "committed_frames": memory.frame_count,
        "unique_frame_pixels_loaded": len(frame_cache),
        "prototype_count": memory.prototype_count,
        "persistent_entity_states": len(entity_state_visits),
        "persistent_entity_state_transitions": entity_state_transitions,
        "persistent_entity_state_visits": entity_state_visits,
        "grid_observations": grid_observations,
        "prototypes": [
            {
                "prototype_id": item.prototype_id,
                "observations": item.observations,
                "frames_observed": item.frames_observed,
                "unique_cells": item.unique_cells,
                "cells": item.cells,
                "spatial_rarity": item.spatial_rarity,
            }
            for item in stats
        ],
        "persistent_rare_prototypes": [
            {
                "prototype_id": item.prototype_id,
                "observations": item.observations,
                "frames_observed": item.frames_observed,
                "unique_cells": item.unique_cells,
                "cells": item.cells,
                "spatial_rarity": item.spatial_rarity,
            }
            for item in persistent_rare
        ],
        "interactions": interactions,
        "unique_interaction_edges": len(memory.interaction_visits),
        "repeated_interaction_edges": sum(
            count > 1 for count in memory.interaction_visits.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit unlabeled visual-patch entities in run telemetry"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--maximum-frames", type=int, default=0)
    parser.add_argument("--match-threshold", type=float, default=0.08)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.maximum_frames < 0:
        parser.error("--maximum-frames must be non-negative")
    if args.match_threshold < 0.0:
        parser.error("--match-threshold must be non-negative")
    result = audit_run(
        args.run_dir,
        maximum_frames=args.maximum_frames,
        match_threshold=args.match_threshold,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
