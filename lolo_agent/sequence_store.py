from __future__ import annotations

import json
import os
import zlib
from pathlib import Path
from typing import Dict, List, Sequence

from .ensemble_world_model import VisualSequence
from .environment import Action
from .pixels import Frame


class SequenceStore:
    """Crash-safe, segmented sequence dataset with deduplicated compressed frames."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.frames_dir = self.root / "frames"
        self.segments_dir = self.root / "segments"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(parents=True, exist_ok=True)

    def segment_path(self, segment_id: str) -> Path:
        if not segment_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in segment_id):
            raise ValueError("segment IDs may contain lowercase letters, digits, dashes, and underscores")
        return self.segments_dir / f"{segment_id}.jsonl"

    def has_segment(self, segment_id: str) -> bool:
        return self.segment_path(segment_id).is_file()

    def append_segment(self, segment_id: str, sequences: Sequence[VisualSequence]) -> Path:
        if not sequences:
            raise ValueError("cannot persist an empty sequence segment")
        destination = self.segment_path(segment_id)
        if destination.exists():
            raise FileExistsError(destination)
        records = []
        for sequence in sequences:
            frame_records = []
            for frame in sequence.frames:
                frame_path = self.frames_dir / f"{frame.digest}.rgb.zlib"
                if not frame_path.exists():
                    temporary_frame = self.frames_dir / f".{frame.digest}.tmp"
                    temporary_frame.write_bytes(zlib.compress(frame.pixels, level=6))
                    os.replace(temporary_frame, frame_path)
                frame_records.append(
                    {
                        "digest": frame.digest,
                        "width": frame.width,
                        "height": frame.height,
                        "channels": frame.channels,
                    }
                )
            records.append(
                {
                    "version": 1,
                    "group": sequence.group,
                    "frames": frame_records,
                    "actions": [action.value for action in sequence.actions],
                    "durations": list(sequence.durations),
                }
            )
        temporary = self.segments_dir / f".{segment_id}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination

    def load(self) -> List[VisualSequence]:
        frame_cache: Dict[str, Frame] = {}
        sequences = []
        for segment in sorted(self.segments_dir.glob("*.jsonl")):
            with segment.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    record = json.loads(line)
                    if record.get("version") != 1:
                        raise ValueError(f"unsupported sequence record in {segment}:{line_number}")
                    frames = []
                    for item in record["frames"]:
                        digest = item["digest"]
                        frame = frame_cache.get(digest)
                        if frame is None:
                            payload_path = self.frames_dir / f"{digest}.rgb.zlib"
                            pixels = zlib.decompress(payload_path.read_bytes())
                            frame = Frame(
                                int(item["width"]),
                                int(item["height"]),
                                int(item["channels"]),
                                pixels,
                            )
                            if frame.digest != digest:
                                raise ValueError(f"dataset frame digest mismatch: {digest}")
                            frame_cache[digest] = frame
                        frames.append(frame)
                    sequences.append(
                        VisualSequence(
                            int(record["group"]),
                            tuple(frames),
                            tuple(Action(value) for value in record["actions"]),
                            tuple(int(value) for value in record.get("durations", [])),
                        )
                    )
        return sequences

    def statistics(self) -> Dict[str, int]:
        segments = sorted(self.segments_dir.glob("*.jsonl"))
        sequences = 0
        for segment in segments:
            with segment.open(encoding="utf-8") as handle:
                sequences += sum(1 for line in handle if line.strip())
        return {
            "segments": len(segments),
            "sequences": sequences,
            "unique_frames": len(list(self.frames_dir.glob("*.rgb.zlib"))),
            "compressed_frame_bytes": sum(
                path.stat().st_size for path in self.frames_dir.glob("*.rgb.zlib")
            ),
        }
