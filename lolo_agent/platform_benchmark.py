from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from .environment import Action
from .native_env import NativeLibretroEnv
from .neural_world_model import choose_torch_device
from .run_logging import sha256_file, utc_now


_ACTIONS = (
    Action.NOOP,
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
    Action.A,
    Action.B,
)


def benchmark(
    host: Path,
    core: Path,
    rom: Path,
    branches: int,
    action_frames: int,
    hourly_rate_usd: float = 0.0,
) -> dict[str, object]:
    if branches <= 0 or action_frames <= 0:
        raise ValueError("branches and action_frames must be positive")
    if hourly_rate_usd < 0:
        raise ValueError("hourly_rate_usd must be non-negative")
    device = choose_torch_device()
    with NativeLibretroEnv(host, core, rom) as env:
        initial = env.reset()
        root = env.save_state()
        digest_accumulator = 0
        started = time.perf_counter()
        for index in range(branches):
            env.load_state(root)
            frame = env.step(
                _ACTIONS[index % len(_ACTIONS)], action_frames
            )
            digest_accumulator ^= int(frame.digest[:16], 16)
        elapsed = time.perf_counter() - started
        env.release_state(root)
        core_name = env.core_name
        core_version = env.core_version
    return {
        "version": 1,
        "completed_at": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "inputs": {
            "host_sha256": sha256_file(host),
            "core_sha256": sha256_file(core),
            "rom_sha256": sha256_file(rom),
            "core_name": core_name,
            "core_version": core_version,
        },
        "frame": {
            "width": initial.width,
            "height": initial.height,
            "channels": initial.channels,
        },
        "benchmark": {
            "branches": branches,
            "action_frames": action_frames,
            "elapsed_seconds": elapsed,
            "branches_per_second": branches / elapsed,
            "frames_per_second": branches * action_frames / elapsed,
            "digest_accumulator": f"{digest_accumulator:016x}",
            "estimated_cost_per_million_branches_usd": (
                None
                if hourly_rate_usd == 0
                else hourly_rate_usd
                / (branches / elapsed * 3600)
                * 1_000_000
            ),
            "hourly_rate_usd": hourly_rate_usd,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic headless save-state branches"
    )
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--branches", type=int, default=2000)
    parser.add_argument("--action-frames", type=int, default=16)
    parser.add_argument("--hourly-rate-usd", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(
        args.host.expanduser().resolve(),
        args.core.expanduser().resolve(),
        args.rom.expanduser().resolve(),
        args.branches,
        args.action_frames,
        args.hourly_rate_usd,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
