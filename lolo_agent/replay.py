from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .environment import Action
from .native_env import NativeLibretroEnv
from .pixels import Frame
from .run_logging import encode_png, read_events, sha256_file, utc_now


@dataclass
class ReplayCapture:
    full: List[Dict[str, Any]]
    step_frames: Dict[int, List[Dict[str, Any]]]
    decision_frames: Dict[int, Dict[str, Any]]
    verified_events: int
    checked_observations: int


def _verify_input(path: Path, recorded: Dict[str, Any], kind: str) -> None:
    expected = recorded.get("sha256") or recorded.get("file_sha256")
    if expected is None:
        raise RuntimeError(f"run manifest has no {kind} digest")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{kind} digest mismatch: expected {expected}, got {actual}")


def _check_frame(frame: Frame, event: Dict[str, Any], field: str = "frame") -> None:
    expected = event.get(field)
    if expected is not None and frame.digest != expected:
        raise RuntimeError(
            f"replay diverged at event {event['seq']} ({event['event']}): "
            f"expected frame {expected}, got {frame.digest}"
        )


def _entry(frame: Frame, event: Dict[str, Any], kind: str, **fields: Any) -> Dict[str, Any]:
    result = {
        "frame": frame.digest,
        "kind": kind,
        "event_seq": event["seq"],
        "attempt": event.get("attempt", 0),
        "elapsed_ms_original": event.get("elapsed_ms"),
    }
    result.update(fields)
    return result


def capture_replay(
    run_dir: Path,
    host_path: Path,
    core_path: Path,
    rom_path: Path,
) -> Tuple[ReplayCapture, Dict[str, Frame]]:
    """Deterministically reconstruct all state operations and individual NES frames."""

    run_dir = Path(run_dir).expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    inputs = manifest.get("metadata", {}).get("inputs", {})
    _verify_input(rom_path, inputs.get("rom", {}), "ROM")
    _verify_input(core_path, inputs.get("core", {}), "core")
    _verify_input(host_path, inputs.get("host", {}), "native host")
    events = list(read_events(run_dir))
    frames: Dict[str, Frame] = {}
    full: List[Dict[str, Any]] = []
    step_frames: Dict[int, List[Dict[str, Any]]] = {}
    decision_frames: Dict[int, Dict[str, Any]] = {}
    handles: Dict[str, object] = {}
    current: Optional[Frame] = None
    checked = 0

    with NativeLibretroEnv(host_path, core_path, rom_path) as env:
        try:
            for event in events:
                kind = event["event"]
                if kind == "env_reset":
                    current = env.reset()
                    _check_frame(current, event)
                    checked += 1
                    frames[current.digest] = current
                    full.append(_entry(current, event, "reset"))
                elif kind == "state_saved":
                    state_id = event.get("state_id")
                    if not state_id or state_id in handles:
                        raise RuntimeError(f"invalid state save alias at event {event['seq']}")
                    handles[state_id] = env.save_state()
                    if current is not None:
                        _check_frame(current, event)
                        checked += 1
                elif kind == "state_loaded":
                    state_id = event.get("state_id")
                    if state_id not in handles:
                        raise RuntimeError(f"unknown state alias {state_id!r} at event {event['seq']}")
                    current = env.load_state(handles[state_id])
                    _check_frame(current, event)
                    checked += 1
                    frames[current.digest] = current
                    full.append(_entry(current, event, "state_load", state_id=state_id))
                elif kind == "state_released":
                    state_id = event.get("state_id")
                    if state_id not in handles:
                        raise RuntimeError(f"unknown state release {state_id!r} at event {event['seq']}")
                    env.release_state(handles.pop(state_id))
                elif kind == "env_step":
                    action = Action(event["action"])
                    action_frames = int(event["action_frames"])
                    captured = []
                    for subframe in range(1, action_frames + 1):
                        current = env.step(action, 1)
                        frames[current.digest] = current
                        item = _entry(
                            current,
                            event,
                            "action_frame",
                            action=action.value,
                            action_frame=subframe,
                            action_frames=action_frames,
                            visual_change=event.get("visual_change"),
                        )
                        captured.append(item)
                        full.append(item)
                    _check_frame(current, event, "target_frame")
                    checked += 1
                    step_frames[int(event["seq"])] = captured
                elif kind == "decision_committed":
                    if current is None:
                        raise RuntimeError(f"decision without emulator frame at event {event['seq']}")
                    _check_frame(current, event)
                    checked += 1
                    frames[current.digest] = current
                    marker = _entry(
                        current,
                        event,
                        "decision",
                        decision=event["decision"],
                        action=event["action"],
                        path=event.get("path", []),
                        score=event.get("score"),
                        restored_archive=event.get("restored_archive", False),
                        committed_state_id=event.get("committed_state_id"),
                    )
                    decision_frames[int(event["seq"])] = marker
                    full.append(marker)
        finally:
            for handle in handles.values():
                try:
                    env.release_state(handle)
                except Exception:
                    pass

    verified_events = sum(event["event"] == "branch_verified" for event in events)
    return ReplayCapture(full, step_frames, decision_frames, verified_events, checked), frames


def committed_timeline(events: Sequence[Dict[str, Any]], capture: ReplayCapture) -> List[Dict[str, Any]]:
    branch_by_state = {
        event.get("state_id"): event
        for event in events
        if event["event"] == "branch_verified" and event.get("state_id")
    }
    timeline: List[Dict[str, Any]] = []
    resets_by_attempt = {
        int(item.get("attempt", 0)): item for item in capture.full if item["kind"] == "reset"
    }
    current_attempt: Optional[int] = None
    for decision in (event for event in events if event["event"] == "decision_committed"):
        attempt = int(decision.get("attempt", 0))
        if attempt != current_attempt:
            reset = resets_by_attempt.get(attempt)
            if reset is not None:
                timeline.append(reset)
            current_attempt = attempt
        state_id = decision.get("committed_state_id")
        branch = branch_by_state.get(state_id)
        if not decision.get("restored_archive") and branch is not None:
            for item in capture.step_frames.get(int(branch["env_step_seq"]), []):
                committed = dict(item)
                committed.update(
                    {
                        "decision": decision["decision"],
                        "branch_id": branch.get("branch_id"),
                        "path": decision.get("path", []),
                        "score": decision.get("score"),
                        "committed": True,
                    }
                )
                timeline.append(committed)
        marker = capture.decision_frames[int(decision["seq"])]
        if decision.get("restored_archive") or not timeline or timeline[-1]["frame"] != marker["frame"]:
            timeline.append(marker)
    return timeline


def _player_html(title: str, timeline: Sequence[Dict[str, Any]], default_speed: int) -> str:
    encoded = json.dumps(list(timeline), separators=(",", ":")).replace("</", "<\\/")
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
  :root {{ color-scheme: dark; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  body {{ margin: 0; background: #090b10; color: #e7eaf0; }}
  main {{ display: grid; grid-template-columns: minmax(320px, 2fr) minmax(280px, 1fr); gap: 20px; padding: 20px; }}
  h1 {{ font: 600 18px system-ui; margin: 0 0 14px; }}
  .screen {{ display: grid; place-items: center; min-height: 60vh; background: #000; border: 1px solid #262b36; border-radius: 8px; }}
  img {{ width: min(100%, 1024px); image-rendering: pixelated; }}
  .controls {{ display: grid; grid-template-columns: auto auto 1fr auto; gap: 8px; align-items: center; margin-top: 12px; }}
  button, select {{ font: inherit; padding: 7px 10px; background: #171b24; color: inherit; border: 1px solid #343b49; border-radius: 5px; }}
  input[type=range] {{ width: 100%; }}
  pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #11141b; border: 1px solid #262b36; border-radius: 8px; padding: 14px; min-height: 260px; }}
  .muted {{ color: #929bad; }}
  @media (max-width: 800px) {{ main {{ grid-template-columns: 1fr; }} .screen {{ min-height: 45vh; }} }}
</style>
</head>
<body><main>
  <section><h1>{safe_title}</h1><div class="screen"><img id="screen" alt="NES replay frame"></div>
    <div class="controls"><button id="play">Play</button><button id="step">Step</button><input id="scrub" type="range" min="0" max="0" value="0"><select id="speed">
      <option value="5">5 fps</option><option value="15">15 fps</option><option value="30">30 fps</option><option value="60">60 fps</option><option value="120">120 fps</option><option value="240">240 fps</option>
    </select></div>
  </section>
  <aside><div id="position" class="muted"></div><pre id="details"></pre><div class="muted">Space: play/pause · ←/→: step · Home/End: jump</div></aside>
</main>
<script>
const timeline={encoded};
let index=0, playing=false, previous=0, carry=0;
const screen=document.querySelector('#screen'), scrub=document.querySelector('#scrub'), details=document.querySelector('#details'), position=document.querySelector('#position'), play=document.querySelector('#play'), speed=document.querySelector('#speed');
speed.value=String({default_speed}); scrub.max=String(Math.max(0,timeline.length-1));
function render() {{
  if (!timeline.length) {{ details.textContent='No replay frames'; return; }}
  const item=timeline[index]; screen.src='frames/'+item.frame+'.png'; scrub.value=String(index);
  position.textContent=`Frame ${{index+1}} / ${{timeline.length}} · ${{speed.value}} fps`;
  details.textContent=JSON.stringify(item,null,2);
}}
function setPlaying(value) {{ playing=value; play.textContent=value?'Pause':'Play'; previous=performance.now(); carry=0; if(value) requestAnimationFrame(tick); }}
function tick(now) {{
  if(!playing) return; carry+=(now-previous)*Number(speed.value)/1000; previous=now;
  const advance=Math.floor(carry); if(advance) {{ carry-=advance; index=Math.min(timeline.length-1,index+advance); render(); if(index===timeline.length-1) return setPlaying(false); }}
  requestAnimationFrame(tick);
}}
play.onclick=()=>setPlaying(!playing); document.querySelector('#step').onclick=()=>{{index=Math.min(timeline.length-1,index+1);render();}};
scrub.oninput=()=>{{index=Number(scrub.value);render();}}; speed.onchange=render;
document.addEventListener('keydown',e=>{{ if(e.key===' '){{e.preventDefault();setPlaying(!playing);}} else if(e.key==='ArrowRight'){{index=Math.min(timeline.length-1,index+1);render();}} else if(e.key==='ArrowLeft'){{index=Math.max(0,index-1);render();}} else if(e.key==='Home'){{index=0;render();}} else if(e.key==='End'){{index=timeline.length-1;render();}} }});
render();
</script></body></html>"""


def write_player(
    output_dir: Path,
    title: str,
    timeline: Sequence[Dict[str, Any]],
    frames: Dict[str, Frame],
    default_speed: int,
    source_run: str,
    integrity: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    required = {item["frame"] for item in timeline}
    for digest in required:
        destination = frames_dir / f"{digest}.png"
        if destination.exists():
            continue
        temporary = frames_dir / f".{digest}.tmp"
        temporary.write_bytes(encode_png(frames[digest]))
        os.replace(temporary, destination)
    (output_dir / "timeline.json").write_text(
        json.dumps(list(timeline), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "index.html").write_text(
        _player_html(title, timeline, default_speed), encoding="utf-8"
    )
    manifest = {
        "created_at": utc_now(),
        "source_run": source_run,
        "title": title,
        "timeline_frames": len(timeline),
        "unique_frames": len(required),
        "default_speed_fps": default_speed,
        "kind_counts": dict(sorted(Counter(item["kind"] for item in timeline).items())),
        "integrity": integrity,
        "artifacts": {"player": "index.html", "timeline": "timeline.json", "frames": "frames/"},
    }
    (output_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def render_replays(
    run_dir: Path,
    host_path: Path,
    core_path: Path,
    rom_path: Path,
    mode: str = "both",
    output: Optional[Path] = None,
    default_speed: int = 120,
) -> Dict[str, Dict[str, Any]]:
    run_dir = Path(run_dir).expanduser().resolve()
    events = list(read_events(run_dir))
    capture, frames = capture_replay(run_dir, host_path, core_path, rom_path)
    destination = (run_dir / "replays") if output is None else Path(output).expanduser().resolve()
    integrity = {
        "status": "pass",
        "checked_observations": capture.checked_observations,
        "verified_branch_events": capture.verified_events,
        "rom_sha256": sha256_file(rom_path),
        "core_sha256": sha256_file(core_path),
        "host_sha256": sha256_file(host_path),
    }
    results = {}
    if mode in ("committed", "both"):
        timeline = committed_timeline(events, capture)
        results["committed"] = write_player(
            destination / "committed",
            f"{run_dir.name} — committed gameplay",
            timeline,
            frames,
            default_speed,
            run_dir.name,
            integrity,
        )
    if mode in ("full", "both"):
        results["full"] = write_player(
            destination / "full",
            f"{run_dir.name} — complete planning exploration",
            capture.full,
            frames,
            default_speed,
            run_dir.name,
            integrity,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Render deterministic high-speed NES replays")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--mode", choices=("committed", "full", "both"), default="both")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--speed", type=int, choices=(5, 15, 30, 60, 120, 240), default=120)
    args = parser.parse_args()
    results = render_replays(
        args.run, args.host, args.core, args.rom, args.mode, args.output, args.speed
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
