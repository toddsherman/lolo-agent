# Local native training

## Components

The local path has three process boundaries:

1. `lolo-libretro-host` owns the ROM, Nestopia core, and serialized states.
2. `NativeLibretroEnv` sends controller commands and receives RGB frames plus
   opaque state capabilities.
3. PyTorch trains `VisualDynamicsModel` on the MPS device.

No native protocol command exposes CPU memory, PPU memory, ROM bytes, tile
tables, sprites, room numbers, or completion flags. Save-state capabilities are
valid only in their originating process and can be explicitly released.

## Build and verify

```bash
source .venv/bin/activate
make -C native

LOLO_ROM_PATH="$PWD/Adventures of Lolo.nes" \
LOLO_CORE_PATH="$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
LOLO_NATIVE_HOST_PATH="$PWD/build/lolo-libretro-host" \
python -m unittest discover -s tests -v
```

The ROM is intentionally ignored by Git. Record its digest separately in the
experiment manifest rather than committing the game content.

## Training smoke run

```bash
lolo-train-smoke \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --decisions 24 \
  --branches 3 \
  --action-frames 4 \
  --epochs 2 \
  --batch-size 8 \
  --checkpoint checkpoints/smoke.pt
```

This collector uses save states to try several controller actions from the same
pixels, records their resulting pixels, and commits one randomly selected
branch. The action selection contains no object names, room knowledge, rules,
or demonstrations.

The smoke model predicts an action-conditioned next latent and next image. Its
current purpose is to validate data flow, device execution, learning, checkpoint
creation, and freezing. It is not yet the planner's production world model.

## Local performance snapshot

On the 24 GB M5 development machine, a 600-frame comparison measured roughly:

| Environment | Rendered frames/s | Load+save branches/s |
| --- | ---: | ---: |
| Python `ctypes` loader | 122 | 60,179 |
| Isolated native host | 4,219 | 14,903 |

The native path is about 35 times faster for frame production. Its branch
operation includes process round trips and RGB frame transfer; later shared
memory batching can reduce that overhead if planning profiles show it matters.

These figures are local diagnostic measurements, not portable benchmark claims.

See [neural-planning.md](neural-planning.md) for the ensemble rollout design,
held-out horizon diagnostics, verification gate, and known limitations.
