# Durable collect–train–evaluate experiments

`lolo-experiment` is the main local learning loop. Each cycle:

1. explores the emulator from pixels using alternative save-state branches;
2. records action sequences with explicit controller press durations;
3. atomically adds the sequences to a persistent replay dataset;
4. trains the ensemble world model on MPS;
5. writes a versioned duration-conditioned checkpoint;
6. runs a fresh frozen evaluation with full telemetry and a parameter digest
   audit.

No room number, sprite identity, success flag, RAM value, demonstration, or
game rule enters the loop.

## Start or resume

```bash
source .venv/bin/activate

lolo-experiment \
  --experiment-dir experiments/lolo1-main \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --cycles 10 \
  --roots 20 \
  --branches 3 \
  --horizon 3 \
  --durations 1,2,4,8,16 \
  --epochs 1 \
  --batch-size 8 \
  --eval-decisions 20
```

`--cycles` means additional completed cycles. Repeating the command resumes the
same experiment and completes ten more. Configuration and ROM/core/host hashes
must match the immutable experiment manifest.

## Crash recovery

`state.json` records the current durable phase: `collecting`, `collected`,
`trained`, or `idle`. Dataset cycles are separate immutable segment files.
Frames are zlib-compressed and content-addressed, so identical observations are
stored once.

If the process stops:

- during collection, the incomplete cycle is collected again unless its atomic
  segment was already committed;
- after collection, training resumes from the committed dataset segment;
- after training, evaluation resumes from the completed checkpoint;
- during evaluation, a new numbered evaluation attempt is created.

An emulator save-state capability is intentionally not persisted across process
restarts. After a restart the collector begins a new pixel interaction stream,
while retaining all learned parameters and prior visual experience.

## Variable-duration actions

New world models embed both the controller button and the number of frames it
is held. The default choices are 1, 2, 4, 8, and 16 frames. Real branch
verification compares button–duration pairs, and telemetry records the selected
duration in planner candidates, branch scores, committed decisions, CSV output,
and replay frames.

Older checkpoints remain loadable in fixed-duration mode. They cannot be used
with multiple durations because they never learned duration as an input.

## Artifacts

```text
experiments/lolo1-main/
├── experiment.json
├── state.json
├── dataset/
│   ├── frames/*.rgb.zlib
│   └── segments/cycle-*.jsonl
├── checkpoints/cycle-*.pt
├── metrics/cycle-*.json
├── collection_runs/run-*/
└── evaluations/cycle-*-attempt-*/
```

Collection and evaluation directories use the same event schema as ordinary
telemetry runs. Evaluation runs can be passed directly to `lolo-replay`.

The first meaningful milestone is not a high cycle count. It is a frozen
evaluation that autonomously moves beyond boot/menu animation into a stable,
interactive visual scene and begins producing diverse action consequences
there. Dataset composition and scene diversity should be inspected before
committing to a long overnight run.
