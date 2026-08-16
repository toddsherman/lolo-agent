# RunPod platform gate — 2026-08-15

## Decision

Do not move the sequential libretro save-state search workload from the M5
MacBook Pro to RunPod. Keep emulator branching, replay, telemetry inspection,
and search orchestration local. Consider RunPod only for bounded offline neural
training or genuinely parallel model inference after a separate benchmark
passes.

## Hypothesis

An RTX A4000 RunPod worker with 18 vCPUs would reduce the time or cost of the
deterministic emulator branching workload relative to the M5 baseline.

## Controlled evidence

Both platforms used the same ROM SHA-256 and the same benchmark logic: restore
one opaque native save-state, take one controller action for 16 frames, and
repeat. Host and core hashes were recorded because native binaries differ by
platform.

| Measurement | M5 MacBook Pro | RunPod RTX A4000 |
|---|---:|---:|
| Branches | 2,000 | 5,000 |
| Branches/second | 586.34 | 135.98 |
| Emulated frames/second | 9,381.46 | 2,175.75 |
| Relative M5 speed | 1.00× | 0.232× |
| Declared GPU rate | local | $0.25/hour |
| Estimated cost/million branches | local | $0.5107 |

The M5 was **4.31× faster**. CUDA was available on the Pod, but this benchmark
measures a sequential CPU-bound emulator path; the GPU does not accelerate it.
The result is therefore evidence against migrating the current search loop,
not evidence against GPU training.

## Artifacts

The machine-readable reports remain outside Git because they contain private
ROM hashes and environment provenance:

- `experiments/platform-benchmarks/m5-baseline.json`
- `experiments/platform-benchmarks/runpod-a4000.json`

The legally obtained ROM remains excluded from Git. It was copied only to the
user's private RunPod volume for this benchmark.

## Reflection

- Finding: more advertised vCPUs do not improve the current single-process
  save-state loop, and Linux/x86-64 was substantially slower than M5/ARM64.
- Decision: revise.
- Plan change: keep real emulator verification local and reserve cloud GPU time
  for work that actually invokes CUDA at meaningful utilization.
- Stop condition: no additional paid emulator-search benchmark is justified
  unless the branch engine is parallelized or its implementation changes
  materially.
- Next hypothesis: a fixed offline world-model training benchmark can achieve
  sufficiently higher examples/second on an inexpensive GPU to lower time or
  cost per validated model improvement versus M5 MPS.

The next cycle begins locally by defining and measuring that fixed training
benchmark on the M5. A paid GPU comparison is permitted only after the local
measurement, artifact schema, validation metric, wall-time limit, and dollar
ceiling are all fixed in advance.
