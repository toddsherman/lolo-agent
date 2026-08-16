# RunPod execution

The same pixel-only emulator and model stack can run on an x86-64 Linux GPU
Pod. The repository now includes a CUDA image, a Linux native host build, a
core installer, a deterministic benchmark, and evidence/cost gates. The ROM
is deliberately excluded from Git and the container image.

## Recommended split

- Use the M5 MacBook Pro for development, deterministic replay, telemetry
  inspection, and cheap emulator-only experiments.
- Use RunPod for CUDA training and only for branch workloads that a benchmark
  proves are faster or cheaper there.
- Keep campaigns and checkpoints on a persistent Pod volume. The bounded
  wrapper stops the Pod after each cycle; a running idle Pod still costs money.

An L4 or A5000-class on-demand worker is an appropriate first comparison.
GPU names and prices change, so enter the current dashboard price in every
cycle plan rather than committing a presumed price to code.

## Build the image

From the repository root:

```bash
docker build -f Dockerfile.runpod -t YOUR_REGISTRY/lolo-agent:cuda13 .
docker push YOUR_REGISTRY/lolo-agent:cuda13
```

The build pins PyTorch and verifies the installed version. If the default CUDA
base tag is unavailable, pass a compatible official tag with
`--build-arg CUDA_IMAGE=...`; retain the build-time PyTorch assertion.

Create a private RunPod template using that image. Mount a persistent Pod
volume at `/workspace`. A 100 GB starting volume is safer than copying the existing
multi-gigabyte experiment history into ephemeral container storage.

For this stop-and-resume workflow, do not attach a network volume: RunPod's
current Pod lifecycle only permits network-volume Pods to be terminated, not
stopped. A Pod volume survives a normal stop, although its storage continues
to incur a smaller charge. Back up important artifacts outside RunPod.

## Install private assets

Open a Pod terminal and fetch the open-source Linux Nestopia core:

```bash
/opt/lolo/scripts/fetch-linux-nestopia.sh
```

The script prints the core SHA-256, which the benchmark records. For strict
repeatability, set `LOLO_CORE_SHA256` on later workers. Transfer the legally
obtained ROM privately to:

```text
/workspace/lolo-assets/Adventures of Lolo.nes
```

Do not put the ROM, emulator states, credentials, or experiment history in the
Git repository or container registry.

## Compare before spending on training

On the Pod, set the actual hourly price and run:

```bash
LOLO_GPU_HOURLY_RATE_USD=0.69 /opt/lolo/scripts/runpod-smoke.sh
```

Run the equivalent `lolo-platform-benchmark` command on the Mac. Compare
`branches_per_second`; for the Pod also compare
`estimated_cost_per_million_branches_usd`. A GPU does not accelerate a
single-threaded emulator automatically. It is valuable when CUDA training or
parallel model inference dominates the cycle.

For the first paid campaign, copy `configs/research-cycle.example.json`, enter
the live hourly price, and keep the campaign ceiling small. The example's
five-minute wall limit is approximately six cents at $0.69/hour and its total
campaign cap is $1. The supervisor refuses any plan whose time ceiling could
exceed its declared cycle budget.

Run the paid cycle through the Pod-aware wrapper:

```bash
/opt/lolo/scripts/runpod-bounded-cycle.sh \
  /workspace/cycle-001.json \
  /workspace/lolo-campaigns/first-pilot
```

RunPod provides each Pod with `runpodctl`, a Pod-scoped API key, and
`RUNPOD_POD_ID`. The wrapper refuses to run without those controls, starts an
outer shutdown watchdog, invokes the tighter experiment supervisor, flushes
artifacts, and stops the Pod whether the cycle succeeds or fails. Stopping
releases the GPU; persistent volume storage remains billable until deleted.

## State portability

Learned PyTorch checkpoints and JSON/JSONL telemetry are portable. Native
libretro save states may depend on the exact core build. The benchmark records
host, core, and ROM hashes; validate a save/load replay on Linux before using
old states, and recreate frontier states when hashes or replay output differ.

## What cloud execution does not solve

RunPod compute does not reduce Codex subscription or API-token usage. The
bounded supervisor is a standalone local process, so it can perform the run
and create the audit without an assistant staying in a long conversational
loop. Codex is best used at the reflection checkpoints to interpret evidence,
modify code, and authorize the next finite experiment.
