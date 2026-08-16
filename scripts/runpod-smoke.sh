#!/usr/bin/env bash
set -euo pipefail

asset_dir="${LOLO_ASSET_DIR:-/workspace/lolo-assets}"
rom_path="${LOLO_ROM_PATH:-${asset_dir}/Adventures of Lolo.nes}"
core_path="${LOLO_CORE_PATH:-${asset_dir}/nestopia_libretro.so}"
host_path="${LOLO_HOST_PATH:-/opt/lolo/build/lolo-libretro-host}"
output_path="${1:-/workspace/platform-benchmark.json}"

for required in "${rom_path}" "${core_path}" "${host_path}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing required file: ${required}" >&2
        exit 1
    fi
done

python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
lolo-platform-benchmark \
    --host "${host_path}" \
    --core "${core_path}" \
    --rom "${rom_path}" \
    --branches "${LOLO_BENCHMARK_BRANCHES:-2000}" \
    --action-frames "${LOLO_BENCHMARK_ACTION_FRAMES:-16}" \
    --hourly-rate-usd "${LOLO_GPU_HOURLY_RATE_USD:-0}" \
    --output "${output_path}"
