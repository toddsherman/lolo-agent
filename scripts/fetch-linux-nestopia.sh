#!/usr/bin/env bash
set -euo pipefail

asset_dir="${LOLO_ASSET_DIR:-/workspace/lolo-assets}"
destination="${asset_dir}/nestopia_libretro.so"
archive_url="https://buildbot.libretro.com/nightly/linux/x86_64/latest/nestopia_libretro.so.zip"
temporary_dir="$(mktemp -d)"
trap 'rm -rf "${temporary_dir}"' EXIT

mkdir -p "${asset_dir}"
curl --fail --location --retry 3 --output "${temporary_dir}/core.zip" "${archive_url}"
unzip -q "${temporary_dir}/core.zip" -d "${temporary_dir}/core"

actual_sha256="$(sha256sum "${temporary_dir}/core/nestopia_libretro.so" | awk '{print $1}')"
if [[ -n "${LOLO_CORE_SHA256:-}" && "${actual_sha256}" != "${LOLO_CORE_SHA256}" ]]; then
    echo "Nestopia SHA-256 mismatch: expected ${LOLO_CORE_SHA256}, got ${actual_sha256}" >&2
    exit 1
fi

install -m 0755 "${temporary_dir}/core/nestopia_libretro.so" "${destination}"
printf '%s  %s\n' "${actual_sha256}" "${destination}"
