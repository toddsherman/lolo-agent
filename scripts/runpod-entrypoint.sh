#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${LOLO_ASSET_DIR:-/workspace/lolo-assets}"
mkdir -p "${LOLO_CAMPAIGN_DIR:-/workspace/lolo-campaigns}"

exec "$@"
