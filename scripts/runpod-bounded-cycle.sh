#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 PLAN_JSON CAMPAIGN_DIR" >&2
    exit 2
fi
if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "RUNPOD_POD_ID is required; refusing an unguarded paid run" >&2
    exit 2
fi
if ! command -v runpodctl >/dev/null 2>&1; then
    echo "runpodctl is required for automatic Pod shutdown" >&2
    exit 2
fi

plan_path="$1"
campaign_dir="$2"
wall_seconds="$(python -c 'import json,sys; print(float(json.load(open(sys.argv[1]))["budgets"]["max_wall_seconds"]))' "${plan_path}")"
watchdog_seconds="$(python -c 'import math,sys; print(math.ceil(float(sys.argv[1])) + 90)' "${wall_seconds}")"

# This outer watchdog covers supervisor startup/teardown faults. The research
# cycle supplies the tighter process and event limits.
(
    sleep "${watchdog_seconds}"
    runpodctl pod stop "${RUNPOD_POD_ID}"
) &
watchdog_pid=$!

set +e
lolo-research-cycle run \
    --plan "${plan_path}" \
    --campaign-dir "${campaign_dir}"
cycle_status=$?
set -e

kill "${watchdog_pid}" 2>/dev/null || true
wait "${watchdog_pid}" 2>/dev/null || true
sync

echo "Cycle ended with status ${cycle_status}; stopping Pod ${RUNPOD_POD_ID}."
runpodctl pod stop "${RUNPOD_POD_ID}"
exit "${cycle_status}"
