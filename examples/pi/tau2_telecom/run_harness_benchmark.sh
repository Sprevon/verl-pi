#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Run benchmark on vGPUN.' >&2; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$PROJECT_DIR"
export STUDENT_MODEL="${PI_PROFILE_MODEL:-$PI_WORK_ROOT/models/Qwen3-1.7B}"
export PI_RUN_DIR="${PI_RUN_DIR:-$PI_WORK_ROOT/runs/pi-harness-$(date +%Y%m%d-%H%M%S)}"
export PI_TRACE_DIR="$PI_RUN_DIR/pi-traces"
export PI_STARTUP_PROFILE=0
export PI_MAX_TURNS=6
export PI_MAX_RESPONSE_LENGTH=256
export PI_ROLLOUT_N=4
mkdir -p "$PI_RUN_DIR"
[[ ! -e "$PI_RUN_DIR/profile_exit_code.txt" ]] || { echo 'Use a new run directory.' >&2; exit 1; }
"$PYTHON_BIN" examples/pi/tau2_telecom/profile_timing.py monitor "$PI_RUN_DIR" > "$PI_RUN_DIR/monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill -TERM "$monitor_pid" 2>/dev/null || true' EXIT
set +e
timeout --signal=TERM --kill-after=15s 900 "$PYTHON_BIN" examples/pi/tau2_telecom/benchmark_harness_pool.py \
  --run-dir "$PI_RUN_DIR" --model "$STUDENT_MODEL" --data "$DATA_DIR/train.parquet" \
  --pool-name "pi-harness-$(basename "$PI_RUN_DIR")" "$@" 2>&1 | tee "$PI_RUN_DIR/train.log"
run_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_status" > "$PI_RUN_DIR/profile_exit_code.txt"
wait "$monitor_pid"
if [[ "$run_status" == 0 ]]; then
  "$PYTHON_BIN" examples/pi/tau2_telecom/profile_timing.py summarize "$PI_RUN_DIR" > "$PI_RUN_DIR/timing-console.json"
fi
exit "$run_status"
