#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Run profiling on vGPUN.' >&2; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$PROJECT_DIR"
export STUDENT_MODEL="${PI_PROFILE_MODEL:-$PI_WORK_ROOT/models/Qwen3-1.7B}"
export PI_RUN_DIR="${PI_RUN_DIR:-$PI_WORK_ROOT/runs/pi-timing-$(date +%Y%m%d-%H%M%S)}"
export PI_ROLLOUT_N="${PI_ROLLOUT_N:-4}"
export PI_MAX_TURNS="${PI_MAX_TURNS:-6}"
export PI_MAX_RESPONSE_LENGTH="${PI_MAX_RESPONSE_LENGTH:-256}"
export PI_DATA_DIR="${PI_DATA_DIR:-$PI_WORK_ROOT/data/pi-tau2-smoke}"
mkdir -p "$PI_RUN_DIR"
[[ ! -e "$PI_RUN_DIR/profile_exit_code.txt" ]] || { echo 'Use a new run directory.' >&2; exit 1; }
"$PYTHON_BIN" examples/pi/tau2_telecom/profile_timing.py monitor "$PI_RUN_DIR" > "$PI_RUN_DIR/monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill -TERM "$monitor_pid" 2>/dev/null || true' EXIT
set +e
timeout --signal=TERM --kill-after=15s 900 bash examples/pi/tau2_telecom/run_single_card.sh \
  actor_rollout_ref.rollout.disable_log_stats=false \
  trainer.save_freq=-1 "$@"
run_status=$?
set -e
printf '%s\n' "$run_status" > "$PI_RUN_DIR/profile_exit_code.txt"
wait "$monitor_pid"
"$PYTHON_BIN" examples/pi/tau2_telecom/profile_timing.py summarize "$PI_RUN_DIR" > "$PI_RUN_DIR/timing-console.json"
printf 'Timing report: %s/timing-report.md\nRun exit code: %s\n' "$PI_RUN_DIR" "$run_status"
exit "$run_status"
