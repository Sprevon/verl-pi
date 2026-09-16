#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Run profiling on vGPUN.' >&2; exit 1; }
exec "${TAU2_PI_REAL_PYTHON:?Set the real Tau2 Python interpreter}" \
  "$(dirname "${BASH_SOURCE[0]}")/profile_tau2_bridge.py" "$@"
