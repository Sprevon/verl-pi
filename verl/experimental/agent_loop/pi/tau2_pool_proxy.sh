#!/usr/bin/env bash
set -euo pipefail
exec "${PI_HARNESS_PYTHON:?Missing Tau2 interpreter}" \
  "$(dirname "${BASH_SOURCE[0]}")/tau2_pool_service.py" proxy "$@"
