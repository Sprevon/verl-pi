#!/usr/bin/env bash
# Run on the remote Linux host after pulling this repository.
set -euo pipefail

[[ "$(uname -s)" == Linux ]] || { echo 'Run this installer on vGPUN, not macOS.' >&2; exit 1; }
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PI_WORK_ROOT="${PI_WORK_ROOT:-/root/autodl-tmp}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-/root/miniconda3/bin/python}"
NODE_VERSION="${NODE_VERSION:-22.22.0}"
TAU2_REVISION="${TAU2_REVISION:-fe8ac67662d9ba765c7324ed732872d362133895}"
export UV_CACHE_DIR="$PI_WORK_ROOT/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$PI_WORK_ROOT/envs/verl-pi"
export UV_HTTP_TIMEOUT=180
export TMPDIR="$PI_WORK_ROOT/tmp"
mkdir -p "$PI_WORK_ROOT/tools" "$PI_WORK_ROOT/envs" "$PI_WORK_ROOT/code" "$TMPDIR"

UV_BIN="$PI_WORK_ROOT/tools/uv/bin/uv"
if [[ ! -x "$UV_BIN" ]]; then
  "$BOOTSTRAP_PYTHON" -m pip install --target "$PI_WORK_ROOT/tools/uv" uv
fi

NODE_DIR="$PI_WORK_ROOT/tools/node-v$NODE_VERSION-linux-x64"
if [[ ! -x "$NODE_DIR/bin/node" ]]; then
  NODE_ARCHIVE="node-v$NODE_VERSION-linux-x64.tar.xz"
  curl --fail --location --retry 3 "https://nodejs.org/dist/v$NODE_VERSION/$NODE_ARCHIVE" -o "$TMPDIR/$NODE_ARCHIVE"
  curl --fail --location --retry 3 "https://nodejs.org/dist/v$NODE_VERSION/SHASUMS256.txt" -o "$TMPDIR/node-SHASUMS256.txt"
  (cd "$TMPDIR" && grep " $NODE_ARCHIVE\$" node-SHASUMS256.txt | sha256sum --check -)
  tar -xJf "$TMPDIR/$NODE_ARCHIVE" -C "$PI_WORK_ROOT/tools"
fi
export PATH="$NODE_DIR/bin:$(dirname "$UV_BIN"):$PATH"

cd "$PROJECT_DIR"
# Reuse the repository's CUDA 13 / Torch 2.11 / vLLM 0.24 lock without regenerating it.
"$UV_BIN" sync --frozen --extra fsdp --extra vllm --extra test --python "$BOOTSTRAP_PYTHON"
npm --prefix "$PROJECT_DIR/verl/experimental/agent_loop/pi/sidecar" install --ignore-scripts --package-lock=false

TAU2_BENCH_ROOT="${TAU2_BENCH_ROOT:-$PI_WORK_ROOT/code/tau2-bench}"
if [[ ! -d "$TAU2_BENCH_ROOT/.git" ]]; then
  git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/Sprevon/tau2-bench.git "$TAU2_BENCH_ROOT"
fi
if [[ "$(git -C "$TAU2_BENCH_ROOT" rev-parse HEAD)" != "$TAU2_REVISION" ]]; then
  [[ -z "$(git -C "$TAU2_BENCH_ROOT" status --porcelain)" ]] || {
    echo 'Tau2 worktree has local changes; refusing to switch revision.' >&2; exit 1;
  }
  git -C "$TAU2_BENCH_ROOT" fetch origin "$TAU2_REVISION"
  git -C "$TAU2_BENCH_ROOT" checkout --detach "$TAU2_REVISION"
fi
TAU2_ENV="$PI_WORK_ROOT/envs/tau2-pi"
[[ -x "$TAU2_ENV/bin/python" ]] || "$UV_BIN" venv --python "$BOOTSTRAP_PYTHON" "$TAU2_ENV"
# Isolate the environment/evaluator's LiteLLM dependencies from the trainer.
"$UV_BIN" pip install --python "$TAU2_ENV/bin/python" -e "$TAU2_BENCH_ROOT[gym]" pyarrow

"$UV_BIN" --version
node --version
"$UV_BIN" pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" > "$PI_WORK_ROOT/verl-pi-environment.txt"
"$UV_BIN" pip freeze --python "$TAU2_ENV/bin/python" > "$PI_WORK_ROOT/tau2-pi-environment.txt"
printf 'Trainer Python: %s\nTau2 Python: %s\nNode: %s\n' \
  "$UV_PROJECT_ENVIRONMENT/bin/python" "$TAU2_ENV/bin/python" "$NODE_DIR/bin/node"
