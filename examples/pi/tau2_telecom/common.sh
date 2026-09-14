#!/usr/bin/env bash
# Source on vGPUN; all paths may be overridden in the launch environment.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PI_WORK_ROOT="${PI_WORK_ROOT:-/root/autodl-tmp}"
NODE_VERSION="${NODE_VERSION:-22.22.0}"
export PYTHON_BIN="${PYTHON_BIN:-$PI_WORK_ROOT/envs/verl-pi/bin/python}"
export TAU2_PI_PYTHON="${TAU2_PI_PYTHON:-$PI_WORK_ROOT/envs/tau2-pi/bin/python}"
export PI_NODE_BINARY="${PI_NODE_BINARY:-$PI_WORK_ROOT/tools/node-v$NODE_VERSION-linux-x64/bin/node}"
export TAU2_BENCH_ROOT="${TAU2_BENCH_ROOT:-$PI_WORK_ROOT/code/tau2-bench}"
export PI_TRAINING_EXTENSION="${PI_TRAINING_EXTENSION:-$TAU2_BENCH_ROOT/.pi/extensions/agent-r1-training.ts}"
export PI_AGENT_DIR="${PI_AGENT_DIR:-$TAU2_BENCH_ROOT/.pi/agent}"
export TAU2_SOLO_MODE=1
export PATH="$(dirname "$PYTHON_BIN"):$(dirname "$PI_NODE_BINARY"):$PATH"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PI_WORK_ROOT/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PI_WORK_ROOT/.cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PI_WORK_ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PI_WORK_ROOT/.cache/torchinductor}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$PI_WORK_ROOT/.cache/flashinfer}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export STUDENT_MODEL="${STUDENT_MODEL:-$PI_WORK_ROOT/models/Qwen3-0.6B}"
DATA_DIR="${PI_DATA_DIR:-$PI_WORK_ROOT/data/pi-tau2-smoke}"
