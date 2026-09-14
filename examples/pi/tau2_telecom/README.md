# Pi coding-agent + verl v1 单卡 GRPO

本示例把真实 Pi coding-agent 作为每条轨迹的执行器。Pi 加载 canonical Tau2
Telecom Skill/extension、执行工具并触发 evaluator；verl 的 LLMServerClient 提供
student generation，原生 v1 trainer 负责 GRPO、FSDP2 和 checkpoint。

## 数据和执行边界

`PiAgentLoop.run()` 返回 `list[AgentLoopOutput]`，每个 output 是一次实际模型调用。
保存当轮完整 prompt token、原始生成 token 和 rollout logprobs；工具结果成为下一段
prompt，response mask 只覆盖该次生成。Pi transcript 不用于事后重建训练 token。
每条轨迹有独立 sidecar；一条轨迹的多轮请求复用同一 LLMServerClient request ID。

v1 TQ 使用 `{uid}_{session_id}_{index}` 组织片段，GRPO 以每个 session 的最终奖励
计算优势，再分配到各片段。Reward 来自 canonical evaluator，没有额外奖励塑形。
组内奖励相同时 GRPO 优势为零；训练代码正常退出不能单独证明模型获得了学习信号。
使用 token-mean loss；不同长度会话的权重遵循原生 token 聚合规则。

当前仅支持文本任务、v1 sync trainer 和 GRPO。多段 OPD、GAE/GiGPO 及异步权重更新
尚未接入，配置误用会在 PiAgentLoop 初始化时报错。Pi 自动压缩和模型调用重试在本示例
的训练及 preflight 中均禁用。达到 prompt 上限会失败，不删除 skill/tool schema。

## 远程准备

以下全部在 vGPUN 执行；仓库代码必须先从本地 commit/push，再在远程拉取。
`bootstrap_vgpun.sh` 使用仓库的 `uv.lock`（fsdp + vllm + test）建立隔离训练环境，
单独建立 Tau2 Python 环境，安装 Node 22.22.0、Pi SDK 0.84.4，并固定 Tau2 revision。

```bash
cd /root/autodl-tmp/code/verl-pi
bash examples/pi/tau2_telecom/bootstrap_vgpun.sh
source examples/pi/tau2_telecom/common.sh
"$TAU2_PI_PYTHON" examples/pi/tau2_telecom/prepare_data.py \
  --output-dir "$DATA_DIR" --max-tasks 1
"$PYTHON_BIN" examples/pi/tau2_telecom/download_model.py \
  --repo-id Qwen/Qwen3-0.6B --output-dir "$STUDENT_MODEL"
```

Hugging Face 不通时，可通过官方 ModelScope 源下载，下载工具单独运行，不改变训练锁：

```bash
UV_CACHE_DIR="$PI_WORK_ROOT/.cache/uv" "$PI_WORK_ROOT/tools/uv/bin/uv" run --no-project \
  --python /root/miniconda3/bin/python --with modelscope==1.34.0 \
  python examples/pi/tau2_telecom/download_model.py --source modelscope \
  --repo-id Qwen/Qwen3-0.6B --output-dir "$STUDENT_MODEL"
```

模型下载记录实际来源、revision 和逐文件 SHA-256；ModelScope 下载额外核对官方文件哈希。
Tau2 使用仓库的 task/allowlist/DB/evaluator。
solo 模式无需用户模拟器服务或外部 LLM 凭据。常用覆盖包括 `PI_WORK_ROOT`、
`PYTHON_BIN`、`TAU2_PI_PYTHON`、`PI_NODE_BINARY`、`STUDENT_MODEL`。

## 检查和单卡训练

```bash
source examples/pi/tau2_telecom/common.sh
"$PYTHON_BIN" -m pytest -q tests/experimental/agent_loop/test_pi_agent_loop_on_cpu.py
npm --prefix verl/experimental/agent_loop/pi/sidecar run check
npm --prefix verl/experimental/agent_loop/pi/sidecar test
PI_RUN_DIR=/root/autodl-tmp/runs/pi-grpo-smoke \
  bash examples/pi/tau2_telecom/run_single_card.sh
"$PYTHON_BIN" examples/pi/tau2_telecom/verify_run.py \
  /root/autodl-tmp/runs/pi-grpo-smoke --require-learning-signal
```

启动器先运行真实 Pi/Tau2 CPU lifecycle probe：一次 canonical read tool、后续 turn、
evaluator 和 session completion。该 probe 的生成回复是脚本指定的，只验证 SDK/环境；
后续 RL 进程每次生成均使用 student vLLM，二者的证据分开记录。

默认：1 GPU，Qwen3-0.6B，1 个 task × 2 rollouts，最多 6 次 Pi 模型调用，
每次生成最多 512 tokens、prompt 最多 24576 tokens，1 次训练更新及终局验证。
`PI_MAX_TURNS`、`PI_ROLLOUT_N`、`PI_MAX_PROMPT_LENGTH`、`PI_MAX_RESPONSE_LENGTH`
可调，启动器的其余参数作为 Hydra override 传入。

若复合任务的少量样本全失败，可额外用 canonical `small` split 做优化器诊断。
该结果只说明训练链路，不作为正式 train/test 成功率；数据 manifest 和 trace 会记录
`source_split=small`。例如以下单故障任务仍使用真实 Pi、student 生成和原始 evaluator：

```bash
"$TAU2_PI_PYTHON" examples/pi/tau2_telecom/prepare_data.py \
  --output-dir "$PI_WORK_ROOT/data/pi-tau2-small-smoke" --train-split small \
  --train-task '[service_issue]airplane_mode_on[PERSONA:None]' --max-tasks 1
PI_DATA_DIR="$PI_WORK_ROOT/data/pi-tau2-small-smoke" PI_ROLLOUT_N=8 PI_MAX_TURNS=4 \
  PI_RUN_DIR="$PI_WORK_ROOT/runs/pi-grpo-small-smoke" \
  bash examples/pi/tau2_telecom/run_single_card.sh
"$PYTHON_BIN" examples/pi/tau2_telecom/verify_run.py \
  "$PI_WORK_ROOT/runs/pi-grpo-small-smoke" --require-learning-signal
```

## 验收证据

- `preflight.json`：canonical prompt 长度、tools、生命周期、依赖与 Git revision。
- `pi-traces/*.jsonl`：每条真实 RL 轨迹的请求、原始 token/logprobs、tool results、
  evaluator、完成事件；失败也写入 `rollout_error`。
- `train.log`、`metrics.jsonl`、`exit_code.txt`：训练 loss、梯度、reward、优势、验证和退出状态。
- `verification.json`：事件/token/梯度/checkpoint 审计，单独标明是否具有非零学习信号。
- `checkpoints/global_step_1`：actor 模型、优化器及训练恢复状态。
- `rollouts/`、`validation/`：verl 原生输出。

需要分别核实工具执行、任务奖励、有效训练 token、有限 loss/梯度及 checkpoint。
组内零优势或零梯度必须明确报告，不能以 checkpoint 存在代替学习信号验证。
2026-09-14 已在 vGPUN 单卡验证：默认正式复合任务完成链路但组内零奖励；small split
诊断任务完成非零 GRPO 更新。详见 [实现、问题修复和完整运行结果](../../../bug&fix/2026-09-14-pi-agent-loop-resolution.md)。
问题定位记录见 `bug&fix/2026-09-14-pi-agent-loop-analysis.md`。
