# Pi coding-agent 接入标准 verl：解决方案与验证记录

## 目标和实现边界

基于标准 verl v1 sync trainer 接入真实 Pi coding-agent。Pi 管理 agent/tool loop，verl 管理 student generation、GRPO 优势和 FSDP2 参数更新。首期只支持文本 GRPO；多段 OPD、GAE/GiGPO、异步权重更新显式拒绝，避免以不完整语义训练。

所有源码在本地编写并提交到 `feat/pi-agent-loop`，运行和验证全部在 ssh vGPUN。原始分析和每个失败的先行定位记录见 [analysis](2026-09-14-pi-agent-loop-analysis.md)。

## 调用链和数据契约

1. 数据集提供 canonical Tau2 task ID；真实任务 prompt、skill 和 tool schemas 由 Tau2 training extension 发布。
2. `PiAgentLoop.run` 为一条轨迹启动一个独立 Node sidecar。sidecar 使用 Pi SDK `createAgentSession`，只加载指定 canonical extension，禁用自动压缩与模型重试。
3. Pi 的每个 generation request 经 JSONL 传到 Python；Python 使用 verl continuous token builder 编码实际上下文，经 LLMServerClient 调 student vLLM。
4. Hermes parser 把生成回复解析为 Pi assistant/tool-call message，工具由真实 Pi SDK 执行。Pi 完成 turn 后发送 generation ID、assistant message 和 tool results。
5. recorder 保留每次实际 prompt IDs、response IDs、rollout logprobs；不会从最终 transcript 反向编码训练目标。工具结果进入下一段 prompt，response mask 只覆盖模型采样 token。
6. 必须满足 generation/turn 一一对应、有限且对齐的 logprobs、终局 evaluator 和 session completion，才返回原生 `list[AgentLoopOutput]`。超时、进程失败、缺失评测直接报错。
7. verl v1 以 `uid/session/index` 存储多段轨迹，按每个 session 最后一段的终局奖励计算组内 GRPO 优势，再广播到该 session 的所有生成段。

实现入口：`verl/experimental/agent_loop/pi_agent_loop.py`；协议与 recorder：同级 `pi/`；配置、数据准备、启动和验收：`examples/pi/tau2_telecom/`。

## 已解决的运行问题

### 依赖安装和下载路径

新 vGPUN 容器只有基础 Python/Torch，未准备 Node、uv、模型和项目环境。使用数据盘中的独立 uv 环境复用仓库 `uv.lock`，避免覆盖系统 Python。Tau2 单独使用自己的 Python 环境。

GitHub 大文件资源直连超时后，仅对依赖安装进程使用已有 SSH HTTP 转发，FlashAttention wheel 和锁定 CUDA 依赖下载完成。Hugging Face 的直连与代理均失败，模型显式使用 [Qwen 官方 ModelScope 仓库](https://modelscope.cn/Qwen/Qwen3-0.6B)；工具使用 ModelScope SDK 1.34.0，记录实际源、revision 和文件 SHA-256，并与官方 API 哈希逐项核对。

实际环境：Torch 2.11.0+cu130、vLLM 0.24.0、Transformers 5.9.0、Ray 2.55.1、TransferQueue 0.1.9.dev0、Node 22.22.0、Pi SDK 0.84.4。GPU 为单张 RTX 4080 SUPER，32760 MiB；驱动 595.71.05。

Tau2 固定为 `fe8ac67662d9ba765c7324ed732872d362133895`。Qwen3-0.6B 的 `model.safetensors` 为 1503300328 字节，SHA-256 为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。

### JSON Schema 兼容

Pi canonical tool 的参数可以含 `anyOf` 等完整 JSON Schema，verl 现有 typed tool property 要求 `type`。Hermes parser 不依赖该 typed schema，因此 Hermes 路径保留原始 schemas 供 tokenizer 使用，解析生成文本时不做不必要的窄化转换。适配器测试覆盖该情况。

### 预检长度统计

Transformers 5.9 的 `apply_chat_template` 默认返回 BatchEncoding。最初预检统计到两个字段而非 token 数；改为复用 verl 的 `normalize_token_ids`，并验证结果为平坦整数序列。实际训练 builder 原本已有正确处理。修正后 canonical 两轮 prompt 长度为 13129 / 13198，低于 24576 上限。

### 单卡 TransferQueue 资源配置

首轮配置 Ray 8 CPU，但默认 SimpleStorage 需要 8 个各占 1 CPU 的 storage actor，controller 还占 1 CPU。Ray 显示整体 placement group 持续等待，GPU 未开始使用。单卡示例显式缩减为 1 个 storage unit，原生 TQ 算法不变。首轮目录保留供诊断；修正后的运行使用新目录，已成功推进到 actor/FSDP2 初始化。

## 已完成验证

- 远程 Python 编译、Node syntax、Bash syntax、Ruff check/format 通过。
- Pi adapter 与原生多轨迹/TQ 相关 Python 测试：14 passed，包含配置注册后延迟导入、连续 Hydra 实例化、canonical 工具失败标志回归。
- 真实 Pi SDK 测试：2 passed，包括真正执行 fixture tool，以及缺失 evaluator 时 fail closed。
- 真实 Pi + canonical Tau2 probe：43 个工具、2 次 generation、2 个完成 turn、实际执行 `check_apn_settings`、收到 evaluator 与 completion；该 probe 使用脚本生成回复，只验证生命周期，不作为 RL 结果。
- 模型文件 SHA-256 校验通过。
- PEP 517 wheel 重新构建通过；zip 清单确认包含 `pi/sidecar/main.mjs` 和 `package.json`。

## 单卡 RL 结果

Run `pi-grpo-smoke-20260914-c`（commit `e6ace235`）已正常完成，退出码 0：

- 1 张 GPU，Qwen3-0.6B，同一训练任务 × 2 条真实 Pi/student 轨迹；分别 2 / 1 轮，共 69 个采样 token，1 次训练轨迹工具执行。
- 额外测试任务 1 条轨迹，2 轮、65 个采样 token、1 次工具执行。
- 真实 response IDs、logprobs、response masks 和 generation/turn/evaluator/completion 事件通过审计。
- 完成一次原生训练更新调用、终局验证、actor/optimizer checkpoint；step 用时 68.42 秒，验证 30.65 秒。
- 训练组 reward 为 `[0, 0]`，advantages、pg_loss、grad_norm 均为 0。**仅证明完整链路可运行，尚不证明非零学习更新。** 严格验收 `--require-learning-signal` 因这一项失败。

Run `pi-grpo-1.7b-20260914-d` 使用 Qwen3-1.7B、同一正式训练任务 × 4 条轨迹、8 轮上限，也已完成，退出码 0：

- 训练轨迹分别为 3 / 8 / 3 / 8 轮，共 22 轮、1229 个采样 token、20 个工具结果。原始 trace 按两种错误标志复核，工具错误 9 次。
- 测试轨迹 8 轮，151 个采样 token、8 个工具结果，达到轮数上限。
- 四条训练轨迹 reward 全为 0，梯度和 loss 为 0。严格验收仍只在非零学习信号一项失败。
- 未发现 token/logprob/mask 对齐、Pi 生命周期、trainer 退出或 checkpoint 缺失；该批用于完整多轮链路检查，不作为任务能力提升证据。
- actor 最大 allocated/reserved 显存约 19.72 / 21.78 GiB，完成单卡运行。

Run `pi-grpo-small-20260914-e`（运行代码 commit `a02c0014`）完成非零更新的严格验收：

- 单张 RTX 4080 SUPER，Qwen3-0.6B；canonical `small` split 的 `[service_issue]airplane_mode_on[PERSONA:None]`，同一任务采样 8 条真实 Pi/student 轨迹，最多 4 轮。
- 奖励为 `[1,1,1,0,1,1,0,1]`，共 14 次模型调用、422 个真实训练 response token、6 次工具执行，工具错误 0。generation、turn、evaluator、completion 一一完整；原始 token、logprobs 和 mask 审计通过。
- 原生 GRPO advantages 范围 `[-1.62018156, 0.54006052]`；`actor/pg_loss=-0.24499539`，`actor/grad_norm=28.28125`，`actor/lr=1e-6`。loss 和梯度有限且非零。
- `global_step=1`，actor 模型、optimizer、extra state、data state 和 latest checkpoint 标记均落盘。额外正式测试任务完成 2 轮、54 个采样 token、1 次工具执行，reward 为 0。
- trainer 退出码 0；`verify_run.py --require-learning-signal` 返回 0，`passed=true`、`learning_signal=true`、`failures=[]`。
- step 用时 103.38 秒，其中 actor update 35.33 秒、checkpoint 4.50 秒；额外验证 34.29 秒。canonical 预检两轮 prompt 为 13099 / 13168 tokens。
- 训练结束后未见本任务 Ray/Node/driver 进程，GPU 利用率 0%，显存恢复至 1 MiB。

**结论：真实 Pi 多轮执行、生成 token 记录、标准 verl GRPO 分组、非零梯度更新和 checkpoint 在单卡已验证。** Small split 结果只是优化器诊断，6/8 是同一任务的采样结果，不能当作任务集成功率。正式复合任务的零奖励结果仍保留；本次不证明模型能力提升或多段 OPD 可用。

最终通过运行的复现命令（在 vGPUN 的仓库根目录）：

```bash
source examples/pi/tau2_telecom/common.sh
"$TAU2_PI_PYTHON" examples/pi/tau2_telecom/prepare_data.py \
  --output-dir /root/autodl-tmp/data/pi-tau2-small-smoke --train-split small \
  --train-task '[service_issue]airplane_mode_on[PERSONA:None]' --max-tasks 1
STUDENT_MODEL=/root/autodl-tmp/models/Qwen3-0.6B \
  PI_DATA_DIR=/root/autodl-tmp/data/pi-tau2-small-smoke PI_ROLLOUT_N=8 PI_MAX_TURNS=4 \
  PI_RUN_DIR=/root/autodl-tmp/runs/pi-grpo-small-20260914-e CUDA_VISIBLE_DEVICES=0 \
  bash examples/pi/tau2_telecom/run_single_card.sh
"$PYTHON_BIN" examples/pi/tau2_telecom/verify_run.py \
  /root/autodl-tmp/runs/pi-grpo-small-20260914-e --require-learning-signal
```

复跑时应使用新的 run 目录保留原始证据。修改采样规模或 seed 后的具体奖励分布可能不同。

Run b 的第一条轨迹完成后，第二条因配置缺失失败；虽然标准 trainer 曾写入 checkpoint，审计仍因不完整组、缺少最终指标和非零退出码判定失败，未作为通过结果。

### Pi 类延迟导入覆盖 YAML 配置

标准 worker 先把 YAML 存入 registry，Hydra 首次实例化时才导入 Pi 模块。原来的 `@register` 会把该 entry 无条件覆盖成只有 `_target_` 的字典，使同组后续实例丢失 `cwd/training_extension`。Pi 现只通过 YAML 注册，去掉重复装饰器，不修改标准 registry；回归测试与 Run c 的两条轨迹均已验证修正。

### canonical 工具错误标志与打包

canonical wrapper 可能在 `isError=false` 时用 `details.error=true` 表示 Python 工具失败。错误计数与预检现在统一检查两种标志，保留原始反馈和 evaluator reward。

PEP 517 使用 pyproject 的 package-data，单独修改 fallback setup.py 不会包含 Node 文件。已在两份配置中列入 sidecar 主文件和 package.json，远程构建产物清单验证通过。构建显式指定训练环境 Python，避免 uv 额外选择解释器。

GitHub 拉取多次超时后，使用本地已 commit/push 的同一提交生成增量 Git bundle，经 SSH 传到 vGPUN，由远程 `git bundle verify` 和 `git pull --ff-only` 应用，commit hash 保持一致；没有在远程手写或修补源码。

## 远程文件

- 代码：`/root/autodl-tmp/code/verl-pi`
- 训练环境：`/root/autodl-tmp/envs/verl-pi`
- Tau2 环境：`/root/autodl-tmp/envs/tau2-pi`
- 模型：`/root/autodl-tmp/models/Qwen3-0.6B`
- 依赖清单：`/root/autodl-tmp/verl-pi-environment.txt`、`tau2-pi-environment.txt`
- Python 测试日志：`/root/autodl-tmp/pi-unit-tests.log`
- canonical probe：`/root/autodl-tmp/pi-canonical-preflight.json`
- 首轮调度等待日志：`/root/autodl-tmp/runs/pi-grpo-smoke-20260914-a/train.log`
- 0.6B 正式任务链路：`/root/autodl-tmp/runs/pi-grpo-smoke-20260914-c/`
- 1.7B 正式任务链路：`/autodl-fs/data/verl-pi/runs/pi-grpo-1.7b-20260914-d/`
- **最终通过的非零更新**：`/root/autodl-tmp/runs/pi-grpo-small-20260914-e/`
- 最终审计：上述目录的 `verification.json`；训练指标：`metrics.jsonl`；原始事件与采样 token：`pi-traces/`
- 最终 checkpoint：`/root/autodl-tmp/runs/pi-grpo-small-20260914-e/checkpoints/global_step_1/actor/`
- 打包检查：`/root/autodl-tmp/pi-wheel-check/verl-0.10.0.dev0-py3-none-any.whl`

启动与复查命令见 [示例 README](../examples/pi/tau2_telecom/README.md)。
