# Pi agent loop 接入与单卡 RL 校验：问题梳理和定位

## 目标与范围

将相邻 `Agent-R1-opd` 仓库的真实 Pi coding-agent 接入迁移到标准 verl v1。
Pi 负责每条任务轨迹的 agent/tool loop；verl 负责生成、训练张量、优势与参数更新。
首期在 `ssh vGPUN` 上用一张 GPU 验证真实 Tau2 Telecom Pi 会话和 GRPO 更新。
所有源码、脚本在本地编写，commit/push 后在远程 clone/pull；不在本机运行项目。

## 已核实的基线

- 本地 `verl-pi`：`main`，`9e0252efba42a5442bbe52f7d0d2ac68e514581e`，开始时工作树干净。
- `verl/trainer/config/ppo_trainer.yaml` 默认 `trainer.use_v1=true`、`trainer_mode=sync`。
- `AgentLoopWorkerTQ._agent_loop_postprocess` 接受 `list[AgentLoopOutput]`，键为
  `{uid}_{session_id}_{index}`；`compute_advantage_for_multi_trajectories` 按 session
  最后一段计算 GRPO 优势并广播至该 session 的其他段。
- 旧 `PiTauAgentFlow` 用真实 Pi `createAgentSession`，JSONL 转发生成请求，
  `PiTrainingRecorder` 输出 Agent-R1 专有 `AgentFlowStep`。
- Tau2 canonical extension 是进程级状态，因此必须一条轨迹一个 Pi sidecar。

## 迁移边界及必须处理的问题

1. **基类接口不同**：旧 `apply_chat_template`、`_postprocess` 不能直接复制。
   新适配器直接记录每次生成实际使用的 prompt token 和返回的 response token，
   返回原生 `AgentLoopOutput`；不从最终 transcript 重建训练目标。
2. **奖励与事件完整性**：每个 generation 必须匹配一个完成的 turn；必须收到
   evaluator 结果和 session completion，才能提交样本。工具错误是环境反馈；
   进程退出、非法协议、超时、缺少评测是运行失败，不能填零奖励冒充有效 rollout。
3. **多段奖励**：v1 会广播 session terminal reward；首期使用 GRPO。
   GAE/GiGPO 的跨步语义不在首期范围。
4. **OPD 限制**：当前 v1 teacher 后处理只计算最后一段，不能宣称多段 OPD 已接通。
   首期配置禁用 distillation，多段 Pi loop 对开启 OPD 明确报错，避免无声漏算。
5. **token/长度边界**：保留真实 response token/logprob；工具结果放入下一轮 prompt。
   prompt 超长必须明确报错，不能删除 canonical skill/tool schema；生成预算在请求前限制。
6. **生命周期隔离**：固定 Pi 版本、只加载指定 training extension；stdout 只放协议，
   stderr 放诊断；保留 abort、超时、进程回收和终局状态。

## vGPUN 环境检查（2026-09-14）

- SSH 初次旧会话曾关闭；本次连接成功。
- GPU：1 × NVIDIA GeForce RTX 4080 SUPER，32760 MiB，总线 00000000:4C:00.0，
  初始占用 1 MiB，无运行中的 GPU 进程。
- 驱动：595.71.05，报告 CUDA 13.2。
- `/root/autodl-tmp` 是新目录，无代码或模型；未发现共享盘中的现有 Qwen 模型。
- 基础解释器：`/root/miniconda3/bin/python` 3.12.3，基础 Torch 2.8.0+cu128；
  默认 PATH 不含该解释器，未安装 uv、Node 或 rg。
- 当前 verl 要求 vLLM >= 0.18.0 和 Transformers >= 5.5.3；不会直接覆盖基础环境。
  使用独立 uv venv，依赖与模型放在数据盘，并记录实际解析版本。

## 计划实现

- `verl/experimental/agent_loop/pi_agent_loop.py`：verl v1 适配与请求处理。
- `verl/experimental/agent_loop/pi/`：协议客户端、逐轮 recorder 和可安装的 Pi sidecar。
- `examples/pi/tau2_telecom/`：数据准备、配置、preflight、单卡启动和验证说明。
- `tests/experimental/agent_loop/`：协议/奖励/token 约束及适配器测试。
- 实际发现新异常时先在本文件补充证据与定位，再修改和复测；通过后另写解决过程文档。

## 验收层次（尚未执行）

1. 远程静态检查与有意义的单元测试。
2. 真实 Pi SDK + canonical Tau2 extension：prompt、工具 schema、工具结果、evaluator、完成事件齐全。
3. 单卡 sync GRPO：有效 rollout、非空 mask、正确 session 分组、有限 loss/梯度、optimizer update。
4. checkpoint、实际训练指标与 Pi trace 落盘；报告工具/任务成功率与训练链路结果各自的证据。
5. 记录依赖版本、远程 commit、运行命令、日志与 checkpoint 路径。任何未通过项目明确保留为未完成。

## 执行中补充：远程拉取与环境版本

- GitHub `ls-remote`、PyPI/npm HTTPS 均成功；首次完整 clone 报
  `GnuTLS recv error (-110): The TLS connection was non-properly terminated`，
  安装脚本尚未开始，没有已安装环境需要回滚。先改用独立分支的浅克隆重试，
  不修改全局 Git/proxy 配置。
- 进一步核实本仓库已有完整 `uv.lock`：Torch 2.11.0+cu130、vLLM 0.24.0、
  Transformers 5.9.0，FSDP 原生轮子来自项目指定 wheelhouse。
  环境准备复用该锁和 `fsdp + vllm + test` extras，不重新解析训练依赖。
- 数据盘 50 GiB，共享盘 200 GiB，初始均为空；环境与 cache 放数据盘。
- HTTP/1.1 浅克隆成功，安装推进到锁定依赖下载。`flash-attn==2.8.3` 的
  wheelhouse GitHub release 重定向到 `release-assets.githubusercontent.com` 后连接超时，
  uv 重试五次后退出。Torch wheel 已缓存，失败发生在下载阶段而不是编译/ABI 或 GPU。
  下一步先只读验证 SSH 已有 17890 转发能否访问同一公开资源；确认后仅对本次安装进程
  使用代理，不修改任何持久 Git/proxy 配置，不换依赖版本绕过锁。
- 同一公开 wheel 经现有 17890 HTTP 转发返回 200，大小 242518291 字节。
  采用进程级代理重试下载。另在源码核对中发现 Pi 的完整 JSON Schema 可能包含
  `anyOf`，而 verl tool schema 的 property 要求 `type`；Hermes parser 本身不依赖
  tools schema，因此该路径保留 Pi 原始 schema，仅解析生成文本中的工具 JSON。

## 执行中补充：模型下载和静态检查

- Hugging Face 模型信息查询经 SSH 转发报 `SSL: UNEXPECTED_EOF_WHILE_READING`，直连 curl 也超时；同一节点访问 ModelScope 官方 API 返回 200。训练尚未启动。计划支持显式 `--source modelscope`，保留实际源、revision 和权重文件校验记录，不修改训练依赖锁。
- 新增验收脚本的一行列表推导超出 Ruff 120 字符限制；远程检查已报告具体行。按项目格式拆行后重新检查，此问题不影响算法语义。

## 执行中补充：canonical 预检长度统计异常

- 真实 Pi/Tau2 预检完成了两次 generation/turn、43 个工具 schema、一次 `check_apn_settings` 实际工具执行和 evaluator，未出现生命周期错误。
- 但报告的两轮 prompt 长度均为 `2`，与完整工具/skill 上下文不符。先定位 Transformers 5.9 的 `apply_chat_template` 返回类型，以及 verl continuous token builder 的处理方式；在厘清前不把该长度作为容量验收依据，不启动 RL。

- 已定位：锁定的 Transformers 5.9 `apply_chat_template` 默认 `return_dict=True`，`len(BatchEncoding)` 统计到两个字段。verl 实际训练 builder 已调用 `normalize_token_ids`，异常仅发生在新预检的统计代码。预检复用同一 normalize utility 并校验平坦整数序列，再跑 canonical 预检核实真实长度。

## 执行中补充：单卡启动时 TransferQueue 资源等待

- Run `pi-grpo-smoke-20260914-a` 配置校验和 Ray 启动成功，但停在 TQ 初始化，GPU 未使用。`ray status` 显示 `1/8 CPU` 已占用，另有 `CPU:1 × 8 (SPREAD)` placement group 等待。
- 根因是示例把 Ray CPU 限为 8，但继承的 SimpleStorage 默认创建 8 个各占 1 CPU 的 storage actor，controller 另占 1 CPU，整组无法调度。锁定的 TQ `simple_storage_bootstrap.py:37` 会等待整组 ready。
- 单卡 smoke 只需 1 个 storage unit；在示例中显式配置 `transfer_queue.backend.SimpleStorage.num_data_storage_units=1`，保留原生 TQ 逻辑和 8 CPU 限额。终止本次无进展的自有 driver，拉取本地修正后在新 run 目录重启，保留原日志。

- 重启前的 Git 拉取再次停滞：当前 SSH HTTP 转发访问 GitHub 报 `SSL unexpected eof`，上一轮已完成的下载不受影响。保留远程既有 commit，终止本次卡住的 Git 传输，先验证直连 HTTP/1.1，再用有低速超时限制的单次 Git 配置重试。

## 执行中补充：并行 Pi session 的注册表覆盖

- Run b 已进入真实 vLLM generation，但同组另一个 session 初始化时报 `PiAgentLoop.__init__ missing cwd and training_extension`。第一条已有 canonical request/实际生成，第二条无 trace。
- 需要检查 YAML agent config 的加载和模块装饰器注册顺序：初步怀疑 Hydra 首次导入 Pi 模块时，`@register` 用只有 target 的默认配置覆盖了已加载的完整 YAML，导致后续实例丢失必要参数。当前运行不能作为完整 GRPO 组验收；先核对调用链再修正，并补充连续实例化回归测试。

- 已确认 `agent_loop.py:445` 的 register 装饰器会无条件覆盖 YAML registry。去掉 Pi 类上的重复装饰器，使用 agent_loop_config_path 注册的完整配置；不修改标准 verl 注册器。回归测试先注册配置，再重新导入 Pi 模块，并通过 Hydra 连续实例化两次。Run b 终止于不完整组导致的空 keys，未取得合格训练结果。

## 执行中补充：链路通过但小模型批次无学习信号

- Run c 正常完成 1 个训练 step、两条真实 student/Pi 轨迹、工具执行、验证和 actor/optimizer checkpoint，退出码 0。并行配置问题已消除。
- 该批两条任务 reward 都为 0，原生 GRPO advantages、pg_loss 和 grad_norm 全为 0。Qwen3-0.6B 对复合任务未完成全部修复，不能把本次结果描述为有效学习更新。
- 保留该基线和原始 evaluator，下一轮使用 Qwen3-1.7B、同一 canonical 训练任务、4 条 rollout 和 8 轮上限，检查能否获得奖励差异及非零梯度。只是单卡训练信号 smoke，不估计任务集成功率。模型/运行目录另建，避免覆盖已通过的链路证据。

- 安装包检查中 `uv build` 未显式指定 Python 时开始下载另一个受管 Python；训练环境本身不受影响。构建改为显式 `--python /root/autodl-tmp/envs/verl-pi/bin/python`，确保使用已验证环境并避免额外解释器。

## 执行中补充：wheel 中缺失 Node sidecar

- 在已验证训练 Python 下执行 `uv build --wheel --no-build-isolation` 成功，但解包清单只有 Pi Python 文件，没有 `sidecar/main.mjs` 和 `sidecar/package.json`。源码 checkout 运行不受影响，但 wheel 安装后会缺少 Pi 运行入口。
- 定位项目实际构建后端及 package-data 来源，修正有效的打包配置后重新构建并检查 zip 清单，不把仅构建成功当作打包完整。

- 根因：PEP 517 构建实际采用 `pyproject.toml` 的 `[tool.setuptools.package-data]`，此前仅在 fallback `setup.py` 增加模式。补齐 pyproject 的同两项 sidecar 文件，不改变依赖版本或锁。
