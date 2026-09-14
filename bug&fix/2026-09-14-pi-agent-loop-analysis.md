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
