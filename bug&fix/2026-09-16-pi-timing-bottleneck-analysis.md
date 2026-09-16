# Pi trajectory 时间瓶颈：测量前的问题定位

## 请求与当前证据

用户希望在 vGPUN 测量真实轨迹中交替发生的 LLM generation、工具执行与其他等待，得到秒级时间线及占比。

2026-09-16 只读检查：远程 `/root/autodl-tmp/code/verl-pi` 位于 `feat/pi-agent-loop`，HEAD 为 `4081284703306e353242b897d05842b2e7a1ca37`；本地 HEAD 为 `7cb87d7cc1a8798a923be5725a9922620d788bdb`。两边工作区均干净。远程单卡 RTX 4080 SUPER 报告 32760 MiB，当前 GPU utilization 为 0%，无活动 verl/vLLM 作业。现有 Qwen3-0.6B、Qwen3-1.7B 及 canonical Tau2 数据和环境。

远程旧结果属于 09-14 实现，不能证明 09-15 的增量 token 路径已经运行通过。测量前先同步当前源码并在远程回归。

## 无法直接从旧日志得到的量

1. `PiAgentLoop._generate` 只保存生成请求总耗时；包含客户端路由、排队、RPC 和推理，不是纯 GPU kernel 时间。
2. `AgentLoopMetrics.tool_calls` 没有填写；默认 0 不代表环境执行为零。
3. JSONL trace 没有完整的接收时刻及逐工具执行区间，不能用事件顺序恢复时间。
4. 每个 trajectory 的 wall time 与并发 rollout 的 wall time 不同。跨轨迹相加会重复计算重叠时间；工具并行时也必须取区间并集。
5. Node 启动、extension/Tau2 环境初始化、tokenization、评估、清理、Ray/Worker 初始化都可能影响首条轨迹，不能统称为 tool wait。

## 测量方案及边界

- 仅在本地编辑；commit + push 后远程 fast-forward 拉取。依赖检查、测试、训练及报告脚本只在 vGPUN 执行。
- Python trace 添加 wall-clock 时间和单调时钟耗时，区分 prompt tokenization、LLM request、解析、回复、事件等待、子进程启动和清理。
- Node 订阅真实 Pi `tool_execution_start/end`，记录工具名、ID 和区间；不伪造工具、生成回复或环境延迟。评估单独计时。
- 使用现有 Qwen3-1.7B 和 canonical Tau2 任务，保留原生 ManagerTQ、WorkerTQ、PiAgentLoop、vLLM 与终局 evaluator。
- 同时采样 GPU utilization 及可用的 vLLM running/waiting 指标；按轨迹和并发 rollout 的区间并集分别汇总。
- 单轨迹的 `LLM request wall / total wall` 只能称为请求耗时占比，不能写成 GPU useful utilization。GPU 采样也不等同于 kernel profiler 的精确有效计算比例。
- 报告保留模型、任务、并发数、输出 token 数及冷启动条件；不把简短 Tau2 工具的结果推广到 bash/python/test 工作负载。

本文件记录测量缺口和拟实施方案；结果需由后续远程实测补充。
