# Ray Harness 资源池：基于细分计时的实验设计

## 已确认的瓶颈

`pi-startup-20260916-b` 完成原生单步 GRPO，退出码 0。四条训练轨迹 Tau2 package import 分别约 5.970、7.048、5.984、6.132 s；验证轨迹 9.643 s。验证首次 LLM submit 12.090 s，其中 package import 占约 79.8%。ResourceLoader ctor 约 1.4 ms、reload 约 27.6 ms、extension load 约 11.5 ms、createAgentSession 约 8.9 ms、set_state 约 0.4 ms。agent/user simulator init_state 未调用。

每轨迹 Node ready 和 Pi SDK import 仍需约 0.26 + 1.10 s，但最大的重复成本是 Python 包导入。资源池实验首先复用已导入 Tau2 的 Python interpreter，而不是缓存任务状态或改变 Agent 推理循环。

## 最小实现与语义边界

- 一个命名 Ray Pool Actor 管理有界会话租约；若干 CPU Harness Actor 各持有一个常驻 Tau2 Python 服务。
- 每条轨迹仍启动独立 Node/Pi session，canonical extension 保持原样。它使用的 Python stdio bridge 被轻量本机 Unix socket 代理连接到同一个 Harness Actor 的常驻解释器。
- 每次连接仍调用原始 `pi_bridge.serve()`，由它创建全新的 `TelecomPiBridge`、environment、任务目录缓存和工具调用记录。只复用 import 后的模块，不复用 bridge/environment 对象。
- PiAgentLoop 的生成、增量 token、reward 与 trajectory 输出保持原路径；可配置选择原来的 subprocess transport 或 Ray transport。默认仍为原路径。
- 同一槽位一次只服务一条轨迹；关闭时销毁 Node/代理，等待 bridge 连接结束后才归还槽位。取消、启动异常和脏槽位需明确收回或隔离，不重放已经执行过的工具。
- Unix socket 只用于 Actor 内的同机子进程；Worker 与 Harness Actor 使用 Ray RPC，因此不要求 Worker 和 Harness 在同一机器。本次仅验证单节点，不声称多机扩展已验证。

## 对照与成本口径

先验证新旧 transport 的真实 Pi/tool/evaluator 生命周期及任务重置。吞吐实验使用原生 `LLMServerManager`/vLLM、原生 Worker 的 `_run_agent_loop` 与 PiAgentLoop，只在 benchmark 中省略训练端后处理和 actor 更新，以测 rollout 本身。不会把该结果称为完整训练 step 加速。

固定 Qwen3-1.7B、任务、并发 4、最多 6 turns、256 response tokens、greedy sampling。GPU 先单独预热；新旧模式交替运行多个 batch。Pool 创建/import 成本单独计量，并在累计成本中加回。保存首请求延迟、batch wall、请求区间并集、输出 tokens、reward/error/truncation，以及每槽位 PID/服务次数；不只报告热池速度。

这是实验功能，不改 Tau2 SDK/业务源码，不改变同步策略版本语义。实际收益和未覆盖边界需在结果文档中据实记录。

## 首次对照入口异常（修复前记录）

`pi-harness-20260916-c` 的 vLLM standalone 初始化完成，但 BenchmarkWorker 在首批 GPU warmup 创建轨迹之前报 `KeyError: 'pi_agent'`。原生 Worker 构造时向其模块注册表加载了 YAML，而定义在脚本入口的实验 Actor 将 `from ... import _agent_loop_registry` 的字典作为序列化全局值携带，读到旧副本。需要在 Actor 方法执行时从原生模块读取 registry，避免跨 Ray 序列化复制可变注册表。该次没有任何可用于速度比较的完成轨迹，不纳入性能结果。GPU 已释放为 0%、1 MiB。

## 独立推理权重加载配置异常（修复前记录）

`pi-harness-20260916-d` 的启动日志显示 `load_format=dummy`。默认训练 rollout 随后由 actor 同步权重，但独立 benchmark 没有该步骤。因此即使它能生成，也不能作为真实 checkpoint 的性能证据。发现后停止这一个自有实验，显式覆盖 rollout.load_format 并增加启动配置断言，然后使用新目录重跑。c/d 两轮均不进入有效结果。
