# Pi 首次 LLM 请求前的细分计时结果

## 实验与结论

远程 `/root/autodl-tmp/runs/pi-startup-20260916-b`，执行提交 `7957b209`，Tau2 `fe8ac676`，2026-09-16 完成。原生 v1 sync GRPO，Qwen3-1.7B，单张 RTX 4080 SUPER，4 条训练轨迹 + 1 条验证轨迹，最多 6 turns、每轮 256 tokens。真实 Pi/canonical extension/tools/evaluator；所有 GPU 生成来自模型，CPU scripted preflight 排除。训练与 profiling 退出码均为 0，结束后 GPU 0%、1 MiB。

验证轨迹首次请求被 Worker 收到在 12.018454 s，Worker 开始调用 LLM client 在 12.090402 s。其中 `import tau2` 单独占 9.642981 s，约占首次提交前时间的 79.8%。`createAgentSession`、extension load 和环境 set_state 都不是主要耗时。

完整逐轨迹区间见 [startup measurements](2026-09-16-pi-startup-measurements.md)。父区间包含子区间，不能把 inclusive 时间全部相加。

## 单条验证轨迹的实际顺序

Session：`485685bf984e465692ac40468ad623aa`。

```text
Python 启动 Node → ready 握手             0.259461 s
↓
Pi coding-agent SDK import                1.103745 s
↓
DefaultResourceLoader ctor                0.001436 s
↓
ResourceLoader.reload                     0.027630 s
  └─ extension load（包含在 reload 中）    0.011452 s
↓
ModelRuntime.create                       0.018469 s
↓
createAgentSession                        0.008877 s
↓
bindExtensions / session_start           10.573148 s
  ├─ Python 启动 / RPC 等余项             0.089601 s（扣除子区间后的时间）
  ├─ import tau2                          9.642981 s
  ├─ import pi_bridge                     0.001745 s
  ├─ 初次 get_environment                 0.054869 s
  │    └─ TelecomEnvironment.__init__      0.000010 s
  ├─ describe_tools                       0.059505 s
  └─ load_task                            0.724277 s
       ├─ 读取并解析 task catalog          0.443015 s
       ├─ 新建任务 get_environment         0.231055 s
       │    └─ TelecomEnvironment.__init__ 0.000011 s
       ├─ environment.set_state           0.000403 s
       │    └─ 执行任务初始化动作          已计入 set_state
       └─ describe_tools / tool_count     0.048291 s
↓
canonical task prompt 发布                t = 11.995035 s（时刻）
↓
session.prompt → 首次 provider 回调       0.014561 s
  包含 skill/template 展开和 Pi 调用准备
↓
provider messages/tools 转换              0.000503 s
↓
generation_request 被 Worker 收到         t = 12.018454 s
↓
prompt tokenization                       0.065930 s
↓
LLM request submitted                     t = 12.090402 s
```

未覆盖的小间隙合计约 0.016643 s；包括 Python/Node 事件分发、IPC、采样代码边界等。上图省略少量几十微秒的探针安装与 bookkeeping，不应按四舍五入后的条目反推精确总和。Python duration 使用 perf_counter，Node 使用 performance.now；Unix 时间仅用于同一远程机器上跨进程对齐。

## init_state 与 prompt 的测量口径

- 当前 Pi solo bridge 不创建 Tau2 LLMAgent/UserSimulator，所以独立 agent/user simulator `init_state()` 没有被调用。初始化在新建 environment、`set_state`、initialization actions 中发生。
- 不能把 `get_environment` 整体耗时称为裸 `TelecomEnvironment.__init__` 耗时。工厂函数还读取 DB/policy、创建 toolkit、设置 solo mode。
- canonical prompt publication 是事件时刻，并未单独测量字符串格式化函数；prompt preparation 是进入 `session.prompt` 到 provider 回调的区间，不包括后续 tokenization。
- `LLM submitted` 定义为 Worker 进入 `server_manager.generate` 等待点，不是 GPU kernel 开始时刻。

## 并发轨迹的复现情况

| 训练 session | 首次 LLM submit s | Tau2 package import s | Pi SDK import s | set_state s |
|---|---:|---:|---:|---:|
| 669fa39e | 8.743118 | 5.970452 | 1.192525 | 0.001343 |
| 88703adf | 9.779503 | 7.048485 | 1.171349 | 0.000414 |
| 951f9761 | 8.288040 | 5.983773 | 1.194780 | 0.000316 |
| d25c5326 | 8.865074 | 6.131698 | 1.158998 | 0.000418 |

工具短、package import 长的结论在五条轨迹中一致。不同任务、冷暖状态与执行时刻未控制，不能用验证轨迹和训练轨迹的差值计算并发加速比。六轮截断、零任务奖励仅适合本次时延诊断，不代表任务成功率或训练改进。

## 后续实验依据

Ray Harness 原型优先复用已导入 Tau2 的 Python runtime，每个 lease 仍由 canonical `serve()` 新建 bridge/environment。暂不复用 Node 或 Pi AgentSession，避免把状态隔离与 runtime amortization 同时改变。Pool 的准备成本和稳态收益分别记录，见 [资源池实验设计](2026-09-16-ray-harness-pool-analysis.md)。本报告证明瓶颈定位，不预先承诺性能收益。

## 复现与验证

```bash
PI_STARTUP_PROFILE=1 \
PI_RUN_DIR=/root/autodl-tmp/runs/pi-startup-new \
bash examples/pi/tau2_telecom/run_timing.sh
```

从 vGPUN 的 verl-pi 根目录执行。结果包括 `startup-summary.json`、`startup-report.md`、`startup-traces/`、`pi-traces/` 及 GPU/server 采样。首次检查 33 个 Python 测试通过、2 个需显式模型路径的测试跳过；2 个真实 Pi SDK 测试、Ruff 及 Node/Bash 语法检查通过。代码只在本地编写并 commit/push，远程仅拉取和执行。
