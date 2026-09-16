# Pi trajectory 计时实现与 vGPUN 实测结果

## 结论

本次真实 Tau2 轨迹的主要等待发生在首次 generation request 之前。单轨迹验证总计 17.110 秒，其中 session 启动和初始化 12.416 秒（72.57%），六轮 LLM request 4.143 秒（24.21%），六次实际工具执行合计 0.0205 秒（0.12%）。不能将其余 75.79% 全部解释为工具执行。

这次解决的是可观测性缺口，尚未修改进程复用或性能调度。启动窗口内部的 Pi SDK import、extension 加载、Tau2 bridge 初始化仍需进一步细分。

## 环境与实验边界

- 日期：2026-09-16；远程 vGPUN，单卡 RTX 4080 SUPER，32760 MiB。
- 模型：`/root/autodl-tmp/models/Qwen3-1.7B`；真实 student vLLM 生成。
- 运行代码：`92ead41bbeb156ff5d77c57d0a2b7d2badc50f03`；最终报告脚本：`eefd0148`。
- Tau2：`fe8ac67662d9ba765c7324ed732872d362133895`；Pi SDK 0.84.4。
- 原生 `PPOTrainerSync → ManagerTQ → WorkerTQ → PiAgentLoop → Node Pi → Tau2` 链路。
- 训练为一个 canonical prompt × 四次采样；验证为另一个 canonical test prompt × 一次采样。
- 每条最多六轮，response 上限 256，prompt 上限 24576；vLLM eager 模式，max_num_seqs=4，max_num_batched_tokens=4096。
- 五条轨迹均完成协议收尾、共 30 次实际生成和 539 个 response token；均在六轮上限截断、终局 reward=0。四条训练轨迹各有一次工具错误；验证六次工具全部无错误。这是实际模型行为，不代表任务成功或非零学习更新。
- preflight 使用 scripted generation 的两轮生命周期检查不计入本报告；数据只来自 Worker 产生的 `pi-traces/`。
- 本次为两种任务上的小样本测量，不能推断 bash/python/test 等其他工作负载的工具成本，也不是控制其他变量后的并发加速比实验。

## 代表性单轨迹：验证任务

Session：`16c7207d26bb48988345215a48de7a12`。

任务：`[mobile_data_issue]data_saver_mode_on|user_abroad_roaming_enabled_off[PERSONA:Easy]`。

以下省略每轮 tokenization、解析和事件间隔，完整明细见原始报告。

```text
Session 初始化                         12.416295 s
LLM generation 1                        1.237598 s
check_data_restriction_status            0.008028 s
LLM generation 2                        0.619925 s
toggle_data_saver_mode                   0.003233 s
LLM generation 3                        0.573255 s
check_network_status                    0.002388 s
LLM generation 4                        0.588365 s
check_network_mode_preference            0.002524 s
LLM generation 5                        0.575622 s
run_speed_test                          0.002198 s
LLM generation 6                        0.547884 s
check_network_status                    0.002175 s
终局 evaluator                          0.086033 s
Node 清理                               0.021408 s
```

| 阶段 | 秒 | trajectory wall 占比 |
|---|---:|---:|
| 启动到第一个 generation request | 12.416295 | 72.57% |
| LLM request（六轮合计） | 4.142649 | 24.21% |
| 工具执行（六次合计） | 0.020545 | 0.12% |
| Tokenization | 0.291602 | 1.70% |
| 终局评估 | 0.086033 | 0.50% |
| Node 清理 | 0.021408 | 0.13% |
| 解析、发送和其他间隔 | 0.131309 | 0.77% |
| 总计 | 17.109839 | 100% |

其中 Node 创建到 ready 握手仅 0.376889 秒，首次请求前剩余约 12.039 秒发生在握手后的 session 准备窗口。不能把全部 12.416 秒归因于操作系统创建子进程。

## 四条并发训练轨迹

| Session 前缀 | 总耗时 s | 初始化 s | LLM request s | 工具执行 s |
|---|---:|---:|---:|---:|
| 89c4bfb6 | 16.925074 | 9.643592 | 6.644207 | 0.018122 |
| 99a17983 | 17.173169 | 10.426320 | 6.076340 | 0.018153 |
| d7c4cff6 | 17.464954 | 9.594718 | 6.802285 | 0.017739 |
| da799d79 | 17.182309 | 8.836045 | 7.696867 | 0.017318 |

四条轨迹的首次请求前时间占各自 wall 的约 51–61%。六轮工具只占约 0.10–0.11%。首轮 LLM 请求比后续更慢，运行日志记录首次推理 Triton kernel JIT；计时未进一步分离 JIT、路由、排队与实际 kernel 时长。

并发组从最早 session 开始到最后 session 清理结束共 17.651182 秒。四条 trajectory 的 LLM request 时长相加为 27.219700 秒，但重叠区间取并集后只有 8.079329 秒，请求覆盖率为 45.77%，无这些请求的时间占 54.23%。

## GPU 与排队观测

| 测量窗口 | 单轨迹验证 | 四条并发训练 |
|---|---:|---:|
| trajectory/group wall s | 17.109839 | 17.651182 |
| 有至少一个 LLM request 的时间占比 | 24.21% | 45.77% |
| nvidia-smi utilization 采样均值 | 11.44% | 10.55% |
| utilization 非零的采样点占比 | 27.91% | 21.35% |
| GPU 采样点数 | 86 | 89 |
| 采到的 vLLM running 最大值 | 1 | 4 |
| 采到的 vLLM waiting 最大值 | 0 | 3 |

GPU 按约 200ms 读取，但 NVIDIA utilization 本身有统计窗口；服务指标也异步更新。它们不能解释为精确 kernel busy time 或有效计算比例。LLM request 包含客户端、路由、排队、RPC、prefill、decode 等，不能将其 24.21% / 45.77% 写为 GPU utilization。

跨轨迹请求及同轮并行工具用区间并集，避免重复计时。Python 耗时采用 perf_counter，Node 采用 performance.now；跨进程对齐使用同主机 wall clock。Node wall timestamp 毫秒精度带来极小区间交叠，最大记录为约 0.000034 秒，不影响上述秒级结论。

## 整个训练 step 的其他成本

原生 trainer 指标：`timing_s/gen=18.228371`，`old_log_prob=25.307583`，`update_actor=95.222411`，`adv=0.587338`，`update_weights=2.815331`，`step=153.395832`；额外 `testing=18.301014`。

所以如果关心完整 RL step 的吞吐，actor 更新的约 95 秒大于这批 rollout 的约 18 秒。该口径与上述单 trajectory 时间不同，不能直接相加或相互替代。也未把整个 Ray/vLLM 冷启动成本计入单 trajectory。

## 改动与验证

1. `PiAgentLoop` 的 trace 记录时间戳与独立阶段；仍保留原始 token IDs、logprobs 和增量上下文，未改变生成/工具顺序。
2. sidecar 订阅真实 Pi 工具开始和结束事件，在 `step_complete` 添加工具时间，终局 evaluator 单独计时。
3. `run_timing.sh` 运行原生单卡流程，`profile_timing.py` 采样 GPU 和 vLLM 指标并按轨迹及 prompt 组生成报告。
4. 远程 34 个 Python 回归通过、2 个真实 Pi SDK 回归通过；Node/Bash 语法检查通过。Ruff check 与 format check 最终全部通过；最终区间并集测试重跑通过。
5. trainer 和 profiling launcher 退出码均为 0。结束后本次 driver/Node/monitor 进程已退出，GPU utilization=0%，显存恢复 1 MiB。

源码在本地修改、commit/push，再由远程 Git bundle fast-forward 拉取；所有依赖/语法/测试/训练/统计执行均在 vGPUN。

## 证据与复跑

- [完整逐轨迹、逐轮明细](2026-09-16-pi-timing-measurements.md)
- [结构化时间区间、工具状态与硬件采样汇总](2026-09-16-pi-timing-measurements.json)
- [测量前分析与实施边界](2026-09-16-pi-timing-bottleneck-analysis.md)
- 原始远程目录：`/root/autodl-tmp/runs/pi-timing-20260916-a/`，包含 `pi-traces/`、`gpu-samples.csv`、`server-samples.jsonl`、`train.log`、`metrics.jsonl` 和退出码。

远程复跑命令（务必使用新目录）：

```bash
cd /root/autodl-tmp/code/verl-pi
PI_RUN_DIR=/root/autodl-tmp/runs/pi-timing-new \
  bash examples/pi/tau2_telecom/run_timing.sh
```
