# Pi 跨轮增量 token 实现记录

## 状态与问题来源

状态：本地实现和回归用例已编写，尚未运行验证。用户要求“暂时先本地修改吧”，本次没有
运行 Python/Node 项目代码、测试、preflight 或 GPU 训练，没有重新开启 vGPUN，也没有
提交、推送或同步远程。本文记录实现与验收计划，不是运行通过报告。

修复前的定位见 [历史 tokenization 核查](2026-09-15-pi-history-tokenization-audit.md)。
旧 Pi 路径每轮完整编码 messages/tools，包括历史 assistant；尽管每个独立训练片段保留了
当次实际 prompt、response 和 logprobs，跨轮历史仍可能因为 BPE 重新分词、工具 JSON
重新序列化而发生变化。本次参照原生 ToolAgentLoop，复用其 Continuous Token builder。

## 数据流与不变量

记 `P_t` 为第 t 次实际生成输入、`A_t` 为该次原始采样 IDs，`C_t` 为新增上下文 tokens：

```text
首轮：Pi messages/tools -> build_initial_tokens -> P1 -> vLLM -> A1
                                     保存 P1/A1/logprobs1

续轮：P1 + A1 + 原生模板边界 + Encode(新增上下文) -> P2 -> vLLM -> A2
                                     保存 P2/A2/logprobs2

训练片段 1：P1 + A1，response_mask 只覆盖 A1
训练片段 2：P2 + A2，response_mask 只覆盖 A2
```

要求 `P_(t+1)` 以 `P_t + A_t` 完整开头。模型生成 `he + llo`，即使解码成的 `hello`
重新编码会选中单个 token，续轮仍保留 `he + llo` 的原始 IDs。工具结果、extension
控制消息和模板新增边界走 tokenizer，但不能替换已有 token。

这项改动保障本桥接路径的跨轮 token 保留；它没有验证任意生产 provider 的消息重编码
接口与这里逐 token 相同，也没有产生新的长期收敛结论。

## 具体实现

### 1. 每个 rollout 独立维护 PiTokenContext

`verl/experimental/agent_loop/pi/token_context.py`：

- `build_prompt`：首轮调用 `build_initial_tokens`。后续检查完整 tools 与消息前缀，再
  调用原生 `merge_context_tokens`。返回 prompt 的独立副本。
- `record_generation`：通过 `merge_assistant_tokens` 追加真实 `response_ids`，并保存
  generation ID 和发给 Pi 的 text/tool_calls，以便验证消息来源。
- `complete_turn`：核对 generation ID、assistant 文本和工具调用，保存该 assistant 的
  OpenAI 消息投影；工具参数按 JSON 对象比较，允许 Python 与 JS 的空格、转义差别。

状态顺序是 `prompt -> response -> turn -> prompt`。上一轮未完成时不能开始新一轮。
下一轮必须包含该 assistant 之后的新上下文。消息、工具定义和生成结果都保存深拷贝，
不让调用方后续修改输入对象改变既有历史。

### 2. 明确哪些 assistant 对应模型采样

`verl/experimental/agent_loop/pi/sidecar/main.mjs` 在 `step_complete` 新增
`assistant_message_openai`，使用发送下一轮 generation_request 的同一个
`contextToOpenAi` 转换器。Python 在保存它前与本轮返回结果核对，再要求它完整出现在
后续请求的历史前缀中。只有已匹配的生成消息关联原始 token；extension 额外追加的
assistant 消息仍是新上下文，不按 role 盲目替换。

启动握手与 `session_started` 升级为协议 v3；`pi/client.py` 拒绝旧协议 sidecar。
Pi 继续负责工具执行、轮次推进和 evaluator，Python 继续负责模型生成与训练记录。

### 3. PiAgentLoop 接入与训练输出

`verl/experimental/agent_loop/pi_agent_loop.py` 在每次 `run()` 内创建 context，避免
同一 Ray worker 的并发/先后会话共用历史。`_prompt_tokens` 在 executor 内构造增量
prompt；`_generate` 记录原始 response；`step_complete` 推进上下文状态。

每个 `AgentLoopOutput` 仍对应一次实际调用。prompt、response、logprobs、response
mask 以及 session 终局奖励分配方式保持原契约。实际送入生成服务的 prompt 原样进入
recorder 和输出；没有从 Pi 最终 transcript 重建训练目标。trace 与 output 分别增加
`tokenization=incremental`、`pi_tokenization=incremental` 标记。

### 4. 失败边界

以下情况中断该 rollout，并沿现有 finally 清理 sidecar、记录 rollout_error：

- 历史消息被删改、压缩、重排，或工具定义发生变化。
- 完成事件缺少 v3 投影、轮次不匹配，或 Pi 改变了该轮生成文本/工具参数。
- 原生 builder 合并时删除或替换已有 runtime token。Qwen 末尾补换行允许，因为它只
  增加新 token；需要裁剪旧 token 的其他模型边界暂不支持。
- 超过 prompt/model context 预算，或没有追加新上下文就再次请求生成。

不对上述情况自动整段重编码。原生模板若无法进行 token 后缀差分，也直接暴露其错误。

### 5. 环境探针与日志验收

`examples/pi/tau2_telecom/preflight.py` 使用相同 context/builder，给脚本指定的 Hermes
回复配上脚本 tokens，以覆盖真实 Pi/Tau2 工具、评价和增量协议。该探针没有学生采样，
其 `probe_uses_scripted_generation=true` 与真实 RL 证据分开。

`verify_run.py` 对新标记的 trace 校验相邻 prompt 的完整原始前缀以及 generation/turn
ID 顺序。`--require-incremental-tokens` 拒绝旧 trace，并要求至少存在一次跨轮生成；
原有 response mask/logprobs、梯度、checkpoint 和学习信号检查继续保留。

## 已编写的回归用例

| 文件 | 覆盖内容 |
| --- | --- |
| `test_pi_token_context_on_cpu.py` | 实际 tokenizers BPE 的 `he + llo -> hello` 非 roundtrip、多轮原始前缀、Qwen 结束换行、多个工具结果与 JSON 空格、历史/tools 改写、错误投影/轮次、extension assistant、对象拷贝、会话隔离、禁止裁剪旧 token |
| 同文件的 checkpoint 用例 | 从 `PI_TEST_TOKENIZER_PATH` 离线读取真实 Qwen tokenizer/config，分别覆盖用户续轮与工具续轮；不设置路径时明确 skip |
| `test_pi_agent_loop_on_cpu.py` | 两轮 mock 生成请求与返回训练片段的 prompt 一致，原始 response/logprobs/mask 与 reward 保留；历史被改写时不发起下一次模型调用，并关闭 sidecar |
| `test_pi_run_verifier_on_cpu.py` | trace 保留前缀时接受，重编码/删除 token 或未追加上下文时拒绝 |
| `pi/sidecar/test/sidecar.test.mjs` | 真实 SDK fixture 的 v3 协议、generation/turn 对应、assistant 投影与下一轮请求一致；原有工具/evaluator 生命周期检查 |

这些用例尚未执行，不能将“已编写”计为“已通过”。BPE fixture 使用简化 ChatML 模板，
不会代替真实 Qwen checkpoint 用例、canonical Tau2 preflight 或单卡 RL 验证。

## 待远程恢复后的验收

按仓库规则先本地 commit/push、远程拉取一致版本，再执行
[README 的远程检查命令](../examples/pi/tau2_telecom/README.md#检查和单卡训练)：

1. Python 回归指定 `PI_TEST_TOKENIZER_PATH="$STUDENT_MODEL"`，确保 checkpoint 两项
   没有 skip；执行 Node check 与真实 SDK fixture 测试。
2. canonical preflight 完成至少一次工具调用和两轮模型请求，终局 evaluator 完整。
3. 新 run 目录做单卡 RL，实际 trace 的每个后续 prompt 保留原始前缀，且具有工具调用。
4. 验证训练片段 token/logprob/mask、有限 loss/梯度、actor/optimizer checkpoint；如
   要声称产生学习信号，仍需组内奖励差异与非零梯度，不能仅以退出码为证。
5. 将实际命令、依赖/commit、trace 与验证结果补充到 bug&fix 的运行解决记录。

09-14 的单卡历史结果只适用于旧实现，不用于替代以上验收。
