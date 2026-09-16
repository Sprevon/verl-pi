# Pi 历史消息重新编码与训练 token 对齐核查

## 核查问题与范围

用户关注：Pi 每轮发回 messages/tools 后，是否会复用已采样的 assistant token，只编码新增的非模型消息；如果历史 assistant 被 decode 后重新 encode，BPE 分词变化是否会破坏 RL 训练。

2026-09-15 首先对本地源码进行只读核查，沿 Pi sidecar、prompt builder、vLLM 输入输出、TQ、训练引擎和已有测试追踪。下文“当前”与源码行号均指修复前的 `4081284703306e353242b897d05842b2e7a1ca37`，保留为问题定位记录；后续授权的实现另见文末及 [增量实现记录](2026-09-15-pi-incremental-token-implementation.md)。vGPUN 已按此前要求关闭，本次没有启动远程环境，也没有运行新的 tokenizer、logprob 或收敛实验。

## 已确认的结论

1. 当前 Pi 路径每次模型调用都会对该次完整 messages/tools 进行模板渲染和编码，包括历史 assistant。没有从 recorder 取出原始 assistant token 替换历史消息，也没有调用 Continuous Token 的 assistant/context merge 接口。
2. 当前训练以一次实际模型调用为一个独立片段。该片段保存并使用当时实际送入生成服务的 prompt IDs、实际采样的 response IDs 和对应 logprobs；不会从最终 transcript 重新编码 response 训练目标。
3. 因此，代码未实现跨轮历史 token 的原样保留，但本次核查没有发现由历史重新编码直接造成的片段内 response/logprob 错配。仅凭历史重新编码，不能认定当前训练一定无法收敛。
4. 尚未测量真实 Qwen 轨迹中 decode/encode 改变 token 的频率，也没有比较真实部署端与训练桥接端的逐轮 prompt IDs。此前“训用一致”应限定为复用 Pi 执行链路以及逐次生成的采样/训练数据对齐；不应扩大成已验证跨轮 token 原样保留或部署端逐 token 等同。

## 源码证据

### 1. Pi 传回的是消息与工具结构

- `verl/experimental/agent_loop/pi/sidecar/main.mjs:78`：`contextToOpenAi` 把 Pi user/assistant/toolResult 转成消息结构，assistant toolCall 参数通过 `JSON.stringify` 序列化。
- 同文件 `:127`：`makeAssistantMessage` 将 Python 回复的 text/tool_calls 变成 Pi assistant message，未包含原始生成 token IDs。
- 同文件 `:284`：`session.agent.streamFunction` 发出 `generation_request`，字段为 messages/tools/generation_id 等，没有历史 token IDs。
- `verl/experimental/agent_loop/tool_parser.py:108`：Hermes parser decode 原始 response IDs，解析工具调用 JSON，再将结构化调用交给 Pi。

这条往返除了 BPE 分词变化，还可能包含工具 JSON 的空格、转义及序列化形式变化。具体变化次数和幅度尚未运行实测。

### 2. 每轮 prompt 是完整编码

调用链：

```text
PiAgentLoop._generate                         pi_agent_loop.py:105
  -> _prompt_tokens(messages, tools)          pi_agent_loop.py:92
  -> builder.build_initial_tokens            continuous_token.py:99
  -> builder._render_tokens                  continuous_token.py:224
  -> apply_chat_template(tokenize=True)
```

`build_initial_tokens` 直接渲染本次全部消息，没有读取上一轮 runtime token 的参数或状态。Qwen 的 builder 继承这个方法；其模型特定边界处理位于 `_merge_context_token_ids`，Pi 当前路径没有调用该 merge 方法。

`pi_agent_loop.py:145` 保存到 recorder 的历史 token 只用于后续输出训练片段，不参与下一轮 `_prompt_tokens`。

### 3. 生成服务直接接收 prompt IDs

- `verl/workers/rollout/vllm_rollout/vllm_async_server.py:642`：将 `prompt_ids` 放入 `TokensPrompt(prompt_token_ids=...)`。
- 同文件 `:711`：取实际生成的 `final_res.outputs[0].token_ids`，按对应 token 取 logprobs。
- `pi_agent_loop.py:125`、`:145`：保存这些原始 IDs/logprobs；文本解析与 special-token 清理作用于交给 Pi 的消息，不重写 recorder 中的 response IDs。

### 4. 训练端直接拼接片段内原始 IDs

- `pi_agent_loop.py:262`：每个已完成 turn 返回一个 `AgentLoopOutput`，使用 recorder 保存的 prompt/response/logprobs，response mask 全为 1。
- `verl/trainer/ppo/v1/agent_loop_tq.py:183`：把这两组 IDs 转成张量，通过 `torch.cat([prompts, responses])` 组成训练输入。
- `verl/experimental/agent_loop/agent_loop.py:890`：文本路径由 attention mask 构造一维 position IDs。
- `verl/workers/engine/fsdp/transformer_impl.py:1165`：读取保存的 input_ids 和 position_ids。
- 同文件 `:1197`、`:1424`：用这些 input IDs 构造 next-token labels 并计算 logprobs。

这里没有把最终 messages 再 tokenize 成训练输入的路径。打包、padding 和数值后端差异是其他验证维度，不能由本次源码核查直接推出逐值概率完全一致。

## 两轮例子与错误边界

设 `P_t` 为第 t 次实际请求的 prompt IDs，`A_t` 为该次实际采样的 response IDs，`L_t` 为对应 rollout logprobs。

```text
第 1 次生成：P1 -> A1, L1
第 2 次 prompt：P2 = Encode(Template(Pi 的完整历史消息和工具))
第 2 次生成：P2 -> A2, L2

训练片段 1：input = P1 + A1，response = A1，logprobs = L1
训练片段 2：input = P2 + A2，response = A2，logprobs = L2
```

假设词表中存在相应 token，模型可能生成 `[he, llo]`，decode 得到 `hello`，确定性 tokenizer 再 encode 时选取 `[hello]`。这是非单射的解码/规范化编码问题，BPE 本身不是位置编码；这里使用符号示意，没有声称已测得 Qwen 对 `hello` 的具体 token IDs。

当前设计下，第二轮历史中的 `hello` 可能已经是另一个分词结果，但第一轮训练目标仍是原来的 `[he, llo]`。第二轮的 A2 是在重新编码后的 P2 上真实采样出来的，训练也使用这个 P2，因此这一变化不会自动造成片段内采样和训练前缀不一致。

危险的另一种实现是：先把整个最终对话重新编码，把 A1 改成了 A1'，却仍给 A1' 使用原先 A1 的 logprobs/mask/位置映射；或者生成 A2 时用 P2，训练 A2 时却改用另一份 P2'。当前 Pi 代码在上述位置保留了原始每轮数据，未发现这两种路径。

如果实际部署采用另一种历史构造方法（例如持续保留原始 token/KV），那么其下一轮输入可能与当前消息重编码方案不同，必须另行比较。部署侧若也逐次接收消息并用同样模板/tokenizer 重新编码，保留原始历史 token 的改造本身也会改变该接口的语义。

## 原生 ToolAgentLoop 的 merge 方式

- `verl/experimental/agent_loop/tool_agent_loop.py:266`：调用 `ct_merge_assistant_token`。
- `verl/utils/tokenizer/continuous_token.py:195`：直接把原始 assistant token IDs 追加到 runtime IDs。
- `tool_agent_loop.py:378`：调用 `ct_merge_context_msg`。
- `continuous_token.py:178`：编码追加的上下文，再与已有 runtime IDs 合并。
- `continuous_token.py:537`：Qwen builder 为 `<|im_end|>` 后的模板换行处理边界。

“复用原始 assistant token，只编码追加上下文”对应这条原生路径。Pi 当前使用同一个 builder 的初始编码方法，不代表已经启用这些增量合并能力。

## 现有测试与尚未验证的内容

`tests/experimental/agent_loop/test_pi_agent_loop_on_cpu.py:118` 的 builder 是 mock：第一轮返回 `[1, 2]`，第二轮返回 `[42, 43, 44]`。

同文件 `:226` 检查各轮真实 response/logprobs、片段 prompt 和 session 元信息被保留。这个测试有意允许第二轮 prompt 改变，支持“各调用独立片段”的契约；它不覆盖真实 BPE roundtrip、历史 assistant token 保留或部署 prompt 等价。

后续针对本问题的验证应包含：

1. 在远程使用实际 tokenizer 构造或寻找 decode 后再 encode 改变 IDs 的可解码序列，不能仅用 mock token 表示已实测。
2. 核对每轮实际送入 vLLM 的 P_t、实际生成的 A_t/L_t 与进入训练引擎的数据相同，并检查响应起点、位置和 mask。
3. 针对相同 Pi 上下文，对照当前完整渲染、原始 token 增量拼接及真实部署请求，分别统计首个差异位置与长度差。
4. 覆盖工具 JSON 序列化、结束 token、模板换行、多个工具结果，以及历史/工具定义发生修改的情形。
5. 如决定增加增量模式，先明确部署侧契约；明确标记模型生成的 assistant 与 extension 注入的 assistant，禁止仅按 role 盲目替换；对非 append-only 历史显式定义行为；保持实际生成输入与训练输入一致。

以上只读核查阶段没有实现增量模式，也不以之前单步非零梯度验收代替这里的 tokenizer 专项验证或长期收敛验证。用户随后授权修复，实施范围见下文。

## 术语参考

- [Hugging Face：Tokenization algorithms](https://huggingface.co/docs/transformers/tokenizer_summary)：BPE 根据词表及合并规则分词。
- [Hugging Face：The tokenization pipeline](https://huggingface.co/docs/tokenizers/main/pipeline)：编码处理流程与解码接口的职责。

## 已授权的修复方案（实施前记录）

用户随后要求参照原生 ToolAgentLoop 改为增量 token 方案。实施范围：

1. 为每次 `run()` 建立独立 `PiTokenContext`。第一轮完整编码，随后用原生 `merge_assistant_tokens` 追加真实采样 IDs，用 `merge_context_tokens` 编码并追加工具、用户或 extension 上下文。
2. Node 在 `step_complete` 中提供通过同一个 `contextToOpenAi` 转换的 assistant 消息。Python 将该消息与本轮返回给 Pi 的 text/tool_calls 核对后，建立“这条 assistant 消息对应这段原始采样 IDs”的明确关联。协议升级为 v3，避免旧 sidecar 被误用。
3. 后续请求必须保留已有消息前缀和完整工具定义。生成消息被改写、历史删改、工具定义变化、缺失轮次关联均报错；不自动回退到完整历史重编码。新增非模型 assistant 消息只作为上下文编码，不冒充已采样输出。
4. 合并使用模型专属边界处理，同时校验已有 runtime token 前缀未被替换或删除。仍返回每次调用的独立训练片段，保持原有 mask/logprob/GRPO session 语义。
5. 调整 canonical preflight 和运行验收器，补充真实 BPE 非 roundtrip 序列、Qwen 换行、工具/历史变化、跨 session 隔离和模型输入/训练输出对齐用例；真实 Pi SDK 测试检查事件投影与下一轮上下文一致。
6. 所有运行按仓库规则放在远程。开始修复时 vGPUN SSH 仍关闭；用户随后明确“暂时先本地修改吧”。本次仅完成本地代码、用例和文档，运行验证留待远程恢复；不启动远程、不在 Mac 执行项目测试，也不报告测试通过。
