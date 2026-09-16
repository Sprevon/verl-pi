# Pi 首次生成前的细分计时：问题定位与实验设计

## 已有证据与本次范围

上一轮真实 v1 GRPO 测量中，验证轨迹首次 generation request 前等待 12.416 s，Node spawn/ready 仅 0.377 s；工具合计 0.0205 s。约 12 s 的剩余窗口尚不能归因于某个函数。本次先补充实际调用顺序和嵌套耗时，再决定 Ray Harness 资源池的复用边界及对照实验。

2026-09-16 核对：本地及 vGPUN verl-pi 均为 db7aa7d2，Tau2 均为 fe8ac676；工作区干净，远程 GPU 空闲。所有执行在 vGPUN，代码在本地编写、commit/push 后同步。

## 从代码确认的边界

- Node ready 后才动态 import Pi coding-agent SDK。
- ResourceLoader.reload 包含 loadFinalExtensionSet；extension load 是子区间，不能与 reload 重复计入总时间。
- createAgentSession 后 bindExtensions 触发 canonical extension 的 session_start，首次 describe_tools 才启动 Python bridge。
- Python `-m tau2.domains.telecom.pi_bridge` 首先执行 tau2 包的导入；其顶层导入 runner、agent、evaluator 等。必须计量这个窗口，而不能把等待 Python 首个 RPC 统称为环境构造。
- TelecomPiBridge 构造时建立一个 environment，load_task 时另建 environment 并 set_state；两次需要分别记录。
- 真实 Pi solo 路径没有独立 Tau2 agent/user simulator 的 init_state。set_state 的初始化数据更新、初始化动作与 sync_tools 才是这里的实际工作，报告应标为未调用而不是捏造 init_state 耗时。

## 探针方案

在 sidecar 增加可开关的阶段事件，使用单调时钟计算 duration、Unix 时间对齐进程。只在 profile 模式包装当前固定 SDK 版本的 ResourceLoader.loadFinalExtensionSet 实例方法；接口缺失应报错，不静默遗漏阶段。记录 SDK import、ResourceLoader ctor/reload/extension load、ModelRuntime、createAgentSession、bindExtensions、prompt 到首次 provider request，以及请求到 tokenization/LLM submit。

Tau2 使用 verl-pi 中的实验性 Python 启动包装器：保留真实 canonical bridge 和业务方法，先计量包导入，再包装具体函数记录 environment 建立、任务表加载、set_state、工具注册信息准备等。不使用全局 sys.setprofile，避免每个 Python 调用均经过探针对启动耗时造成明显干扰。每条轨迹独立 PID/文件，数据写到 startup-traces，不混进业务 JSONL 协议。

报告同时保留 parent/child 嵌套区间与非重叠主时间线；精确命名未单独测量的区间。生成回复必须来自真实模型，脚本化 CPU preflight 单独标注，不混入 GPU 样本。

## 后续 Harness Pool 验证边界

只有计时证明可预热部分后才选择实现。保留 Pi 对 agent/tool loop 的所有权、每条轨迹的状态隔离、token/logprob 记录和同步策略版本边界。对比需要固定模型、任务、采样和总轨迹数，分别报告准备成本、稳态 rollout wall、端到端成本及资源占用；不把成本移到计时窗口外就声称消除了成本。
