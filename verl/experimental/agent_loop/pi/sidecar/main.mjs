// Pi lifecycle bridge adapted from Agent-R1; generation and training belong to verl.
import { createInterface } from "node:readline";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import {
  createAssistantMessageEventStream,
  createProvider,
} from "@earendil-works/pi-ai";

const EMPTY_USAGE = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const BRIDGE_MODEL = {
  id: "verl-rollout",
  name: "verl rollout bridge",
  api: "openai-completions",
  provider: "verl",
  baseUrl: "",
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 131072,
  maxTokens: 32768,
};

const sessions = new Map();
const pendingHostRequests = new Map();
let requestSequence = 0;
let nodeReadyUnix = 0;
const startupProfile = process.env.PI_STARTUP_PROFILE === "1";
if (process.env.PI_HARNESS_OWNER_PID) {
  const owner = Number(process.env.PI_HARNESS_OWNER_PID);
  setInterval(() => { if (process.ppid !== owner) process.exit(1); }, 1000).unref();
}

function emit(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function startupSpan(sessionId, phase) {
  if (!startupProfile) return () => {};
  const started = performance.now();
  const startUnix = (performance.timeOrigin + started) / 1000;
  return (fields = {}) => {
    const duration = (performance.now() - started) / 1000;
    emit({
      type: "startup_timing", source: "node", pid: process.pid,
      session_id: sessionId, phase, start_unix_s: startUnix,
      end_unix_s: startUnix + duration, duration_s: duration, ...fields,
    });
  };
}

async function startupAsync(sessionId, phase, fn) {
  const finish = startupSpan(sessionId, phase);
  try { return await fn(); }
  finally { finish(); }
}

function startupSync(sessionId, phase, fn) {
  const finish = startupSpan(sessionId, phase);
  try { return fn(); }
  finally { finish(); }
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function requestHost(sessionId, type, payload, abortSignal) {
  const id = `${sessionId}:${++requestSequence}`;
  emit({ type, id, session_id: sessionId, ...payload });
  return new Promise((resolveRequest, rejectRequest) => {
    const pending = { resolve: resolveRequest, reject: rejectRequest, cleanup: () => {} };
    pendingHostRequests.set(id, pending);
    if (!abortSignal) return;
    const onAbort = () => {
      if (pendingHostRequests.get(id) !== pending) return;
      pendingHostRequests.delete(id);
      pending.cleanup();
      pending.reject(new Error("Generation aborted"));
    };
    if (abortSignal.aborted) {
      onAbort();
      return;
    }
    abortSignal.addEventListener("abort", onAbort, { once: true });
    pending.cleanup = () => abortSignal.removeEventListener("abort", onAbort);
  });
}

function contentText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content) && content.some((item) => item?.type === "image")) {
    throw new Error("PiAgentLoop currently supports text-only tasks");
  }
  if (!Array.isArray(content)) return "";
  return content
    .filter((item) => item && item.type === "text")
    .map((item) => item.text)
    .join("\n");
}

function contextToOpenAi(context) {
  const messages = [];
  if (context.systemPrompt) messages.push({ role: "system", content: context.systemPrompt });
  for (const message of context.messages ?? []) {
    if (message.role === "user") {
      messages.push({ role: "user", content: contentText(message.content) });
      continue;
    }
    if (message.role === "assistant") {
      const content = Array.isArray(message.content) ? message.content : [];
      const toolCalls = content
        .filter((item) => item && item.type === "toolCall")
        .map((item) => ({
          id: item.id,
          type: "function",
          function: { name: item.name, arguments: JSON.stringify(item.arguments ?? {}) },
        }));
      const converted = { role: "assistant", content: contentText(message.content) };
      if (toolCalls.length > 0) converted.tool_calls = toolCalls;
      messages.push(converted);
      continue;
    }
    if (message.role === "toolResult") {
      messages.push({
        role: "tool",
        tool_call_id: message.toolCallId,
        name: message.toolName,
        content: contentText(message.content),
      });
    }
  }
  const tools = (context.tools ?? []).map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }));
  return { messages, tools };
}

function canonicalAnchor(messages, tools) {
  return JSON.stringify({
    messages,
    tools: tools.map((tool) => tool.function.name).sort(),
  });
}

function makeAssistantMessage(result) {
  const toolCalls = Array.isArray(result.tool_calls) ? result.tool_calls : [];
  const text = String(result.text ?? "");
  const content = [];
  if (text) content.push({ type: "text", text });
  content.push(
    ...toolCalls.map((toolCall) => ({
      type: "toolCall",
      id: String(toolCall.id),
      name: String(toolCall.name),
      arguments: toolCall.arguments ?? {},
    })),
  );
  if (content.length === 0) content.push({ type: "text", text: "" });
  const stopReason = result.stop_reason ?? (toolCalls.length > 0 ? "toolUse" : "stop");
  return {
    role: "assistant",
    content,
    api: BRIDGE_MODEL.api,
    provider: BRIDGE_MODEL.provider,
    model: BRIDGE_MODEL.id,
    usage: EMPTY_USAGE,
    stopReason,
    timestamp: Date.now(),
  };
}

function makeErrorMessage(message, stopReason = "error") {
  return {
    role: "assistant",
    content: [{ type: "text", text: "" }],
    api: BRIDGE_MODEL.api,
    provider: BRIDGE_MODEL.provider,
    model: BRIDGE_MODEL.id,
    usage: EMPTY_USAGE,
    stopReason,
    errorMessage: message,
    timestamp: Date.now(),
  };
}

function createBridgeProvider() {
  const unavailableStream = () => {
    const stream = createAssistantMessageEventStream();
    stream.push({
      type: "error",
      reason: "error",
      error: makeErrorMessage(
        "verl bridge provider must be invoked through session.agent.streamFunction",
      ),
    });
    return stream;
  };
  return createProvider({
    id: BRIDGE_MODEL.provider,
    name: "verl host bridge",
    auth: {
      apiKey: {
        name: "verl host bridge",
        resolve: async () => ({ auth: {} }),
      },
    },
    models: [BRIDGE_MODEL],
    api: {
      stream: unavailableStream,
      streamSimple: unavailableStream,
    },
  });
}

async function loadCodingAgent(entrypoint) {
  if (!entrypoint) return import("@earendil-works/pi-coding-agent");
  return import(pathToFileURL(resolve(entrypoint)).href);
}

function serializable(value) {
  try {
    JSON.stringify(value);
    return value;
  } catch {
    return String(value);
  }
}

async function createSession(command) {
  const id = String(command.session_id);
  if (startupProfile) {
    emit({
      type: "startup_timing", source: "node", pid: process.pid,
      session_id: id, phase: "node.ready", start_unix_s: nodeReadyUnix,
      end_unix_s: nodeReadyUnix, duration_s: 0,
    });
  }
  if (sessions.size > 0) {
    throw new Error(
      "Run one Pi sidecar per trajectory to isolate extension and environment state",
    );
  }
  if (sessions.has(id)) throw new Error(`Session already exists: ${id}`);

  const cwd = String(command.cwd ?? "");
  if (!cwd) throw new Error("start_session.cwd is required");
  const trainingExtension = String(command.training_extension ?? "");
  if (!trainingExtension) throw new Error("start_session.training_extension is required");
  const agentDir = String(command.agent_dir ?? resolve(cwd, ".pi", "agent"));
  const taskId = String(command.task_id ?? "");
  const promptEntryType = command.prompt_entry_type ?? "pi-training-task-prompt";
  const evaluationEntryType = command.evaluation_entry_type ?? "pi-training-evaluation";
  const entrypoint = String(
    command.pi_coding_agent_entrypoint ?? process.env.PI_CODING_AGENT_ENTRYPOINT ?? "",
  );
  const { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } =
    await startupAsync(id, "pi.sdk_import", () => loadCodingAgent(entrypoint));

  if (!taskId) throw new Error("start_session.task_id is required");

  const state = {
    id,
    maxTurns: Number(command.max_turns ?? 30),
    turnCount: 0,
    currentGenerationId: null,
    closed: false,
    evaluation: null,
    taskPrompt: null,
    extensionErrors: [],
    toolStarts: new Map(),
    toolTimings: [],
    diagnostics: {
      generation_requests: 0,
      completed_turns: 0,
      tool_turns: 0,
      text_turns: 0,
    },
  };
  sessions.set(id, state);

  const resourceLoader = startupSync(id, "pi.resource_loader_ctor", () => new DefaultResourceLoader({
    cwd,
    agentDir,
    noExtensions: true,
    additionalExtensionPaths: [trainingExtension],
  }));
  if (startupProfile) {
    // Profiling-only hook for the pinned SDK. Keep reload's real loading path.
    if (typeof resourceLoader.loadFinalExtensionSet !== "function") {
      throw new Error("Startup profiling requires ResourceLoader.loadFinalExtensionSet (Pi 0.84.4)");
    }
    const loadExtensions = resourceLoader.loadFinalExtensionSet.bind(resourceLoader);
    resourceLoader.loadFinalExtensionSet = (...args) =>
      startupAsync(id, "pi.extension_load", () => loadExtensions(...args));
  }
  await startupAsync(id, "pi.resource_loader_reload", () => resourceLoader.reload());
  const extensionErrors = resourceLoader.getExtensions().errors ?? [];
  if (extensionErrors.length > 0) {
    throw new Error(`Pi extension loading failed: ${JSON.stringify(extensionErrors)}`);
  }

  const modelRuntime = await startupAsync(id, "pi.model_runtime_create", () => ModelRuntime.create({ modelsPath: null }));
  modelRuntime.registerNativeProvider(createBridgeProvider());

  const { session } = await startupAsync(id, "pi.create_agent_session", () => createAgentSession({
    cwd,
    agentDir,
    model: BRIDGE_MODEL,
    thinkingLevel: "off",
    noTools: "builtin",
    modelRuntime,
    resourceLoader,
    sessionManager: SessionManager.inMemory(cwd),
    settingsManager: SettingsManager.inMemory({
      compaction: { enabled: false },
      retry: { enabled: false },
    }),
  }));
  state.session = session;
  let finishPromptPreparation = null;
  session.agent.streamFunction = (_model, context, options) => {
    finishPromptPreparation?.();
    finishPromptPreparation = null;
    const generationId = `${id}:generation:${++state.turnCount}`;
    state.currentGenerationId = generationId;
    state.diagnostics.generation_requests += 1;
    const converted = startupSync(id, "pi.provider_context_conversion", () => contextToOpenAi(context));
    const stream = createAssistantMessageEventStream();
    const abortSignal = options?.signal;
    const fail = (error, aborted = false) => {
      const reason = aborted ? "aborted" : "error";
      stream.push({
        type: "error",
        reason,
        error: makeErrorMessage(errorMessage(error), reason),
      });
    };
    if (abortSignal?.aborted) {
      fail("Generation aborted", true);
      return stream;
    }
    requestHost(
      id,
      "generation_request",
      {
        generation_id: generationId,
        messages: converted.messages,
        tools: converted.tools,
        anchor_obs: canonicalAnchor(converted.messages, converted.tools),
      },
      abortSignal,
    )
      .then((result) => {
        const message = makeAssistantMessage(result);
        if (message.stopReason === "error" || message.stopReason === "aborted") {
          stream.push({ type: "error", reason: message.stopReason, error: message });
        } else {
          stream.push({ type: "done", reason: message.stopReason, message });
        }
      })
      .catch((error) => fail(error, abortSignal?.aborted || errorMessage(error).includes("aborted")));
    return stream;
  };
  session.agent.shouldStopAfterTurn = () => state.closed || state.turnCount >= state.maxTurns;

  session.subscribe((event) => {
    if (event.type === "entry_appended" && event.entry?.type === "custom") {
      if (event.entry.customType === evaluationEntryType) {
        state.evaluation = event.entry.data;
        emit({
          type: "evaluation_result",
          session_id: id,
          result: serializable(event.entry.data),
        });
      } else if (event.entry.customType === promptEntryType) {
        const returnedTaskId = String(event.entry.data?.task_id ?? "");
        const prompt = String(event.entry.data?.prompt ?? "");
        if (returnedTaskId !== taskId) {
          state.extensionErrors.push(
            `Training extension returned task ${returnedTaskId || "<empty>"}, expected ${taskId}`,
          );
        } else if (!prompt.trim()) {
          state.extensionErrors.push("Training extension returned an empty task prompt");
        } else {
          state.taskPrompt = prompt;
          startupSpan(id, "pi.canonical_prompt_published")();
        }
      }
    }
  });
  session.agent.subscribe((event) => {
    if (event.type === "tool_execution_start") {
      state.toolStarts.set(event.toolCallId, {
        name: event.toolName,
        tool_call_id: event.toolCallId,
        start_unix_s: Date.now() / 1000,
        started: performance.now(),
      });
      return;
    }
    if (event.type === "tool_execution_end") {
      const started = state.toolStarts.get(event.toolCallId);
      if (started) {
        const duration = (performance.now() - started.started) / 1000;
        state.toolTimings.push({
          name: started.name,
          tool_call_id: started.tool_call_id,
          start_unix_s: started.start_unix_s,
          end_unix_s: started.start_unix_s + duration,
          duration_s: duration,
          is_error: Boolean(event.isError),
        });
        state.toolStarts.delete(event.toolCallId);
      }
      return;
    }
    if (event.type !== "turn_end" || event.message?.role !== "assistant") return;
    const generationId = state.currentGenerationId;
    if (!generationId) return;
    const hasToolCalls = Array.isArray(event.message.content) &&
      event.message.content.some((item) => item?.type === "toolCall");
    if (hasToolCalls) state.diagnostics.tool_turns += 1;
    else state.diagnostics.text_turns += 1;
    state.diagnostics.completed_turns += 1;
    emit({
      type: "step_complete",
      session_id: id,
      generation_id: generationId,
      reward: 0,
      terminated: false,
      truncated: state.turnCount >= state.maxTurns,
      invalid_action: false,
      diagnostics: { ...state.diagnostics },
      assistant_message: serializable(event.message),
      // Use the same conversion as the next generation_request. This identifies
      // the exact assistant message backed by the host's sampled token IDs.
      assistant_message_openai: contextToOpenAi({ messages: [event.message] }).messages[0],
      tool_results: serializable(event.toolResults ?? []),
      tool_timings: state.toolTimings.splice(0),
    });
  });

  emit({ type: "session_started", session_id: id, protocol_version: 3, runtime: "pi-coding-agent" });
  try {
    await startupAsync(id, "pi.bind_extensions", () => session.bindExtensions({
      mode: "rpc",
      onError: (error) => state.extensionErrors.push(serializable(error)),
    }));
    if (state.extensionErrors.length > 0) {
      throw new Error(`Pi extension startup failed: ${JSON.stringify(state.extensionErrors)}`);
    }
    if (!state.taskPrompt) {
      throw new Error("Training extension did not publish the canonical task prompt");
    }
    finishPromptPreparation = startupSpan(id, "pi.prompt_to_first_provider");
    await session.prompt(state.taskPrompt, { expandPromptTemplates: true });
    if (state.extensionErrors.length > 0) {
      throw new Error(`Pi extension runtime failed: ${JSON.stringify(state.extensionErrors)}`);
    }
    await session.waitForIdle();
    const evaluationStarted = performance.now();
    const evaluationUnix = Date.now() / 1000;
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    const evaluationDuration = (performance.now() - evaluationStarted) / 1000;
    if (state.extensionErrors.length > 0) {
      throw new Error(`Pi extension shutdown failed: ${JSON.stringify(state.extensionErrors)}`);
    }
    if (state.evaluation === null) {
      throw new Error("Training extension did not publish a evaluation result");
    }
    session.dispose();
    emit({
      type: "session_complete",
      session_id: id,
      terminated: Boolean(state.evaluation.terminated),
      truncated: Boolean(state.evaluation.truncated) ||
        (state.turnCount >= state.maxTurns && !state.evaluation.terminated),
      turns: state.turnCount,
      diagnostics: { ...state.diagnostics },
      evaluation_timing: {
        start_unix_s: evaluationUnix,
        end_unix_s: evaluationUnix + evaluationDuration,
        duration_s: evaluationDuration,
      },
    });
  } catch (error) {
    emit({ type: "session_error", session_id: id, error: errorMessage(error) });
    try {
      session.dispose();
    } catch {
      // Best-effort cleanup; the host still receives the session_error above.
    }
  } finally {
    sessions.delete(id);
  }
}

function closeSession(sessionId) {
  const state = sessions.get(sessionId);
  if (!state) return;
  state.closed = true;
  state.session?.agent.abort();
  sessions.delete(sessionId);
  for (const [requestId, pending] of pendingHostRequests.entries()) {
    if (requestId.startsWith(`${sessionId}:`)) {
      pendingHostRequests.delete(requestId);
      pending.cleanup();
      pending.reject(new Error(`Pi session closed: ${sessionId}`));
    }
  }
}

function handleResponse(message) {
  const pending = pendingHostRequests.get(message.response_to);
  if (!pending) return;
  pendingHostRequests.delete(message.response_to);
  pending.cleanup();
  if (message.ok) pending.resolve(message.result ?? {});
  else pending.reject(new Error(String(message.error ?? "Host request failed")));
}

async function handleCommand(message) {
  if (message.type === "response") {
    handleResponse(message);
    return;
  }
  if (message.type === "start_session") {
    void createSession(message).catch((error) => {
      emit({ type: "session_error", session_id: String(message.session_id), error: errorMessage(error) });
    });
    return;
  }
  if (message.type === "close_session") {
    closeSession(String(message.session_id));
    return;
  }
  if (message.type === "shutdown") {
    for (const id of [...sessions.keys()]) closeSession(id);
    process.exit(0);
  }
  throw new Error(`Unknown command: ${message.type}`);
}

process.on("unhandledRejection", (error) => {
  process.stderr.write(`Unhandled Pi error: ${errorMessage(error)}\n`);
  process.exit(1);
});
const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on("close", () => {
  for (const id of [...sessions.keys()]) closeSession(id);
  process.exit(0);
});
input.on("line", (line) => {
  let message;
  try {
    message = JSON.parse(line);
  } catch (error) {
    process.stderr.write(`Invalid JSON command: ${errorMessage(error)}\n`);
    return;
  }
  handleCommand(message).catch((error) => {
    const sessionId = message.session_id ? String(message.session_id) : undefined;
    if (sessionId) emit({ type: "session_error", session_id: sessionId, error: errorMessage(error) });
    else process.stderr.write(`${errorMessage(error)}\n`);
  });
});

nodeReadyUnix = (performance.timeOrigin + performance.now()) / 1000;
emit({ type: "ready", protocol_version: 3, pi_runtime: "pi-coding-agent" });

export { canonicalAnchor, contextToOpenAi, makeAssistantMessage };
