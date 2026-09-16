# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Train real Pi coding-agent turns with verl's native v1 GRPO pipeline.

Each output contains the exact prompt and response from one generation request.
Sampled assistant tokens are preserved in later prompts through native Continuous
Token merges. Pi alone runs tools and decides the next turn.
"""

import asyncio
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.pi.client import PiSidecarClient, PiSidecarError
from verl.experimental.agent_loop.pi.recorder import PiTrainingRecorder, is_tool_error
from verl.experimental.agent_loop.pi.token_context import PiTokenContext
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.trainer.distillation import is_distillation_enabled


def _as_dict(value):
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "item"):
        return _as_dict(value.item())
    raise TypeError(f"Expected an object, got {type(value).__name__}")


@contextmanager
def _timed_phase(record_trace, phase, **fields):
    started, started_unix = perf_counter(), time()
    timing = dict(fields)
    try:
        yield timing
    finally:
        duration = perf_counter() - started
        timing.update(start_unix_s=started_unix, end_unix_s=started_unix + duration, duration_s=duration)
        if record_trace is not None:
            record_trace({"type": "phase_timing", "phase": phase, **timing})


class PiAgentLoop(AgentLoopBase):
    # Registered by agent_loop_config_path: decorating this lazily imported class
    # would replace that full YAML config with a target-only registry entry.
    def __init__(
        self,
        *args,
        cwd: str,
        training_extension: str,
        agent_dir: str | None = None,
        node_binary: str = "node",
        sidecar_entrypoint: str | None = None,
        coding_agent_entrypoint: str = "",
        prompt_entry_type: str = "pi-training-task-prompt",
        evaluation_entry_type: str = "pi-training-evaluation",
        task_id_env: str = "",
        environment: dict | None = None,
        max_turns: int = 8,
        event_timeout: float = 300,
        generation_timeout: float = 600,
        trace_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        max_turns = int(max_turns)
        event_timeout = float(event_timeout)
        generation_timeout = float(generation_timeout)
        if not self.config.trainer.get("use_v1", False):
            raise ValueError("PiAgentLoop requires trainer.use_v1=true for multi-output trajectories")
        if self.config.trainer.v1.trainer_mode != "sync":
            raise ValueError("PiAgentLoop currently supports trainer.v1.trainer_mode=sync")
        if self.config.algorithm.adv_estimator != "grpo":
            raise ValueError("PiAgentLoop currently supports GRPO session-level credit assignment")
        if is_distillation_enabled(self.config.distillation):
            raise ValueError("Multi-output Pi OPD requires teacher logprobs for every turn; currently unsupported")
        if self.processor is not None:
            raise ValueError("PiAgentLoop currently supports text-only models and tasks")
        if max_turns < 1 or event_timeout <= 0 or generation_timeout <= 0:
            raise ValueError("Pi turn and timeout limits must be positive")
        self.cwd = str(Path(cwd).resolve())
        self.training_extension = str(Path(training_extension).resolve())
        self.agent_dir = agent_dir or str(Path(self.cwd) / ".pi" / "agent")
        self.node_binary = node_binary
        self.sidecar_entrypoint = sidecar_entrypoint or str(Path(__file__).parent / "pi" / "sidecar" / "main.mjs")
        self.coding_agent_entrypoint = coding_agent_entrypoint
        self.prompt_entry_type = prompt_entry_type
        self.evaluation_entry_type = evaluation_entry_type
        self.task_id_env = task_id_env
        self.environment = {str(key): str(value) for key, value in (environment or {}).items()}
        self.max_turns = max_turns
        self.event_timeout = event_timeout
        self.generation_timeout = generation_timeout
        self.trace_dir = trace_dir
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)

    async def _prompt_tokens(self, messages, tools, token_context):
        # Initial encoding once per trajectory; subsequent turns preserve sampled
        # assistant tokens and only encode appended context, as in ToolAgentLoop.
        prompt_ids = await self.loop.run_in_executor(None, lambda: token_context.build_prompt(messages, tools))
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError(
                f"Pi prompt has {len(prompt_ids)} tokens, exceeding rollout.prompt_length="
                f"{self.rollout_config.prompt_length}; canonical context was not truncated"
            )
        return prompt_ids

    async def _generate(self, event, sampling_params, request_id, recorder, token_context, record_trace=None):
        messages, tools = event.get("messages"), event.get("tools")
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise PiSidecarError("Pi generation_request must contain messages and tools arrays")
        with _timed_phase(record_trace, "tokenization", generation_id=event["generation_id"]):
            prompt_ids = await self._prompt_tokens(messages, tools, token_context)
        budget = int(self.rollout_config.response_length)
        if self.rollout_config.max_model_len:
            budget = min(budget, int(self.rollout_config.max_model_len) - len(prompt_ids))
        if budget < 1:
            raise ValueError("Pi generation has no remaining model context budget")
        params = dict(sampling_params)
        params["max_tokens"] = min(budget, int(params.get("max_tokens", budget)))
        if self.tool_parser.stop_token_ids:
            params["stop_token_ids"] = sorted(set(params.get("stop_token_ids", []) + self.tool_parser.stop_token_ids))
        with _timed_phase(record_trace, "llm_request", generation_id=event["generation_id"]) as generation_timing:
            output = await asyncio.wait_for(
                self.server_manager.generate(request_id=request_id, prompt_ids=prompt_ids, sampling_params=params),
                timeout=self.generation_timeout,
            )
        elapsed = generation_timing["duration_s"]
        parsing_started, parsing_unix = perf_counter(), time()
        response_ids = list(output.token_ids)
        if len(response_ids) > params["max_tokens"]:
            raise ValueError("Rollout server exceeded the requested Pi token budget")
        logprobs = list(output.log_probs) if output.log_probs is not None else None
        if self.rollout_config.calculate_log_probs and logprobs is None:
            raise ValueError("Pi rollout requested logprobs but the server returned none")
        # Parsing affects the message Pi executes, never the recorded sampling tokens.
        # Hermes reads the generated JSON directly. Do not force Pi's full JSON
        # Schema (e.g. anyOf/$ref parameters) through verl's narrower tool schema.
        schemas = None
        if self.rollout_config.multi_turn.format != "hermes":
            schemas = [OpenAIFunctionToolSchema.model_validate(tool) for tool in tools]
        text, tool_calls = await self.tool_parser.extract_tool_calls(response_ids, schemas)
        payload = []
        for call in tool_calls:
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                arguments = {"__invalid_json__": call.arguments}
            payload.append({"id": call.tool_call_id or uuid4().hex, "name": call.name, "arguments": arguments})
        turn = recorder.record_generation(
            str(event["generation_id"]),
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=logprobs,
            extra_fields=output.extra_fields,
            routed_experts=output.routed_experts,
            generation_seconds=elapsed,
            num_turns=sum(message.get("role") != "system" for message in messages) + 1,
        )
        for special in (self.tokenizer.eos_token, self.tokenizer.bos_token, self.tokenizer.pad_token):
            if special and text:
                text = text.replace(special, "")
        result = {"text": text, "tool_calls": payload, "stop_reason": "toolUse" if payload else "stop"}
        token_context.record_generation(str(event["generation_id"]), response_ids, result)
        if record_trace is not None:
            duration = perf_counter() - parsing_started
            record_trace(
                {
                    "type": "phase_timing",
                    "phase": "parse_and_record",
                    "generation_id": event["generation_id"],
                    "start_unix_s": parsing_unix,
                    "end_unix_s": parsing_unix + duration,
                    "duration_s": duration,
                    "prompt_tokens": len(prompt_ids),
                    "response_tokens": len(response_ids),
                }
            )
        return result, turn

    async def run(self, sampling_params: dict, **kwargs) -> list[AgentLoopOutput]:
        run_started = perf_counter()
        extra_info = _as_dict(kwargs.get("extra_info"))
        task_id = str(extra_info.get("task_id") or kwargs.get("task_id") or "")
        if not task_id:
            raise ValueError("Pi sample is missing extra_info.task_id")
        session_id = uuid4().hex
        environment = dict(self.environment)
        if self.task_id_env:
            environment[self.task_id_env] = task_id
        client = PiSidecarClient(node_binary=self.node_binary, entrypoint=self.sidecar_entrypoint, env=environment)
        recorder = PiTrainingRecorder()
        token_context = PiTokenContext(self.continuous_token_builder)
        evaluation = completion = None
        trace = None
        if self.trace_dir:
            directory = Path(os.path.expandvars(self.trace_dir)).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            trace = (directory / f"{session_id}.jsonl").open("x", encoding="utf-8")

        def record_trace(data):
            if trace:
                trace.write(
                    json.dumps(
                        {**data, "trace_unix_s": time(), "trace_elapsed_s": perf_counter() - run_started},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                trace.flush()

        record_trace(
            {
                "type": "sample",
                "task_id": task_id,
                "uid": kwargs.get("uid"),
                "session_id": session_id,
                "split": extra_info.get("split"),
                "source_split": extra_info.get("source_split", extra_info.get("split")),
                "tokenization": "incremental",
            }
        )
        try:
            with _timed_phase(record_trace, "sidecar_start"):
                await client.start(
                    {
                        "session_id": session_id,
                        "task_id": task_id,
                        "cwd": self.cwd,
                        "training_extension": self.training_extension,
                        "agent_dir": self.agent_dir,
                        "pi_coding_agent_entrypoint": self.coding_agent_entrypoint,
                        "prompt_entry_type": self.prompt_entry_type,
                        "evaluation_entry_type": self.evaluation_entry_type,
                        "max_turns": self.max_turns,
                    },
                    timeout=self.event_timeout,
                )
            while completion is None:
                with _timed_phase(record_trace, "wait_event") as waiting:
                    event = await client.next_event(self.event_timeout)
                    waiting["next_event_type"] = event.get("type")
                record_trace(event)
                event_type = event.get("type")
                if event_type in {"session_started", "session_info"}:
                    continue
                if event_type == "generation_request":
                    if evaluation is not None or len(recorder.turns) >= self.max_turns:
                        raise PiSidecarError("Pi requested generation after its terminal boundary")
                    result, turn = await self._generate(
                        event, sampling_params, session_id, recorder, token_context, record_trace
                    )
                    record_trace(
                        {
                            "type": "generation_tokens",
                            "generation_id": event["generation_id"],
                            "prompt_ids": turn.prompt_ids,
                            "response_ids": turn.response_ids,
                            "response_logprobs": turn.response_logprobs,
                            "response_mask": [1] * len(turn.response_ids),
                        }
                    )
                    with _timed_phase(record_trace, "response_send", generation_id=event["generation_id"]):
                        await client.respond(event["id"], result)
                elif event_type == "step_complete":
                    if evaluation is not None:
                        raise PiSidecarError("Pi completed a turn after its terminal evaluation")
                    recorder.complete_turn(event)
                    token_context.complete_turn(event)
                elif event_type == "evaluation_result":
                    if evaluation is not None:
                        raise PiSidecarError("Pi returned duplicate terminal evaluations")
                    evaluation = event.get("result")
                    if not isinstance(evaluation, dict):
                        raise PiSidecarError("Pi evaluation_result must contain a JSON object")
                elif event_type == "session_complete":
                    completion = event
                else:
                    raise PiSidecarError(f"Unknown Pi event: {event_type}")
            reward = recorder.finalize(evaluation, completion)
        except BaseException as exc:
            record_trace({"type": "rollout_error", "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            try:
                with _timed_phase(record_trace, "sidecar_close"):
                    await client.close()
            finally:
                record_trace({"type": "session_closed", "session_id": session_id})
                if trace:
                    trace.close()

        tool_results = [result for turn in recorder.turns for result in turn.event.get("tool_results", [])]
        reward_info = {
            "score": reward,
            "pi_turns": len(recorder.turns),
            "pi_tool_calls": len(tool_results),
            "pi_tool_errors": sum(is_tool_error(result) for result in tool_results),
            "pi_truncated": int(bool(completion.get("truncated"))),
        }
        return [
            AgentLoopOutput(
                prompt_ids=turn.prompt_ids,
                response_ids=turn.response_ids,
                response_mask=[1] * len(turn.response_ids),
                response_logprobs=turn.response_logprobs,
                routed_experts=turn.routed_experts,
                reward_score=reward,
                num_turns=turn.num_turns,
                metrics=AgentLoopMetrics(generate_sequences=turn.generation_seconds),
                extra_fields={
                    **turn.extra_fields,
                    "pi_session_id": session_id,
                    "pi_generation_id": turn.generation_id,
                    "pi_step_index": index,
                    "pi_task_id": task_id,
                    "pi_tokenization": "incremental",
                    "pi_terminated": bool(completion.get("terminated")),
                    "pi_truncated": bool(completion.get("truncated")),
                    "reward_extra_info": dict(reward_info),
                },
            )
            for index, turn in enumerate(recorder.turns)
        ]
