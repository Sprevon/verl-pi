# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl.experimental.agent_loop import agent_loop, pi_agent_loop
from verl.experimental.agent_loop.pi.recorder import PiTrainingRecorder, is_tool_error
from verl.experimental.agent_loop.pi_agent_loop import PiAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall


def test_yaml_registration_survives_lazy_import_and_multiple_instances(monkeypatch, tmp_path):
    configured = OmegaConf.create(
        {
            "_target_": "verl.experimental.agent_loop.pi_agent_loop.PiAgentLoop",
            "cwd": str(tmp_path),
            "training_extension": str(tmp_path / "canonical.ts"),
            "max_turns": 7,
        }
    )
    monkeypatch.setitem(agent_loop._agent_loop_registry, "pi_agent", configured)
    # YAML is registered before Hydra imports the target in a fresh Ray worker.
    importlib.reload(pi_agent_loop)

    def base_init(self, **kwargs):
        self.config = OmegaConf.create(
            {
                "trainer": {"use_v1": True, "v1": {"trainer_mode": "sync"}},
                "algorithm": {"adv_estimator": "grpo"},
                "distillation": {"enabled": False},
            }
        )
        self.processor = self.tokenizer = None
        self.rollout_config = SimpleNamespace(multi_turn=SimpleNamespace(format="hermes"))

    monkeypatch.setattr(agent_loop.AgentLoopBase, "__init__", base_init)
    monkeypatch.setattr(pi_agent_loop.ToolParser, "get_tool_parser", lambda *args: None)
    for _ in range(2):
        instance = instantiate(agent_loop._agent_loop_registry["pi_agent"])
        assert instance.cwd == str(tmp_path)
        assert instance.training_extension == str(tmp_path / "canonical.ts")
        assert instance.max_turns == 7


def record(recorder, name="g0", **kwargs):
    return recorder.record_generation(
        name, prompt_ids=[1, 2], response_ids=[3, 4], response_logprobs=kwargs.pop("logprobs", [-0.1, -0.2]), **kwargs
    )


def test_tool_failure_includes_canonical_error_details():
    assert is_tool_error({"isError": False, "details": {"error": True}})
    assert is_tool_error({"isError": True, "details": {}})
    assert not is_tool_error({"isError": False, "details": {"error": False}})
    assert not is_tool_error({"details": None})


def completed(name="g0"):
    return {"generation_id": name, "assistant_message": {"role": "assistant", "stopReason": "stop"}, "tool_results": []}


def test_recorder_requires_evaluation_completion_and_exact_token_metadata():
    recorder = PiTrainingRecorder()
    turn = record(recorder)
    assert turn.response_ids == [3, 4]
    with pytest.raises(ValueError, match="unfinished"):
        recorder.finalize({"reward": 1}, {"turns": 1})
    recorder.complete_turn(completed())
    with pytest.raises(ValueError, match="terminal evaluator"):
        recorder.finalize(None, {"turns": 1})
    with pytest.raises(ValueError, match="turn count"):
        recorder.finalize({"reward": 1}, {"turns": 2})
    assert recorder.finalize({"reward": 0}, {"turns": 1}) == 0
    assert recorder.finalize({"reward": 0.75}, {"turns": 1}) == 0.75


@pytest.mark.parametrize("logprobs", [[-0.1], [-0.1, float("nan")]])
def test_recorder_rejects_misaligned_or_nonfinite_logprobs(logprobs):
    with pytest.raises(ValueError):
        record(PiTrainingRecorder(), logprobs=logprobs)


def test_recorder_rejects_duplicate_unknown_failed_and_empty_generations():
    recorder = PiTrainingRecorder()
    record(recorder)
    with pytest.raises(ValueError, match="Duplicate or overlapping"):
        record(recorder)
    with pytest.raises(ValueError, match="unknown generation"):
        recorder.complete_turn(completed("other"))
    bad_turn = completed()
    bad_turn["assistant_message"]["stopReason"] = "error"
    with pytest.raises(ValueError, match="valid assistant"):
        recorder.complete_turn(bad_turn)
    with pytest.raises(ValueError, match="nonempty"):
        PiTrainingRecorder().record_generation("empty", prompt_ids=[1], response_ids=[], response_logprobs=[])


def test_recorder_copies_sampling_tokens_and_rejects_nonfinite_reward():
    prompt, response = [1, 2], [3, 4]
    recorder = PiTrainingRecorder()
    recorder.record_generation("g0", prompt_ids=prompt, response_ids=response, response_logprobs=None)
    prompt.clear()
    response.clear()
    recorder.complete_turn(completed())
    assert recorder.turns[0].prompt_ids == [1, 2]
    assert recorder.turns[0].response_ids == [3, 4]
    with pytest.raises(ValueError, match="finite"):
        recorder.finalize({"reward": float("inf")}, {"turns": 1})


def make_loop(tmp_path):
    loop = object.__new__(PiAgentLoop)
    loop.loop = asyncio.get_running_loop()
    loop.rollout_config = SimpleNamespace(
        prompt_length=8,
        response_length=3,
        max_model_len=12,
        calculate_log_probs=True,
        multi_turn=SimpleNamespace(format="hermes"),
    )
    loop.tokenizer = SimpleNamespace(eos_token="<eos>", bos_token=None, pad_token=None)
    loop.continuous_token_builder = SimpleNamespace(
        # Context can change its serialized prefix; each request remains an independent segment.
        build_initial_tokens=lambda messages, tools: [1, 2] if len(messages) == 1 else [42, 43, 44]
    )
    loop.server_manager = SimpleNamespace(
        generate=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    token_ids=[7, 8, 9],
                    log_probs=[-0.1, -0.2, -0.3],
                    routed_experts=None,
                    extra_fields={"min_global_steps": 1, "max_global_steps": 1},
                ),
                SimpleNamespace(
                    token_ids=[10, 11],
                    log_probs=[-0.4, -0.5],
                    routed_experts=None,
                    extra_fields={"min_global_steps": 1, "max_global_steps": 1},
                ),
            ]
        )
    )
    loop.tool_parser = SimpleNamespace(
        stop_token_ids=[],
        extract_tool_calls=AsyncMock(
            side_effect=[("<eos>", [FunctionCall(name="lookup", arguments="{}")]), ("Done<eos>", [])]
        ),
    )
    loop.cwd = loop.agent_dir = str(tmp_path)
    loop.training_extension = str(tmp_path / "extension.ts")
    loop.node_binary = "node"
    loop.sidecar_entrypoint = "sidecar.mjs"
    loop.coding_agent_entrypoint = ""
    loop.prompt_entry_type = "prompt"
    loop.evaluation_entry_type = "evaluation"
    loop.task_id_env = "TASK_ID"
    loop.environment = {"TASK_PYTHON": "/isolated/python"}
    loop.max_turns = 4
    loop.event_timeout = loop.generation_timeout = 2
    loop.trace_dir = str(tmp_path / "traces")
    return loop


class ScriptedTransport:
    instances = []
    missing_evaluation = False

    def __init__(self, **kwargs):
        self.options = kwargs
        self.closed = False
        self.replies = []
        self.instances.append(self)

    async def start(self, payload, timeout):
        self.payload = payload
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup a record",
                    "parameters": {
                        "type": "object",
                        "properties": {"optional": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                    },
                },
            }
        ]
        request = {"type": "generation_request", "tools": tools}
        self.events = [
            {"type": "session_started"},
            {**request, "id": "r0", "generation_id": "g0", "messages": [{"role": "user", "content": "task"}]},
            {"type": "step_complete", **completed(), "tool_results": [{"isError": False}]},
            {
                **request,
                "id": "r1",
                "generation_id": "g1",
                "messages": [{"role": "user", "content": "task"}, {"role": "tool", "content": "observation"}],
            },
            {"type": "step_complete", **completed("g1")},
            {"type": "evaluation_result", "result": {"reward": 0.75}},
            {"type": "session_complete", "turns": 2, "terminated": True, "truncated": False},
        ]
        if self.missing_evaluation:
            self.events = [event for event in self.events if event["type"] != "evaluation_result"]

    async def next_event(self, timeout):
        return self.events.pop(0)

    async def respond(self, request_id, result):
        self.replies.append((request_id, result))

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_two_pi_turns_keep_actual_tokens_group_metadata_and_reward(monkeypatch, tmp_path):
    monkeypatch.setattr("verl.experimental.agent_loop.pi_agent_loop.PiSidecarClient", ScriptedTransport)
    loop = make_loop(tmp_path)
    sampling_params = {"temperature": 1.0, "logprobs": True}
    outputs = await loop.run(sampling_params, extra_info={"task_id": "task0"}, uid="group0")
    assert len(outputs) == 2
    assert outputs[0].prompt_ids == [1, 2]
    assert outputs[1].prompt_ids == [42, 43, 44]
    assert outputs[0].response_ids == [7, 8, 9]
    assert outputs[1].response_ids == [10, 11]
    assert [output.response_mask for output in outputs] == [[1, 1, 1], [1, 1]]
    assert outputs[1].response_logprobs == [-0.4, -0.5]
    assert all(output.reward_score == 0.75 for output in outputs)
    assert outputs[0].extra_fields["pi_session_id"] == outputs[1].extra_fields["pi_session_id"]
    assert outputs[1].extra_fields["pi_step_index"] == 1
    assert outputs[0].extra_fields["reward_extra_info"]["pi_tool_calls"] == 1
    assert sampling_params == {"temperature": 1.0, "logprobs": True}
    requests = loop.server_manager.generate.call_args_list
    assert requests[0].kwargs["request_id"] == requests[1].kwargs["request_id"]
    assert requests[0].kwargs["sampling_params"]["max_tokens"] == 3
    transport = ScriptedTransport.instances[-1]
    assert transport.closed
    assert transport.options["env"]["TASK_ID"] == "task0"
    assert transport.replies[1][1]["text"] == "Done"
    assert list((tmp_path / "traces").glob("*.jsonl"))


@pytest.mark.asyncio
async def test_missing_evaluation_fails_and_closes_sidecar(monkeypatch, tmp_path):
    monkeypatch.setattr("verl.experimental.agent_loop.pi_agent_loop.PiSidecarClient", ScriptedTransport)
    monkeypatch.setattr(ScriptedTransport, "missing_evaluation", True)
    with pytest.raises(ValueError, match="terminal evaluator"):
        await make_loop(tmp_path).run({}, extra_info={"task_id": "task0"})
    assert ScriptedTransport.instances[-1].closed


@pytest.mark.asyncio
async def test_overlong_canonical_prompt_is_not_truncated(tmp_path):
    loop = make_loop(tmp_path)
    loop.rollout_config.prompt_length = 1
    with pytest.raises(ValueError, match="canonical context was not truncated"):
        await loop._prompt_tokens([{"role": "user", "content": "task"}], [])
    loop.server_manager.generate.assert_not_called()


@pytest.mark.asyncio
async def test_generation_timeout_propagates_and_closes_sidecar(monkeypatch, tmp_path):
    monkeypatch.setattr("verl.experimental.agent_loop.pi_agent_loop.PiSidecarClient", ScriptedTransport)
    loop = make_loop(tmp_path)
    loop.server_manager.generate.side_effect = TimeoutError("generation timed out")
    with pytest.raises(TimeoutError):
        await loop.run({}, extra_info={"task_id": "task0"})
    assert ScriptedTransport.instances[-1].closed
