# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Incremental Pi regressions; run on the remote host, including CPU tests."""

import json
import os
import string
from copy import deepcopy
from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer, decoders, models
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast

from verl.experimental.agent_loop.pi.token_context import PiTokenContext
from verl.utils.tokenizer.continuous_token import MergeResult, QwenContinuousTokenBuilder
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

# A small real BPE tokenizer and ChatML template, not a mock encode/decode pair.
# This fixture tests token identity; the optional checkpoint test uses real Qwen.
CHAT_TEMPLATE = (
    "{% if tools %}{{ '<|im_start|>system\\n' }}{{ tools | tojson }}{{ '<|im_end|>\\n' }}{% endif %}"
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message.get('content', '') }}"
    "{% for call in message.get('tool_calls', []) %}"
    "{{ '\\n<tool_call>\\n' }}{{ call['function'] | tojson }}{{ '\\n</tool_call>' }}"
    "{% endfor %}{{ '<|im_end|>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup a record",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    }
]


@pytest.fixture
def tokenizer():
    tokens = ["<unk>", "<|im_start|>", "<|im_end|>", *string.printable, "he", "ll", "llo", "hello"]
    backend = Tokenizer(
        models.BPE(
            vocab={token: index for index, token in enumerate(tokens)},
            merges=[("h", "e"), ("l", "l"), ("ll", "o"), ("he", "llo")],
            unk_token="<unk>",
        )
    )
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
        chat_template=CHAT_TEMPLATE,
        clean_up_tokenization_spaces=False,
    )


def complete(context, generation_id="g0", message=None):
    context.complete_turn(
        {
            "generation_id": generation_id,
            "assistant_message_openai": message if message is not None else {"role": "assistant", "content": "hello"},
        }
    )


def sampled_hello(tokenizer):
    response = tokenizer.convert_tokens_to_ids(["he", "llo"])
    assert tokenizer.decode(response) == "hello"
    assert tokenizer.encode("hello", add_special_tokens=False) != response
    return response + [tokenizer.eos_token_id]


def test_nonroundtrip_bpe_response_survives_multiple_prompts(tokenizer):
    builder = QwenContinuousTokenBuilder(tokenizer)
    builder.build_initial_tokens = Mock(wraps=builder.build_initial_tokens)
    context = PiTokenContext(builder)
    messages = [{"role": "user", "content": "task"}]
    prompt = context.build_prompt(messages, [])
    response = sampled_hello(tokenizer)
    for index in range(3):
        context.record_generation(f"g{index}", response, {"text": "hello", "tool_calls": []})
        complete(context, f"g{index}")
        messages += [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "continue"}]
        next_prompt = context.build_prompt(messages, [])
        prefix = prompt + response
        assert next_prompt[: len(prefix)] == prefix
        assert next_prompt[len(prefix)] == tokenizer.encode("\n", add_special_tokens=False)[0]
        assert next_prompt != builder._render_tokens(messages, add_generation_prompt=True, tools=[])
        prompt = next_prompt
    builder.build_initial_tokens.assert_called_once()


def test_tool_json_and_multiple_results_do_not_reencode_assistant(tokenizer):
    builder = QwenContinuousTokenBuilder(tokenizer)
    context = PiTokenContext(builder)
    messages = [{"role": "user", "content": "task"}]
    prompt = context.build_prompt(messages, TOOLS)
    calls = [{"id": f"lookup{i}", "name": "lookup", "arguments": {"x": i}} for i in range(2)]
    raw_text = "".join(
        "\n<tool_call>\n" + json.dumps({"name": call["name"], "arguments": call["arguments"]}) + "\n</tool_call>"
        for call in calls
    )
    response = tokenizer.encode(raw_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    context.record_generation("g0", response, {"text": "", "tool_calls": calls})
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], separators=(",", ":"))},
            }
            for call in calls
        ],
    }
    complete(context, message=assistant)
    results = [
        {"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": f"result {index}"}
        for index, call in enumerate(calls)
    ]
    next_prompt = context.build_prompt(messages + [assistant] + results, TOOLS)
    prefix = prompt + response
    assert next_prompt[: len(prefix)] == prefix
    suffix = tokenizer.decode(next_prompt[len(prefix) :], skip_special_tokens=False)
    assert "result 0" in suffix and "result 1" in suffix
    assert suffix.endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("change", ["system", "assistant", "remove", "tools"])
def test_history_or_tool_schema_changes_fail_without_reencoding(tokenizer, change):
    builder = QwenContinuousTokenBuilder(tokenizer)
    builder.build_initial_tokens = Mock(wraps=builder.build_initial_tokens)
    context = PiTokenContext(builder)
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "task"}]
    context.build_prompt(messages, TOOLS)
    context.record_generation("g0", sampled_hello(tokenizer), {"text": "hello", "tool_calls": []})
    complete(context)
    updated = messages + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "continue"}]
    tools = deepcopy(TOOLS)
    if change == "system":
        updated[0] = {"role": "system", "content": "new policy"}
    elif change == "assistant":
        updated[2] = {"role": "assistant", "content": "rewritten"}
    elif change == "remove":
        updated = updated[1:]
    else:
        tools[0]["function"]["parameters"]["properties"]["x"]["type"] = "string"
    with pytest.raises(ValueError, match="history changed|tool definitions changed"):
        context.build_prompt(updated, tools)
    builder.build_initial_tokens.assert_called_once()


@pytest.mark.parametrize(
    "event, error",
    [
        ({"generation_id": "other"}, "matching incremental generation"),
        ({"generation_id": "g0"}, "protocol v3 required"),
        (
            {"generation_id": "g0", "assistant_message_openai": {"role": "assistant", "content": "rewritten"}},
            "changed the generated assistant",
        ),
        (
            {
                "generation_id": "g0",
                "assistant_message_openai": {"role": "assistant", "content": "hello", "tool_calls": [{}]},
            },
            "malformed assistant tool calls",
        ),
    ],
)
def test_turn_must_match_the_generated_assistant(tokenizer, event, error):
    context = PiTokenContext(QwenContinuousTokenBuilder(tokenizer))
    context.build_prompt([{"role": "user", "content": "task"}], [])
    context.record_generation("g0", sampled_hello(tokenizer), {"text": "hello", "tool_calls": []})
    with pytest.raises(ValueError, match=error):
        context.complete_turn(event)


def test_prompt_requires_completed_turn_and_new_context(tokenizer):
    context = PiTokenContext(QwenContinuousTokenBuilder(tokenizer))
    messages = [{"role": "user", "content": "task"}]
    context.build_prompt(messages, [])
    with pytest.raises(ValueError, match="previous generation/turn"):
        context.build_prompt(messages, [])
    context.record_generation("g0", sampled_hello(tokenizer), {"text": "hello", "tool_calls": []})
    with pytest.raises(ValueError, match="previous generation/turn"):
        context.build_prompt(messages, [])
    complete(context)
    with pytest.raises(ValueError, match="append context"):
        context.build_prompt(messages + [{"role": "assistant", "content": "hello"}], [])


def test_extension_assistant_is_encoded_as_new_context(tokenizer):
    context = PiTokenContext(QwenContinuousTokenBuilder(tokenizer))
    messages = [{"role": "user", "content": "task"}]
    prompt = context.build_prompt(messages, [])
    response = sampled_hello(tokenizer)
    context.record_generation("g0", response, {"text": "hello", "tool_calls": []})
    complete(context)
    updated = messages + [
        {"role": "assistant", "content": "hello"},  # Matched sampled message.
        {"role": "assistant", "content": "hello"},  # Extension-supplied context.
        {"role": "user", "content": "continue"},
    ]
    next_prompt = context.build_prompt(updated, [])
    prefix = prompt + response
    assert next_prompt[: len(prefix)] == prefix
    assert tokenizer.convert_tokens_to_ids("hello") in next_prompt[len(prefix) :]


def test_sessions_and_caller_owned_lists_do_not_share_state(tokenizer):
    builder = QwenContinuousTokenBuilder(tokenizer)
    first, second = PiTokenContext(builder), PiTokenContext(builder)
    messages = [{"role": "user", "content": "task"}]
    tools = deepcopy(TOOLS)
    first_prompt = first.build_prompt(messages, tools)
    original_prompt = list(first_prompt)
    first_prompt.clear()
    messages[0]["content"] = "changed"
    tools.clear()
    other_messages = [{"role": "user", "content": "another task"}]
    other_prompt = second.build_prompt(other_messages, [])
    response = sampled_hello(tokenizer)
    first.record_generation("g0", response, {"text": "hello", "tool_calls": []})
    saved_response = list(response)
    response.clear()
    complete(first)
    updated = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "continue"},
    ]
    next_prompt = first.build_prompt(updated, TOOLS)
    prefix = original_prompt + saved_response
    assert next_prompt[: len(prefix)] == prefix
    second.record_generation("g0", saved_response, {"text": "hello", "tool_calls": []})
    complete(second)
    other_next = second.build_prompt(other_messages + updated[1:], [])
    other_prefix = other_prompt + saved_response
    assert other_next[: len(other_prefix)] == other_prefix
    assert other_prefix != prefix


def test_builder_cannot_trim_sampled_prefix(tokenizer):
    builder = QwenContinuousTokenBuilder(tokenizer)
    context = PiTokenContext(builder)
    messages = [{"role": "user", "content": "task"}]
    context.build_prompt(messages, [])
    context.record_generation("g0", sampled_hello(tokenizer), {"text": "hello", "tool_calls": []})
    complete(context)
    builder.merge_context_tokens = Mock(
        return_value=MergeResult(token_ids=[42], appended_token_count=1, kind="context", removed_prefix_token_count=1)
    )
    with pytest.raises(ValueError, match="changed existing token IDs"):
        context.build_prompt(
            messages + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "continue"}], []
        )


@pytest.mark.parametrize("tool_turn", [False, True])
def test_checkpoint_qwen_tokenizer_preserves_nonroundtrip_tokens(tool_turn):
    model_path = os.environ.get("PI_TEST_TOKENIZER_PATH")
    if not model_path:
        pytest.skip("Set PI_TEST_TOKENIZER_PATH to the remote Qwen checkpoint; no model download in this test")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    builder = create_continuous_token_builder(
        tokenizer, hf_model_type=config.model_type, chat_template_kwargs={"enable_thinking": False}
    )
    assert isinstance(builder, QwenContinuousTokenBuilder), "PI_TEST_TOKENIZER_PATH must point to a Qwen text model"
    context = PiTokenContext(builder)
    messages = [{"role": "user", "content": "task"}]
    tools = TOOLS if tool_turn else []
    prompt = context.build_prompt(messages, tools)
    body_ids = [token for char in "hello" for token in tokenizer.encode(char, add_special_tokens=False)]
    assert tokenizer.decode(body_ids) == "hello"
    assert body_ids != tokenizer.encode("hello", add_special_tokens=False)
    result = {"text": "hello", "tool_calls": []}
    assistant = {"role": "assistant", "content": "hello"}
    appended = {"role": "user", "content": "continue"}
    if tool_turn:
        # Deliberately preserve the model's whitespace, unlike Pi's JSON.stringify.
        raw_call = '\n<tool_call>\n{ "name": "lookup", "arguments": { "x": 7 } }\n</tool_call>'
        body_ids += tokenizer.encode(raw_call, add_special_tokens=False)
        result["tool_calls"] = [{"id": "lookup0", "name": "lookup", "arguments": {"x": 7}}]
        assistant["tool_calls"] = [
            {"id": "lookup0", "type": "function", "function": {"name": "lookup", "arguments": '{"x":7}'}}
        ]
        appended = {"role": "tool", "name": "lookup", "tool_call_id": "lookup0", "content": "record 7"}
    response = body_ids + [tokenizer.eos_token_id]
    context.record_generation("g0", response, result)
    complete(context, message=assistant)
    updated = messages + [assistant, appended]
    next_prompt = context.build_prompt(updated, tools)
    prefix = prompt + response
    assert next_prompt[: len(prefix)] == prefix
    assert next_prompt[len(prefix)] == tokenizer.encode("\n", add_special_tokens=False)[0]
    assert next_prompt != builder.build_initial_tokens(updated, tools=tools)
