# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Preserve sampled assistant tokens while Pi appends messages to a trajectory."""

import json
from copy import deepcopy

from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder


class PiTokenContext:
    """One trajectory's message provenance and incremental runtime token stream.

    Message snapshots detect history edits; only a matched generation/turn pair
    associates an assistant message with sampled tokens. Other appended messages
    are context and go through the native Continuous Token context builder.
    """

    def __init__(self, builder: ContinuousTokenBuilder):
        self.builder = builder
        self._messages: list[dict] | None = None
        self._tools: list[dict] | None = None
        self._runtime_ids: list[int] = []
        self._phase = "prompt"
        self._generation_id: str | None = None
        self._result: dict | None = None

    def build_prompt(self, messages: list[dict], tools: list[dict]) -> list[int]:
        if self._phase != "prompt":
            raise ValueError("Pi requested a prompt before the previous generation/turn completed")
        if self._messages is None:
            prompt_ids = self.builder.build_initial_tokens(messages, tools=tools)
        else:
            if tools != self._tools:
                raise ValueError("Pi tool definitions changed during incremental rollout")
            if messages[: len(self._messages)] != self._messages:
                raise ValueError("Pi history changed: incremental rollout requires an unchanged message prefix")
            if len(messages) <= len(self._messages):
                raise ValueError("Pi next generation must append context after the completed assistant")
            merged = self.builder.merge_context_tokens(self._messages, messages, list(self._runtime_ids), tools=tools)
            if merged.removed_prefix_token_count or merged.token_ids[: len(self._runtime_ids)] != self._runtime_ids:
                raise ValueError("Pi context merge changed existing token IDs; this boundary is unsupported")
            prompt_ids = merged.token_ids
        if not prompt_ids:
            raise ValueError("Pi incremental prompt is empty")
        self._messages = deepcopy(messages)
        self._tools = deepcopy(tools)
        self._runtime_ids = list(prompt_ids)
        self._phase = "response"
        return list(prompt_ids)

    def record_generation(self, generation_id: str, response_ids: list[int], result: dict) -> None:
        if self._phase != "response" or not generation_id or not response_ids:
            raise ValueError("Pi generation has no matching prompt or has empty response tokens")
        merged = self.builder.merge_assistant_tokens(list(self._runtime_ids), list(response_ids))
        if merged.token_ids != self._runtime_ids + list(response_ids):
            raise ValueError("Pi assistant merge must preserve the exact sampled token IDs")
        self._runtime_ids = list(merged.token_ids)
        self._generation_id = generation_id
        self._result = deepcopy(result)
        self._phase = "turn"

    def complete_turn(self, event: dict) -> None:
        if self._phase != "turn" or event.get("generation_id") != self._generation_id:
            raise ValueError("Pi completed a turn without a matching incremental generation")
        message = event.get("assistant_message_openai")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError("Pi turn is missing its assistant_message_openai projection (protocol v3 required)")
        # Node serializes function.arguments with JSON.stringify. Compare parsed
        # arguments to the host result so whitespace/Unicode escaping is immaterial.
        try:
            calls = []
            for call in message.get("tool_calls", []):
                function = call["function"]
                arguments = function["arguments"]
                calls.append(
                    {
                        "id": call["id"],
                        "name": function["name"],
                        "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments,
                    }
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Pi turn contains malformed assistant tool calls") from exc
        if message.get("content") != self._result.get("text", "") or calls != self._result.get("tool_calls", []):
            raise ValueError("Pi changed the generated assistant message before turn completion")
        self._messages.append(deepcopy(message))
        self._generation_id = None
        self._result = None
        self._phase = "prompt"
