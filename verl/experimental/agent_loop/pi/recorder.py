# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Record actual sampling inputs/outputs, independently of Pi's text transcript."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def is_tool_error(result: dict) -> bool:
    details = result.get("details")
    return bool(result.get("isError") or (isinstance(details, Mapping) and details.get("error")))


@dataclass
class PiTurn:
    generation_id: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float] | None
    extra_fields: dict[str, Any] = field(default_factory=dict)
    routed_experts: Any = None
    generation_seconds: float = 0.0
    event: dict[str, Any] = field(default_factory=dict)
    num_turns: int = 0


class PiTrainingRecorder:
    def __init__(self):
        self.turns: list[PiTurn] = []
        self._pending: PiTurn | None = None
        self._seen: set[str] = set()

    def record_generation(
        self,
        generation_id: str,
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        response_logprobs: list[float] | None,
        extra_fields: dict | None = None,
        routed_experts: Any = None,
        generation_seconds: float = 0,
        num_turns: int = 0,
    ):
        if not generation_id or generation_id in self._seen or self._pending is not None:
            raise ValueError(f"Duplicate or overlapping Pi generation: {generation_id}")
        if not prompt_ids or not response_ids:
            raise ValueError("Pi generation must have nonempty prompt and response tokens")
        if response_logprobs is not None:
            if len(response_logprobs) != len(response_ids):
                raise ValueError("Pi response tokens and logprobs have different lengths")
            if not all(math.isfinite(value) for value in response_logprobs):
                raise ValueError("Pi response logprobs must be finite")
        self._seen.add(generation_id)
        self._pending = PiTurn(
            generation_id=generation_id,
            prompt_ids=list(prompt_ids),
            response_ids=list(response_ids),
            response_logprobs=list(response_logprobs) if response_logprobs is not None else None,
            extra_fields=dict(extra_fields or {}),
            routed_experts=routed_experts,
            generation_seconds=generation_seconds,
            num_turns=num_turns,
        )
        return self._pending

    def complete_turn(self, event: dict):
        if self._pending is None or event.get("generation_id") != self._pending.generation_id:
            raise ValueError(f"Pi completed an unknown generation: {event.get('generation_id')}")
        message = event.get("assistant_message")
        if not isinstance(message, dict) or message.get("stopReason") in {"error", "aborted"}:
            raise ValueError("Pi turn did not complete with a valid assistant message")
        self._pending.event = dict(event)
        self._pending.num_turns += len(event.get("tool_results", []))
        self.turns.append(self._pending)
        self._pending = None

    def finalize(self, evaluation: dict | None, completion: dict | None) -> float:
        if self._pending is not None:
            raise ValueError("Pi session ended with an unfinished generation")
        if not self.turns:
            raise ValueError("Pi session produced no trainable turns")
        if not isinstance(evaluation, dict) or "reward" not in evaluation:
            raise ValueError("Pi session is missing its terminal evaluator reward")
        if not isinstance(completion, dict) or completion.get("turns") != len(self.turns):
            raise ValueError("Pi completion turn count does not match the recorded generations")
        reward = float(evaluation["reward"])
        if not math.isfinite(reward):
            raise ValueError("Pi evaluator reward must be finite")
        return reward
