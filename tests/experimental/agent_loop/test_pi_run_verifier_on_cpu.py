# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from examples.pi.tau2_telecom.verify_run import incremental_prefix_errors


def test_trace_audit_accepts_raw_tokens_across_multiple_generations():
    tokens = [
        {"generation_id": "g0", "prompt_ids": [1, 2], "response_ids": [7, 8]},
        {"generation_id": "g1", "prompt_ids": [1, 2, 7, 8, 42], "response_ids": [9]},
        {"generation_id": "g2", "prompt_ids": [1, 2, 7, 8, 42, 9, 43], "response_ids": [10]},
    ]
    assert incremental_prefix_errors(tokens) == []


@pytest.mark.parametrize(
    "next_prompt, error",
    [
        ([1, 2, 78, 42], "token prefix changed"),  # Same decoded text may use a different token.
        ([1, 2, 7, 42], "token prefix changed"),  # A generated boundary token was removed.
        ([2, 7, 8, 42], "token prefix changed"),  # History was truncated.
        ([1, 2, 7, 8], "no appended context"),
    ],
)
def test_trace_audit_rejects_changed_or_incomplete_incremental_prompt(next_prompt, error):
    tokens = [
        {"generation_id": "g0", "prompt_ids": [1, 2], "response_ids": [7, 8]},
        {"generation_id": "g1", "prompt_ids": next_prompt, "response_ids": [9]},
    ]
    failures = incremental_prefix_errors(tokens)
    assert len(failures) == 1 and "g1" in failures[0] and error in failures[0]
