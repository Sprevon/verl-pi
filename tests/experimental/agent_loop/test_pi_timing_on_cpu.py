# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from examples.pi.tau2_telecom.profile_startup import exclusive_seconds
from examples.pi.tau2_telecom.profile_timing import union_seconds


def test_parallel_requests_and_nested_tools_do_not_double_count_wall_time():
    assert union_seconds([(0, 4), (1, 2), (3, 7), (10, 12)]) == 9
    assert union_seconds([]) == 0
    assert union_seconds([(1, 1), (2, 3), (3, 4)]) == 2
    with pytest.raises(ValueError, match="Reversed interval"):
        union_seconds([(4, 3)])


def test_startup_nested_cross_process_spans_are_not_added_twice():
    parent = {"start_unix_s": 0.0, "end_unix_s": 12.0}
    python = {"start_unix_s": 1.0, "end_unix_s": 11.0}
    child = {"start_unix_s": 2.0, "end_unix_s": 8.0}
    marker = {"start_unix_s": 6.0, "end_unix_s": 6.0}
    rows = [parent, python, child, marker]
    assert exclusive_seconds(parent, rows) == 2.0
    assert exclusive_seconds(python, rows) == 4.0
    assert sum(exclusive_seconds(row, rows) for row in rows) == 12.0
