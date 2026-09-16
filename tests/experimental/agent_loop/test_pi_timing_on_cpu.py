# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from examples.pi.tau2_telecom.profile_timing import union_seconds


def test_parallel_requests_and_nested_tools_do_not_double_count_wall_time():
    assert union_seconds([(0, 4), (1, 2), (3, 7), (10, 12)]) == 9
    assert union_seconds([]) == 0
    assert union_seconds([(1, 1), (2, 3), (3, 4)]) == 2
    with pytest.raises(ValueError, match="Reversed interval"):
        union_seconds([(4, 3)])
