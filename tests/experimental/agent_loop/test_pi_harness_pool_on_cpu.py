# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest
import ray

from verl.experimental.agent_loop.pi.client import PiSidecarClient
from verl.experimental.agent_loop.pi.harness_pool import RayHarnessClient, get_pool, signature
from verl.experimental.agent_loop.pi.recorder import is_tool_error


async def session(client, config, task_id):
    results, initial = [], None
    evaluation = None
    count = 0
    try:
        await client.start(
            {
                "session_id": uuid4().hex,
                "task_id": task_id,
                "cwd": config["cwd"],
                "agent_dir": str(Path(config["cwd"]) / ".pi/agent"),
                "training_extension": str(Path(config["cwd"]) / ".pi/extensions/agent-r1-training.ts"),
                "prompt_entry_type": "agent-r1-task-prompt",
                "evaluation_entry_type": "agent-r1-evaluation",
                "max_turns": 3,
            },
            timeout=120,
        )
        while True:
            event = await client.next_event(120)
            if event["type"] == "generation_request":
                count += 1
                initial = initial or (event["messages"], event["tools"])
                names = {tool["function"]["name"] for tool in event["tools"]}
                if count <= 2:
                    tool = "check_network_status" if count == 1 else "toggle_airplane_mode"
                    assert tool in names
                    response = {"text": "", "tool_calls": [{"id": f"probe{count}", "name": tool, "arguments": {}}]}
                else:
                    response = {"text": "Done", "tool_calls": []}
                await client.respond(event["id"], response)
            elif event["type"] == "step_complete":
                assert not any(is_tool_error(result) for result in event["tool_results"])
                results.extend(result["content"] for result in event["tool_results"])
            elif event["type"] == "evaluation_result":
                evaluation = event["result"]
            elif event["type"] == "session_complete":
                assert event["turns"] == 3
                break
        assert evaluation is not None and len(results) == 2
        return initial, results, evaluation
    finally:
        await client.close()


@pytest.mark.skipif(not os.environ.get("PI_TEST_TAU2_ROOT"), reason="Requires remote canonical Tau2 and Node SDK")
@pytest.mark.asyncio
async def test_ray_pool_bounds_capacity_and_resets_mutated_tasks():
    project = Path(__file__).resolve().parents[3]
    config = {
        "node_binary": os.environ["PI_NODE_BINARY"],
        "entrypoint": str(project / "verl/experimental/agent_loop/pi/sidecar/main.mjs"),
        "cwd": os.environ["PI_TEST_TAU2_ROOT"],
        "env": {"TAU2_PI_PYTHON": os.environ["TAU2_PI_PYTHON"], "PI_STARTUP_PROFILE": "0"},
    }
    tasks = [
        "[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]",
        "[mobile_data_issue]data_saver_mode_on|user_abroad_roaming_enabled_off[PERSONA:Easy]",
    ]
    name = f"pi-pool-test-{uuid4().hex}"
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    pool = get_pool(name, config, 1)
    try:
        preparation = await pool.prepare.remote()
        first_pid = preparation["slots"][0]["pid"]
        slot = await pool.acquire.remote("held", signature(config), 30)
        waiter = asyncio.ensure_future(pool.acquire.remote("waiting", signature(config), 30))
        await asyncio.sleep(0.1)
        assert not waiter.done(), "A one-slot pool admitted a second simultaneous lease"
        await pool.release.remote("waiting")
        with pytest.raises((asyncio.CancelledError, ray.exceptions.RayTaskError, ray.exceptions.TaskCancelledError)):
            await waiter
        await pool.release.remote("held")
        with pytest.raises(Exception, match="Stale"):
            await slot.next_event.remote("held", 1)

        reference = {}
        for task in tasks:
            client = PiSidecarClient(
                node_binary=config["node_binary"],
                entrypoint=config["entrypoint"],
                env={**config["env"], "TAU2_TELECOM_TASK_ID": task},
            )
            reference[task] = await session(client, config, task)
        for task in (tasks[0], tasks[1], tasks[0]):
            client = RayHarnessClient(
                pool_name=name,
                pool_size=1,
                cpus_per_actor=1,
                config=config,
                env={"TAU2_TELECOM_TASK_ID": task},
            )
            assert await session(client, config, task) == reference[task]
        same = await pool.prepare.remote()
        assert same["slots"][0]["pid"] == first_pid
        await pool.acquire.remote("inspect", signature(config), 30)
        status = await pool.release.remote("inspect")
        assert status["pid"] == first_pid and status["sessions"] == 3
    finally:
        await pool.shutdown.remote()
        ray.kill(pool, no_restart=True)
        ray.shutdown()
