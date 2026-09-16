#!/usr/bin/env python3
"""Opt-in timing launcher for the canonical Tau2 bridge, without editing Tau2.

Only selected lifecycle functions are wrapped. No global call profiler, fake
environment, alternate tools, or generation is introduced. Execute on vGPUN.
"""

import functools
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, time


def main():
    if sys.argv[1:] != ["-m", "tau2.domains.telecom.pi_bridge"]:
        raise SystemExit("This profiling launcher only supports -m tau2.domains.telecom.pi_bridge")
    directory = Path(os.environ["PI_STARTUP_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    session_id = os.environ["PI_TIMING_SESSION_ID"]
    stack = []
    sequence = 0
    with (directory / f"{session_id}-{os.getpid()}.jsonl").open("x") as output:

        @contextmanager
        def span(phase):
            nonlocal sequence
            sequence += 1
            span_id = sequence
            parent = stack[-1] if stack else None
            stack.append(span_id)
            started, started_unix = perf_counter(), time()
            ok = False
            try:
                yield
                ok = True
            finally:
                duration = perf_counter() - started
                stack.pop()
                output.write(
                    json.dumps(
                        {
                            "type": "startup_timing",
                            "source": "python",
                            "pid": os.getpid(),
                            "session_id": session_id,
                            "phase": phase,
                            "span_id": span_id,
                            "parent_span_id": parent,
                            "start_unix_s": started_unix,
                            "end_unix_s": started_unix + duration,
                            "duration_s": duration,
                            "ok": ok,
                        }
                    )
                    + "\n"
                )
                output.flush()

        def instrument(target, name, phase):
            original = getattr(target, name)

            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                with span(phase):
                    return original(*args, **kwargs)

            setattr(target, name, wrapped)

        with span("tau2.package_import"):
            importlib.import_module("tau2")
        with span("tau2.bridge_module_import"):
            bridge = importlib.import_module("tau2.domains.telecom.pi_bridge")
        with span("tau2.install_timing_hooks"):
            env = importlib.import_module("tau2.environment.environment")
            tools = importlib.import_module("tau2.domains.telecom.tools")
            user_tools = importlib.import_module("tau2.domains.telecom.user_tools")
            instrument(bridge, "get_environment", "tau2.get_environment")
            instrument(bridge.TelecomEnvironment, "__init__", "tau2.environment_ctor")
            instrument(bridge, "get_tasks", "tau2.task_catalog_load")
            instrument(bridge.TelecomPiBridge, "__init__", "tau2.bridge_ctor")
            instrument(bridge.TelecomPiBridge, "load_task", "tau2.load_task")
            instrument(bridge.TelecomPiBridge, "describe_tools", "tau2.describe_tools")
            instrument(env.Environment, "set_state", "tau2.set_state")
            instrument(env.Environment, "run_env_function_call", "tau2.initialization_action")
            instrument(tools.TelecomTools, "update_db", "tau2.agent_environment_update_db")
            instrument(user_tools.TelecomUserTools, "update_db", "tau2.user_environment_update_db")
        bridge.serve()


if __name__ == "__main__":
    main()
