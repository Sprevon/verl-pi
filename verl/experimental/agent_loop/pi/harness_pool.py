# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental bounded Ray Harness pool; warm imports, fresh Pi/task state."""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import signal
import tempfile
from pathlib import Path
from time import perf_counter

import ray

from verl.experimental.agent_loop.pi.client import PiSidecarClient, PiSidecarError

logger = logging.getLogger(__name__)


def signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


@ray.remote
class HarnessActor:
    def __init__(self, config):
        self.config = config
        self.directory = tempfile.TemporaryDirectory(prefix="pi-harness-")
        self.socket = str(Path(self.directory.name) / "bridge.sock")
        self.process = self.client = self.stderr_task = None
        self.lease = None
        self.ready = None

    async def prepare(self):
        if self.ready is not None:
            if self.process.returncode is not None:
                raise PiSidecarError("Warm Tau2 interpreter exited")
            return self.ready
        if self.process is not None:
            raise PiSidecarError("Harness interpreter had an incomplete startup; recreate this slot")
        config = self.config
        env = {**os.environ, **config["env"]}
        python = env.get("TAU2_PI_REAL_PYTHON") or env["TAU2_PI_PYTHON"]
        source = str(Path(config["cwd"]) / "src")
        env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = ""
        started = perf_counter()
        self.process = await asyncio.create_subprocess_exec(
            python,
            str(Path(__file__).with_name("tau2_pool_service.py")),
            "serve",
            "--socket",
            self.socket,
            cwd=config["cwd"],
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        async def drain_stderr():
            while line := await self.process.stderr.readline():
                logger.info("harness[%s]: %s", self.process.pid, line.decode(errors="replace").rstrip())

        self.stderr_task = asyncio.create_task(drain_stderr())
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout=120)
        self.ready = json.loads(line)
        if not self.ready.get("ready"):
            raise PiSidecarError("Warm Tau2 interpreter did not become ready")
        self.ready.update(prepare_s=perf_counter() - started, actor_pid=os.getpid())
        return self.ready

    async def _ping(self):
        reader, writer = await asyncio.open_unix_connection(self.socket)
        try:
            writer.write(b'{"op":"ping"}\n')
            await writer.drain()
            return json.loads(await asyncio.wait_for(reader.readline(), timeout=10))
        finally:
            writer.close()
            await writer.wait_closed()

    def _check_lease(self, lease):
        if self.lease != lease or self.client is None:
            raise PiSidecarError("Stale or unallocated Harness lease")

    async def start(self, lease, payload, env, timeout):
        if self.lease is not None:
            raise PiSidecarError("Harness slot is already occupied")
        await self.prepare()
        self.lease = lease
        base_env = {**self.config["env"], **env}
        python = base_env.get("TAU2_PI_REAL_PYTHON") or base_env["TAU2_PI_PYTHON"]
        base_env.update(
            TAU2_PI_PYTHON=str(Path(__file__).with_name("tau2_pool_proxy.sh")),
            PI_HARNESS_PYTHON=python,
            PI_HARNESS_SOCKET=self.socket,
            PI_HARNESS_OWNER_PID=str(os.getpid()),
        )
        self.client = PiSidecarClient(
            node_binary=self.config["node_binary"], entrypoint=self.config["entrypoint"], env=base_env
        )
        await self.client.start(payload, timeout)

    async def next_event(self, lease, timeout):
        self._check_lease(lease)
        return await self.client.next_event(timeout)

    async def respond(self, lease, request_id, result):
        self._check_lease(lease)
        await self.client.respond(request_id, result)

    async def close(self, lease):
        if self.lease not in (None, lease):
            raise PiSidecarError("Cannot close another trajectory's Harness")
        if self.client is not None:
            await self.client.close()
        # A new connection is accepted only after canonical serve() has returned.
        # This is the reset fence before this slot becomes available again.
        status = await self._ping()
        self.client = None
        self.lease = None
        return status

    async def shutdown(self):
        try:
            if self.client is not None:
                await self.client.close()
        finally:
            if self.process is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
            if self.stderr_task is not None:
                self.stderr_task.cancel()
                await asyncio.gather(self.stderr_task, return_exceptions=True)
            self.directory.cleanup()


@ray.remote(num_cpus=0)
class HarnessPool:
    def __init__(self, config, size, cpus_per_actor):
        self.config_signature = signature(config)
        self.slots = [HarnessActor.options(num_cpus=cpus_per_actor).remote(config) for _ in range(size)]
        self.available = asyncio.Queue()
        self.leases = {}
        self.waiting = {}
        self.lock = asyncio.Lock()
        self.ready = None
        self.failure = None

    async def prepare(self):
        async with self.lock:
            if self.failure:
                raise PiSidecarError(self.failure)
            if self.ready is None:
                started = perf_counter()
                try:
                    slots = await asyncio.gather(*[slot.prepare.remote() for slot in self.slots])
                except BaseException as error:
                    self.failure = f"Harness pool initialization failed: {error}"
                    raise
                self.ready = {"prepare_s": perf_counter() - started, "slots": slots}
                for index in range(len(self.slots)):
                    self.available.put_nowait(index)
            return self.ready

    async def acquire(self, lease, config_signature, timeout):
        if config_signature != self.config_signature:
            raise PiSidecarError("Harness pool name was reused with a different configuration")
        if lease in self.leases or lease in self.waiting:
            raise PiSidecarError("Duplicate Harness lease")
        self.waiting[lease] = asyncio.current_task()
        try:
            await self.prepare()
            index = await asyncio.wait_for(self.available.get(), timeout=timeout)
            self.leases[lease] = index
            return self.slots[index]
        finally:
            self.waiting.pop(lease, None)

    async def release(self, lease):
        pending = self.waiting.pop(lease, None)
        if pending is not None:
            pending.cancel()
        index = self.leases.pop(lease, None)
        if index is None:
            return
        try:
            status = await self.slots[index].close.remote(lease)
        except BaseException as error:
            self.failure = f"Harness slot failed cleanup and was quarantined: {error}"
            for task in list(self.waiting.values()):
                task.cancel()
            raise
        else:
            self.available.put_nowait(index)
            return status

    async def shutdown(self):
        self.failure = "Harness pool is shut down"
        for task in list(self.waiting.values()):
            task.cancel()
        try:
            return await asyncio.gather(*[slot.shutdown.remote() for slot in self.slots], return_exceptions=True)
        finally:
            for slot in self.slots:
                ray.kill(slot, no_restart=True)


def get_pool(name, config, size, cpus_per_actor=1):
    if not name or size < 1 or cpus_per_actor <= 0:
        raise ValueError("Harness pool requires a name, positive capacity, and positive CPU reservation")
    return HarnessPool.options(name=name, get_if_exists=True).remote(config, size, cpus_per_actor)


class RayHarnessClient:
    """PiSidecarClient-shaped transport; token generation remains in the Worker."""

    def __init__(self, *, pool_name, pool_size, cpus_per_actor, config, env):
        self.pool = get_pool(pool_name, config, pool_size, cpus_per_actor)
        self.config_signature = signature(config)
        self.env = env
        self.slot = self.lease = None
        self.final_status = None

    async def start(self, payload, timeout=300):
        self.lease = payload["session_id"]
        self.slot = await self.pool.acquire.remote(self.lease, self.config_signature, timeout)
        await self.slot.start.remote(self.lease, payload, self.env, timeout)

    async def next_event(self, timeout):
        return await self.slot.next_event.remote(self.lease, timeout)

    async def respond(self, request_id, result):
        await self.slot.respond.remote(self.lease, request_id, result)

    async def close(self):
        if self.lease is not None:
            self.final_status = await self.pool.release.remote(self.lease)
            self.lease = self.slot = None
