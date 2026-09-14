# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One subprocess per trajectory; Pi owns the agent and tool lifecycle."""

import asyncio
import contextlib
import json
import logging
import os
import signal
from pathlib import Path

logger = logging.getLogger(__name__)


class PiSidecarError(RuntimeError):
    """The Pi process or its generation protocol failed."""


class PiSidecarClient:
    def __init__(self, *, node_binary: str, entrypoint: str, env: dict[str, str] | None = None):
        self.node_binary = node_binary
        self.entrypoint = str(Path(entrypoint).resolve())
        self.env = dict(env or {})
        self.process = None
        self.session_id = None
        self._events = asyncio.Queue()
        self._reader = None
        self._stderr = None
        self._write_lock = asyncio.Lock()

    async def start(self, payload: dict, timeout: float = 60):
        if self.process is not None:
            raise PiSidecarError("A Pi sidecar can only host one trajectory")
        self.session_id = payload["session_id"]
        self.process = await asyncio.create_subprocess_exec(
            self.node_binary,
            self.entrypoint,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self.env},
            limit=16 * 1024 * 1024,
            start_new_session=True,
        )
        self._reader = asyncio.create_task(self._read_stdout())
        self._stderr = asyncio.create_task(self._read_stderr())
        ready = await self.next_event(timeout)
        if ready.get("type") != "ready" or ready.get("protocol_version") != 2:
            raise PiSidecarError(f"Unsupported Pi startup handshake: {ready}")
        await self.send({"type": "start_session", **payload})

    async def next_event(self, timeout: float):
        event = await asyncio.wait_for(self._events.get(), timeout=timeout)
        if event.get("type") in {"sidecar_error", "session_error"}:
            raise PiSidecarError(str(event.get("error", event)))
        if event.get("session_id") not in (None, self.session_id):
            raise PiSidecarError("Pi returned an event for a different session")
        return event

    async def send(self, payload: dict):
        if self.process is None or self.process.returncode is not None:
            raise PiSidecarError("Pi sidecar is not running")
        async with self._write_lock:
            self.process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
            await self.process.stdin.drain()

    async def respond(self, request_id: str, result: dict):
        await self.send({"type": "response", "response_to": request_id, "ok": True, "result": result})

    async def _read_stdout(self):
        try:
            while line := await self.process.stdout.readline():
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise PiSidecarError("Pi protocol event must be a JSON object")
                await self._events.put(event)
            code = await self.process.wait()
            await self._events.put({"type": "sidecar_error", "error": f"Pi process exited with status {code}"})
        except Exception as exc:
            await self._events.put({"type": "sidecar_error", "error": f"Invalid Pi protocol: {exc}"})

    async def _read_stderr(self):
        while line := await self.process.stderr.readline():
            logger.info("pi[%s]: %s", self.session_id, line.decode(errors="replace").rstrip())

    async def close(self):
        if self.process is None:
            return
        try:
            if self.process.returncode is None:
                with contextlib.suppress(BrokenPipeError, ConnectionResetError, PiSidecarError):
                    await self.send({"type": "shutdown"})
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=5)
                    except TimeoutError:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(self.process.pid, signal.SIGKILL)
                        await self.process.wait()
        finally:
            # Reap any bridge descendants even if Node exited before its cleanup hook.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            tasks = [task for task in (self._reader, self._stderr) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
