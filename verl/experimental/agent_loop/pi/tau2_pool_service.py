# Copyright 2026 The verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Actor-local warm Python service and light stdio proxy; run with Tau2 Python.

Every lease calls the original bridge.serve(), which creates a NEW bridge and
environment. Only imported modules persist across leases. No tools are reimplemented.
"""

import argparse
import importlib
import json
import os
import resource
import socket
import sys
import threading
import time


def watchdog(parent_pid):
    while os.getppid() == parent_pid:
        time.sleep(1)
    os._exit(1)


def serve(path, parent_pid):
    threading.Thread(target=watchdog, args=(parent_pid,), daemon=True).start()
    started = time.perf_counter()
    bridge = importlib.import_module("tau2.domains.telecom.pi_bridge")
    import_s = time.perf_counter() - started
    sessions = 0
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(path)
        os.chmod(path, 0o600)
        server.listen(4)
        print(
            json.dumps(
                {
                    "ready": True,
                    "pid": os.getpid(),
                    "import_s": import_s,
                    "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            ),
            flush=True,
        )
        while True:
            connection, _ = server.accept()
            with connection, connection.makefile("r", encoding="utf-8") as incoming:
                with connection.makefile("w", encoding="utf-8", buffering=1) as outgoing:
                    header = json.loads(incoming.readline())
                    outgoing.write(
                        json.dumps(
                            {
                                "ok": True,
                                "pid": os.getpid(),
                                "sessions": sessions,
                                "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            }
                        )
                        + "\n"
                    )
                    outgoing.flush()
                    if header["op"] == "ping":
                        continue
                    if header["op"] != "session":
                        raise ValueError("Invalid Harness connection")
                    sessions += 1
                    old_stdin, old_stdout = sys.stdin, sys.stdout
                    try:
                        sys.stdin, sys.stdout = incoming, outgoing
                        bridge.serve()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        sys.stdin, sys.stdout = old_stdin, old_stdout


def proxy(path):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(path)
        connection.sendall(b'{"op":"session"}\n')
        incoming = connection.makefile("rb")
        if not json.loads(incoming.readline())["ok"]:
            raise RuntimeError("Harness refused session")

        def send_requests():
            try:
                for line in sys.stdin.buffer:
                    connection.sendall(line)
            finally:
                connection.shutdown(socket.SHUT_WR)

        threading.Thread(target=send_requests, daemon=True).start()
        for line in incoming:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    if sys.argv[1:2] == ["proxy"]:
        if sys.argv[2:] != ["-m", "tau2.domains.telecom.pi_bridge"]:
            raise SystemExit("Proxy only supports -m tau2.domains.telecom.pi_bridge")
        proxy(os.environ["PI_HARNESS_SOCKET"])
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve",))
    parser.add_argument("--socket", default=os.environ.get("PI_HARNESS_SOCKET"))
    parser.add_argument("--parent-pid", type=int, default=os.getppid())
    args = parser.parse_args()
    if not args.socket:
        parser.error("--socket or PI_HARNESS_SOCKET required")
    serve(args.socket, args.parent_pid)
