from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any


class IPCError(RuntimeError):
    pass


@dataclass
class IPCMessage:
    type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "payload": self.payload}


class JsonLineConnection:
    def __init__(self, sock: socket.socket, *, name: str = "ipc"):
        self._sock = sock
        self._name = name
        self._file = sock.makefile("rwb", buffering=0)
        self._lock = threading.Lock()

    def set_timeout(self, timeout_sec: float | None) -> None:
        self._sock.settimeout(timeout_sec)

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            self._sock.close()

    def send(self, msg_type: str, payload: dict[str, Any] | None = None) -> None:
        message = {"type": msg_type, "payload": payload or {}}
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            try:
                self._file.write(data)
            except OSError as ex:
                raise IPCError(f"{self._name}: failed to send {msg_type}: {ex}") from ex

    def recv(self, timeout_sec: float | None = None) -> IPCMessage:
        previous_timeout = self._sock.gettimeout()
        if timeout_sec is not None:
            self._sock.settimeout(timeout_sec)
        try:
            line = self._file.readline()
        except socket.timeout as ex:
            raise TimeoutError(f"{self._name}: timed out waiting for message") from ex
        except OSError as ex:
            raise IPCError(f"{self._name}: failed to receive message: {ex}") from ex
        finally:
            if timeout_sec is not None:
                self._sock.settimeout(previous_timeout)

        if not line:
            raise IPCError(f"{self._name}: peer disconnected")
        try:
            raw = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as ex:
            raise IPCError(f"{self._name}: invalid JSON line: {line!r}") from ex
        msg_type = raw.get("type")
        payload = raw.get("payload", {})
        if not isinstance(msg_type, str) or not isinstance(payload, dict):
            raise IPCError(f"{self._name}: invalid message envelope: {raw!r}")
        return IPCMessage(type=msg_type, payload=payload)


class IPCServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, *, backlog: int = 1):
        self.host = host
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(backlog)
        self.port = int(self._sock.getsockname()[1])

    def close(self) -> None:
        self._sock.close()

    def accept(self, timeout_sec: float | None = None) -> JsonLineConnection:
        self._sock.settimeout(timeout_sec)
        try:
            conn, addr = self._sock.accept()
        except socket.timeout as ex:
            raise TimeoutError("Timed out waiting for Gazebo RL bridge connection") from ex
        return JsonLineConnection(conn, name=f"server:{addr[0]}:{addr[1]}")


def connect_with_retry(
    host: str,
    port: int,
    *,
    timeout_sec: float = 60.0,
    retry_period_sec: float = 0.25,
) -> JsonLineConnection:
    deadline = time.monotonic() + timeout_sec
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=retry_period_sec)
            return JsonLineConnection(sock, name=f"client:{host}:{port}")
        except OSError as ex:
            last_error = ex
            time.sleep(retry_period_sec)
    raise TimeoutError(f"Could not connect to Gazebo RL server at {host}:{port}: {last_error}")
