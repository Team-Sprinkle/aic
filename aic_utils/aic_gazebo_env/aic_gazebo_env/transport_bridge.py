"""IPC helpers for the persistent C++ Gazebo transport bridge."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import threading
import time
from typing import Any


class GazeboTransportBridgeError(RuntimeError):
    """Raised when the C++ transport bridge process fails."""


@dataclass(frozen=True)
class GazeboTransportBridgeConfig:
    """Configuration for the C++ Gazebo transport helper process."""

    world_name: str
    state_topic: str
    pose_topic: str
    helper_executable: str | None = None
    startup_timeout_s: float = 5.0
    request_timeout_s: float = 5.0
    startup_settle_s: float = 0.0


class GazeboTransportBridge:
    """Thin JSONL client around the persistent C++ Gazebo transport helper."""

    def __init__(self, config: GazeboTransportBridgeConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._lock = threading.Lock()

    @staticmethod
    def find_helper_explicit_or_on_path(helper_executable: str | None = None) -> str | None:
        """Resolve the helper executable from explicit config, env, or PATH."""
        candidates = [
            helper_executable,
            os.environ.get("AIC_GZ_TRANSPORT_BRIDGE_EXECUTABLE"),
            shutil.which("aic_gz_transport_bridge"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            resolved = shutil.which(candidate) if os.sep not in candidate else candidate
            if resolved and Path(resolved).exists():
                return resolved
        return None

    def start(self) -> None:
        """Start the helper process and verify it responds."""
        process = self._process
        if process is not None and process.poll() is None:
            return

        executable = self.find_helper_explicit_or_on_path(
            self._config.helper_executable
        )
        if executable is None:
            raise FileNotFoundError(
                "Could not resolve `aic_gz_transport_bridge`. "
                "Set `AIC_GZ_TRANSPORT_BRIDGE_EXECUTABLE` or source a workspace "
                "that installs the helper."
            )

        self._process = subprocess.Popen(
            [
                executable,
                "--state-topic",
                self._config.state_topic,
                "--pose-topic",
                self._config.pose_topic,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            self.request({"op": "ping"}, timeout_s=self._config.startup_timeout_s)
            if self._config.startup_settle_s > 0.0:
                time.sleep(self._config.startup_settle_s)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Shut down the helper process."""
        process = self._process
        if process is None:
            return

        if process.poll() is None:
            try:
                self.request({"op": "shutdown"}, timeout_s=1.0)
            except Exception:
                pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        self._process = None

    def request(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Send one JSON request and return one JSON response."""
        self.start()
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise GazeboTransportBridgeError("transport bridge process is not available")

        timeout = self._config.request_timeout_s if timeout_s is None else timeout_s
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            request_payload = dict(payload)
            request_payload["id"] = request_id
            process.stdin.write(json.dumps(request_payload) + "\n")
            process.stdin.flush()
            response_line = self._read_line(process, timeout_s=timeout)
            response = json.loads(response_line)
            if response.get("id") != request_id:
                raise GazeboTransportBridgeError(
                    f"transport bridge response id mismatch: expected {request_id}, got {response.get('id')}"
                )
            if response.get("ok") is not True:
                raise GazeboTransportBridgeError(
                    str(response.get("error", "transport bridge request failed"))
                )
            return response

    def _read_line(
        self,
        process: subprocess.Popen[str],
        *,
        timeout_s: float,
    ) -> str:
        stdout = process.stdout
        stderr = process.stderr
        if stdout is None:
            raise GazeboTransportBridgeError("transport bridge stdout is not available")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr_text = ""
                if stderr is not None:
                    stderr_text = stderr.read().strip()
                raise GazeboTransportBridgeError(
                    "transport bridge exited unexpectedly"
                    + (f": {stderr_text}" if stderr_text else "")
                )
            remaining = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([stdout], [], [], remaining)
            if ready:
                line = stdout.readline()
                if line:
                    return line.strip()
        raise GazeboTransportBridgeError("timed out waiting for transport bridge response")
