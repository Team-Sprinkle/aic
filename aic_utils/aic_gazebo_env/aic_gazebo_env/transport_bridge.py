"""IPC helpers for the persistent C++ Gazebo transport bridge."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import json
import os
from pathlib import Path
import select
import subprocess
import threading
import time
from typing import Any

from .discovery import resolve_transport_helper_executable


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
    require_pose_for_ready: bool = True


class GazeboTransportBridge:
    """Thin JSONL client around the persistent C++ Gazebo transport helper."""

    def __init__(self, config: GazeboTransportBridgeConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._stderr_lines: deque[str] = deque(maxlen=20)
        self._stderr_thread: threading.Thread | None = None
        self._startup_ok = False
        self._ready_ok = False
        self._last_status: dict[str, Any] | None = None

    @staticmethod
    def find_helper_explicit_or_on_path(helper_executable: str | None = None) -> str | None:
        """Resolve the helper executable from explicit config, env, or PATH."""
        resolution = resolve_transport_helper_executable(helper_executable)
        return resolution.resolved_path

    def start(self) -> None:
        """Start the helper process and verify it responds."""
        process = self._process
        if process is not None and process.poll() is None:
            return
        self._startup_ok = False
        self._ready_ok = False
        self._last_status = None

        resolution = resolve_transport_helper_executable(self._config.helper_executable)
        executable = resolution.resolved_path
        if executable is None:
            searched = "\n".join(f"  - {path}" for path in resolution.searched_locations)
            setup_hint = ""
            if resolution.discovered_setup_script:
                setup_hint = (
                    "\nSetup script found but not sourced. Try:\n"
                    f"  bash -lc \"source {resolution.discovered_setup_script} && <your command>\""
                )
            raise FileNotFoundError(
                "Could not resolve `aic_gz_transport_bridge`. "
                f"status={resolution.status}. setup_status={resolution.setup_status}. "
                f"Searched:\n{searched}{setup_hint}"
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
        self._stderr_lines.clear()
        self._stderr_thread = threading.Thread(
            target=self._stderr_reader_loop,
            name="aic-gz-transport-bridge-stderr",
            daemon=True,
        )
        self._stderr_thread.start()
        try:
            self._request_no_start(
                {"op": "ping"},
                timeout_s=self._config.startup_timeout_s,
            )
            self._startup_ok = True
            self.wait_until_ready(timeout_s=self._config.startup_timeout_s)
            if self._config.startup_settle_s > 0.0:
                time.sleep(self._config.startup_settle_s)
        except Exception as exc:
            status = self._last_status
            stderr_snippet = self.recent_stderr_snippet()
            self.close()
            raise GazeboTransportBridgeError(
                "transport helper failed readiness handshake"
                f" (last_status={status}, stderr={stderr_snippet or '<empty>'}): {exc}"
            ) from exc

    def close(self) -> None:
        """Shut down the helper process."""
        process = self._process
        if process is None:
            return

        if process.poll() is None:
            try:
                self._request_no_start({"op": "shutdown"}, timeout_s=1.0)
            except Exception:
                pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        thread = self._stderr_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)
        self._stderr_thread = None
        self._process = None

    def request(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Send one JSON request and return one JSON response."""
        self.start()
        return self._request_no_start(payload, timeout_s=timeout_s)

    def _request_no_start(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Send one JSON request without recursively triggering start()."""
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
            deadline = time.monotonic() + max(float(timeout), 0.1)
            skipped_ids: list[int | None] = []
            while True:
                remaining = max(deadline - time.monotonic(), 0.0)
                if remaining <= 0.0:
                    raise GazeboTransportBridgeError(
                        "transport bridge timed out waiting for the matching response id: "
                        f"expected {request_id}, skipped={skipped_ids}"
                    )
                response_line = self._read_line(process, timeout_s=remaining)
                response = json.loads(response_line)
                response_id = response.get("id")
                if response_id == request_id:
                    break
                skipped_ids.append(response_id)
            if response.get("ok") is not True:
                raise GazeboTransportBridgeError(
                    str(response.get("error", "transport bridge request failed"))
                )
            return response

    def wait_until_ready(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Block until the helper reports initial state readiness."""
        timeout = self._config.startup_timeout_s if timeout_s is None else timeout_s
        parse_failures_before = 0
        try:
            initial_status = self._request_no_start({"op": "status"}, timeout_s=min(timeout, 1.0))
            self._last_status = initial_status
            parse_failures_before = int(initial_status.get("state_parse_failures", 0))
        except Exception:
            initial_status = None
        response = self._request_no_start(
            {
                "op": "wait_until_ready",
                "timeout_ms": int(timeout * 1000),
                "require_pose": self._config.require_pose_for_ready,
            },
            timeout_s=timeout + 1.0,
        )
        self._last_status = response
        parse_failures_after = int(response.get("state_parse_failures", 0))
        if parse_failures_after > parse_failures_before and int(response.get("state_generation", 0)) <= 0:
            raise GazeboTransportBridgeError(
                "transport helper readiness failed while state parse failures increased: "
                f"status={response}"
            )
        self._ready_ok = True
        return response

    def health_flags(self) -> dict[str, Any]:
        """Return helper health flags for runtime diagnostics."""
        return {
            "helper_startup_ok": self._startup_ok,
            "helper_ready_ok": self._ready_ok,
            "helper_last_status": self._last_status,
        }

    def recent_stderr_snippet(self) -> str:
        """Return a recent stderr tail for diagnostics."""
        return "\n".join(self._stderr_lines).strip()

    def _stderr_reader_loop(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            line = process.stderr.readline()
            if not line:
                break
            self._stderr_lines.append(line.rstrip())

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
