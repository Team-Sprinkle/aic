#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import json
import os
import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when YAML support is missing
    yaml = None

from aic_example_policies.controllers import PIDController
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


class PIDControllerMode(Enum):
    ON = auto()
    OFF = auto()
    CHEATCODE_DEFAULT = auto()


@dataclass(frozen=True)
class ControllerConfig:
    config_path: Path
    raw_config: dict[str, Any]
    pid_x_gains: tuple[float, float, float]
    pid_y_gains: tuple[float, float, float]
    pid_mode: PIDControllerMode
    cheatcode_i_gain: float
    cheatcode_max_integrator_windup: float
    xy_alignment_tolerance_m: float
    xy_alignment_stable_cycles: int

    pid_plots_enabled: bool
    plot_output_dir: Path

    transport_initial_z_offset_m: float
    transport_duration_sec: float
    transport_dt_sec: float

    transport_force_injection_enabled: bool
    transport_force_injection_randomize_time: bool
    transport_force_injection_config_time_sec: float | None
    transport_force_injection_randomize_force: bool
    transport_force_injection_min_n: float
    transport_force_injection_max_n: float
    transport_force_injection_force_n: float | None
    transport_force_injection_randomize_axis: bool
    transport_force_injection_axis: str | None
    transport_force_injection_randomize_direction: bool
    transport_force_injection_direction_sign: float | None
    transport_force_injection_seed: int

    recover_lift_m: float
    recover_duration_sec: float
    recover_motion_dt_sec: float
    recover_pid_x_gains: tuple[float, float, float]
    recover_pid_y_gains: tuple[float, float, float]

    pre_insertion_z_offset_m: float

    insert_min_z_offset_m: float
    insert_duration_sec: float
    insert_dt_sec: float
    insert_wait_timeout_sec: float

    force_log_interval_sec: float
    collision_force_threshold_n: float
    collision_duration_threshold_sec: float
    contact_force_threshold_n: float
    contact_duration_threshold_sec: float
    lateral_contact_force_threshold_n: float
    lateral_contact_ratio_threshold: float
    lateral_contact_duration_threshold_sec: float
    stuck_fz_threshold_n: float
    stuck_duration_threshold_sec: float
    stuck_z_stationary_threshold_m: float
    stuck_force_increase_threshold_n: float
    stuck_debug_log_interval_sec: float


class ConfigLoader:
    @staticmethod
    def _coerce_pid_mode_name(raw_mode: Any) -> str:
        mode_name = str(raw_mode).strip().upper()
        if mode_name in ("TRUE", "1", "YES", "ON"):
            return "ON"
        if mode_name in ("FALSE", "0", "NO", "OFF"):
            return "OFF"
        return mode_name

    @classmethod
    def load(cls, pre_insertion_default: float = 0.08) -> ControllerConfig:
        config_path = cls._resolve_config_path()
        config = cls._load_config_file(config_path)

        def cfg_value(key_path: str, default: Any, env_name: str | None = None) -> Any:
            if env_name:
                env_value = os.environ.get(env_name)
                if env_value is not None:
                    return env_value
            value = cls._lookup_config(config, key_path)
            return default if value is None else value

        def cfg_bool(key_path: str, default: bool, env_name: str | None = None) -> bool:
            value = cfg_value(key_path, default, env_name)
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() not in ("0", "false", "no", "off", "")

        def cfg_optional_float(key_path: str, env_name: str | None = None) -> float | None:
            if env_name:
                env_value = os.environ.get(env_name)
                if env_value is not None and env_value.strip().lower() not in (
                    "",
                    "null",
                    "none",
                ):
                    return float(env_value)
            value = cls._lookup_config(config, key_path)
            if value is None:
                return None
            return float(value)

        def cfg_optional_str(key_path: str, env_name: str | None = None) -> str | None:
            if env_name:
                env_value = os.environ.get(env_name)
                if env_value is not None and env_value.strip().lower() not in (
                    "",
                    "null",
                    "none",
                ):
                    return env_value
            value = cls._lookup_config(config, key_path)
            if value is None:
                return None
            return str(value)

        pid_x_gains = (
            float(cfg_value("controller.pid.x.kp", 17.0, "AIC_PID_KP_X")),
            float(cfg_value("controller.pid.x.ki", 0.08, "AIC_PID_KI_X")),
            float(cfg_value("controller.pid.x.kd", 2.0, "AIC_PID_KD_X")),
        )
        pid_y_gains = (
            float(cfg_value("controller.pid.y.kp", 17.0, "AIC_PID_KP_Y")),
            float(cfg_value("controller.pid.y.ki", 0.08, "AIC_PID_KI_Y")),
            float(cfg_value("controller.pid.y.kd", 2.0, "AIC_PID_KD_Y")),
        )

        pid_mode_name = cls._coerce_pid_mode_name(
            cfg_value("controller.mode", "ON", "AIC_PID_CONTROLLER_MODE")
        )
        try:
            pid_mode = PIDControllerMode[pid_mode_name]
        except KeyError as ex:
            valid_modes = ", ".join(mode.name for mode in PIDControllerMode)
            raise RuntimeError(
                "AIC_PID_CONTROLLER_MODE must be one of "
                f"{valid_modes}; got '{pid_mode_name}'"
            ) from ex

        configured_dir = os.environ.get("AIC_PID_TUNING_PLOTS_DIR")
        if not configured_dir:
            configured_dir = config.get("plots", {}).get("output_dir")
        if configured_dir:
            plot_output_dir = Path(configured_dir).expanduser()
        else:
            plot_output_dir = Path.cwd() / "outputs" / "pid_tuning_plots"

        cfg = ControllerConfig(
            config_path=config_path,
            raw_config=config,
            pid_x_gains=pid_x_gains,
            pid_y_gains=pid_y_gains,
            pid_mode=pid_mode,
            cheatcode_i_gain=float(
                cfg_value(
                    "controller.cheatcode_i_gain", 0.15, "AIC_PID_CHEATCODE_I_GAIN"
                )
            ),
            cheatcode_max_integrator_windup=float(
                cfg_value(
                    "controller.cheatcode_max_integrator_windup",
                    0.05,
                    "AIC_PID_CHEATCODE_MAX_INTEGRATOR_WINDUP",
                )
            ),
            xy_alignment_tolerance_m=float(
                cfg_value(
                    "controller.xy_alignment_tolerance_m",
                    0.01,
                    "AIC_PID_XY_ALIGNMENT_TOLERANCE_M",
                )
            ),
            xy_alignment_stable_cycles=int(
                cfg_value(
                    "controller.xy_alignment_stable_cycles",
                    5,
                    "AIC_PID_XY_ALIGNMENT_STABLE_CYCLES",
                )
            ),
            pid_plots_enabled=cfg_bool("plots.enabled", False, "AIC_PID_TUNING_PLOTS"),
            plot_output_dir=plot_output_dir,
            transport_initial_z_offset_m=float(
                cfg_value(
                    "transport.initial_z_offset_m",
                    0.2,
                    "AIC_PID_TRANSPORT_INITIAL_Z_OFFSET_M",
                )
            ),
            transport_duration_sec=float(
                cfg_value("transport.duration_sec", 5.0, "AIC_PID_TRANSPORT_DURATION_SEC")
            ),
            transport_dt_sec=float(
                cfg_value("transport.dt_sec", 0.05, "AIC_PID_TRANSPORT_DT_SEC")
            ),
            transport_force_injection_enabled=cfg_bool(
                "transport.force_injection.enabled",
                False,
                "AIC_PID_TRANSPORT_FORCE_INJECTION_ENABLED",
            ),
            transport_force_injection_randomize_time=cfg_bool(
                "transport.force_injection.randomize_time",
                True,
                "AIC_PID_TRANSPORT_FORCE_INJECTION_RANDOMIZE_TIME",
            ),
            transport_force_injection_config_time_sec=cfg_optional_float(
                "transport.force_injection.time_sec",
                "AIC_PID_TRANSPORT_FORCE_INJECTION_TIME_SEC",
            ),
            transport_force_injection_randomize_force=cfg_bool(
                "transport.force_injection.randomize_force",
                True,
                "AIC_PID_TRANSPORT_FORCE_INJECTION_RANDOMIZE_FORCE",
            ),
            transport_force_injection_min_n=float(
                cfg_value(
                    "transport.force_injection.min_n",
                    5.0,
                    "AIC_PID_TRANSPORT_FORCE_INJECTION_MIN_N",
                )
            ),
            transport_force_injection_max_n=float(
                cfg_value(
                    "transport.force_injection.max_n",
                    20.0,
                    "AIC_PID_TRANSPORT_FORCE_INJECTION_MAX_N",
                )
            ),
            transport_force_injection_force_n=cfg_optional_float(
                "transport.force_injection.force_n",
                "AIC_PID_TRANSPORT_FORCE_INJECTION_FORCE_N",
            ),
            transport_force_injection_randomize_axis=cfg_bool(
                "transport.force_injection.randomize_axis",
                True,
                "AIC_PID_TRANSPORT_FORCE_INJECTION_RANDOMIZE_AXIS",
            ),
            transport_force_injection_axis=cfg_optional_str(
                "transport.force_injection.axis",
                "AIC_PID_TRANSPORT_FORCE_INJECTION_AXIS",
            ),
            transport_force_injection_randomize_direction=cfg_bool(
                "transport.force_injection.randomize_direction",
                True,
                "AIC_PID_TRANSPORT_FORCE_INJECTION_RANDOMIZE_DIRECTION",
            ),
            transport_force_injection_direction_sign=cfg_optional_float(
                "transport.force_injection.direction_sign",
                "AIC_PID_TRANSPORT_FORCE_INJECTION_DIRECTION_SIGN",
            ),
            transport_force_injection_seed=int(
                cfg_value(
                    "transport.force_injection.seed",
                    7,
                    "AIC_PID_TRANSPORT_FORCE_INJECTION_SEED",
                )
            ),
            recover_lift_m=float(cfg_value("recover.lift_m", 0.01, "AIC_PID_RECOVER_LIFT_M")),
            recover_duration_sec=float(
                cfg_value("recover.duration_sec", 1.0, "AIC_PID_RECOVER_DURATION_SEC")
            ),
            recover_motion_dt_sec=float(
                cfg_value("recover.dt_sec", 0.05, "AIC_PID_RECOVER_DT_SEC")
            ),
            recover_pid_x_gains=(
                float(
                    cfg_value(
                        "recover.pid.x.kp",
                        pid_x_gains[0],
                        "AIC_PID_RECOVER_KP_X",
                    )
                ),
                float(
                    cfg_value(
                        "recover.pid.x.ki",
                        pid_x_gains[1],
                        "AIC_PID_RECOVER_KI_X",
                    )
                ),
                float(
                    cfg_value(
                        "recover.pid.x.kd",
                        pid_x_gains[2],
                        "AIC_PID_RECOVER_KD_X",
                    )
                ),
            ),
            recover_pid_y_gains=(
                float(
                    cfg_value(
                        "recover.pid.y.kp",
                        pid_y_gains[0],
                        "AIC_PID_RECOVER_KP_Y",
                    )
                ),
                float(
                    cfg_value(
                        "recover.pid.y.ki",
                        pid_y_gains[1],
                        "AIC_PID_RECOVER_KI_Y",
                    )
                ),
                float(
                    cfg_value(
                        "recover.pid.y.kd",
                        pid_y_gains[2],
                        "AIC_PID_RECOVER_KD_Y",
                    )
                ),
            ),
            pre_insertion_z_offset_m=float(
                cfg_value(
                    "pre_insertion.z_offset_m",
                    pre_insertion_default,
                    "AIC_PID_PRE_INSERTION_Z_OFFSET_M",
                )
            ),
            insert_min_z_offset_m=float(
                cfg_value("insert.min_z_offset_m", -0.015, "AIC_PID_INSERT_MIN_Z_OFFSET_M")
            ),
            insert_duration_sec=float(
                cfg_value("insert.duration_sec", 4.0, "AIC_PID_INSERT_DURATION_SEC")
            ),
            insert_dt_sec=float(cfg_value("insert.dt_sec", 0.05, "AIC_PID_INSERT_DT_SEC")),
            insert_wait_timeout_sec=float(
                cfg_value("insert.wait_timeout_sec", 5.0, "AIC_PID_INSERT_WAIT_TIMEOUT_SEC")
            ),
            force_log_interval_sec=float(
                cfg_value(
                    "diagnostics.force_log_interval_sec",
                    0.25,
                    "AIC_PID_FORCE_LOG_INTERVAL_SEC",
                )
            ),
            collision_force_threshold_n=float(
                cfg_value(
                    "diagnostics.collision_force_threshold_n",
                    20.0,
                    "AIC_PID_COLLISION_FORCE_THRESHOLD_N",
                )
            ),
            collision_duration_threshold_sec=float(
                cfg_value(
                    "diagnostics.collision_duration_threshold_sec",
                    1.0,
                    "AIC_PID_COLLISION_DURATION_THRESHOLD_SEC",
                )
            ),
            contact_force_threshold_n=float(
                cfg_value(
                    "diagnostics.contact_force_threshold_n",
                    5.0,
                    "AIC_PID_CONTACT_FORCE_THRESHOLD_N",
                )
            ),
            contact_duration_threshold_sec=float(
                cfg_value(
                    "diagnostics.contact_duration_threshold_sec",
                    1.0,
                    "AIC_PID_CONTACT_DURATION_THRESHOLD_SEC",
                )
            ),
            lateral_contact_force_threshold_n=float(
                cfg_value(
                    "diagnostics.lateral_contact_force_threshold_n",
                    10.0,
                    "AIC_PID_LATERAL_CONTACT_FORCE_THRESHOLD_N",
                )
            ),
            lateral_contact_ratio_threshold=float(
                cfg_value(
                    "diagnostics.lateral_contact_ratio_threshold",
                    0.65,
                    "AIC_PID_LATERAL_CONTACT_RATIO_THRESHOLD",
                )
            ),
            lateral_contact_duration_threshold_sec=float(
                cfg_value(
                    "diagnostics.lateral_contact_duration_threshold_sec",
                    0.3,
                    "AIC_PID_LATERAL_CONTACT_DURATION_THRESHOLD_SEC",
                )
            ),
            stuck_fz_threshold_n=float(
                cfg_value(
                    "diagnostics.stuck_fz_threshold_n",
                    5.0,
                    "AIC_PID_STUCK_FZ_THRESHOLD_N",
                )
            ),
            stuck_duration_threshold_sec=float(
                cfg_value(
                    "diagnostics.stuck_duration_threshold_sec",
                    1.0,
                    "AIC_PID_STUCK_DURATION_THRESHOLD_SEC",
                )
            ),
            stuck_z_stationary_threshold_m=float(
                cfg_value(
                    "diagnostics.stuck_z_stationary_threshold_m",
                    0.0001,
                    "AIC_PID_STUCK_Z_STATIONARY_THRESHOLD_M",
                )
            ),
            stuck_force_increase_threshold_n=float(
                cfg_value(
                    "diagnostics.stuck_force_increase_threshold_n",
                    0.05,
                    "AIC_PID_STUCK_FORCE_INCREASE_THRESHOLD_N",
                )
            ),
            stuck_debug_log_interval_sec=float(
                cfg_value(
                    "diagnostics.stuck_debug_log_interval_sec",
                    0.5,
                    "AIC_PID_STUCK_DEBUG_LOG_INTERVAL_SEC",
                )
            ),
        )

        if (
            cfg.transport_force_injection_enabled
            and cfg.transport_force_injection_min_n >= cfg.transport_force_injection_max_n
        ):
            raise RuntimeError(
                "AIC_PID_TRANSPORT_FORCE_INJECTION_MIN_N must be less than "
                "AIC_PID_TRANSPORT_FORCE_INJECTION_MAX_N"
            )

        if cfg.pid_plots_enabled and not _HAS_MATPLOTLIB:
            raise RuntimeError(
                "PID tuning plots are enabled but matplotlib is not installed. "
                "Hint: pixi add matplotlib"
            )

        return cfg

    @staticmethod
    def _resolve_config_path() -> Path:
        configured_path = os.environ.get("AIC_PID_CONFIG_FILE")
        if configured_path:
            candidate = Path(configured_path).expanduser()
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"CheatCodePID config file not found: {candidate}")

        candidates = [
            Path(__file__).resolve().parent.parent
            / "assets"
            / "cheatcode_pid"
            / "config.yaml",
            Path(__file__).resolve().parent.parent
            / "assets"
            / "cheatcode_pid"
            / "config.json",
            Path.cwd()
            / "aic_example_policies"
            / "aic_example_policies"
            / "assets"
            / "cheatcode_pid"
            / "config.yaml",
            Path.cwd()
            / "aic_example_policies"
            / "aic_example_policies"
            / "assets"
            / "cheatcode_pid"
            / "config.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not locate CheatCodePID config file. Set AIC_PID_CONFIG_FILE or "
            "place config.yaml/json under aic_example_policies/aic_example_policies/assets/cheatcode_pid."
        )

    @staticmethod
    def _load_config_file(path: Path) -> dict[str, Any]:
        suffix = path.suffix.lower()
        raw_text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            data = json.loads(raw_text)
        else:
            if yaml is None:
                raise RuntimeError(f"Cannot load YAML config without PyYAML installed: {path}")
            data = yaml.safe_load(raw_text)
        if not isinstance(data, dict):
            raise RuntimeError(f"CheatCodePID config must be a mapping: {path}")
        return data

    @staticmethod
    def _lookup_config(config: dict[str, Any], key_path: str) -> Any | None:
        current: Any = config
        for key in key_path.split("."):
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current


class ForceMonitor:
    def __init__(self, config: ControllerConfig, logger_getter: Callable[[], Any]):
        self._cfg = config
        self._logger_getter = logger_getter
        self._active = False
        self.reset()

    @property
    def collision_detected(self) -> bool:
        return self._collision_detected

    @property
    def contact_detected(self) -> bool:
        return self._contact_detected

    @property
    def lateral_contact_detected(self) -> bool:
        return self._lateral_contact_detected

    @property
    def stuck_detected(self) -> bool:
        return self._stuck_detected

    @property
    def blocked_descent_detected(self) -> bool:
        return self._blocked_descent_detected

    @property
    def latest_tared_force_z_n(self) -> float:
        return self._latest_tared_force_z_n

    def reset(self) -> None:
        self._collision_detected = False
        self._contact_detected = False
        self._lateral_contact_detected = False
        self._stuck_detected = False
        self._blocked_descent_detected = False

        self._latest_force_mag_n = 0.0
        self._latest_tared_force_z_n = 0.0
        self._latest_lateral_force_mag_n = 0.0
        self._latest_lateral_force_ratio = 0.0
        self._max_force_mag_n = 0.0

        self._force_above_threshold_sec = 0.0
        self._contact_force_sec = 0.0
        self._lateral_contact_sec = 0.0
        self._stuck_sec = 0.0

        self._last_force_sample_time_sec = None
        self._last_force_log_time_sec = None
        self._last_stuck_log_time_sec = None

        self._last_blocked_descent_check_time_sec = None
        self._last_plug_port_z_m = None
        self._stuck_window_start_abs_fz_n = None

    def set_active(self, active: bool) -> None:
        self._active = active

    def reset_blocked_descent_state(self) -> None:
        # Recovery should clear both blocked-descent state and contact latches so
        # subsequent INSERT attempts can re-detect fresh events.
        self._collision_detected = False
        self._contact_detected = False
        self._lateral_contact_detected = False

        self._force_above_threshold_sec = 0.0
        self._contact_force_sec = 0.0
        self._lateral_contact_sec = 0.0

        self._stuck_detected = False
        self._blocked_descent_detected = False
        self._stuck_sec = 0.0
        self._last_blocked_descent_check_time_sec = None
        self._last_plug_port_z_m = None
        self._stuck_window_start_abs_fz_n = None
        self._last_stuck_log_time_sec = None

    def update_observation(self, msg: Observation, fallback_time_sec: float) -> None:
        raw_force = msg.wrist_wrench.wrench.force
        tare = msg.controller_state.fts_tare_offset.wrench.force
        tared_force = np.array(
            [
                raw_force.x - tare.x,
                raw_force.y - tare.y,
                raw_force.z - tare.z,
            ]
        )

        self._latest_force_mag_n = float(np.linalg.norm(tared_force))
        self._latest_tared_force_z_n = float(tared_force[2])
        self._latest_lateral_force_mag_n = float(np.linalg.norm(tared_force[:2]))
        self._latest_lateral_force_ratio = self._latest_lateral_force_mag_n / max(
            self._latest_force_mag_n, 1e-6
        )
        self._max_force_mag_n = max(self._max_force_mag_n, self._latest_force_mag_n)

        if not self._active:
            return

        stamp = msg.wrist_wrench.header.stamp
        sample_time_sec = stamp.sec + stamp.nanosec / 1e9
        if sample_time_sec <= 0.0:
            sample_time_sec = fallback_time_sec

        dt = 0.0
        if self._last_force_sample_time_sec is not None:
            dt = max(0.0, sample_time_sec - self._last_force_sample_time_sec)
        self._last_force_sample_time_sec = sample_time_sec

        if self._latest_force_mag_n > self._cfg.collision_force_threshold_n:
            self._force_above_threshold_sec += dt
        else:
            self._force_above_threshold_sec = 0.0

        if self._latest_force_mag_n >= self._cfg.contact_force_threshold_n:
            self._contact_force_sec += dt
        else:
            self._contact_force_sec = 0.0

        lateral_contact_active = (
            self._latest_lateral_force_mag_n >= self._cfg.lateral_contact_force_threshold_n
            or (
                self._latest_force_mag_n >= self._cfg.contact_force_threshold_n
                and self._latest_lateral_force_ratio
                >= self._cfg.lateral_contact_ratio_threshold
            )
        )
        if lateral_contact_active:
            self._lateral_contact_sec += dt
        else:
            self._lateral_contact_sec = 0.0

        if self._cfg.force_log_interval_sec > 0.0 and (
            self._last_force_log_time_sec is None
            or sample_time_sec - self._last_force_log_time_sec >= self._cfg.force_log_interval_sec
        ):
            self._logger_getter().info(
                "Tared force: "
                f"|F|={self._latest_force_mag_n:.2f} N, "
                f"Fx={tared_force[0]:.2f} N, "
                f"Fy={tared_force[1]:.2f} N, "
                f"Fz={tared_force[2]:.2f} N, "
                f"Fxy={self._latest_lateral_force_mag_n:.2f} N, "
                f"lateral_ratio={self._latest_lateral_force_ratio:.2f}, "
                f"contact_time={self._contact_force_sec:.2f} s, "
                f"lateral_contact_time={self._lateral_contact_sec:.2f} s, "
                f"stuck_time={self._stuck_sec:.2f} s, "
                f"time_above_20N={self._force_above_threshold_sec:.2f} s"
            )
            self._last_force_log_time_sec = sample_time_sec

        if (
            not self._contact_detected
            and self._contact_force_sec > self._cfg.contact_duration_threshold_sec
        ):
            self._logger_getter().warn(
                "Contact detected: sustained tared force between "
                f"{self._cfg.contact_force_threshold_n:.1f} N and "
                f"{self._cfg.collision_force_threshold_n:.1f} N for "
                f"{self._contact_force_sec:.2f} seconds. "
                "This is contact or blocked descent, not a scoring insertion collision."
            )
            self._contact_detected = True

        if (
            not self._lateral_contact_detected
            and self._lateral_contact_sec > self._cfg.lateral_contact_duration_threshold_sec
        ):
            self._logger_getter().warn(
                "Lateral contact detected: sustained off-axis tared force. "
                f"Fxy={self._latest_lateral_force_mag_n:.2f} N, "
                f"|F|={self._latest_force_mag_n:.2f} N, "
                f"lateral_ratio={self._latest_lateral_force_ratio:.2f}, "
                f"duration={self._lateral_contact_sec:.2f} s. "
                "This indicates binding/contact even before scoring collision."
            )
            self._lateral_contact_detected = True

        if (
            not self._collision_detected
            and self._force_above_threshold_sec > self._cfg.collision_duration_threshold_sec
        ):
            self._logger_getter().warn(
                "Collision detected: insertion force above "
                f"{self._cfg.collision_force_threshold_n:.1f} N for "
                f"{self._force_above_threshold_sec:.2f} seconds. "
                f"Max tared force: {self._max_force_mag_n:.2f} N"
            )
            self._collision_detected = True

    def update_blocked_descent(self, plug_port_z_m: float, now_sec: float) -> None:
        if (
            self._last_plug_port_z_m is None
            or self._last_blocked_descent_check_time_sec is None
        ):
            self._last_plug_port_z_m = plug_port_z_m
            self._last_blocked_descent_check_time_sec = now_sec
            return

        dt = max(0.0, now_sec - self._last_blocked_descent_check_time_sec)
        z_progress_m = self._last_plug_port_z_m - plug_port_z_m
        abs_fz_n = abs(self._latest_tared_force_z_n)
        self._last_plug_port_z_m = plug_port_z_m
        self._last_blocked_descent_check_time_sec = now_sec

        fz_contact_present = abs_fz_n > self._cfg.stuck_fz_threshold_n
        z_stationary = abs(z_progress_m) <= self._cfg.stuck_z_stationary_threshold_m
        if fz_contact_present and z_stationary:
            if self._stuck_window_start_abs_fz_n is None:
                self._stuck_window_start_abs_fz_n = abs_fz_n
            self._stuck_sec += dt
        else:
            self._stuck_sec = 0.0
            self._stuck_window_start_abs_fz_n = None

        window_force_delta_n = (
            0.0
            if self._stuck_window_start_abs_fz_n is None
            else abs_fz_n - self._stuck_window_start_abs_fz_n
        )
        force_increased = window_force_delta_n >= self._cfg.stuck_force_increase_threshold_n
        # Handle force-baseline shift / plateaued contact: even when |Fz| no longer
        # rises, prolonged stationary high-force contact still indicates blockage.
        plateau_force_floor_n = (
            self._cfg.stuck_fz_threshold_n
            + 0.5 * self._cfg.stuck_force_increase_threshold_n
        )
        plateau_time_threshold_sec = 1.5 * self._cfg.stuck_duration_threshold_sec
        plateau_blocked = (
            self._stuck_sec > plateau_time_threshold_sec
            and abs_fz_n >= plateau_force_floor_n
        )

        if (
            self._cfg.stuck_debug_log_interval_sec > 0.0
            and fz_contact_present
            and (
                self._last_stuck_log_time_sec is None
                or now_sec - self._last_stuck_log_time_sec >= self._cfg.stuck_debug_log_interval_sec
            )
        ):
            self._logger_getter().info(
                "[CheatCodePID] STUCK check: "
                f"|Fz|={abs_fz_n:.2f} N "
                f"(threshold>{self._cfg.stuck_fz_threshold_n:.2f}), "
                f"z_change={z_progress_m * 1e3:.3f} mm "
                f"(stationary<={self._cfg.stuck_z_stationary_threshold_m * 1e3:.3f} mm), "
                f"|Fz|_window_delta={window_force_delta_n:.2f} N "
                f"(threshold>={self._cfg.stuck_force_increase_threshold_n:.2f}), "
                f"plateau_time_threshold={plateau_time_threshold_sec:.2f} s, "
                f"stuck_time={self._stuck_sec:.2f}/"
                f"{self._cfg.stuck_duration_threshold_sec:.2f} s"
            )
            self._last_stuck_log_time_sec = now_sec

        if (
            not self._stuck_detected
            and self._stuck_sec > self._cfg.stuck_duration_threshold_sec
            and (force_increased or plateau_blocked)
        ):
            trigger_reason = (
                "force-rise-trigger"
                if force_increased
                else "plateau-trigger"
            )
            self._logger_getter().warn(
                "[CheatCodePID] STUCK condition detected during INSERT: "
                "z_offset is decreasing, plug-to-port Z is stationary, "
                "and blocked contact persists. "
                f"trigger={trigger_reason}, "
                f"plug_port_z={plug_port_z_m:.4f} m, "
                f"last_z_change={z_progress_m * 1e3:.2f} mm, "
                f"Fz={self._latest_tared_force_z_n:.2f} N, "
                f"|Fz|_window_delta={window_force_delta_n:.2f} N, "
                f"stuck_time={self._stuck_sec:.2f} s"
            )
            self._stuck_detected = True
            self._blocked_descent_detected = True


class TelemetryManager:
    def __init__(
        self,
        config: ControllerConfig,
        logger_getter: Callable[[], Any],
        now_ns_getter: Callable[[], int],
    ):
        self._cfg = config
        self._logger_getter = logger_getter
        self._now_ns_getter = now_ns_getter
        self.reset()

    def reset(self) -> None:
        self.history_time: list[float] = []
        self.history_err_x: list[float] = []
        self.history_err_y: list[float] = []
        self.history_state: list[str] = []
        self.history_collision_detected: list[bool] = []
        self.history_stuck_detected: list[bool] = []
        self.start_time_sec: float | None = None
        self.pre_descent_index: int | None = None

    def record(
        self,
        pid_error_x: float,
        pid_error_y: float,
        state_name: str,
        collision_detected: bool,
        stuck_detected: bool,
        now_sec: float,
    ) -> None:
        if self.start_time_sec is None:
            self.start_time_sec = now_sec
        current_time = now_sec - self.start_time_sec
        self.history_time.append(current_time)
        self.history_err_x.append(pid_error_x)
        self.history_err_y.append(pid_error_y)
        self.history_state.append(state_name)
        self.history_collision_detected.append(collision_detected)
        self.history_stuck_detected.append(stuck_detected)

    def mark_pre_descent(self) -> None:
        self.pre_descent_index = len(self.history_err_x) - 1

    def safe_plot(
        self,
        success: bool,
        task: Task | None,
        pid_mode: PIDControllerMode,
        pid_x: PIDController,
        pid_y: PIDController,
        recover_pid_x_gains: tuple[float, float, float],
        recover_pid_y_gains: tuple[float, float, float],
    ) -> None:
        if not self._cfg.pid_plots_enabled:
            return
        try:
            self._plot(
                success=success,
                task=task,
                pid_mode=pid_mode,
                pid_x=pid_x,
                pid_y=pid_y,
                recover_pid_x_gains=recover_pid_x_gains,
                recover_pid_y_gains=recover_pid_y_gains,
            )
        except Exception as ex:
            self._logger_getter().error(f"Failed to save PID telemetry plot: {ex}")

    def _build_plot_path(self, task: Task | None) -> Path:
        task_name = "task"
        if task is not None:
            task_name = f"{task.target_module_name}_{task.port_name}"
        safe_task_name = re.sub(r"[^A-Za-z0-9._-]+", "_", task_name).strip("_")
        if not safe_task_name:
            safe_task_name = "task"
        timestamp_ns = self._now_ns_getter()
        return self._cfg.plot_output_dir / f"{safe_task_name}_{timestamp_ns}.png"

    def _plot(
        self,
        success: bool,
        task: Task | None,
        pid_mode: PIDControllerMode,
        pid_x: PIDController,
        pid_y: PIDController,
        recover_pid_x_gains: tuple[float, float, float],
        recover_pid_y_gains: tuple[float, float, float],
    ) -> None:
        if not self.history_time:
            self._logger_getter().warn("No history recorded to plot.")
            return

        plt.figure(figsize=(10, 6))
        err_x_mm = [e * 1e3 for e in self.history_err_x]
        err_y_mm = [e * 1e3 for e in self.history_err_y]

        plt.plot(self.history_time, err_x_mm, label="Error X (mm)", color="red", linewidth=1.5)
        plt.plot(self.history_time, err_y_mm, label="Error Y (mm)", color="blue", linewidth=1.5)

        final_x = err_x_mm[-1]
        final_y = err_y_mm[-1]

        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Error (mm)")

        task_label = task.target_module_name if task else "Task"
        status = "SUCCESS" if success else "FAILURE"
        plt.title(
            f"PID Performance: {task_label} [{status}]\n"
            f"Mode={pid_mode.name}  |  "
            f"X: Kp={pid_x.kp} Ki={pid_x.ki} Kd={pid_x.kd}  |  "
            f"Y: Kp={pid_y.kp} Ki={pid_y.ki} Kd={pid_y.kd}  |  "
            f"Recover X: Kp={recover_pid_x_gains[0]} "
            f"Ki={recover_pid_x_gains[1]} "
            f"Kd={recover_pid_x_gains[2]}  |  "
            f"Recover Y: Kp={recover_pid_y_gains[0]} "
            f"Ki={recover_pid_y_gains[1]} "
            f"Kd={recover_pid_y_gains[2]}\n"
            f"Final Error: X={final_x:.3f} mm, Y={final_y:.3f} mm"
        )

        if self.pre_descent_index is not None and self.pre_descent_index < len(self.history_time):
            desc_idx = self.pre_descent_index
            plt.axvline(self.history_time[desc_idx], color="green", linestyle=":", label="Descent Start")
            desc_x = err_x_mm[desc_idx]
            desc_y = err_y_mm[desc_idx]
            plt.annotate(
                f"DESCENT START\nX: {desc_x:.3f} mm\nY: {desc_y:.3f} mm",
                xy=(self.history_time[desc_idx], max(desc_x, desc_y)),
                xytext=(-10, 20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", alpha=0.3),
                ha="right",
            )

        plt.legend(loc="upper right")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.annotate(
            f"FINAL ERROR\nX: {final_x:.3f} mm\nY: {final_y:.3f} mm",
            xy=(self.history_time[-1], (final_x + final_y) / 2),
            xytext=(10, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3),
        )

        plt.tight_layout()
        plot_path = self._build_plot_path(task)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        self._logger_getter().info(f"Telemetry plot saved as {plot_path}")
