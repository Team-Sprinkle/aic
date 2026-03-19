"""Diversified TF-assisted cable insertion policy.

This policy keeps the same "cheat" core behavior as `CheatCode`:
- use ground-truth TF for port, plug, and gripper poses
- continuously solve a target gripper pose that aligns plug to port

It then diversifies trajectories for data collection by randomizing:
- temporal profile (linear / smoothstep / minimum-jerk)
- approach timing (loop dt and total segment duration)
- approach geometry (optional intermediate waypoints)
- low-frequency micro-jitter (small sinusoidal offsets)
- insertion style (constant, staged, or peck descent)
- mild controller bias terms (integrator windup + i_gain)
"""

import os
from typing import TypedDict

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


class TrajectoryParams(TypedDict):
    """Per-episode randomized trajectory parameters.

    Each field is sampled once in `insert_cable()` and reused for the whole
    episode to keep motions smooth while still producing inter-episode
    variability for dataset collection.
    """

    profile: str
    approach_dt_sec: float
    approach_duration_sec: float
    start_z_offset: float
    waypoints: list[tuple[float, float, float]]
    micro_jitter_amp_x: float
    micro_jitter_amp_y: float
    micro_jitter_amp_z: float
    micro_jitter_freq_x: float
    micro_jitter_freq_y: float
    micro_jitter_freq_z: float
    micro_jitter_phase_x: float
    micro_jitter_phase_y: float
    micro_jitter_phase_z: float
    i_gain: float
    max_integrator_windup: float
    descent_mode: str
    descent_base_step: float
    descent_dt_sec: float
    descent_end_offset: float


class DiversifiedCheatCode(Policy):
    """Ground-truth insertion policy with bounded randomized trajectory styles.

    The policy is intended for trajectory data generation, not real deployment.
    Randomization ranges are intentionally conservative to preserve successful
    insertion while increasing path diversity.
    """

    def __init__(self, parent_node):
        """Initialize policy state and seeded RNG (optional env override)."""
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        self._rng = self._build_rng()
        super().__init__(parent_node)

    def _build_rng(self):
        """Build numpy RNG.

        Returns:
            np.random.Generator: Deterministic if `AIC_DIVERSIFIED_SEED` is set
            to an integer; otherwise random-seeded.
        """
        seed_str = os.getenv("AIC_DIVERSIFIED_SEED")
        if seed_str is not None:
            try:
                seed = int(seed_str)
                self.get_logger().info(f"DiversifiedCheatCode deterministic seed: {seed}")
                return np.random.default_rng(seed)
            except ValueError:
                self.get_logger().warn(
                    f"Invalid AIC_DIVERSIFIED_SEED '{seed_str}', using random seed"
                )
        return np.random.default_rng()

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        """Wait for a TF transform to become available.

        Args:
            target_frame: Destination frame for lookup.
            source_frame: Source frame for lookup.
            timeout_sec: Max wait time in seconds.

        Returns:
            bool: True when transform exists before timeout.
        """
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'... -- are you running eval with `ground_truth:=true`?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def _profile_fraction(self, raw_fraction: float, profile: str) -> float:
        """Map linear progress to a shaped interpolation progress.

        Args:
            raw_fraction: Unshaped progress in [0, 1].
            profile: Time profile name ("linear", "smoothstep", "min_jerk").

        Returns:
            float: Shaped progress in [0, 1].
        """
        f = float(np.clip(raw_fraction, 0.0, 1.0))
        if profile == "smoothstep":
            return f * f * (3.0 - 2.0 * f)
        if profile == "min_jerk":
            return 10.0 * f**3 - 15.0 * f**4 + 6.0 * f**5
        return f

    def _sample_trajectory_params(self) -> TrajectoryParams:
        """Sample one randomized trajectory configuration for an episode.

        Diversification groups:
        - Approach geometry: `waypoints`, `start_z_offset`
        - Approach timing: `approach_dt_sec`, `approach_duration_sec`, `profile`
        - Micro-perturbations: jitter amplitudes/frequencies/phases
        - Insertion style: `descent_mode`, `descent_base_step`, `descent_dt_sec`
        - Correction dynamics: `i_gain`, `max_integrator_windup`

        Returns:
            TrajectoryParams: Randomized but bounded trajectory parameters.
        """
        profile = self._rng.choice(["linear", "smoothstep", "min_jerk"])
        descent_mode = self._rng.choice(["constant", "staged", "peck"])
        num_waypoints = int(self._rng.integers(0, 3))
        waypoints = []
        for _ in range(num_waypoints):
            waypoints.append(
                (
                    float(self._rng.uniform(-0.02, 0.02)),
                    float(self._rng.uniform(-0.02, 0.02)),
                    float(self._rng.uniform(-0.01, 0.03)),
                )
            )

        return {
            "profile": str(profile),
            "approach_dt_sec": float(self._rng.uniform(0.04, 0.07)),
            "approach_duration_sec": float(self._rng.uniform(3.0, 7.0)),
            "start_z_offset": float(self._rng.uniform(0.15, 0.30)),
            "waypoints": waypoints,
            "micro_jitter_amp_x": float(self._rng.uniform(0.0, 0.003)),
            "micro_jitter_amp_y": float(self._rng.uniform(0.0, 0.003)),
            "micro_jitter_amp_z": float(self._rng.uniform(0.0, 0.002)),
            "micro_jitter_freq_x": float(self._rng.uniform(0.2, 0.8)),
            "micro_jitter_freq_y": float(self._rng.uniform(0.2, 0.8)),
            "micro_jitter_freq_z": float(self._rng.uniform(0.2, 0.8)),
            "micro_jitter_phase_x": float(self._rng.uniform(0.0, 2.0 * np.pi)),
            "micro_jitter_phase_y": float(self._rng.uniform(0.0, 2.0 * np.pi)),
            "micro_jitter_phase_z": float(self._rng.uniform(0.0, 2.0 * np.pi)),
            "i_gain": float(self._rng.uniform(0.12, 0.20)),
            "max_integrator_windup": float(self._rng.uniform(0.03, 0.07)),
            "descent_mode": str(descent_mode),
            "descent_base_step": float(self._rng.uniform(0.00035, 0.00095)),
            "descent_dt_sec": float(self._rng.uniform(0.04, 0.07)),
            "descent_end_offset": float(self._rng.uniform(-0.020, -0.010)),
        }

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
        xy_offset: tuple[float, float] = (0.0, 0.0),
        z_extra: float = 0.0,
        i_gain: float = 0.15,
    ) -> Pose:
        """Compute a blended target gripper pose for alignment.

        This function is the core pose solver. It aligns orientation with
        quaternion slerp, and blends position from current gripper pose toward
        a target above/near the port.

        Args:
            port_transform: Transform of target port in `base_link`.
            slerp_fraction: Orientation interpolation factor [0, 1].
            position_fraction: Position interpolation factor [0, 1].
            z_offset: Base height above/below the port for insertion phase.
            reset_xy_integrator: If True, clears XY error integral.
            xy_offset: Additional lateral offset for waypoint shaping/jitter.
            z_extra: Additional vertical offset for waypoint shaping/jitter.
            i_gain: Integral gain used to bias XY target toward error history.

        Returns:
            Pose: Gripper target pose in `base_link`.
        """
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        q_plug = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_plug_inv = (
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        )
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        q_gripper = (
            gripper_tf_stamped.transform.rotation.w,
            gripper_tf_stamped.transform.rotation.x,
            gripper_tf_stamped.transform.rotation.y,
            gripper_tf_stamped.transform.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (
            gripper_tf_stamped.transform.translation.x,
            gripper_tf_stamped.transform.translation.y,
            gripper_tf_stamped.transform.translation.z,
        )
        port_xy = (
            port_transform.translation.x,
            port_transform.translation.y,
        )
        plug_xyz = (
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        )
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )
            self._tip_y_error_integrator = np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )

        target_x = (
            port_xy[0]
            + xy_offset[0]
            + i_gain * self._tip_x_error_integrator
        )
        target_y = (
            port_xy[1]
            + xy_offset[1]
            + i_gain * self._tip_y_error_integrator
        )
        target_z = (
            port_transform.translation.z
            + z_offset
            + z_extra
            - plug_tip_gripper_offset[2]
        )

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(
                x=blend_xyz[0],
                y=blend_xyz[1],
                z=blend_xyz[2],
            ),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )

    def _micro_jitter(
        self, params: TrajectoryParams, t_sec: float, decay_fraction: float
    ) -> tuple[float, float, float]:
        """Compute smooth micro-jitter for data diversity.

        Args:
            params: Episode parameter set with jitter amplitudes/frequencies.
            t_sec: Elapsed trajectory time.
            decay_fraction: Progress-based decay term in [0, 1].

        Returns:
            tuple[float, float, float]: Small (x, y, z) offsets in meters.
        """
        # Taper perturbations near the final alignment target.
        decay = float(np.clip(1.0 - decay_fraction, 0.0, 1.0))
        jx = params["micro_jitter_amp_x"] * np.sin(
            2.0 * np.pi * params["micro_jitter_freq_x"] * t_sec
            + params["micro_jitter_phase_x"]
        )
        jy = params["micro_jitter_amp_y"] * np.sin(
            2.0 * np.pi * params["micro_jitter_freq_y"] * t_sec
            + params["micro_jitter_phase_y"]
        )
        jz = params["micro_jitter_amp_z"] * np.sin(
            2.0 * np.pi * params["micro_jitter_freq_z"] * t_sec
            + params["micro_jitter_phase_z"]
        )
        return (decay * jx, decay * jy, decay * jz)

    def _run_interpolation_segment(
        self,
        move_robot: MoveRobotCallback,
        port_transform: Transform,
        params: TrajectoryParams,
        z_offset: float,
        xy_offset: tuple[float, float],
        z_extra: float,
        elapsed_t_sec: float,
    ) -> float:
        """Run one randomized approach segment toward a local target.

        Args:
            move_robot: Controller callback used to send pose targets.
            port_transform: Port transform in `base_link`.
            params: Episode randomization parameters.
            z_offset: Base offset above port for this segment.
            xy_offset: Segment-level lateral offset (usually waypoint).
            z_extra: Segment-level vertical offset (usually waypoint).
            elapsed_t_sec: Running elapsed time for phase-continuous jitter.

        Returns:
            float: Updated elapsed time after this segment.
        """
        num_steps = max(
            2, int(params["approach_duration_sec"] / params["approach_dt_sec"])
        )
        for step in range(num_steps):
            raw_fraction = step / float(num_steps)
            interp_fraction = self._profile_fraction(raw_fraction, params["profile"])
            # Waypoint offset shapes the macro path; jitter adds smooth local
            # variation for richer demonstrations.
            jitter = self._micro_jitter(params, elapsed_t_sec, interp_fraction)
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform=port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                        xy_offset=(xy_offset[0] + jitter[0], xy_offset[1] + jitter[1]),
                        z_extra=z_extra + jitter[2],
                        i_gain=params["i_gain"],
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(
                    f"TF lookup failed during diversified interpolation: {ex}"
                )
            self.sleep_for(params["approach_dt_sec"])
            elapsed_t_sec += params["approach_dt_sec"]
        return elapsed_t_sec

    def _run_diversified_descent(
        self,
        move_robot: MoveRobotCallback,
        port_transform: Transform,
        params: TrajectoryParams,
        z_offset: float,
    ):
        """Run insertion descent with a sampled descent strategy.

        Descent modes:
        - constant: fixed per-step decrement
        - staged: faster far away, slower near socket
        - peck: occasional small retreat before continuing downward

        Args:
            move_robot: Controller callback used to send pose targets.
            port_transform: Port transform in `base_link`.
            params: Episode randomization parameters.
            z_offset: Initial descent offset above the port.
        """
        z_end = params["descent_end_offset"]
        iteration = 0
        max_iters = 2000
        peck_counter = 0
        while z_offset > z_end and iteration < max_iters:
            step = params["descent_base_step"]
            if params["descent_mode"] == "staged":
                if z_offset > 0.04:
                    step *= 1.8
                else:
                    step *= 0.7
            elif params["descent_mode"] == "peck":
                peck_counter += 1
                if peck_counter % 35 == 0:
                    z_offset += 0.0025
                    self.sleep_for(params["descent_dt_sec"] * 2.0)

            z_offset -= step
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform=port_transform,
                        z_offset=z_offset,
                        i_gain=params["i_gain"],
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(
                    f"TF lookup failed during diversified insertion: {ex}"
                )
            self.sleep_for(params["descent_dt_sec"])
            iteration += 1

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        """Execute one full insertion rollout with diversified motion.

        Pipeline:
        1. Sample a per-episode trajectory configuration.
        2. Wait for required ground-truth TF frames.
        3. Run zero or more waypoint approach segments.
        4. Run a final centered approach segment.
        5. Descend with sampled insertion strategy.

        Args:
            task: Challenge task descriptor (port/module/cable names).
            get_observation: Unused callback required by policy interface.
            move_robot: Callback used to send motion updates.
            send_feedback: Callback for external status reporting.

        Returns:
            bool: True on completion, False on missing TF prerequisites.
        """
        self.get_logger().info(f"DiversifiedCheatCode.insert_cable() task: {task}")
        self._task = task
        params = self._sample_trajectory_params()
        self._max_integrator_windup = params["max_integrator_windup"]

        send_feedback(
            (
                "diversified trajectory active: "
                f"profile={params['profile']} descent={params['descent_mode']}"
            )
        )
        self.get_logger().info(
            "Trajectory params: "
            f"profile={params['profile']} "
            f"dt={params['approach_dt_sec']:.3f} "
            f"dur={params['approach_duration_sec']:.2f} "
            f"start_z={params['start_z_offset']:.3f} "
            f"waypoints={params['waypoints']} "
            f"descent_mode={params['descent_mode']} "
            f"descent_step={params['descent_base_step']:.6f}"
        )

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                port_frame,
                Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False
        port_transform = port_tf_stamped.transform

        elapsed_t_sec = 0.0
        z_offset = params["start_z_offset"]

        for waypoint in params["waypoints"]:
            elapsed_t_sec = self._run_interpolation_segment(
                move_robot=move_robot,
                port_transform=port_transform,
                params=params,
                z_offset=z_offset,
                xy_offset=(waypoint[0], waypoint[1]),
                z_extra=waypoint[2],
                elapsed_t_sec=elapsed_t_sec,
            )

        self._run_interpolation_segment(
            move_robot=move_robot,
            port_transform=port_transform,
            params=params,
            z_offset=z_offset,
            xy_offset=(0.0, 0.0),
            z_extra=0.0,
            elapsed_t_sec=elapsed_t_sec,
        )

        self._run_diversified_descent(
            move_robot=move_robot,
            port_transform=port_transform,
            params=params,
            z_offset=z_offset,
        )

        self.get_logger().info("Waiting for connector to stabilize...")
        self.sleep_for(5.0)

        self.get_logger().info("DiversifiedCheatCode.insert_cable() exiting...")
        return True
