# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Small, package-local RB-Y1 SDK adapter.

The public calling surface intentionally follows the proven
``projects.rby1_teleop.rby1_api_utils.RobotMwithBase`` usage from
``demo_rby1_xr_robot_teleop_v9_hw_all.py``.  The implementation is limited to
the body/head joint stream used by this package; the legacy monolith remains
the source for experimental Cartesian and wheel-odometry controllers.
"""

from __future__ import annotations

from typing import Any

import numpy as np


RIGHT_READY_RAD = np.array(
    [-0.261799, -0.261799, 0.0, -1.894395, 0.0, 0.0, 0.0], dtype=float
)
LEFT_READY_RAD = np.array(
    [-0.261799, 0.261799, 0.0, -1.894395, 0.0, 0.0, 0.0], dtype=float
)
TORSO_READY_RAD = np.array([0.0, 0.5, -1.0, 0.5, 0.0, 0.0], dtype=float)
HEAD_READY_RAD = np.array([0.0, 0.6], dtype=float)


def _require_sdk():
    try:
        import rby1_sdk
    except ImportError as exc:
        raise ImportError(
            "RB-Y1 hardware support is optional. Install with "
            "`pip install 'rby1-teleop[hw]'` before using --hw."
        ) from exc
    return rby1_sdk


def _goal_value(goals: Any, name: str, default=None):
    if isinstance(goals, dict):
        return goals.get(name, default)
    return getattr(goals, name, default)


class RBY1HardwareController:
    """Connected Model-M body/head joint controller.

    Construction connects, powers the requested devices, enables servos and
    enables the control manager. It does *not* move the robot. Call
    :meth:`reset_egoengine_ready_pose` explicitly, matching the v9 demo.
    """

    def __init__(
        self,
        address: str,
        servo: str = ".*",
        power_device: str = ".*",
        controller: str | None = None,
        use_egoengine: bool = False,
        *,
        position_control: bool = True,
        sdk=None,
        robot=None,
    ):
        del controller, use_egoengine  # accepted for rby1_api_utils compatibility
        self.sdk = sdk or _require_sdk()
        self.address = address
        self.position_control = bool(position_control)
        self.robot = robot or self.sdk.create_robot_m(address)
        self._connect_and_enable(power_device, servo)
        self.stream = self.robot.create_command_stream()
        self.q_lower, self.q_upper = self._read_joint_limits()

    def _connect_and_enable(self, power_device: str, servo: str) -> None:
        if not self.robot.connect():
            raise ConnectionError(f"Could not connect to RB-Y1 at {self.address}")
        if hasattr(self.robot, "is_connected") and not self.robot.is_connected():
            raise ConnectionError(f"RB-Y1 at {self.address} did not stay connected")
        if not self.robot.is_power_on(power_device) and not self.robot.power_on(power_device):
            raise RuntimeError(f"Failed to power on RB-Y1 devices matching {power_device!r}")
        if not self.robot.is_servo_on(servo) and not self.robot.servo_on(servo):
            raise RuntimeError(f"Failed to enable RB-Y1 servos matching {servo!r}")

        state = self.robot.get_control_manager_state()
        fault_states = {
            self.sdk.ControlManagerState.State.MinorFault,
            self.sdk.ControlManagerState.State.MajorFault,
        }
        if state.state in fault_states and not self.robot.reset_fault_control_manager():
            raise RuntimeError("Failed to reset the RB-Y1 control-manager fault")
        if not self.robot.enable_control_manager():
            raise RuntimeError("Failed to enable the RB-Y1 control manager")

    def _read_joint_limits(self):
        try:
            dynamics = self.robot.get_dynamics()
            model = self.sdk.Model_M()
            state = dynamics.make_state(["base"], model.robot_joint_names)
            return (
                np.asarray(dynamics.get_limit_q_lower(state), dtype=float),
                np.asarray(dynamics.get_limit_q_upper(state), dtype=float),
            )
        except Exception:
            return None, None

    @staticmethod
    def real_angle(q_goal, q_current):
        q_goal = np.asarray(q_goal, dtype=float)
        q_current = np.asarray(q_current, dtype=float)
        return q_current + (q_goal - q_current + np.pi) % (2.0 * np.pi) - np.pi

    def get_hardware_state(self) -> dict[str, np.ndarray]:
        """Return the Model-M state keys consumed by the MuJoCo controller."""
        state = self.robot.get_state()
        q = np.asarray(state.position, dtype=float)
        if q.size < 26:
            raise RuntimeError(f"Expected at least 26 RB-Y1 positions, received {q.size}")
        qd_raw = getattr(state, "velocity", None)
        qd = np.zeros_like(q) if qd_raw is None else np.asarray(qd_raw, dtype=float)
        return {
            "q_base": q[0:4].copy(), "qd_base": qd[0:4].copy(),
            "q_torso": q[4:10].copy(), "qd_torso": qd[4:10].copy(),
            "q_right": q[10:17].copy(), "qd_right": qd[10:17].copy(),
            "q_left": q[17:24].copy(), "qd_left": qd[17:24].copy(),
            "q_head": q[24:26].copy(), "qd_head": qd[24:26].copy(),
        }

    def _clip(self, values, start: int, size: int):
        values = np.asarray(values, dtype=float).reshape(size)
        if self.q_lower is None or self.q_upper is None:
            return values
        return np.clip(values, self.q_lower[start:start + size], self.q_upper[start:start + size])

    def send_joint_command_arms_head_base(
        self,
        q_goal_left,
        q_goal_right,
        q_goal_torso=None,
        q_goal_head=None,
        q_arm_lower_limit=None,
        q_arm_upper_limit=None,
        q_torso_lower_limit=None,
        q_torso_upper_limit=None,
        q_head_lower_limit=None,
        q_head_upper_limit=None,
        position_control=None,
        delta_x=0.0,
        delta_y=0.0,
        delta_yaw=0.0,
        fix_head_pose=False,
        fix_base_pose=True,
        min_time=0.05,
    ):
        """Stream torso/arms/head goals with the legacy method signature.

        Wheel deltas are deliberately rejected unless fixed: the monolith's
        base controller requires live odometry and a separate command stream,
        and silently dropping base motion would be unsafe.
        """
        del q_torso_lower_limit, q_torso_upper_limit
        if not fix_base_pose and any(abs(x) > 1e-12 for x in (delta_x, delta_y, delta_yaw)):
            raise NotImplementedError(
                "Package-local hardware control currently fixes the mobile base. "
                "Use the monolith RobotMwithBase for odometry-feedback base motion."
            )
        current = self.get_hardware_state()
        q_torso = current["q_torso"] if q_goal_torso is None else np.asarray(q_goal_torso, dtype=float)
        q_head = HEAD_READY_RAD if fix_head_pose else (
            current["q_head"] if q_goal_head is None else np.asarray(q_goal_head, dtype=float)
        )
        q_right = np.asarray(q_goal_right, dtype=float).reshape(7)
        q_left = np.asarray(q_goal_left, dtype=float).reshape(7)

        if q_arm_lower_limit is not None and q_arm_upper_limit is not None:
            lo, hi = np.asarray(q_arm_lower_limit), np.asarray(q_arm_upper_limit)
            q_right, q_left = np.clip(q_right, lo[:7], hi[:7]), np.clip(q_left, lo[7:14], hi[7:14])
        else:
            q_right, q_left = self._clip(q_right, 10, 7), self._clip(q_left, 17, 7)
        if q_head_lower_limit is not None and q_head_upper_limit is not None:
            q_head = np.clip(q_head, q_head_lower_limit, q_head_upper_limit)
        else:
            q_head = self._clip(q_head, 24, 2)
        q_torso = self._clip(q_torso, 4, 6)

        q_body = np.concatenate((q_torso, q_right, q_left))
        hold_s = 1.0
        position = self.position_control if position_control is None else bool(position_control)
        header = self.sdk.CommandHeaderBuilder().set_control_hold_time(hold_s)
        if position:
            body = (self.sdk.JointPositionCommandBuilder()
                    .set_command_header(header)
                    .set_position(q_body.tolist())
                    .set_velocity_limit(([5.0] * 6 + [15.0] * 14))
                    .set_acceleration_limit(([5.0] * 6 + [15.0] * 14))
                    .set_minimum_time(max(float(min_time), 0.05)))
        else:
            body = (self.sdk.JointImpedanceControlCommandBuilder()
                    .set_command_header(header)
                    .set_position(q_body.tolist())
                    .set_torque_limit(([500.0] * 6 + [70.0, 70.0, 70.0, 36.0, 36.0, 36.0, 36.0] * 2))
                    .set_stiffness(([800.0] * 6 + [200.0, 200.0, 200.0, 200.0, 40.0, 100.0, 100.0] * 2))
                    .set_damping_ratio(0.8)
                    .set_minimum_time(max(float(min_time), 0.05)))
        head = (self.sdk.JointPositionCommandBuilder()
                .set_command_header(self.sdk.CommandHeaderBuilder().set_control_hold_time(hold_s))
                .set_position(q_head.tolist())
                .set_minimum_time(max(float(min_time), 0.05)))
        components = self.sdk.ComponentBasedCommandBuilder().set_body_command(body).set_head_command(head)
        return self.stream.send_command(self.sdk.RobotCommandBuilder().set_command(components))

    def send_goals(self, goals, *, zero_torso_yaw=True, min_time=0.05):
        """Send a ``RetargetOutput`` or legacy joint-target dictionary."""
        current = self.get_hardware_state()
        q_right = self.real_angle(_goal_value(goals, "q_goal_right"), current["q_right"])
        q_left = self.real_angle(_goal_value(goals, "q_goal_left"), current["q_left"])
        q_torso = _goal_value(goals, "q_goal_torso", current["q_torso"])
        if q_torso is not None:
            q_torso = np.asarray(q_torso, dtype=float).copy()
            if zero_torso_yaw:
                q_torso[-1] = 0.0
        return self.send_joint_command_arms_head_base(
            q_goal_left=q_left, q_goal_right=q_right, q_goal_torso=q_torso,
            q_goal_head=_goal_value(goals, "q_goal_head", current["q_head"]),
            fix_base_pose=True, min_time=min_time,
        )

    def reset_egoengine_ready_pose(self, min_time=3.0):
        return self.send_joint_command_arms_head_base(
            q_goal_left=LEFT_READY_RAD, q_goal_right=RIGHT_READY_RAD,
            q_goal_torso=TORSO_READY_RAD, q_goal_head=HEAD_READY_RAD,
            position_control=True, min_time=min_time,
        )

    def shutdown(self):
        for obj, method in ((getattr(self, "stream", None), "cancel"),
                            (getattr(self, "robot", None), "cancel_control"),
                            (getattr(self, "robot", None), "stop_state_update"),
                            (getattr(self, "robot", None), "disconnect")):
            try:
                if obj is not None and hasattr(obj, method):
                    getattr(obj, method)()
            except Exception:
                pass

    close = shutdown

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.shutdown()


# Compatibility name used by the v9 monolith demo.
RobotMwithBase = RBY1HardwareController
