# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Hardware contract tests with fake SDK objects; no robot is contacted."""

from types import SimpleNamespace

import numpy as np

from rby1_teleop.control.hw.rby1 import RobotMwithBase
from rby1_teleop.control.hw.xhand import XHAND_Q_UPPER, XHandPair


class _Builder:
    def __init__(self):
        self.values = {}

    def __getattr__(self, name):
        if not name.startswith("set_"):
            raise AttributeError(name)
        def set_value(value):
            self.values[name] = value
            return self
        return set_value


class _SDK:
    class ControlManagerState:
        class State:
            MinorFault = "minor"
            MajorFault = "major"

    class Model_M:
        robot_joint_names = [f"joint_{i}" for i in range(26)]

    CommandHeaderBuilder = JointPositionCommandBuilder = _Builder
    JointImpedanceControlCommandBuilder = ComponentBasedCommandBuilder = _Builder
    RobotCommandBuilder = _Builder


class _Dynamics:
    def make_state(self, *_args):
        return object()

    def get_limit_q_lower(self, _state):
        return np.full(26, -2.0)

    def get_limit_q_upper(self, _state):
        return np.full(26, 2.0)


class _Stream:
    def __init__(self):
        self.commands = []

    def send_command(self, command):
        self.commands.append(command)
        return command

    def cancel(self):
        pass


class _Robot:
    def __init__(self):
        self.stream = _Stream()
        self.state = SimpleNamespace(position=np.arange(26) / 100.0,
                                     velocity=np.arange(26) / 1000.0)

    def connect(self): return True
    def is_connected(self): return True
    def is_power_on(self, _pattern): return True
    def power_on(self, _pattern): return True
    def is_servo_on(self, _pattern): return True
    def servo_on(self, _pattern): return True
    def get_control_manager_state(self): return SimpleNamespace(state="normal")
    def reset_fault_control_manager(self): return True
    def enable_control_manager(self): return True
    def create_command_stream(self): return self.stream
    def get_dynamics(self): return _Dynamics()
    def get_state(self): return self.state
    def cancel_control(self): pass
    def disconnect(self): pass


def test_robot_m_compatible_usage_and_state_mapping():
    robot = _Robot()
    hw = RobotMwithBase("fake:50051", servo=".*", power_device=".*",
                        controller=None, sdk=_SDK, robot=robot)
    state = hw.get_hardware_state()
    np.testing.assert_allclose(state["q_torso"], np.arange(4, 10) / 100.0)
    np.testing.assert_allclose(state["q_right"], np.arange(10, 17) / 100.0)
    np.testing.assert_allclose(state["q_left"], np.arange(17, 24) / 100.0)
    hw.send_joint_command_arms_head_base(
        q_goal_left=np.zeros(7), q_goal_right=np.zeros(7),
        q_goal_torso=np.zeros(6), q_goal_head=np.zeros(2), min_time=0.05,
    )
    assert len(robot.stream.commands) == 1


def test_send_goals_wraps_and_streams():
    robot = _Robot()
    hw = RobotMwithBase("fake:50051", sdk=_SDK, robot=robot)
    goals = SimpleNamespace(
        q_goal_right=np.full(7, 2 * np.pi + 0.2),
        q_goal_left=np.full(7, -2 * np.pi - 0.2),
        q_goal_torso=np.zeros(6), q_goal_head=np.zeros(2),
    )
    hw.send_goals(goals)
    assert len(robot.stream.commands) == 1


class _Hand:
    def __init__(self, comport):
        self.comport, self.commands = comport, []

    def grasp(self, q):
        self.commands.append(q)


def test_xhand_pair_clips_and_preserves_v9_goal_names():
    pair = XHandPair("unused", left_serial="L", right_serial="R", controller_cls=_Hand)
    goals = {"q_goal_left_hand": np.full(12, -10.0),
             "q_goal_right_hand": np.full(12, 10.0)}
    pair.send(goals)
    np.testing.assert_allclose(pair.right.commands[0], XHAND_Q_UPPER)
    assert len(pair.left.commands[0]) == 12
