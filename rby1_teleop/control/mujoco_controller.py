# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""MuJoCo controllers for the RB-Y1 (base robot + XHand variant).

Ported from the monolith's ``rby1_teleop/rby1_mujoco_controller_v3.py``
(RBY1MuJoCoController) and
``xhand_teleop/controllers/rby1_with_xhand_mujoco_controller_v3.py``
(RBY1WithXHandMuJoCoController). Behavior preserved; the only addition is that
``set_joint_goals`` also accepts a ``geo_kin_core.types.RetargetOutput``
(attribute access) in place of the legacy goals dict.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from typing import Mapping, Optional


_GOAL_KEYS = (
    "q_goal_right", "q_goal_left", "q_goal_torso", "q_goal_head",
    "q_goal_right_hand", "q_goal_left_hand",
    "left_gripper_val", "right_gripper_val",
)


def _as_goal_dict(goals):
    """Accept the legacy goals dict or a RetargetOutput-shaped object."""
    if isinstance(goals, dict):
        return goals
    return {key: getattr(goals, key, None) for key in _GOAL_KEYS}


class RBY1MuJoCoController:
    """
    MuJoCo Controller for RBY1 v3 robot.
    Applies joint positions (position/torque/kinematic modes).
    """

    def __init__(self, mujoco_model, mujoco_data, kp=150.0, kd=None, debug=False):
        """
        Args:
            mujoco_model: MuJoCo model
            mujoco_data: MuJoCo data
            kp: Proportional gain for joint control
            kd: Derivative gain for joint control (auto-computed if None)
            debug: Enable debug output
        """
        self.model = mujoco_model
        self.data = mujoco_data
        self.debug = debug

        # Enable torso gravity compensation by default
        self.torso_gravity_compensation_enabled = True

        # Get joint indices in MuJoCo model
        self._setup_joint_indices()

        # Control gains
        self.kp = kp
        self.kd = 1 * np.sqrt(kp) if kd is None else kd

        # Torso control gains (more conservative to prevent unwanted movement)
        self.Kp_torso = 100.0  # Lower gain for torso stability
        self.Kd_torso = 20.0  # Damping for torso

        # Initial positions
        self.RIGHT_READY_RAD = np.array(
            [-0.261799, -0.261799, 0.0, -1.894395, 0.0, 0.0, 0.0], dtype=float
        )
        self.LEFT_READY_RAD = np.array(
            [-0.261799, 0.261799, 0.0, -1.894395, 0.0, 0.0, 0.0], dtype=float
        )
        self.TORSO_READY_POS_RAD = np.array([0.0, 0.5, -1.0, 0.5, 0.0, 0.0])

        self.RIGHT_GRIPER_READY_RAD = np.array([0.05])
        self.LEFT_GRIPER_READY_RAD = np.array([0.05])

        # Goal joint angles
        self.q_goal_right = np.zeros(7)
        self.q_goal_left = np.zeros(7)
        self.q_goal_torso = None
        self.q_goal_head = None
        self.q_goal_right_hand = None
        self.q_goal_left_hand = None

        # Initialize with current joint positions
        self._update_current_positions()

        # Initialize goals to current positions
        self.q_goal_right = self.RIGHT_READY_RAD.copy()
        self.q_goal_left = self.LEFT_READY_RAD.copy()
        self.q_goal_torso = self.TORSO_READY_POS_RAD.copy()
        self.q_goal_head = self.q_current_head.copy()
        self.q_goal_right_hand = self.RIGHT_GRIPER_READY_RAD.copy()
        self.q_goal_left_hand = self.LEFT_GRIPER_READY_RAD.copy()

        # Build actuator cache for position control
        self._build_actuator_cache()

        if self.debug:
            print("RBY1 MuJoCo Controller initialized")

    def _build_actuator_cache(self):
        """Cache actuator IDs for position control."""
        self.right_arm_actuator_ids = []
        self.left_arm_actuator_ids = []
        self.torso_actuator_ids = []
        self.head_actuator_ids = []
        self.right_hand_actuator_ids = []
        self.left_hand_actuator_ids = []
        self.other_actuator_ids = []

        controlled_actuators = set()

        # Helper to find actuator for joint
        def get_actuator_for_joint(joint_id):
            for i in range(self.model.nu):
                if self.model.actuator_trnid[i, 0] == joint_id:
                    return i
            return None

        for joint_ids, actuator_list in (
            (self.right_arm_joint_ids, self.right_arm_actuator_ids),
            (self.left_arm_joint_ids, self.left_arm_actuator_ids),
            (self.torso_joint_ids, self.torso_actuator_ids),
            (self.head_joint_ids, self.head_actuator_ids),
            (self.right_hand_joint_ids, self.right_hand_actuator_ids),
            (self.left_hand_joint_ids, self.left_hand_actuator_ids),
        ):
            for joint_id in joint_ids:
                aid = get_actuator_for_joint(joint_id)
                if aid is not None:
                    actuator_list.append(aid)
                    controlled_actuators.add(aid)

        for i in range(self.model.nu):
            if i not in controlled_actuators:
                self.other_actuator_ids.append(i)

        # Build joint-name to actuator-id map for modular torque command interfaces.
        self.joint_name_to_actuator_id = {}
        group_pairs = (
            (self.right_arm_joint_names, self.right_arm_actuator_ids),
            (self.left_arm_joint_names, self.left_arm_actuator_ids),
            (self.torso_joint_names, self.torso_actuator_ids),
            (self.head_joint_names, self.head_actuator_ids),
            (self.right_hand_joint_names, self.right_hand_actuator_ids),
            (self.left_hand_joint_names, self.left_hand_actuator_ids),
        )
        for joint_names, actuator_ids in group_pairs:
            for joint_name, actuator_id in zip(joint_names, actuator_ids):
                self.joint_name_to_actuator_id[joint_name] = actuator_id

    def update_torque_control(
        self,
        joint_torque_by_name,
        reset_ctrl=False,
    ):
        """Apply joint-name torque commands to actuators with ctrlrange clipping.

        Args:
            joint_torque_by_name: Mapping of joint name -> torque command.
            reset_ctrl: If True, zero all actuator commands before applying torques.
        """
        if reset_ctrl:
            self.data.ctrl[:] = 0.0

        for joint_name, torque in joint_torque_by_name.items():
            actuator_id = self.joint_name_to_actuator_id.get(joint_name)
            if actuator_id is None:
                if self.debug:
                    print(f"[update_torque_control] Unknown joint name: {joint_name}")
                continue

            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = np.clip(float(torque), low, high)

    def _setup_joint_indices(self):
        """Setup joint indices and names for both arms."""
        self.right_arm_joint_names = [f"right_arm_{i}" for i in range(7)]
        self.left_arm_joint_names = [f"left_arm_{i}" for i in range(7)]
        self.torso_joint_names = [f"torso_{i}" for i in range(6)]
        self.head_joint_names = ["head_0", "head_1"]

        # Default 1-DOF gripper joints (overridden by the XHand subclass)
        self.right_hand_joint_names = ["gripper_finger_r2"]
        self.left_hand_joint_names = ["gripper_finger_l1"]

        # Build joint name to ID mapping
        joint_name2id = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                joint_name2id[joint_name] = i
        self._joint_name2id = joint_name2id

        def qpos_addrs(names):
            return [self.model.jnt_qposadr[joint_name2id[n]] for n in names if n in joint_name2id]

        def qvel_addrs(names):
            return [self.model.jnt_dofadr[joint_name2id[n]] for n in names if n in joint_name2id]

        def joint_ids(names):
            return [joint_name2id[n] for n in names if n in joint_name2id]

        self.right_arm_qpos_addrs = qpos_addrs(self.right_arm_joint_names)
        self.left_arm_qpos_addrs = qpos_addrs(self.left_arm_joint_names)
        self.torso_qpos_addrs = qpos_addrs(self.torso_joint_names)
        self.head_qpos_addrs = qpos_addrs(self.head_joint_names)
        self.right_hand_qpos_addrs = qpos_addrs(self.right_hand_joint_names)
        self.left_hand_qpos_addrs = qpos_addrs(self.left_hand_joint_names)

        self.right_arm_qvel_addrs = qvel_addrs(self.right_arm_joint_names)
        self.left_arm_qvel_addrs = qvel_addrs(self.left_arm_joint_names)
        self.torso_qvel_addrs = qvel_addrs(self.torso_joint_names)
        self.head_qvel_addrs = qvel_addrs(self.head_joint_names)
        self.right_hand_qvel_addrs = qvel_addrs(self.right_hand_joint_names)
        self.left_hand_qvel_addrs = qvel_addrs(self.left_hand_joint_names)

        self.right_arm_joint_ids = joint_ids(self.right_arm_joint_names)
        self.left_arm_joint_ids = joint_ids(self.left_arm_joint_names)
        self.torso_joint_ids = joint_ids(self.torso_joint_names)
        self.head_joint_ids = joint_ids(self.head_joint_names)
        self.right_hand_joint_ids = joint_ids(self.right_hand_joint_names)
        self.left_hand_joint_ids = joint_ids(self.left_hand_joint_names)

    def _update_current_positions(self):
        """Update current joint positions from MuJoCo data."""
        self.q_current_right = np.array(
            [self.data.qpos[addr] for addr in self.right_arm_qpos_addrs]
        )
        self.q_current_left = np.array([self.data.qpos[addr] for addr in self.left_arm_qpos_addrs])
        self.q_current_torso = np.array([self.data.qpos[addr] for addr in self.torso_qpos_addrs])
        self.q_current_head = np.array([self.data.qpos[addr] for addr in self.head_qpos_addrs])

        self.qd_current_right = np.array(
            [self.data.qvel[addr] for addr in self.right_arm_qvel_addrs]
        )
        self.qd_current_left = np.array([self.data.qvel[addr] for addr in self.left_arm_qvel_addrs])
        self.qd_current_torso = np.array([self.data.qvel[addr] for addr in self.torso_qvel_addrs])
        self.qd_current_head = np.array([self.data.qvel[addr] for addr in self.head_qvel_addrs])

    def set_joint_goals(self, goals):
        """
        Set target joint angles from a goals dict or a RetargetOutput.

        Args:
            goals: dict with 'q_goal_right', 'q_goal_left', 'q_goal_torso',
                'q_goal_head', 'q_goal_*_hand', '*_gripper_val' keys — or a
                geo_kin_core RetargetOutput carrying the same attributes.
        """
        goals = _as_goal_dict(goals)
        if goals.get("q_goal_right") is not None:
            self.q_goal_right = goals["q_goal_right"]
        if goals.get("q_goal_left") is not None:
            self.q_goal_left = goals["q_goal_left"]
        if goals.get("q_goal_torso") is not None:
            self.q_goal_torso = goals["q_goal_torso"]
        if goals.get("q_goal_head") is not None:
            self.q_goal_head = goals["q_goal_head"]
        if goals.get("q_goal_right_hand") is not None:
            self.q_goal_right_hand = goals["q_goal_right_hand"]
        if goals.get("q_goal_left_hand") is not None:
            self.q_goal_left_hand = goals["q_goal_left_hand"]

        if goals.get("left_gripper_val") is not None:
            # Map distance to joint position (assuming 0-0.1m distance maps to 0-0.05m joint)
            val = np.clip(goals["left_gripper_val"] / 2.0, 0.0, 0.05)
            self.q_goal_left_hand = np.array([val])

        if goals.get("right_gripper_val") is not None:
            val = np.clip(goals["right_gripper_val"] / 2.0, 0.0, 0.05)
            self.q_goal_right_hand = np.array([val])

    @staticmethod
    def real_angle(q_goal, q_curr):
        # Safety function to wrap the angle to [-pi, pi], avoid the sign flipping
        q_diff = q_goal - q_curr
        q_diff_wrapped = (q_diff + np.pi) % (2 * np.pi) - np.pi
        q_goal_real = q_curr + q_diff_wrapped

        return q_goal_real

    def update(self, engaged=True):
        """
        Compute and apply torques based on current goals.

        Args:
            engaged: If False, applies position holding control instead of tracking goals.
        """
        self._update_current_positions()
        torques_right, torques_left, torques_torso, torques_head = self.compute_control_torques(
            engaged
        )
        self.apply_torques(torques_right, torques_left, torques_torso, torques_head)

    def update_position_control(self):
        """
        Apply joint position goals directly to actuators (for position actuators).
        Sets untracked actuators (wheels, etc.) to zero.
        """
        self._update_current_positions()

        # Right arm
        q_goal_right_real = self.real_angle(self.q_goal_right, self.q_current_right)
        for i, aid in enumerate(self.right_arm_actuator_ids):
            if i < len(q_goal_right_real):
                self.data.ctrl[aid] = q_goal_right_real[i]

        # Left arm
        q_goal_left_real = self.real_angle(self.q_goal_left, self.q_current_left)
        for i, aid in enumerate(self.left_arm_actuator_ids):
            if i < len(q_goal_left_real):
                self.data.ctrl[aid] = q_goal_left_real[i]

        # Torso
        if self.q_goal_torso is not None:
            for i, aid in enumerate(self.torso_actuator_ids):
                if i < len(self.q_goal_torso):
                    self.data.ctrl[aid] = self.q_goal_torso[i]

        # Head
        if self.q_goal_head is not None:
            for i, aid in enumerate(self.head_actuator_ids):
                if i < len(self.q_goal_head):
                    self.data.ctrl[aid] = self.q_goal_head[i]

        # Right Hand
        if self.q_goal_right_hand is not None:
            for i, aid in enumerate(self.right_hand_actuator_ids):
                if i < len(self.q_goal_right_hand):
                    self.data.ctrl[aid] = -self.q_goal_right_hand[i]  # [-0.05,0] for right hand

        # Left Hand
        if self.q_goal_left_hand is not None:
            for i, aid in enumerate(self.left_hand_actuator_ids):
                if i < len(self.q_goal_left_hand):
                    self.data.ctrl[aid] = self.q_goal_left_hand[i]

        # Others (Wheels) -> 0
        for aid in self.other_actuator_ids:
            self.data.ctrl[aid] = 0.0

    def compute_control_torques(self, engaged=True):
        """
        Compute joint torques for both arms using inverse dynamics.
        """
        right_mass_matrix = self._get_arm_mass_matrix("right")
        left_mass_matrix = self._get_arm_mass_matrix("left")
        torso_mass_matrix = self._get_torso_mass_matrix()

        right_compensation = self._get_arm_torque_compensation("right")
        left_compensation = self._get_arm_torque_compensation("left")

        # Always apply gravity compensation for torso joints (if enabled)
        torso_compensation = (
            self._get_torso_gravity_compensation()
            if self.torso_gravity_compensation_enabled
            else np.zeros(len(self.torso_qvel_addrs))
        )

        if not engaged:
            # Use lower gains when disengaged but still maintain proper control
            hold_kp = 50.0
            hold_kd = 10.0

            # No position error when holding (damping only)
            qdd_desired_right = hold_kp * np.zeros(7) + hold_kd * (-self.qd_current_right)
            qdd_desired_left = hold_kp * np.zeros(7) + hold_kd * (-self.qd_current_left)
        else:
            # Normal PD control when engaged
            q_error_right = self.q_goal_right - self.q_current_right
            q_error_right = (q_error_right + np.pi) % (2 * np.pi) - np.pi
            qdd_desired_right = self.kp * q_error_right + self.kd * (-self.qd_current_right)

            q_error_left = self.q_goal_left - self.q_current_left
            q_error_left = (q_error_left + np.pi) % (2 * np.pi) - np.pi
            qdd_desired_left = self.kp * q_error_left + self.kd * (-self.qd_current_left)

        # Compute torques using inverse dynamics: tau = M * qdd + compensation
        torques_right = np.dot(right_mass_matrix, qdd_desired_right) + right_compensation
        torques_left = np.dot(left_mass_matrix, qdd_desired_left) + left_compensation

        # Torso PD control to hold initial position and prevent drift
        q_error_torso = self.q_goal_torso - self.q_current_torso
        q_error_torso = (q_error_torso + np.pi) % (2 * np.pi) - np.pi
        qdd_desired_torso = self.Kp_torso * q_error_torso + self.Kd_torso * (-self.qd_current_torso)

        torques_torso = np.dot(torso_mass_matrix, qdd_desired_torso) + torso_compensation

        # Head PD control (simple PD, no dynamics)
        kp_head = 50.0
        kd_head = 5.0
        q_error_head = self.q_goal_head - self.q_current_head
        q_error_head = (q_error_head + np.pi) % (2 * np.pi) - np.pi
        torques_head = kp_head * q_error_head + kd_head * (-self.qd_current_head)

        return torques_right, torques_left, torques_torso, torques_head

    def _get_arm_mass_matrix(self, arm_side):
        if arm_side == "right":
            qvel_addrs = self.right_arm_qvel_addrs
        else:
            qvel_addrs = self.left_arm_qvel_addrs

        full_mass_matrix = np.ndarray(
            shape=(self.model.nv, self.model.nv), dtype=np.float64, order="C"
        )
        mujoco.mj_fullM(self.model, full_mass_matrix, self.data.qM)
        full_mass_matrix = np.reshape(full_mass_matrix, (len(self.data.qvel), len(self.data.qvel)))

        return full_mass_matrix[qvel_addrs, :][:, qvel_addrs]

    def _get_arm_torque_compensation(self, arm_side):
        if arm_side == "right":
            qvel_addrs = self.right_arm_qvel_addrs
        else:
            qvel_addrs = self.left_arm_qvel_addrs

        bias_forces = np.zeros(7)
        for i, qvel_addr in enumerate(qvel_addrs[:7]):
            if qvel_addr < self.model.nv:
                bias_forces[i] = self.data.qfrc_bias[qvel_addr]
        return bias_forces

    def _get_torso_mass_matrix(self):
        qvel_addrs = self.torso_qvel_addrs
        full_mass_matrix = np.ndarray(
            shape=(self.model.nv, self.model.nv), dtype=np.float64, order="C"
        )
        mujoco.mj_fullM(self.model, full_mass_matrix, self.data.qM)
        full_mass_matrix = np.reshape(full_mass_matrix, (len(self.data.qvel), len(self.data.qvel)))

        torso_mass_matrix = full_mass_matrix[qvel_addrs, :][:, qvel_addrs]
        torso_mass_matrix += np.eye(len(qvel_addrs)) * 1e-6
        return torso_mass_matrix

    def _get_torso_gravity_compensation(self):
        gravity_compensation = np.zeros(len(self.torso_qvel_addrs))
        for i, qvel_addr in enumerate(self.torso_qvel_addrs):
            if qvel_addr < self.model.nv:
                gravity_compensation[i] = self.data.qfrc_bias[qvel_addr]
        return gravity_compensation

    def apply_torques(self, torques_right, torques_left, torques_torso=None, torques_head=None):
        """Apply computed torques to MuJoCo actuators."""
        groups = (
            (self.right_arm_joint_ids, torques_right),
            (self.left_arm_joint_ids, torques_left),
            (self.torso_joint_ids, torques_torso),
            (self.head_joint_ids, torques_head),
        )
        for joint_ids, torques in groups:
            if torques is None:
                continue
            for i, joint_id in enumerate(joint_ids):
                if i < len(torques):
                    for actuator_id in range(self.model.nu):
                        if self.model.actuator_trnid[actuator_id, 0] == joint_id:
                            self.data.ctrl[actuator_id] = torques[i]
                            break

    def update_kinematic(self):
        """
        Updates robot joint positions kinematically based on controller goals.
        This bypasses physics simulation and sets joint positions directly from goals.
        """
        # Ensure controller has latest qpos to compute real_angle correctly
        self._update_current_positions()

        # Right Arm
        if self.q_goal_right is not None:
            q_goal_right_real = self.real_angle(self.q_goal_right, self.q_current_right)
            for i, addr in enumerate(self.right_arm_qpos_addrs):
                if i < len(q_goal_right_real):
                    self.data.qpos[addr] = q_goal_right_real[i]

        # Left Arm
        if self.q_goal_left is not None:
            q_goal_left_real = self.real_angle(self.q_goal_left, self.q_current_left)
            for i, addr in enumerate(self.left_arm_qpos_addrs):
                if i < len(q_goal_left_real):
                    self.data.qpos[addr] = q_goal_left_real[i]

        # Torso
        if self.q_goal_torso is not None:
            for i, addr in enumerate(self.torso_qpos_addrs):
                if i < len(self.q_goal_torso):
                    self.data.qpos[addr] = self.q_goal_torso[i]

        # Head
        if self.q_goal_head is not None:
            for i, addr in enumerate(self.head_qpos_addrs):
                if i < len(self.q_goal_head):
                    self.data.qpos[addr] = self.q_goal_head[i]

        # Right Hand (sign flip: legacy 1-DOF gripper convention)
        if self.q_goal_right_hand is not None:
            for i, addr in enumerate(self.right_hand_qpos_addrs):
                if i < len(self.q_goal_right_hand):
                    self.data.qpos[addr] = -self.q_goal_right_hand[i]

        # Left Hand
        if self.q_goal_left_hand is not None:
            for i, addr in enumerate(self.left_hand_qpos_addrs):
                if i < len(self.q_goal_left_hand):
                    self.data.qpos[addr] = self.q_goal_left_hand[i]

        # Update Forward Kinematics
        mujoco.mj_forward(self.model, self.data)

        # Update controller internal state again so IK gets clean values
        self._update_current_positions()

    def update_sim_from_hardware(self, hardware_state):
        """
        Update MuJoCo simulation state directly from hardware readings.

        Args:
            hardware_state: Dict with 'q_right'/'qd_right'/'q_left'/'qd_left'/
                'q_torso'/'qd_torso'/'q_head'/'qd_head'/'q_*_hand'/'qd_*_hand'.
        """
        groups = (
            ("q_right", "qd_right", self.right_arm_qpos_addrs, self.right_arm_qvel_addrs),
            ("q_left", "qd_left", self.left_arm_qpos_addrs, self.left_arm_qvel_addrs),
            ("q_torso", "qd_torso", self.torso_qpos_addrs, self.torso_qvel_addrs),
            ("q_head", "qd_head", self.head_qpos_addrs, self.head_qvel_addrs),
            ("q_right_hand", "qd_right_hand", self.right_hand_qpos_addrs, self.right_hand_qvel_addrs),
            ("q_left_hand", "qd_left_hand", self.left_hand_qpos_addrs, self.left_hand_qvel_addrs),
        )
        for q_key, qd_key, qpos_addrs, qvel_addrs in groups:
            if q_key in hardware_state and len(hardware_state[q_key]) == len(qpos_addrs):
                for i, addr in enumerate(qpos_addrs):
                    self.data.qpos[addr] = hardware_state[q_key][i]
            if qd_key in hardware_state and len(hardware_state[qd_key]) == len(qvel_addrs):
                for i, addr in enumerate(qvel_addrs):
                    self.data.qvel[addr] = hardware_state[qd_key][i]

        # Update current position tracking variables for consistency
        self._update_current_positions()

    def get_current_joint_angles(self):
        """
        Get current joint angles for both arms, torso, and head.
        """
        self._update_current_positions()
        return (
            self.q_current_right.copy(),
            self.q_current_left.copy(),
            self.q_current_torso.copy(),
            self.q_current_head.copy(),
        )

    # --- Mocap Body Management ---
    def setup_mocap_body(self, mocap_body_name="base_mocap_mover"):
        self.mocap_body_name = mocap_body_name
        self.mocap_enabled = False
        self.mocap_index = None
        mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
        if mocap_body_id >= 0:
            for i in range(self.model.nmocap):
                if self.model.body_mocapid[mocap_body_id] == i:
                    self.mocap_index = i
                    self.mocap_enabled = True
                    break
            if not self.mocap_enabled and self.model.nmocap > 0:
                self.mocap_index = 0
                self.mocap_enabled = True

        if self.debug:
            print(f"Mocap body setup: {self.mocap_enabled}, index: {self.mocap_index}")

    def update_mocap_body(self, position, R_world_body):
        if not getattr(self, "mocap_enabled", False) or self.mocap_index is None:
            return
        try:
            self.data.mocap_pos[self.mocap_index] = position

            quat_xyzw = Rotation.from_matrix(R_world_body).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            self.data.mocap_quat[self.mocap_index] = quat_wxyz
        except Exception as e:
            if self.debug:
                print(f"Error updating mocap: {e}")

    def get_sew_transform(self):
        """
        Returns a function that transforms points from SEW frame to World frame.
        """
        # Find torso_5 joint ID
        torso_5_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "torso_5")

        if torso_5_id != -1:
            # Get the body ID driven by this joint
            body_id = self.model.jnt_bodyid[torso_5_id]

            # Get joint position (anchor) in world frame
            pos_joint_world = self.data.xanchor[torso_5_id]

            # Get body orientation in world frame
            rot_body_world = self.data.xmat[body_id].reshape(3, 3)

            # Offset from joint to SEW origin (in joint/body frame)
            offset = np.array([0, 0, 0.0])

            # SEW Origin in World Frame
            sew_origin_world = pos_joint_world + rot_body_world @ offset
            sew_rot_world = rot_body_world
        else:
            # Fallback if torso_5 not found (e.g. fixed base or different model)
            sew_origin_world = np.zeros(3)
            sew_rot_world = np.eye(3)

        def to_world(p_local):
            return sew_origin_world + sew_rot_world @ p_local

        return to_world


class RBY1WithXHandMuJoCoController(RBY1MuJoCoController):
    """
    MuJoCo Controller for RBY1 v3 robot with XHand.
    Applies joint positions including 12-DOF hand control.
    """

    def __init__(self, mujoco_model, mujoco_data, kp=150.0, kd=None, debug=False):
        super().__init__(mujoco_model, mujoco_data, kp, kd, debug)

        # Override initial hand goals to zeros (12 joints per hand)
        self.RIGHT_GRIPER_READY_RAD = np.zeros(12)
        self.LEFT_GRIPER_READY_RAD = np.zeros(12)

        self.q_goal_right_hand = self.RIGHT_GRIPER_READY_RAD.copy()
        self.q_goal_left_hand = self.LEFT_GRIPER_READY_RAD.copy()

    def _set_ctrl_with_clip(self, actuator_id, value):
        low, high = self.model.actuator_ctrlrange[actuator_id]
        self.data.ctrl[actuator_id] = np.clip(float(value), low, high)

    def _apply_hand_position_control(self):
        """Apply current hand goals to hand actuators in position-control mode."""
        if self.q_goal_right_hand is not None:
            for i, actuator_id in enumerate(self.right_hand_actuator_ids):
                if i < len(self.q_goal_right_hand):
                    self._set_ctrl_with_clip(actuator_id, self.q_goal_right_hand[i])

        if self.q_goal_left_hand is not None:
            for i, actuator_id in enumerate(self.left_hand_actuator_ids):
                if i < len(self.q_goal_left_hand):
                    self._set_ctrl_with_clip(actuator_id, self.q_goal_left_hand[i])

    def update_torque_control(
        self,
        torques_by_joint: Optional[Mapping[str, float]],
        zero_unset: bool = True,
        apply_hand_position: bool = True,
    ):
        """Map joint torques to MuJoCo actuators, optionally applying hand position control."""
        if zero_unset:
            self.data.ctrl[:] = 0.0

        if torques_by_joint is not None:
            for joint_name, tau in torques_by_joint.items():
                actuator_id = self.joint_name_to_actuator_id.get(joint_name)
                if actuator_id is None:
                    continue
                self._set_ctrl_with_clip(actuator_id, tau)

        if apply_hand_position:
            self._apply_hand_position_control()

    def _update_current_positions(self):
        super()._update_current_positions()
        if hasattr(self, 'right_hand_qpos_addrs') and len(self.right_hand_qpos_addrs) == 12:
            self.q_current_right_hand = np.array([self.data.qpos[addr] for addr in self.right_hand_qpos_addrs])
        else:
            self.q_current_right_hand = np.zeros(12)

        if hasattr(self, 'left_hand_qpos_addrs') and len(self.left_hand_qpos_addrs) == 12:
            self.q_current_left_hand = np.array([self.data.qpos[addr] for addr in self.left_hand_qpos_addrs])
        else:
            self.q_current_left_hand = np.zeros(12)

    def _setup_joint_indices(self):
        """Setup joint indices and names for both arms and xhands."""
        super()._setup_joint_indices()

        # Override hand joint names for XHand (12 DOF per hand)
        self.right_hand_joint_names = [
            "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
            "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
            "right_hand_mid_joint1", "right_hand_mid_joint2",
            "right_hand_ring_joint1", "right_hand_ring_joint2",
            "right_hand_pinky_joint1", "right_hand_pinky_joint2"
        ]
        self.left_hand_joint_names = [
            "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
            "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
            "left_hand_mid_joint1", "left_hand_mid_joint2",
            "left_hand_ring_joint1", "left_hand_ring_joint2",
            "left_hand_pinky_joint1", "left_hand_pinky_joint2"
        ]

        joint_name2id = self._joint_name2id

        self.right_hand_qpos_addrs = [
            self.model.jnt_qposadr[joint_name2id[n]] for n in self.right_hand_joint_names if n in joint_name2id]
        self.left_hand_qpos_addrs = [
            self.model.jnt_qposadr[joint_name2id[n]] for n in self.left_hand_joint_names if n in joint_name2id]
        self.right_hand_qvel_addrs = [
            self.model.jnt_dofadr[joint_name2id[n]] for n in self.right_hand_joint_names if n in joint_name2id]
        self.left_hand_qvel_addrs = [
            self.model.jnt_dofadr[joint_name2id[n]] for n in self.left_hand_joint_names if n in joint_name2id]
        self.right_hand_joint_ids = [
            joint_name2id[n] for n in self.right_hand_joint_names if n in joint_name2id]
        self.left_hand_joint_ids = [
            joint_name2id[n] for n in self.left_hand_joint_names if n in joint_name2id]

    def set_joint_goals(self, goals):
        """Set target joint angles without overriding hand goals with gripper vals."""
        goals = _as_goal_dict(goals)
        if goals.get("q_goal_right") is not None:
            self.q_goal_right = goals["q_goal_right"]
        if goals.get("q_goal_left") is not None:
            self.q_goal_left = goals["q_goal_left"]
        if goals.get("q_goal_torso") is not None:
            self.q_goal_torso = goals["q_goal_torso"]
        if goals.get("q_goal_head") is not None:
            self.q_goal_head = goals["q_goal_head"]
        if goals.get("q_goal_right_hand") is not None:
            self.q_goal_right_hand = goals["q_goal_right_hand"]
        if goals.get("q_goal_left_hand") is not None:
            self.q_goal_left_hand = goals["q_goal_left_hand"]

    def update_position_control(self):
        """Apply joint position goals directly to actuators."""
        super().update_position_control()
        # Base RBY1MuJoCoController negates the *right* "gripper" ctrl only (legacy 1-DoF convention).
        # XHand right hand must be re-applied without '-'.
        self._apply_hand_position_control()

    def update_kinematic(self):
        """Updates robot joint positions kinematically based on controller goals."""
        super().update_kinematic()
        # Same as update_position_control: undo base right-hand qpos negation for XHand joints.
        if self.q_goal_right_hand is not None:
            for i, addr in enumerate(self.right_hand_qpos_addrs):
                if i < len(self.q_goal_right_hand):
                    self.data.qpos[addr] = self.q_goal_right_hand[i]
        # Re-compute FK because qpos changed AFTER the base class ran mj_forward.
        mujoco.mj_fwdPosition(self.model, self.data)
        self._update_current_positions()
