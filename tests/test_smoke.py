# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Smoke tests: package imports, assets exist, models load, controllers build."""

import mujoco
import numpy as np

import rby1_teleop
from rby1_teleop import (
    SAMPLE_MOTION,
    SPECS_DIR,
    URDF_RBY1,
    URDF_RBY1_XHAND,
    XML_RBY1,
    XML_RBY1_MOCAP,
    XML_RBY1_XHAND,
    XML_XHAND_LEFT,
    XML_XHAND_RIGHT,
)
from rby1_teleop.control import RBY1MuJoCoController, RBY1WithXHandMuJoCoController


def test_assets_exist():
    for path in (SAMPLE_MOTION, URDF_RBY1, URDF_RBY1_XHAND, XML_RBY1,
                 XML_RBY1_MOCAP, XML_RBY1_XHAND, XML_XHAND_LEFT, XML_XHAND_RIGHT):
        assert path.exists(), path
    for npz in ("rby1_v1_3.npz", "rby1_v1_3_xhand.npz", "xhand_left.npz", "xhand_right.npz"):
        assert (SPECS_DIR / npz).exists(), npz


def test_spec_signatures_match_vendored_urdfs():
    """The committed spec npz must match the committed source models byte-for-byte."""
    from geo_kin_core.spec import verify_signature

    assert verify_signature(str(SPECS_DIR / "rby1_v1_3.npz"), str(URDF_RBY1))
    assert verify_signature(str(SPECS_DIR / "rby1_v1_3_xhand.npz"), str(URDF_RBY1_XHAND))
    assert verify_signature(str(SPECS_DIR / "xhand_left.npz"), str(XML_XHAND_LEFT))
    assert verify_signature(str(SPECS_DIR / "xhand_right.npz"), str(XML_XHAND_RIGHT))


def test_models_load_and_controllers_build():
    for xml, cls, n_hand in ((XML_RBY1_MOCAP, RBY1MuJoCoController, 1),
                             (XML_RBY1_XHAND, RBY1WithXHandMuJoCoController, 12)):
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        controller = cls(model, data)
        assert len(controller.right_arm_qpos_addrs) == 7
        assert len(controller.left_arm_qpos_addrs) == 7
        assert len(controller.torso_qpos_addrs) == 6
        assert len(controller.head_qpos_addrs) == 2
        assert len(controller.right_hand_qpos_addrs) == n_hand
        controller.setup_mocap_body("base_mocap_mover")
        assert controller.mocap_enabled
        controller.set_joint_goals({
            "q_goal_right": controller.RIGHT_READY_RAD,
            "q_goal_left": controller.LEFT_READY_RAD,
            "q_goal_torso": controller.TORSO_READY_POS_RAD,
        })
        controller.update_kinematic()
        np.testing.assert_allclose(controller.q_current_right,
                                   controller.RIGHT_READY_RAD, atol=1e-9)


def test_sample_stream_loads():
    from geo_kin_core.frames import load_frames

    stream = load_frames(SAMPLE_MOTION)
    assert len(stream) > 0
    frame = stream[0]
    assert frame.left_sew is not None and frame.right_sew is not None
    assert frame.R_world_upper_body is not None
    assert frame.left_fingers is not None
    # The tcp / elbow retarget modes need the MCP centroid TCP goals.
    assert "left_finger_mcp_centroid" in frame.extras
    assert "right_finger_mcp_centroid" in frame.extras


def test_version():
    assert rby1_teleop.__version__
