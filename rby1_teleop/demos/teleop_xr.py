# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""RBY1 live XR teleoperation demo (sim; hardware hook pending).

Live loop: XR headset (Meta Quest via the monolith's XRRTCBodyPoseDevice) ->
``resolve_session(robot='rby1', hand=...)`` -> MuJoCo (kinematic follow).

The XR device stack is imported from a SEW-Geometric-Teleop checkout
(``--monolith_path`` / ``GEO_TELEOP_MONOLITH``) until the xrt_device repo is
split out — see :mod:`rby1_teleop.input.xr_adapter`.

``--hw`` (RB-Y1 hardware over rby1_sdk + XHand serial) is NOT wired up yet:
the hardware plumbing (rby1_api_utils / realtime_control / xhand APIs) is the
next port from the monolith into ``rby1_teleop.control.hw``. Until then use
the monolith's demo_rby1_xr_robot_teleop_v9_hw_all.py for hardware runs.

Example:
    python -m rby1_teleop.demos.teleop_xr --retarget_mode tcp --mobile_base
"""

import argparse
import time
import traceback

import mujoco
import mujoco.viewer
import numpy as np

from geo_kin_core.session import resolve_session
from geo_kin_core.viz import HumanCapsuleViz, capsules, draw_filtered_sew

from rby1_teleop import XML_RBY1_MOCAP, XML_RBY1_XHAND
from rby1_teleop.control import RBY1MuJoCoController, RBY1WithXHandMuJoCoController
from rby1_teleop.input import XRDeviceAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="RBY1 XR teleoperation (sim)")
    parser.add_argument("--hand", choices=["xhand", "none"], default="xhand")
    parser.add_argument("--retarget_mode", default="tcp",
                        choices=["pose", "tcp", "left_elbow", "right_elbow"])
    parser.add_argument("--mobile_base", action="store_true",
                        help="Enable SE(2) base placement")
    parser.add_argument("--base_alignment", choices=["manual", "mocap"], default="manual")
    parser.add_argument("--max_fr", type=int, default=60, help="Control rate cap (Hz)")
    parser.add_argument("--no_safety_filter", action="store_true")
    parser.add_argument("--elbow_filter_hz", type=float, default=None)
    parser.add_argument("--keep_torso_yaw", action="store_true",
                        help="Keep the solved torso joint 5 (default: zeroed, the "
                             "omni base owns that yaw)")
    parser.add_argument("--monolith_path", default=None,
                        help="SEW-Geometric-Teleop checkout (else GEO_TELEOP_MONOLITH)")
    parser.add_argument("--record_data", action="store_true",
                        help="Record incoming bone CSV via the device")
    parser.add_argument("--no_human_overlay", action="store_true")
    parser.add_argument("--hw", action="store_true",
                        help="RB-Y1 hardware (NOT wired up yet in this repo)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hw:
        raise NotImplementedError(
            "--hw is not wired up in rby1_teleop yet: the hardware plumbing "
            "(rby1_api_utils / realtime_control / xhand APIs) is the next port "
            "into rby1_teleop.control.hw. Use the monolith's "
            "demo_rby1_xr_robot_teleop_v9_hw_all.py for hardware runs meanwhile.")

    if args.hand == "xhand":
        xml, controller_cls, hand = XML_RBY1_XHAND, RBY1WithXHandMuJoCoController, "xhand"
    else:
        xml, controller_cls, hand = XML_RBY1_MOCAP, RBY1MuJoCoController, None

    print(f"Loading model from: {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    model.opt.timestep = 0.005

    session = resolve_session(
        robot="rby1",
        hand=hand,
        model_xml=xml,
        control_rate_hz=float(args.max_fr),
        elbow_filter_cutoff_hz=args.elbow_filter_hz,
        collision_avoidance=not args.no_safety_filter,
        retarget_mode=args.retarget_mode,
        base_alignment_mode=args.base_alignment,
        mobile_base=args.mobile_base,
    )
    print(f"Solver backend: {type(session).__module__}.{type(session).__name__}")

    controller = controller_cls(model, data, debug=False)
    controller.setup_mocap_body("base_mocap_mover")
    session.reset(controller.RIGHT_READY_RAD, controller.LEFT_READY_RAD)

    device = XRDeviceAdapter(monolith_path=args.monolith_path,
                             record_data=args.record_data)
    print("Waiting for the headset to connect...")

    frame_interval = 1.0 / args.max_fr
    with_hand_kwargs = args.hand == "xhand"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False,
    ) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0, 0, 1.0]

        overlay = None if args.no_human_overlay else HumanCapsuleViz(viewer)

        while not device.is_connected and viewer.is_running():
            viewer.sync()
            time.sleep(0.1)
        if device.is_connected:
            print("Headset connected — teleoperation live.")

        while viewer.is_running():
            loop_start = time.time()
            try:
                frame = device.get_frame()
                if frame is not None:
                    kwargs = dict(
                        q_current_right=controller.q_current_right,
                        q_current_left=controller.q_current_left,
                    )
                    if with_hand_kwargs:
                        try:
                            out = session.solve(
                                frame,
                                q_current_right_hand=getattr(controller, "q_current_right_hand", None),
                                q_current_left_hand=getattr(controller, "q_current_left_hand", None),
                                **kwargs)
                        except TypeError:
                            with_hand_kwargs = False
                            out = session.solve(frame, **kwargs)
                    else:
                        out = session.solve(frame, **kwargs)

                    if not args.keep_torso_yaw and out.q_goal_torso is not None:
                        out.q_goal_torso = np.asarray(out.q_goal_torso, dtype=float).copy()
                        out.q_goal_torso[-1] = 0.0
                    if out.p_world_base is not None and out.R_world_base is not None:
                        controller.update_mocap_body(out.p_world_base, out.R_world_base)
                    controller.set_joint_goals(out)
            except Exception as e:
                print(f"Error during solve: {e}")
                traceback.print_exc()

            controller.update_kinematic()

            capsules.clear(viewer)
            draw_filtered_sew(viewer, session, to_world=controller.get_sew_transform())
            if overlay is not None and frame is not None:
                R_mocap = getattr(session, "R_mocap_world", None)
                p_mocap = getattr(session, "p_mocap_world", None)
                if R_mocap is not None and p_mocap is not None:
                    overlay.set_base_offset(p_mocap, np.asarray(R_mocap).T)
                overlay.draw(frame)
            viewer.sync()

            lag = time.time() - loop_start
            if lag < frame_interval:
                time.sleep(frame_interval - lag)

    device.cleanup()
    print("Teleoperation ended.")


if __name__ == "__main__":
    main()
