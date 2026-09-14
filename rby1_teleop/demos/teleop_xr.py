# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Live XRT or MediaPipe teleoperation in simulation, with explicit optional hardware."""

import argparse
import time
import traceback

import mujoco
import mujoco.viewer
import numpy as np

from geo_kin_core.session import resolve_session
from geo_kin_core.viz import HumanCapsuleViz, capsules, draw_filtered_sew

from rby1_teleop import SPECS_DIR, XML_RBY1_MOCAP, XML_RBY1_XHAND
from rby1_teleop.control import RBY1MuJoCoController, RBY1WithXHandMuJoCoController
from rby1_teleop.input import XRDeviceAdapter, MediaPipeDeviceAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="RBY1 XR teleoperation (sim)")
    parser.add_argument("--hand", choices=["xhand", "none"], default="xhand")
    parser.add_argument("--retarget_mode", default="pose",
                        choices=["pose", "tcp", "left_elbow", "right_elbow"])
    parser.add_argument("--mobile_base", action="store_true",
                        help="Enable SE(2) base placement")
    parser.add_argument("--base_alignment", choices=["manual", "mocap"], default="manual")
    parser.add_argument("--max_fr", type=int, default=60, help="Control rate cap (Hz)")
    parser.add_argument("--geometry_config", choices=["sf_tapered_capsule", "sf_tapered_capsule_xhand"],
                        help="Licensed backend collision preset; xhand relaxes only torso/upper-arm pairs")
    parser.add_argument("--torso_upperarm_distances", type=float, nargs=3, metavar=("MIN", "ACT", "REL"),
                        default=None,
                        help="Licensed backend pair distances in metres (default: 0, 0.015, 0.025)")
    parser.add_argument("--torso_radius_scale", type=float, default=None,
                        help="Torso proxy radius multiplier (default: 1.0 on the tuned proxy)")
    parser.add_argument("--no_safety_filter", action="store_true")
    parser.add_argument("--elbow_filter_hz", type=float, default=None)
    parser.add_argument("--keep_torso_yaw", action="store_true",
                        help="Keep the solved torso joint 5 (default: zeroed, the "
                             "omni base owns that yaw)")
    parser.add_argument("--device", choices=["xrt", "mediapipe"], default="mediapipe")
    parser.add_argument("--backend", choices=["auto", "licensed", "reference", "mink"], default="auto")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--stale_after", type=float, default=0.5)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--input_diagnostics", action="store_true", help="Print XRT receive age and consumer rate every 5 seconds")
    parser.add_argument("--pose_model")
    parser.add_argument("--hand_model")
    parser.add_argument("--output_dir", default="recordings")
    parser.add_argument("--xhand_vendor_path", help="Optional legacy XHand vendor checkout root")
    parser.add_argument("--record_data", action="store_true",
                        help="Record incoming bone CSV via the device")
    parser.add_argument("--no_human_overlay", action="store_true")
    parser.add_argument("--hw", action="store_true",
                        help="Stream solved goals to RB-Y1 hardware")
    parser.add_argument("--robot_address", default="192.168.30.1:50051")
    parser.add_argument("--power_device", default=".*")
    parser.add_argument("--servo", default=".*")
    parser.add_argument("--hw_reset_ready", action="store_true",
                        help="Move hardware to the ready pose before teleop")
    parser.add_argument("--hw_impedance", action="store_true",
                        help="Use joint impedance commands (default: position)")
    parser.add_argument("--hand_serial_left", default=None)
    parser.add_argument("--hand_serial_right", default=None)
    args = parser.parse_args()
    if args.max_fr <= 0 or args.stale_after <= 0:
        parser.error("max_fr and stale_after must be positive")
    if args.device == "mediapipe":
        if args.record_data or args.mobile_base:
            parser.error("MediaPipe currently supports fixed-base input without CSV recording")
    return args


def main():
    args = parse_args()
    from contextlib import ExitStack
    with ExitStack() as resources:
        run(args, resources)


def run(args, resources):
    if args.hw and args.mobile_base:
        raise NotImplementedError(
            "--hw currently keeps the wheel base fixed; omit --mobile_base. "
            "The monolith RobotMwithBase remains required for odometry-feedback base motion.")

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
        backend=args.backend,
        model_xml=xml,
        hand=hand,
        control_rate_hz=float(args.max_fr),
        elbow_filter_cutoff_hz=args.elbow_filter_hz,
        collision_avoidance=not args.no_safety_filter,
        retarget_mode=args.retarget_mode,
        base_alignment_mode=args.base_alignment,
        mobile_base=args.mobile_base,
        spec_dir=SPECS_DIR,
        **({"geometry_config": args.geometry_config} if getattr(args, "geometry_config", None) else {}),
        **({"torso_upperarm_distances": tuple(args.torso_upperarm_distances)}
           if getattr(args, "torso_upperarm_distances", None) else {}),
        **({"torso_radius_scale": args.torso_radius_scale}
           if getattr(args, "torso_radius_scale", None) is not None else {}),
    )
    print(f"Solver backend: {type(session).__module__}.{type(session).__name__}")

    controller = controller_cls(model, data, debug=False)
    controller.setup_mocap_body("base_mocap_mover")

    if args.device == "xrt":
        device = XRDeviceAdapter(host=args.host, port=args.port, stale_after=args.stale_after,
                                 record_data=args.record_data, output_dir=args.output_dir,
                                 diagnostics=args.input_diagnostics)
    else:
        device = MediaPipeDeviceAdapter(camera_id=args.camera_id, pose_model=args.pose_model,
                                       hand_model=args.hand_model, display=True,
                                       stale_after=args.stale_after)
    resources.callback(device.cleanup)

    hardware = hands = None
    if args.hw:
        from rby1_teleop.control.hw import RobotMwithBase, XHandPair

        hardware = RobotMwithBase(
            address=args.robot_address, servo=args.servo,
            power_device=args.power_device, controller=None,
            position_control=not args.hw_impedance,
        )
        resources.callback(hardware.shutdown)
        if args.hw_reset_ready:
            hardware.reset_egoengine_ready_pose()
        state = hardware.get_hardware_state()
        controller.update_sim_from_hardware(state)
        if args.hand == "xhand" and (args.hand_serial_left or args.hand_serial_right):
            hands = XHandPair(
                args.xhand_vendor_path,
                left_serial=args.hand_serial_left,
                right_serial=args.hand_serial_right,
            )
            resources.callback(hands.shutdown)
        print(f"Hardware connected: {args.robot_address} (mobile base fixed)")
    session.reset(controller.q_current_right, controller.q_current_left)

    print(f"Waiting for {args.device} tracking...")

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
            frame = None
            try:
                frame = device.get_frame()
                if frame is not None:
                    if hardware is not None:
                        controller.update_sim_from_hardware(hardware.get_hardware_state())
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

                    if args.device == "mediapipe":
                        # Camera input lacks global torso/base/head tracking.
                        out.q_goal_torso = None
                        out.q_goal_head = None
                        out.p_world_base = None
                        out.R_world_base = None

                    if not args.keep_torso_yaw and out.q_goal_torso is not None:
                        out.q_goal_torso = np.asarray(out.q_goal_torso, dtype=float).copy()
                        out.q_goal_torso[-1] = 0.0
                    if out.p_world_base is not None and out.R_world_base is not None:
                        controller.update_mocap_body(out.p_world_base, out.R_world_base)
                    controller.set_joint_goals(out)
                    if hardware is not None:
                        hardware.send_goals(out, zero_torso_yaw=not args.keep_torso_yaw,
                                            min_time=max(frame_interval, 0.05))
                    if hands is not None:
                        hands.send(out)
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

    print("Teleoperation ended.")


if __name__ == "__main__":
    main()
