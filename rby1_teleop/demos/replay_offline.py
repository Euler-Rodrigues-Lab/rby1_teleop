# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""RBY1 offline replay demo: recorded human motion -> retargeting -> MuJoCo.

Port of the monolith demo_rby1_xr_robot_teleop_v3_offline.py (and the xhand
variant) onto the geo_kin_core interface: solver =
``resolve_session(robot='rby1', hand=...)`` (licensed `geo_kin` wheel ->
private geo_kin_ref -> public fallback), controller =
:class:`rby1_teleop.control.RBY1WithXHandMuJoCoController` (xhand) or
:class:`rby1_teleop.control.RBY1MuJoCoController` (parallel gripper),
overlays = :mod:`geo_kin_core.viz`.

Two motion sources, same interface:

* a **frame stream** (``--frames``, the default) — the vendored sample motion,
  device-neutral numpy, needs nothing but this repo;
* a **recorded CSV** (``--csv_file``) — needs a SEW-Geometric-Teleop checkout
  for the device stack (``--monolith_path`` / ``GEO_TELEOP_MONOLITH``); use
  ``rby1_teleop.scripts.transcode_recording`` to turn one into a frame stream.

No headset and no robot required — this is the loop for tuning the solver,
exercising the retarget modes (``--retarget_mode pose|tcp|left_elbow|
right_elbow``), eyeballing the XPBD self-collision filter, and measuring solve
time. Playback is kinematic (``update_kinematic``), so the robot follows goals
exactly, and the recording is sampled on a fixed 1/max_fr clock so a replay is
reproducible regardless of solve speed (--wall_clock restores wall-clock
sampling).

Kept from the monolith offline demo: torso joint 5 is zeroed by default (the
omni-directional base owns that yaw; pass --keep_torso_yaw to disable), and
base alignment defaults to 'manual' so the human-capsule overlay lands on the
robot.

Examples:
    # Vendored sample motion, xhand, viewer + overlays (no setup)
    python -m rby1_teleop.demos.replay_offline

    # Functional (tcp) retargeting on a recorded CSV, mobile base enabled
    python -m rby1_teleop.demos.replay_offline --retarget_mode tcp --mobile_base \
        --csv_file $GEO_TELEOP_MONOLITH/References/recordings/ipman_roll.csv

    # Headless timing/collision sweep over one pass, stats to npz
    python -m rby1_teleop.demos.replay_offline --headless --no-loop --log_stats stats.npz
"""

import argparse
import sys
import time
import traceback

import mujoco
import numpy as np

from geo_kin_core.session import resolve_session
from geo_kin_core.viz import HumanCapsuleViz, capsules, draw_filtered_sew

from rby1_teleop import SAMPLE_MOTION, XML_RBY1_MOCAP, XML_RBY1_XHAND
from rby1_teleop.control import RBY1MuJoCoController, RBY1WithXHandMuJoCoController
from rby1_teleop.input import open_motion_source


def count_self_contacts(model, data) -> int:
    """Self-collision contacts in the current (kinematic) configuration.

    Contacts against static world geoms (floor) are ignored — the replayed
    robot is posed kinematically, so only body-vs-body contacts matter.
    """
    n = 0
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = model.geom_bodyid[con.geom1]
        b2 = model.geom_bodyid[con.geom2]
        if b1 != 0 and b2 != 0 and b1 != b2:
            n += 1
    return n


def parse_args():
    parser = argparse.ArgumentParser(description="RBY1 offline replay")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--frames", default=None,
                        help=f"geo_kin_core frame stream (.npz); default: the vendored "
                             f"sample motion ({SAMPLE_MOTION.name})")
    source.add_argument("--csv_file", default=None,
                        help="Recorded OpenXR body-pose CSV (needs a monolith checkout)")
    parser.add_argument("--hand", choices=["xhand", "none"], default="xhand",
                        help="Hand embodiment (selects sim model + controller + hand IK)")
    parser.add_argument("--retarget_mode", default="tcp",
                        choices=["pose", "tcp", "left_elbow", "right_elbow"],
                        help="Retargeting mode ('tcp' = functional retargeting, the "
                             "monolith offline demo default)")
    parser.add_argument("--mobile_base", action="store_true",
                        help="Enable SE(2) base placement (spring-damper follows the human)")
    parser.add_argument("--playback_speed", type=float, default=1.0,
                        help="Playback speed multiplier (1.0 = real time)")
    parser.add_argument("--no-loop", dest="loop", action="store_false",
                        help="Stop at the end of the recording instead of looping")
    parser.add_argument("--max_fr", type=int, default=60,
                        help="Solve/step rate cap in Hz")
    parser.add_argument("--no_safety_filter", action="store_true",
                        help="Disable the XPBD self-collision SEW filter")
    parser.add_argument("--base_alignment", choices=["manual", "mocap"], default="manual",
                        help="Base alignment mode. 'manual' (default) anchors the capture "
                             "frame to the robot, so the human overlay lands on it; 'mocap' "
                             "leaves the human at raw capture coordinates")
    parser.add_argument("--keep_torso_yaw", action="store_true",
                        help="Keep the solved torso joint 5 (the monolith offline demo "
                             "zeroes it: the omni base owns that yaw)")
    parser.add_argument("--elbow_filter_hz", type=float, default=None,
                        help="Stereographic elbow-angle low-pass cutoff (Hz); default off "
                             "(monolith default)")
    parser.add_argument("--monolith_path", default=None,
                        help="SEW-Geometric-Teleop checkout (else GEO_TELEOP_MONOLITH)")
    parser.add_argument("--headless", action="store_true",
                        help="No viewer (timing/collision sweeps, CI)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Stop after this many solved frames")
    parser.add_argument("--log_stats", default=None,
                        help="Write per-frame solve time + contact count to this .npz")
    parser.add_argument("--no_human_overlay", action="store_true",
                        help="Skip the human-skeleton capsule overlay")
    parser.add_argument("--wall_clock", action="store_true",
                        help="Sample the recording by wall-clock time instead of a "
                             "fixed 1/max_fr step (non-reproducible if solving lags)")
    return parser.parse_args()


def build(args):
    """Model, data, controller, session, and playback source."""
    if args.hand == "xhand":
        xml, controller_cls, hand = XML_RBY1_XHAND, RBY1WithXHandMuJoCoController, "xhand"
    else:
        xml, controller_cls, hand = XML_RBY1_MOCAP, RBY1MuJoCoController, None
    print(f"Loading model from: {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    frames = args.frames
    if frames is None and args.csv_file is None:
        frames = SAMPLE_MOTION  # vendored sample: runs on a clean checkout
    source = open_motion_source(
        frames=frames,
        csv_file=args.csv_file,
        playback_speed=args.playback_speed,
        loop=args.loop,
        monolith_path=args.monolith_path,
    )
    print(f"Motion source: {source.describe()}")
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
    return model, data, controller, session, source


class _Solver:
    """Wrap session.solve, passing hand-q kwargs only to backends that take them."""

    def __init__(self, session, with_hands: bool):
        self.session = session
        self.with_hands = with_hands

    def __call__(self, frame, controller):
        kwargs = dict(
            q_current_right=controller.q_current_right,
            q_current_left=controller.q_current_left,
        )
        if self.with_hands:
            try:
                return self.session.solve(
                    frame,
                    q_current_right_hand=getattr(controller, "q_current_right_hand", None),
                    q_current_left_hand=getattr(controller, "q_current_left_hand", None),
                    **kwargs)
            except TypeError:
                self.with_hands = False  # backend without hand-q kwargs
        return self.session.solve(frame, **kwargs)


def step(model, data, controller, solver, source, elapsed, zero_torso_yaw=True):
    """Solve + apply one replay frame. Returns ``(frame, solve_seconds)``."""
    frame = source.frame_at_time(elapsed)
    solve_time = 0.0
    if frame is not None:
        t0 = time.perf_counter()
        out = solver(frame, controller)
        solve_time = time.perf_counter() - t0

        if zero_torso_yaw and out.q_goal_torso is not None:
            # Torso joint 5 is owned by the omni base (monolith offline demo).
            out.q_goal_torso = np.asarray(out.q_goal_torso, dtype=float).copy()
            out.q_goal_torso[-1] = 0.0

        if out.p_world_base is not None and out.R_world_base is not None:
            controller.update_mocap_body(out.p_world_base, out.R_world_base)
        controller.set_joint_goals(out)

    controller.update_kinematic()
    return frame, solve_time


def main():
    args = parse_args()
    try:
        model, data, controller, session, source = build(args)
    except Exception as e:
        print(f"Error initializing replay: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"Recording duration: {source.duration:.2f}s "
          f"(playback speed {args.playback_speed}x, loop={args.loop})")
    model.opt.timestep = 0.005
    solver = _Solver(session, with_hands=(args.hand == "xhand"))
    solve_times, contacts = [], []
    frame_interval = 1.0 / args.max_fr

    def run(viewer=None):
        overlay = None
        if viewer is not None and not args.no_human_overlay:
            overlay = HumanCapsuleViz(viewer)
        start = time.time()
        sim_time = 0.0  # deterministic playback clock (see --wall_clock)
        n = 0
        finished = False
        while viewer is None or viewer.is_running():
            loop_start = time.time()
            elapsed = (loop_start - start) if args.wall_clock else sim_time
            sim_time += frame_interval * args.playback_speed
            try:
                frame, solve_time = step(model, data, controller, solver, source,
                                         elapsed, zero_torso_yaw=not args.keep_torso_yaw)
            except Exception as e:
                print(f"Error processing frame: {e}")
                traceback.print_exc()
                break

            if frame is None:
                if not args.loop and not finished:
                    print("Playback finished.")
                    finished = True
                    if viewer is None:
                        break
            else:
                n += 1
                solve_times.append(solve_time)
                contacts.append(count_self_contacts(model, data))

            if viewer is not None:
                capsules.clear(viewer)
                draw_filtered_sew(viewer, session, to_world=controller.get_sew_transform())
                if overlay is not None and frame is not None:
                    R_mocap = getattr(session, "R_mocap_world", None)
                    p_mocap = getattr(session, "p_mocap_world", None)
                    if R_mocap is not None and p_mocap is not None:
                        overlay.set_base_offset(p_mocap, np.asarray(R_mocap).T)
                    overlay.draw(frame)
                viewer.sync()

            if args.max_frames is not None and n >= args.max_frames:
                break
            lag = time.time() - loop_start
            if lag < frame_interval:
                time.sleep(frame_interval - lag)
        return n

    if args.headless:
        n = run(None)
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False,
        ) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -15
            viewer.cam.lookat[:] = [0, 0, 1.0]
            n = run(viewer)

    if solve_times:
        ms = np.asarray(solve_times) * 1e3
        print(f"Replayed {n} frames | solve {ms.mean():.2f} ms mean, "
              f"{ms.max():.2f} ms max | self-collision frames: "
              f"{int(np.count_nonzero(contacts))}/{len(contacts)}")
    if args.log_stats:
        np.savez(args.log_stats,
                 solve_time_s=np.asarray(solve_times),
                 self_contacts=np.asarray(contacts))
        print(f"Wrote stats to {args.log_stats}")


if __name__ == "__main__":
    main()
    print("Demo completed.")
