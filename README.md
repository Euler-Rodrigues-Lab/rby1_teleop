# rby1_teleop

Teleoperation for the Rainbow Robotics **RB-Y1** (dual 7-DOF arms, 6-DOF torso,
2-DOF head, SE(2) omni base) with the **XHand** (12-DOF per hand), built on the
closed-form SEW geometric retargeting stack. The solver is obtained through the
`geo_kin_core.session.RetargetingSolver` interface — `resolve_session(robot="rby1",
hand=...)` picks the licensed `geo_kin` wheel, the private reference, or the
public fallback, whichever is available — and everything around it lives here:
MuJoCo models and controllers, device adapters, demos, and hardware glue.

- **Project Web & Demos**: [https://sew-mimic.com/](https://sew-mimic.com/)
- **Paper**: [A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot Teleoperation (arXiv:2602.01632)](https://arxiv.org/abs/2602.01632)
- **Input Devices & Teleop Server**: [XRT_devices](https://github.com/Euler-Rodrigues-Lab/XRT_devices) | [XR-Robot-Teleop System](https://xr-robot-teleop-website.pages.dev/)
- **WARP Embodiment**: [WARP Project](https://warp-retargeting.github.io)

Retargeting scope: both arms + wrist orientation, the full 6-DOF torso, the
2-DOF head, SE(2) mobile-base placement (spring-damper follow + stability
clamp), XPBD self-collision filtering, and per-finger XHand IK. Retarget modes:
`pose` (raw SEW), `tcp` (functional retargeting — bimanual TCP alignment),
`left_elbow` / `right_elbow` (one-side elbow-rigid + TCP-preserving FR on the
other arm). The hand is a config choice, not a separate codebase; Sharpa
support lands next through the same session `hand=` switch.

## Install

Clone with submodules (or run `git submodule update --init --recursive` in an existing checkout):

```bash
git clone --recurse-submodules https://github.com/Euler-Rodrigues-Lab/rby1_teleop.git
cd rby1_teleop
uv venv
uv pip install -e 'external/geo_kin_core[fallback]' -e 'external/XRT_devices[xr,recording]' -e . pytest
source .venv/bin/activate
```

The install includes `external/geo_kin_core`, the public fallback and
the `geo-kin-provision` command. To use WARP, TCP/C-SEW, or another licensed
mode, register the supplied RBY1/XHand wheel and license once per user, then
link the central build here:

```bash
geo-kin-provision register \
  --product rby1-xhand \
  --wheel /path/to/geo_kin-0.1.0-cp310-abi3-manylinux_2_35_x86_64.whl \
  --license /path/to/geo_kin_license.toml \
  --name my-rby1-license \
  --activate
geo-kin-provision install
```

The private binary is stored once outside every checkout and remains linked
across normal `uv sync` and `uv run` operations.

This robot is the reference embodiment of
[WARP](https://warp-retargeting.github.io) (Whole-Body Retargeting for Learning
from Offline Human Demonstrations); the WARP experiment/policy pipelines live
in the WARP repo and consume this one as their teleoperation base.

## Layout

```
rby1_teleop/
├── assets/            # models + data
│   ├── specs/         # geo_kin_core spec npz (URDF/MJCF -> R,p,h,limits; signed)
│   ├── urdf/          # spec sources (byte-locked by the npz signatures)
│   ├── rby1m/         # MuJoCo model + the single shared mesh tree
│   ├── rby1_with_xhand/  # XHand-equipped MuJoCo model (+ hand kinematic MJCFs)
│   └── sample_motion/ # vendored device-neutral recording (frame stream npz)
├── control/           # RBY1MuJoCoController (+ XHand variant); hw/ = hardware glue
├── input/             # XR device + offline CSV adapters, frame-stream playback
├── demos/             # replay_offline.py, teleop_xr.py
└── scripts/           # transcode_recording.py (CSV -> frame stream)
```

## Try it without a headset or a robot

```bash
pip install -e .
python -m rby1_teleop.demos.replay_offline                      # viewer, xhand, tcp mode
python -m rby1_teleop.demos.replay_offline --retarget_mode pose --mobile_base
python -m rby1_teleop.demos.replay_offline --headless --no-loop --log_stats stats.npz
```

The offline demo replays the vendored sample motion through the solver and a
kinematic MuJoCo follow, drawing the filtered SEW capsules and the human
skeleton overlay. Live teleop: `python -m rby1_teleop.demos.teleop_xr`
(Meta Quest over WebRTC; see below).

## Test Quest or webcam in simulation

```bash
# Quest: enter this computer's IP and port 8080 in XRT-Client.
python -m rby1_teleop.demos.teleop_xr --device xrt --backend auto

# Optional bone CSV recording:
python -m rby1_teleop.demos.teleop_xr --device xrt --record_data

# Webcam: default models download once, then are reused from the user cache.
uv pip install -e 'external/XRT_devices[mediapipe]'
python -m rby1_teleop.demos.teleop_xr --device mediapipe \
  --camera_id 0 --camera_display --backend auto
```

Simulation is the default. Use `--backend licensed` to require Rust, or
`--backend mink` for public differential IK. MINK does not reproduce analytic
TCP/C-SEW modes, finger retargeting or the SEW safety filter. MediaPipe currently
drives arms/hands; torso, head and base commands are suppressed for camera input.
Live webcam/headset validation remains pending. Close the viewer or press Ctrl-C
to release the device and hardware connections.

## Hardware (optional)

Simulation remains the default and does not import vendor SDKs. Install the
RB-Y1 hardware dependencies explicitly:

```bash
pip install -e '.[hw]'
```

The live demo follows the calling pattern of the monolith's
`demo_rby1_xr_robot_teleop_v9_hw_all.py` and keeps the wheel base fixed:

```bash
python -m rby1_teleop.demos.teleop_xr --hw \
    --robot_address 192.168.30.1:50051 \
    --hand_serial_right /dev/ttyUSB0
```

Add `--hw_reset_ready` only when the area is clear and an intentional move to
the ready pose is safe. Add `--hw_impedance` to use the impedance command
builder instead of joint-position commands. The XHand vendor class is not on
PyPI and is loaded lazily from `<monolith>/TeleVision`; omit both hand serial
arguments for body-only hardware.

Other projects can use the v9-compatible surface directly:

```python
from rby1_teleop.control.hw import RobotMwithBase

RBY1 = RobotMwithBase(
    address="192.168.30.1:50051",
    servo=".*",
    power_device=".*",
    controller=None,
)
RBY1.reset_egoengine_ready_pose()  # explicit: this moves the robot
state = RBY1.robot.get_state()     # same raw SDK access used by v9
RBY1.send_joint_command_arms_head_base(
    q_goal_left=q_left,
    q_goal_right=q_right,
    q_goal_torso=q_torso,
    q_goal_head=q_head,
    fix_base_pose=True,
    min_time=0.05,
)
RBY1.shutdown()
```

Package-local hardware control intentionally rejects nonzero wheel deltas.
The monolith's `rby1_api_utils.RobotMwithBase` remains necessary for mobile
base motion because its separate feedback loop consumes live wheel odometry.

## Known external dependencies (interim)

XR, MediaPipe, CSV replay and transcoding use the shared public [`XRT_devices`](https://github.com/Euler-Rodrigues-Lab/XRT_devices)
submodule under `external/XRT_devices`. They have no private-checkout dependency. The XHand serial vendor class
still needs its separately supplied legacy checkout via `--xhand_vendor_path`;
that source is absent from this workspace and has not been repackaged.
RB-Y1 body/head control is package-local under
`rby1_teleop/control/hw` and installed with the `hw` extra.

## Citation

If you use this retargeting stack, RB-Y1 teleoperation setup, or SEW solver in your research, please cite:

```bibtex
@article{kong2026closedform,
  title={A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot Teleoperation},
  author={Kong, Chuizheng and Cho, Yunho and Jung, Wonsuhk and Wibowo, Idris and Shinde, Parth and Vinodh-Sangeetha, Sundhar and Chung, Long Kiu and Chen, Zhenyang and others},
  journal={arXiv preprint arXiv:2602.01632},
  year={2026},
  url={https://arxiv.org/abs/2602.01632}
}
```

## Licensing

MIT licensed. The SEW retargeting solver itself is patented & licensed
separately (`geo_kin` wheel); this repo runs against the public fallback out
of the box.
