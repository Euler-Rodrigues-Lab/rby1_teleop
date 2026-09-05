# rby1_teleop

Teleoperation for the Rainbow Robotics **RB-Y1** (dual 7-DOF arms, 6-DOF torso,
2-DOF head, SE(2) omni base) with the **XHand** (12-DOF per hand), built on the
closed-form SEW geometric retargeting stack. The solver is obtained through the
`geo_kin_core.session.RetargetingSolver` interface — `resolve_session(robot="rby1",
hand=...)` picks the licensed `geo_kin` wheel, the private reference, or the
public fallback, whichever is available — and everything around it lives here:
MuJoCo models and controllers, device adapters, demos, and hardware glue.

Retargeting scope: both arms + wrist orientation, the full 6-DOF torso, the
2-DOF head, SE(2) mobile-base placement (spring-damper follow + stability
clamp), XPBD self-collision filtering, and per-finger XHand IK. Retarget modes:
`pose` (raw SEW), `tcp` (functional retargeting — bimanual TCP alignment),
`left_elbow` / `right_elbow` (one-side elbow-rigid + TCP-preserving FR on the
other arm). The hand is a config choice, not a separate codebase; Sharpa
support lands next through the same session `hand=` switch.

## Install

Clone with the pinned public-core submodule, then sync normally:

```bash
git clone --recurse-submodules https://github.com/Euler-Rodrigues-Lab/rby1_teleop.git
cd rby1_teleop
uv sync
```

For an existing checkout:

```bash
git pull
git submodule update --init --recursive
uv sync
```

The sync installs `external/geo_kin_core`, including the public fallback and
the `geo-kin-provision` command. To use WARP, TCP/C-SEW, or another licensed
mode, register the supplied RBY1/XHand wheel and license once per user, then
link the central build here:

```bash
uv run geo-kin-provision register \
  --product rby1-xhand \
  --wheel /path/to/geo_kin-0.1.0-cp310-abi3-manylinux_2_35_x86_64.whl \
  --license /path/to/geo_kin_license.toml \
  --name my-rby1-license \
  --activate
uv run geo-kin-provision install
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

## Known external dependencies (interim)

The live XR device and the CSV reader still live in the SEW-Geometric-Teleop
monolith: point `--monolith_path` or `GEO_TELEOP_MONOLITH` at a checkout for
`teleop_xr`, `--csv_file` replay, and `scripts/transcode_recording`. The
`--hw` hardware path (rby1_sdk + XHand serial) is being ported into
`rby1_teleop/control/hw` — until then hardware runs use the monolith's
`demo_rby1_xr_robot_teleop_v9_hw_all.py`.

## Licensing

MIT licensed. The SEW retargeting solver itself is patented & licensed
separately (`geo_kin` wheel); this repo runs against the public fallback out
of the box.

## Citation

If you use the retargeting solver, please cite the paper using the format published on the
[project website](https://sew-mimic.com/):
