"""Device integration and unlicensed simulation without a private checkout."""
import sys
import numpy as np
import pytest


@pytest.mark.parametrize("demo", ["teleop_xr", "replay_offline"])
def test_experimental_torso_flags(monkeypatch, demo):
    import importlib
    module = importlib.import_module(f"rby1_teleop.demos.{demo}")
    monkeypatch.setattr(sys, "argv", [demo, "--backend", "licensed",
        "--torso_upperarm_distances", "0", "0.015", "0.025",
        "--torso_radius_scale", "0.99"])
    args = module.parse_args()
    assert args.torso_upperarm_distances == [0, .015, .025]
    assert args.torso_radius_scale == .99


def test_shared_adapter_identity():
    from rby1_teleop.input import XRDeviceAdapter, MediaPipeDeviceAdapter, OfflineCSVAdapter
    from xrt_devices.integrations import geo_kin
    assert XRDeviceAdapter is geo_kin.XRDeviceAdapter
    assert MediaPipeDeviceAdapter is geo_kin.MediaPipeDeviceAdapter
    assert OfflineCSVAdapter is geo_kin.OfflineCSVAdapter


def test_public_replay_fallback(monkeypatch):
    from rby1_teleop.demos.replay_offline import parse_args, build, step, _Solver
    monkeypatch.setitem(sys.modules, "geo_kin", None)
    monkeypatch.setitem(sys.modules, "geo_kin_ref", None)
    monkeypatch.setitem(sys.modules, "projects", None)
    monkeypatch.setattr(sys, "argv", ["replay", "--headless", "--backend", "mink"])
    model, data, controller, session, source = build(parse_args())
    solver = _Solver(session, with_hands=True)
    for i in range(2):
        step(model, data, controller, solver, source, i / 60)
    assert np.isfinite(data.qpos).all()


def test_live_camera_arguments(monkeypatch):
    from rby1_teleop.demos.teleop_xr import parse_args
    monkeypatch.setattr(sys, "argv", ["live", "--device", "mediapipe", "--backend", "mink"])
    args = parse_args()
    assert args.device == "mediapipe" and not args.hw and not args.mobile_base
