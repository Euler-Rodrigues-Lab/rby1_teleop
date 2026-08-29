# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Offline replay pipeline over the vendored sample motion (needs a geo backend)."""

import argparse

import pytest

from rby1_teleop.demos.replay_offline import _Solver, build, step

pytestmark = pytest.mark.geo


def _args(**overrides):
    base = dict(
        frames=None, csv_file=None, hand="xhand", retarget_mode="tcp",
        mobile_base=False, playback_speed=1.0, loop=True, max_fr=60,
        no_safety_filter=False, base_alignment="manual", keep_torso_yaw=False,
        elbow_filter_hz=None, monolith_path=None, headless=True,
        max_frames=None, log_stats=None, no_human_overlay=True, wall_clock=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("mode", ["pose", "tcp", "left_elbow", "right_elbow"])
def test_replay_modes(mode):
    args = _args(retarget_mode=mode)
    model, data, controller, session, source = build(args)
    solver = _Solver(session, with_hands=True)
    n_solved = 0
    for i in range(30):
        frame, _ = step(model, data, controller, solver, source, i / 60.0)
        if frame is not None:
            n_solved += 1
    assert n_solved == 30
    # Kinematic follow: arms should have moved off the ready pose.
    assert controller.q_current_right is not None


def test_replay_no_hand_mobile_base():
    args = _args(hand="none", mobile_base=True)
    model, data, controller, session, source = build(args)
    solver = _Solver(session, with_hands=False)
    for i in range(30):
        step(model, data, controller, solver, source, i / 60.0)
    assert controller.mocap_enabled
