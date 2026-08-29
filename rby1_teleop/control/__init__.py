# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Sim + hardware control for the RB-Y1.

MuJoCo controllers import cleanly everywhere; hardware modules under
``rby1_teleop.control.hw`` are imported lazily so a sim-only install (no
``rby1_sdk`` / vendor SDKs) works.
"""

from .mujoco_controller import (  # noqa: F401
    RBY1MuJoCoController,
    RBY1WithXHandMuJoCoController,
)
