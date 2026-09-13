# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Lazy XHand serial adapter using the established TeleVision vendor class."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

XHAND_Q_LOWER = np.array([0.0, -0.698, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
XHAND_Q_UPPER = np.array([1.832, 1.745, 1.745, 0.6, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919])


def _controller_class(vendor_path):
    vendor_path = vendor_path or os.environ.get("XHAND_VENDOR_PATH")
    if vendor_path:
        root = Path(vendor_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"XHand vendor root not found: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        from TeleVision.xhand.xhand_2f_gripper import XHandController
    except ImportError as exc:
        raise ImportError(
            "XHand hardware needs the independently supplied TeleVision vendor class. "
            "Install it or pass --xhand_vendor_path / XHAND_VENDOR_PATH pointing "
            "to the directory containing TeleVision/."
        ) from exc
    return XHandController


class XHandPair:
    """Optional left/right serial XHands with clipped 12-DOF commands."""

    def __init__(self, vendor_path=None, *, left_serial=None, right_serial=None, controller_cls=None):
        cls = controller_cls or _controller_class(vendor_path)
        self.left = cls(comport=left_serial) if left_serial else None
        self.right = cls(comport=right_serial) if right_serial else None

    @staticmethod
    def _send(controller, q):
        if controller is None or q is None:
            return
        q = np.asarray(q, dtype=float).reshape(12)
        controller.grasp(np.clip(q, XHAND_Q_LOWER, XHAND_Q_UPPER).tolist())

    def send(self, goals):
        getter = goals.get if isinstance(goals, dict) else lambda key, default=None: getattr(goals, key, default)
        self._send(self.left, getter("q_goal_left_hand", None))
        self._send(self.right, getter("q_goal_right_hand", None))

    def shutdown(self):
        for hand in (self.left, self.right):
            if hand is not None:
                for method in ("close", "disconnect", "stop"):
                    if hasattr(hand, method):
                        try:
                            getattr(hand, method)()
                        except Exception:
                            pass
                        break

    close = shutdown
