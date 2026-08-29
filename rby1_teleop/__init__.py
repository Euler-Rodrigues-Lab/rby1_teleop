# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""rby1_teleop — Rainbow Robotics RB-Y1 teleoperation on the geo_kin stack.

SEW geometric retargeting for the RB-Y1 (dual 7-DOF arms + 6-DOF torso +
2-DOF head + SE(2) wheeled base), with the XHand (12-DOF per hand) as a hand
option. The solver is resolved through ``geo_kin_core.session.resolve_session``
(licensed ``geo_kin`` wheel -> private ``geo_kin_ref`` -> public fallback);
this repo holds everything around it: MuJoCo models + controllers, device
adapters, demos, and hardware glue.
"""

from pathlib import Path

__version__ = "0.1.0"

PACKAGE_DIR = Path(__file__).parent
ASSETS_DIR = PACKAGE_DIR / "assets"
SPECS_DIR = ASSETS_DIR / "specs"
SAMPLE_MOTION_DIR = ASSETS_DIR / "sample_motion"

#: Vendored sample recording (device-neutral geo_kin_core frame stream).
SAMPLE_MOTION = SAMPLE_MOTION_DIR / "ipman_roll.npz"

# MuJoCo models (curated from the monolith; one shared mesh tree under
# assets/rby1m/assets, the xhand models reference it relatively).
XML_RBY1 = ASSETS_DIR / "rby1m" / "model_v1.3_act.xml"
XML_RBY1_MOCAP = ASSETS_DIR / "rby1m" / "model_v1.3_act_mocap.xml"
XML_RBY1_XHAND = ASSETS_DIR / "rby1_with_xhand" / "model_v1.3_xhand_act.xml"

# Hand kinematic models (spec-npz sources; byte-locked by the npz signatures).
XML_XHAND_LEFT = ASSETS_DIR / "rby1_with_xhand" / "xhand_left.xml"
XML_XHAND_RIGHT = ASSETS_DIR / "rby1_with_xhand" / "xhand_right.xml"

# URDFs (spec-npz sources; byte-locked by the npz signatures).
URDF_RBY1 = ASSETS_DIR / "urdf" / "model_v1.3_modified.urdf"
URDF_RBY1_XHAND = ASSETS_DIR / "urdf" / "rby1_v1.3_xhand.urdf"
