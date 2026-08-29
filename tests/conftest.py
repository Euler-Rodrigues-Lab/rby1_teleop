# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
import importlib.util

import pytest


def _geo_backend_available() -> bool:
    return (importlib.util.find_spec("geo_kin") is not None
            or importlib.util.find_spec("geo_kin_ref") is not None)


def pytest_collection_modifyitems(config, items):
    if _geo_backend_available():
        return
    skip = pytest.mark.skip(reason="no geo_kin wheel / geo_kin_ref available")
    for item in items:
        if "geo" in item.keywords:
            item.add_marker(skip)
