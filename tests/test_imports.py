"""Basic import test for wpt_tools."""

import importlib

import pytest


def test_import_wpt_tools():
    """Ensure the package is importable without raising ImportError."""
    try:
        importlib.import_module("wpt_tools")
    except ImportError:
        pytest.fail("Failed to import wpt_tools")
