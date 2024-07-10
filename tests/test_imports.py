import pytest

def test_import_wpt_tools():
    """
    Test that the wpt_tools module can be imported successfully.
    """
    try:
        import wpt_tools
    except ImportError:
        pytest.fail("Failed to import wpt_tools")