def test_import_wpt_tools():
    try:
        import wpt_tools
    except ImportError as e:
        raise AssertionError(f"Importing wpt_tools failed: {e}")