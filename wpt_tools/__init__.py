try:
    from importlib.metadata import version
    __version__ = version("wpt_tools")
except ImportError:
    __version__ = "unknown"
