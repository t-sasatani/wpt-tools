"""Tests for loading and configuring touchstone networks."""

import os

import numpy as np
import pytest
import skrf as rf

from wpt_tools.analysis import nw_with_config


@pytest.fixture
def sample_touchstone_file(tmp_path):
    """
    Fixture to provide a temporary copy of a sample touchstone file.

    The sample touchstone file is copied from a predefined 'data' directory
    to a temporary directory for isolated testing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory provided by pytest.

    Returns
    -------
    str
        Path to the temporary touchstone file.

    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    sample_touchstone_src = os.path.join(data_dir, "sample.s2p")

    # Create a temporary copy of the sample file
    sample_touchstone_dst = tmp_path / "sample.s2p"
    sample_touchstone_dst.write_bytes(open(sample_touchstone_src, "rb").read())

    return str(sample_touchstone_dst)


def test_import_touchstone(sample_touchstone_file):
    """
    Test loading a touchstone and creating config via nw_with_config.
    """
    cfg = nw_with_config.import_touchstone(sample_touchstone_file)
    assert isinstance(cfg.nw, rf.Network)
    assert cfg.sweeppoint == np.size(cfg.nw.frequency.f)


def test_set_f_target_range(sample_touchstone_file):
    """
    Test the set_f_target_range method of the nw_tools class.

    This function tests if the target frequency range is correctly set and
    the corresponding indices in the frequency sweep points are accurately identified.

    Parameters
    ----------
    sample_touchstone_file : str
        Path to the temporary touchstone file provided by the fixture.

    """
    cfg = nw_with_config.import_touchstone(sample_touchstone_file)

    target_f = 6.78e6
    range_f = 100e3

    cfg.set_f_target_range(target_f, range_f)

    assert cfg.target_f == target_f
    assert cfg.range_f == range_f
    assert isinstance(cfg.f_narrow_index_start, int)
    assert isinstance(cfg.f_narrow_index_stop, int)
    assert isinstance(cfg.target_f_index, int)

    # Further checks to ensure the correct calculation of indices
    f_narrow_start_f = cfg.nw.frequency.f[cfg.f_narrow_index_start]
    f_narrow_stop_f = cfg.nw.frequency.f[cfg.f_narrow_index_stop]
    f_target_index_f = cfg.nw.frequency.f[cfg.target_f_index]

    assert abs(target_f - f_target_index_f) < range_f / 2
    assert f_narrow_start_f <= target_f <= f_narrow_stop_f

    # Ensure the narrow indices span the range properly
    assert cfg.f_narrow_index_start <= cfg.f_narrow_index_stop
    assert cfg.f_narrow_index_start >= 0
    assert cfg.f_narrow_index_stop < cfg.sweeppoint
