"""Tests for data classes."""

import numpy as np
import pytest
import skrf as rf

from wpt_tools.data_classes import (
    EfficiencyResults,
    LCRFittingResults,
    RichNetwork,
    ValR2,
)


class TestValR2:
    """Test ValR2 data class."""

    def test_init(self):
        """Test ValR2 initialization."""
        val = ValR2(value=1.0, r2=0.95)
        assert val.value == 1.0
        assert val.r2 == 0.95

    def test_none_values(self):
        """Test ValR2 with None values."""
        val = ValR2(value=None, r2=None)
        assert val.value is None
        assert val.r2 is None


class TestEfficiencyResults:
    """Test EfficiencyResults data class."""

    def test_init(self):
        """Test EfficiencyResults initialization."""
        results = EfficiencyResults()
        assert results.f_plot == []
        assert results.r_opt == []
        assert results.x_opt == []
        assert results.eff_opt == []
        assert results.max_f_plot is None
        assert results.max_eff_opt is None
        assert results.max_r_opt is None
        assert results.max_x_opt is None

    def test_validate_success(self):
        """Test successful validation."""
        results = EfficiencyResults()
        results.max_f_plot = 1e9
        results.max_eff_opt = 0.85
        results.max_r_opt = 50.0
        results.max_x_opt = 25.0

        # Should not raise
        results.validate()

    @pytest.mark.parametrize(
        "missing_field,expected_error",
        [
            ("max_f_plot", "max_f_plot is None"),
            ("max_eff_opt", "max_eff_opt is None"),
            ("max_r_opt", "max_r_opt is None"),
            ("max_x_opt", "max_x_opt is None"),
        ],
    )
    def test_validate_failures(self, missing_field, expected_error):
        """Test validation failures for missing fields."""
        results = EfficiencyResults()
        results.max_f_plot = 1e9
        results.max_eff_opt = 0.85
        results.max_r_opt = 50.0
        results.max_x_opt = 25.0

        # Remove the field being tested
        setattr(results, missing_field, None)

        with pytest.raises(ValueError, match=expected_error):
            results.validate()


class TestLCRFittingResults:
    """Test LCRFittingResults data class."""

    def test_init(self):
        """Test LCRFittingResults initialization."""
        results = LCRFittingResults()
        # LCRFittingResults initializes with ValR2 objects, not None
        assert results.ls1 is not None
        assert results.cs1 is not None
        assert results.rs1 is not None
        assert results.ls2 is not None
        assert results.cs2 is not None
        assert results.rs2 is not None
        assert results.lm is not None
        # km is calculated dynamically, not stored as attribute


class TestRichNetwork:
    """Test RichNetwork data class."""

    def test_from_touchstone_string(self):
        """Test creating RichNetwork from string path."""
        # Create a simple 2-port network for testing
        f = np.linspace(1e9, 2e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.1
        s[:, 1, 1] = 0.1
        s[:, 0, 1] = 0.9
        s[:, 1, 0] = 0.9

        nw = rf.Network(frequency=f, s=s)

        # Test from_touchstone with rf.Network
        rich_nw = RichNetwork.from_touchstone(nw)
        assert rich_nw.nw is nw
        assert rich_nw.target_f is None
        assert rich_nw.range_f is None

    def test_set_f_target_range(self):
        """Test set_f_target_range method."""
        # Create a simple 2-port network
        f = np.linspace(1e9, 2e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.1
        s[:, 1, 1] = 0.1
        s[:, 0, 1] = 0.9
        s[:, 1, 0] = 0.9

        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)

        target_f = 1.5e9
        range_f = 0.1e9

        rich_nw.set_f_target_range(target_f, range_f)

        assert rich_nw.target_f == target_f
        assert rich_nw.range_f == range_f
        assert rich_nw.target_f_index is not None
        assert rich_nw.f_narrow_index_start is not None
        assert rich_nw.f_narrow_index_stop is not None
        assert rich_nw.sweeppoint is not None

        # Check that indices are within bounds
        assert 0 <= rich_nw.target_f_index < len(f)
        assert 0 <= rich_nw.f_narrow_index_start < len(f)
        assert 0 <= rich_nw.f_narrow_index_stop <= len(f)
        assert rich_nw.f_narrow_index_start < rich_nw.f_narrow_index_stop
