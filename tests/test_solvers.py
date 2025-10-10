"""Tests for solver functions."""

import pytest
import numpy as np
import skrf as rf
from wpt_tools.data_classes import RichNetwork
from wpt_tools.solvers import (
    series_lcr_xself,
    series_lcr_rself,
    series_lcr_xm,
    efficiency_calculator,
    compute_load_sweep,
    compute_rxc_filter,
    lcr_fitting,
)


class TestSeriesLCRFunctions:
    """Test series LCR model functions."""
    
    def test_series_lcr_xself(self):
        """Test series LCR reactance calculation."""
        f = np.array([1e9, 2e9])
        ls = 1e-6
        cs = 1e-9
        
        result = series_lcr_xself(f, ls, cs)
        expected = 2 * np.pi * f * ls - 1 / (2 * np.pi * f * cs)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_series_lcr_rself(self):
        """Test series LCR resistance calculation."""
        f = np.array([1e9, 2e9])
        r = 5.0
        
        result = series_lcr_rself(f, r)
        expected = np.zeros_like(f) + r
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_series_lcr_xm(self):
        """Test series LCR mutual reactance calculation."""
        f = np.array([1e9, 2e9])
        lm = 0.5e-6
        
        result = series_lcr_xm(f, lm)
        expected = 2 * np.pi * f * lm
        
        np.testing.assert_array_almost_equal(result, expected)


class TestEfficiencyCalculator:
    """Test efficiency calculator function."""
    
    def create_test_network(self):
        """Create a simple 2-port network for testing."""
        f = np.linspace(1e9, 2e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.1
        s[:, 1, 1] = 0.1
        s[:, 0, 1] = 0.9
        s[:, 1, 0] = 0.9
        
        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)
        rich_nw.set_f_target_range(1.5e9, 0.1e9)
        
        return rich_nw
    
    def test_efficiency_calculator_basic(self):
        """Test basic efficiency calculation."""
        rich_nw = self.create_test_network()
        
        results = efficiency_calculator(rich_nw, rx_port=1, target_f=1.5e9, range_f=0.1e9)
        
        from wpt_tools.data_classes import EfficiencyResults
        assert isinstance(results, EfficiencyResults)
        assert len(results.f_plot) > 0
        assert len(results.r_opt) > 0
        assert len(results.x_opt) > 0
        assert len(results.eff_opt) > 0
        assert results.max_f_plot is not None
        assert results.max_eff_opt is not None
        assert results.max_r_opt is not None
        assert results.max_x_opt is not None
    
    def test_efficiency_calculator_missing_target_f(self):
        """Test efficiency calculator with missing target frequency."""
        rich_nw = self.create_test_network()
        rich_nw.target_f = None
        
        with pytest.raises(ValueError, match="Target frequency is not set"):
            efficiency_calculator(rich_nw, rx_port=1, target_f=None, range_f=0.1e9)
    
    def test_efficiency_calculator_missing_range_f(self):
        """Test efficiency calculator with missing range frequency."""
        rich_nw = self.create_test_network()
        rich_nw.range_f = None
        
        with pytest.raises(ValueError, match="Range frequency is not set"):
            efficiency_calculator(rich_nw, rx_port=1, target_f=1.5e9, range_f=None)


class TestComputeLoadSweep:
    """Test compute_load_sweep function."""
    
    def create_test_network(self):
        """Create a simple 2-port network for testing."""
        f = np.linspace(1e9, 2e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.1
        s[:, 1, 1] = 0.1
        s[:, 0, 1] = 0.9
        s[:, 1, 0] = 0.9
        
        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)
        rich_nw.set_f_target_range(1.5e9, 0.1e9)
        
        return rich_nw
    
    def test_compute_load_sweep_basic(self):
        """Test basic load sweep computation."""
        rich_nw = self.create_test_network()
        
        results = compute_load_sweep(
            rich_nw,
            rez_min=0.1,
            rez_max=10.0,
            rez_step=1.0,
            imz_min=-10.0,
            imz_max=10.0,
            imz_step=2.0,
            rx_port=1,
            input_voltage=1.0,
            target_f=1.5e9,
            range_f=0.1e9,
        )
        
        assert hasattr(results, 'rez_list')
        assert hasattr(results, 'imz_list')
        assert hasattr(results, 'eff_grid')
        assert hasattr(results, 'Pin')
        assert hasattr(results, 'Pout')
        assert hasattr(results, 'target_f')
        
        assert len(results.rez_list) > 0
        assert len(results.imz_list) > 0
        assert results.eff_grid.shape == (len(results.rez_list), len(results.imz_list))
        assert results.Pin.shape == results.eff_grid.shape
        assert results.Pout.shape == results.eff_grid.shape
    
    def test_compute_load_sweep_invalid_rx_port(self):
        """Test compute_load_sweep with invalid rx_port."""
        rich_nw = self.create_test_network()
        
        with pytest.raises(ValueError, match="set rx_port to 1 or 2"):
            compute_load_sweep(
                rich_nw,
                rez_min=0.1,
                rez_max=10.0,
                rez_step=1.0,
                imz_min=-10.0,
                imz_max=10.0,
                imz_step=2.0,
                rx_port=3,  # Invalid port
                input_voltage=1.0,
                target_f=1.5e9,
                range_f=0.1e9,
            )


class TestComputeRXCFilter:
    """Test compute_rxc_filter function."""
    
    def create_test_network(self):
        """Create a simple 2-port network for testing."""
        f = np.linspace(1e9, 2e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.1
        s[:, 1, 1] = 0.1
        s[:, 0, 1] = 0.9
        s[:, 1, 0] = 0.9
        
        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)
        rich_nw.set_f_target_range(1.5e9, 0.1e9)
        
        return rich_nw
    
    def test_compute_rxc_filter_basic(self):
        """Test basic RXC filter computation."""
        rich_nw = self.create_test_network()
        
        results = compute_rxc_filter(
            rich_nw,
            rx_port=1,
            rload=100.0,
            c_network="CpCsRl",
            target_f=1.5e9,
            range_f=0.1e9,
        )
        
        assert hasattr(results, 'cp')
        assert hasattr(results, 'cs')
        assert hasattr(results, 'rload')
        assert hasattr(results, 'max_r_opt')
        assert hasattr(results, 'max_f_plot')
        assert hasattr(results, 'lrx')
        
        assert results.cp > 0
        assert results.cs > 0
        assert results.rload == 100.0


class TestLCRFitting:
    """Test LCR fitting function."""
    
    def create_test_network(self):
        """Create a simple 2-port network for testing."""
        f = np.linspace(1e9, 2e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.1
        s[:, 1, 1] = 0.1
        s[:, 0, 1] = 0.9
        s[:, 1, 0] = 0.9
        
        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)
        rich_nw.set_f_target_range(1.5e9, 0.1e9)
        
        return rich_nw
    
    def test_lcr_fitting_basic(self):
        """Test basic LCR fitting."""
        rich_nw = self.create_test_network()
        
        results = lcr_fitting(rich_nw, target_f=1.5e9, range_f=0.1e9)
        
        assert hasattr(results, 'ls1')
        assert hasattr(results, 'cs1')
        assert hasattr(results, 'rs1')
        
        # Check that fitted values are present (may be negative for poor fits)
        assert results.ls1.value is not None
        assert results.cs1.value is not None
        assert results.rs1.value is not None
        
        # Check R2 scores are present
        assert results.ls1.r2 is not None
        assert results.cs1.r2 is not None
        assert results.rs1.r2 is not None
