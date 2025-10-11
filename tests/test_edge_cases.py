"""Tests for edge cases and error conditions."""

import numpy as np
import pytest
import skrf as rf

from wpt_tools.data_classes import RichNetwork
from wpt_tools.solver import compute_load_sweep, efficiency_calculator


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_rich_network_empty_frequency_range(self):
        """Test RichNetwork with very small frequency range."""
        f = np.array([1e9, 1e9 + 1e6])  # Very small range
        s = np.zeros((2, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.1
        s[:, 1, 1] = 0.1
        s[:, 0, 1] = 0.9
        s[:, 1, 0] = 0.9

        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)

        # This should still work but with minimal range
        rich_nw.set_f_target_range(1e9, 0.1e6)

        assert rich_nw.target_f == 1e9
        assert rich_nw.range_f == 0.1e6
        assert rich_nw.f_narrow_index_start is not None
        assert rich_nw.f_narrow_index_stop is not None

    def test_rich_network_single_frequency(self):
        """Test RichNetwork with single frequency point."""
        f = np.array([1e9])
        s = np.zeros((1, 2, 2), dtype=complex)
        s[0, 0, 0] = 0.1
        s[0, 1, 1] = 0.1
        s[0, 0, 1] = 0.9
        s[0, 1, 0] = 0.9

        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)

        # This should work with single frequency
        rich_nw.set_f_target_range(1e9, 0.1e6)

        assert rich_nw.target_f == 1e9
        assert rich_nw.range_f == 0.1e6

    def test_efficiency_calculator_zero_efficiency(self):
        """Test efficiency calculator with network that gives zero efficiency."""
        # Create a network with very poor coupling
        f = np.linspace(1e9, 2e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.99  # High reflection
        s[:, 1, 1] = 0.99  # High reflection
        s[:, 0, 1] = 0.01  # Very low coupling
        s[:, 1, 0] = 0.01  # Very low coupling

        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)
        rich_nw.set_f_target_range(1.5e9, 0.1e9)

        results = efficiency_calculator(
            rich_nw, rx_port=1, target_f=1.5e9, range_f=0.1e9
        )

        # Should still return valid results, even if efficiency is low
        assert len(results.f_plot) > 0
        assert len(results.eff_opt) > 0
        assert all(
            eff >= 0 for eff in results.eff_opt
        )  # Efficiency should be non-negative
        assert all(eff <= 1 for eff in results.eff_opt)  # Efficiency should be <= 1

    def test_compute_load_sweep_extreme_values(self):
        """Test load sweep with extreme impedance values."""
        f = np.linspace(1e9, 2e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.1
        s[:, 1, 1] = 0.1
        s[:, 0, 1] = 0.9
        s[:, 1, 0] = 0.9

        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)
        rich_nw.set_f_target_range(1.5e9, 0.1e9)

        # Test with very small and very large impedance ranges
        results = compute_load_sweep(
            rich_nw,
            rez_min=1e-6,  # Very small
            rez_max=1e6,  # Very large
            rez_step=1e3,
            imz_min=-1e6,  # Very large negative
            imz_max=1e6,  # Very large positive
            imz_step=1e3,
            rx_port=1,
            input_voltage=1.0,
            target_f=1.5e9,
            range_f=0.1e9,
        )

        assert len(results.rez_list) > 0
        assert len(results.imz_list) > 0
        assert results.eff_grid.shape == (len(results.rez_list), len(results.imz_list))

    def test_compute_load_sweep_single_point(self):
        """Test load sweep with single impedance point."""
        f = np.linspace(1e9, 2e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.1
        s[:, 1, 1] = 0.1
        s[:, 0, 1] = 0.9
        s[:, 1, 0] = 0.9

        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)
        rich_nw.set_f_target_range(1.5e9, 0.1e9)

        # Test with single point (min == max)
        results = compute_load_sweep(
            rich_nw,
            rez_min=50.0,
            rez_max=50.0,  # Same as min
            rez_step=1.0,
            imz_min=0.0,
            imz_max=0.0,  # Same as min
            imz_step=1.0,
            rx_port=1,
            input_voltage=1.0,
            target_f=1.5e9,
            range_f=0.1e9,
        )

        # When min == max, numpy.arange returns empty array
        # This is expected behavior
        assert len(results.rez_list) == 0
        assert len(results.imz_list) == 0
        assert results.eff_grid.shape == (0, 0)

    def test_rich_network_target_frequency_out_of_range(self):
        """Test RichNetwork with target frequency outside the network range."""
        f = np.linspace(1e9, 2e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.1
        s[:, 1, 1] = 0.1
        s[:, 0, 1] = 0.9
        s[:, 1, 0] = 0.9

        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)

        # Target frequency outside range should raise an error
        with pytest.raises(
            ValueError, match="Target frequency not found within specified range"
        ):
            rich_nw.set_f_target_range(0.5e9, 0.1e9)  # Below range

    def test_rich_network_very_large_range(self):
        """Test RichNetwork with very large frequency range."""
        f = np.linspace(1e9, 2e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        s[:, 0, 0] = 0.1
        s[:, 1, 1] = 0.1
        s[:, 0, 1] = 0.9
        s[:, 1, 0] = 0.9

        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)

        # Very large range should use the entire frequency range
        rich_nw.set_f_target_range(1.5e9, 10e9)  # Much larger than network range

        assert rich_nw.target_f == 1.5e9
        assert rich_nw.range_f == 10e9
        assert rich_nw.f_narrow_index_start is not None
        assert rich_nw.f_narrow_index_stop is not None
        # Should use the entire available range (stop is exclusive)
        assert rich_nw.f_narrow_index_start == 0
        assert rich_nw.f_narrow_index_stop == len(f) - 1
