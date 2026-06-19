"""Tests for edge cases and error conditions."""

import numpy as np
import pytest
import skrf as rf

from wpt_tools.analysis import nw_tools
from wpt_tools.data_classes import EfficiencyResults, RichNetwork
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
        rich_nw.set_f_target_range(13.56e6, 1e6)

        # This should work even when target_f does not match the only point.
        rich_nw.set_f_target_range(13.56e6, 0.1e6)

        assert rich_nw.target_f == 13.56e6
        assert rich_nw.range_f == 0.1e6
        assert rich_nw.target_f_index == 0
        assert rich_nw.f_narrow_index_start == 0
        assert rich_nw.f_narrow_index_stop == 1

    def test_single_frequency_skips_efficiency_plot(self, monkeypatch):
        """Single-frequency data should skip frequency plotting gracefully."""
        f = np.array([1e9])
        s = np.zeros((1, 2, 2), dtype=complex)
        s[0, 0, 0] = 0.1
        s[0, 1, 1] = 0.1
        s[0, 0, 1] = 0.9
        s[0, 1, 0] = 0.9

        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)

        called = {"plot": False}

        def _stub_plot_efficiency(*args, **kwargs):
            called["plot"] = True

        monkeypatch.setattr("wpt_tools.analysis.plot_efficiency", _stub_plot_efficiency)

        results = nw_tools.analyze_efficiency(
            rich_nw=rich_nw,
            rx_port=2,
            show_plot=True,
            show_data=False,
            target_f=13.56e6,
            range_f=1e6,
        )

        # Plotting is skipped, but the single point must still be analyzed even
        # though the requested target_f does not match the only frequency.
        assert called["plot"] is False
        assert results.max_eff_opt is not None
        assert results.f_plot == [1e9]

    def test_single_frequency_runs_without_target_range(self):
        """Single-frequency analysis should work with no target_f/range_f."""
        f = np.array([1e9])
        s = np.zeros((1, 2, 2), dtype=complex)
        s[0, 0, 0] = 0.1
        s[0, 1, 1] = 0.1
        s[0, 0, 1] = 0.9
        s[0, 1, 0] = 0.9

        nw = rf.Network(frequency=f, s=s)
        rich_nw = RichNetwork.from_touchstone(nw)

        results = nw_tools.analyze_efficiency(
            rich_nw=rich_nw, rx_port=2, show_plot=False, show_data=False
        )

        assert results.max_eff_opt is not None
        assert results.f_plot == [1e9]

    def test_kq_inverts_efficiency_relation(self):
        """results.kq should invert the solver's gmax(kQ) relation."""
        kq_true = 5.0
        # Forward relation used by efficiency_calculator.
        gmax = kq_true**2 / (1.0 + np.sqrt(1.0 + kq_true**2)) ** 2

        results = EfficiencyResults()
        results.max_eff_opt = float(gmax)

        assert results.kq == pytest.approx(kq_true)

    def test_kq_undefined_for_nonpositive_efficiency(self):
        """kQ is not real when the peak efficiency is outside (0, 1)."""
        results = EfficiencyResults()
        results.max_eff_opt = -0.01
        with pytest.raises(ValueError):
            _ = results.kq

    def test_kq_raises_when_efficiency_missing(self):
        """kQ requires a computed peak efficiency."""
        results = EfficiencyResults()
        with pytest.raises(ValueError):
            _ = results.kq

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
