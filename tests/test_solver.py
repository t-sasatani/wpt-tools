"""Tests for solver functions."""

from pathlib import Path

import numpy as np
import pytest
import skrf as rf
from scipy.optimize import minimize

from wpt_tools.data_classes import RichNetwork
from wpt_tools.solver import (
    compute_load_sweep,
    compute_rxc_filter,
    efficiency_calculator,
    lcr_fitting,
    series_lcr_rself,
    series_lcr_xm,
    series_lcr_xself,
    solve_cpcs_closed_form,
)

SAMPLE_S2P = Path(__file__).parent / "data" / "sample.s2p"


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

        results = efficiency_calculator(
            rich_nw, rx_port=1, target_f=1.5e9, range_f=0.1e9
        )

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

    @staticmethod
    def _rm_network():
        """Single-frequency network with non-zero mutual resistance Rm.

        Measured 10 MHz sample (TX stretchable coil, RX 30x30 mm coil, air).
        """
        z11 = 4.18 + 38.49j
        z22 = 1.32 - 1.69j
        zm = 0.836 + 28.171j
        z = np.array([[[z11, zm], [zm, z22]]], dtype=complex)
        nw = rf.Network.from_z(z, frequency=rf.Frequency.from_f([10e6], unit="Hz"))
        return RichNetwork.from_touchstone(nw), z11, z22, zm

    @staticmethod
    def _efficiency(z11, z22, zm, zload):
        """Link efficiency of the two-port terminated with ``zload`` at port 2."""
        zin = z11 - zm**2 / (z22 + zload)
        return zload.real * abs(zm / (z22 + zload)) ** 2 / zin.real

    def test_x_opt_closed_form_with_mutual_resistance(self):
        """x_opt must be Rm*Xm/R11 - X22, not Rm*Xm - X22 (regression)."""
        rich_nw, z11, z22, zm = self._rm_network()

        results = efficiency_calculator(rich_nw, rx_port=2, target_f=None, range_f=None)

        r_det = z11.real * z22.real - zm.real**2
        expected_r = np.sqrt(r_det * (r_det + abs(zm) ** 2)) / z11.real
        expected_x = zm.real * zm.imag / z11.real - z22.imag

        assert results.max_r_opt == pytest.approx(expected_r, rel=1e-9)
        assert results.max_x_opt == pytest.approx(expected_x, rel=1e-9)
        assert results.max_x_opt == pytest.approx(7.3242, abs=1e-4)

    def test_x_opt_load_reaches_max_efficiency(self):
        """Efficiency at the reported optimum equals max_eff_opt and is a maximum."""
        rich_nw, z11, z22, zm = self._rm_network()

        results = efficiency_calculator(rich_nw, rx_port=2, target_f=None, range_f=None)
        zload_opt = results.max_r_opt + 1j * results.max_x_opt
        eff_opt = self._efficiency(z11, z22, zm, zload_opt)

        assert eff_opt == pytest.approx(results.max_eff_opt, abs=1e-9)

        for dr, dx in [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5), (0.3, -0.3)]:
            eff_perturbed = self._efficiency(z11, z22, zm, zload_opt + dr + 1j * dx)
            assert eff_perturbed < eff_opt

        # The pre-fix formula (Rm*Xm - X22) gave a visibly worse load.
        zload_buggy = results.max_r_opt + 1j * (zm.real * zm.imag - z22.imag)
        assert self._efficiency(z11, z22, zm, zload_buggy) < eff_opt - 0.05

    def test_x_opt_rx_port_swap(self):
        """rx_port=1 must apply the same formula with the ports swapped."""
        rich_nw, z11, z22, zm = self._rm_network()

        results = efficiency_calculator(rich_nw, rx_port=1, target_f=None, range_f=None)
        # Now the receiver is the original port 1, transmitter is port 2.
        expected_x = zm.real * zm.imag / z22.real - z11.imag

        assert results.max_x_opt == pytest.approx(expected_x, rel=1e-9)


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

        assert hasattr(results, "rez_list")
        assert hasattr(results, "imz_list")
        assert hasattr(results, "eff_grid")
        assert hasattr(results, "Pin")
        assert hasattr(results, "Pout")
        assert hasattr(results, "target_f")

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

        assert hasattr(results, "cp")
        assert hasattr(results, "cs")
        assert hasattr(results, "rload")
        assert hasattr(results, "max_r_opt")
        assert hasattr(results, "max_f_plot")
        assert hasattr(results, "lrx")

        assert results.cp > 0
        assert results.cs > 0
        assert results.rload == 100.0


class TestRXCFilterDesign:
    """Verify the RXC filter realises the maximum-efficiency load.

    The checks deliberately use routes that are independent of each other and
    of the solver's own Z-matrix efficiency formula:

    * the network impedance equals ``R_opt + j*X_opt`` (solver target),
    * a closed form derived here from the admittance equations,
    * an ABCD-matrix circuit simulation of coil + Cp + Cs + Rload,
    * a direct numerical maximisation of that simulated efficiency.
    """

    RLOAD = 100.0
    TARGET_F = 8e6
    RANGE_F = 2.8e6

    @staticmethod
    def reciprocal_sample():
        """Measured sample with Z21 := Z12 so the two-port is exactly reciprocal.

        The solver uses Z12 as the mutual impedance; the raw measurement has a
        ~0.7% Z12/Z21 asymmetry that would otherwise dominate the comparison.
        """
        nw = rf.Network(str(SAMPLE_S2P))
        z = nw.z.copy()
        z[:, 1, 0] = z[:, 0, 1]
        return RichNetwork.from_touchstone(rf.Network.from_z(z, frequency=nw.frequency))

    @staticmethod
    def z_network(cp, cs, rload, w):
        """Impedance of the Cp || (Cs + Rload) network seen from the coil."""
        return 1 / (1j * w * cp + 1 / (rload + 1 / (1j * w * cs)))

    @staticmethod
    def abcd_efficiency(nw, f_index, cp, cs, rload):
        """Pload/Pin from an ABCD cascade: coil -> shunt Cp -> series Cs -> Rload.

        Power flows from port 1 to port 2 of ``nw``; callers flip the network
        when the receiver is port 1.
        """
        w = 2 * np.pi * nw.frequency.f[f_index]
        shunt = np.array([[1, 0], [1j * w * cp, 1]])
        series = np.array([[1, 1 / (1j * w * cs)], [0, 1]])
        chain = nw.a[f_index] @ shunt @ series
        v_load = 1.0
        i_load = v_load / rload
        v_in, i_in = chain @ np.array([v_load, i_load])
        p_load = (v_load * np.conj(i_load)).real
        p_in = (v_in * np.conj(i_in)).real
        return p_load / p_in

    def design(self, rx_port):
        """Run the RXC design and gather what the verification routes need."""
        rich_nw = self.reciprocal_sample()
        res = compute_rxc_filter(
            rich_nw,
            rx_port=rx_port,
            rload=self.RLOAD,
            target_f=self.TARGET_F,
            range_f=self.RANGE_F,
        )
        eff = efficiency_calculator(
            rich_nw, rx_port=rx_port, target_f=self.TARGET_F, range_f=self.RANGE_F
        )
        f_index = int(np.argmin(np.abs(rich_nw.nw.frequency.f - res.max_f_plot)))
        net = rich_nw.nw if rx_port == 2 else rich_nw.nw.flipped()
        return res, eff, net, f_index

    @pytest.mark.parametrize("rx_port", [1, 2])
    def test_network_presents_optimal_load(self, rx_port):
        """Route 1: Z_network(Cp, Cs) == R_opt + j*X_opt at the coil terminals."""
        res, eff, _, _ = self.design(rx_port)
        w = 2 * np.pi * res.max_f_plot
        z_net = self.z_network(res.cp, res.cs, self.RLOAD, w)
        z_opt = complex(eff.max_r_opt, eff.max_x_opt)

        assert abs(z_net - z_opt) / abs(z_opt) < 1e-6
        assert res.cp > 0 and res.cs > 0

    @pytest.mark.parametrize("rx_port", [1, 2])
    def test_matches_independent_closed_form(self, rx_port):
        """Route 2: Cp, Cs agree with an algebraic solution derived here."""
        res, eff, _, _ = self.design(rx_port)
        w = 2 * np.pi * res.max_f_plot
        z_opt = complex(eff.max_r_opt, eff.max_x_opt)

        # Y_net = j*w*Cp + 1/(Rload - j*Xs) with Xs = 1/(w*Cs) must equal 1/Z_opt.
        y_opt = 1 / z_opt
        # Re(Y): Rload / (Rload^2 + Xs^2) = Re(y_opt)
        xs = np.sqrt(self.RLOAD / y_opt.real - self.RLOAD**2)
        cs_expected = 1 / (w * xs)
        # Im(Y): w*Cp + Xs / (Rload^2 + Xs^2) = Im(y_opt)
        cp_expected = (y_opt.imag - xs / (self.RLOAD**2 + xs**2)) / w

        assert res.cs == pytest.approx(cs_expected, rel=1e-6)
        assert res.cp == pytest.approx(cp_expected, rel=1e-6)

    @pytest.mark.parametrize("rx_port", [1, 2])
    def test_abcd_simulation_reaches_max_efficiency(self, rx_port):
        """Route 3: circuit simulation of coil + network + Rload hits max_eff_opt."""
        res, eff, net, f_index = self.design(rx_port)

        eff_sim = self.abcd_efficiency(net, f_index, res.cp, res.cs, self.RLOAD)
        assert eff_sim == pytest.approx(eff.max_eff_opt, abs=1e-9)

        # The design is a maximum: detuning either capacitor lowers efficiency.
        for dcp, dcs in [(1.05, 1.0), (0.95, 1.0), (1.0, 1.05), (1.0, 0.95)]:
            eff_detuned = self.abcd_efficiency(
                net, f_index, res.cp * dcp, res.cs * dcs, self.RLOAD
            )
            assert eff_detuned < eff_sim

    @pytest.mark.parametrize("rx_port", [1, 2])
    def test_direct_efficiency_maximisation_agrees(self, rx_port):
        """Route 4: maximising simulated efficiency over (Cp, Cs) finds the same values."""
        res, eff, net, f_index = self.design(rx_port)

        def neg_eff(y):
            return -self.abcd_efficiency(
                net, f_index, 10 ** y[0], 10 ** y[1], self.RLOAD
            )

        # Start deliberately off the solution.
        y0 = [np.log10(res.cp) + 0.3, np.log10(res.cs) - 0.3]
        opt = minimize(
            neg_eff,
            y0,
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 20000},
        )

        assert -opt.fun == pytest.approx(eff.max_eff_opt, abs=1e-9)
        assert 10 ** opt.x[0] == pytest.approx(res.cp, rel=1e-4)
        assert 10 ** opt.x[1] == pytest.approx(res.cs, rel=1e-4)

    def test_pre_fix_design_was_far_from_optimum(self):
        """Document the bug: resonating the fitted coil inductance misses badly."""
        res, eff, net, f_index = self.design(2)
        w = 2 * np.pi * res.max_f_plot

        # Legacy target: Z_network + j*w*Lrx == R_opt (imaginary part driven to 0).
        z_legacy = complex(eff.max_r_opt, -w * res.lrx)
        cp_old, cs_old = solve_cpcs_closed_form(z_legacy, self.RLOAD, w)
        eff_old = self.abcd_efficiency(net, f_index, cp_old, cs_old, self.RLOAD)

        assert eff_old < 0.05
        assert eff.max_eff_opt > 0.5

    def test_closed_form_roundtrip(self):
        """solve_cpcs_closed_form returns capacitors that reproduce the target."""
        w = 2 * np.pi * 6.78e6
        for z_target in [10 - 250j, 7.75 - 198.6j, 2 - 50j, 40 - 400j]:
            cp, cs = solve_cpcs_closed_form(z_target, 100.0, w)
            assert cp > 0 and cs > 0
            z_net = self.z_network(cp, cs, 100.0, w)
            assert z_net == pytest.approx(z_target, rel=1e-12)

    def test_closed_form_unrealisable(self):
        """No positive Cp/Cs exists for inductive targets or too-small Rload."""
        w = 2 * np.pi * 10e6
        # Inductive optimum (positive X_opt) cannot be built from two capacitors.
        assert solve_cpcs_closed_form(14.85 + 7.33j, 100.0, w) is None
        # |Z|^2 < Re(Z) * Rload: Rload cannot be stepped down to Re(Z).
        assert solve_cpcs_closed_form(10 - 5j, 100.0, w) is None

    def test_warns_when_optimum_not_realisable(self, caplog):
        """A network with positive X_opt gets a best-effort design and a warning."""
        # Short sweep around 10 MHz with the Rm != 0 impedances of
        # TestEfficiencyCalculator._rm_network (reactances scaled with f so the
        # LCR fit inside compute_rxc_filter has something to fit).
        f = np.linspace(9.9e6, 10.1e6, 5)
        scale = f / 10e6
        z11 = 4.18 + 38.49j * scale
        z22 = 1.32 - 1.69j * scale
        zm = 0.836 + 28.171j * scale
        z = np.stack([np.stack([z11, zm], -1), np.stack([zm, z22], -1)], -2)
        nw = rf.Network.from_z(z, frequency=rf.Frequency.from_f(f, unit="Hz"))
        rich_nw = RichNetwork.from_touchstone(nw)
        with caplog.at_level("WARNING", logger="wpt_tools.solver"):
            res = compute_rxc_filter(
                rich_nw, rx_port=2, rload=100.0, target_f=10e6, range_f=0.3e6
            )
        assert res.max_x_opt > 0  # inductive optimum: unrealisable with Cp/Cs
        assert res.cp > 0 and res.cs > 0
        assert any("cannot realise" in rec.getMessage() for rec in caplog.records)


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

        assert hasattr(results, "ls1")
        assert hasattr(results, "cs1")
        assert hasattr(results, "rs1")

        # Check that fitted values are present (may be negative for poor fits)
        assert results.ls1.value is not None
        assert results.cs1.value is not None
        assert results.rs1.value is not None

        # Check R2 scores are present
        assert results.ls1.r2 is not None
        assert results.cs1.r2 is not None
        assert results.rs1.r2 is not None
