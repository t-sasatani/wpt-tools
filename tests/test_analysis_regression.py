"""Regression tests for analysis module using sample.s2p data."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from wpt_tools.analysis import MinMax, nw_tools
from wpt_tools.data_classes import RichNetwork


class TestAnalysisRegression:
    """Regression tests using sample.s2p data with archived results."""

    def setup_method(self):
        """Set up test data and load sample network."""
        self.sample_path = Path(__file__).parent / "data" / "sample.s2p"
        self.results_path = (
            Path(__file__).parent / "data" / "analysis_regression_results.pkl"
        )

        # Load the sample network
        self.rich_nw = RichNetwork.from_touchstone(self.sample_path)

        # Use dynamic frequency range like the old test
        f_start = float(self.rich_nw.nw.frequency.f[0])
        f_stop = float(self.rich_nw.nw.frequency.f[-1])
        self.target_f = (f_start + f_stop) / 2.0
        self.range_f = max((f_stop - f_start) * 0.2, 1.0)
        self.rich_nw.set_f_target_range(self.target_f, self.range_f)

    def test_analyze_efficiency_regression(self):
        """Test efficiency analysis against archived results."""
        # Test both ports if available (like the old test)
        ports = [1, 2] if getattr(self.rich_nw.nw, "nports", 1) >= 2 else [1]

        for rx_port in ports:
            results = nw_tools.analyze_efficiency(
                rich_nw=self.rich_nw, show_plot=False, show_data=False, rx_port=rx_port
            )

            # Load archived results if they exist
            if self.results_path.exists():
                with open(self.results_path, "rb") as f:
                    archived_results = pickle.load(f)

                # Compare key metrics for this port
                port_key = f"efficiency_port_{rx_port}"
                if port_key in archived_results:
                    port_data = archived_results[port_key]
                    np.testing.assert_allclose(
                        results.max_f_plot,
                        port_data["max_f_plot"],
                        rtol=1e-6,
                        err_msg=f"Max frequency mismatch for port {rx_port}",
                    )
                    np.testing.assert_allclose(
                        results.max_eff_opt,
                        port_data["max_eff_opt"],
                        rtol=1e-6,
                        err_msg=f"Max efficiency mismatch for port {rx_port}",
                    )
                    np.testing.assert_allclose(
                        results.max_r_opt,
                        port_data["max_r_opt"],
                        rtol=1e-6,
                        err_msg=f"Max resistance mismatch for port {rx_port}",
                    )
                    np.testing.assert_allclose(
                        results.max_x_opt,
                        port_data["max_x_opt"],
                        rtol=1e-6,
                        err_msg=f"Max reactance mismatch for port {rx_port}",
                    )
                else:
                    # Create initial archive for this port
                    self._create_archive(results, f"efficiency_port_{rx_port}")
                    pytest.skip(
                        f"Created initial archive for port {rx_port}. Re-run test to verify regression."
                    )
                    return
            else:
                # Create initial archive for port 1
                self._create_archive(results, f"efficiency_port_{rx_port}")
                pytest.skip(
                    f"Created initial archive for port {rx_port}. Re-run test to verify regression."
                )
                return

    @pytest.mark.xfail(
        condition=sys.platform == "win32",
        reason="LCR fitting results differ significantly on Windows due to numerical precision differences",
        strict=False,
    )
    def test_fit_z_narrow_regression(self):
        """Test LCR fitting against archived results."""
        from wpt_tools.solvers import lcr_fitting

        # Test wrapper parity: fit_z_narrow vs lcr_fitting
        results_wrapper = nw_tools.fit_z_narrow(
            rich_nw=self.rich_nw,
            show_plot=False,
            show_data=False,
            target_f=self.target_f,
            range_f=self.range_f,
        )

        # Test core function for comparison
        results_core = lcr_fitting(
            rich_nw=self.rich_nw,
            target_f=self.target_f,
            range_f=self.range_f,
        )

        # Verify wrapper parity (like the old test)
        np.testing.assert_allclose(
            results_wrapper.ls1.value,
            results_core.ls1.value,
            rtol=1e-6,
            err_msg="Wrapper parity: Ls1 value mismatch",
        )
        np.testing.assert_allclose(
            results_wrapper.cs1.value,
            results_core.cs1.value,
            rtol=1e-6,
            err_msg="Wrapper parity: Cs1 value mismatch",
        )
        np.testing.assert_allclose(
            results_wrapper.rs1.value,
            results_core.rs1.value,
            rtol=1e-6,
            err_msg="Wrapper parity: Rs1 value mismatch",
        )

        # Use wrapper results for regression testing
        results = results_wrapper

        # Load archived results if they exist
        if self.results_path.exists():
            with open(self.results_path, "rb") as f:
                archived_results = pickle.load(f)

            # Check if lcr_fitting data exists
            if "lcr_fitting" not in archived_results:
                self._create_archive(results, "lcr_fitting")
                pytest.skip(
                    "Created initial archive for lcr_fitting. Re-run test to verify regression."
                )
                return

            # Compare LCR parameters (very loose tolerance for optimization results)
            lcr_data = archived_results["lcr_fitting"]
            np.testing.assert_allclose(
                results.ls1.value,
                lcr_data["ls1_value"],
                rtol=1e-2,
                err_msg="Ls1 value mismatch",
            )
            np.testing.assert_allclose(
                results.cs1.value,
                lcr_data["cs1_value"],
                rtol=1e-2,
                err_msg="Cs1 value mismatch",
            )
            np.testing.assert_allclose(
                results.rs1.value,
                lcr_data["rs1_value"],
                rtol=1e-2,
                err_msg="Rs1 value mismatch",
            )
            np.testing.assert_allclose(
                results.ls2.value,
                lcr_data["ls2_value"],
                rtol=1e-2,
                err_msg="Ls2 value mismatch",
            )
            np.testing.assert_allclose(
                results.cs2.value,
                lcr_data["cs2_value"],
                rtol=1e-2,
                err_msg="Cs2 value mismatch",
            )
            np.testing.assert_allclose(
                results.rs2.value,
                lcr_data["rs2_value"],
                rtol=1e-2,
                err_msg="Rs2 value mismatch",
            )
            np.testing.assert_allclose(
                results.lm.value,
                lcr_data["lm_value"],
                rtol=1e-2,
                err_msg="Lm value mismatch",
            )

            # Compare R² values (looser tolerance for R²)
            np.testing.assert_allclose(
                results.ls1.r2, lcr_data["ls1_r2"], rtol=1e-2, err_msg="Ls1 R² mismatch"
            )
            np.testing.assert_allclose(
                results.cs1.r2, lcr_data["cs1_r2"], rtol=1e-2, err_msg="Cs1 R² mismatch"
            )
        else:
            # Create initial archive
            self._create_archive(results, "lcr_fitting")

    def test_sweep_load_regression(self):
        """Test load sweep against archived results."""
        results = nw_tools.sweep_load(
            rich_nw=self.rich_nw,
            rez_range=MinMax(min=0.1, max=50, step=0.2),
            imz_range=MinMax(min=-200, max=200, step=1),
            input_voltage=5,
            rx_port=1,
            show_plot=False,
            target_f=self.target_f,
            range_f=self.range_f,
        )

        # Load archived results if they exist
        if self.results_path.exists():
            with open(self.results_path, "rb") as f:
                archived_results = pickle.load(f)

            # Check if load_sweep data exists
            if "load_sweep" not in archived_results:
                self._create_archive(results, "load_sweep")
                pytest.skip(
                    "Created initial archive for load_sweep. Re-run test to verify regression."
                )
                return

            # Compare load sweep data
            sweep_data = archived_results["load_sweep"]
            np.testing.assert_allclose(
                results.rez_list,
                sweep_data["rez_list"],
                rtol=1e-6,
                err_msg="Resistance list mismatch",
            )
            np.testing.assert_allclose(
                results.imz_list,
                sweep_data["imz_list"],
                rtol=1e-6,
                err_msg="Reactance list mismatch",
            )
            np.testing.assert_allclose(
                results.eff_grid,
                sweep_data["eff_grid"],
                rtol=1e-6,
                err_msg="Efficiency grid mismatch",
            )
        else:
            # Create initial archive
            self._create_archive(results, "load_sweep")

    def test_calc_rxc_filter_regression(self):
        """Test RXC filter calculation against archived results."""
        from wpt_tools.solvers import compute_rxc_filter

        # Test wrapper parity: calc_rxc_filter vs compute_rxc_filter
        # calc_rxc_filter doesn't return results, it just prints them
        nw_tools.calc_rxc_filter(
            rich_nw=self.rich_nw,
            rx_port=1,
            rload=100,
            c_network="CpCsRl",
            show_data=False,
            target_f=self.target_f,
            range_f=self.range_f,
        )

        # Now test compute_rxc_filter for value comparison
        results = compute_rxc_filter(
            rich_nw=self.rich_nw,
            rx_port=1,
            rload=100.0,
            c_network="CpCsRl",
            target_f=self.target_f,
            range_f=self.range_f,
        )

        # Load archived results if they exist
        if self.results_path.exists():
            with open(self.results_path, "rb") as f:
                archived_results = pickle.load(f)

            # Check if rxc_filter data exists
            if "rxc_filter" not in archived_results:
                self._create_archive(results, "rxc_filter")
                pytest.skip(
                    "Created initial archive for rxc_filter. Re-run test to verify regression."
                )
                return

            # Compare RXC filter values (looser tolerance for optimization results)
            rxc_data = archived_results["rxc_filter"]
            np.testing.assert_allclose(
                results.cp, rxc_data["cp"], rtol=1e-3, err_msg="Cp value mismatch"
            )
            np.testing.assert_allclose(
                results.cs, rxc_data["cs"], rtol=1e-3, err_msg="Cs value mismatch"
            )
        else:
            # Create initial archive
            self._create_archive(results, "rxc_filter")

    def _create_archive(self, results, test_type):
        """Create initial archive of results for regression testing."""
        archive_data = {}

        if test_type.startswith("efficiency_port_"):
            archive_data[test_type] = {
                "max_f_plot": results.max_f_plot,
                "max_eff_opt": results.max_eff_opt,
                "max_r_opt": results.max_r_opt,
                "max_x_opt": results.max_x_opt,
            }
        elif test_type == "lcr_fitting":
            archive_data["lcr_fitting"] = {
                "ls1_value": results.ls1.value,
                "cs1_value": results.cs1.value,
                "rs1_value": results.rs1.value,
                "ls2_value": results.ls2.value,
                "cs2_value": results.cs2.value,
                "rs2_value": results.rs2.value,
                "lm_value": results.lm.value,
                "ls1_r2": results.ls1.r2,
                "cs1_r2": results.cs1.r2,
            }
        elif test_type == "load_sweep":
            archive_data["load_sweep"] = {
                "rez_list": results.rez_list,
                "imz_list": results.imz_list,
                "eff_grid": results.eff_grid,
            }
        elif test_type == "rxc_filter":
            archive_data["rxc_filter"] = {
                "cp": results.cp,
                "cs": results.cs,
            }

        # Load existing archive and update
        if self.results_path.exists():
            with open(self.results_path, "rb") as f:
                existing_data = pickle.load(f)
            existing_data.update(archive_data)
            archive_data = existing_data

        # Save updated archive
        with open(self.results_path, "wb") as f:
            pickle.dump(archive_data, f)

        pytest.skip(
            f"Created initial archive for {test_type}. Re-run test to verify regression."
        )
