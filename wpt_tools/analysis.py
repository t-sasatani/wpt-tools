"""
Analysis code for wireless power transfer systems.
"""

from typing import Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin
from dataclasses import dataclass
from tabulate import tabulate

from wpt_tools.data_classes import RichNetwork, override_frange, EfficiencyResults, LCRFittingResults
from wpt_tools.solvers import efficiency_calculator, lcr_fitting
from wpt_tools.plotter import plot_efficiency, plot_z_matrix, plot_z_matrix_narrow, plot_z11
from wpt_tools.logger import WPTToolsLogger

logger = WPTToolsLogger().get_logger(__name__)

@dataclass
class MinMax:
    """
    Class for minimum and maximum values and step size.

    Parameters
    ----------
    min: float
        The minimum value.
    max: float
        The maximum value.
    step: Optional[float]
        The step size.

    """

    min: float
    max: float
    step: Optional[float]

    def __init__(self, min: float, max: float, step: Optional[float]):
        """
        Initialize the class.
        """
        self.min = min
        self.max = max
        self.step = step

class nw_tools:
    """
    Stateless utilities for analyzing scikit-rf `rf.Network` objects.
    """

    @staticmethod
    def analyze_efficiency(
        rich_nw: RichNetwork,
        rx_port: int = 2,
        show_plot: bool = True,
        show_data: bool = True,
        target_f: Optional[float] = None,
        range_f: Optional[float] = None,
    ) -> EfficiencyResults:
        """
        Analyze the efficiency of the network.

        Parameters
        ----------
        rich_nw: RichNetwork
            The network to analyze.
        rx_port: int
            The port to analyze.
        show_plot: bool
            Whether to show the plot.
        show_data: bool
            Whether to show the data.
        target_f: Optional[float]
            The target frequency.
        range_f: Optional[float]
            The range of the target frequency.

        Returns
        -------
        EfficiencyResults
            The results of the efficiency solver.

        Raises
        ------
        ValueError
            If the target frequency is not found within the specified range.
        TypeError
            If the network is not a rf.Network or nw_with_config.

        """
        results = efficiency_calculator(
            rich_nw, rx_port=rx_port, target_f=target_f, range_f=range_f
        )

        if show_plot is True:
            plot_efficiency(results, rich_nw)

        if show_data is True:
            print(tabulate([
                ["Target frequency", results.max_f_plot],
                ["Maximum efficiency", results.max_eff_opt],
                ["Optimum Re(Zload)", results.max_r_opt],
                ["Optimum Im(Zload)", results.max_x_opt]],
                headers=["Parameter", "Value"],
                stralign='left',
                numalign='right',
                floatfmt='.3e',
                tablefmt='fancy_grid',
                ))

        return results

    @staticmethod
    def plot_z_full(
        rich_nw: RichNetwork,
        target_f: Optional[float] = None,
    ) -> None:
        """
        Plot the full impedance matrix of the network.

        Parameters
        ----------
        rich_nw: RichNetwork
            The network to plot.
        target_f: Optional[float]
            The target frequency.

        Raises
        ------
        ValueError
            If the target frequency is not found within the specified range.
        TypeError
            If the network is not a rf.Network or nw_with_config.

        """
        plot_z_matrix(rich_nw, target_f)

        return None

    @staticmethod
    def fit_z_narrow(
        rich_nw: RichNetwork,
        show_plot: bool = True,
        show_data: bool = True,
        target_f: Optional[float] = None,
        range_f: Optional[float] = None,
    ) -> LCRFittingResults:
        """
        Fit simple LCR models in a narrow frequency band and optionally plot.

        Parameters
        ----------
        rich_nw: RichNetwork
            The network to fit.
        show_plot: bool
            Whether to show the plot.
        show_data: bool
            Whether to show the fit.
        target_f: Optional[float]
            The target frequency.
        range_f: Optional[float]
            The range of the target frequency.

        Returns
        -------
        LCRFittingResults
            The results of the LCR fitting.

        """
        results = lcr_fitting(rich_nw, target_f=target_f, range_f=range_f)

        if show_data is True:

            print("Fitting values assuming a pair of series LCR resonators\n")
            print(
                tabulate([
                    ["Ls1", results.ls1.value, results.ls1.r2],
                    ["Cs1", results.cs1.value, results.cs1.r2],
                    ["Rs1", results.rs1.value, results.rs1.r2],
                    ["f_1", 1 / (2 * np.pi * np.sqrt(results.ls1 * results.cs1))],
                    [f"Q_1 (approx., @{rich_nw.target_f:.3e} Hz)", (2 * np.pi * rich_nw.target_f * results.ls1 / results.rs1)[0]]
                ], headers=["Parameter", "Value", "R2"], stralign='left', numalign='right', floatfmt='.3e', tablefmt='fancy_grid')
            )
            if rich_nw.nw.nports == 2:
                print(
                    tabulate([
                        ["Ls2", results.ls2.value, results.ls2.r2],
                        ["Cs2", results.cs2.value, results.cs2.r2],
                        ["Rs2", results.rs2.value, results.rs2.r2],
                        ["f_2", 1 / (2 * np.pi * np.sqrt(results.ls2 * results.cs2))],
                        [f"Q_2 (approx., @{rich_nw.target_f:.3e} Hz)", (2 * np.pi * rich_nw.target_f * results.ls2 / results.rs2)[0]]
                    ], headers=["Parameter", "Value", "R2"], stralign='left', numalign='right', floatfmt='.3e', tablefmt='fancy_grid')
                )
                print(
                    tabulate([
                        ["Lm", results.lm.value, results.lm.r2],
                        ["km", results.lm.value / np.sqrt(results.ls1.value * results.ls2.value)]
                    ], headers=["Parameter", "Value", "R2"], stralign='left', numalign='right', floatfmt='.3e', tablefmt='fancy_grid')
                )

        if show_plot is True:
            if rich_nw.nw.nports == 1:
                plot_z11(rich_nw, results, target_f)
            if rich_nw.nw.nports == 2:
                plot_z_matrix_narrow(rich_nw, results, target_f)


        return results

    @staticmethod
    def calc_rxc_filter(
        rich_nw: RichNetwork,
        rx_port,
        rload,
        *,
        c_network: str = "CpCsRl",
        target_f: Optional[float] = None,
        range_f: Optional[float] = None,
    ):
        """Compute receiver capacitor values for a target load at the optimal point."""
        rich_nw = override_frange(rich_nw, target_f=target_f, range_f=range_f)

        results = nw_tools.fit_z_narrow(
            rich_nw, show_plot=0, show_data=0
        )
        max_f_plot, max_eff_opt, max_r_opt, max_x_opt = nw_tools.analyze_efficiency(
            rich_nw, rx_port=rx_port, show_plot=0, show_data=0
        )

        max_w_plot = 2 * np.pi * max_f_plot
        if rx_port == 1:
            lrx = results.ls1
        elif rx_port == 2:
            lrx = results.ls2
        else:
            raise ValueError("set rx_port parameter to 1 or 2")

        print("-----------Analysis results-----------")
        print(
            tabulate([
                ["Target frequency", results.max_f_plot],
                ["Maximum efficiency", max_eff_opt],
                ["Receiver inductance", lrx],
                ["Optimum load", max_r_opt],
                ["Target Rload", rload]
            ], headers=["Parameter", "Value"], stralign='left', numalign='right', floatfmt='.3e', tablefmt='fancy_grid')
        )

        if c_network == "CpCsRl":

            def Z(params):
                cp, cs = params
                return (
                    1
                    / (
                        (1j * max_w_plot * cp)
                        + 1 / ((1 / (1j * max_w_plot * cs) + rload))
                    )
                    + 1j * max_w_plot * lrx
                )

            def Zerror(params):
                return np.linalg.norm([Z(params).real - max_r_opt, Z(params).imag])

        sol = fmin(Zerror, np.array([100e-12, 100e-12]), xtol=1e-9, ftol=1e-9)
        logger.info(sol)

    @staticmethod
    def plot_optimal_load(
        rich_nw: RichNetwork,
        rez_range: MinMax,
        imz_range: MinMax,
        rx_port: Literal[1, 2],
        input_voltage: Optional[float] = 1,
        target_f: Optional[float] = None,
        range_f: Optional[float] = None,
    ):
        """Plot efficiency, input power and output power over a grid of complex loads.

        Optimal load visualization
        Imura, "Wireless Power Transfer: Using Magnetic and Electric Resonance Coupling Techniques," Springer Singapore 2020.

        Parameters
        ----------
        rich_nw: RichNetwork
            The network to plot.

        rez_range: MinMax
            The range of the real(Zload).
        imz_range: MinMax
            The range of the imaginary(Zload).
        rx_port: int
            The port to plot.
        input_voltage: Optional[float]
            The input voltage.
        target_f: Optional[float]
            The target frequency.
        range_f: Optional[float]
            The range of the target frequency.\

        """
        rich_nw = override_frange(rich_nw, target_f=target_f, range_f=range_f)

        if rich_nw.target_f is None:
            raise ValueError("target frequency is not set.")

        rez_list = np.arange(rez_range.min, rez_range.max, rez_range.step)
        imz_list = np.arange(imz_range.min, imz_range.max, imz_range.step)
        eff_grid = np.zeros((rez_list.size, imz_list.size))
        Pin = np.zeros((rez_list.size, imz_list.size))
        Pout = np.zeros((rez_list.size, imz_list.size))

        if rx_port == 2:
            Z11 = rich_nw.nw.z[rich_nw.target_f_index, 0, 0]
            Z22 = rich_nw.nw.z[rich_nw.target_f_index, 1, 1]
        elif rx_port == 1:
            Z11 = rich_nw.nw.z[rich_nw.target_f_index, 1, 1]
            Z22 = rich_nw.nw.z[rich_nw.target_f_index, 0, 0]

        Zm = rich_nw.nw.z[rich_nw.target_f_index, 0, 1]

        for rez_index in range(rez_list.size):
            for imz_index in range(imz_list.size):
                ZL = rez_list[rez_index] + 1j * imz_list[imz_index]
                V1 = input_voltage  # arbitrary
                I1 = (Z22 + ZL) / (Z11 * (Z22 + ZL) - Zm**2) * V1
                I2 = -Zm / (Z11 * (Z22 + ZL) - Zm**2) * V1
                V2 = Zm * ZL / (Z11 * (Z22 + ZL) - Zm**2) * V1

                Pin[rez_index][imz_index] = (V1 * I1.conjugate()).real
                Pout[rez_index][imz_index] = (V2 * (-I2.conjugate())).real
                eff_grid[rez_index][imz_index] = (V2 * (-I2.conjugate())).real / (
                    V1 * I1.conjugate()
                ).real

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        c = axs[0].pcolor(
            imz_list, rez_list, eff_grid, cmap="hot", vmin=0, vmax=1, shading="auto"
        )
        fig.colorbar(c, ax=axs[0])
        axs[0].set_title(
            "Efficiency @ "
            + format(rich_nw.nw.frequency.f[rich_nw.target_f_index], "3.2e")
            + " Hz"
        )
        axs[0].set_ylabel("Re(Z_load)")
        axs[0].set_xlabel("Im(Z_load)")

        c = axs[1].pcolor(
            imz_list, rez_list, Pin, cmap="hot", vmin=0, vmax=Pin.max(), shading="auto"
        )
        fig.colorbar(c, ax=axs[1])
        axs[1].set_title(
            "Input Power (W) @ "
            + format(rich_nw.nw.frequency.f[rich_nw.target_f_index], "3.2e")
            + " Hz"
        )
        axs[1].set_ylabel("Re(Z_load)")
        axs[1].set_xlabel("Im(Z_load)")

        c = axs[2].pcolor(
            imz_list, rez_list, Pout, cmap="hot", vmin=0, vmax=Pin.max(), shading="auto"
        )
        fig.colorbar(c, ax=axs[2])
        axs[2].set_title(
            "Output Power (W) @ "
            + format(rich_nw.nw.frequency.f[rich_nw.target_f_index], "3.2e")
            + " Hz"
        )
        axs[2].set_ylabel("Re(Z_load)")
        axs[2].set_xlabel("Im(Z_load)")

        fig.tight_layout()
