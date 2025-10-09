"""
Analysis code for wireless power transfer systems.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from tabulate import tabulate

from wpt_tools.data_classes import EfficiencyResults, LCRFittingResults, RichNetwork
from wpt_tools.logger import WPTToolsLogger
from wpt_tools.plotter import plot_efficiency, plot_impedance, plot_load_sweep
from wpt_tools.solvers import (
    compute_load_sweep,
    compute_rxc_filter,
    efficiency_calculator,
    lcr_fitting,
)

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
            print(
                tabulate(
                    [
                        ["Target frequency", results.max_f_plot],
                        ["Maximum efficiency", results.max_eff_opt],
                        ["Optimum Re(Zload)", results.max_r_opt],
                        ["Optimum Im(Zload)", results.max_x_opt],
                    ],
                    headers=["Parameter", "Value"],
                    stralign="left",
                    numalign="right",
                    floatfmt=".3e",
                    tablefmt="fancy_grid",
                )
            )

        return results

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
        # Provide context so results can print tables without external inputs
        results._target_f = (
            float(rich_nw.target_f) if rich_nw.target_f is not None else None
        )
        results._nports = int(getattr(rich_nw.nw, "nports", 1))

        if show_data is True:
            results.print_tables()

        if show_plot is True:
            # Plot within the narrow range by default; overlay fits when available
            plot_impedance(
                rich_nw, results=results, full_range=False, target_f=target_f
            )

        return results

    @staticmethod
    def calc_rxc_filter(
        rich_nw: RichNetwork,
        rx_port: Literal[1, 2],
        rload: float,
        *,
        target_f: Optional[float] = None,
        range_f: Optional[float] = None,
        show_data: bool = True,
        c_network: Literal["CpCsRl"] = "CpCsRl",
    ):
        """Compute receiver RXC filter using solver and print a summary."""
        res = compute_rxc_filter(
            rich_nw,
            rx_port=rx_port,
            rload=rload,
            c_network=c_network,
            target_f=target_f,
            range_f=range_f,
        )

        if show_data:
            print("-----------RXC Filter results-----------")
            print(
                tabulate(
                    [
                        ["Target frequency", res.max_f_plot],
                        ["Optimum Re(Zload)", res.max_r_opt],
                        ["Optimum Im(Zload)", res.max_x_opt],
                        ["Receiver inductance", res.lrx],
                        ["Target Rload", res.rload],
                        ["Cp", res.cp],
                        ["Cs", res.cs],
                    ],
                    headers=["Parameter", "Value"],
                    stralign="left",
                    numalign="right",
                    floatfmt=".3e",
                    tablefmt="fancy_grid",
                )
            )

    @staticmethod
    def sweep_load(
        rich_nw: RichNetwork,
        rez_range: MinMax,
        imz_range: MinMax,
        rx_port: Literal[1, 2],
        input_voltage: Optional[float] = 1,
        target_f: Optional[float] = None,
        range_f: Optional[float] = None,
        show_plot: bool = True,
    ):
        """Compute load sweep results and optionally plot them. Returns results."""
        results = compute_load_sweep(
            rich_nw,
            rez_min=rez_range.min,
            rez_max=rez_range.max,
            rez_step=rez_range.step,
            imz_min=imz_range.min,
            imz_max=imz_range.max,
            imz_step=imz_range.step,
            rx_port=rx_port,
            input_voltage=input_voltage,
            target_f=target_f,
            range_f=range_f,
        )
        if show_plot:
            plot_load_sweep(results)
        return results
