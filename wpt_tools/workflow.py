"""Workflow definitions for wpt-tools."""

from contextlib import chdir
from pathlib import Path

from wpt_tools.analysis import MinMax, nw_tools
from wpt_tools.data_classes import RichNetwork
from wpt_tools.plotter import plot_impedance


def demo_workflow(show_plot: bool = False) -> None:
    """
    Run the wireless power tools demo workflow.

    Parameters
    ----------
    show_plot : bool
        Whether to show interactive plots (default: False)

    """
    # Get the path to the example assets
    examples_dir = Path(__file__).parent.parent / "examples"

    with chdir(examples_dir):
        example_nw = RichNetwork.from_touchstone("./assets/sample.s2p")
        example_nw.set_f_target_range(target_f=6.78e6, range_f=1e6)

        # Full-range impedance plot (ESC to close)
        if show_plot:
            plot_impedance(example_nw, results=None, full_range=True, target_f=6.78e6)

        _ = nw_tools.analyze_efficiency(
            rich_nw=example_nw, show_plot=show_plot, show_data=True, rx_port=1
        )

        # For maximum efficiency analysis
        target_f = 6.78e6
        range_f = 1e6

        _ = nw_tools.fit_z_narrow(
            rich_nw=example_nw, show_plot=show_plot, target_f=target_f, range_f=range_f
        )

        # Load sweep (returns model; also plots by default, ESC to close)
        _ = nw_tools.sweep_load(
            rich_nw=example_nw,
            rez_range=MinMax(min=0.1, max=50, step=0.2),
            imz_range=MinMax(min=-200, max=200, step=1),
            input_voltage=5,
            rx_port=1,
        )
        # Access model fields if needed, e.g. sweep.eff_grid

        nw_tools.calc_rxc_filter(
            rich_nw=example_nw, rx_port=1, rload=100, c_network="CpCsRl"
        )
