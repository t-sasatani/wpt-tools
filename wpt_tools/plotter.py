"""
Plotting functions for wpt-tools.
"""

from wpt_tools.logger import WPTToolsLogger
import matplotlib.pyplot as plt
from wpt_tools.data_classes import RichNetwork, EfficiencyResults
from IPython import get_ipython

logger = WPTToolsLogger().get_logger(__name__)

def plot_efficiency(results: EfficiencyResults, rich_nw: RichNetwork):
    """
    Plot the efficiency results.

    Parameters
    ----------
    results: EfficiencyResults
        The results of the efficiency solver.
    rich_nw: RichNetwork
        The network to plot.

    Returns
    -------
    None

    """
    logger.info("Plotting efficiency results.")
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    axs[0].plot(results.f_plot, results.eff_opt)
    axs[0].set_title("Maximum efficiency")
    axs[0].set_xlabel("Frequency")
    axs[0].set_ylabel("Efficiency")
    axs[0].axvline(rich_nw.target_f, color="gray", lw=1)

    axs[1].plot(results.f_plot, results.r_opt)
    axs[1].set_title("Optimum Re(Zload)")
    axs[1].set_xlabel("Frequency")
    axs[1].set_ylabel("Optimum Re(Zload) (Ohm)")
    axs[1].axvline(rich_nw.target_f, color="gray", lw=1)

    axs[2].plot(results.f_plot, results.x_opt)
    axs[2].set_title("Optimum Im(Zload)")
    axs[2].set_xlabel("Frequency")
    axs[2].set_ylabel("Optimum Im(Zload) (Ohm)")
    axs[2].axvline(rich_nw.target_f, color="gray", lw=1)

    fig.tight_layout()

    # Show the plot if not in Jupyter Notebook
    if not get_ipython():
        plt.show()
