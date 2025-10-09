"""
Plotting functions for wpt-tools.
"""

from typing import Optional

import matplotlib.pyplot as plt
from IPython import get_ipython

from wpt_tools.data_classes import (
    EfficiencyResults,
    LCRFittingResults,
    OptimalLoadGridResults,
    RichNetwork,
)
from wpt_tools.logger import WPTToolsLogger
from wpt_tools.solvers import series_lcr_xm, series_lcr_xself

logger = WPTToolsLogger().get_logger(__name__)


def _attach_escape_close(fig):
    """Close the given figure when ESC is pressed."""

    def _on_key(event):
        if event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)


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
    _attach_escape_close(fig)

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

    return None


def plot_impedance(
    rich_nw: RichNetwork,
    results: Optional[LCRFittingResults] = None,
    *,
    full_range: bool = False,
    target_f: Optional[float] = None,
) -> None:
    """
    Unified impedance plotter.

    - For 1-port: plots Z11 real/imag. If results provided, overlays fit for imag(Z11).
    - For 2-port: plots 4 subplots for Z11, Z12, Z21, Z22 real/imag. If results provided, overlays fits for self and mutual reactances.
    - By default, plots within `rich_nw` narrow range; set `full_range=True` to plot full sweep.
    """
    logger.info("Plotting impedance. full_range=%s" % (full_range))

    # Determine frequency slice
    if full_range:
        f_vec = rich_nw.nw.frequency.f
        z_slice = rich_nw.nw.z
    else:
        f_vec = rich_nw.nw.frequency.f[
            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
        ]
        z_slice = rich_nw.nw.z[
            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, :, :
        ]

    nports = rich_nw.nw.nports

    if nports == 1:
        fig, axs = plt.subplots(1, 1, figsize=(5, 3.5))
        _attach_escape_close(fig)
        twin = axs.twinx()

        lw = 3 if not full_range else 1
        (pr,) = axs.plot(f_vec, z_slice[:, 0, 0].real, label="real(z)", lw=lw)
        (pi,) = twin.plot(f_vec, z_slice[:, 0, 0].imag, "r-", label="imag(z)", lw=lw)

        axs.set_title("Z11")
        axs.set_xlabel("frequency")
        axs.set_ylabel("re(Z11) Ohm")
        twin.set_ylabel("im(Z11) Ohm")
        axs.yaxis.label.set_color(pr.get_color())
        twin.yaxis.label.set_color(pi.get_color())
        vline_f = rich_nw.target_f if rich_nw.target_f is not None else target_f
        if vline_f is not None:
            axs.axvline(vline_f, color="gray", lw=1)
        # Set symmetric y-limits consistent with legacy behavior
        if full_range:
            axs.set_ylim(
                (
                    -abs(z_slice[:, 0, 0].real).max(),
                    abs(z_slice[:, 0, 0].real).max(),
                )
            )
            twin.set_ylim(
                (
                    -abs(z_slice[:, 0, 0].imag).max(),
                    abs(z_slice[:, 0, 0].imag).max(),
                )
            )
        else:
            axs.set_ylim(
                (
                    -1.5 * abs(z_slice[:, 0, 0].real).max(),
                    1.5 * abs(z_slice[:, 0, 0].real).max(),
                )
            )
            twin.set_ylim(
                (
                    -1.5 * abs(z_slice[:, 0, 0].imag).max(),
                    1.5 * abs(z_slice[:, 0, 0].imag).max(),
                )
            )
        axs.axhline(0, color="gray", lw=1)

        # Overlay fit if available and not full range (fits are narrowband)
        if results is not None and not full_range:
            twin.plot(
                f_vec,
                series_lcr_xself(
                    f_vec,
                    results.ls1.value,
                    results.cs1.value,
                ),
                label="imag(z) fitting",
                color="green",
            )

        fig.tight_layout()
        if not get_ipython():
            plt.show()
        return None

    # Two-port plotting
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.5))
    _attach_escape_close(fig)
    twins = [axs[i].twinx() for i in range(4)]

    for rx_port in range(1, 3):
        for tx_port in range(1, 3):
            plot_index = (rx_port - 1) * 2 + (tx_port - 1) * 1
            axs[plot_index].set_title("Z" + str(rx_port) + str(tx_port))
            lw = 3 if not full_range else 1
            (pr,) = axs[plot_index].plot(
                f_vec,
                z_slice[:, rx_port - 1, tx_port - 1].real,
                label="real(z)",
                lw=lw,
            )
            (pi,) = twins[plot_index].plot(
                f_vec,
                z_slice[:, rx_port - 1, tx_port - 1].imag,
                "r-",
                label="imag(z)",
                lw=lw,
            )
            axs[plot_index].set_xlabel("frequency")
            axs[plot_index].set_ylabel("re(Z%s%s) Ohm" % (rx_port, tx_port))
            twins[plot_index].set_ylabel("im(Z%s%s) Ohm" % (rx_port, tx_port))
            axs[plot_index].yaxis.label.set_color(pr.get_color())
            twins[plot_index].yaxis.label.set_color(pi.get_color())
            vline_f = rich_nw.target_f if rich_nw.target_f is not None else target_f
            if vline_f is not None:
                axs[plot_index].axvline(vline_f, color="gray", lw=1)
            # Set symmetric y-limits consistent with legacy behavior
            if full_range:
                axs[plot_index].set_ylim(
                    (
                        -abs(z_slice[:, rx_port - 1, tx_port - 1].real).max(),
                        abs(z_slice[:, rx_port - 1, tx_port - 1].real).max(),
                    )
                )
                twins[plot_index].set_ylim(
                    (
                        -abs(z_slice[:, rx_port - 1, tx_port - 1].imag).max(),
                        abs(z_slice[:, rx_port - 1, tx_port - 1].imag).max(),
                    )
                )
            else:
                axs[plot_index].set_ylim(
                    (
                        -1.5 * abs(z_slice[:, rx_port - 1, tx_port - 1].real).max(),
                        1.5 * abs(z_slice[:, rx_port - 1, tx_port - 1].real).max(),
                    )
                )
                twins[plot_index].set_ylim(
                    (
                        -1.5 * abs(z_slice[:, rx_port - 1, tx_port - 1].imag).max(),
                        1.5 * abs(z_slice[:, rx_port - 1, tx_port - 1].imag).max(),
                    )
                )
            axs[plot_index].axhline(0, color="gray", lw=1)

    # Overlay fits when provided and plotting narrow range
    if results is not None and not full_range:
        # Z11 fit on subplot 0
        twins[0].plot(
            f_vec,
            series_lcr_xself(
                f_vec,
                results.ls1.value,
                results.cs1.value,
            ),
            label="imag(z) fitting",
            color="green",
        )
        # Z22 fit on subplot 3
        twins[3].plot(
            f_vec,
            series_lcr_xself(
                f_vec,
                results.ls2.value,
                results.cs2.value,
            ),
            label="imag(z) fitting",
            color="green",
        )
        # Mutual Z12 and Z21 use series_lcr_xm
        twins[1].plot(
            f_vec,
            series_lcr_xm(
                f_vec,
                results.lm.value,
            ),
            label="imag(z) fitting",
            color="green",
        )
        twins[2].plot(
            f_vec,
            series_lcr_xm(
                f_vec,
                results.lm.value,
            ),
            label="imag(z) fitting",
            color="green",
        )

    fig.tight_layout()
    if not get_ipython():
        plt.show()
    return None


def plot_load_sweep(results: OptimalLoadGridResults):
    """Plot efficiency, input power and output power over a grid (model-driven)."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    _attach_escape_close(fig)

    c = axs[0].pcolor(
        results.imz_list,
        results.rez_list,
        results.eff_grid,
        cmap="hot",
        vmin=0,
        vmax=1,
        shading="auto",
    )
    fig.colorbar(c, ax=axs[0])
    title_f = f" @ {results.target_f:.2e} Hz" if results.target_f is not None else ""
    axs[0].set_title("Efficiency" + title_f)
    axs[0].set_ylabel("Re(Z_load)")
    axs[0].set_xlabel("Im(Z_load)")

    c = axs[1].pcolor(
        results.imz_list,
        results.rez_list,
        results.Pin,
        cmap="hot",
        vmin=0,
        vmax=results.Pin.max(),
        shading="auto",
    )
    fig.colorbar(c, ax=axs[1])
    axs[1].set_title("Input Power (W)" + title_f)
    axs[1].set_ylabel("Re(Z_load)")
    axs[1].set_xlabel("Im(Z_load)")

    c = axs[2].pcolor(
        results.imz_list,
        results.rez_list,
        results.Pout,
        cmap="hot",
        vmin=0,
        vmax=results.Pin.max(),
        shading="auto",
    )
    fig.colorbar(c, ax=axs[2])
    axs[2].set_title("Output Power (W)" + title_f)
    axs[2].set_ylabel("Re(Z_load)")
    axs[2].set_xlabel("Im(Z_load)")

    fig.tight_layout()
    if not get_ipython():
        plt.show()
    return None
