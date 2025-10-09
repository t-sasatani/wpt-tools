"""
Plotting functions for wpt-tools.
"""

from wpt_tools.logger import WPTToolsLogger
import matplotlib.pyplot as plt
from wpt_tools.data_classes import RichNetwork, EfficiencyResults, LCRFittingResults
from IPython import get_ipython
from typing import Optional
from wpt_tools.solvers import series_lcr_xself, series_lcr_xm

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

    return None

def plot_z_matrix(rich_nw: RichNetwork, target_f: Optional[float] = None):
    """
    Plot the impedance matrix of the network.

    Parameters
    ----------
    rich_nw: RichNetwork
        The network to plot.
    target_f: Optional[float]
        The target frequency.

    """
    logger.info("Plotting impedance matrix.")
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.5))
    twin = ["init"] * 4
    pr = ["init"] * 4
    pi = ["init"] * 4

    for rx_port in range(1, 3):
        for tx_port in range(1, 3):
            plot_index = (rx_port - 1) * 2 + (tx_port - 1) * 1
            axs[plot_index].set_title("Z" + str(rx_port) + str(tx_port))
            twin[plot_index] = axs[plot_index].twinx()
            (pr[plot_index],) = axs[plot_index].plot(
                rich_nw.nw.frequency.f,
                rich_nw.nw.z[:, rx_port - 1, tx_port - 1].real,
                label="real(z)",
            )
            (pi[plot_index],) = twin[plot_index].plot(
                rich_nw.nw.frequency.f,
                rich_nw.nw.z[:, rx_port - 1, tx_port - 1].imag,
                "r-",
                label="imag(z)",
            )
            axs[plot_index].set_xlabel("frequency")
            axs[plot_index].set_ylabel(
                "re(Z" + str(rx_port) + str(tx_port) + ") Ohm"
            )
            twin[plot_index].set_ylabel(
                "im(Z" + str(rx_port) + str(tx_port) + ") Ohm"
            )
            axs[plot_index].yaxis.label.set_color(pr[plot_index].get_color())
            twin[plot_index].yaxis.label.set_color(pi[plot_index].get_color())
            vline_f = rich_nw.target_f if rich_nw.target_f is not None else target_f
            if vline_f is not None:
                axs[plot_index].axvline(vline_f, color="gray", lw=1)
            axs[plot_index].set_ylim(
                (
                    -abs(rich_nw.nw.z[:, rx_port - 1, tx_port - 1].real).max(),
                    abs(rich_nw.nw.z[:, rx_port - 1, tx_port - 1].real).max(),
                )
            )
            twin[plot_index].set_ylim(
                (
                    -abs(rich_nw.nw.z[:, rx_port - 1, tx_port - 1].imag).max(),
                    abs(rich_nw.nw.z[:, rx_port - 1, tx_port - 1].imag).max(),
                )
            )
            axs[plot_index].axhline(0, color="gray", lw=1)
    fig.tight_layout()

    # Show the plot if not in Jupyter Notebook
    if not get_ipython():
        plt.show()

    return None

def plot_z_matrix_narrow(rich_nw: RichNetwork, results: LCRFittingResults, target_f: Optional[float] = None):
    """
    Plot the impedance matrix of the network in a narrow frequency band.
    """
    logger.info("Plotting impedance matrix in a narrow frequency band.")
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.5))
    twin = ["init"] * 4
    pr = ["init"] * 4
    pi = ["init"] * 4

    for rx_port in range(1, 3):
        for tx_port in range(1, 3):
            plot_index = (rx_port - 1) * 2 + (tx_port - 1) * 1
            axs[plot_index].set_title("Z" + str(rx_port) + str(tx_port))
            twin[plot_index] = axs[plot_index].twinx()
            (pr[plot_index],) = axs[plot_index].plot(
                rich_nw.nw.frequency.f[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
                ],
                rich_nw.nw.z[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop,
                    rx_port - 1,
                    tx_port - 1,
                ].real,
                label="real(z)",
                lw=3,
            )
            (pi[plot_index],) = twin[plot_index].plot(
                rich_nw.nw.frequency.f[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
                ],
                rich_nw.nw.z[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop,
                    rx_port - 1,
                    tx_port - 1,
                ].imag,
                "r-",
                label="imag(z)",
                lw=3,
            )
            axs[plot_index].set_xlabel("frequency")
            axs[plot_index].set_ylabel(
                "re(Z" + str(rx_port) + str(tx_port) + ") Ohm"
            )
            twin[plot_index].set_ylabel(
                "im(Z" + str(rx_port) + str(tx_port) + ") Ohm"
            )
            axs[plot_index].yaxis.label.set_color(
                pr[plot_index].get_color()
            )
            twin[plot_index].yaxis.label.set_color(
                pi[plot_index].get_color()
            )
            axs[plot_index].axvline(rich_nw.target_f, color="gray", lw=1)
            axs[plot_index].set_ylim(
                (
                    -1.5
                    * abs(
                        rich_nw.nw.z[
                            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop,
                            rx_port - 1,
                            tx_port - 1,
                        ].real
                    ).max(),
                    1.5
                    * abs(
                        rich_nw.nw.z[
                            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop,
                            rx_port - 1,
                            tx_port - 1,
                        ].real
                    ).max(),
                )
            )
            twin[plot_index].set_ylim(
                (
                    -1.5
                    * abs(
                        rich_nw.nw.z[
                            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop,
                            rx_port - 1,
                            tx_port - 1,
                        ].imag
                    ).max(),
                    1.5
                    * abs(
                        rich_nw.nw.z[
                            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop,
                            rx_port - 1,
                            tx_port - 1,
                        ].imag
                    ).max(),
                )
            )
            axs[plot_index].axhline(0, color="gray", lw=1)

    twin[0].plot(
        rich_nw.nw.frequency.f[
            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
        ],
        series_lcr_xself(
            rich_nw.nw.frequency.f[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
            ],
            results.ls1.value,
            results.cs1.value,
        ),
        label="imag(z) fitting",
        color="green",
    )
    twin[3].plot(
        rich_nw.nw.frequency.f[
            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
        ],
        series_lcr_xself(
            rich_nw.nw.frequency.f[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
            ],
            results.ls2.value,
            results.cs2.value,
        ),
        label="imag(z) fitting",
        color="green",
    )
    twin[1].plot(
        rich_nw.nw.frequency.f[
            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
        ],
        series_lcr_xm(
            rich_nw.nw.frequency.f[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
            ],
            results.lm.value,
        ),
        label="imag(z) fitting",
        color="green",
    )
    twin[2].plot(
        rich_nw.nw.frequency.f[
            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
        ],
        series_lcr_xm(
            rich_nw.nw.frequency.f[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
            ],
            results.lm.value,
        ),
        label="imag(z) fitting",
        color="green",
    )

    fig.tight_layout()

    # Show the plot if not in Jupyter Notebook
    if not get_ipython():
        plt.show()

    return None

def plot_z11(rich_nw: RichNetwork, results: LCRFittingResults, target_f: Optional[float] = None):
    """
    Plot the impedance of the network at port 1.

    Parameters
    ----------
    rich_nw: RichNetwork
        The network to plot.
    results: LCRFittingResults
        The results of the LCR fitting.
    target_f: Optional[float]
        The target frequency.

    """
    fig, axs = plt.subplots(1, 1, figsize=(5, 3.5))
    twin = ["init"] * 1
    pr = ["init"] * 1
    pi = ["init"] * 1

    axs.set_title("Z11")
    twin = axs.twinx()
    (pr[0],) = axs.plot(
        rich_nw.nw.frequency.f[
            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
        ],
        rich_nw.nw.z[
            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 0, 0
        ].real,
        label="real(z)",
        lw=3,
    )
    (pi[0],) = twin.plot(
        rich_nw.nw.frequency.f[
            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
        ],
        rich_nw.nw.z[
            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 0, 0
        ].imag,
        "r-",
        label="imag(z)",
        lw=3,
    )
    axs.set_xlabel("frequency")
    axs.set_ylabel("re(Z11) Ohm")
    twin.set_ylabel("im(Z11) Ohm")
    axs.yaxis.label.set_color(pr[0].get_color())
    twin.yaxis.label.set_color(pi[0].get_color())
    axs.axvline(rich_nw.target_f, color="gray", lw=1)
    axs.set_ylim(
        (
            -1.5
            * abs(
                rich_nw.nw.z[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop,
                    0,
                    0,
                ].real
            ).max(),
            1.5
            * abs(
                rich_nw.nw.z[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop,
                    0,
                    0,
                ].real
            ).max(),
        )
    )
    twin.set_ylim(
        (
            -1.5
            * abs(
                rich_nw.nw.z[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop,
                    0,
                    0,
                ].imag
            ).max(),
            1.5
            * abs(
                rich_nw.nw.z[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop,
                    0,
                    0,
                ].imag
            ).max(),
        )
    )
    axs.axhline(0, color="gray", lw=1)

    twin.plot(
        rich_nw.nw.frequency.f[
            rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
        ],
        series_lcr_xself(
            rich_nw.nw.frequency.f[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
            ],
            results.ls1,
            results.cs1,
        ),
        label="imag(z) fitting",
        color="green",
    )
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
    logger.info("Plotting impedance (unified). full_range=%s" % (full_range))

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
