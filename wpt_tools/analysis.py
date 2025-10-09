"""
Analysis code for wireless power transfer systems.
"""

import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from scipy.optimize import curve_fit, fmin

from wpt_tools.data_classes import RichNetwork, override_frange
from wpt_tools.logger import WPTToolsLogger

logger = WPTToolsLogger().get_logger(__name__)


def compute_efficiency_vectors(
    rich_nw: RichNetwork,
    rx_port: int,
    target_f: Optional[float],
    range_f: Optional[float],
) -> Tuple[
    list[float], list[float], list[float], list[float], float, float, float, float
]:
    """Compute efficiency vectors and maxima (public compute entry point).

    Returns (f_plot, r_opt, x_opt, eff_opt, max_f_plot, max_eff_opt, max_r_opt, max_x_opt).
    """
    rich_nw = override_frange(rich_nw, target_f=target_f, range_f=range_f)

    f_plot: list[float] = []
    r_opt: list[float] = []
    x_opt: list[float] = []
    eff_opt: list[float] = []

    max_eff_opt = 0.0
    max_x_opt = 0.0
    max_r_opt = 0.0
    max_f_plot = float(rich_nw.target_f) if rich_nw.target_f is not None else 0.0

    for f_index in range(rich_nw.sweeppoint):
        if rich_nw.target_f is None:
            raise ValueError("Target frequency is not set.")
        if rich_nw.range_f is None:
            raise ValueError("Range frequency is not set.")
        if (
            abs(rich_nw.target_f - rich_nw.nw.frequency.f[f_index])
            < rich_nw.range_f / 2
        ):
            if rx_port == 2:
                Z11 = rich_nw.nw.z[f_index, 0, 0]
                Z22 = rich_nw.nw.z[f_index, 1, 1]
            elif rx_port == 1:
                Z11 = rich_nw.nw.z[f_index, 1, 1]
                Z22 = rich_nw.nw.z[f_index, 0, 0]
            else:
                raise ValueError("set rx_port to 1 or 2.")
            Zm = rich_nw.nw.z[f_index, 0, 1]
            f_temp = rich_nw.nw.frequency.f[f_index]
            r_det_temp = Z11.real * Z22.real - Zm.real**2

            kq2_temp = (Zm.real**2 + Zm.imag**2) / r_det_temp
            r_opt_temp = r_det_temp / Z11.real * np.sqrt(1 + kq2_temp)
            x_opt_temp = Zm.real * Zm.imag - Z11.real * Z22.imag / Z11.real
            eff_opt_temp = kq2_temp / (1 + np.sqrt(1 + kq2_temp)) ** 2

            f_plot.append(float(f_temp))
            r_opt.append(float(r_opt_temp))
            x_opt.append(float(x_opt_temp))
            eff_opt.append(float(eff_opt_temp))

            if max_eff_opt < eff_opt_temp:
                max_f_plot = float(f_temp)
                max_eff_opt = float(eff_opt_temp)
                max_r_opt = float(r_opt_temp)
                max_x_opt = float(x_opt_temp)

    return f_plot, r_opt, x_opt, eff_opt, max_f_plot, max_eff_opt, max_r_opt, max_x_opt


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
    ) -> Tuple[float, float, float, float]:
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
        max_f_plot: float
            The target frequency.
        max_eff_opt: float
            The maximum efficiency.
        max_r_opt: float
            The maximum real(Zload).
        max_x_opt: float
            The maximum imaginary(Zload).

        Raises
        ------
        ValueError
            If the target frequency is not found within the specified range.
        TypeError
            If the network is not a rf.Network or nw_with_config.

        """
        (
            f_plot,
            r_opt,
            x_opt,
            eff_opt,
            max_f_plot,
            max_eff_opt,
            max_r_opt,
            max_x_opt,
        ) = compute_efficiency_vectors(
            rich_nw, rx_port=rx_port, target_f=target_f, range_f=range_f
        )

        if show_plot is True:
            fig, axs = plt.subplots(1, 3, figsize=(18, 4))

            axs[0].plot(f_plot, eff_opt)
            axs[0].set_title("Maximum efficiency")
            axs[0].set_xlabel("Frequency")
            axs[0].set_ylabel("Efficiency")
            axs[0].axvline(rich_nw.target_f, color="gray", lw=1)

            axs[1].plot(f_plot, r_opt)
            axs[1].set_title("Optimum Re(Zload)")
            axs[1].set_xlabel("Frequency")
            axs[1].set_ylabel("Optimum Re(Zload) (Ohm)")
            axs[1].axvline(rich_nw.target_f, color="gray", lw=1)

            axs[2].plot(f_plot, x_opt)
            axs[2].set_title("Optimum Im(Zload)")
            axs[2].set_xlabel("Frequency")
            axs[2].set_ylabel("Optimum Im(Zload) (Ohm)")
            axs[2].axvline(rich_nw.target_f, color="gray", lw=1)

            fig.tight_layout()

        if show_data is True:
            print("----Analysis results----")
            print("Target frequency: %.3e" % (max_f_plot))
            print("Maximum efficiency: %.2f" % (max_eff_opt))
            print("Optimum Re(Zload): %.2f" % (max_r_opt))
            print("Optimum Im(Zload): %.2f" % (max_x_opt))

        return max_f_plot, max_eff_opt, max_r_opt, max_x_opt

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

    @staticmethod
    def fit_z_narrow(
        rich_nw: RichNetwork,
        show_plot: int = 1,
        show_fit: int = 1,
        target_f: Optional[float] = None,
        range_f: Optional[float] = None,
    ):
        """Fit simple LCR models in a narrow frequency band and optionally plot.

        Returns a tuple of fitted component values for each port and coupling.
        """

        def series_lcr_xself(x, ls, cs):
            return 2 * math.pi * x * ls - 1 / (2 * math.pi * x * cs)

        def series_lcr_rself(x, r):
            return 0 * x + r

        def series_lcr_xm(x, lm):
            return 2 * math.pi * x * lm

        popt, _ = curve_fit(
            series_lcr_xself,
            rich_nw.nw.frequency.f[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
            ],
            rich_nw.nw.z[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 0, 0
            ].imag,
            p0=np.asarray([1e-6, 1e-9]),
            maxfev=10000,
        )
        ls1, cs1 = popt
        r2 = metrics.r2_score(
            rich_nw.nw.z[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 0, 0
            ].imag,
            series_lcr_xself(
                rich_nw.nw.frequency.f[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
                ],
                ls1,
                cs1,
            ),
        )
        if show_fit is True:
            print("R2 for fitting Ls1, Cs1: %f" % (r2))

        popt, _ = curve_fit(
            series_lcr_rself,
            rich_nw.nw.frequency.f[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
            ],
            rich_nw.nw.z[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 0, 0
            ].real,
            p0=np.asarray([1]),
            maxfev=10000,
        )
        rs1 = popt

        if rich_nw.nw.nports == 2:
            popt, _ = curve_fit(
                series_lcr_xself,
                rich_nw.nw.frequency.f[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
                ],
                rich_nw.nw.z[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 1, 1
                ].imag,
                p0=np.asarray([1e-6, 1e-9]),
                maxfev=10000,
            )
            ls2, cs2 = popt

            r2 = metrics.r2_score(
                rich_nw.nw.z[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 1, 1
                ].imag,
                series_lcr_xself(
                    rich_nw.nw.frequency.f[
                        rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
                    ],
                    ls2,
                    cs2,
                ),
            )
            if show_fit == 1:
                print("R2 for fitting Ls2, Cs2: %f" % (r2))

            popt, _ = curve_fit(
                series_lcr_rself,
                rich_nw.nw.frequency.f[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
                ],
                rich_nw.nw.z[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 1, 1
                ].real,
                p0=np.asarray([1]),
                maxfev=10000,
            )
            rs2 = popt

            popt, _ = curve_fit(
                series_lcr_xm,
                rich_nw.nw.frequency.f[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
                ],
                rich_nw.nw.z[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 0, 1
                ].imag,
                p0=np.asarray([1e-6]),
                maxfev=10000,
            )
            lm = popt
            r2 = metrics.r2_score(
                rich_nw.nw.z[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 0, 1
                ].imag,
                series_lcr_xm(
                    rich_nw.nw.frequency.f[
                        rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
                    ],
                    lm,
                ),
            )
            print('R2 for fitting Lm: %f' % (r2))

        if show_fit == 1:
            print(
                "Re(Z11): %.2e\nIm(Z11): %.2e\n"
                % (
                    rich_nw.nw.z[rich_nw.target_f_index, 0, 0].real,
                    rich_nw.nw.z[rich_nw.target_f_index, 0, 0].imag,
                )
            )

            if rich_nw.nw.nports == 2:
                print(
                    "Re(Z22): %.2e\nIm(Z22) %.2e\n"
                    % (
                        rich_nw.nw.z[rich_nw.target_f_index, 1, 1].real,
                        rich_nw.nw.z[rich_nw.target_f_index, 1, 1].imag,
                    )
                )

            logger.debug("Fitting values assuming a pair of series LCR resonators\n")
            print(
                f"Ls1: {ls1}, Cs1: {cs1}, Rs1: {rs1}, f_1: {1 / (2 * np.pi * np.sqrt(ls1 * cs1))}, Q_1 (approximate, @{rich_nw.target_f} Hz): {(2 * np.pi * rich_nw.target_f * ls1 / rs1)[0]}"
            )
            if rich_nw.nw.nports == 2:
                print(
                    f"Ls2: {ls2}, Cs2: {cs2}, Rs2: {rs2}, f_2: {1 / (2 * np.pi * np.sqrt(ls2 * cs2))}, Q_2 (approximate, @{rich_nw.target_f} Hz): {(2 * np.pi * rich_nw.target_f * ls2 / rs2)[0]}"
                )
                print(f"Lm: {lm}, km: {lm / np.sqrt(ls1 * ls2)}")

        if show_plot == 1:
            if rich_nw.nw.nports == 1:
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
                        ls1,
                        cs1,
                    ),
                    label="imag(z) fitting",
                    color="green",
                )
                fig.tight_layout()

            if rich_nw.nw.nports == 2:
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
                        ls1,
                        cs1,
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
                        ls2,
                        cs2,
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
                        lm,
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
                        lm,
                    ),
                    label="imag(z) fitting",
                    color="green",
                )

                fig.tight_layout()

        return ls1, cs1, rs1, ls2, cs2, rs2, lm

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

        ls1, _, rs1, ls2, _, rs2, _ = nw_tools.fit_z_narrow(
            rich_nw, show_plot=0, show_fit=0
        )
        max_f_plot, max_eff_opt, max_r_opt, max_x_opt = nw_tools.analyze_efficiency(
            rich_nw, rx_port=rx_port, show_plot=0, show_data=0
        )

        max_w_plot = 2 * np.pi * max_f_plot
        if rx_port == 1:
            lrx = ls1
        elif rx_port == 2:
            lrx = ls2
        else:
            raise ValueError("set rx_port parameter to 1 or 2")

        print("-----------Analysis results-----------")
        print("Target frequency: %.3e" % (max_f_plot))
        print("Maximum efficiency: %.2f" % (max_eff_opt))
        print("Receiver inductance: %.2e" % (lrx))
        print("Optimum load: %.2f" % (max_r_opt))
        print("Target Rload: %.2f\n" % (rload))

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
        min_rez,
        min_imz,
        max_rez,
        max_imz,
        step_rez,
        step_imz,
        input_voltage,
        *,
        rx_port=2,
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
        min_rez: float
            The minimum real(Zload).
        min_imz: float
            The minimum imaginary(Zload).
        max_rez: float
            The maximum real(Zload).
        max_imz: float
            The maximum imaginary(Zload).
        step_rez: float
            The step size for the real(Zload).
        step_imz: float
            The step size for the imaginary(Zload).
        input_voltage: float
            The input voltage.
        rx_port: int
            The port to analyze.
        target_f: Optional[float]
            The target frequency.
        range_f: Optional[float]
            The range of the target frequency.

        Raises
        ------
        ValueError
            If the target frequency is not found within the specified range.
        TypeError
            If the network is not a RichNetwork.
        ValueError
            If the rx_port is not 1 or 2.

        """
        rich_nw = override_frange(rich_nw, target_f=target_f, range_f=range_f)

        rez_list = np.arange(min_rez, max_rez, step_rez)
        imz_list = np.arange(min_imz, max_imz, step_imz)
        eff_grid = np.zeros((rez_list.size, imz_list.size))
        Pin = np.zeros((rez_list.size, imz_list.size))
        Pout = np.zeros((rez_list.size, imz_list.size))

        if rx_port == 2:
            Z11 = rich_nw.nw.z[rich_nw.target_f_index, 0, 0]
            Z22 = rich_nw.nw.z[rich_nw.target_f_index, 1, 1]
        elif rx_port == 1:
            Z11 = rich_nw.nw.z[rich_nw.target_f_index, 1, 1]
            Z22 = rich_nw.nw.z[rich_nw.target_f_index, 0, 0]
        else:
            raise ValueError("set rx_port to 1 or 2.")

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
