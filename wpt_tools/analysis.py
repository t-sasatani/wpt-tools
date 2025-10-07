"""
Analysis code for wireless power transfer systems.
"""

import dataclasses
import math
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import skrf as rf
from scipy.optimize import curve_fit, fmin

import wpt_tools.logger

wpt_tools.logger.init_logger("mymodule.log")  # Initialize logger
logger = getLogger(__name__)  # Get a logger for this module


@dataclasses.dataclass
class nw_with_config:
    """
    A dataclass for analyzing wireless power transfer systems.

    Parameters
    ----------
    nw: rf.Network
        The network to analyze.
    f_narrow_index_start: Optional[int]
        The start index of the narrowband range.
    f_narrow_index_stop: Optional[int]
        The stop index of the narrowband range.
    target_f_index: Optional[int]
        The index of the target frequency.
    sweeppoint: Optional[int]
        The number of points in the sweep.
    range_f: Optional[float]
        The range of the target frequency.
    target_f: float
        The target frequency.

    """

    nw: rf.Network
    f_narrow_index_start: Optional[int] = None
    f_narrow_index_stop: Optional[int] = None
    target_f_index: Optional[int] = None
    sweeppoint: Optional[int] = None
    range_f: Optional[float] = None
    target_f: Optional[float] = None

    @classmethod
    def from_touchstone(cls, source: Union[str, Path, rf.Network]) -> "nw_with_config":
        """
        Create a nw_with_config instance from a touchstone file or an rf.Network object.
        """
        if isinstance(source, str):
            nw = rf.Network(source)
        elif isinstance(source, Path):
            nw = rf.Network(str(source))
        elif isinstance(source, rf.Network):
            nw = source
        return cls(nw=nw)

    def set_f_target_range(self, target_f: float, range_f: float) -> None:
        """
        Set the target frequency and range for the analysis.

        Parameters
        ----------
        target_f: float
            The target frequency.
        range_f: float
            The range of the target frequency.

        """
        self.target_f = float(target_f)
        self.range_f = float(range_f)
        if self.nw is None:
            raise ValueError("Network is not loaded")
        if self.sweeppoint is None:
            self.sweeppoint = int(np.size(self.nw.frequency.f))

        self.f_narrow_index_start = int(self.sweeppoint)
        self.f_narrow_index_stop = 0

        d_target_f = self.range_f
        target_index: Optional[int] = None
        for f_index in range(self.sweeppoint):
            if abs(self.target_f - self.nw.frequency.f[f_index]) < self.range_f / 2:
                if self.f_narrow_index_start > f_index:
                    self.f_narrow_index_start = f_index
                if self.f_narrow_index_stop < f_index:
                    self.f_narrow_index_stop = f_index
                f_temp = self.nw.frequency.f[f_index]
                if abs(self.target_f - f_temp) < d_target_f:
                    d_target_f = abs(self.target_f - f_temp)
                    target_index = f_index
        if target_index is None:
            raise ValueError("Target frequency not found within specified range")
        self.target_f_index = int(target_index)


class nw_tools:
    """
    Stateless utilities for analyzing scikit-rf `rf.Network` objects.
    """

    @staticmethod
    def analyze_efficiency(
        rich_nw: Union[rf.Network, "nw_with_config"],
        *,
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
        rich_nw: Union[rf.Network, "nw_with_config"]
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
        # inline ensure-config
        if isinstance(rich_nw, nw_with_config):
            cfg = rich_nw
            if cfg.sweeppoint is None:
                cfg.sweeppoint = int(np.size(cfg.nw.frequency.f))
            if (
                cfg.f_narrow_index_start is None
                or cfg.f_narrow_index_stop is None
                or cfg.target_f_index is None
            ):
                if target_f is None or range_f is None:
                    raise ValueError(
                        "target_f and range_f are required to compute configuration"
                    )
                cfg.set_f_target_range(target_f=float(target_f), range_f=float(range_f))
        elif isinstance(rich_nw, rf.Network):
            if target_f is None or range_f is None:
                raise ValueError(
                    "target_f and range_f are required when passing rf.Network"
                )
            cfg = nw_with_config(
                nw=rich_nw, sweeppoint=int(np.size(rich_nw.frequency.f))
            )
            cfg.set_f_target_range(target_f=float(target_f), range_f=float(range_f))
        else:
            raise TypeError("nw_or_config must be rf.Network or nw_with_config")

        f_plot = []
        r_det = []
        kq2 = []
        r_opt = []
        x_opt = []
        eff_opt = []

        max_eff_opt = 0
        max_x_opt = 0
        max_r_opt = 0

        for f_index in range(cfg.sweeppoint):
            if abs(cfg.target_f - cfg.nw.frequency.f[f_index]) < cfg.range_f / 2:
                if rx_port == 2:
                    Z11 = cfg.nw.z[f_index, 0, 0]
                    Z22 = cfg.nw.z[f_index, 1, 1]
                elif rx_port == 1:
                    Z11 = cfg.nw.z[f_index, 1, 1]
                    Z22 = cfg.nw.z[f_index, 0, 0]
                else:
                    raise ValueError("set rx_port to 1 or 2.")
                Zm = cfg.nw.z[f_index, 0, 1]
                f_temp = cfg.nw.frequency.f[f_index]
                r_det_temp = Z11.real * Z22.real - Zm.real**2

                kq2_temp = (Zm.real**2 + Zm.imag**2) / r_det_temp
                r_opt_temp = r_det_temp / Z11.real * np.sqrt(1 + kq2_temp)
                x_opt_temp = Zm.real * Zm.imag - Z11.real * Z22.imag / Z11.real
                eff_opt_temp = kq2_temp / (1 + np.sqrt(1 + kq2_temp)) ** 2

                f_plot.append(f_temp)
                r_det.append(r_det_temp)
                kq2.append(kq2_temp)
                r_opt.append(r_opt_temp)
                x_opt.append(x_opt_temp)
                eff_opt.append(eff_opt_temp)

                if max_eff_opt < eff_opt_temp:
                    max_f_plot = f_temp
                    max_eff_opt = eff_opt_temp
                    max_r_opt = r_opt_temp
                    max_x_opt = x_opt_temp

        if show_plot is True:
            fig, axs = plt.subplots(1, 3, figsize=(18, 4))

            axs[0].plot(f_plot, eff_opt)
            axs[0].set_title("Maximum efficiency")
            axs[0].set_xlabel("Frequency")
            axs[0].set_ylabel("Efficiency")
            axs[0].axvline(cfg.target_f, color="gray", lw=1)

            axs[1].plot(f_plot, r_opt)
            axs[1].set_title("Optimum Re(Zload)")
            axs[1].set_xlabel("Frequency")
            axs[1].set_ylabel("Optimum Re(Zload) (Ohm)")
            axs[1].axvline(cfg.target_f, color="gray", lw=1)

            axs[2].plot(f_plot, x_opt)
            axs[2].set_title("Optimum Im(Zload)")
            axs[2].set_xlabel("Frequency")
            axs[2].set_ylabel("Optimum Im(Zload) (Ohm)")
            axs[2].axvline(cfg.target_f, color="gray", lw=1)

            fig.tight_layout()

        if show_data is True:
            logger.info("Target frequency: %.3e" % (max_f_plot))
            logger.info("Maximum efficiency: %.2f" % (max_eff_opt))
            logger.info("Optimum Re(Zload): %.2f" % (max_r_opt))
            logger.info("Optimum Im(Zload): %.2f" % (max_x_opt))

        return max_f_plot, max_eff_opt, max_r_opt, max_x_opt

    @staticmethod
    def plot_z_full(
        rich_nw: Union[rf.Network, "nw_with_config"],
        *,
        target_f: Optional[float] = None,
    ) -> None:
        """
        Plot the full impedance matrix of the network.

        Parameters
        ----------
        rich_nw: Union[rf.Network, "nw_with_config"]
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
        cfg = None
        if isinstance(rich_nw, nw_with_config):
            cfg = rich_nw
            nw = cfg.nw
        else:
            nw = rich_nw

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
                    nw.frequency.f,
                    nw.z[:, rx_port - 1, tx_port - 1].real,
                    label="real(z)",
                )
                (pi[plot_index],) = twin[plot_index].plot(
                    nw.frequency.f,
                    nw.z[:, rx_port - 1, tx_port - 1].imag,
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
                vline_f = (
                    cfg.target_f
                    if cfg is not None and cfg.target_f is not None
                    else target_f
                )
                if vline_f is not None:
                    axs[plot_index].axvline(vline_f, color="gray", lw=1)
                axs[plot_index].set_ylim(
                    (
                        -abs(nw.z[:, rx_port - 1, tx_port - 1].real).max(),
                        abs(nw.z[:, rx_port - 1, tx_port - 1].real).max(),
                    )
                )
                twin[plot_index].set_ylim(
                    (
                        -abs(nw.z[:, rx_port - 1, tx_port - 1].imag).max(),
                        abs(nw.z[:, rx_port - 1, tx_port - 1].imag).max(),
                    )
                )
                axs[plot_index].axhline(0, color="gray", lw=1)
        fig.tight_layout()

    @staticmethod
    def fit_z_narrow(
        rich_nw: Union[rf.Network, "nw_with_config"],
        *,
        show_plot: int = 1,
        show_fit: int = 1,
        target_f: Optional[float] = None,
        range_f: Optional[float] = None,
    ):
        """Fit simple LCR models in a narrow frequency band and optionally plot.

        Returns a tuple of fitted component values for each port and coupling.
        """
        # inline ensure-config
        if isinstance(rich_nw, nw_with_config):
            cfg = rich_nw
            if cfg.sweeppoint is None:
                cfg.sweeppoint = int(np.size(cfg.nw.frequency.f))
            if (
                cfg.f_narrow_index_start is None
                or cfg.f_narrow_index_stop is None
                or cfg.target_f_index is None
            ):
                if target_f is None or range_f is None:
                    raise ValueError(
                        "target_f and range_f are required to compute configuration"
                    )
                cfg.set_f_target_range(target_f=float(target_f), range_f=float(range_f))
        elif isinstance(rich_nw, rf.Network):
            if target_f is None or range_f is None:
                raise ValueError(
                    "target_f and range_f are required when passing rf.Network"
                )
            cfg = nw_with_config(
                nw=rich_nw, sweeppoint=int(np.size(rich_nw.frequency.f))
            )
            cfg.set_f_target_range(target_f=float(target_f), range_f=float(range_f))
        else:
            raise TypeError("nw_or_config must be rf.Network or nw_with_config")

        def series_lcr_xself(x, ls, cs):
            return 2 * math.pi * x * ls - 1 / (2 * math.pi * x * cs)

        def series_lcr_rself(x, r):
            return 0 * x + r

        def series_lcr_xm(x, lm):
            return 2 * math.pi * x * lm

        popt, _ = curve_fit(
            series_lcr_xself,
            cfg.nw.frequency.f[cfg.f_narrow_index_start : cfg.f_narrow_index_stop],
            cfg.nw.z[cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 0, 0].imag,
            p0=np.asarray([1e-6, 1e-9]),
            maxfev=10000,
        )
        ls1, cs1 = popt
        r2 = metrics.r2_score(
            cfg.nw.z[cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 0, 0].imag,
            series_lcr_xself(
                cfg.nw.frequency.f[cfg.f_narrow_index_start : cfg.f_narrow_index_stop],
                ls1,
                cs1,
            ),
        )
        if show_fit is True:
            logger.info("R2 for fitting Ls1, Cs1: %f" % (r2))

        popt, _ = curve_fit(
            series_lcr_rself,
            cfg.nw.frequency.f[cfg.f_narrow_index_start : cfg.f_narrow_index_stop],
            cfg.nw.z[cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 0, 0].real,
            p0=np.asarray([1]),
            maxfev=10000,
        )
        rs1 = popt

        if cfg.nw.nports == 2:
            popt, _ = curve_fit(
                series_lcr_xself,
                cfg.nw.frequency.f[cfg.f_narrow_index_start : cfg.f_narrow_index_stop],
                cfg.nw.z[cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 1, 1].imag,
                p0=np.asarray([1e-6, 1e-9]),
                maxfev=10000,
            )
            ls2, cs2 = popt

            r2 = metrics.r2_score(
                cfg.nw.z[cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 1, 1].imag,
                series_lcr_xself(
                    cfg.nw.frequency.f[
                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                    ],
                    ls2,
                    cs2,
                ),
            )
            if show_fit == 1:
                logger.info("R2 for fitting Ls2, Cs2: %f" % (r2))

            popt, _ = curve_fit(
                series_lcr_rself,
                cfg.nw.frequency.f[cfg.f_narrow_index_start : cfg.f_narrow_index_stop],
                cfg.nw.z[cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 1, 1].real,
                p0=np.asarray([1]),
                maxfev=10000,
            )
            rs2 = popt

            popt, _ = curve_fit(
                series_lcr_xm,
                cfg.nw.frequency.f[cfg.f_narrow_index_start : cfg.f_narrow_index_stop],
                cfg.nw.z[cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 0, 1].imag,
                p0=np.asarray([1e-6]),
                maxfev=10000,
            )
            lm = popt
            r2 = metrics.r2_score(
                cfg.nw.z[cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 0, 1].imag,
                series_lcr_xm(
                    cfg.nw.frequency.f[
                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                    ],
                    lm,
                ),
            )
            # logger.info('R2 for fitting Lm: %f' % (r2))

        if show_fit == 1:
            logger.info("Self impedance at target frequency\n")
            logger.info(
                "Re(Z11): %.2e\nIm(Z11): %.2e\n"
                % (
                    cfg.nw.z[cfg.target_f_index, 0, 0].real,
                    cfg.nw.z[cfg.target_f_index, 0, 0].imag,
                )
            )

            if cfg.nw.nports == 2:
                logger.info(
                    "Re(Z22): %.2e\nIm(Z22) %.2e\n"
                    % (
                        cfg.nw.z[cfg.target_f_index, 1, 1].real,
                        cfg.nw.z[cfg.target_f_index, 1, 1].imag,
                    )
                )

            logger.info("Fitting values assuming a pair of series LCR resonators\n")
            logger.info(
                "Ls1: %.2e, Cs1: %.2e, Rs1: %.2e, f_1: %.3e, Q_1 (approximate, @%.3e Hz): %.2e"
                % (
                    ls1,
                    cs1,
                    rs1,
                    1 / (2 * np.pi * np.sqrt(ls1 * cs1)),
                    cfg.target_f,
                    2 * np.pi * cfg.target_f * ls1 / rs1,
                )
            )
            if cfg.nw.nports == 2:
                logger.info(
                    "Ls2: %.2e, Cs2: %.2e, Rs2: %.2e, f_2: %.3e, Q_2 (approximate, @%.3e Hz): %.2e"
                    % (
                        ls2,
                        cs2,
                        rs2,
                        1 / (2 * np.pi * np.sqrt(ls2 * cs2)),
                        cfg.target_f,
                        2 * np.pi * cfg.target_f * ls2 / rs2,
                    )
                )
                logger.info("Lm: %.2e, km: %.3f" % (lm, lm / np.sqrt(ls1 * ls2)))

        if show_plot == 1:
            if cfg.nw.nports == 1:
                fig, axs = plt.subplots(1, 1, figsize=(5, 3.5))
                twin = ["init"] * 1
                pr = ["init"] * 1
                pi = ["init"] * 1

                axs.set_title("Z11")
                twin = axs.twinx()
                (pr[0],) = axs.plot(
                    cfg.nw.frequency.f[
                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                    ],
                    cfg.nw.z[
                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 0, 0
                    ].real,
                    label="real(z)",
                    lw=3,
                )
                (pi[0],) = twin.plot(
                    cfg.nw.frequency.f[
                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                    ],
                    cfg.nw.z[
                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 0, 0
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
                axs.axvline(cfg.target_f, color="gray", lw=1)
                axs.set_ylim(
                    (
                        -1.5
                        * abs(
                            cfg.nw.z[
                                cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 0, 0
                            ].real
                        ).max(),
                        1.5
                        * abs(
                            cfg.nw.z[
                                cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 0, 0
                            ].real
                        ).max(),
                    )
                )
                twin.set_ylim(
                    (
                        -1.5
                        * abs(
                            cfg.nw.z[
                                cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 0, 0
                            ].imag
                        ).max(),
                        1.5
                        * abs(
                            cfg.nw.z[
                                cfg.f_narrow_index_start : cfg.f_narrow_index_stop, 0, 0
                            ].imag
                        ).max(),
                    )
                )
                axs.axhline(0, color="gray", lw=1)

                twin.plot(
                    cfg.nw.frequency.f[
                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                    ],
                    series_lcr_xself(
                        cfg.nw.frequency.f[
                            cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                        ],
                        ls1,
                        cs1,
                    ),
                    label="imag(z) fitting",
                    color="green",
                )
                fig.tight_layout()

            if cfg.nw.nports == 2:
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
                            cfg.nw.frequency.f[
                                cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                            ],
                            cfg.nw.z[
                                cfg.f_narrow_index_start : cfg.f_narrow_index_stop,
                                rx_port - 1,
                                tx_port - 1,
                            ].real,
                            label="real(z)",
                            lw=3,
                        )
                        (pi[plot_index],) = twin[plot_index].plot(
                            cfg.nw.frequency.f[
                                cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                            ],
                            cfg.nw.z[
                                cfg.f_narrow_index_start : cfg.f_narrow_index_stop,
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
                        axs[plot_index].axvline(cfg.target_f, color="gray", lw=1)
                        axs[plot_index].set_ylim(
                            (
                                -1.5
                                * abs(
                                    cfg.nw.z[
                                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop,
                                        rx_port - 1,
                                        tx_port - 1,
                                    ].real
                                ).max(),
                                1.5
                                * abs(
                                    cfg.nw.z[
                                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop,
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
                                    cfg.nw.z[
                                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop,
                                        rx_port - 1,
                                        tx_port - 1,
                                    ].imag
                                ).max(),
                                1.5
                                * abs(
                                    cfg.nw.z[
                                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop,
                                        rx_port - 1,
                                        tx_port - 1,
                                    ].imag
                                ).max(),
                            )
                        )
                        axs[plot_index].axhline(0, color="gray", lw=1)

                twin[0].plot(
                    cfg.nw.frequency.f[
                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                    ],
                    series_lcr_xself(
                        cfg.nw.frequency.f[
                            cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                        ],
                        ls1,
                        cs1,
                    ),
                    label="imag(z) fitting",
                    color="green",
                )
                twin[3].plot(
                    cfg.nw.frequency.f[
                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                    ],
                    series_lcr_xself(
                        cfg.nw.frequency.f[
                            cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                        ],
                        ls2,
                        cs2,
                    ),
                    label="imag(z) fitting",
                    color="green",
                )
                twin[1].plot(
                    cfg.nw.frequency.f[
                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                    ],
                    series_lcr_xm(
                        cfg.nw.frequency.f[
                            cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                        ],
                        lm,
                    ),
                    label="imag(z) fitting",
                    color="green",
                )
                twin[2].plot(
                    cfg.nw.frequency.f[
                        cfg.f_narrow_index_start : cfg.f_narrow_index_stop
                    ],
                    series_lcr_xm(
                        cfg.nw.frequency.f[
                            cfg.f_narrow_index_start : cfg.f_narrow_index_stop
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
        rich_nw: Union[rf.Network, "nw_with_config"],
        rx_port,
        rload,
        *,
        c_network: str = "CpCsRl",
        target_f: Optional[float] = None,
        range_f: Optional[float] = None,
    ):
        """Compute receiver capacitor values for a target load at the optimal point."""
        # inline ensure-config
        if isinstance(rich_nw, nw_with_config):
            cfg = rich_nw
            if cfg.sweeppoint is None:
                cfg.sweeppoint = int(np.size(cfg.nw.frequency.f))
            if (
                cfg.f_narrow_index_start is None
                or cfg.f_narrow_index_stop is None
                or cfg.target_f_index is None
            ):
                if target_f is None or range_f is None:
                    raise ValueError(
                        "target_f and range_f are required to compute configuration"
                    )
                cfg.set_f_target_range(target_f=float(target_f), range_f=float(range_f))
        elif isinstance(rich_nw, rf.Network):
            if target_f is None or range_f is None:
                raise ValueError(
                    "target_f and range_f are required when passing rf.Network"
                )
            cfg = nw_with_config(
                nw=rich_nw, sweeppoint=int(np.size(rich_nw.frequency.f))
            )
            cfg.set_f_target_range(target_f=float(target_f), range_f=float(range_f))
        else:
            raise TypeError("nw_or_config must be rf.Network or nw_with_config")

        ls1, _, rs1, ls2, _, rs2, _ = nw_tools.fit_z_narrow(
            cfg, show_plot=0, show_fit=0
        )
        max_f_plot, max_eff_opt, max_r_opt, max_x_opt = nw_tools.analyze_efficiency(
            cfg, rx_port=rx_port, show_plot=0, show_data=0
        )

        max_w_plot = 2 * np.pi * max_f_plot
        if rx_port == 1:
            lrx = ls1
        elif rx_port == 2:
            lrx = ls2
        else:
            raise ValueError("set rx_port parameter to 1 or 2")

        logger.info("Target frequency: %.3e" % (max_f_plot))
        logger.info("Maximum efficiency: %.2f" % (max_eff_opt))
        logger.info("Receiver inductance: %.2e" % (lrx))
        logger.info("Optimum load: %.2f" % (max_r_opt))
        logger.info("Target Rload: %.2f\n" % (rload))

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
        rich_nw: Union[rf.Network, "nw_with_config"],
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
        """Plot efficiency, input power and output power over a grid of complex loads."""
        # inline ensure-config
        if isinstance(rich_nw, nw_with_config):
            cfg = rich_nw
            if cfg.sweeppoint is None:
                cfg.sweeppoint = int(np.size(cfg.nw.frequency.f))
            if (
                cfg.f_narrow_index_start is None
                or cfg.f_narrow_index_stop is None
                or cfg.target_f_index is None
            ):
                if target_f is None or range_f is None:
                    raise ValueError(
                        "target_f and range_f are required to compute configuration"
                    )
                cfg.set_f_target_range(target_f=float(target_f), range_f=float(range_f))
        elif isinstance(rich_nw, rf.Network):
            if target_f is None or range_f is None:
                raise ValueError(
                    "target_f and range_f are required when passing rf.Network"
                )
            cfg = nw_with_config(
                nw=rich_nw, sweeppoint=int(np.size(rich_nw.frequency.f))
            )
            cfg.set_f_target_range(target_f=float(target_f), range_f=float(range_f))
        else:
            raise TypeError("nw_or_config must be rf.Network or nw_with_config")

        # Optimal load visualization
        # Imura, "Wireless Power Transfer: Using Magnetic and Electric Resonance Coupling Techniques," Springer Singapore 2020.
        rez_list = np.arange(min_rez, max_rez, step_rez)
        imz_list = np.arange(min_imz, max_imz, step_imz)
        eff_grid = np.zeros((rez_list.size, imz_list.size))
        Pin = np.zeros((rez_list.size, imz_list.size))
        Pout = np.zeros((rez_list.size, imz_list.size))

        if rx_port == 2:
            Z11 = cfg.nw.z[cfg.target_f_index, 0, 0]
            Z22 = cfg.nw.z[cfg.target_f_index, 1, 1]
        elif rx_port == 1:
            Z11 = cfg.nw.z[cfg.target_f_index, 1, 1]
            Z22 = cfg.nw.z[cfg.target_f_index, 0, 0]
        else:
            raise ValueError("set rx_port to 1 or 2.")

        Zm = cfg.nw.z[cfg.target_f_index, 0, 1]

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
            + format(cfg.nw.frequency.f[cfg.target_f_index], "3.2e")
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
            + format(cfg.nw.frequency.f[cfg.target_f_index], "3.2e")
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
            + format(cfg.nw.frequency.f[cfg.target_f_index], "3.2e")
            + " Hz"
        )
        axs[2].set_ylabel("Re(Z_load)")
        axs[2].set_xlabel("Im(Z_load)")

        fig.tight_layout()
