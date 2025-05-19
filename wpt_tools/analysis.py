"""
Analysis code for wireless power transfer systems.
"""

import math
import sys
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import skrf as rf
from scipy.optimize import curve_fit, fmin

import wpt_tools.logger

wpt_tools.logger.init_logger("mymodule.log")  # Initialize logger
logger = getLogger(__name__)  # Get a logger for this module


class nw_tools:
    """
    Class for analyzing wireless power transfer systems based on touchstone (`.snp`) files.

    Attributes
    ----------
    nw : skrf.network.Network
        Network object representing the touchstone file.
    f_narrow_index_start : type
        (Description of f_narrow_index_start)
    f_narrow_index_stop : type
        (Description of f_narrow_index_stop)
    target_f_index : type
        (Description of target_f_index)
    sweeppoint : numpy.ndarray
        Sweep points in the frequency range of the network.
    range_f : type
        (Description of range_f)
    target_f : type
        (Description of target_f)

    """

    def __init__(self):
        """
        Initialize the nw_tools class.
        """
        self.nw = None
        self.f_narrow_index_start = None
        self.f_narrow_index_stop = None
        self.target_f_index = None
        self.sweeppoint = None
        self.range_f = None
        self.target_f = None

    def import_touchstone(self, filename: str) -> None:
        """
        Load touchstone file into the network.

        Parameters
        ----------
        filename : str
            Name of the touchstone file.

        Raises
        ------
        ValueError
            If filename is not provided or the file does not exist.

        """
        if filename is None:
            raise ValueError("Filename must be provided")

        try:
            self.nw = rf.Network(filename)
        except FileNotFoundError:
            raise ValueError(f"File '{filename}' not found")

        self.sweeppoint = np.size(self.nw.frequency.f)
        logger.info(f"Loaded touchstone file: {filename}")

    def set_f_target_range(self, target_f: float, range_f: float) -> None:
        """
        Set the target frequency range for the analysis.

        This attempts to find an index range close to the target frequency with a given range. The closest
        frequency is stored in `self.target_f_index`.

        Parameters
        ----------
        target_f : float
            Target frequency.
        range_f : float
            The frequency range centered around the target frequency.

        Note
        ----
        This method relies on previously imported frequencies via the `import_touchstone` method. If no
        frequency data is found, this function does nothing.

        """
        self.target_f = target_f
        self.range_f = range_f

        # Initialize start and stop indices to the sweep point extremes.
        self.f_narrow_index_start = self.sweeppoint
        self.f_narrow_index_stop = 0

        d_target_f = self.range_f

        for f_index in range(self.sweeppoint):
            # Check if the frequency indexed by f_index falls within the target range.
            if abs(target_f - self.nw.frequency.f[f_index]) < range_f / 2:
                # Update start and stop indices.
                if self.f_narrow_index_start > f_index:
                    self.f_narrow_index_start = f_index
                if self.f_narrow_index_stop < f_index:
                    self.f_narrow_index_stop = f_index

                # Preparing to update the target index if this frequency is closest to the target.
                f_temp = self.nw.frequency.f[f_index]

                # Update the target index if this is the closest frequency to the target.
                if abs(target_f - f_temp) < d_target_f:
                    d_target_f = abs(target_f - f_temp)
                    self.target_f_index = f_index

    def efficiency_load_analysis(
        self, rx_port: int = 2, show_plot: bool = True, show_data: bool = True
    ):
        """
        Perform efficiency and optimal load analysis (for general 2-port networks).

        The function is unstable when far from the resonant frequency.
        Reference: Y. Narusue, et al., "Load optimization factors for analyzing the
        efficiency of wireless power transfer systems using two-port network parameters,"
        IEICE ELEX, 2020.

        Parameters
        ----------
        rx_port : int, optional
            The port to receive on, either 1 or 2 (default is 2).
        show_plot : int, optional
            Whether to show a plot of the analysis (default is 1).
        show_data : int, optional
            Whether to logger.info data from the analysis (default is 1).

        Returns
        -------
        max_f_plot : float
            Frequency at maximum efficiency.
        max_eff_opt : float
            Maximum efficiency.
        max_r_opt : float
            Optimum resistance for maximum efficiency.
        max_x_opt : float
            Optimum reactance for maximum efficiency.

        Raises
        ------
        SystemExit
            If the method 'set_f_target_range' was not executed before this operation.

        """
        f_plot = []
        r_det = []
        kq2 = []
        r_opt = []
        x_opt = []
        eff_opt = []

        max_eff_opt = 0
        max_x_opt = 0
        max_r_opt = 0

        if self.target_f is None:
            raise ValueError("execute set_f_target_range() before this operation")

        for f_index in range(self.sweeppoint):
            if abs(self.target_f - self.nw.frequency.f[f_index]) < self.range_f / 2:
                if rx_port == 2:
                    Z11 = self.nw.z[f_index, 0, 0]
                    Z22 = self.nw.z[f_index, 1, 1]
                elif rx_port == 1:
                    Z11 = self.nw.z[f_index, 1, 1]
                    Z22 = self.nw.z[f_index, 0, 0]
                else:
                    logger.info("set rx_port to 1 or 2.")
                    sys.exit()
                Zm = self.nw.z[f_index, 0, 1]
                f_temp = self.nw.frequency.f[f_index]
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
            axs[0].axvline(self.target_f, color="gray", lw=1)

            axs[1].plot(f_plot, r_opt)
            axs[1].set_title(r"Optimum Re($Z_\mathrm{load}$)")
            axs[1].set_xlabel("Frequency")
            axs[1].set_ylabel(r"Optimum Re($Z_\mathrm{load}$) ($\Omega$)")
            axs[1].axvline(self.target_f, color="gray", lw=1)

            axs[2].plot(f_plot, x_opt)
            axs[2].set_title(r"Optimum Im($Z_\mathrm{load}$)")
            axs[2].set_xlabel("Frequency")
            axs[2].set_ylabel(r"Optimum Im($Z_\mathrm{load}$) ($\Omega$)")
            axs[2].axvline(self.target_f, color="gray", lw=1)

            fig.tight_layout()

        if show_data is True:
            logger.info("Target frequency: %.3e" % (max_f_plot))
            logger.info("Maximum efficiency: %.2f" % (max_eff_opt))
            logger.info("Optimum Re(Zload): %.2f" % (max_r_opt))
            logger.info("Optimum Im(Zload): %.2f" % (max_x_opt))

        return max_f_plot, max_eff_opt, max_r_opt, max_x_opt

    # Plot Z-parameters (full-range)
    def plot_z_full(self):
        """
        Plot the Z-parameters (full-range).
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
                    self.nw.frequency.f,
                    self.nw.z[:, rx_port - 1, tx_port - 1].real,
                    label="real(z)",
                )
                (pi[plot_index],) = twin[plot_index].plot(
                    self.nw.frequency.f,
                    self.nw.z[:, rx_port - 1, tx_port - 1].imag,
                    "r-",
                    label="imag(z)",
                )
                axs[plot_index].set_xlabel("frequency")
                axs[plot_index].set_ylabel(
                    "re(Z" + str(rx_port) + str(tx_port) + r") ($\Omega$)"
                )
                twin[plot_index].set_ylabel(
                    "im(Z" + str(rx_port) + str(tx_port) + r") ($\Omega$)"
                )
                axs[plot_index].yaxis.label.set_color(pr[plot_index].get_color())
                twin[plot_index].yaxis.label.set_color(pi[plot_index].get_color())
                if self.target_f is not None:
                    axs[plot_index].axvline(self.target_f, color="gray", lw=1)
                axs[plot_index].set_ylim(
                    (
                        -abs(self.nw.z[:, rx_port - 1, tx_port - 1].real).max(),
                        abs(self.nw.z[:, rx_port - 1, tx_port - 1].real).max(),
                    )
                )
                twin[plot_index].set_ylim(
                    (
                        -abs(self.nw.z[:, rx_port - 1, tx_port - 1].imag).max(),
                        abs(self.nw.z[:, rx_port - 1, tx_port - 1].imag).max(),
                    )
                )
                axs[plot_index].axhline(0, color="gray", lw=1)
        fig.tight_layout()

    # Curve-fitting and Z-matrix plot (narrow-range)
    def plot_z_narrow_fit(self, show_plot=1, show_fit=1):
        """
        Curve-fitting and Z-matrix plot (narrow-range).

        Parameters
        ----------
        show_plot : bool, optional
            Whether to show a plot of the analysis (default is 1).
        show_fit : bool, optional
            Whether to logger.info data from the analysis (default is 1).

        """
        if self.target_f is None:
            logger.info("execute set_f_target_range() before this operation")
            sys.exit()

        def series_lcr_xself(x, ls, cs):
            return 2 * math.pi * x * ls - 1 / (2 * math.pi * x * cs)

        def series_lcr_rself(x, r):
            return 0 * x + r

        def series_lcr_xm(x, lm):
            return 2 * math.pi * x * lm

        popt, _ = curve_fit(
            series_lcr_xself,
            self.nw.frequency.f[self.f_narrow_index_start : self.f_narrow_index_stop],
            self.nw.z[self.f_narrow_index_start : self.f_narrow_index_stop, 0, 0].imag,
            p0=np.asarray([1e-6, 1e-9]),
            maxfev=10000,
        )
        ls1, cs1 = popt
        r2 = metrics.r2_score(
            self.nw.z[self.f_narrow_index_start : self.f_narrow_index_stop, 0, 0].imag,
            series_lcr_xself(
                self.nw.frequency.f[
                    self.f_narrow_index_start : self.f_narrow_index_stop
                ],
                ls1,
                cs1,
            ),
        )
        if show_fit is True:
            logger.info("R2 for fitting Ls1, Cs1: %f" % (r2))

        popt, _ = curve_fit(
            series_lcr_rself,
            self.nw.frequency.f[self.f_narrow_index_start : self.f_narrow_index_stop],
            self.nw.z[self.f_narrow_index_start : self.f_narrow_index_stop, 0, 0].real,
            p0=np.asarray([1]),
            maxfev=10000,
        )
        rs1 = popt

        if self.nw.nports == 2:
            popt, _ = curve_fit(
                series_lcr_xself,
                self.nw.frequency.f[
                    self.f_narrow_index_start : self.f_narrow_index_stop
                ],
                self.nw.z[
                    self.f_narrow_index_start : self.f_narrow_index_stop, 1, 1
                ].imag,
                p0=np.asarray([1e-6, 1e-9]),
                maxfev=10000,
            )
            ls2, cs2 = popt

            r2 = metrics.r2_score(
                self.nw.z[
                    self.f_narrow_index_start : self.f_narrow_index_stop, 1, 1
                ].imag,
                series_lcr_xself(
                    self.nw.frequency.f[
                        self.f_narrow_index_start : self.f_narrow_index_stop
                    ],
                    ls2,
                    cs2,
                ),
            )
            if show_fit == 1:
                logger.info("R2 for fitting Ls2, Cs2: %f" % (r2))

            popt, _ = curve_fit(
                series_lcr_rself,
                self.nw.frequency.f[
                    self.f_narrow_index_start : self.f_narrow_index_stop
                ],
                self.nw.z[
                    self.f_narrow_index_start : self.f_narrow_index_stop, 1, 1
                ].real,
                p0=np.asarray([1]),
                maxfev=10000,
            )
            rs2 = popt

            popt, _ = curve_fit(
                series_lcr_xm,
                self.nw.frequency.f[
                    self.f_narrow_index_start : self.f_narrow_index_stop
                ],
                self.nw.z[
                    self.f_narrow_index_start : self.f_narrow_index_stop, 0, 1
                ].imag,
                p0=np.asarray([1e-6]),
                maxfev=10000,
            )
            lm = popt
            r2 = metrics.r2_score(
                self.nw.z[
                    self.f_narrow_index_start : self.f_narrow_index_stop, 0, 1
                ].imag,
                series_lcr_xm(
                    self.nw.frequency.f[
                        self.f_narrow_index_start : self.f_narrow_index_stop
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
                    self.nw.z[self.target_f_index, 0, 0].real,
                    self.nw.z[self.target_f_index, 0, 0].imag,
                )
            )

            if self.nw.nports == 2:
                logger.info(
                    "Re(Z22): %.2e\nIm(Z22) %.2e\n"
                    % (
                        self.nw.z[self.target_f_index, 1, 1].real,
                        self.nw.z[self.target_f_index, 1, 1].imag,
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
                    self.target_f,
                    2 * np.pi * self.target_f * ls1 / rs1,
                )
            )
            if self.nw.nports == 2:
                logger.info(
                    "Ls2: %.2e, Cs2: %.2e, Rs2: %.2e, f_2: %.3e, Q_2 (approximate, @%.3e Hz): %.2e"
                    % (
                        ls2,
                        cs2,
                        rs2,
                        1 / (2 * np.pi * np.sqrt(ls2 * cs2)),
                        self.target_f,
                        2 * np.pi * self.target_f * ls2 / rs2,
                    )
                )
                logger.info("Lm: %.2e, km: %.3f" % (lm, lm / np.sqrt(ls1 * ls2)))

        if show_plot == 1:
            if self.nw.nports == 1:
                fig, axs = plt.subplots(1, 1, figsize=(5, 3.5))
                twin = ["init"] * 1
                pr = ["init"] * 1
                pi = ["init"] * 1

                axs.set_title("Z11")
                twin = axs.twinx()
                (pr[0],) = axs.plot(
                    self.nw.frequency.f[
                        self.f_narrow_index_start : self.f_narrow_index_stop
                    ],
                    self.nw.z[
                        self.f_narrow_index_start : self.f_narrow_index_stop, 0, 0
                    ].real,
                    label="real(z)",
                    lw=3,
                )
                (pi[0],) = twin.plot(
                    self.nw.frequency.f[
                        self.f_narrow_index_start : self.f_narrow_index_stop
                    ],
                    self.nw.z[
                        self.f_narrow_index_start : self.f_narrow_index_stop, 0, 0
                    ].imag,
                    "r-",
                    label="imag(z)",
                    lw=3,
                )
                axs.set_xlabel("frequency")
                axs.set_ylabel(r"re(Z11) ($\Omega$)")
                twin.set_ylabel(r"im(Z) ($\Omega$)")
                axs.yaxis.label.set_color(pr[0].get_color())
                twin.yaxis.label.set_color(pi[0].get_color())
                axs.axvline(self.target_f, color="gray", lw=1)
                axs.set_ylim(
                    (
                        -1.5
                        * abs(
                            self.nw.z[
                                self.f_narrow_index_start : self.f_narrow_index_stop,
                                0,
                                0,
                            ].real
                        ).max(),
                        1.5
                        * abs(
                            self.nw.z[
                                self.f_narrow_index_start : self.f_narrow_index_stop,
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
                            self.nw.z[
                                self.f_narrow_index_start : self.f_narrow_index_stop,
                                0,
                                0,
                            ].imag
                        ).max(),
                        1.5
                        * abs(
                            self.nw.z[
                                self.f_narrow_index_start : self.f_narrow_index_stop,
                                0,
                                0,
                            ].imag
                        ).max(),
                    )
                )
                axs.axhline(0, color="gray", lw=1)

                twin.plot(
                    self.nw.frequency.f[
                        self.f_narrow_index_start : self.f_narrow_index_stop
                    ],
                    series_lcr_xself(
                        self.nw.frequency.f[
                            self.f_narrow_index_start : self.f_narrow_index_stop
                        ],
                        ls1,
                        cs1,
                    ),
                    label="imag(z) fitting",
                    color="green",
                )
                fig.tight_layout()

            if self.nw.nports == 2:
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
                            self.nw.frequency.f[
                                self.f_narrow_index_start : self.f_narrow_index_stop
                            ],
                            self.nw.z[
                                self.f_narrow_index_start : self.f_narrow_index_stop,
                                rx_port - 1,
                                tx_port - 1,
                            ].real,
                            label="real(z)",
                            lw=3,
                        )
                        (pi[plot_index],) = twin[plot_index].plot(
                            self.nw.frequency.f[
                                self.f_narrow_index_start : self.f_narrow_index_stop
                            ],
                            self.nw.z[
                                self.f_narrow_index_start : self.f_narrow_index_stop,
                                rx_port - 1,
                                tx_port - 1,
                            ].imag,
                            "r-",
                            label="imag(z)",
                            lw=3,
                        )
                        axs[plot_index].set_xlabel("frequency")
                        axs[plot_index].set_ylabel(
                            "re(Z" + str(rx_port) + str(tx_port) + r") ($\Omega$)"
                        )
                        twin[plot_index].set_ylabel(
                            "im(Z" + str(rx_port) + str(tx_port) + r") ($\Omega$)"
                        )
                        axs[plot_index].yaxis.label.set_color(
                            pr[plot_index].get_color()
                        )
                        twin[plot_index].yaxis.label.set_color(
                            pi[plot_index].get_color()
                        )
                        axs[plot_index].axvline(self.target_f, color="gray", lw=1)
                        axs[plot_index].set_ylim(
                            (
                                -1.5
                                * abs(
                                    self.nw.z[
                                        self.f_narrow_index_start : self.f_narrow_index_stop,
                                        rx_port - 1,
                                        tx_port - 1,
                                    ].real
                                ).max(),
                                1.5
                                * abs(
                                    self.nw.z[
                                        self.f_narrow_index_start : self.f_narrow_index_stop,
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
                                    self.nw.z[
                                        self.f_narrow_index_start : self.f_narrow_index_stop,
                                        rx_port - 1,
                                        tx_port - 1,
                                    ].imag
                                ).max(),
                                1.5
                                * abs(
                                    self.nw.z[
                                        self.f_narrow_index_start : self.f_narrow_index_stop,
                                        rx_port - 1,
                                        tx_port - 1,
                                    ].imag
                                ).max(),
                            )
                        )
                        axs[plot_index].axhline(0, color="gray", lw=1)

                twin[0].plot(
                    self.nw.frequency.f[
                        self.f_narrow_index_start : self.f_narrow_index_stop
                    ],
                    series_lcr_xself(
                        self.nw.frequency.f[
                            self.f_narrow_index_start : self.f_narrow_index_stop
                        ],
                        ls1,
                        cs1,
                    ),
                    label="imag(z) fitting",
                    color="green",
                )
                twin[3].plot(
                    self.nw.frequency.f[
                        self.f_narrow_index_start : self.f_narrow_index_stop
                    ],
                    series_lcr_xself(
                        self.nw.frequency.f[
                            self.f_narrow_index_start : self.f_narrow_index_stop
                        ],
                        ls2,
                        cs2,
                    ),
                    label="imag(z) fitting",
                    color="green",
                )
                twin[1].plot(
                    self.nw.frequency.f[
                        self.f_narrow_index_start : self.f_narrow_index_stop
                    ],
                    series_lcr_xm(
                        self.nw.frequency.f[
                            self.f_narrow_index_start : self.f_narrow_index_stop
                        ],
                        lm,
                    ),
                    label="imag(z) fitting",
                    color="green",
                )
                twin[2].plot(
                    self.nw.frequency.f[
                        self.f_narrow_index_start : self.f_narrow_index_stop
                    ],
                    series_lcr_xm(
                        self.nw.frequency.f[
                            self.f_narrow_index_start : self.f_narrow_index_stop
                        ],
                        lm,
                    ),
                    label="imag(z) fitting",
                    color="green",
                )

                fig.tight_layout()

        return ls1, cs1, rs1, ls2, cs2, rs2, lm

    def rxc_filter_calc(self, rx_port, rload, c_network="CpCsRl"):
        """
        Calculate the receiver capacitor filter optimized for a given load impedance.

        Parameters
        ----------
        rx_port : int
            The port number of the receiver (1 or 2).

        rload : float
            The load impedance.

        c_network : str, optional
            The network type to use for the calculation (default is 'CpCsRl').

        """
        ls1, _, rs1, ls2, _, rs2, _ = self.plot_z_narrow_fit(show_plot=0, show_fit=0)
        max_f_plot, max_eff_opt, max_r_opt, max_x_opt = self.efficiency_load_analysis(
            rx_port=rx_port, show_plot=0, show_data=0
        )

        max_w_plot = 2 * np.pi * max_f_plot
        if rx_port == 1:
            lrx = ls1
        elif rx_port == 2:
            lrx = ls2
        else:
            logger.info("set rx_port parameter to 1 or 2")
            sys.exit()

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

    def optimal_load_plot(
        self,
        min_rez,
        min_imz,
        max_rez,
        max_imz,
        step_rez,
        step_imz,
        input_voltage,
        rx_port=2,
    ):
        """
        Plot the optimal load for a given range of load impedance.

        Parameters
        ----------
        min_rez : float
            The minimum real part of the load impedance.
        min_imz : float
            The minimum imaginary part of the load impedance.
        max_rez : float
            The maximum real part of the load impedance.
        max_imz : float
            The maximum imaginary part of the load impedance.
        step_rez : float
            The step size for the real part of the load impedance.
        step_imz : float
            The step size for the imaginary part of the load impedance.
        input_voltage : float
            The input voltage of the system.
        rx_port : int, optional
            The port number of the receiver (1 or 2).

        """
        # Optimal load visualization
        # Imura, "Wireless Power Transfer: Using Magnetic and Electric Resonance Coupling Techniques," Springer Singapore 2020.
        rez_list = np.arange(min_rez, max_rez, step_rez)
        imz_list = np.arange(min_imz, max_imz, step_imz)
        eff_grid = np.zeros((rez_list.size, imz_list.size))
        Pin = np.zeros((rez_list.size, imz_list.size))
        Pout = np.zeros((rez_list.size, imz_list.size))

        if rx_port == 2:
            Z11 = self.nw.z[self.target_f_index, 0, 0]
            Z22 = self.nw.z[self.target_f_index, 1, 1]
        elif rx_port == 1:
            Z11 = self.nw.z[self.target_f_index, 1, 1]
            Z22 = self.nw.z[self.target_f_index, 0, 0]
        else:
            logger.info("set rx_port to 1 or 2.")
            sys.exit()

        Zm = self.nw.z[self.target_f_index, 0, 1]

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
            + format(self.nw.frequency.f[self.target_f_index], "3.2e")
            + " Hz"
        )
        axs[0].set_ylabel(r"Re($Z_{\mathrm{load}}$)")
        axs[0].set_xlabel(r"Im($Z_{\mathrm{load}}$)")

        c = axs[1].pcolor(
            imz_list, rez_list, Pin, cmap="hot", vmin=0, vmax=Pin.max(), shading="auto"
        )
        fig.colorbar(c, ax=axs[1])
        axs[1].set_title(
            "Input Power (W) @ "
            + format(self.nw.frequency.f[self.target_f_index], "3.2e")
            + " Hz"
        )
        axs[1].set_ylabel(r"Re($Z_{\mathrm{load}}$)")
        axs[1].set_xlabel(r"Im($Z_{\mathrm{load}}$)")

        c = axs[2].pcolor(
            imz_list, rez_list, Pout, cmap="hot", vmin=0, vmax=Pin.max(), shading="auto"
        )
        fig.colorbar(c, ax=axs[2])
        axs[2].set_title(
            "Output Power (W) @ "
            + format(self.nw.frequency.f[self.target_f_index], "3.2e")
            + " Hz"
        )
        axs[2].set_ylabel(r"Re($Z_{\mathrm{load}}$)")
        axs[2].set_xlabel(r"Im($Z_{\mathrm{load}}$)")

        fig.tight_layout()
