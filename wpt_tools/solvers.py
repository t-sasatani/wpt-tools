"""
Solvers for wpt-tools.
"""

from typing import Literal, Optional

import numpy as np
import sklearn.metrics as metrics
from scipy.optimize import curve_fit

from wpt_tools.data_classes import (
    EfficiencyResults,
    LCRFittingResults,
    RichNetwork,
    ValR2,
    override_frange,
)
from wpt_tools.logger import WPTToolsLogger

logger = WPTToolsLogger().get_logger(__name__)


def series_lcr_xself(x, ls, cs):
    """
    Series LCR model for self reactance.
    """
    return 2 * np.pi * x * ls - 1 / (2 * np.pi * x * cs)


def series_lcr_rself(x, r):
    """
    Series LCR model for self resistance.
    """
    return 0 * x + r


def series_lcr_xm(x, lm):
    """
    Series LCR model for mutual reactance.
    """
    return 2 * np.pi * x * lm


def efficiency_calculator(
    rich_nw: RichNetwork,
    rx_port: Literal[1, 2],
    target_f: Optional[float],
    range_f: Optional[float],
) -> EfficiencyResults:
    """Compute efficiency vectors and maxima.

    Parameters
    ----------
    rich_nw: RichNetwork
        The network to analyze.
    rx_port: Literal[1, 2]
        The port to analyze.
    target_f: Optional[float]
        The target frequency.
    range_f: Optional[float]
        The range of the target frequency.

    Returns
    -------
    EfficiencyResults
        The results of the efficiency solver.

    """
    if rich_nw.target_f is None:
        raise ValueError("Target frequency is not set.")
    if rich_nw.range_f is None:
        raise ValueError("Range frequency is not set.")

    rich_nw = override_frange(rich_nw, target_f=target_f, range_f=range_f)
    results = EfficiencyResults()

    for f_index in range(rich_nw.sweeppoint):
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

            results.f_plot.append(float(f_temp))
            results.r_opt.append(float(r_opt_temp))
            results.x_opt.append(float(x_opt_temp))
            results.eff_opt.append(float(eff_opt_temp))

            if results.max_eff_opt is None or results.max_eff_opt < eff_opt_temp:
                results.max_f_plot = float(f_temp)
                results.max_eff_opt = float(eff_opt_temp)
                results.max_r_opt = float(r_opt_temp)
                results.max_x_opt = float(x_opt_temp)

    return results


def lcr_fitting(
    rich_nw: RichNetwork,
    target_f: Optional[float] = None,
    range_f: Optional[float] = None,
) -> LCRFittingResults:
    """
    Fit the LCR model to the network.
    """
    rich_nw = override_frange(rich_nw, target_f=target_f, range_f=range_f)
    results = LCRFittingResults()

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

    logger.info("R2 for fitting Ls1, Cs1: %f" % (r2))

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
    # Ensure scalar for rs1 value
    rs1 = ValR2(value=float(popt[0]), r2=r2)

    # Store fitted values with R2 for port 1
    results.ls1 = ValR2(value=float(ls1), r2=r2)
    results.cs1 = ValR2(value=float(cs1), r2=r2)
    results.rs1 = rs1

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
        ls2 = ValR2(value=float(popt[0]), r2=r2)
        cs2 = ValR2(value=float(popt[1]), r2=r2)

        r2 = metrics.r2_score(
            rich_nw.nw.z[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 1, 1
            ].imag,
            series_lcr_xself(
                rich_nw.nw.frequency.f[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
                ],
                ls2.value,
                cs2.value,
            ),
        )
        logger.info("R2 for fitting Ls2, Cs2: %f" % (r2))

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
        rs2 = ValR2(value=float(popt[0]), r2=r2)

        results.ls2 = ls2
        results.cs2 = cs2
        results.rs2 = rs2

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
        lm_value = float(popt[0])
        r2 = metrics.r2_score(
            rich_nw.nw.z[
                rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop, 0, 1
            ].imag,
            series_lcr_xm(
                rich_nw.nw.frequency.f[
                    rich_nw.f_narrow_index_start : rich_nw.f_narrow_index_stop
                ],
                lm_value,
            ),
        )
        logger.info("R2 for fitting Lm: %f" % (r2))
        results.lm = ValR2(value=lm_value, r2=r2)

    return results
