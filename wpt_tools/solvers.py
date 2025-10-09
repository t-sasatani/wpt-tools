"""
Solvers for wpt-tools.
"""

from typing import Literal, Optional

import numpy as np
import sklearn.metrics as metrics
from scipy.optimize import curve_fit, fmin

from wpt_tools.data_classes import (
    EfficiencyResults,
    LCRFittingResults,
    OptimalLoadGridResults,
    RichNetwork,
    RXCFilterResults,
    ValR2,
    override_frange,
)
from wpt_tools.logger import WPTToolsLogger

logger = WPTToolsLogger().get_logger(__name__)
r2_threshold = 0.9


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


def compute_load_sweep(
    rich_nw: RichNetwork,
    rez_min: float,
    rez_max: float,
    rez_step: float,
    imz_min: float,
    imz_max: float,
    imz_step: float,
    rx_port: Literal[1, 2],
    input_voltage: Optional[float] = 1,
    target_f: Optional[float] = None,
    range_f: Optional[float] = None,
) -> OptimalLoadGridResults:
    """Compute efficiency, input and output power over a grid (load sweep)."""
    rich_nw = override_frange(rich_nw, target_f=target_f, range_f=range_f)
    if rich_nw.target_f is None:
        raise ValueError("target frequency is not set.")

    rez_list = np.arange(rez_min, rez_max, rez_step)
    imz_list = np.arange(imz_min, imz_max, imz_step)
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

    return OptimalLoadGridResults(
        rez_list=rez_list,
        imz_list=imz_list,
        eff_grid=eff_grid,
        Pin=Pin,
        Pout=Pout,
        target_f=float(rich_nw.nw.frequency.f[rich_nw.target_f_index]),
    )


def compute_rxc_filter(
    rich_nw: RichNetwork,
    rx_port: Literal[1, 2],
    rload: float,
    *,
    c_network: Literal["CpCsRl"] = "CpCsRl",
    target_f: Optional[float] = None,
    range_f: Optional[float] = None,
) -> RXCFilterResults:
    """Compute receiver capacitor values for a target load at the optimal point."""
    rich_nw = override_frange(rich_nw, target_f=target_f, range_f=range_f)

    fit = lcr_fitting(rich_nw, target_f=target_f, range_f=range_f)
    eff = efficiency_calculator(
        rich_nw, rx_port=rx_port, target_f=target_f, range_f=range_f
    )

    max_f_plot = eff.max_f_plot
    max_r_opt = eff.max_r_opt
    max_x_opt = eff.max_x_opt

    max_w_plot = 2 * np.pi * max_f_plot
    # Prefer port 2 inductance only if a valid fitted value exists
    if (
        rx_port == 2
        and hasattr(fit, "ls2")
        and isinstance(fit.ls2, ValR2)
        and fit.ls2 is not ValR2
        and fit.ls2.value is not None
    ):
        lrx = float(fit.ls2.value)
    else:
        lrx = float(fit.ls1.value)

    def Z(params):
        if c_network == "CpCsRl":
            cp, cs = params
            return (
                1
                / ((1j * max_w_plot * cp) + 1 / ((1 / (1j * max_w_plot * cs) + rload)))
                + 1j * max_w_plot * lrx
            )
        raise NotImplementedError(f"Unsupported c_network: {c_network}")

    def Zerror(params):
        Zp = Z(params)
        # Legacy behavior: match Re(Z) to Ropt and drive Im(Z) to 0
        return np.linalg.norm([Zp.real - max_r_opt, Zp.imag])

    sol = fmin(Zerror, np.array([100e-12, 100e-12]), xtol=1e-9, ftol=1e-9)
    logger.info(sol)
    cp, cs = float(sol[0]), float(sol[1])

    return RXCFilterResults(
        cp=cp,
        cs=cs,
        rload=float(rload),
        max_r_opt=float(max_r_opt),
        max_x_opt=float(max_x_opt),
        max_f_plot=float(max_f_plot),
        lrx=float(lrx),
    )


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
    if r2 < r2_threshold:
        logger.warning(f"R2 for fitting Ls1, Cs1 is {r2}")

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
        if r2 < r2_threshold:
            logger.warning(f"R2 for fitting Ls2, Cs2 is {ls2.r2}")
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

        if r2 < r2_threshold:
            logger.warning(f"R2 for fitting Rs2 is {rs2.r2}")

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
        if r2 < r2_threshold:
            logger.warning(f"R2 for fitting Lm is {r2}")
        results.lm = ValR2(value=lm_value, r2=r2)

    # Set context for downstream printing/plotting
    results._target_f = (
        float(rich_nw.target_f) if rich_nw.target_f is not None else None
    )
    results._nports = int(getattr(rich_nw.nw, "nports", 1))
    return results
