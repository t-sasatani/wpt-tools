"""
Solvers for wpt-tools.
"""

from typing import Literal, Optional

import numpy as np
from wpt_tools.data_classes import RichNetwork, override_frange, EfficiencyResults

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
