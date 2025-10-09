"""
Data classes for analyzing wireless power transfer systems.
"""

import dataclasses
from pathlib import Path
from typing import Optional, Union

import numpy as np
import skrf as rf


@dataclasses.dataclass
class EfficiencyResults:
    """
    Results of efficiency solver.

    """

    f_plot: list[float]
    r_opt: list[float]
    x_opt: list[float]
    eff_opt: list[float]
    max_f_plot: float
    max_eff_opt: float
    max_r_opt: float
    max_x_opt: float

    def __init__(self):
        """
        Initialize the results.
        """
        self.f_plot = []
        self.r_opt = []
        self.x_opt = []
        self.eff_opt = []
        self.max_f_plot = None
        self.max_eff_opt = None
        self.max_r_opt = None
        self.max_x_opt = None

class LCRFittingResults:
    """
    Results of LCR fitting.

    Parameters
    ----------
    ls1: float
        The inductance of the first series LCR.
    cs1: float
        The capacitance of the first series LCR.
    rs1: float
        The resistance of the first series LCR.
    ls2: float
        The inductance of the second series LCR.
    cs2: float
        The capacitance of the second series LCR.
    rs2: float
        The resistance of the second series LCR.
    lm: float
        The inductance of the mutual coupling.
    km: float
        The mutual coupling coefficient.

    """

    ls1: float
    cs1: float
    rs1: float
    ls2: float
    cs2: float
    rs2: float
    lm: float

    def __init__(self):
        """
        Initialize the results.
        """
        self.ls1 = None
        self.cs1 = None
        self.rs1 = None
        self.ls2 = None
        self.cs2 = None
        self.rs2 = None
        self.lm = None

@dataclasses.dataclass
class RichNetwork:
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
    def from_touchstone(cls, source: Union[str, Path, rf.Network]) -> "RichNetwork":
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


def override_frange(
    rich_nw: RichNetwork, target_f: Optional[float], range_f: Optional[float]
) -> RichNetwork:
    """
    Override target frequency and range (or validate existing), and compute indices.

    - If both are provided, applies set_f_target_range.
    - Otherwise, validates existing values and computes missing indices.
    """
    # If caller provided both, compute and set derived fields.
    if target_f is not None and range_f is not None:
        rich_nw.set_f_target_range(target_f=float(target_f), range_f=float(range_f))
    else:
        # Otherwise require they already exist on the instance.
        if rich_nw.target_f is None or rich_nw.range_f is None:
            raise ValueError(
                "Target frequency is not set. Provide target_f and range_f or call set_f_target_range."
            )
        # Ensure sweeppoint at minimum.
        if rich_nw.sweeppoint is None:
            rich_nw.sweeppoint = int(np.size(rich_nw.nw.frequency.f))
        # If indices are not computed yet, compute them using existing target/range
        if (
            rich_nw.f_narrow_index_start is None
            or rich_nw.f_narrow_index_stop is None
            or rich_nw.target_f_index is None
        ):
            rich_nw.set_f_target_range(
                target_f=float(rich_nw.target_f), range_f=float(rich_nw.range_f)
            )
    return rich_nw
