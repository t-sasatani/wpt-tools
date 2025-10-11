"""
Data classes for analyzing wireless power transfer systems.
"""

import dataclasses
from pathlib import Path
from typing import Optional, Union

import numpy as np
import skrf as rf
from tabulate import tabulate


@dataclasses.dataclass
class ValR2:
    """
    Pair of value and R2 score.

    Parameters
    ----------
    value: float
        The value.
    r2: float
        The R2 score.

    """

    value: Optional[float]
    r2: Optional[float]

    def __init__(self, value: Optional[float], r2: Optional[float]):
        """
        Initialize the class.
        """
        self.value = value
        self.r2 = r2


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

    def print_table(self) -> None:
        """Print a summary table of peak efficiency and optimal load."""
        self.validate()
        print("Efficiency summary\n")

    def validate(self) -> None:
        """Validate required fields for printing/plotting."""
        if self.max_f_plot is None:
            raise ValueError("EfficiencyResults.max_f_plot is None")
        if self.max_eff_opt is None:
            raise ValueError("EfficiencyResults.max_eff_opt is None")
        if self.max_r_opt is None:
            raise ValueError("EfficiencyResults.max_r_opt is None")
        if self.max_x_opt is None:
            raise ValueError("EfficiencyResults.max_x_opt is None")
        print(
            tabulate(
                [
                    ["Target frequency", self.max_f_plot],
                    ["Maximum efficiency", self.max_eff_opt],
                    ["Optimum Re(Zload)", self.max_r_opt],
                    ["Optimum Im(Zload)", self.max_x_opt],
                ],
                headers=["Parameter", "Value"],
                stralign="left",
                numalign="right",
                floatfmt=".3e",
                tablefmt="fancy_grid",
            )
        )


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

    ls1: ValR2
    cs1: ValR2
    rs1: ValR2
    ls2: ValR2
    cs2: ValR2
    rs2: ValR2
    lm: ValR2

    def __init__(self):
        """
        Initialize the results.
        """
        self.ls1 = ValR2(None, None)
        self.cs1 = ValR2(None, None)
        self.rs1 = ValR2(None, None)
        self.ls2 = ValR2(None, None)
        self.cs2 = ValR2(None, None)
        self.rs2 = ValR2(None, None)
        self.lm = ValR2(None, None)
        # Context for printing without external inputs
        self._target_f: Optional[float] = None
        self._nports: Optional[int] = None

    def print_table(self) -> None:
        """Print port-1, port-2 (if present), and mutual tables separately."""
        self.validate()
        target_f = self._target_f
        nports = (
            self._nports
            if self._nports is not None
            else (2 if hasattr(self, "ls2") and isinstance(self.ls2, ValR2) else 1)
        )
        # Port 1 table
        print(
            tabulate(
                [
                    ["Ls1", self.ls1.value, f"{self.ls1.r2:.3e}"],
                    ["Cs1", self.cs1.value, f"{self.cs1.r2:.3e}"],
                    ["Rs1", self.rs1.value, f"{self.rs1.r2:.3e}"],
                    [
                        "f_1",
                        1 / (2 * np.pi * np.sqrt(self.ls1.value * self.cs1.value)),
                        "",
                    ],
                    (
                        [
                            f"Q_1 (approx., @{target_f:.3e} Hz)",
                            2
                            * np.pi
                            * float(target_f)
                            * self.ls1.value
                            / self.rs1.value,
                            "",
                        ]
                        if target_f is not None
                        else ["Q_1 (approx.)", "", ""]
                    ),
                ],
                headers=["Fitting params (port 1)", "Value", "R2"],
                stralign="left",
                numalign="right",
                floatfmt=".3e",
                tablefmt="fancy_grid",
            )
        )
        # Port 2 table
        if nports == 2:
            print(
                tabulate(
                    [
                        ["Ls2", self.ls2.value, f"{self.ls2.r2:.3e}"],
                        ["Cs2", self.cs2.value, f"{self.cs2.r2:.3e}"],
                        ["Rs2", self.rs2.value, f"{self.rs2.r2:.3e}"],
                        [
                            "f_2",
                            1 / (2 * np.pi * np.sqrt(self.ls2.value * self.cs2.value)),
                            "",
                        ],
                        (
                            [
                                f"Q_2 (approx., @{target_f:.3e} Hz)",
                                2
                                * np.pi
                                * float(target_f)
                                * self.ls2.value
                                / self.rs2.value,
                                "",
                            ]
                            if target_f is not None
                            else ["Q_2 (approx.)", "", ""]
                        ),
                    ],
                    headers=["Fitting params (port 2)", "Value", "R2"],
                    stralign="left",
                    numalign="right",
                    floatfmt=".3e",
                    tablefmt="fancy_grid",
                )
            )
        # Mutual table (print if fitted)
        if isinstance(self.lm, ValR2) and self.lm.value is not None:
            ls_for_km = (
                self.ls2.value
                if (isinstance(self.ls2, ValR2) and self.ls2.value is not None)
                else self.ls1.value
            )
            print(
                tabulate(
                    [
                        ["Fitting params (mutual)", "", ""],
                        ["Lm", self.lm.value, f"{self.lm.r2:.3e}"],
                        ["km", self.lm.value / np.sqrt(self.ls1.value * ls_for_km), ""],
                    ],
                    headers=["Parameter", "Value", "R2"],
                    stralign="left",
                    numalign="right",
                    floatfmt=".3e",
                    tablefmt="fancy_grid",
                )
            )

    def validate(self) -> None:
        """Validate required fitted values exist before printing/plotting."""
        if not isinstance(self.ls1, ValR2) or self.ls1.value is None:
            raise ValueError("LCRFittingResults.ls1 missing")
        if not isinstance(self.cs1, ValR2) or self.cs1.value is None:
            raise ValueError("LCRFittingResults.cs1 missing")
        if not isinstance(self.rs1, ValR2) or self.rs1.value is None:
            raise ValueError("LCRFittingResults.rs1 missing")


@dataclasses.dataclass
class OptimalLoadGridResults:
    """
    Results for optimal load grid sweep (efficiency, Pin, Pout).
    """

    rez_list: np.ndarray
    imz_list: np.ndarray
    eff_grid: np.ndarray
    Pin: np.ndarray
    Pout: np.ndarray
    target_f: Optional[float]

    def __init__(
        self,
        rez_list: np.ndarray,
        imz_list: np.ndarray,
        eff_grid: np.ndarray,
        Pin: np.ndarray,
        Pout: np.ndarray,
        target_f: Optional[float],
    ):
        """Initialize the optimal load grid results."""
        self.rez_list = rez_list
        self.imz_list = imz_list
        self.eff_grid = eff_grid
        self.Pin = Pin
        self.Pout = Pout
        self.target_f = target_f

    def print_table(self) -> None:
        """Print sweep metadata (dimensions and target freq)."""
        self.validate()
        print("Load sweep metadata\n")

    def validate(self) -> None:
        """Validate required arrays for plotting/printing."""
        if self.rez_list is None or self.imz_list is None:
            raise ValueError("Load sweep axes missing")
        if self.eff_grid is None or self.Pin is None or self.Pout is None:
            raise ValueError("Load sweep grids missing")
        print(
            tabulate(
                [
                    ["Re(Z) points", self.rez_list.size],
                    ["Im(Z) points", self.imz_list.size],
                    ["Target frequency", self.target_f],
                ],
                headers=["Parameter", "Value"],
                stralign="left",
                numalign="right",
                floatfmt=".3e",
                tablefmt="fancy_grid",
            )
        )


@dataclasses.dataclass
class RXCFilterResults:
    """
    Results for receiver RXC filter calculation at optimal point.
    """

    cp: float
    cs: float
    rload: float
    max_r_opt: float
    max_x_opt: float
    max_f_plot: float
    lrx: float

    def print_table(self) -> None:
        """Print RXC filter summary values."""
        self.validate()
        print("RXC filter summary\n")

    def validate(self) -> None:
        """Validate RXC filter fields before printing/plotting."""
        for name in (
            "cp",
            "cs",
            "rload",
            "max_r_opt",
            "max_x_opt",
            "max_f_plot",
            "lrx",
        ):
            if getattr(self, name, None) is None:
                raise ValueError(f"RXCFilterResults.{name} is None")
        print(
            tabulate(
                [
                    ["Target frequency", self.max_f_plot],
                    ["Optimum Re(Zload)", self.max_r_opt],
                    ["Optimum Im(Zload)", self.max_x_opt],
                    ["Receiver inductance", self.lrx],
                    ["Target Rload", self.rload],
                    ["Cp", self.cp],
                    ["Cs", self.cs],
                ],
                headers=["Parameter", "Value"],
                stralign="left",
                numalign="right",
                floatfmt=".3e",
                tablefmt="fancy_grid",
            )
        )


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
