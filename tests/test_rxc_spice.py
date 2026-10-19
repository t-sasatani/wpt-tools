"""Independent ngspice check of the RXC filter design.

At the design frequency the (reciprocal) measured Z-matrix is realised exactly
as a T-network: series arms ``Z11 - Zm`` and ``Z22 - Zm`` and a shunt arm
``Zm``. Each complex impedance becomes R in series with L (X > 0) or C (X < 0)
at that frequency. The designed Cp/Cs/Rload are attached to the receiver side
and ngspice runs a single-point AC analysis; efficiency is P(Rload) / P(in).

Skipped when ``ngspice`` is not on PATH (it is not installed on CI).
"""

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import skrf as rf

from wpt_tools.data_classes import RichNetwork
from wpt_tools.solver import (
    compute_rxc_filter,
    efficiency_calculator,
    solve_cpcs_closed_form,
)

NGSPICE = shutil.which("ngspice")
SAMPLE_S2P = Path(__file__).parent / "data" / "sample.s2p"
RLOAD = 100.0
TARGET_F = 8e6
RANGE_F = 2.8e6

pytestmark = pytest.mark.skipif(NGSPICE is None, reason="ngspice not installed")


def _branch(name, n1, n2, z, w):
    """SPICE lines realising complex impedance ``z`` between ``n1`` and ``n2``."""
    mid = f"{name}_m"
    lines = [f"R{name} {n1} {mid} {z.real:.12e}"]
    if z.imag >= 0:
        lines.append(f"L{name} {mid} {n2} {z.imag / w:.12e}")
    else:
        lines.append(f"C{name} {mid} {n2} {-1.0 / (w * z.imag):.12e}")
    return lines


def _netlist(z11, z22, zm, cp, cs, rload, f0):
    w = 2 * np.pi * f0
    lines = ["* wpt-tools RXC filter check", "Vin in 0 AC 1"]
    lines += _branch("a", "in", "t", z11 - zm, w)
    lines += _branch("c", "t", "0", zm, w)
    lines += _branch("b", "t", "rx", z22 - zm, w)
    lines += [
        f"Cp rx 0 {cp:.12e}",
        f"Cs rx ld {cs:.12e}",
        f"Rload ld 0 {rload:.12e}",
        ".control",
        f"ac lin 1 {f0:.6f} {f0:.6f}",
        "let pin = real(v(in) * conj(-i(Vin)))",
        "let pout = real(v(ld) * conj(v(ld) / @Rload[resistance]))",
        "let eff = pout / pin",
        "print eff",
        "quit",
        ".endc",
        ".end",
    ]
    return "\n".join(lines) + "\n"


def _ngspice_efficiency(tmp_path, netlist_text):
    cir = tmp_path / "rxc.cir"
    cir.write_text(netlist_text)
    out = subprocess.run(
        [NGSPICE, "-b", str(cir)], capture_output=True, text=True, check=False
    )
    for line in (out.stdout + out.stderr).splitlines():
        if line.strip().startswith("eff ="):
            return float(line.split("=")[1].split()[0])
    raise RuntimeError(f"ngspice produced no 'eff' line:\n{out.stdout}\n{out.stderr}")


def _reciprocal_sample():
    nw = rf.Network(str(SAMPLE_S2P))
    z = nw.z.copy()
    z[:, 1, 0] = z[:, 0, 1]
    return RichNetwork.from_touchstone(rf.Network.from_z(z, frequency=nw.frequency))


@pytest.mark.parametrize("rx_port", [1, 2])
def test_rxc_design_in_ngspice(tmp_path, rx_port, caplog):
    """Efficiency simulated by ngspice with the designed Cp/Cs equals max_eff_opt."""
    caplog.set_level(logging.ERROR)
    rich_nw = _reciprocal_sample()
    res = compute_rxc_filter(
        rich_nw, rx_port=rx_port, rload=RLOAD, target_f=TARGET_F, range_f=RANGE_F
    )
    eff = efficiency_calculator(
        rich_nw, rx_port=rx_port, target_f=TARGET_F, range_f=RANGE_F
    )
    i = int(np.argmin(np.abs(rich_nw.nw.frequency.f - res.max_f_plot)))
    z = rich_nw.nw.z[i]
    z11, z22 = (z[0, 0], z[1, 1]) if rx_port == 2 else (z[1, 1], z[0, 0])
    zm = z[0, 1]
    f0 = res.max_f_plot

    eff_spice = _ngspice_efficiency(
        tmp_path, _netlist(z11, z22, zm, res.cp, res.cs, RLOAD, f0)
    )
    assert eff_spice == pytest.approx(eff.max_eff_opt, abs=1e-6)

    # Legacy design (resonate fitted Lrx, Im -> 0) on the same topology is far off.
    w = 2 * np.pi * f0
    legacy = solve_cpcs_closed_form(complex(res.max_r_opt, -w * res.lrx), RLOAD, w)
    assert legacy is not None
    eff_legacy = _ngspice_efficiency(
        tmp_path, _netlist(z11, z22, zm, legacy[0], legacy[1], RLOAD, f0)
    )
    assert eff_legacy < eff.max_eff_opt - 0.15
