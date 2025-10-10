"""Regression test against snapshot for the bundled sample.s2p."""

import json
import os
import sys

import numpy as np
import pytest

from wpt_tools.analysis import nw_tools
from wpt_tools.data_classes import RichNetwork
from wpt_tools.solvers import (
    compute_load_sweep,
    compute_rxc_filter,
    lcr_fitting,
)


@pytest.mark.xfail(
    condition=sys.platform == "win32",
    reason="LCR fitting results differ on Windows due to numerical precision differences",
    strict=False,
)
def test_sample_s2p_regression():
    """
    Test the regression of the sample.s2p.
    """
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample.s2p")
    assert os.path.isfile(sample_path), "missing sample.s2p"

    nw = RichNetwork.from_touchstone(sample_path)
    # Choose mid frequency as target
    f_start = float(nw.nw.frequency.f[0])
    f_stop = float(nw.nw.frequency.f[-1])
    target_f = (f_start + f_stop) / 2.0
    range_f = max((f_stop - f_start) * 0.2, 1.0)
    nw.set_f_target_range(target_f, range_f)

    # Compute metrics (both ports if 2-port)
    ports = [1, 2] if getattr(nw.nw, "nports", 1) >= 2 else [1]
    generated = {}
    for rx_port in ports:
        eff = nw_tools.analyze_efficiency(
            nw, rx_port=rx_port, show_plot=False, show_data=False
        )
        generated[str(rx_port)] = {
            "max_f": float(eff.max_f_plot),
            "eff": float(eff.max_eff_opt),
            "Ropt": float(eff.max_r_opt),
            "Xopt": float(eff.max_x_opt),
        }

    # LCR fitting snapshot (narrow around target)
    fit = lcr_fitting(nw, target_f=target_f, range_f=range_f)
    generated["fit"] = {
        "ls1": float(fit.ls1.value),
        "cs1": float(fit.cs1.value),
        "rs1": float(fit.rs1.value),
    }
    if (
        getattr(nw.nw, "nports", 1) >= 2
        and hasattr(fit, "ls2")
        and fit.ls2.value is not None
    ):
        generated["fit"].update(
            {
                "ls2": float(fit.ls2.value),
                "cs2": float(fit.cs2.value),
                "rs2": float(fit.rs2.value),
            }
        )
    if hasattr(fit, "lm") and fit.lm.value is not None:
        generated["fit"]["lm"] = float(fit.lm.value)

    # RXC filter snapshot for a canonical case
    rxc = compute_rxc_filter(
        nw,
        rx_port=1,
        rload=100.0,
        c_network="CpCsRl",
        target_f=target_f,
        range_f=range_f,
    )
    generated["rxc"] = {"cp": float(rxc.cp), "cs": float(rxc.cs)}

    # Load sweep snapshot summary (rx_port=1)
    sweep = compute_load_sweep(
        nw,
        rez_min=0.1,
        rez_max=50.0,
        rez_step=0.5,
        imz_min=-200.0,
        imz_max=200.0,
        imz_step=2.0,
        rx_port=1,
        input_voltage=1.0,
        target_f=target_f,
        range_f=range_f,
    )
    # Argmax summary
    arg_idx = np.unravel_index(np.argmax(sweep.eff_grid), sweep.eff_grid.shape)
    generated["sweep"] = {
        "eff_max": float(sweep.eff_grid[arg_idx]),
        "rez_at_max": float(sweep.rez_list[arg_idx[0]]),
        "imz_at_max": float(sweep.imz_list[arg_idx[1]]),
    }

    # Snapshot path
    expected_path = os.path.join(os.path.dirname(__file__), "expected_sample.json")

    # If asked, write snapshot
    if os.environ.get("WPT_UPDATE_EXPECTED", "0") == "1":
        with open(expected_path, "w", encoding="utf-8") as f:
            json.dump(generated, f, indent=2, sort_keys=True)
        pytest.skip("Snapshot updated; re-run without WPT_UPDATE_EXPECTED=1")

    # Otherwise compare
    assert os.path.isfile(
        expected_path
    ), "expected snapshot missing; run with WPT_UPDATE_EXPECTED=1"
    with open(expected_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    # Compare efficiency per-port
    for port in ports:
        key = str(port)
        vals = generated[key]
        exp = expected.get(key)
        assert exp is not None, f"Missing snapshot for port {key}"
        assert vals["max_f"] == pytest.approx(exp["max_f"], rel=1e-3)
        assert vals["eff"] == pytest.approx(exp["eff"], rel=5e-3)
        assert vals["Ropt"] == pytest.approx(exp["Ropt"], rel=5e-2, abs=1e-3)
        assert vals["Xopt"] == pytest.approx(exp["Xopt"], rel=5e-2, abs=1e-3)

    # Compare LCR fit values
    assert "fit" in expected, "Missing snapshot for fit"
    for k, v in generated["fit"].items():
        assert v == pytest.approx(expected["fit"][k], rel=5e-2)

    # Compare RXC filter cp/cs
    assert "rxc" in expected, "Missing snapshot for rxc"
    assert generated["rxc"]["cp"] == pytest.approx(expected["rxc"]["cp"], rel=5e-2)
    assert generated["rxc"]["cs"] == pytest.approx(expected["rxc"]["cs"], rel=5e-2)

    # Compare load sweep summary
    assert "sweep" in expected, "Missing snapshot for sweep"
    for k, v in generated["sweep"].items():
        assert v == pytest.approx(expected["sweep"][k], rel=5e-2)

    # Wrapper parity checks (no snapshot): fit_z_narrow vs lcr_fitting
    fit_wrap = nw_tools.fit_z_narrow(
        nw, show_plot=False, show_data=False, target_f=target_f, range_f=range_f
    )
    assert float(fit_wrap.ls1.value) == pytest.approx(float(fit.ls1.value), rel=1e-6)
    assert float(fit_wrap.cs1.value) == pytest.approx(float(fit.cs1.value), rel=1e-6)
    assert float(fit_wrap.rs1.value) == pytest.approx(float(fit.rs1.value), rel=1e-6)
    if (
        getattr(nw.nw, "nports", 1) >= 2
        and hasattr(fit, "ls2")
        and fit.ls2.value is not None
    ):
        assert float(fit_wrap.ls2.value) == pytest.approx(
            float(fit.ls2.value), rel=1e-6
        )
        assert float(fit_wrap.cs2.value) == pytest.approx(
            float(fit.cs2.value), rel=1e-6
        )
        assert float(fit_wrap.rs2.value) == pytest.approx(
            float(fit.rs2.value), rel=1e-6
        )

    # calc_rxc_filter wrapper parity with compute_rxc_filter
    nw_tools.calc_rxc_filter(
        nw, rx_port=1, rload=100.0, target_f=target_f, range_f=range_f, show_data=False
    )
    # calc_rxc_filter prints only; ensure compute_rxc_filter returns values we snapshot
    rxc2 = compute_rxc_filter(
        nw,
        rx_port=1,
        rload=100.0,
        c_network="CpCsRl",
        target_f=target_f,
        range_f=range_f,
    )
    assert float(rxc2.cp) == pytest.approx(float(generated["rxc"]["cp"]), rel=1e-6)
    assert float(rxc2.cs) == pytest.approx(float(generated["rxc"]["cs"]), rel=1e-6)
