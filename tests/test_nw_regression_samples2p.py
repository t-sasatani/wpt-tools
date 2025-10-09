"""Regression test against snapshot for the bundled sample.s2p."""

import json
import os

import pytest

from wpt_tools.analysis import nw_tools
from wpt_tools.data_classes import RichNetwork


def test_sample_s2p_regression(tmp_path):
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

    # Snapshot path
    expected_path = os.path.join(os.path.dirname(__file__), "expected_sample.json")

    # If asked, write snapshot
    if os.environ.get("WPT_UPDATE_EXPECTED", "0") == "1":
        with open(expected_path, "w", encoding="utf-8") as f:
            json.dump(generated, f, indent=2, sort_keys=True)
        pytest.skip("Snapshot updated; re-run without WPT_UPDATE_EXPECTED=1")

    # Otherwise compare
    assert os.path.isfile(expected_path), "expected snapshot missing; run with WPT_UPDATE_EXPECTED=1"
    with open(expected_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    for port, vals in generated.items():
        exp = expected.get(port)
        assert exp is not None, f"Missing snapshot for port {port}"
        assert vals["max_f"] == pytest.approx(exp["max_f"], rel=1e-3)
        assert vals["eff"] == pytest.approx(exp["eff"], rel=5e-3)
        assert vals["Ropt"] == pytest.approx(exp["Ropt"], rel=5e-2, abs=1e-3)
        assert vals["Xopt"] == pytest.approx(exp["Xopt"], rel=5e-2, abs=1e-3)
