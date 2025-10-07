"""Snapshot-based regression tests for .s2p sample networks."""

import glob
import json
import os

import numpy as np
import pytest

from wpt_tools.analysis import nw_tools, nw_with_config

samples_dir = os.path.join(os.path.dirname(__file__), "data", "samples2p")
default_sample = os.path.join(os.path.dirname(__file__), "data", "sample.s2p")
collected = []
if os.path.isfile(default_sample):
    collected.append(default_sample)
if os.path.isdir(samples_dir):
    collected.extend(glob.glob(os.path.join(samples_dir, "*.s2p")))


@pytest.mark.parametrize("s2p_path", collected)
def test_config_and_efficiency_on_samples(s2p_path):
    """Compute and assert efficiency-related metrics against stored snapshots."""
    cfg = nw_with_config.import_touchstone(s2p_path)
    # choose mid frequency as target
    f_start = float(cfg.nw.frequency.f[0])
    f_stop = float(cfg.nw.npoints and cfg.nw.frequency.f[-1] or cfg.nw.frequency.f[0])
    target_f = (f_start + f_stop) / 2.0
    range_f = max((f_stop - f_start) * 0.2, 1.0)

    cfg.set_f_target_range(target_f, range_f)

    assert isinstance(cfg.target_f_index, int)
    assert 0 <= cfg.f_narrow_index_start <= cfg.f_narrow_index_stop < cfg.sweeppoint

    # Run efficiency analysis without plotting for speed
    # If 2-port, check both rx_port 1 and 2; if 1-port, only port 1
    ports_to_test = [1, 2] if getattr(cfg.nw, "nports", 1) >= 2 else [1]
    # Attempt to load expected snapshot
    expected_path = os.path.join(os.path.dirname(__file__), "expected_samples2p.json")
    expected = {}
    if os.path.isfile(expected_path):
        with open(expected_path, "r", encoding="utf-8") as f:
            expected = json.load(f)

    generated = {}

    for rx_port in ports_to_test:
        max_f_plot, max_eff_opt, max_r_opt, max_x_opt = nw_tools.analyze_efficiency(
            cfg, rx_port=rx_port, show_plot=False, show_data=False
        )

        generated[str(rx_port)] = {
            "target_f": float(target_f),
            "range_f": float(range_f),
            "max_f": float(max_f_plot),
            "eff": float(max_eff_opt),
            "Ropt": float(max_r_opt),
            "Xopt": float(max_x_opt),
        }

        # Sanity checks on results
        assert isinstance(max_f_plot, float)
        assert f_start <= max_f_plot <= f_stop
        assert 0.0 <= max_eff_opt <= 1.0
        assert np.isfinite(max_r_opt)
        assert np.isfinite(max_x_opt)

    # If asked, update snapshot file
    if os.environ.get("WPT_UPDATE_EXPECTED", "0") == "1":
        # merge with existing
        key = os.path.basename(s2p_path)
        if os.path.isfile(expected_path):
            with open(expected_path, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = {}
        else:
            existing = {}
        existing[key] = generated
        with open(expected_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, sort_keys=True)
        pytest.skip(
            "Updated expected snapshot; re-run tests without WPT_UPDATE_EXPECTED=1"
        )

    # Otherwise, assert against snapshot if present
    key = os.path.basename(s2p_path)
    if expected and key in expected:
        for rx_port, vals in generated.items():
            exp = expected[key].get(rx_port)
            assert exp is not None, f"Missing expected for {key} port {rx_port}"
            # Compare with tolerances
            assert vals["max_f"] == pytest.approx(exp["max_f"], rel=1e-3, abs=0)
            assert vals["eff"] == pytest.approx(exp["eff"], rel=5e-3, abs=0)
            assert vals["Ropt"] == pytest.approx(exp["Ropt"], rel=5e-2, abs=1e-3)
            assert vals["Xopt"] == pytest.approx(exp["Xopt"], rel=5e-2, abs=1e-3)
    else:
        pytest.skip(
            "No expected snapshot found. Run with WPT_UPDATE_EXPECTED=1 to generate expected values."
        )
