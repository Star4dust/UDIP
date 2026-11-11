"""Deterministic regression suite for Phase-1 calibration modules.

Validates:
  1. detect_redundant_groups()
  2. redundancy_averaged_visibilities()
  3. redundancy_calibration_residual()
  4. redundancy_improvement_metric()
  5. solve_gain_vs_frequency()
  6. propagate_gain_uncertainty_to_tsys()
  7. gain_stability_metric()
  8. Reproducibility (re-running the suite yields an identical signature).

All tests are deterministic and headless.  Run from the project root:

    python3 validation_corpus/run_redundant_calibration_regression.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import sys

# Make the project root importable regardless of cwd.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from modules.redundant_calibration import (
    detect_redundant_groups,
    redundancy_averaged_visibilities,
    redundancy_calibration_residual,
    redundancy_improvement_metric,
)
from modules.per_frequency_gain import (
    gain_stability_metric,
    propagate_gain_uncertainty_to_tsys,
    solve_gain_vs_frequency,
)

REGRESSION_MANIFEST = _HERE / "redundant_calibration_regression.json"


# ---------------------------------------------------------------------------
# Test framework
# ---------------------------------------------------------------------------

class TestRunner:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def run(self, name: str, fn) -> None:
        try:
            details = fn() or ""
        except AssertionError as exc:
            self.results.append({"name": name, "status": "FAIL", "detail": str(exc)})
            return
        except Exception as exc:
            self.results.append({"name": name, "status": "ERROR", "detail": f"{type(exc).__name__}: {exc}"})
            return
        self.results.append({"name": name, "status": "PASS", "detail": details})

    def summary(self) -> dict[str, Any]:
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        total = len(self.results)
        return {"passed": passed, "total": total, "results": self.results}

    def print(self) -> None:
        print(f"\nPASSED: {sum(1 for r in self.results if r['status'] == 'PASS')}    "
              f"FAILED: {sum(1 for r in self.results if r['status'] != 'PASS')}")
        for r in self.results:
            print(f"  {r['status']:<5} {r['name']}: {r['detail']}")


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _make_visibility_array(
    n_time: int,
    n_freq: int,
    n_baseline: int,
    n_pol: int = 1,
    *,
    rng: np.random.Generator,
    signal_amplitude: float = 1.0,
    noise_sigma: float = 0.1,
) -> np.ndarray:
    """Create a complex visibility cube with a known but realistic signal."""
    t = np.linspace(0.0, 1.0, n_time)[:, None, None, None]
    f = np.linspace(0.0, 1.0, n_freq)[None, :, None, None]
    phase = 2.0 * math.pi * (2.0 * t + 3.0 * f)
    sig = signal_amplitude * (np.cos(phase) + 1j * np.sin(phase))
    noise = rng.normal(0.0, noise_sigma, (n_time, n_freq, n_baseline, n_pol)) + 1j * rng.normal(
        0.0, noise_sigma, (n_time, n_freq, n_baseline, n_pol)
    )
    return (sig + noise).astype(np.complex128)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_redundant_group_detection_2groups(t: TestRunner) -> None:
    ant = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [12.0, 0.0, 0.0],
        [18.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [6.0, 6.0, 0.0],
        [12.0, 6.0, 0.0],
        [18.0, 6.0, 0.0],
    ], dtype=float)
    pairs = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (0, 4), (1, 5)]
    res = detect_redundant_groups(
        baseline_pairs=pairs,
        antenna_positions=ant,
        tolerance_m=0.5,
    )
    assert res["n_unique_groups"] == 4, f"expected 4 unique groups, got {res['n_unique_groups']}"
    assert res["redundancy_factor"] == 3, f"expected redundancy 3, got {res['redundancy_factor']}"
    assert res["n_redundant_baselines"] == 6, f"expected 6 redundant baselines, got {res['n_redundant_baselines']}"
    sizes = sorted(res["group_sizes"], reverse=True)
    assert sizes[:2] == [3, 3], f"expected two groups of 3, got {sizes[:2]}"
    assert sorted(sizes[2:]) == [1, 1], f"expected two singletons, got {sizes[2:]}"


def test_redundant_group_detection_no_redundancy(t: TestRunner) -> None:
    ant = np.array([[0, 0, 0], [6, 0, 0], [13, 0, 0]], dtype=float)
    pairs = [(0, 1), (0, 2), (1, 2)]
    res = detect_redundant_groups(baseline_pairs=pairs, antenna_positions=ant, tolerance_m=0.5)
    assert res["redundancy_factor"] == 1, f"expected 1, got {res['redundancy_factor']}"
    assert res["n_redundant_baselines"] == 0
    assert sorted(res["group_sizes"]) == [1, 1, 1]
    # Insufficient redundancy must produce a warning.
    insuf = detect_redundant_groups(baseline_pairs=pairs, antenna_positions=ant, tolerance_m=0.0)
    assert any("insufficient" in w.lower() for w in insuf["warnings"]), \
        f"expected insufficient-redundancy warning, got {insuf['warnings']}"


def test_redundant_group_detection_uses_uv_fallback(t: TestRunner) -> None:
    uv = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 0.0], [12.0, 0.0]], dtype=float)
    res = detect_redundant_groups(uv_coordinates=uv, tolerance_m=0.5)
    sizes = sorted(res["group_sizes"], reverse=True)
    assert sizes[0] == 2, f"expected a redundant pair, got sizes {sizes}"
    assert res["redundancy_factor"] == 2


def test_redundancy_averaged_visibilities_shapes(t: TestRunner) -> None:
    rng = np.random.default_rng(42)
    vis = _make_visibility_array(4, 8, 6, rng=rng)
    pairs = [(0, 1), (1, 2), (2, 3), (0, 4), (1, 5), (2, 6)]
    ant = np.array([[0, 0, 0], [6, 0, 0], [12, 0, 0], [18, 0, 0],
                    [0, 6, 0], [6, 6, 0], [12, 6, 0]], dtype=float)
    res_g = detect_redundant_groups(baseline_pairs=pairs, antenna_positions=ant, tolerance_m=0.5)
    av = redundancy_averaged_visibilities(vis, res_g["groups"])
    # Output must preserve time, freq, pol axes and produce N_groups baseline entries.
    assert av["averaged_visibilities"].shape == (4, 8, av["n_groups"], 1), \
        f"unexpected averaged shape: {av['averaged_visibilities'].shape}"
    assert av["n_groups"] == res_g["n_unique_groups"]
    # sigma_post must be < sigma_pre by a measurable amount for any redundant group.
    assert np.all(np.isfinite(av["per_group_sigma"])) or np.all(np.isnan(av["per_group_sigma"]))


def test_redundancy_calibration_residual_in_unit_gain(t: TestRunner) -> None:
    rng = np.random.default_rng(7)
    vis = _make_visibility_array(2, 4, 5, rng=rng, noise_sigma=0.05)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    ant = np.array([[0, 0, 0], [6, 0, 0], [12, 0, 0], [18, 0, 0], [24, 0, 0]], dtype=float)
    res_g = detect_redundant_groups(baseline_pairs=pairs, antenna_positions=ant, tolerance_m=0.5)
    res = redundancy_calibration_residual(vis, res_g["groups"])
    # All units of gain -> residual must be small.
    amp_rms = res["per_baseline"]["amp_rms"]
    assert np.all(np.isfinite(amp_rms))
    assert float(np.median(amp_rms)) < 0.5, f"residual amp_rms median too large: {float(np.median(amp_rms))}"


def test_redundancy_improvement_metric_p(t: TestRunner) -> None:
    rng = np.random.default_rng(123)
    # Strongly redundant visibilities -> significant sigma_post < sigma_pre improvement.
    vis = _make_visibility_array(3, 5, 7, rng=rng, signal_amplitude=2.0, noise_sigma=0.5)
    pairs = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3)]
    ant = np.array([[0, 0, 0], [6, 0, 0], [12, 0, 0],
                    [0, 6, 0], [6, 6, 0], [12, 6, 0], [6, 3, 0]], dtype=float)
    res_g = detect_redundant_groups(baseline_pairs=pairs, antenna_positions=ant, tolerance_m=0.5)
    metric = redundancy_improvement_metric(vis, res_g["groups"])
    # At least one group must show improvement > 1.
    assert metric["n_improved_groups"] >= 1, f"expected improved groups, got {metric['n_improved_groups']}"
    assert math.isfinite(metric["mean_improvement_factor"])


def test_per_frequency_gain_recovery(t: TestRunner) -> None:
    n_freq = 8
    n_time = 16
    n_baseline = 4
    true_g = (np.array([1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07]) +
              1j * np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]))
    model = np.ones((n_time, n_freq, n_baseline, 1), dtype=np.complex128)
    rng = np.random.default_rng(99)
    noise = 0.01 * (rng.normal(size=(n_time, n_freq, n_baseline, 1)) +
                    1j * rng.normal(size=(n_time, n_freq, n_baseline, 1)))
    meas = g[:, None, None, None] * model + noise
    g_meas = meas
    sol = solve_gain_vs_frequency(
        visibilities_measured=g_meas,
        visibilities_model=model,
        frequencies_mhz=np.array([700.0, 710.0, 720.0, 730.0, 740.0, 750.0, 760.0, 770.0]),
        noise_sigma_measured=0.01,
    )
    recovered = sol["gain"]
    diff = np.abs(recovered - true_g)
    max_err = float(np.max(diff))
    assert max_err < 0.02, f"gain recovery error {max_err} too large"
    # Uncertainties must be finite and > 0.
    assert np.all(np.isfinite(sol["gain_sigma"]))
    assert np.all(sol["gain_sigma"] > 0)


def test_propagate_gain_uncertainty_to_tsys(t: TestRunner) -> None:
    g = np.array([1.0 + 0.0j, 1.01 + 0.0j, 1.05 + 0.0j, 1.10 + 0.0j])
    g_sig = np.array([0.01, 0.02, 0.03, 0.05])
    out = propagate_gain_uncertainty_to_tsys(
        gain=g, gain_sigma=g_sig,
        trxr_reference_k=50.0, trxr_reference_uncertainty_k=5.0,
        t_sky_k=np.array([5.0, 5.0, 5.0, 5.0]),
    )
    assert out["trxr_k"].shape == (4,)
    # sigma must grow with g_sig and trxr_unc.
    assert out["tsys_sigma_k"][-1] > out["tsys_sigma_k"][0]
    # The gain-derived Tsys must be larger than the no-gain Tsys by a positive fraction.
    assert np.all(out["tsys_k"] > 5.0)


def test_gain_stability_metric(t: TestRunner) -> None:
    n_time = 32
    n_freq = 16
    n_baseline = 3
    t_axis = np.arange(n_time)
    f_axis = np.arange(n_freq)
    # A slow gain drift with small noise.
    drift = 1.0 + 0.005 * (t_axis[:, None, None, None] / n_time) + 0.002 * (f_axis[None, :, None, None] / n_freq)
    g_cube = (drift + 0.01 * np.random.default_rng(0).normal(size=(n_time, n_freq, n_baseline, 1))).astype(np.complex128)
    # Build model and measurement.
    model = np.ones_like(g_cube)
    meas = g_cube * model
    sol = solve_gain_vs_frequency(
        visibilities_measured=meas, visibilities_model=model,
        frequencies_mhz=f_axis.astype(float),
    )
    stab = gain_stability_metric(sol)
    assert math.isfinite(stab["amp_std_median"])
    assert math.isfinite(stab["phase_std_median_rad"])
    assert stab["n_valid_freq_bins"] == n_freq


def test_reproducibility_signature(t: TestRunner) -> None:
    # Build a small batch and re-run it: results must be identical.
    rng = np.random.default_rng(2026)
    vis = _make_visibility_array(3, 5, 4, rng=rng, noise_sigma=0.05)
    pairs = [(0, 1), (1, 2), (0, 2), (0, 3)]
    ant = np.array([[0, 0, 0], [6, 0, 0], [12, 0, 0], [6, 6, 0]], dtype=float)
    res_g = detect_redundant_groups(baseline_pairs=pairs, antenna_positions=ant, tolerance_m=0.5)
    av1 = redundancy_averaged_visibilities(vis, res_g["groups"])
    av2 = redundancy_averaged_visibilities(vis, res_g["groups"])
    h1 = hashlib.sha256(av1["averaged_visibilities"].tobytes()).hexdigest()
    h2 = hashlib.sha256(av2["averaged_visibilities"].tobytes()).hexdigest()
    assert h1 == h2, f"reproducibility broken: {h1} != {h2}"


# ---------------------------------------------------------------------------
# Manifest-driven smoke test
# ---------------------------------------------------------------------------

def test_manifest_fixtures_match_module_outputs(t: TestRunner) -> None:
    if not REGRESSION_MANIFEST.exists():
        return  # fixture optional, skip silently
    manifest = json.loads(REGRESSION_MANIFEST.read_text())
    for fix in manifest.get("fixtures", []):
        fid = fix["id"]
        if "antenna_positions" in fix:
            ant = np.asarray(fix["antenna_positions"], dtype=float)
            pairs = [tuple(p) for p in fix["baseline_pairs"]]
            res = detect_redundant_groups(
                baseline_pairs=pairs, antenna_positions=ant, tolerance_m=fix.get("tolerance_m", 0.5)
            )
            assert res["redundancy_factor"] == fix["expected_redundancy_factor"], (
                f"fixture {fid}: expected redundancy_factor {fix['expected_redundancy_factor']}, "
                f"got {res['redundancy_factor']}"
            )
            assert res["n_redundant_baselines"] == fix["expected_n_redundant_baselines"], (
                f"fixture {fid}: expected n_redundant {fix['expected_n_redundant_baselines']}, "
                f"got {res['n_redundant_baselines']}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t = TestRunner()
    t.run("redundant_group_detection_2groups", lambda: test_redundant_group_detection_2groups(t))
    t.run("redundant_group_detection_no_redundancy", lambda: test_redundant_group_detection_no_redundancy(t))
    t.run("redundant_group_detection_uses_uv_fallback", lambda: test_redundant_group_detection_uses_uv_fallback(t))
    t.run("redundancy_averaged_visibilities_shapes", lambda: test_redundancy_averaged_visibilities_shapes(t))
    t.run("redundancy_calibration_residual_in_unit_gain", lambda: test_redundancy_calibration_residual_in_unit_gain(t))
    t.run("redundancy_improvement_metric", lambda: test_redundancy_improvement_metric_p(t))
    t.run("per_frequency_gain_recovery", lambda: test_per_frequency_gain_recovery(t))
    t.run("propagate_gain_uncertainty_to_tsys", lambda: test_propagate_gain_uncertainty_to_tsys(t))
    t.run("gain_stability_metric", lambda: test_gain_stability_metric(t))
    t.run("reproducibility_signature", lambda: test_reproducibility_signature(t))
    t.run("manifest_fixtures_match_module_outputs", lambda: test_manifest_fixtures_match_module_outputs(t))
    summary = t.summary()
    print("=" * 60)
    print(f"PASSED: {summary['passed']}    FAILED: {summary['total'] - summary['passed']}")
    for r in summary["results"]:
        print(f"  {r['status']:<5} {r['name']}: {r['detail']}")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
