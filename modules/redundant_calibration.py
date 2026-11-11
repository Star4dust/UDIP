"""Redundant baseline calibration for HIRAX.

A redundant array (HIRAX has 8 close-packed dishes in a regular grid
configuration) observes the same (u,v) baseline multiple times via
different physical antenna pairs.  Because the sky signal sampled by
all redundant measurements is identical, the only thing that can differ
between them is gain-like noise.  This module:

  * detects redundant baseline groups from an explicit `antenna_positions`
    (preferred) or from a list of `baseline_pairs`;
  * averages the visibilities within each redundant group to obtain a
    lower-variance estimate of V_red(uv);
  * computes per-pair residual statistics (a single complex gain per
    baseline per polarization);
  * summarises the calibration improvement (sigma_pre / sigma_post);
  * returns all of the above as plain dicts/ndarrays with explicit
    `warnings` and `uncertainties` blocks, so downstream stages can
    distinguish measured from estimated values.

Headless, dependency-light, and the routine does not mutate the input
visibility array.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float) -> tuple[float, bool]:
    if value is None:
        return default, False
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default, False
    if not math.isfinite(x):
        return default, False
    return x, True


def _baseline_uv_from_pair(
    pair: tuple[Any, Any],
    antenna_positions: np.ndarray | None,
) -> np.ndarray | None:
    if antenna_positions is None or antenna_positions.ndim != 2:
        return None
    if antenna_positions.shape[1] < 2:
        return None
    a, b = pair
    try:
        ia, ib = int(a), int(b)
    except (TypeError, ValueError):
        return None
    if ia < 0 or ib < 0 or ia >= antenna_positions.shape[0] or ib >= antenna_positions.shape[0]:
        return None
    uv = antenna_positions[ib, :2] - antenna_positions[ia, :2]
    return np.asarray(uv, dtype=float)


def _baseline_uv_observed(
    pair: tuple[Any, Any],
    uv_coordinates: np.ndarray | None,
    baseline_index: int,
) -> np.ndarray | None:
    if uv_coordinates is None or uv_coordinates.ndim != 2:
        return None
    if baseline_index < 0 or baseline_index >= uv_coordinates.shape[0]:
        return None
    if uv_coordinates.shape[1] < 2:
        return None
    return np.asarray(uv_coordinates[baseline_index, :2], dtype=float)


# ---------------------------------------------------------------------------
# Redundant-group detection
# ---------------------------------------------------------------------------

def detect_redundant_groups(
    baseline_pairs: Sequence[tuple[Any, Any]] | None = None,
    *,
    antenna_positions: np.ndarray | None = None,
    uv_coordinates: np.ndarray | None = None,
    tolerance_m: float = 0.5,
    wavelength_m: float | None = None,
) -> dict[str, Any]:
    """Detect groups of baselines that sample the same (u,v) coordinate.

    Parameters
    ----------
    baseline_pairs : sequence of (antenna_i, antenna_j) tuples
        Preferred. Used with `antenna_positions` to derive uv.
    antenna_positions : ndarray, shape (N_ant, 2 or 3)
        Required when `baseline_pairs` is given. Provides positions in
        metres.
    uv_coordinates : ndarray, shape (N_baseline, 2 or 3)
        Fallback if `baseline_pairs` is not available. Grouping is then
        done on the observed uv positions only.
    tolerance_m : float
        Two baselines are redundant when their physical separation
        differs by less than this many metres.  In a regular 8-element
        grid (HIRAX prototype) the default 0.5 m is well below the
        dish pitch and is a safe threshold.
    wavelength_m : float, optional
        If supplied, also reports the (u,v) vector in wavelengths. The
        actual grouping is still done in metres.

    Returns
    -------
    dict with:
        groups : list[list[int]]   baseline-index lists per redundant group
        uv_physical_m : list[ndarray]  representative uv per group (m)
        group_sizes : list[int]
        redundancy_factor : int    max number of baselines per group
        n_unique_groups : int
        n_redundant_baselines : int
        warnings : list[str]
    """
    out_warnings: list[str] = []

    uv_m: list[np.ndarray] = []
    if baseline_pairs is not None and antenna_positions is not None:
        n_pairs = len(baseline_pairs)
        for i, pair in enumerate(baseline_pairs):
            uv = _baseline_uv_from_pair(pair, antenna_positions)
            if uv is None:
                out_warnings.append(f"baseline {i}: unable to derive uv from pair, skipped")
                uv = np.array([np.nan, np.nan])
            uv_m.append(uv)
    elif uv_coordinates is not None:
        for i in range(uv_coordinates.shape[0]):
            uv = _baseline_uv_observed(None, uv_coordinates, i)
            if uv is None:
                out_warnings.append(f"baseline {i}: unable to read uv, skipped")
                uv = np.array([np.nan, np.nan])
            uv_m.append(uv)
    else:
        return {
            "groups": [],
            "uv_physical_m": [],
            "group_sizes": [],
            "redundancy_factor": 0,
            "n_unique_groups": 0,
            "n_redundant_baselines": 0,
            "n_baselines": 0,
            "warnings": out_warnings + ["no baseline_pairs or uv_coordinates supplied"],
        }

    uv_arr = np.asarray(uv_m, dtype=float)
    n = uv_arr.shape[0]

    # Build groups via a single-linkage-style union-find on the (u,v) positions.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    valid_mask = np.all(np.isfinite(uv_arr), axis=1)
    valid_idx = np.where(valid_mask)[0]
    for i_pos in range(len(valid_idx)):
        i = int(valid_idx[i_pos])
        for j_pos in range(i_pos + 1, len(valid_idx)):
            j = int(valid_idx[j_pos])
            if np.linalg.norm(uv_arr[i] - uv_arr[j]) < tolerance_m:
                union(i, j)

    # Collect groups.
    groups_map: dict[int, list[int]] = {}
    for k in range(n):
        if not valid_mask[k]:
            continue
        r = find(k)
        groups_map.setdefault(r, []).append(k)

    sorted_groups = sorted(groups_map.values(), key=lambda g: g[0])
    group_sizes = [len(g) for g in sorted_groups]
    uv_rep = [np.mean(uv_arr[g], axis=0).tolist() for g in sorted_groups]
    redundancy_factor = int(max(group_sizes)) if group_sizes else 0
    n_redundant = int(sum(s for s in group_sizes if s > 1))

    # 1) Insufficient-redundancy warning.
    if redundancy_factor < 2:
        out_warnings.append(
            "insufficient redundancy for noise-averaging calibration: "
            f"max group size is {redundancy_factor}"
        )

    return {
        "groups": sorted_groups,
        "uv_physical_m": uv_rep,
        "group_sizes": group_sizes,
        "redundancy_factor": redundancy_factor,
        "n_unique_groups": len(sorted_groups),
        "n_redundant_baselines": n_redundant,
        "n_baselines": int(np.sum(valid_mask)),
        "tolerance_m": float(tolerance_m),
        "wavelength_m": float(wavelength_m) if wavelength_m is not None else None,
        "warnings": out_warnings,
    }


# ---------------------------------------------------------------------------
# Redundancy-averaged visibilities
# ---------------------------------------------------------------------------

def redundancy_averaged_visibilities(
    visibilities: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Average visibilities within each redundant group.

    The input visibility is preserved in shape (..., N_baseline, ...);
    the output is the same shape with the trailing baseline axis
    replaced by the redundancy-averaged estimate at each group's
    representative baseline.

    Parameters
    ----------
    visibilities : ndarray
        Shape (..., N_baseline, [pol]).  Complex expected.
    groups : list[list[int]]
        Baseline indices belonging to each redundant group.
    weights : ndarray, optional
        Per-baseline weights of the same length as the baseline axis
        (or broadcastable).  Defaults to uniform weights.

    Returns
    -------
    dict with:
        averaged_visibilities : ndarray
            Shape (..., N_groups, [pol])  with complex dtype.
        group_weights : ndarray   (N_groups,)  sum of weights per group
        per_group_sigma : ndarray (N_groups,)  scatter within each group
        n_groups : int
        warnings : list[str]
    """
    out_warnings: list[str] = []
    vis = np.asarray(visibilities)
    if vis.ndim < 1:
        return _empty_redundancy_output(out_warnings + ["empty input"])
    n_baseline = vis.shape[-1] if vis.ndim >= 2 else 1
    pol_axis = vis.ndim - 1 if vis.ndim >= 2 and np.iscomplexobj(vis) else None
    if pol_axis is None:
        pol_axis = vis.ndim  # synthetic; real data collapses to last axis

    if n_baseline == 0 or len(groups) == 0:
        return _empty_redundancy_output(out_warnings + ["no groups supplied"])

    if weights is None:
        w_per = np.ones(n_baseline, dtype=float)
    else:
        w_per = np.asarray(weights, dtype=float)
        if w_per.shape != (n_baseline,):
            w_per = np.broadcast_to(w_per.ravel(), (n_baseline,)).copy()

    # Stack groups as boolean masks for clarity.
    n_groups = len(groups)
    averaged = []
    group_weights = np.zeros(n_groups, dtype=float)
    per_group_sigma = np.zeros(n_groups, dtype=float)
    for g_idx, group in enumerate(groups):
        idx = [b for b in group if 0 <= b < n_baseline]
        if not idx:
            averaged.append(np.zeros(vis.shape[:-1], dtype=complex))
            continue
        sub = vis[..., idx]                 # (..., k, [pol])
        w_sub = w_per[idx][None, ...]       # (1, k)
        if sub.ndim >= 2 and sub.shape[-1] != len(idx):
            # Collapse baseline axis only.
            pass
        weighted = sub * w_sub
        denom = float(np.sum(w_sub))
        group_weights[g_idx] = denom
        avg = np.sum(weighted, axis=-1) / denom if denom > 0 else np.zeros(sub.shape[:-1], dtype=complex)
        # Scatter: 1-sigma deviation within the group.
        if sub.shape[-1] > 1:
            amp = np.abs(sub)
            per_group_sigma[g_idx] = float(np.sqrt(np.mean(amp ** 2) - np.mean(amp) ** 2)) if sub.size else 0.0
        else:
            per_group_sigma[g_idx] = 0.0
        averaged.append(avg)

    # Stack: shape becomes (..., N_groups, [pol])
    # Always ensure a trailing pol axis of size 1 so downstream consumers
    # can rely on the (..., N_groups, 1) shape contract.
    try:
        out = np.stack(averaged, axis=-1)
    except Exception as exc:  # pragma: no cover - defensive
        out_warnings.append(f"stack failed: {exc}; returning first group only")
        out = averaged[0][..., None]
    if out.ndim < vis.ndim:
        out = out[..., None]

    return {
        "averaged_visibilities": out,
        "group_weights": group_weights,
        "per_group_sigma": per_group_sigma,
        "n_groups": int(n_groups),
        "warnings": out_warnings,
    }


# ---------------------------------------------------------------------------
# Per-baseline gain estimates
# ---------------------------------------------------------------------------

def _complex_gain(measured: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Single complex gain via the closed-form ratio V_meas / V_model.

    Returns 1.0 where V_model is too small.
    """
    m = np.asarray(measured, dtype=np.complex128)
    r = np.asarray(model, dtype=np.complex128)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(np.abs(r) > 0, m / r, 1.0 + 0j)
    return g


def _sigma_complex(gains: np.ndarray) -> tuple[float, float]:
    """Estimate the 1-sigma amplitude and phase scatter of a complex
    gain sample.  Returns (sigma_amp, sigma_phase_rad).
    """
    arr = np.asarray(gains, dtype=np.complex128)
    if arr.size == 0:
        return float("nan"), float("nan")
    amp = np.abs(arr)
    ph = np.angle(arr)
    return float(np.std(amp)), float(np.std(ph))


def redundancy_calibration_residual(
    visibilities: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    averaged: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute per-baseline gain residuals and per-group residual
    statistics.  The 'residual' is the deviation of the per-baseline
    measurement from the redundancy-averaged estimate, expressed as
    both an amplitude ratio and a phase offset.

    Parameters
    ----------
    visibilities : ndarray
        Input visibilities, shape (..., N_baseline, [pol]).
    groups : list[list[int]]
        Baseline indices per redundant group.
    averaged : ndarray, optional
        Pre-computed redundancy-averaged visibilities (from
        `redundancy_averaged_visibilities`).  If `None`, they are
        computed internally.
    weights : ndarray, optional
        Passed to the averaging if `averaged` is None.

    Returns
    -------
    dict with:
        gain_per_baseline : ndarray of complex, shape (N_baseline,)
        sigma_amp : ndarray (N_baseline,)  per-baseline 1-sigma |g|
        sigma_phase_rad : ndarray (N_baseline,)  per-baseline 1-sigma phi
        amplitude_residual_rms : float
        phase_residual_rms_rad : float
        averaged_visibilities : ndarray  (echoed)
        warnings : list[str]
    """
    out_warnings: list[str] = []
    vis = np.asarray(visibilities, dtype=np.complex128)
    n_baseline = vis.shape[-1]
    if averaged is None:
        avg_bundle = redundancy_averaged_visibilities(vis, groups, weights=weights)
        averaged_vis = avg_bundle["averaged_visibilities"]
    else:
        averaged_vis = np.asarray(averaged, dtype=np.complex128)
        avg_bundle = {"warnings": []}

    # Map each baseline to a group and the averaged visibility at that slot.
    baseline_to_group: list[int] = [-1] * n_baseline
    for g_idx, group in enumerate(groups):
        for b in group:
            if 0 <= b < n_baseline and baseline_to_group[b] == -1:
                baseline_to_group[b] = g_idx

    g_per = np.ones(n_baseline, dtype=np.complex128)
    sig_amp = np.zeros(n_baseline, dtype=float)
    sig_phi = np.zeros(n_baseline, dtype=float)
    for b in range(n_baseline):
        g_idx = baseline_to_group[b]
        if g_idx < 0 or g_idx >= averaged_vis.shape[-1]:
            out_warnings.append(f"baseline {b} not in any group; gain set to 1+0j")
            continue
        # Average over all non-baseline axes for a single complex ratio.
        m_flat = vis[..., b].ravel()
        a_flat = averaged_vis[..., g_idx].ravel()
        # If sizes don't match (e.g. single time vs full cube), use mean.
        m_scalar = np.mean(m_flat) if m_flat.size else 1.0 + 0j
        a_scalar = np.mean(a_flat) if a_flat.size else 1.0 + 0j
        g = _complex_gain(np.atleast_1d(m_scalar), np.atleast_1d(a_scalar))
        g_per[b] = g[0]
        sig_amp[b], sig_phi[b] = _sigma_complex(g)

    # Aggregate residual statistics (over all baselines and all axes).
    ratios = []
    phase_offsets = []
    for b in range(n_baseline):
        g_idx = baseline_to_group[b]
        if g_idx < 0:
            continue
        m_flat = vis[..., b].ravel()
        a_flat = averaged_vis[..., g_idx].ravel()
        n = min(m_flat.size, a_flat.size)
        if n == 0:
            continue
        for k in range(n):
            denom = a_flat[k]
            if abs(denom) == 0:
                continue
            r = m_flat[k] / denom
            ratios.append(abs(r) - 1.0)
            phase_offsets.append(np.angle(r))

    ratios_arr = np.asarray(ratios, dtype=float)
    phase_arr = np.asarray(phase_offsets, dtype=float)
    amp_rms = float(np.sqrt(np.mean(np.square(ratios_arr)))) if ratios_arr.size else float("nan")
    phase_rms = float(np.sqrt(np.mean(np.square(phase_arr)))) if phase_arr.size else float("nan")

    return {
        "gain_per_baseline": g_per,
        "sigma_amp": sig_amp,
        "sigma_phase_rad": sig_phi,
        "baseline_to_group": np.asarray(baseline_to_group, dtype=int),
        "amplitude_residual_rms": amp_rms,
        "phase_residual_rms_rad": phase_rms,
        "averaged_visibilities": averaged_vis,
        "n_baselines_calibrated": int(np.sum(np.asarray(baseline_to_group) >= 0)),
        "per_baseline": {
            "gain": g_per,
            "amp_rms": sig_amp,
            "phase_rms_rad": sig_phi,
            "amplitude_residual_rms": amp_rms,
            "phase_residual_rms_rad": phase_rms,
        },
        "warnings": out_warnings + avg_bundle.get("warnings", []),
    }


# ---------------------------------------------------------------------------
# Improvement metric
# ---------------------------------------------------------------------------

def redundancy_improvement_metric(
    visibilities: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    averaged: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Quantify how much redundancy averaging reduces noise.

    Compares the 1-sigma amplitude scatter per redundant group *before*
    averaging (input scatter across the redundant baselines) to the
    scatter *after* averaging (the per-group 1-sigma returned by
    `redundancy_averaged_visibilities`).  A value > 1 means the
    averaging reduces noise (i.e., the array is truly redundant and
    the baselines were noisy).

    Returns a dict with `improvement_factor`, `sigma_pre`, `sigma_post`,
    `n_groups_compared`, and `warnings`.
    """
    out_warnings: list[str] = []
    vis = np.asarray(visibilities, dtype=np.complex128)
    n_baseline = vis.shape[-1] if vis.ndim >= 2 else 0
    if averaged is None:
        bundle = redundancy_averaged_visibilities(vis, groups, weights=weights)
        averaged = bundle["averaged_visibilities"]
        sigma_post = bundle["per_group_sigma"]
    else:
        sigma_post = np.zeros(len(groups), dtype=float)

    sigma_pre = np.zeros(len(groups), dtype=float)
    for g_idx, group in enumerate(groups):
        idx = [b for b in group if 0 <= b < n_baseline]
        if len(idx) < 2:
            sigma_pre[g_idx] = 0.0
            continue
        amp = np.abs(vis[..., idx])
        if amp.size == 0:
            sigma_pre[g_idx] = 0.0
            continue
        sigma_pre[g_idx] = float(np.sqrt(np.mean(amp ** 2) - np.mean(amp) ** 2))

    # Compare only groups with at least 2 baselines.
    valid = sigma_pre > 0
    if not np.any(valid):
        out_warnings.append("no group has >=2 baselines; improvement undefined")
        return {
            "improvement_factor": float("nan"),
            "sigma_pre": sigma_pre,
            "sigma_post": sigma_post,
            "n_groups_compared": 0,
            "warnings": out_warnings,
        }
    pre = sigma_pre[valid]
    post = sigma_post[valid]
    # Improvement factor: median ratio.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(post > 0, pre / np.maximum(post, 1e-30), 1.0)
    improvement = float(np.median(ratios))
    n_improved = int(np.sum(ratios > 1.0))
    return {
        "improvement_factor": improvement,
        "sigma_pre": sigma_pre,
        "sigma_post": sigma_post,
        "n_groups_compared": int(np.sum(valid)),
        "n_improved_groups": n_improved,
        "mean_improvement_factor": float(np.mean(ratios)) if ratios.size else float("nan"),
        "median_improvement_factor": improvement,
        "warnings": out_warnings,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_redundancy_output(warnings: list[str]) -> dict[str, Any]:
    return {
        "averaged_visibilities": np.zeros((0,), dtype=complex),
        "group_weights": np.zeros((0,), dtype=float),
        "per_group_sigma": np.zeros((0,), dtype=float),
        "n_groups": 0,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper used by the batch pipeline
# ---------------------------------------------------------------------------

def redundant_calibration_summary(
    visibilities: np.ndarray,
    *,
    baseline_pairs: Sequence[tuple[Any, Any]] | None = None,
    antenna_positions: np.ndarray | None = None,
    uv_coordinates: np.ndarray | None = None,
    tolerance_m: float = 0.5,
    weights: np.ndarray | None = None,
    wavelength_m: float | None = None,
) -> dict[str, Any]:
    """One-call bundle used by the per-file research pipeline.

    Returns the groups, averaged visibilities, calibration residual
    statistics, and improvement metric.  All inputs are preserved
    (the function never mutates `visibilities` or any of the metadata
    arrays).
    """
    groups_bundle = detect_redundant_groups(
        baseline_pairs=baseline_pairs,
        antenna_positions=antenna_positions,
        uv_coordinates=uv_coordinates,
        tolerance_m=tolerance_m,
        wavelength_m=wavelength_m,
    )
    groups = groups_bundle["groups"]
    warnings: list[str] = list(groups_bundle["warnings"])
    if not groups:
        return {
            "groups": groups_bundle,
            "averaged": None,
            "residual": None,
            "improvement": None,
            "warnings": warnings + ["no redundant groups detected"],
        }
    averaged = redundancy_averaged_visibilities(visibilities, groups, weights=weights)
    warnings.extend(averaged["warnings"])
    residual = redundancy_calibration_residual(visibilities, groups, averaged=averaged["averaged_visibilities"])
    warnings.extend(residual["warnings"])
    improvement = redundancy_improvement_metric(visibilities, groups, averaged=averaged["averaged_visibilities"])
    warnings.extend(improvement["warnings"])
    return {
        "groups": groups_bundle,
        "averaged": averaged,
        "residual": residual,
        "improvement": improvement,
        "warnings": warnings,
    }


__all__ = [
    "detect_redundant_groups",
    "redundancy_averaged_visibilities",
    "redundancy_calibration_residual",
    "redundancy_improvement_metric",
    "redundant_calibration_summary",
]
