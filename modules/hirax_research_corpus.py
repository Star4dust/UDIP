"""\
Deterministic HIRAX-style validation corpus generator.

Produces 50 HDF5 files covering a realistic range of:
  * clean observations
  * noisy observations
  * RFI-heavy sessions
  * calibration drift
  * closure inconsistencies
  * polarization inconsistencies
  * malformed / missing-metadata variants
  * mixed-condition sessions

Determinism
-----------
A single master seed is used to derive per-file seeds with a hash of
the file id. Re-running the script with the same seed and the same
output directory produces byte-identical files.

Public surface
--------------
  * `generate_validation_corpus(output_dir, *, n_files=50, seed=20260610)`
  * `build_corpus_manifest(corpus_dir)`
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


N_ANTENNAS = 8
N_BASELINES = N_ANTENNAS * (N_ANTENNAS - 1) // 2  # 28
N_FREQ = 64
N_TIME = 32
N_POL = 4
REST_FREQUENCY_MHZ = 1420.40575177
BAND_START_MHZ = 700.0
BAND_END_MHZ = 800.0
DISH_DIAMETER_M = 6.0


def _seed_for(file_id: str, master_seed: int) -> np.random.Generator:
    h = hashlib.sha256(f"{file_id}|{master_seed}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _gaussian_beam(ant_pos: np.ndarray, source_lm: np.ndarray) -> np.ndarray:
    """|beam|^2 at each antenna for a point source at direction (l, m)."""
    # crude: 1 at boresight, falls off as exp(-|theta|^2 / sigma^2)
    ant_pos = np.asarray(ant_pos, dtype=float)
    if ant_pos.ndim == 1:
        ant_pos = ant_pos.reshape(1, -1)
    # use x,y separation only
    theta2 = ant_pos[:, 0] ** 2 + ant_pos[:, 1] ** 2
    sigma2 = (DISH_DIAMETER_M * 0.3) ** 2
    return np.exp(-0.5 * theta2 / sigma2)


def _simulate_visibilities(
    rng: np.random.Generator,
    *,
    n_time: int,
    n_freq: int,
    n_baselines: int,
    n_pol: int,
    noise_level: float,
    source_lm: tuple[float, float] = (0.0, 0.0),
    point_source_flux: float = 1.0,
    drift_rfi: bool = False,
    burst_rfi: bool = False,
    rfi_strength: float = 5.0,
    closure_perturbation: float = 0.0,
    gain_drift_per_time: float = 0.0,
    polarization_leak: float = 0.0,
) -> np.ndarray:
    """Simulate complex visibilities for one observation."""
    vis = np.zeros((n_time, n_freq, n_baselines, n_pol), dtype=np.complex64)
    # Per-antenna positions in wavelengths, just for beam factor.
    ant_pos = rng.normal(0.0, DISH_DIAMETER_M, size=(N_ANTENNAS, 2))
    beam = _gaussian_beam(ant_pos, np.array(source_lm))  # (N_ANT,)
    # Construct antenna pairs.
    pairs = []
    for i in range(N_ANTENNAS):
        for j in range(i + 1, N_ANTENNAS):
            pairs.append((i, j))
    pairs = pairs[:n_baselines]
    # Per-time, per-frequency phase (geometric) + thermal noise.
    for t in range(n_time):
        for f in range(n_freq):
            for bi, (i, j) in enumerate(pairs):
                # Geometric phase from the source direction.
                beam_factor = beam[i] * beam[j]
                phase = 2.0 * math.pi * rng.uniform(0, 1)
                amp = beam_factor * point_source_flux / n_freq
                # Per-polarisation.
                for p in range(n_pol):
                    signal = amp * np.exp(1j * phase)
                    noise = rng.normal(0, noise_level / n_freq) + 1j * rng.normal(0, noise_level / n_freq)
                    vis[t, f, bi, p] = signal + noise
    # Per-time gain drift (applies to amplitude scaling).
    if gain_drift_per_time > 0:
        gain_amp = 1.0 + gain_drift_per_time * np.linspace(-1.0, 1.0, n_time)[:, None, None, None]
        gain_phase = gain_drift_per_time * np.cumsum(rng.normal(0, 0.01, n_time))[:, None, None, None]
        vis = vis * gain_amp * np.exp(1j * gain_phase)
    # Polarisation leakage between pol 0 -> 1.
    if polarization_leak > 0:
        leak = polarization_leak
        vis[..., 1] = (1.0 - leak) * vis[..., 1] + leak * vis[..., 0]
    # Closure phase perturbation: add a per-baseline phase ramp.
    if closure_perturbation > 0:
        phase_pert = closure_perturbation * np.linspace(-1.0, 1.0, n_baselines)[None, None, :, None]
        vis = vis * np.exp(1j * phase_pert)
    # Drifting RFI: a moving line in (t, f).
    if drift_rfi:
        slope = rng.uniform(0.2, 0.6) * rng.choice([-1.0, 1.0])
        for t in range(n_time):
            f = int(0.4 * n_freq + slope * t)
            f = max(0, min(f, n_freq - 1))
            vis[t, max(0, f - 1):min(n_freq, f + 2), :, :] += rfi_strength + 0j
    # Burst RFI: a hot band of channels and times.
    if burst_rfi:
        f0 = rng.integers(0.2 * n_freq, 0.8 * n_freq)
        t0 = rng.integers(0, n_time)
        vis[max(0, t0 - 2):min(n_time, t0 + 3), max(0, f0 - 2):min(n_freq, f0 + 3), :, :] += rfi_strength + 0j
    return vis.astype(np.complex64)


def _build_antenna_positions() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0, DISH_DIAMETER_M, size=(N_ANTENNAS, 3))


def _baselines_from_antennas(ant_pos: np.ndarray) -> np.ndarray:
    out = []
    for i in range(N_ANTENNAS):
        for j in range(i + 1, N_ANTENNAS):
            out.append(ant_pos[j] - ant_pos[i])
    return np.asarray(out)[:N_BASELINES]


def _frequency_axis(n_freq: int) -> np.ndarray:
    return np.linspace(BAND_START_MHZ, BAND_END_MHZ, n_freq) * 1.0e6  # Hz


def _timestamp(seed: int) -> datetime:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # Spread observations over ~30 days.
    return base + timedelta(hours=int((seed * 17) % (30 * 24)))


def _write_one(
    path: Path,
    file_id: str,
    *,
    noise_level: float,
    point_source_flux: float,
    drift_rfi: bool,
    burst_rfi: bool,
    rfi_strength: float,
    closure_perturbation: float,
    gain_drift_per_time: float,
    polarization_leak: float,
    malformed: bool,
    missing_uv: bool,
    missing_calib: bool,
    missing_timestamp: bool,
    source_lm: tuple[float, float],
    master_seed: int,
) -> dict[str, Any]:
    rng = _seed_for(file_id, master_seed)
    ant_pos = _build_antenna_positions()
    baselines = _baselines_from_antennas(ant_pos)
    freq_hz = _frequency_axis(N_FREQ)
    pol_labels = ["XX", "XY", "YX", "YY"][:N_POL]
    vis = _simulate_visibilities(
        rng,
        n_time=N_TIME,
        n_freq=N_FREQ,
        n_baselines=N_BASELINES,
        n_pol=N_POL,
        noise_level=noise_level,
        source_lm=source_lm,
        point_source_flux=point_source_flux,
        drift_rfi=drift_rfi,
        burst_rfi=burst_rfi,
        rfi_strength=rfi_strength,
        closure_perturbation=closure_perturbation,
        gain_drift_per_time=gain_drift_per_time,
        polarization_leak=polarization_leak,
    )
    timestamp = _timestamp(rng.integers(0, 1000))
    with h5py.File(path, "w") as f:
        # Top-level attrs (instruments / observation metadata).
        f.attrs["instrument"] = "HIRAX"
        f.attrs["dish_diameter_m"] = DISH_DIAMETER_M
        f.attrs["n_antennas"] = N_ANTENNAS
        f.attrs["n_baselines"] = N_BASELINES
        f.attrs["telescope_latitude_deg"] = -30.7139     # Klerefontein
        f.attrs["telescope_longitude_deg"] = 21.4089
        if not missing_timestamp:
            f.attrs["observation_time"] = timestamp.isoformat()
        # Synthetic hot/cold calibration measurements for the Y-factor.
        if not missing_calib:
            f.attrs["calibration_trxr_k"] = 50.0
            f.attrs["calibration_trxr_uncertainty_k"] = 5.0
            f.attrs["calibration_t_hot_k"] = 300.0
            f.attrs["calibration_t_cold_k"] = 10.0
            f.attrs["aperture_efficiency"] = 0.65
        if not malformed:
            vis_grp = f.create_group("visibilities")
            vis_grp.create_dataset("data", data=vis, compression="gzip", compression_opts=4)
            vis_grp.attrs["axis_labels"] = np.array(["time", "frequency", "baseline", "polarization"], dtype="S16")
            vis_grp.attrs["component_labels"] = np.array(pol_labels, dtype="S2")
            f.create_dataset("frequencies", data=freq_hz)
            f["frequencies"].attrs["axis_labels"] = np.array(["frequency"], dtype="S9")
            f["frequencies"].attrs["units"] = "Hz"
            f.create_dataset("antenna_positions", data=ant_pos)
            f["antenna_positions"].attrs["axis_labels"] = np.array(["antenna", "component"], dtype="S18")
            f["antenna_positions"].attrs["component_labels"] = np.array(["x", "y", "z"], dtype="S1")
            if not missing_uv:
                f.create_dataset("uv_coordinates", data=baselines[:, :2])
                f["uv_coordinates"].attrs["axis_labels"] = np.array(["baseline", "component"], dtype="S18")
                f["uv_coordinates"].attrs["component_labels"] = np.array(["u", "v"], dtype="S1")
            if not missing_calib:
                # Y-factor: per-frequency power (average over time + baselines + pols).
                power_hot_arr = np.abs(vis)
                f.create_dataset("power_hot", data=power_hot_arr.mean(axis=(0, 2)))
                f.create_dataset("power_cold", data=power_hot_arr.mean(axis=(0, 2)) * 0.2)
        else:
            # Malformed: write a corrupted / empty dataset.
            vis_grp = f.create_group("visibilities")
            vis_grp.create_dataset("data", data=np.zeros((2, 2, 2, 2), dtype=np.complex64))
            f.create_dataset("frequencies", data=np.zeros(2))
            f.create_dataset("antenna_positions", data=np.zeros((1, 3)))
    return {
        "id": file_id,
        "file": path.name,
        "category": _category_label(
            drift_rfi=drift_rfi, burst_rfi=burst_rfi, noise_level=noise_level,
            closure_perturbation=closure_perturbation, gain_drift_per_time=gain_drift_per_time,
            polarization_leak=polarization_leak, malformed=malformed, missing_uv=missing_uv,
            missing_calib=missing_calib,
        ),
        "observation_time": timestamp.isoformat(),
        "noise_level": noise_level,
        "rfi": (drift_rfi or burst_rfi),
        "rfi_strength": rfi_strength,
        "closure_perturbation": closure_perturbation,
        "gain_drift_per_time": gain_drift_per_time,
        "polarization_leak": polarization_leak,
        "malformed": malformed,
        "missing_uv": missing_uv,
        "missing_calib": missing_calib,
        "missing_timestamp": missing_timestamp,
    }


def _category_label(
    *,
    drift_rfi: bool, burst_rfi: bool, noise_level: float,
    closure_perturbation: float, gain_drift_per_time: float,
    polarization_leak: float, malformed: bool, missing_uv: bool, missing_calib: bool,
) -> str:
    if malformed:
        return "malformed metadata"
    if missing_uv:
        return "missing uv geometry"
    if missing_calib:
        return "missing calibration"
    if drift_rfi and burst_rfi:
        return "rfi-heavy sessions"
    if drift_rfi or burst_rfi:
        return "rfi-contaminated sessions"
    if closure_perturbation > 0.1:
        return "closure inconsistencies"
    if polarization_leak > 0.1:
        return "polarization inconsistencies"
    if gain_drift_per_time > 0.1:
        return "partial calibration drift"
    if noise_level > 0.2:
        return "noisy observations"
    if noise_level < 0.05:
        return "clean observations"
    return "mixed-condition sessions"


def _file_specs(master_seed: int) -> list[dict[str, Any]]:
    """Build the deterministic 50-file recipe."""
    specs: list[dict[str, Any]] = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # 8 clean
    for k in range(8):
        specs.append(dict(
            file_id=f"clean_{k:02d}",
            noise_level=0.02 + 0.005 * k,
            point_source_flux=1.0,
            drift_rfi=False, burst_rfi=False, rfi_strength=0.0,
            closure_perturbation=0.0, gain_drift_per_time=0.0, polarization_leak=0.0,
            malformed=False, missing_uv=False, missing_calib=False, missing_timestamp=False,
            source_lm=(0.0, 0.0),
        ))
    # 8 noisy
    for k in range(8):
        specs.append(dict(
            file_id=f"noisy_{k:02d}",
            noise_level=0.25 + 0.04 * k,
            point_source_flux=1.0,
            drift_rfi=False, burst_rfi=False, rfi_strength=0.0,
            closure_perturbation=0.0, gain_drift_per_time=0.0, polarization_leak=0.0,
            malformed=False, missing_uv=False, missing_calib=False, missing_timestamp=False,
            source_lm=(0.0, 0.0),
        ))
    # 8 RFI-heavy (mix of drift + burst)
    for k in range(8):
        specs.append(dict(
            file_id=f"rfi_heavy_{k:02d}",
            noise_level=0.05,
            point_source_flux=1.0,
            drift_rfi=(k % 2 == 0), burst_rfi=(k % 2 == 1), rfi_strength=8.0 + 0.4 * k,
            closure_perturbation=0.0, gain_drift_per_time=0.0, polarization_leak=0.0,
            malformed=False, missing_uv=False, missing_calib=False, missing_timestamp=False,
            source_lm=(0.0, 0.0),
        ))
    # 4 partial calibration drift
    for k in range(4):
        specs.append(dict(
            file_id=f"calib_drift_{k:02d}",
            noise_level=0.05,
            point_source_flux=1.0,
            drift_rfi=False, burst_rfi=False, rfi_strength=0.0,
            closure_perturbation=0.0, gain_drift_per_time=0.15 + 0.05 * k, polarization_leak=0.0,
            malformed=False, missing_uv=False, missing_calib=False, missing_timestamp=False,
            source_lm=(0.0, 0.0),
        ))
    # 4 closure inconsistencies
    for k in range(4):
        specs.append(dict(
            file_id=f"closure_inconsistent_{k:02d}",
            noise_level=0.05,
            point_source_flux=1.0,
            drift_rfi=False, burst_rfi=False, rfi_strength=0.0,
            closure_perturbation=0.2 + 0.05 * k, gain_drift_per_time=0.0, polarization_leak=0.0,
            malformed=False, missing_uv=False, missing_calib=False, missing_timestamp=False,
            source_lm=(0.0, 0.0),
        ))
    # 4 polarization inconsistencies
    for k in range(4):
        specs.append(dict(
            file_id=f"pol_inconsistent_{k:02d}",
            noise_level=0.05,
            point_source_flux=1.0,
            drift_rfi=False, burst_rfi=False, rfi_strength=0.0,
            closure_perturbation=0.0, gain_drift_per_time=0.0, polarization_leak=0.15 + 0.05 * k,
            malformed=False, missing_uv=False, missing_calib=False, missing_timestamp=False,
            source_lm=(0.0, 0.0),
        ))
    # 4 missing uv
    for k in range(4):
        specs.append(dict(
            file_id=f"missing_uv_{k:02d}",
            noise_level=0.05,
            point_source_flux=1.0,
            drift_rfi=False, burst_rfi=False, rfi_strength=0.0,
            closure_perturbation=0.0, gain_drift_per_time=0.0, polarization_leak=0.0,
            malformed=False, missing_uv=True, missing_calib=False, missing_timestamp=False,
            source_lm=(0.0, 0.0),
        ))
    # 4 missing calib
    for k in range(4):
        specs.append(dict(
            file_id=f"missing_calib_{k:02d}",
            noise_level=0.05,
            point_source_flux=1.0,
            drift_rfi=False, burst_rfi=False, rfi_strength=0.0,
            closure_perturbation=0.0, gain_drift_per_time=0.0, polarization_leak=0.0,
            malformed=False, missing_uv=False, missing_calib=True, missing_timestamp=False,
            source_lm=(0.0, 0.0),
        ))
    # 2 malformed
    for k in range(2):
        specs.append(dict(
            file_id=f"malformed_{k:02d}",
            noise_level=0.0,
            point_source_flux=0.0,
            drift_rfi=False, burst_rfi=False, rfi_strength=0.0,
            closure_perturbation=0.0, gain_drift_per_time=0.0, polarization_leak=0.0,
            malformed=True, missing_uv=True, missing_calib=True, missing_timestamp=True,
            source_lm=(0.0, 0.0),
        ))
    # 4 mixed-condition (light drift + light closure)
    for k in range(4):
        specs.append(dict(
            file_id=f"mixed_{k:02d}",
            noise_level=0.10,
            point_source_flux=1.0,
            drift_rfi=(k % 2 == 0), burst_rfi=False, rfi_strength=3.0,
            closure_perturbation=0.05, gain_drift_per_time=0.05, polarization_leak=0.05,
            malformed=False, missing_uv=False, missing_calib=False, missing_timestamp=False,
            source_lm=(0.01, 0.0),
        ))
    return specs


def generate_validation_corpus(
    output_dir: str | Path,
    *,
    n_files: int = 50,
    seed: int = 20260610,
) -> dict[str, Any]:
    """Generate the deterministic 50-file HIRAX-style validation corpus."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _file_specs(seed)[:n_files]
    assert len(specs) == n_files, f"expected {n_files} specs, got {len(specs)}"
    catalog: list[dict[str, Any]] = []
    for spec in specs:
        path = output_dir / f"{spec['file_id']}.h5"
        file_id = spec["file_id"]
        spec_for_call = {k: v for k, v in spec.items() if k != "file_id"}
        meta = _write_one(path, file_id, master_seed=seed, **spec_for_call)
        catalog.append(meta)
    return {
        "n_files": n_files,
        "seed": seed,
        "output_dir": str(output_dir),
        "files": [c["file"] for c in catalog],
        "catalog": catalog,
    }


def build_corpus_manifest(corpus_dir: str | Path) -> dict[str, Any]:
    corpus_dir = Path(corpus_dir)
    files = sorted(corpus_dir.glob("*.h5"))
    manifest = {
        "version": 2,
        "description": "Deterministic 50-file HIRAX-style research-grade regression corpus.",
        "datasets": [],
    }
    for path in files:
        try:
            with h5py.File(path, "r") as f:
                attrs = dict(f.attrs)
                expected_behavior = _expected_behavior_for(path.name, attrs)
                manifest["datasets"].append({
                    "id": path.stem,
                    "category": _category_from_filename(path.stem),
                    "file": path.name,
                    "preferred_dataset_path": "visibilities/data",
                    "target_freq_bins": 32,
                    "expected_scientific_behavior": expected_behavior,
                    "observation_metadata_summary": {
                        "instrument": attrs.get("instrument", "HIRAX"),
                        "n_antennas": int(attrs.get("n_antennas", 0)),
                        "n_baselines": int(attrs.get("n_baselines", 0)),
                        "has_uv_coordinates": "uv_coordinates" in f,
                        "has_calibration_metadata": ("calibration_trxr_k" in attrs) or ("power_hot" in f),
                        "has_timestamp": "observation_time" in attrs,
                    },
                })
        except Exception as exc:  # pragma: no cover - defensive
            manifest["datasets"].append({
                "id": path.stem,
                "category": "uncategorized",
                "file": path.name,
                "error": str(exc),
            })
    return manifest


def _category_from_filename(name: str) -> str:
    if name.startswith("clean"): return "clean observations"
    if name.startswith("noisy"): return "noisy observations"
    if name.startswith("rfi_heavy"): return "rfi-heavy sessions"
    if name.startswith("calib_drift"): return "partial calibration drift"
    if name.startswith("closure_inconsistent"): return "closure inconsistencies"
    if name.startswith("pol_inconsistent"): return "polarization inconsistencies"
    if name.startswith("missing_uv"): return "missing uv geometry"
    if name.startswith("missing_calib"): return "missing calibration"
    if name.startswith("malformed"): return "malformed metadata"
    if name.startswith("mixed"): return "mixed-condition sessions"
    return "uncategorized"


def _expected_behavior_for(name: str, attrs: dict) -> str:
    cat = _category_from_filename(name)
    if cat == "clean observations":
        return "Tsys, beam-squint and wedge/window power all nominal; high research confidence."
    if cat == "noisy observations":
        return "Higher Tsys / lower confidence; foreground wedge still dominates."
    if cat == "rfi-heavy sessions":
        return "Spectral kurtosis + drifting-RFI flag a substantial fraction; confidence drops."
    if cat == "partial calibration drift":
        return "Beam-squint increases; gain-amplitude drift appears in stability maps."
    if cat == "closure inconsistencies":
        return "Beam-squint offset and wedge/window ratio degraded; confidence drops."
    if cat == "polarization inconsistencies":
        return "Polarisation diagnostics flag; cross-pol leakage visible in Tsys estimate."
    if cat == "missing uv geometry":
        return "uv-dependent diagnostics disabled; P(k_perp, k_parallel) emitted as warning."
    if cat == "missing calibration":
        return "Trxr unavailable; default HRMS value with uncertainty band reported."
    if cat == "malformed metadata":
        return "Record is quarantined; calibration research block emits warnings only."
    if cat == "mixed-condition sessions":
        return "Moderate confidence; wedge / window power ratio close to 1."
    return "Physics-consistent output."


__all__ = [
    "generate_validation_corpus",
    "build_corpus_manifest",
]
