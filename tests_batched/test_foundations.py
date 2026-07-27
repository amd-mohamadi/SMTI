#!/usr/bin/env python
"""
M1 stage-1 foundation tests (CPU only).

Covers the three behaviour-preserving production fixes:

  1. numerically stable ``src.inversion_blackjax.softplus_inv``
  2. dtype-agnostic ``src.blockwise_rmh.build_blockwise_scale_overrides``
  3. ``InversionBlackJAX._prepare_event_arrays`` extraction from
     ``_invert_single_event`` (pure refactor)

Run with:
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        conda run -n smti python tests_batched/test_foundations.py

Anything under ``--quick`` skips the (few-minute) eq00124 inversion.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse
import importlib.util
import json
import math
import sys
import time

import numpy as np

# SMTI_ROOT lets test_refactor_equivalence.py point this module at a pristine
# git-HEAD copy of src/; it defaults to the m1-batched-gpu worktree.
SMTI_DEV = os.environ.get("SMTI_ROOT", "/s0/data/CAPE/smti_workflow/SMTI_dev")
WORKFLOW = "/s0/data/CAPE/smti_workflow/smti_inversion"
M0_SPIKE = os.path.join(WORKFLOW, "m0_spike")
sys.path.insert(0, SMTI_DEV)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.nn as jnn  # noqa: E402

# NOTE: importing src.inversion_blackjax pulls in src.tape_jax, which forces
# jax_enable_x64=True process-wide.
from src.inversion_blackjax import InversionBlackJAX  # noqa: E402

try:  # absent in the pre-fix git-HEAD copy used by test_refactor_equivalence
    from src.inversion_blackjax import softplus_inv  # noqa: E402
except ImportError:  # pragma: no cover
    softplus_inv = None
from src.blockwise_rmh import (  # noqa: E402
    RMHBlock,
    default_rmh_blocks,
    build_blockwise_scale_overrides,
)
from src.data_loader import read_data  # noqa: E402
from src.data_prep import build_location_samples_from_errors  # noqa: E402


# --------------------------------------------------------------------------
# tiny test harness
# --------------------------------------------------------------------------
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


EPS = 1e-6


def old_softplus_inv(x):
    """The pre-fix production expression (inversion_blackjax.py:1001, old)."""
    return jnp.log(jnp.expm1(jnp.maximum(x, EPS)) + EPS)


# --------------------------------------------------------------------------
# Test 1 -- softplus_inv
# --------------------------------------------------------------------------
def test_softplus_inv():
    print("\n=== Test 1: numerically stable softplus_inv ===")
    assert jax.config.jax_enable_x64, "expected x64 enabled by src.tape_jax"
    if softplus_inv is None:
        check("1. src.inversion_blackjax.softplus_inv is importable", False,
              "module-level softplus_inv not found")
        return

    # (a) FP64 agreement with the old (exact in this range) expression.
    x = np.unique(
        np.concatenate(
            [
                np.logspace(-6, np.log10(88.0), 20001),
                np.array([1e-6, 1e-3, 0.5, np.log(2.0), 1.0, 19.9, 20.0, 20.1, 88.0]),
            ]
        )
    )
    xj = jnp.asarray(x, dtype=jnp.float64)
    old = np.asarray(old_softplus_inv(xj))
    new = np.asarray(softplus_inv(xj))
    check(
        "1a.dtype: fp64 input -> fp64 output",
        new.dtype == np.float64 and old.dtype == np.float64,
        f"{new.dtype}",
    )
    absdiff = np.abs(new - old)
    rel = np.where(absdiff == 0.0, 0.0, absdiff / np.maximum(np.abs(old), 1e-300))
    max_rel = float(np.max(rel))
    n_exact = int(np.sum(absdiff == 0.0))
    check(
        "1a. FP64 new vs old rel < 1e-12 on x in [1e-6, 88]",
        np.all(np.isfinite(old)) and max_rel < 1e-12,
        f"max_rel={max_rel:.3e}, bit-identical for {n_exact}/{x.size} points, "
        f"max_abs={float(np.max(absdiff)):.3e}",
    )

    # (b) FP32: finite everywhere up to the clip value 100 (old form overflows).
    x32_np = np.unique(
        np.concatenate(
            [
                np.logspace(-6, 2, 20001),
                np.array([1e-6, 1.0, 20.0, 80.0, 88.0, 88.7, 89.0, 95.0, 100.0]),
            ]
        )
    ).astype(np.float32)
    x32 = jnp.asarray(x32_np, dtype=jnp.float32)
    new32 = softplus_inv(x32)
    old32 = old_softplus_inv(x32)
    check(
        "1b.dtype: fp32 input -> fp32 output (no silent upcast)",
        new32.dtype == jnp.float32,
        f"{new32.dtype}",
    )
    new32_np = np.asarray(new32)
    old32_np = np.asarray(old32)
    n_bad_old = int(np.sum(~np.isfinite(old32_np)))
    check(
        "1b. FP32 new: no inf/nan for x up to 100",
        bool(np.all(np.isfinite(new32_np))),
        f"n_nonfinite_new=0 of {new32_np.size}; old form non-finite at "
        f"{n_bad_old} points (first bad x={float(x32_np[~np.isfinite(old32_np)][0]) if n_bad_old else float('nan'):.4f})",
    )
    # explicit spot check at the clip value used by init_particle
    v100 = float(softplus_inv(jnp.asarray(100.0, dtype=jnp.float32)))
    check(
        "1b. FP32 softplus_inv(100.0) finite and ~= 100",
        math.isfinite(v100) and abs(v100 - 100.0) < 1e-3,
        f"value={v100!r}",
    )

    # extra: with jax_enable_x64 turned off entirely (the fp32 process pattern)
    try:
        jax.config.update("jax_enable_x64", False)
        y = softplus_inv(jnp.asarray(x32_np))
        ok = y.dtype == jnp.float32 and bool(np.all(np.isfinite(np.asarray(y))))
        check(
            "1b. FP32 with jax_enable_x64=False: finite, float32",
            ok,
            f"dtype={y.dtype}",
        )
    finally:
        jax.config.update("jax_enable_x64", True)

    # (c) round trip softplus(softplus_inv(x)) ~= x
    xr = np.logspace(-3, 2, 5001)
    z64 = np.asarray(jnn.softplus(softplus_inv(jnp.asarray(xr, dtype=jnp.float64))))
    err64 = np.abs(z64 - xr)
    tol64 = 3e-6 + 1e-9 * xr  # the +eps inside the log costs ~eps*exp(-x)
    check(
        "1c. round trip FP64: softplus(softplus_inv(x)) ~= x, x in [1e-3, 100]",
        bool(np.all(err64 <= tol64)),
        f"max_abs_err={float(np.max(err64)):.3e}, "
        f"max_rel_err={float(np.max(err64 / xr)):.3e}",
    )
    xr32 = xr.astype(np.float32)
    z32 = np.asarray(
        jnn.softplus(softplus_inv(jnp.asarray(xr32, dtype=jnp.float32)))
    ).astype(np.float64)
    err32 = np.abs(z32 - xr32.astype(np.float64))
    tol32 = 3e-6 + 2e-5 * xr
    check(
        "1c. round trip FP32: softplus(softplus_inv(x)) ~= x, x in [1e-3, 100]",
        bool(np.all(err32 <= tol32)),
        f"max_abs_err={float(np.max(err32)):.3e}, "
        f"max_rel_err={float(np.max(err32 / xr)):.3e}",
    )


# --------------------------------------------------------------------------
# Test 2 -- blockwise_rmh dtype agnosticism
# --------------------------------------------------------------------------
def _reference_scales_old(blocks, initial_stds, current_stds, baseline_scale, min_ratio):
    """Recreate the pre-fix computation with the hardcoded float64 factor."""
    out = {}
    baseline = jnp.asarray(baseline_scale)
    for block in blocks:
        d = len(block.fields)
        optimal_factor = 2.38 / jnp.sqrt(jnp.asarray(d, dtype=jnp.float64))
        initial_mean = jnp.mean(jnp.asarray([initial_stds[f] for f in block.fields]))
        current_mean = jnp.mean(jnp.asarray([current_stds[f] for f in block.fields]))
        scale = optimal_factor * current_mean
        floor = min_ratio * optimal_factor * initial_mean
        scale = jnp.maximum(scale, floor)
        out[block.name] = baseline * scale * jnp.asarray(1.0)
    return out


def test_blockwise_dtype():
    print("\n=== Test 2: dtype-agnostic optimal_factor in blockwise_rmh ===")
    blocks = default_rmh_blocks(dc=False, has_ar=True)
    fields = [f for b in blocks for f in b.fields]
    rng = np.random.default_rng(7)
    initial = {f: float(v) for f, v in zip(fields, rng.uniform(0.05, 1.5, len(fields)))}
    # include a case where the floor binds (current << initial)
    current = {f: 0.02 * v for f, v in initial.items()}
    current[fields[0]] = initial[fields[0]] * 3.0

    new = build_blockwise_scale_overrides(
        blocks, initial, current, baseline_scale=0.7, min_ratio=0.1
    )
    ref = _reference_scales_old(blocks, initial, current, 0.7, 0.1)
    identical = all(
        np.asarray(new[k]).dtype == np.asarray(ref[k]).dtype
        and np.array_equal(np.asarray(new[k]), np.asarray(ref[k]))
        for k in ref
    )
    check(
        "2a. FP64 bit-identical to the pre-fix formula",
        identical,
        ", ".join(f"{k}={float(new[k]):.17g}" for k in sorted(new)),
    )

    # with acceptance adaptation active as well
    acc = {b.name: 0.2 + 0.1 * i for i, b in enumerate(blocks)}
    new_a = build_blockwise_scale_overrides(
        blocks, initial, current, baseline_scale=1.0, block_acceptance_rate=acc
    )
    ref_a = {}
    for b in blocks:
        d = len(b.fields)
        of = 2.38 / jnp.sqrt(jnp.asarray(d, dtype=jnp.float64))
        im = jnp.mean(jnp.asarray([initial[f] for f in b.fields]))
        cm = jnp.mean(jnp.asarray([current[f] for f in b.fields]))
        sc = jnp.maximum(of * cm, 0.1 * of * im)
        tgt = {"source_type": 0.44, "mechanism": 0.30, "noise": 0.44}[b.name]
        mult = jnp.clip(jnp.exp(1.5 * (jnp.asarray(acc[b.name]) - tgt)), 0.5, 4.0)
        ref_a[b.name] = jnp.asarray(1.0) * sc * mult
    identical_a = all(
        np.array_equal(np.asarray(new_a[k]), np.asarray(ref_a[k])) for k in ref_a
    )
    check(
        "2a. FP64 bit-identical with acceptance adaptation",
        identical_a,
        ", ".join(f"{k}={float(new_a[k]):.17g}" for k in sorted(new_a)),
    )

    # float32 inputs must stay float32 (old code upcast to float64)
    initial32 = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in initial.items()}
    current32 = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in current.items()}
    new32 = build_blockwise_scale_overrides(blocks, initial32, current32)
    old32 = _reference_scales_old(blocks, initial32, current32, 1.0, 0.1)
    dtypes_new = {k: str(np.asarray(v).dtype) for k, v in new32.items()}
    dtypes_old = {k: str(np.asarray(v).dtype) for k, v in old32.items()}
    check(
        "2b. float32 stds -> float32 scales (no forced upcast)",
        all(v == "float32" for v in dtypes_new.values()),
        f"new={dtypes_new} old={dtypes_old}",
    )
    close32 = all(
        np.allclose(np.asarray(new32[k], dtype=np.float64),
                    np.asarray(old32[k], dtype=np.float64), rtol=1e-6)
        for k in new32
    )
    check(
        "2b. float32 values match the float64 reference to 1e-6",
        close32,
        ", ".join(f"{k}={float(new32[k]):.8g}" for k in sorted(new32)),
    )

    # and under jax_enable_x64=False there must be no float64 request at all
    try:
        jax.config.update("jax_enable_x64", False)
        s = build_blockwise_scale_overrides(blocks, initial, current)
        ok = all(str(np.asarray(v).dtype) == "float32" for v in s.values())
        check(
            "2b. x64 disabled -> float32 scales",
            ok,
            f"{ {k: str(np.asarray(v).dtype) for k, v in s.items()} }",
        )
    finally:
        jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# eq00124 loading (identical to m0_spike/prepare_eq00124_arrays.py)
# --------------------------------------------------------------------------
EVENT_ID = "eq00124"
DAT_PATH = f"/s0/data/CAPE/smti_workflow/inversion_data/data/{EVENT_ID}/data.dat"
VELOCITY_MODEL_PATH = "/s0/data/CAPE/smti_workflow/cape_surface_corrected.tvel"
INVERSION_OPTIONS = [
    "PPolarity",
    "SHPolarity",
    "P/SHAmplitudeRatio",
    "P/SVAmplitudeRatio",
]
READ_DATA_KW = dict(
    min_p_phase_score=0.55,
    min_s_phase_score=0.55,
    min_ppl_score=0.6,
    min_shpl_score=0.7,
    min_svpl_score=0.7,
    p_polarity_phase="Pz",
)
LOCATION_SAMPLES_N = 20
AZIMUTH_ERROR = 5
TAKEOFF_ERROR = 10
LOCATION_SAMPLE_SEED = 20240216


def _filter_step4(dat_path, include_stacked_stations=True):
    spec = importlib.util.spec_from_file_location(
        "driver4", os.path.join(WORKFLOW, "4_run_inversion.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["driver4"] = mod
    cwd = os.getcwd()
    os.chdir(WORKFLOW)
    sys.path.insert(0, WORKFLOW)
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod._filter_step4_data_dat(
        dat_path, include_stacked_stations=include_stacked_stations
    )


def load_eq00124():
    filtered_path, _summary = _filter_step4(DAT_PATH)
    try:
        data = read_data(
            filtered_path,
            inversion_options=INVERSION_OPTIONS,
            velocity_model_path=VELOCITY_MODEL_PATH,
            **READ_DATA_KW,
        )
    finally:
        try:
            os.unlink(filtered_path)
        except OSError:
            pass
    return data


def make_inversion(data, num_particles):
    return InversionBlackJAX(
        data,
        inversion_options=INVERSION_OPTIONS,
        num_particles=num_particles,
        dc=False,
        gamma_beta_prior=(3.0, 3.0),
        delta_beta_prior=(3.0, 3.0),
        amp_ratio_sigma_prior=2.0,
        amp_ratio_noise_mode="global",
        random_seed=LOCATION_SAMPLE_SEED,
        location_samples_n=LOCATION_SAMPLES_N,
        azimuth_error=AZIMUTH_ERROR,
        takeoff_error=TAKEOFF_ERROR,
        mcmc_kernel="rmh",
        num_mcmc_steps=5,
        smc_target_ess_ratio=0.9,
        max_smc_iterations=60,
        rmh_proposal_scale=0.02,
        num_chains=1,
        chain_execution="sequential",
    )


# --------------------------------------------------------------------------
# Test 3 -- _prepare_event_arrays refactor
# --------------------------------------------------------------------------
def test_prepare_event_arrays(data, inv):
    print("\n=== Test 3a: _prepare_event_arrays vs m0_spike/eq00124_arrays.npz ===")
    rng = np.random.default_rng(LOCATION_SAMPLE_SEED)
    location_samples = build_location_samples_from_errors(
        data,
        rng=rng,
        n_samples=LOCATION_SAMPLES_N,
        azimuth_error=AZIMUTH_ERROR,
        takeoff_error=TAKEOFF_ERROR,
    )
    prep = inv._prepare_event_arrays(data, location_samples=location_samples)

    check(
        "3a. flags: has_pol/has_ar/station_smooth_ar",
        prep["has_pol"] is True
        and prep["has_ar"] is True
        and prep["station_smooth_ar"] is False,
        f"has_pol={prep['has_pol']} has_ar={prep['has_ar']} "
        f"station_smooth_ar={prep['station_smooth_ar']} "
        f"ar_station_context={prep['ar_station_context']}",
    )
    check(
        "3a. location_samples passed through unchanged (identity)",
        prep["location_samples"] is location_samples,
    )

    ref = np.load(os.path.join(M0_SPIKE, "eq00124_arrays.npz"))
    with open(os.path.join(M0_SPIKE, "eq00124_arrays.json")) as fh:
        meta = json.load(fh)
    keys = [
        "a_pol",
        "error_pol",
        "pol_obs",
        "incorrect_prob",
        "a1_ar",
        "a2_ar",
        "amp_ratio_obs",
        "log_ratio_sigma",
    ]
    all_ok = True
    for k in keys:
        got = np.asarray(prep[k])
        want = ref[k]
        same_shape = got.shape == want.shape
        same_dtype = got.dtype == want.dtype
        same_bits = same_shape and np.array_equal(got, want)
        ok = same_shape and same_dtype and same_bits
        all_ok = all_ok and ok
        check(
            f"3a. {k}: bit-identical to npz",
            ok,
            f"shape {got.shape} vs {want.shape}, dtype {got.dtype} vs {want.dtype}"
            + ("" if same_bits else
               f", max|diff|={float(np.max(np.abs(got.astype(float) - want.astype(float)))):.3e}"),
        )
    check(
        "3a. shapes match the recorded N_pol/N_ar/N_loc",
        np.asarray(prep["a_pol"]).shape
        == (meta["N_pol"], meta["N_loc"], 6)
        and np.asarray(prep["a1_ar"]).shape == (meta["N_ar"], meta["N_loc"], 6),
        f"N_pol={meta['N_pol']} N_ar={meta['N_ar']} N_loc={meta['N_loc']}",
    )
    return all_ok, location_samples


# --------------------------------------------------------------------------
# Test 4 -- end-to-end single-event inversion through the refactored path
# --------------------------------------------------------------------------
GAMMA_REF, GAMMA_TOL = 0.247, 0.06
DELTA_REF, DELTA_TOL = -0.145, 0.08


def test_single_event_inversion(data, inv, location_samples, num_particles):
    print("\n=== Test 3b: _invert_single_event through the refactored data prep ===")
    t0 = time.time()
    res = inv._invert_single_event(
        data,
        location_samples=location_samples,
        progress_callback=lambda _m: None,
    )
    wall = time.time() - t0

    w = np.asarray(res.weights, dtype=np.float64)
    if w.size == 0 or not np.isfinite(w).all() or w.sum() <= 0:
        w = None
    gamma = np.asarray(res.gamma, dtype=np.float64)
    delta = np.asarray(res.delta, dtype=np.float64)
    g_mean = float(np.average(gamma, weights=w))
    d_mean = float(np.average(delta, weights=w))
    sar = res.sigma_amp_ratio
    sar = None if sar is None else np.asarray(sar, dtype=np.float64)

    check(
        "3b. inversion completed",
        gamma.size == num_particles and np.isfinite(gamma).all(),
        f"n_samples={gamma.size}, wall={wall:.1f}s",
    )
    check(
        f"3b. gamma mean {g_mean:+.4f} within {GAMMA_TOL} of production {GAMMA_REF}",
        abs(g_mean - GAMMA_REF) <= GAMMA_TOL,
        f"|diff|={abs(g_mean - GAMMA_REF):.4f}",
    )
    check(
        f"3b. delta mean {d_mean:+.4f} within {DELTA_TOL} of production {DELTA_REF}",
        abs(d_mean - DELTA_REF) <= DELTA_TOL,
        f"|diff|={abs(d_mean - DELTA_REF):.4f}",
    )
    if sar is not None:
        check(
            "3b. sigma_amp_ratio finite and non-degenerate "
            "(noise block not frozen by softplus_inv)",
            bool(np.all(np.isfinite(sar))) and float(np.std(sar)) > 0.0,
            f"mean={float(np.mean(sar)):.4f} std={float(np.std(sar)):.4f}",
        )
    print(
        f"  summary: gamma={g_mean:+.4f} delta={d_mean:+.4f} "
        f"kappa_circmean={float(np.angle(np.mean(np.exp(1j * np.asarray(res.kappa))))) % (2 * np.pi):.3f} "
        f"h={float(np.mean(res.h)):.3f} sigma={float(np.mean(res.sigma)):.3f} "
        f"wall={wall:.1f}s"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the eq00124 inversion")
    ap.add_argument("--particles", type=int, default=500)
    args = ap.parse_args()

    test_softplus_inv()
    test_blockwise_dtype()

    print("\n=== loading eq00124 (step-4 filter + read_data) ===")
    data = load_eq00124()
    inv = make_inversion(data, args.particles)
    _ok, location_samples = test_prepare_event_arrays(data, inv)

    if not args.quick:
        test_single_event_inversion(data, inv, location_samples, args.particles)
    else:
        print("\n(skipping the eq00124 inversion: --quick)")

    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    print("\n" + "=" * 70)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        for n, ok, d in RESULTS:
            if not ok:
                print(f"  FAILED: {n}  --  {d}")
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
