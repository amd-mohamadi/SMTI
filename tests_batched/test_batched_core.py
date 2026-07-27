#!/usr/bin/env python
"""
M1 stage-2 tests for ``src/inversion_blackjax_batched.py`` (CPU only).

  0. ``src.tape_jax_dtype.jax_Tape_MT6`` is bit-identical to ``src.tape_jax``'s
     in float64 and stays float32 for float32 inputs.
  1. pad/mask equivalence: the masked, padded batched log-likelihood equals the
     unpadded production log-likelihood (m0_spike/harness.py, a verbatim copy of
     ``_invert_single_event``'s math) on 20 random positions.
  2. bucketing: events with different station counts land in different buckets;
     entry count = events x chains; padded shapes are multiples of 8.
  3. mini end-to-end: 2 events x 2 chains (B=4), P=200, num_mcmc_steps=2.
  4. fp32 smoke, run in a SEPARATE subprocess with x64 disabled.

Run with:
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        conda run -n smti python tests_batched/test_batched_core.py
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
import subprocess
import sys
import time

import numpy as np

SMTI_DEV = os.environ.get("SMTI_ROOT", "/s0/data/CAPE/smti_workflow/SMTI_dev")
WORKFLOW = "/s0/data/CAPE/smti_workflow/smti_inversion"
M0_SPIKE = os.path.join(WORKFLOW, "m0_spike")
for _p in (SMTI_DEV, M0_SPIKE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FP32 = "--fp32" in sys.argv

import jax  # noqa: E402

# NOTE: this import pulls in src.tape_jax, which force-enables x64 process-wide.
# For the fp32 subprocess we turn it back off immediately afterwards -- before
# any array exists -- which is the documented recipe in the batched module.
from src.inversion_blackjax import InversionBlackJAX  # noqa: E402
import src.inversion_blackjax_batched as bx  # noqa: E402

if FP32:
    jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402


RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""),
        flush=True,
    )
    return bool(ok)


# --------------------------------------------------------------------------
# eq00124 loading (identical to tests_batched/test_foundations.py)
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
SEED = 20240216


def _filter_step4(dat_path):
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
    return mod._filter_step4_data_dat(dat_path, include_stacked_stations=True)


def load_eq00124():
    from src.data_loader import read_data

    filtered_path, _summary = _filter_step4(DAT_PATH)
    try:
        return read_data(
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


def make_inversion(data, num_particles, num_chains, num_mcmc_steps=5, cls=None):
    cls = cls or InversionBlackJAX
    return cls(
        data,
        inversion_options=INVERSION_OPTIONS,
        num_particles=num_particles,
        dc=False,
        gamma_beta_prior=(3.0, 3.0),
        delta_beta_prior=(3.0, 3.0),
        amp_ratio_sigma_prior=2.0,
        amp_ratio_noise_mode="global",
        random_seed=SEED,
        location_samples_n=LOCATION_SAMPLES_N,
        azimuth_error=AZIMUTH_ERROR,
        takeoff_error=TAKEOFF_ERROR,
        mcmc_kernel="rmh",
        num_mcmc_steps=num_mcmc_steps,
        mechanism_steps=5,
        smc_target_ess_ratio=0.9,
        max_smc_iterations=60,
        num_chains=num_chains,
        chain_execution="sequential",
    )


# --------------------------------------------------------------------------
# Test 0 -- dtype-agnostic tape
# --------------------------------------------------------------------------
def test_tape():
    print("\n=== Test 0: src.tape_jax_dtype vs src.tape_jax ===")
    from src.tape_jax import jax_Tape_MT6 as tape_ref
    from src.tape_jax_dtype import jax_Tape_MT6 as tape_new

    rng = np.random.default_rng(3)
    n = 500
    g = rng.uniform(-np.pi / 6, np.pi / 6, n)
    d = rng.uniform(-np.pi / 2, np.pi / 2, n)
    k = rng.uniform(0, 2 * np.pi, n)
    h = rng.uniform(0, 1, n)
    s = rng.uniform(-np.pi / 2, np.pi / 2, n)

    tape_new_jit = jax.jit(tape_new)

    def eval_all(fn):
        return np.stack(
            [
                np.asarray(fn(*(jnp.asarray(v[i]) for v in (g, d, k, h, s))))
                for i in range(n)
            ]
        )

    ref = eval_all(tape_ref)
    new = eval_all(tape_new)
    new_jit = eval_all(tape_new_jit)
    # ``src.tape_jax``'s entry points are @jax.jit, so the like-for-like
    # comparison is against the jitted copy (which is how the sampler uses it).
    check(
        "0a. float64 bit-identical to src.tape_jax.jax_Tape_MT6 (both jitted)",
        np.array_equal(ref, new_jit),
        f"n={n}, max|diff|={float(np.max(np.abs(ref - new_jit))):.3e}",
    )
    check(
        "0a'. float64 eager evaluation agrees to <= 1e-15 (XLA fusion only)",
        float(np.max(np.abs(ref - new))) <= 1e-15,
        f"max|diff|={float(np.max(np.abs(ref - new))):.3e}",
    )
    out32 = tape_new(
        jnp.float32(g[0]), jnp.float32(d[0]), jnp.float32(k[0]),
        jnp.float32(h[0]), jnp.float32(s[0]),
    )
    check(
        "0b. float32 inputs -> float32 output",
        out32.dtype == jnp.float32,
        f"dtype={out32.dtype}, max|diff| vs fp64 = "
        f"{float(np.max(np.abs(np.asarray(out32, dtype=np.float64) - ref[0]))):.3e}",
    )


# --------------------------------------------------------------------------
# Test 1 -- pad/mask equivalence
# --------------------------------------------------------------------------
def test_pad_mask_equivalence():
    print("\n=== Test 1: masked padded log-likelihood == unpadded production ===")
    from harness import load_arrays, make_model
    from src.tape_jax import jax_Tape_MT6 as tape_ref

    npz = np.load(os.path.join(M0_SPIKE, "eq00124_arrays.npz"))
    with open(os.path.join(M0_SPIKE, "eq00124_arrays.json")) as fh:
        meta = json.load(fh)
    n_pol, n_ar, n_loc = meta["N_pol"], meta["N_ar"], meta["N_loc"]

    # reference: verbatim copy of _invert_single_event's math, unpadded
    arrays = load_arrays()
    _, ref_loglik, _, _ = make_model(
        arrays,
        dc=False,
        gamma_beta_prior=(3.0, 3.0),
        delta_beta_prior=(3.0, 3.0),
        amp_ratio_sigma_prior=2.0,
        jax_Tape_MT6=tape_ref,
    )

    prep = {
        "has_pol": True,
        "has_ar": True,
        "a_pol": npz["a_pol"],
        "error_pol": npz["error_pol"],
        "incorrect_prob": float(npz["incorrect_prob"]),
        "a1_ar": npz["a1_ar"],
        "a2_ar": npz["a2_ar"],
        "amp_ratio_obs": npz["amp_ratio_obs"],
        "log_ratio_sigma": npz["log_ratio_sigma"],
    }
    pad_pol = bx._round_up(n_pol)
    pad_ar = bx._round_up(n_ar)
    check(
        "1a. padded widths are multiples of 8",
        (pad_pol, pad_ar) == (48, 64),
        f"N_pol {n_pol}->{pad_pol}, N_ar {n_ar}->{pad_ar}",
    )

    variants = {}
    for tag, (np_, na_) in {
        "unpadded(mask=1)": (n_pol, n_ar),
        "padded(48,64)": (pad_pol, pad_ar),
    }.items():
        data = {
            k: jnp.asarray(v)
            for k, v in bx._pad_event_arrays(prep, np_, na_, n_loc).items()
        }
        _, loglik = bx._make_model(
            data,
            has_pol=True,
            has_ar=True,
            gamma_beta_prior=(3.0, 3.0),
            delta_beta_prior=(3.0, 3.0),
            amp_ratio_sigma_prior=2.0,
        )
        variants[tag] = loglik

    rng = np.random.default_rng(11)
    worst = {tag: 0.0 for tag in variants}
    finite = True
    for _ in range(20):
        pos = {
            "gamma": jnp.asarray(rng.normal(0, 1.5)),
            "delta": jnp.asarray(rng.normal(0, 1.5)),
            "kappa": jnp.asarray(rng.normal(0, 2.0)),
            "h": jnp.asarray(rng.normal(0, 1.5)),
            "sigma": jnp.asarray(rng.normal(0, 1.5)),
            "sigma_amp_ratio": jnp.asarray(rng.normal(0, 1.0)),
        }
        ref = float(ref_loglik(pos))
        finite = finite and np.isfinite(ref)
        for tag, fn in variants.items():
            got = float(fn(pos))
            # NaN must FAIL this test (a masked/padded row leaking a NaN is
            # exactly what the padding scheme exists to prevent), so check
            # finiteness explicitly and use a NaN-propagating accumulator --
            # ``max(0.0, nan)`` is 0.0 in Python and would silently pass.
            finite = finite and bool(np.isfinite(got))
            rel = abs(got - ref) / max(1.0, abs(ref))
            if not (rel < worst[tag]):
                worst[tag] = rel

    for tag in variants:
        check(
            f"1b. {tag} vs unpadded production loglik, rel < 1e-12 on 20 positions",
            bool(worst[tag] < 1e-12) and finite,
            f"max_rel={worst[tag]:.3e}, all values finite={finite}",
        )

    # padded rows must produce finite garbage, not NaN
    data48 = {
        k: jnp.asarray(v)
        for k, v in bx._pad_event_arrays(prep, pad_pol, pad_ar, n_loc).items()
    }
    check(
        "1c. padded arrays finite and masks correct",
        all(bool(np.all(np.isfinite(np.asarray(v)))) for v in data48.values())
        and int(np.sum(np.asarray(data48["pol_mask"]))) == n_pol
        and int(np.sum(np.asarray(data48["ar_mask"]))) == n_ar,
        f"pol_mask sum={int(np.sum(np.asarray(data48['pol_mask'])))}, "
        f"ar_mask sum={int(np.sum(np.asarray(data48['ar_mask'])))}",
    )


# --------------------------------------------------------------------------
# Test 2 -- bucketing
# --------------------------------------------------------------------------
class _TruncInversion(InversionBlackJAX):
    """InversionBlackJAX whose per-event arrays are truncated to a queued size.

    Lets the bucketing test build 'synthetic' events with different station
    counts out of the one real event, without faking the data-prep pipeline.
    """

    def set_truncations(self, sizes):
        self._trunc = list(sizes)

    def _prepare_event_arrays(self, event, location_samples=None):
        prep = super()._prepare_event_arrays(event, location_samples=location_samples)
        n_pol, n_ar = self._trunc.pop(0)
        for key in ("a_pol", "error_pol", "pol_obs"):
            if prep[key] is not None:
                prep[key] = prep[key][:n_pol]
        for key in ("a1_ar", "a2_ar", "amp_ratio_obs", "log_ratio_sigma"):
            if prep[key] is not None:
                prep[key] = prep[key][:n_ar]
        return prep


def test_bucketing(data):
    print("\n=== Test 2: shape bucketing ===")
    num_chains = 3
    inv = make_inversion(data, num_particles=50, num_chains=num_chains, cls=_TruncInversion)
    sizes = [(44, 62), (10, 62), (44, 62), (10, 62)]
    inv.set_truncations(sizes)
    events = [(f"ev{i}", data) for i in range(len(sizes))]
    buckets = bx.prepare_batch(inv, events)

    by_key = {b.key: b for b in buckets}
    check(
        "2a. two buckets: (48,64) and (16,64)",
        len(buckets) == 2
        and (48, 64, 20, True, True) in by_key
        and (16, 64, 20, True, True) in by_key,
        f"keys={[b.key for b in buckets]}",
    )
    total = sum(b.size for b in buckets)
    check(
        "2b. entry count = events x chains",
        total == len(sizes) * num_chains
        and all(b.size == 2 * num_chains for b in buckets),
        f"total={total} (expected {len(sizes) * num_chains}), "
        f"per bucket {[b.size for b in buckets]}",
    )
    b48 = by_key[(48, 64, 20, True, True)]
    shapes_ok = (
        b48.data["a_pol"].shape == (2 * num_chains, 48, 20, 6)
        and b48.data["a1_ar"].shape == (2 * num_chains, 64, 20, 6)
        and b48.data["pol_mask"].shape == (2 * num_chains, 48)
    )
    check(
        "2c. stacked array shapes (B, N_pad, N_loc, 6)",
        shapes_ok,
        f"a_pol={b48.data['a_pol'].shape}, a1_ar={b48.data['a1_ar'].shape}",
    )
    masks_ok = bool(
        np.all(b48.data["pol_mask"][:, :44] == 1.0)
        and np.all(b48.data["pol_mask"][:, 44:] == 0.0)
        and np.all(by_key[(16, 64, 20, True, True)].data["pol_mask"][:, 10:] == 0.0)
    )
    check("2d. masks mark exactly the real rows", masks_ok)

    seeds = sorted({e.seed for e in buckets[0].entries})
    ss = np.random.SeedSequence(SEED)
    expect = sorted(int(s) for s in ss.generate_state(num_chains))
    check(
        "2e. chain seeds from SeedSequence(random_seed)",
        seeds == expect,
        f"{seeds}",
    )
    chains_per_event = {}
    for b in buckets:
        for e in b.entries:
            chains_per_event.setdefault(e.event_id, set()).add(e.chain_index)
    check(
        "2f. every event has all chain indices",
        all(v == set(range(num_chains)) for v in chains_per_event.values())
        and len(chains_per_event) == len(sizes),
        f"{ {k: sorted(v) for k, v in chains_per_event.items()} }",
    )

    # scope guards
    guards = []
    for attr, value, tag in (
        ("smc_method", "adaptive_persistent", "persistent"),
        ("mcmc_kernel", "nuts", "NUTS"),
        ("amp_ratio_noise_mode", "station_smooth", "station_smooth"),
        ("dc", True, "dc"),
        ("adapt_proposal", False, "adapt_proposal=False"),
    ):
        probe = make_inversion(data, 50, 2)
        setattr(probe, attr, value)
        try:
            bx.prepare_batch(probe, [("ev", data)])
            guards.append(f"{tag}: NO RAISE")
        except NotImplementedError:
            pass
        except Exception as exc:  # pragma: no cover
            guards.append(f"{tag}: {type(exc).__name__}")
    check(
        "2g. NotImplementedError for persistent/NUTS/station_smooth/dc/no-adapt",
        not guards,
        "; ".join(guards) if guards else "all five raise NotImplementedError",
    )

    # duplicate event ids would merge two events' chains into one result
    dup_inv = make_inversion(data, 50, 2)
    try:
        bx.prepare_batch(dup_inv, [("ev", data), ("ev", data)])
        dup_msg, dup_ok = "NO RAISE", False
    except ValueError as exc:
        dup_msg, dup_ok = str(exc).splitlines()[0][:60], True
    except Exception as exc:  # pragma: no cover
        dup_msg, dup_ok = f"{type(exc).__name__}", False
    check("2h. duplicate event_id raises ValueError", dup_ok, dup_msg)

    # num_chains == 1 must use forward()'s seed (PRNGKey(random_seed)), not the
    # SeedSequence derivation used for multi-chain runs.
    inv1 = make_inversion(data, 50, num_chains=1)
    b1 = bx.prepare_batch(inv1, [("ev", data)])
    check(
        "2i. num_chains=1 seed == random_seed (matches _invert_single_event)",
        [e.seed for e in b1[0].entries] == [SEED],
        f"seeds={[e.seed for e in b1[0].entries]}, random_seed={SEED}",
    )

    # max_batch chunking: fixed B per program, last chunk padded and discarded
    inv_c = make_inversion(data, 50, num_chains=2, cls=_TruncInversion)
    inv_c.set_truncations([(44, 62)] * 3)
    big = bx.prepare_batch(inv_c, [(f"c{i}", data) for i in range(3)])[0]
    chunks = bx._chunk_bucket(big, 4)
    ok_chunk = (
        big.size == 6
        and len(chunks) == 2
        and all(c.size == 4 for c in chunks)
        and [c.active for c in chunks] == [4, 2]
        and all(
            np.array_equal(c.data["a_pol"], big.data["a_pol"][idx])
            for c, idx in zip(chunks, ([0, 1, 2, 3], [4, 5, 4, 4]))
        )
    )
    check(
        "2j. max_batch chunking: fixed B, last chunk pad-repeated and marked",
        ok_chunk,
        f"B={big.size} -> chunks {[c.size for c in chunks]} active {[c.active for c in chunks]}",
    )
    check(
        "2k. max_batch=None leaves the bucket intact",
        bx._chunk_bucket(big, None)[0] is big and bx._chunk_bucket(big, 6)[0] is big,
        f"size={big.size}",
    )


# --------------------------------------------------------------------------
# Test 3 / 4 -- mini end-to-end
# --------------------------------------------------------------------------
def run_mini(data, smc_dtype, num_particles=200, num_chains=2, quiet=True):
    # Two GENUINELY different events (different station subsets) that still land
    # in the same (48, 64) bucket, so the test covers "different data in one
    # vmapped program" and not just "the same event twice".
    inv = make_inversion(
        data,
        num_particles=num_particles,
        num_chains=num_chains,
        num_mcmc_steps=2,
        cls=_TruncInversion,
    )
    inv.set_truncations([(44, 62), (42, 58)])
    events = [("evA", data), ("evB", data)]

    captured = []
    orig = bx._run_bucket

    def spy(*a, **k):
        out = orig(*a, **k)
        captured.append(out)
        return out

    bx._run_bucket = spy
    lines = []
    try:
        t0 = time.time()
        results = bx.run_batched(
            inv,
            events,
            smc_dtype=smc_dtype,
            progress_callback=(lines.append if quiet else print),
        )
        wall = time.time() - t0
    finally:
        bx._run_bucket = orig
    return results, captured, wall, lines, num_particles, num_chains


def check_mini(results, captured, wall, num_particles, num_chains, tag):
    n_bucket = len(captured)
    out = captured[0]
    lam = out["lambda"]
    check(
        f"3a{tag}. single bucket, B = events x chains = 4",
        n_bucket == 1 and lam.size == 4,
        f"buckets={n_bucket}, B={lam.size}, stages={out['stages']}, "
        f"wall={wall:.1f}s (SMC {out['wall']:.1f}s, init+compile {out['t_init']:.1f}s)",
    )
    check(
        f"3b{tag}. all entries reached lambda = 1",
        bool(np.all(lam >= 1.0 - 1e-6)) and not bool(np.any(out["stalled"])),
        f"lambda={np.array2string(lam, precision=6)}, "
        f"stalled={out['stalled'].tolist()}",
    )
    check(
        f"3c{tag}. result dict has both event ids",
        set(results) == {"evA", "evB"},
        f"{sorted(results)}",
    )
    ok_shapes = True
    detail = []
    for eid, res in sorted(results.items()):
        shapes = {
            "mt6": res.mt6.shape,
            "gamma": res.gamma.shape,
            "weights": res.weights.shape,
            "sigma_amp_ratio": None if res.sigma_amp_ratio is None else res.sigma_amp_ratio.shape,
        }
        ok = (
            shapes["mt6"] == (num_chains, 6, num_particles)
            and shapes["gamma"] == (num_chains, num_particles)
            and res.delta.shape == (num_chains, num_particles)
            and res.kappa.shape == (num_chains, num_particles)
            and res.h.shape == (num_chains, num_particles)
            and res.sigma.shape == (num_chains, num_particles)
            and shapes["weights"] == (num_chains, num_particles)
            and shapes["sigma_amp_ratio"] == (num_chains, num_particles)
            and res.num_chains == num_chains
        )
        ok_shapes = ok_shapes and ok
        detail.append(f"{eid}: mt6{shapes['mt6']} num_chains={res.num_chains}")
    check(
        f"3d{tag}. stacked structure matches _invert_multi_chain",
        ok_shapes,
        "; ".join(detail),
    )
    finite = all(
        np.all(np.isfinite(getattr(r, f)))
        for r in results.values()
        for f in ("mt6", "gamma", "delta", "kappa", "h", "sigma", "weights")
    )
    wsum = [float(np.sum(r.weights[c])) for r in results.values() for c in range(num_chains)]
    check(
        f"3e{tag}. all outputs finite, weights normalised per chain",
        finite and all(abs(w - 1.0) < 1e-4 for w in wsum),
        f"weight sums={['%.6f' % w for w in wsum]}",
    )
    summ = {}
    for eid, res in sorted(results.items()):
        g = float(np.average(res.gamma.ravel(), weights=res.weights.ravel()))
        d = float(np.average(res.delta.ravel(), weights=res.weights.ravel()))
        sar = float(np.mean(res.sigma_amp_ratio))
        summ[eid] = (g, d)
        print(
            f"    {eid}: gamma={g:+.4f} delta={d:+.4f} "
            f"h={float(np.mean(res.h)):.3f} sigma_amp_ratio={sar:.4f}"
        )
    # evA and evB are different station subsets of the same earthquake: their
    # particles must not be identical (that would mean one entry's data leaked
    # into the other's lane), but their posteriors should stay in the same
    # region of parameter space.
    ga, gb = results["evA"].gamma, results["evB"].gamma
    check(
        f"3f{tag}. the two events in the bucket get genuinely distinct results",
        not np.array_equal(np.asarray(ga), np.asarray(gb))
        and abs(summ["evA"][0] - summ["evB"][0]) > 1e-6,
        f"dgamma={summ['evA'][0] - summ['evB'][0]:+.4f}, "
        f"ddelta={summ['evA'][1] - summ['evB'][1]:+.4f}",
    )


def test_mini_e2e(data):
    print("\n=== Test 3: mini end-to-end, 2 events x 2 chains, P=200 (float64) ===")
    results, captured, wall, _lines, P, C = run_mini(data, "float64")
    check(
        "3.dtype. particles stay float64",
        captured[0]["particle_dtype"] == np.float64,
        f"{captured[0]['particle_dtype']}",
    )
    check_mini(results, captured, wall, P, C, "")


def test_fp32(data):
    print("\n=== Test 4: fp32 smoke (x64 disabled in this process) ===")
    check(
        "4a. jax_enable_x64 is False",
        not jax.config.jax_enable_x64,
        f"x64={jax.config.jax_enable_x64}",
    )
    results, captured, wall, _lines, P, C = run_mini(data, "float32")
    check(
        "4b. SMC particles are float32",
        captured[0]["particle_dtype"] == np.float32,
        f"{captured[0]['particle_dtype']}",
    )
    check_mini(results, captured, wall, P, C, " (fp32)")


# --------------------------------------------------------------------------
# Test 5 (opt-in) -- batched vs unbatched posterior on the same seed
# --------------------------------------------------------------------------
def test_vs_unbatched(data, num_particles=500):
    """Same event, same chain seed, same station-angle realisation.

    The two paths are not bit-identical (``jnp.sum(mask * x)`` differs from
    ``jnp.sum(x)`` at the 3e-16 level, which MCMC accept/reject decisions
    amplify chaotically), so this is a statistical comparison at the level of
    single-chain Monte-Carlo noise.
    """
    from src.data_prep import build_location_samples_from_errors

    print(f"\n=== Test 5: batched (B=1) vs unbatched, P={num_particles} ===")
    chain_seed = int(np.random.SeedSequence(SEED).generate_state(1)[0])

    inv_b = make_inversion(data, num_particles, num_chains=1)
    res_b = bx.run_batched(
        inv_b, [("evA", data)], smc_dtype="float64", progress_callback=lambda _m: None
    )["evA"]

    # prepare_batch drew the location samples from a fresh default_rng(SEED)
    location_samples = build_location_samples_from_errors(
        data,
        rng=np.random.default_rng(SEED),
        n_samples=LOCATION_SAMPLES_N,
        azimuth_error=AZIMUTH_ERROR,
        takeoff_error=TAKEOFF_ERROR,
    )
    inv_u = make_inversion(data, num_particles, num_chains=1)
    inv_u.random_seed = chain_seed
    res_u = inv_u._invert_single_event(
        data, location_samples=location_samples, progress_callback=lambda _m: None
    )

    def summ(r):
        w = np.asarray(r.weights, dtype=float).ravel()
        return {
            k: float(np.average(np.asarray(getattr(r, k), dtype=float).ravel(), weights=w))
            for k in ("gamma", "delta", "h", "sigma", "sigma_amp_ratio")
        }

    sb, su = summ(res_b), summ(res_u)
    tols = {"gamma": 0.06, "delta": 0.08, "h": 0.10, "sigma": 0.20,
            "sigma_amp_ratio": 0.10}
    for k in sb:
        check(
            f"5. {k}: batched {sb[k]:+.4f} vs unbatched {su[k]:+.4f} "
            f"(tol {tols[k]})",
            abs(sb[k] - su[k]) <= tols[k],
            f"|diff|={abs(sb[k] - su[k]):.4f}",
        )
    # Because the key streams are constructed identically, the two runs in
    # practice take the *same* accept/reject decisions and agree particle for
    # particle to round-off.  A chaotic divergence would show up as O(0.1)
    # differences, so the 1e-6 threshold separates the two regimes cleanly.
    worst = []
    for f in ("gamma", "delta", "kappa", "h", "sigma", "sigma_amp_ratio",
              "weights", "mt6"):
        a = np.asarray(getattr(res_b, f), dtype=float)
        b = np.asarray(getattr(res_u, f), dtype=float)
        worst.append((f, float(np.max(np.abs(a - b)))))
    check(
        "5. particle-for-particle agreement with the unbatched path (< 1e-6)",
        all(d < 1e-6 for _f, d in worst),
        ", ".join(f"{f} {d:.3e}" for f, d in worst),
    )


# --------------------------------------------------------------------------
# Test 6 -- stall guard freezes an entry (review finding: major 1)
# --------------------------------------------------------------------------
def test_stall_freeze(data, num_particles=200):
    """A stalled entry must stop evolving, exactly like the unbatched ``break``.

    With a deliberately aggressive ``min_tempering_increment`` some chains trip
    the stall guard early while others run on.  Before the fix the guard only
    exempted an entry from the loop-exit test, so a stalled entry kept being
    tempered to lambda=1 by the rest of the batch and was still labelled
    ``tempering_stalled``.  Now it is frozen at the state the unbatched loop
    would have returned, and the flag is derived from the final lambda.
    """
    from src.data_prep import build_location_samples_from_errors

    print("\n=== Test 6: stall guard freezes the entry (batched == unbatched) ===")
    # Tuned so the batch is MIXED: chains 0 and 1 trip the guard at stage 4
    # (lambda ~= 0.013) while chains 2 and 3 temper all the way to 1 over 33
    # stages -- i.e. the frozen entries sit out 29 stages of the batch.
    num_chains = 4
    stall_kw = dict(min_tempering_increment=0.004, tempering_stall_patience=4)

    inv_b = make_inversion(data, num_particles, num_chains, num_mcmc_steps=2)
    for k, v in stall_kw.items():
        setattr(inv_b, k, v)

    captured = []
    orig = bx._run_bucket

    def spy(*a, **k):
        out = orig(*a, **k)
        captured.append(out)
        return out

    bx._run_bucket = spy
    try:
        res_b = bx.run_batched(
            inv_b, [("ev", data)], smc_dtype="float64",
            progress_callback=lambda _m: None,
        )["ev"]
    finally:
        bx._run_bucket = orig

    out = captured[0]
    lam = np.asarray(out["lambda"])
    stalled = np.asarray(out["stalled"])
    guard = np.asarray(out["stall_guard_fired"])
    check(
        "6a. setup produced a mixed batch (some entries stalled, some finished)",
        bool(stalled.any()) and bool((~stalled).any()),
        f"lambda={np.array2string(lam, precision=4)}, stalled={stalled.tolist()}",
    )
    check(
        "6b. tempering_stalled <=> final lambda < 1 (flag describes the samples)",
        bool(np.array_equal(stalled, lam < 1.0 - 1e-6))
        and bool(np.array_equal(stalled, guard)),
        f"stalled={stalled.tolist()}, lambda<1={(lam < 1 - 1e-6).tolist()}, "
        f"guard_fired={guard.tolist()}",
    )
    check(
        "6c. event-level tempering_stalled propagates through the stacking",
        bool(res_b.tempering_stalled) == bool(stalled.any()),
        f"result.tempering_stalled={res_b.tempering_stalled}",
    )

    # unbatched twins: same seeds, same station-angle realisation, same guard
    chain_seeds = [
        int(s) for s in np.random.SeedSequence(SEED).generate_state(num_chains)
    ]
    location_samples = build_location_samples_from_errors(
        data,
        rng=np.random.default_rng(SEED),
        n_samples=LOCATION_SAMPLES_N,
        azimuth_error=AZIMUTH_ERROR,
        takeoff_error=TAKEOFF_ERROR,
    )
    worst = []
    for c, seed in enumerate(chain_seeds):
        inv_u = make_inversion(data, num_particles, num_chains=1, num_mcmc_steps=2)
        for k, v in stall_kw.items():
            setattr(inv_u, k, v)
        inv_u.random_seed = seed
        res_u = inv_u._invert_single_event(
            data, location_samples=location_samples, progress_callback=lambda _m: None
        )
        d = max(
            float(np.max(np.abs(np.asarray(getattr(res_b, f))[c] - np.asarray(getattr(res_u, f)))))
            for f in ("gamma", "delta", "kappa", "h", "sigma", "weights")
        )
        worst.append((c, d, bool(res_u.tempering_stalled)))
    check(
        "6d. every chain (stalled or not) matches its unbatched twin (< 1e-6)",
        all(d < 1e-6 for _c, d, _s in worst),
        ", ".join(f"chain{c}{' (stalled)' if s else ''} {d:.2e}" for c, d, s in worst),
    )
    check(
        "6e. unbatched _invert_single_event now reports tempering_stalled too",
        [s for _c, _d, s in worst] == stalled.tolist(),
        f"unbatched={[s for _c, _d, s in worst]}, batched={stalled.tolist()}",
    )


# --------------------------------------------------------------------------
# Test 7 -- failure isolation and chunked batches
# --------------------------------------------------------------------------
def test_failure_isolation(data, num_particles=100):
    """A failing bucket must not destroy the results of the buckets that worked."""
    print("\n=== Test 7: failure isolation + max_batch chunking end-to-end ===")
    inv = make_inversion(data, num_particles, num_chains=2, num_mcmc_steps=2,
                         cls=_TruncInversion)
    inv.set_truncations([(44, 62), (10, 62)])
    orig = bx._run_bucket
    calls = {"n": 0}

    def flaky(inv_, bucket, dtype, progress, label):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated XLA/OOM failure")
        return orig(inv_, bucket, dtype, progress, label)

    bx._run_bucket = flaky
    try:
        results, failures = bx.run_batched(
            inv, [("good", data), ("bad", data)], smc_dtype="float64",
            progress_callback=lambda _m: None, return_failures=True,
        )
    finally:
        bx._run_bucket = orig
    check(
        "7a. surviving bucket returned, failed bucket reported not raised",
        set(results) == {"good"} and set(failures) == {"bad"},
        f"results={sorted(results)}, failures={sorted(failures)}",
    )

    # same events, chunked to one entry per program
    inv2 = make_inversion(data, num_particles, num_chains=2, num_mcmc_steps=2)
    ref = bx.run_batched(inv2, [("e0", data)], smc_dtype="float64",
                         progress_callback=lambda _m: None)["e0"]
    inv3 = make_inversion(data, num_particles, num_chains=2, num_mcmc_steps=2)
    chunked = bx.run_batched(inv3, [("e0", data)], smc_dtype="float64",
                             max_batch=1, progress_callback=lambda _m: None)["e0"]
    d = max(
        float(np.max(np.abs(np.asarray(getattr(ref, f)) - np.asarray(getattr(chunked, f)))))
        for f in ("gamma", "delta", "kappa", "h", "sigma", "weights", "mt6")
    )
    check(
        "7b. max_batch chunking does not change results (B=2 vs 2x B=1, < 1e-6)",
        d < 1e-6,
        f"max|diff|={d:.3e}",
    )

    # float64 requested in an x64-disabled process must raise, not demote
    if jax.config.jax_enable_x64:
        check(
            "7c. smc_dtype='float64' guard is inactive when x64 is on",
            True,
            "x64 enabled",
        )
    else:
        try:
            bx.run_batched(inv2, [("e0", data)], smc_dtype="float64",
                           progress_callback=lambda _m: None)
            ok, msg = False, "NO RAISE"
        except RuntimeError as exc:
            ok, msg = True, str(exc).splitlines()[0][:60]
        check("7c. smc_dtype='float64' without x64 raises", ok, msg)

    # a raising progress callback must not abort the run
    inv4 = make_inversion(data, num_particles, num_chains=2, num_mcmc_steps=2)

    def boom(_msg):
        raise IOError("log file closed")

    try:
        r = bx.run_batched(inv4, [("e0", data)], smc_dtype="float64",
                           progress_callback=boom)
        cb_ok, cb_msg = set(r) == {"e0"}, "completed with a raising callback"
    except Exception as exc:  # noqa: BLE001
        cb_ok, cb_msg = False, f"{type(exc).__name__}: {exc}"
    check("7d. exception from progress_callback does not abort the bucket",
          cb_ok, cb_msg)


# --------------------------------------------------------------------------
# Test 8 -- program cache (no recompile on a repeated call)
# --------------------------------------------------------------------------
def test_program_cache(data, num_particles=60):
    print("\n=== Test 8: jitted programs are reused across run_batched calls ===")
    import logging

    bx.clear_program_cache()
    jax.clear_caches()
    logs: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            logs.append(record.getMessage())

    logger = logging.getLogger("jax")
    handler = _H()
    logger.addHandler(handler)
    prev = jax.config.jax_log_compiles
    jax.config.update("jax_log_compiles", True)
    try:
        for _ in range(2):
            inv = make_inversion(data, num_particles, num_chains=1, num_mcmc_steps=2)
            bx.run_batched(inv, [("e0", data)], smc_dtype="float64",
                           progress_callback=lambda _m: None)
            logs.append("__CALL_BOUNDARY__")
    finally:
        jax.config.update("jax_log_compiles", prev)
        logger.removeHandler(handler)

    boundary = logs.index("__CALL_BOUNDARY__")
    key = "Finished XLA compilation"
    first = [m for m in logs[:boundary] if key in m]
    second = [m for m in logs[boundary + 1:] if key in m]
    check(
        "8a. second identical run_batched call compiles nothing new",
        len(first) > 0 and len(second) == 0,
        f"XLA compilations: first call {len(first)}, second call {len(second)}"
        + ("" if not second else " -- " + "; ".join(sorted(set(second))[:2])),
    )
    check(
        "8b. one cache entry, one traced signature per program",
        len(bx._PROGRAM_CACHE) == 1
        and all(
            (not hasattr(p, "_cache_size")) or p._cache_size() == 1
            for p in list(bx._PROGRAM_CACHE.values())[0]
        ),
        f"{len(bx._PROGRAM_CACHE)} entries, jit cache sizes "
        + str([getattr(p, "_cache_size", lambda: None)()
               for p in list(bx._PROGRAM_CACHE.values())[0]]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--no-subprocess", action="store_true")
    ap.add_argument("--compare-unbatched", action="store_true")
    args = ap.parse_args()

    print(f"jax x64={jax.config.jax_enable_x64} devices={jax.devices()}", flush=True)

    if args.fp32:
        data = load_eq00124()
        test_fp32(data)
    else:
        test_tape()
        test_pad_mask_equivalence()
        print("\n=== loading eq00124 (step-4 filter + read_data) ===")
        data = load_eq00124()
        test_bucketing(data)
        test_mini_e2e(data)
        test_stall_freeze(data)
        test_failure_isolation(data)
        test_program_cache(data)
        if args.compare_unbatched:
            test_vs_unbatched(data)

    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    print("\n" + "=" * 70)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed"
          + (" (fp32 subprocess)" if args.fp32 else ""))
    for n, ok, d in RESULTS:
        if not ok:
            print(f"  FAILED: {n}  --  {d}")
    print("=" * 70, flush=True)

    if not args.fp32 and not args.no_subprocess:
        print("\n=== spawning fp32 subprocess ===", flush=True)
        env = dict(os.environ)
        rc = subprocess.call(
            [sys.executable, os.path.abspath(__file__), "--fp32"], env=env
        )
        print(f"fp32 subprocess exit code: {rc}", flush=True)
        if rc != 0:
            n_fail += 1

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
