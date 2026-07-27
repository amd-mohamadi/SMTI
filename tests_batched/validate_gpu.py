#!/usr/bin/env python
"""
M1 stage-3 GPU validation of ``src/inversion_blackjax_batched.py`` (A40).

    1. BATCHED == UNBATCHED  (primary M1 correctness bar)
       eq00124, 4 chains, P=2000, num_mcmc_steps=5, target_ess=0.9, fp64.
       (a) unbatched ``_invert_multi_chain(chain_execution='sequential')``
       (b) ``run_batched`` on the same event x 4 chains
       compared per chain and pooled.
    2. FP32 vs FP64 batched: the same batch in a separate fp32 process.
    3. MULTI-EVENT PADDING: 4 real events with different station counts, all
       chains batched in fp32, each event also inverted unbatched in fp64
       (P=1000).  Plus two padding-leak tests:
         (m1) padded rows filled with wild garbage -> results must not move
         (m2) the whole batch re-padded to a coarser multiple -> results must
              not move
    4. Perf snapshot: one homogeneous 4-event bucket (B=16, P=2000, fp32)
       against M0 spike 4 (107 ms/stage).

Usage (GPU; run the guard first)::

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
        conda run -n smti python tests_batched/validate_gpu.py --stage all

``--stage fp64`` / ``--stage fp32`` run one process each (fp32 needs
``jax_enable_x64=False`` before any array exists); ``--stage compare`` prints
the tables from the JSON both stages leave in ``--out-dir``.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
import time

import numpy as np

SMTI_DEV = os.environ.get("SMTI_ROOT", "/s0/data/CAPE/smti_workflow/SMTI_dev")
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (SMTI_DEV, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FP32 = "--stage" in sys.argv and "fp32" in sys.argv

import jax  # noqa: E402

# pulls in src.tape_jax, which force-enables x64 process-wide
from src.inversion_blackjax import InversionBlackJAX  # noqa: E402
import src.inversion_blackjax_batched as bx  # noqa: E402

if FP32:
    jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402

from event_loading import SEED, count_obs, load_event, make_inversion  # noqa: E402


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
REF_EVENT = "eq00124"
#: heterogeneous set for test 3 (station counts from tests_batched/survey_events.py)
MULTI_EVENTS = ["eq00126", "eq00124", "eq00167", "eq00047"]
#: homogeneous set for the perf snapshot: all four land in the (48, 64) bucket
PERF_EVENTS = ["eq00124", "eq00167", "eq00141", "eq00188"]

PROD_REF = {"gamma": (0.247, 0.06), "delta": (-0.145, 0.08)}
DOMINANT_BAND = (0.68, 0.92)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""),
        flush=True,
    )
    return bool(ok)


# ---------------------------------------------------------------------------
# GPU guard
# ---------------------------------------------------------------------------
def gpu_guard(min_free_gib: float = 1.0) -> None:
    """Abort if a process we do not own holds more than 1 GiB on the GPU."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: nvidia-smi unavailable ({exc}); continuing")
        return
    mine = {os.getpid(), os.getppid()}
    foreign = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid_s, mem_s = [p.strip() for p in line.split(",")[:2]]
        pid, mem = int(pid_s), float(mem_s)
        if pid not in mine and mem > min_free_gib * 1024:
            foreign.append((pid, mem))
    if foreign:
        raise SystemExit(
            "GPU occupied: "
            + ", ".join(f"pid {p} holds {m:.0f} MiB" for p, m in foreign)
        )
    print(f"GPU guard OK (compute apps: {out.strip() or 'none'})", flush=True)


def gpu_mem_mib() -> dict:
    try:
        dev = jax.devices()[0]
        st = dev.memory_stats() or {}
        return {
            k: round(st[k] / 1024**2, 1)
            for k in ("bytes_in_use", "peak_bytes_in_use")
            if k in st
        }
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# posterior summaries (definitions copied from m0_spike/spike4_precision_bench.py)
# ---------------------------------------------------------------------------
def kmeans2(X, n_restarts=20, iters=200, seed=0):
    X = np.asarray(X, dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_restarts):
        idx = rng.choice(len(X), size=2, replace=False)
        C = X[idx].copy()
        labels = np.zeros(len(X), dtype=int)
        for _ in range(iters):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            new_labels = d.argmin(1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(2):
                if np.any(labels == k):
                    C[k] = X[labels == k].mean(0)
        inertia = ((X - C[labels]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, labels.copy())
    labels = best[1]
    w = np.array([np.mean(labels == 0), np.mean(labels == 1)])
    return labels, w


def circmean(x, w=None):
    x = np.asarray(x, dtype=np.float64)
    if w is None:
        s, c = np.mean(np.sin(x)), np.mean(np.cos(x))
    else:
        w = np.asarray(w, dtype=np.float64)
        w = w / w.sum()
        s, c = np.sum(w * np.sin(x)), np.sum(w * np.cos(x))
    return float(np.arctan2(s, c) % (2 * np.pi))


def _2d(a):
    a = np.asarray(a, dtype=np.float64)
    return a[None, :] if a.ndim == 1 else a


def summarize(res, kmeans_seed=7) -> dict:
    """Weighted + unweighted posterior summaries of an ``InversionResult``."""
    g, d = _2d(res.gamma), _2d(res.delta)
    k, h, s = _2d(res.kappa), _2d(res.h), _2d(res.sigma)
    w = _2d(res.weights)
    sar = _2d(res.sigma_amp_ratio) if res.sigma_amp_ratio is not None else None
    n_chain, n_part = g.shape

    wf = (w / w.sum(axis=1, keepdims=True)).ravel() / n_chain
    out = {
        "n_chains": int(n_chain),
        "n_particles": int(n_part),
        "gamma": float(np.sum(wf * g.ravel())),
        "delta": float(np.sum(wf * d.ravel())),
        "h": float(np.sum(wf * h.ravel())),
        "sigma": float(np.sum(wf * s.ravel())),
        "kappa_circmean": circmean(k.ravel(), wf),
        "gamma_unw": float(np.mean(g)),
        "delta_unw": float(np.mean(d)),
        "kappa_circmean_unw": circmean(k.ravel()),
        "sigma_amp_ratio": None if sar is None else float(np.sum(wf * sar.ravel())),
        "gamma_per_chain": [float(np.average(g[c], weights=w[c])) for c in range(n_chain)],
        "delta_per_chain": [float(np.average(d[c], weights=w[c])) for c in range(n_chain)],
        "kappa_per_chain": [circmean(k[c], w[c]) for c in range(n_chain)],
        "all_finite": bool(
            np.all(np.isfinite(g)) and np.all(np.isfinite(d)) and np.all(np.isfinite(k))
            and np.all(np.isfinite(h)) and np.all(np.isfinite(s))
            and np.all(np.isfinite(w)) and np.all(np.isfinite(np.asarray(res.mt6)))
            and (sar is None or np.all(np.isfinite(sar)))
        ),
        "weight_sums": [float(np.sum(w[c])) for c in range(n_chain)],
    }
    X = np.column_stack(
        [np.cos(k.ravel()), np.sin(k.ravel()), h.ravel(), s.ravel()]
    )
    labels, cw = kmeans2(X, seed=kmeans_seed)
    order = np.argsort(-cw)
    out["dominant_weight"] = float(cw[order[0]])
    out["cluster_weights"] = [float(cw[order[0]]), float(cw[order[1]])]
    out["dominant_occupancy_per_chain"] = (
        (labels.reshape(n_chain, n_part) == order[0]).mean(axis=1).tolist()
    )
    return out


ELEMENT_FIELDS = (
    "gamma", "delta", "kappa", "h", "sigma", "sigma_amp_ratio", "weights", "mt6",
)


def elementwise_diff(a, b, chain=None) -> dict:
    """max |a - b| per field, optionally restricted to one chain row."""
    out = {}
    for f in ELEMENT_FIELDS:
        x, y = getattr(a, f, None), getattr(b, f, None)
        if x is None or y is None:
            continue
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            out[f] = float("nan")
            continue
        if chain is not None and x.ndim >= 2:
            x, y = x[chain], y[chain]
        out[f] = float(np.max(np.abs(x - y)))
    return out


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------
class _Tee(io.StringIO):
    """Collect the unbatched path's prints while echoing a heartbeat."""

    def __init__(self, echo_every=25):
        super().__init__()
        self.lines: list[str] = []
        self._n = 0
        self._echo = echo_every

    def write(self, s):  # noqa: D102
        for line in s.splitlines():
            if line.strip():
                self.lines.append(line)
                self._n += 1
                if self._n % self._echo == 0:
                    print(f"    ... {line.strip()[:100]}", file=sys.__stdout__, flush=True)
        return len(s)


_STAGE_RE = re.compile(r"^\s*Stage (\d+): beta=([0-9.]+)")
_CHAIN_RE = re.compile(r"Chain (\d+)/(\d+)")


def _parse_stages(lines):
    """Per-chain stage counts and final beta from ``_invert_single_event`` prints."""
    chains, cur = [], None
    for line in lines:
        if _CHAIN_RE.search(line):
            cur = {"stages": 0, "beta": float("nan")}
            chains.append(cur)
        m = _STAGE_RE.match(line)
        if m:
            if cur is None:
                cur = {"stages": 0, "beta": float("nan")}
                chains.append(cur)
            cur["stages"] = int(m.group(1))
            cur["beta"] = float(m.group(2))
    return chains


def run_unbatched(event_id, data, num_particles, num_chains, tag="", inv=None):
    """Unbatched reference run.

    ``inv`` may be an instance shared across several events: both
    ``_invert_multi_chain`` (line 670) and ``prepare_batch`` draw the
    station-angle realisation from ``inv.rng``, so a multi-event reference must
    reuse ONE instance to consume that stream in the same order the batched path
    does -- otherwise every event but the first is compared against a different
    location-sample realisation.
    """
    inv = inv if inv is not None else make_inversion(data, num_particles, num_chains)
    filtered = inv._filter_event_by_options(data)
    tee = _Tee()
    t0 = time.time()
    with contextlib.redirect_stdout(tee):
        if num_chains > 1:
            res = inv._invert_multi_chain(filtered)
        else:
            res = inv._invert_single_event(filtered)
    wall = time.time() - t0
    stages = _parse_stages(tee.lines)
    print(
        f"  unbatched {event_id}{tag}: {wall:.1f}s, stages/chain="
        f"{[c['stages'] for c in stages]}",
        flush=True,
    )
    return res, {"wall": wall, "stages_per_chain": [c["stages"] for c in stages],
                 "beta_per_chain": [c["beta"] for c in stages]}


def _garbage_hook(orig_prepare):
    """prepare_batch wrapper that fills every masked-out row with wild values."""

    def wrapped(inv, events, **kw):
        buckets = orig_prepare(inv, events, **kw)
        for bkt in buckets:
            d = bkt.data
            if "pol_mask" in d:
                m = d["pol_mask"][..., None, None] > 0.5
                d["a_pol"] = np.where(m, d["a_pol"], -1.0e3)
                m1 = d["pol_mask"] > 0.5
                d["error_pol"] = np.where(m1, d["error_pol"], 1.0e-3)
                d["incorrect_prob"] = np.where(m1, d["incorrect_prob"], 0.49)
            if "ar_mask" in d:
                m = d["ar_mask"][..., None, None] > 0.5
                d["a1_ar"] = np.where(m, d["a1_ar"], 1.0e3)
                d["a2_ar"] = np.where(m, d["a2_ar"], -1.0e-3)
                m1 = d["ar_mask"] > 0.5
                d["amp_ratio_obs"] = np.where(m1, d["amp_ratio_obs"], 1.0e4)
                d["log_ratio_sigma"] = np.where(m1, d["log_ratio_sigma"], 1.0e-3)
        return buckets

    return wrapped


def loglik_vs_padwidth(event_id, event_data, smc_dtype, n_pos=20, widths=(8, 32)):
    """Masked log-likelihood of the same event at two padding widths.

    Isolates the *only* mechanism by which a wider pad can move a result once
    masking is correct: the XLA reduction tree over the padded axis regroups, so
    the sum of the REAL terms rounds differently.  Reported as a relative
    difference; it is round-off, not leakage (the garbage-content test proves
    the padded values themselves never enter).
    """
    dtype = jnp.float32 if smc_dtype == "float32" else jnp.float64
    fns, shapes = {}, {}
    for pm in widths:
        inv = make_inversion(event_data[event_id], 50, 1)
        bkt = bx.prepare_batch(inv, [(event_id, event_data[event_id])], pad_multiple=pm)[0]
        data = {k: jnp.asarray(v[0], dtype=dtype) for k, v in bkt.data.items()}
        _lp, ll = bx._make_model(
            data, has_pol=bkt.has_pol, has_ar=bkt.has_ar,
            gamma_beta_prior=inv.gamma_beta_prior,
            delta_beta_prior=inv.delta_beta_prior,
            amp_ratio_sigma_prior=inv.amp_ratio_sigma_prior,
        )
        fns[pm] = jax.jit(ll)
        shapes[pm] = (bkt.n_pol, bkt.n_ar)
    rng = np.random.default_rng(11)
    worst, n_ident = 0.0, 0
    for _ in range(n_pos):
        pos = {
            k: jnp.asarray(rng.normal(0, s), dtype=dtype)
            for k, s in (("gamma", 1.5), ("delta", 1.5), ("kappa", 2.0),
                         ("h", 1.5), ("sigma", 1.5), ("sigma_amp_ratio", 1.0))
        }
        vals = [float(fns[pm](pos)) for pm in widths]
        worst = max(worst, abs(vals[0] - vals[1]) / max(1.0, abs(vals[0])))
        n_ident += int(vals[0] == vals[1])
    return {"max_rel": worst, "n_bit_identical": n_ident, "n_pos": n_pos,
            "shapes": {str(k): list(v) for k, v in shapes.items()}}


_BUCKET_LABEL_RE = re.compile(r"^(\[bucket [^\]]+\])")


def _stage_timings(stamps):
    """Wall time per SMC stage, per bucket, from timestamped progress lines.

    ``_run_bucket`` emits one line after each stage (each line follows a host
    read-back, so the timestamps are true per-stage wall times).  The first
    stage of a bucket also carries the XLA compile of the vmapped step, so it is
    reported separately from the median of the remaining stages.
    """
    per_label: dict[str, list[float]] = {}
    prev_t = prev_label = None
    for t, msg in stamps:
        m = _BUCKET_LABEL_RE.match(msg)
        label = m.group(1) if m else None
        if label is not None and " stage " in msg and prev_label == label:
            per_label.setdefault(label, []).append(t - prev_t)
        prev_t, prev_label = t, label
    out = {}
    for label, d in per_label.items():
        out[label] = {
            "first_stage_s": float(d[0]) if d else float("nan"),
            "median_ms": float(np.median(d[1:]) * 1e3) if len(d) > 1 else float("nan"),
            "mean_ms_after_first": float(np.mean(d[1:]) * 1e3) if len(d) > 1 else float("nan"),
            "min_ms": float(np.min(d[1:]) * 1e3) if len(d) > 1 else float("nan"),
            "n_stages": len(d),
        }
    return out


def run_batched(
    event_ids, event_data, num_particles, num_chains, smc_dtype,
    pad_multiple=8, garbage=False, quiet=True, num_mcmc_steps=5,
):
    inv = make_inversion(
        event_data[event_ids[0]], num_particles, num_chains,
        num_mcmc_steps=num_mcmc_steps,
    )
    events = [(eid, event_data[eid]) for eid in event_ids]

    captured, orig_run = [], bx._run_bucket

    def spy(inv_, bucket, dtype, progress, label):
        out = orig_run(inv_, bucket, dtype, progress, label)
        captured.append(
            {
                "label": label,
                "B": bucket.size,
                "n_pol": bucket.n_pol,
                "n_ar": bucket.n_ar,
                "event_ids": sorted({e.event_id for e in bucket.entries}),
                "stages": int(out["stages"]),
                "wall": float(out["wall"]),
                "t_init": float(out["t_init"]),
                "lambda": np.asarray(out["lambda"]).tolist(),
                "stalled": np.asarray(out["stalled"]).tolist(),
                "particle_dtype": str(out["particle_dtype"]),
            }
        )
        return out

    orig_prepare = bx.prepare_batch
    bx._run_bucket = spy
    if garbage:
        bx.prepare_batch = _garbage_hook(orig_prepare)
    stamps: list[tuple[float, str]] = []

    def cb(msg):
        stamps.append((time.time(), msg))
        if not quiet:
            print(msg, flush=True)

    try:
        t0 = time.time()
        results = bx.run_batched(
            inv, events, smc_dtype=smc_dtype, pad_multiple=pad_multiple,
            progress_callback=cb,
        )
        wall = time.time() - t0
    finally:
        bx._run_bucket = orig_run
        bx.prepare_batch = orig_prepare

    timings = _stage_timings(stamps)
    for c in captured:
        c["timing"] = timings.get(c["label"], {})
        t = c["timing"]
        print(
            f"  {c['label']} events={c['event_ids']} stages={c['stages']} "
            f"SMC {c['wall']:.2f}s | steady-state {t.get('median_ms', float('nan')):.1f} "
            f"ms/stage (median), first stage {t.get('first_stage_s', float('nan')):.1f}s "
            f"(incl. step compile), init {c['t_init']:.1f}s dtype={c['particle_dtype']}",
            flush=True,
        )
    return results, {"wall": wall, "buckets": captured}


# ---------------------------------------------------------------------------
# stage fp64
# ---------------------------------------------------------------------------
def stage_fp64(args) -> dict:
    out: dict = {"precision": "float64", "x64": bool(jax.config.jax_enable_x64)}
    P = args.particles
    data = {eid: load_event(eid) for eid in dict.fromkeys(
        [REF_EVENT] + MULTI_EVENTS)}
    out["t3_counts"] = {}
    for eid in MULTI_EVENTS:
        n_pol, n_ar = count_obs(data[eid])
        out["t3_counts"][eid] = {"n_pol": int(n_pol), "n_ar": int(n_ar)}
        print(f"  {eid}: N_pol={n_pol} N_ar={n_ar}", flush=True)

    # ---- test 1 ----------------------------------------------------------
    print(f"\n=== Test 1: batched vs unbatched, {REF_EVENT}, 4 chains, P={P}, fp64 ===",
          flush=True)
    res_u, diag_u = run_unbatched(REF_EVENT, data[REF_EVENT], P, 4)
    res_bd, diag_b = run_batched([REF_EVENT], data, P, 4, "float64")
    res_b = res_bd[REF_EVENT]

    su, sb = summarize(res_u), summarize(res_b)
    out["t1_unbatched"] = {**su, **diag_u}
    out["t1_batched"] = {**sb, **diag_b}
    out["t1_elementwise"] = elementwise_diff(res_b, res_u)

    dg = abs(sb["gamma"] - su["gamma"])
    dd = abs(sb["delta"] - su["delta"])
    ddom = abs(sb["dominant_weight"] - su["dominant_weight"])
    dk = abs(((sb["kappa_circmean"] - su["kappa_circmean"] + np.pi) % (2 * np.pi)) - np.pi)
    stages_b = diag_b["buckets"][0]["stages"]
    check("1a. pooled gamma |batched - unbatched| < 0.02",
          dg < 0.02, f"batched {sb['gamma']:+.4f} vs unbatched {su['gamma']:+.4f}, |d|={dg:.5f}")
    check("1b. pooled delta |batched - unbatched| < 0.02",
          dd < 0.02, f"batched {sb['delta']:+.4f} vs unbatched {su['delta']:+.4f}, |d|={dd:.5f}")
    check("1c. dominant kappa-mode weight |d| < 0.05",
          ddom < 0.05,
          f"batched {sb['dominant_weight']:.4f} vs unbatched {su['dominant_weight']:.4f}, "
          f"|d|={ddom:.5f}")
    check("1d. kappa circmean agrees (< 0.05 rad)", dk < 0.05,
          f"batched {sb['kappa_circmean']:.4f} vs unbatched {su['kappa_circmean']:.4f}, "
          f"|d|={dk:.5f}")
    per_chain_ok = all(
        abs(a - b) < 0.02 for a, b in zip(sb["gamma_per_chain"], su["gamma_per_chain"])
    ) and all(
        abs(a - b) < 0.02 for a, b in zip(sb["delta_per_chain"], su["delta_per_chain"])
    )
    check("1e. per-chain gamma/delta agree (< 0.02 each)", per_chain_ok,
          "gamma b=" + np.array2string(np.array(sb["gamma_per_chain"]), precision=4)
          + " u=" + np.array2string(np.array(su["gamma_per_chain"]), precision=4))
    check("1f. stage counts agree",
          stages_b == max(diag_u["stages_per_chain"]),
          f"batched {stages_b} stages (all entries) vs unbatched per chain "
          f"{diag_u['stages_per_chain']}")
    # The batched driver steps EVERY entry until the slowest one reaches
    # lambda=1, so a chain that finishes early takes extra rejuvenation moves at
    # lambda=1 (plan section 1: "statistically harmless").  Only chains whose
    # unbatched stage count equals the batched stage count take exactly the same
    # moves as their unbatched twin and can be compared particle-for-particle.
    per_chain = [
        elementwise_diff(res_b, res_u, chain=c) for c in range(sb["n_chains"])
    ]
    out["t1_elementwise_per_chain"] = per_chain
    same = [c for c, n in enumerate(diag_u["stages_per_chain"]) if n == stages_b]
    early = [c for c in range(sb["n_chains"]) if c not in same]
    worst_same = max(
        (max(per_chain[c].values()) for c in same), default=float("nan")
    )
    check(
        "1g. particle-for-particle agreement for chains with equal stage counts "
        "(< 1e-6)",
        bool(same) and worst_same < 1e-6,
        f"chains {same}: max|diff|={worst_same:.3e}"
        + (
            f"; chains {early} took "
            f"{[stages_b - diag_u['stages_per_chain'][c] for c in early]} extra "
            "lambda=1 rejuvenation stage(s) in the batch, max|diff|="
            + ", ".join(f"{max(per_chain[c].values()):.2e}" for c in early)
            if early
            else "; all chains ran the same number of stages"
        ),
    )
    check("1h. batched pooled gamma/delta inside the production band",
          abs(sb["gamma"] - PROD_REF["gamma"][0]) <= PROD_REF["gamma"][1]
          and abs(sb["delta"] - PROD_REF["delta"][0]) <= PROD_REF["delta"][1],
          f"gamma {sb['gamma']:+.4f} (ref {PROD_REF['gamma'][0]} +-{PROD_REF['gamma'][1]}), "
          f"delta {sb['delta']:+.4f} (ref {PROD_REF['delta'][0]} +-{PROD_REF['delta'][1]})")
    check("1i. all batched outputs finite, weights normalised",
          sb["all_finite"] and all(abs(w - 1) < 1e-6 for w in sb["weight_sums"]),
          f"weight sums {['%.6f' % w for w in sb['weight_sums']]}")
    out["gpu_mem_after_t1"] = gpu_mem_mib()

    # ---- test 3, fp64 per-event references -------------------------------
    print(f"\n=== Test 3a: per-event unbatched fp64 references, P={args.ref_particles} ===",
          flush=True)
    out["t3_unbatched"] = {}
    # ONE instance for all four events: see run_unbatched's docstring -- the
    # station-angle RNG stream must advance exactly as it does inside
    # prepare_batch (and inside production's forward() loop).
    shared_inv = make_inversion(
        data[MULTI_EVENTS[0]], args.ref_particles, args.ref_chains
    )
    for eid in MULTI_EVENTS:
        r, d = run_unbatched(eid, data[eid], args.ref_particles, args.ref_chains,
                             inv=shared_inv)
        out["t3_unbatched"][eid] = {**summarize(r), **d}
        print(f"    {eid}: gamma={out['t3_unbatched'][eid]['gamma']:+.4f} "
              f"delta={out['t3_unbatched'][eid]['delta']:+.4f}", flush=True)

    # ---- padding-leak tests in fp64 --------------------------------------
    print("\n=== Test 3b: padding-leak tests (fp64, P=%d, 1 chain) ===" % args.mask_particles,
          flush=True)
    out["mask_particles"] = int(args.mask_particles)
    mask_events = ["eq00124", "eq00126"]
    base, _ = run_batched(mask_events, data, args.mask_particles, 1, "float64")
    garb, _ = run_batched(mask_events, data, args.mask_particles, 1, "float64", garbage=True)
    wide, _ = run_batched(mask_events, data, args.mask_particles, 1, "float64",
                          pad_multiple=32)
    d_g = {e: elementwise_diff(base[e], garb[e]) for e in mask_events}
    d_w = {e: elementwise_diff(base[e], wide[e]) for e in mask_events}
    out["t3_mask_fp64_garbage"] = d_g
    out["t3_mask_fp64_widepad"] = d_w
    check("3b-1. garbage in padded rows does not change fp64 results (< 1e-12)",
          max(max(v.values()) for v in d_g.values()) < 1e-12,
          "; ".join(f"{e}: " + ", ".join(f"{k} {v:.1e}" for k, v in d.items())
                    for e, d in d_g.items()))
    out["t3_loglik_padwidth"] = {
        e: loglik_vs_padwidth(e, data, "float64") for e in mask_events
    }
    check("3b-2. re-padding to a multiple of 32 does not change fp64 results (< 1e-12)",
          max(max(v.values()) for v in d_w.values()) < 1e-12,
          "; ".join(f"{e}: " + ", ".join(f"{k} {v:.1e}" for k, v in d.items())
                    for e, d in d_w.items()))
    print("  loglik vs pad width (fp64): "
          + "; ".join(f"{e}: rel {v['max_rel']:.2e}, bit-identical "
                      f"{v['n_bit_identical']}/{v['n_pos']}"
                      for e, v in out["t3_loglik_padwidth"].items()), flush=True)
    out["gpu_mem_end"] = gpu_mem_mib()
    return out


# ---------------------------------------------------------------------------
# stage fp32
# ---------------------------------------------------------------------------
def stage_fp32(args) -> dict:
    out: dict = {"precision": "float32", "x64": bool(jax.config.jax_enable_x64)}
    check("0. jax_enable_x64 is False in the fp32 process",
          not jax.config.jax_enable_x64, f"x64={jax.config.jax_enable_x64}")
    P = args.particles
    ids = list(dict.fromkeys([REF_EVENT] + MULTI_EVENTS + PERF_EVENTS))
    data = {eid: load_event(eid) for eid in ids}

    # ---- test 2 ----------------------------------------------------------
    print(f"\n=== Test 2: fp32 batched, {REF_EVENT}, 4 chains, P={P} ===", flush=True)
    res, diag = run_batched([REF_EVENT], data, P, 4, "float32")
    out["t2_batched32"] = {**summarize(res[REF_EVENT]), **diag}
    out["t2_stages"] = diag["buckets"][0]["stages"]
    check("2a. fp32 particles are float32",
          diag["buckets"][0]["particle_dtype"] == "float32",
          diag["buckets"][0]["particle_dtype"])
    check("2b. fp32 outputs finite, all entries reached lambda=1",
          out["t2_batched32"]["all_finite"]
          and all(l >= 1 - 1e-6 for l in diag["buckets"][0]["lambda"])
          and not any(diag["buckets"][0]["stalled"]),
          f"lambda={diag['buckets'][0]['lambda']}")

    # ---- test 3, fp32 multi-event batch ----------------------------------
    print(f"\n=== Test 3c: 4 heterogeneous events x 4 chains, fp32, P={P} ===", flush=True)
    res_m, diag_m = run_batched(MULTI_EVENTS, data, P, 4, "float32")
    out["t3_batched32"] = {e: summarize(r) for e, r in res_m.items()}
    out["t3_batched32_diag"] = diag_m
    check("3c-1. every event present, all outputs finite",
          set(res_m) == set(MULTI_EVENTS)
          and all(out["t3_batched32"][e]["all_finite"] for e in MULTI_EVENTS),
          f"events={sorted(res_m)}")
    lam_ok = all(all(l >= 1 - 1e-6 for l in b["lambda"]) for b in diag_m["buckets"])
    check("3c-2. all entries reached lambda=1, none stalled",
          lam_ok and not any(any(b["stalled"]) for b in diag_m["buckets"]),
          "; ".join(f"{b['label']} lambda min={min(b['lambda']):.4f}"
                    for b in diag_m["buckets"]))

    # ---- padding-leak tests in fp32 --------------------------------------
    print("\n=== Test 3d: padding-leak tests (fp32, same 4-event batch) ===", flush=True)
    res_g, _ = run_batched(MULTI_EVENTS, data, P, 4, "float32", garbage=True)
    res_w, _ = run_batched(MULTI_EVENTS, data, P, 4, "float32", pad_multiple=32)
    d_g = {e: elementwise_diff(res_m[e], res_g[e]) for e in MULTI_EVENTS}
    d_w = {e: elementwise_diff(res_m[e], res_w[e]) for e in MULTI_EVENTS}
    out["t3_mask_fp32_garbage"] = d_g
    out["t3_mask_fp32_widepad"] = d_w
    out["t3_mask_fp32_summ_widepad"] = {e: summarize(r) for e, r in res_w.items()}
    out["t3_loglik_padwidth"] = {
        e: loglik_vs_padwidth(e, data, "float32") for e in MULTI_EVENTS
    }
    check("3d-1. garbage in padded rows does not change fp32 results (< 1e-6)",
          max(max(v.values()) for v in d_g.values()) < 1e-6,
          "; ".join(f"{e}: max {max(v.values()):.2e}" for e, v in d_g.items()))
    # A wider pad changes the XLA reduction tree, so in fp32 the log-likelihood
    # moves in its last bits and MCMC accept/reject amplifies that chaotically.
    # The bar there is therefore statistical, not element-wise (the element-wise
    # numbers are reported below and the fp64 run has the strict version).
    ll_rel = max(v["max_rel"] for v in out["t3_loglik_padwidth"].values())
    check("3d-2. wider padding perturbs the fp32 log-likelihood only at fp32 "
          "round-off (rel < 1e-5)",
          ll_rel < 1e-5,
          "; ".join(f"{e}: rel {v['max_rel']:.2e}, bit-identical "
                    f"{v['n_bit_identical']}/{v['n_pos']}"
                    for e, v in out["t3_loglik_padwidth"].items()))
    sw = out["t3_mask_fp32_summ_widepad"]
    stat_ok = all(
        abs(sw[e]["gamma"] - out["t3_batched32"][e]["gamma"]) < 0.03
        and abs(sw[e]["delta"] - out["t3_batched32"][e]["delta"]) < 0.03
        for e in MULTI_EVENTS
    )
    check("3d-3. wider padding leaves the fp32 posterior unchanged within MC "
          "noise (gamma/delta < 0.03)",
          stat_ok,
          "; ".join(
              f"{e}: dgamma {sw[e]['gamma'] - out['t3_batched32'][e]['gamma']:+.4f} "
              f"ddelta {sw[e]['delta'] - out['t3_batched32'][e]['delta']:+.4f} "
              f"(elementwise max {max(d_w[e].values()):.1e})"
              for e in MULTI_EVENTS))

    # ---- test 4, perf snapshot ------------------------------------------
    print(f"\n=== Test 4: perf snapshot, {len(PERF_EVENTS)} same-shape events x 4 "
          f"chains (B=16), fp32, P={P} ===", flush=True)
    jax.clear_caches()
    res_p, diag_p = run_batched(PERF_EVENTS, data, P, 4, "float32")
    out["t4_perf"] = diag_p
    out["t4_summaries"] = {e: summarize(r) for e, r in res_p.items()}
    out["gpu_mem_end"] = gpu_mem_mib()
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        out["nvidia_smi_used_mib"] = float(smi.splitlines()[0])
    except Exception:  # noqa: BLE001
        pass
    b = diag_p["buckets"][0]
    ms = b["timing"]["median_ms"]
    out["t4_ms_per_stage"] = ms
    out["t4_ms_per_stage_incl_compile"] = b["wall"] / max(b["stages"], 1) * 1e3
    out["t4_compile_s"] = b["timing"]["first_stage_s"]
    check(f"4a. perf snapshot is a single B={4 * len(PERF_EVENTS)} bucket",
          len(diag_p["buckets"]) == 1 and b["B"] == 4 * len(PERF_EVENTS),
          f"buckets={len(diag_p['buckets'])}, B={b['B']}, n_pol={b['n_pol']}, "
          f"n_ar={b['n_ar']}")
    print(f"    {ms:.1f} ms/stage vs M0 spike4 fp32 B=16: 107.5 ms/stage "
          f"(full-run 3.68 s / 34 stages = 108.2)", flush=True)
    return out


# ---------------------------------------------------------------------------
# comparison stage
# ---------------------------------------------------------------------------
def _row(name, a, b, tol=None):
    d = a - b
    flag = "" if tol is None else ("  PASS" if abs(d) <= tol else "  FAIL")
    return f"  {name:<24s} {a:+10.4f} {b:+10.4f} {d:+10.4f}{flag}"


def stage_compare(f64: dict, f32: dict) -> None:
    print("\n" + "=" * 78)
    print("TABLE 1 -- batched vs unbatched (fp64, eq00124, 4 chains, "
          f"P={f64['t1_batched']['n_particles']})")
    print("=" * 78)
    b, u = f64["t1_batched"], f64["t1_unbatched"]
    print(f"  {'quantity':<24s} {'batched':>10s} {'unbatched':>10s} {'diff':>10s}")
    print(_row("gamma (pooled)", b["gamma"], u["gamma"], 0.02))
    print(_row("delta (pooled)", b["delta"], u["delta"], 0.02))
    print(_row("kappa circmean", b["kappa_circmean"], u["kappa_circmean"]))
    print(_row("dominant mode weight", b["dominant_weight"], u["dominant_weight"], 0.05))
    print(_row("h", b["h"], u["h"]))
    print(_row("sigma", b["sigma"], u["sigma"]))
    print(_row("sigma_amp_ratio", b["sigma_amp_ratio"], u["sigma_amp_ratio"]))
    for c in range(b["n_chains"]):
        print(_row(f"  chain {c} gamma", b["gamma_per_chain"][c],
                   u["gamma_per_chain"][c], 0.02))
        print(_row(f"  chain {c} delta", b["delta_per_chain"][c],
                   u["delta_per_chain"][c], 0.02))
    print(f"  stages: batched {b['buckets'][0]['stages']}  "
          f"unbatched per chain {u['stages_per_chain']}")
    print(f"  wall:   batched {b['wall']:.1f}s (SMC {b['buckets'][0]['wall']:.1f}s + "
          f"init/compile {b['buckets'][0]['t_init']:.1f}s)  "
          f"unbatched {u['wall']:.1f}s")
    print("  element-wise max|diff| (all chains): "
          + ", ".join(f"{k} {v:.2e}" for k, v in f64["t1_elementwise"].items()))
    for c, pc in enumerate(f64.get("t1_elementwise_per_chain", [])):
        extra = f64["t1_batched"]["buckets"][0]["stages"] - u["stages_per_chain"][c]
        print(f"    chain {c} (+{extra} lambda=1 stage(s) in the batch): "
              + ", ".join(f"{k} {v:.2e}" for k, v in pc.items()))

    if f32:
        print("\n" + "=" * 78)
        print("TABLE 2 -- fp32 vs fp64 batched (eq00124, 4 chains)")
        print("=" * 78)
        a, c = f32["t2_batched32"], f64["t1_batched"]
        print(f"  {'quantity':<24s} {'fp32':>10s} {'fp64':>10s} {'diff':>10s}")
        print(_row("gamma", a["gamma"], c["gamma"], 0.03))
        print(_row("delta", a["delta"], c["delta"], 0.03))
        print(_row("dominant mode weight", a["dominant_weight"], c["dominant_weight"], 0.06))
        print(_row("kappa circmean", a["kappa_circmean"], c["kappa_circmean"]))
        print(_row("h", a["h"], c["h"]))
        print(_row("sigma", a["sigma"], c["sigma"]))
        print(_row("sigma_amp_ratio", a["sigma_amp_ratio"], c["sigma_amp_ratio"]))
        s32, s64 = a["buckets"][0]["stages"], c["buckets"][0]["stages"]
        print(f"  stages: fp32 {s32}  fp64 {s64}  (diff {s32 - s64:+d}, tol +-2)")
        t32 = a["buckets"][0].get("timing", {})
        t64 = c["buckets"][0].get("timing", {})
        if t32.get("median_ms") and t64.get("median_ms"):
            print(f"  steady-state per stage (B=4): fp32 {t32['median_ms']:.1f} ms  "
                  f"fp64 {t64['median_ms']:.1f} ms  "
                  f"speedup {t64['median_ms'] / t32['median_ms']:.1f}x")
        print(f"  SMC wall (incl. step compile): fp32 {a['buckets'][0]['wall']:.2f}s  "
              f"fp64 {c['buckets'][0]['wall']:.2f}s")
        check("2c. fp32 vs fp64 gamma |d| < 0.03",
              abs(a["gamma"] - c["gamma"]) < 0.03,
              f"|d|={abs(a['gamma'] - c['gamma']):.4f}")
        check("2d. fp32 vs fp64 delta |d| < 0.03",
              abs(a["delta"] - c["delta"]) < 0.03,
              f"|d|={abs(a['delta'] - c['delta']):.4f}")
        check("2e. fp32 vs fp64 dominant weight |d| < 0.06",
              abs(a["dominant_weight"] - c["dominant_weight"]) < 0.06,
              f"|d|={abs(a['dominant_weight'] - c['dominant_weight']):.4f}")
        check("2f. fp32 stage count within +-2 of fp64",
              abs(s32 - s64) <= 2, f"fp32 {s32} vs fp64 {s64}")

        print("\n" + "=" * 78)
        print("TABLE 3 -- multi-event: fp32 batched vs fp64 unbatched per event")
        print("=" * 78)
        print(f"  {'event':<10s} {'N_pol':>5s} {'N_ar':>5s} {'bucket':>12s} "
              f"{'g32':>8s} {'g64':>8s} {'dg':>8s} {'d32':>8s} {'d64':>8s} {'dd':>8s}")
        okall = True
        bkt_of = {}
        for bk in f32["t3_batched32_diag"]["buckets"]:
            for e in bk["event_ids"]:
                bkt_of[e] = f"({bk['n_pol']},{bk['n_ar']})"
        for e in MULTI_EVENTS:
            x, y = f32["t3_batched32"][e], f64["t3_unbatched"][e]
            dg, dd = x["gamma"] - y["gamma"], x["delta"] - y["delta"]
            ok = abs(dg) <= 0.05 and abs(dd) <= 0.05
            okall = okall and ok
            n = f64.get("t3_counts", {}).get(e, {})
            print(f"  {e:<10s} {n.get('n_pol', -1):>5d} {n.get('n_ar', -1):>5d} "
                  f"{bkt_of.get(e, '?'):>12s} "
                  f"{x['gamma']:+8.4f} {y['gamma']:+8.4f} {dg:+8.4f} "
                  f"{x['delta']:+8.4f} {y['delta']:+8.4f} {dd:+8.4f}"
                  f"{'' if ok else '   FAIL'}")
        check("3e. per-event gamma/delta agree within combined MC tolerance (0.05)",
              okall, "see TABLE 3")

        print("\n" + "=" * 78)
        print("TABLE 4 -- perf snapshot (fp32, B=16, P=%d)" % f32["t2_batched32"]["n_particles"])
        print("=" * 78)
        for bk in f32["t4_perf"]["buckets"]:
            t = bk.get("timing", {})
            print(f"  bucket {bk['n_pol']}x{bk['n_ar']} B={bk['B']}: "
                  f"{bk['stages']} stages, SMC {bk['wall']:.2f}s total; "
                  f"steady state {t.get('median_ms', float('nan')):.1f} ms/stage "
                  f"(median), {t.get('min_ms', float('nan')):.1f} ms min; "
                  f"first stage {t.get('first_stage_s', float('nan')):.1f}s "
                  f"(= XLA compile), init {bk['t_init']:.1f}s")
        ms = f32["t4_ms_per_stage"]
        ref = 107.47
        print(f"  M0 spike4 fp32 B=16 P=2000: {ref} ms/stage (pure step), "
              f"full-run 3.68 s / 34 stages = 108.2 ms/stage")
        print(f"  productionised: {ms:.1f} ms/stage -> {ms / ref:.2f}x of the spike "
              + ("(>25% slower: FLAG)" if ms > 1.25 * ref else "(within 25%)"))
        print(f"  (whole-run incl. compile: "
              f"{f32.get('t4_ms_per_stage_incl_compile', float('nan')):.1f} ms/stage; "
              f"compile {f32.get('t4_compile_s', float('nan')):.1f}s)")
        print(f"  GPU memory: {f32.get('gpu_mem_end')} "
              f"nvidia-smi used {f32.get('nvidia_smi_used_mib', float('nan')):.0f} MiB")
        print("  multi-event fp32 run:")
        for bk in f32["t3_batched32_diag"]["buckets"]:
            t = bk.get("timing", {})
            print(f"    bucket {bk['n_pol']}x{bk['n_ar']} B={bk['B']} "
                  f"events={bk['event_ids']}: {bk['stages']} stages, {bk['wall']:.2f}s "
                  f"(steady state {t.get('median_ms', float('nan')):.1f} ms/stage, "
                  f"first stage {t.get('first_stage_s', float('nan')):.1f}s)")
        print(f"    total wall for the 4-event batch: "
              f"{f32['t3_batched32_diag']['wall']:.1f}s")

        print("\n" + "=" * 78)
        print("TABLE 5 -- padding / mask leak tests")
        print("=" * 78)
        print("  garbage in every masked-out row (element-wise max|diff| vs the "
              "normal run):")
        for tag, blob in (("fp64 (P=%d, 1 chain)" % f64.get("mask_particles", 0),
                           f64.get("t3_mask_fp64_garbage", {})),
                          ("fp32 (4 events x 4 chains)",
                           f32.get("t3_mask_fp32_garbage", {}))):
            for e, d in blob.items():
                print(f"    {tag:<28s} {e}: {max(d.values()):.2e}")
        print("  re-padded to a multiple of 32 (element-wise max|diff|):")
        for tag, blob in (("fp64", f64.get("t3_mask_fp64_widepad", {})),
                          ("fp32", f32.get("t3_mask_fp32_widepad", {}))):
            for e, d in blob.items():
                print(f"    {tag:<28s} {e}: {max(d.values()):.2e}")
        print("  log-likelihood at 20 fixed positions, pad 8 vs pad 32 "
              "(relative):")
        for tag, blob in (("fp64", f64.get("t3_loglik_padwidth", {})),
                          ("fp32", f32.get("t3_loglik_padwidth", {}))):
            for e, v in blob.items():
                print(f"    {tag:<28s} {e}: rel {v['max_rel']:.2e}, "
                      f"bit-identical {v['n_bit_identical']}/{v['n_pos']}, "
                      f"shapes {v['shapes']}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "fp64", "fp32", "compare"])
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--ref-particles", type=int, default=1000)
    ap.add_argument("--ref-chains", type=int, default=4)
    ap.add_argument("--mask-particles", type=int, default=400)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "gpu_results"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="2 events instead of 4 (smoke test of the harness itself)")
    args = ap.parse_args()

    if args.quick:
        global MULTI_EVENTS, PERF_EVENTS
        MULTI_EVENTS = MULTI_EVENTS[:2]
        PERF_EVENTS = PERF_EVENTS[:2]

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.no_cache:
        cache = os.path.join(args.out_dir, "jax_cache")
        os.makedirs(cache, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    print(f"stage={args.stage} x64={jax.config.jax_enable_x64} "
          f"devices={jax.devices()}", flush=True)

    f64_path = os.path.join(args.out_dir, "fp64.json")
    f32_path = os.path.join(args.out_dir, "fp32.json")

    if args.stage in ("all", "fp64"):
        gpu_guard()
        out = stage_fp64(args)
        out["checks"] = [(n, ok, d) for n, ok, d in RESULTS]
        with open(f64_path, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print("wrote", f64_path, flush=True)

    if args.stage == "fp32":
        gpu_guard()
        out = stage_fp32(args)
        out["checks"] = [(n, ok, d) for n, ok, d in RESULTS]
        with open(f32_path, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print("wrote", f32_path, flush=True)

    if args.stage == "all":
        print("\n=== spawning fp32 subprocess ===", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--stage", "fp32",
               "--particles", str(args.particles), "--out-dir", args.out_dir]
        if args.no_cache:
            cmd.append("--no-cache")
        rc = subprocess.call(cmd, env=dict(os.environ))
        print(f"fp32 subprocess exit code: {rc}", flush=True)

    if args.stage in ("all", "compare"):
        f64 = json.load(open(f64_path)) if os.path.exists(f64_path) else {}
        f32 = json.load(open(f32_path)) if os.path.exists(f32_path) else {}
        if args.stage == "compare":
            RESULTS.extend((n, bool(ok), d) for n, ok, d in f64.get("checks", []))
        if f32:
            RESULTS.extend((n, bool(ok), d) for n, ok, d in f32.get("checks", []))
        if f64:
            stage_compare(f64, f32)

    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    print("\n" + "=" * 78)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed"
          + (f"  ({args.stage} stage)" if args.stage != "all" else ""))
    for n, ok, d in RESULTS:
        if not ok:
            print(f"  FAILED: {n}  --  {d}")
    print("=" * 78, flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
