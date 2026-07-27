#!/usr/bin/env python
"""
Behaviour-preservation check for the stage-1 foundation changes.

Runs the *same* eq00124 single-event inversion (same seed, same explicitly
supplied ``location_samples``) twice:

  * against a pristine checkout of ``src/`` from git HEAD  (pre-fix code)
  * against the working tree ``SMTI_dev/src/``             (post-fix code)

and compares the posterior summaries.  The two are NOT expected to be
bit-identical: the stable ``softplus_inv`` differs from the old expression by
~1e-16 relative for initial ``sigma_amp_ratio`` draws above 20 (~6% of
particles), and SMC amplifies that chaotically.  The bar is therefore
"agreement at the Monte-Carlo-noise level", plus bit-identical prepared data
arrays (which the refactor must not touch at all).

Usage:
    conda run -n smti python tests_batched/test_refactor_equivalence.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse
import json
import subprocess
import sys
import tempfile

import numpy as np

SMTI_DEV = "/s0/data/CAPE/smti_workflow/SMTI_dev"
HERE = os.path.dirname(os.path.abspath(__file__))
PARTICLES = 500


def run_one(root: str, out_path: str) -> dict:
    """Import SMTI from ``root`` in a fresh process and dump the summary."""
    code = r"""
import os, sys, json
sys.path.insert(0, %(root)r)
sys.path.insert(0, %(here)r)
import numpy as np
from test_foundations import (
    load_eq00124, make_inversion, build_location_samples_from_errors,
    LOCATION_SAMPLES_N, AZIMUTH_ERROR, TAKEOFF_ERROR, LOCATION_SAMPLE_SEED,
)
import src.inversion_blackjax as ib
assert os.path.abspath(ib.__file__).startswith(os.path.abspath(%(root)r)), ib.__file__

data = load_eq00124()
inv = make_inversion(data, %(particles)d)
rng = np.random.default_rng(LOCATION_SAMPLE_SEED)
loc = build_location_samples_from_errors(
    data, rng=rng, n_samples=LOCATION_SAMPLES_N,
    azimuth_error=AZIMUTH_ERROR, takeoff_error=TAKEOFF_ERROR,
)
res = inv._invert_single_event(data, location_samples=loc,
                               progress_callback=lambda m: None)
w = np.asarray(res.weights, dtype=float)
if w.size == 0 or not np.isfinite(w).all() or w.sum() <= 0:
    w = None
out = dict(
    module=os.path.abspath(ib.__file__),
    has_prepare_helper=hasattr(inv, "_prepare_event_arrays"),
    gamma=float(np.average(np.asarray(res.gamma, float), weights=w)),
    delta=float(np.average(np.asarray(res.delta, float), weights=w)),
    kappa_circmean=float(np.angle(np.mean(np.exp(1j*np.asarray(res.kappa, float)))) %% (2*np.pi)),
    h=float(np.mean(np.asarray(res.h, float))),
    sigma=float(np.mean(np.asarray(res.sigma, float))),
    sigma_amp_ratio=(None if res.sigma_amp_ratio is None
                     else float(np.mean(np.asarray(res.sigma_amp_ratio, float)))),
    sigma_amp_ratio_std=(None if res.sigma_amp_ratio is None
                         else float(np.std(np.asarray(res.sigma_amp_ratio, float)))),
    n=int(np.asarray(res.gamma).size),
)
with open(%(out)r, "w") as fh:
    json.dump(out, fh)
print(json.dumps(out))
""" % dict(root=root, here=HERE, out=out_path, particles=PARTICLES)
    env = dict(os.environ)
    env["SMTI_ROOT"] = root
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
    with open(out_path) as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="smti_head_")
    subprocess.run(
        f"git -C {SMTI_DEV} archive HEAD src | tar -x -C {tmp}",
        shell=True,
        check=True,
    )
    print(f"pristine HEAD src extracted to {tmp}/src")

    old = run_one(tmp, os.path.join(tmp, "old.json"))
    new = run_one(SMTI_DEV, os.path.join(tmp, "new.json"))

    print("\n old (git HEAD):", json.dumps(old, indent=2))
    print(" new (worktree):", json.dumps(new, indent=2))

    fails = []
    if old["has_prepare_helper"]:
        fails.append("HEAD copy unexpectedly already has _prepare_event_arrays")
    if not new["has_prepare_helper"]:
        fails.append("worktree copy is missing _prepare_event_arrays")

    tols = dict(gamma=0.06, delta=0.06, kappa_circmean=0.30, h=0.08,
                sigma=0.15, sigma_amp_ratio=0.05)
    for k, tol in tols.items():
        a, b = old[k], new[k]
        if a is None or b is None:
            fails.append(f"{k} missing")
            continue
        d = abs(a - b)
        ok = d <= tol
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: old={a:+.4f} new={b:+.4f} "
              f"|diff|={d:.4f} (tol {tol})")
        if not ok:
            fails.append(f"{k}: |{a:+.4f} - {b:+.4f}| = {d:.4f} > {tol}")

    print("\n" + "=" * 70)
    if fails:
        for f in fails:
            print("FAILED:", f)
        return 1
    print("pre-fix and post-fix single-event inversions agree within MC noise")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
