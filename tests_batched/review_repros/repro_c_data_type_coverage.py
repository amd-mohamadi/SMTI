"""REPRO C: bucketing pathologies the stage-2/3 tests never exercised.

  C1  pol-only event  (has_ar=False)   -> does the AR-less bucket run?
  C2  AR-only event   (has_pol=False)  -> does the pol-less bucket run?
  C3  mixed data types in one call     -> separate buckets, correct results?
  C4  bucket of exactly 1 entry (B=1)
  C5  per-observation ``incorrect_prob`` array mixed with the scalar form
  C6  single event, single chain (num_chains=1)
"""
from stub import StubInv, make_prep, QUIET  # noqa: E402

import numpy as np
import src.inversion_blackjax_batched as bx

OK = []


def check(name, ok, detail=""):
    OK.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def shapes(res, C, P, want_ar=True):
    return (
        res.mt6.shape == ((C, 6, P) if C > 1 else (6, P))
        and res.weights.shape == ((C, P) if C > 1 else (P,))
        and ((res.sigma_amp_ratio is not None) == want_ar)
    )


def main():
    P, C = 40, 2

    # ---- C1: polarity only -------------------------------------------------
    print("\nC1: polarity-only event (has_ar=False)")
    try:
        inv = StubInv(num_particles=P, num_chains=C)
        ev = make_prep(seed=1, has_ar=False)
        res = bx.run_batched(inv, [("pol", ev)], progress_callback=QUIET)["pol"]
        check("C1 pol-only runs and returns AR-free result",
              shapes(res, C, P, want_ar=False)
              and np.all(np.isfinite(res.mt6)),
              f"mt6{res.mt6.shape} sigma_amp_ratio={res.sigma_amp_ratio}")
    except Exception as exc:
        check("C1 pol-only runs", False, f"{type(exc).__name__}: {exc}")

    # ---- C2: amplitude ratio only -----------------------------------------
    print("\nC2: amplitude-ratio-only event (has_pol=False)")
    try:
        inv = StubInv(num_particles=P, num_chains=C)
        ev = make_prep(seed=2, has_pol=False)
        res = bx.run_batched(inv, [("ar", ev)], progress_callback=QUIET)["ar"]
        check("C2 AR-only runs", shapes(res, C, P) and np.all(np.isfinite(res.mt6)),
              f"mt6{res.mt6.shape}")
    except Exception as exc:
        check("C2 AR-only runs", False, f"{type(exc).__name__}: {exc}")

    # ---- C3: all three data types in one run_batched call ------------------
    print("\nC3: mixed pol-only / AR-only / both in one call")
    try:
        inv = StubInv(num_particles=P, num_chains=C)
        events = [
            ("both", make_prep(seed=3)),
            ("pol", make_prep(seed=4, has_ar=False)),
            ("ar", make_prep(seed=5, has_pol=False)),
        ]
        buckets = bx.prepare_batch(StubInv(num_particles=P, num_chains=C), events)
        check("C3a mixed data types land in 3 separate buckets",
              len(buckets) == 3,
              f"keys={[b.key for b in buckets]}")
        res = bx.run_batched(inv, events, progress_callback=QUIET)
        check("C3b all three events returned, finite",
              set(res) == {"both", "pol", "ar"}
              and all(np.all(np.isfinite(r.mt6)) for r in res.values()),
              f"{sorted(res)}")
    except Exception as exc:
        check("C3 mixed data types", False, f"{type(exc).__name__}: {exc}")

    # ---- C4: bucket of one entry ------------------------------------------
    print("\nC4: bucket of exactly one entry (B=1)")
    try:
        inv = StubInv(num_particles=P, num_chains=1)
        res = bx.run_batched(inv, [("solo", make_prep(seed=6))],
                             progress_callback=QUIET)["solo"]
        check("C4 B=1 runs, unstacked single-chain result",
              res.mt6.shape == (6, P) and res.num_chains == 1,
              f"mt6{res.mt6.shape} num_chains={res.num_chains}")
    except Exception as exc:
        check("C4 B=1", False, f"{type(exc).__name__}: {exc}")

    # ---- C5: per-observation incorrect_prob mixed with the scalar form -----
    print("\nC5: scalar and per-observation incorrect_prob in the same bucket")
    try:
        inv = StubInv(num_particles=P, num_chains=1)
        events = [
            ("scalar", make_prep(seed=7, scalar_incorrect=True)),
            ("vector", make_prep(seed=7, scalar_incorrect=False)),
        ]
        buckets = bx.prepare_batch(StubInv(num_particles=P, num_chains=1), events)
        same_bucket = len(buckets) == 1
        inc = buckets[0].data["incorrect_prob"]
        check("C5 both forms normalise to the same padded (B, N_pol) array",
              same_bucket and np.allclose(inc[0], inc[1]),
              f"buckets={len(buckets)}, row0={inc[0][:3]}, row1={inc[1][:3]}")
    except Exception as exc:
        check("C5 incorrect_prob normalisation", False, f"{type(exc).__name__}: {exc}")

    # ---- C6: num_chains = 1 result is NOT chain-stacked --------------------
    print("\nC6: num_chains=1 -> unstacked result (matches _invert_single_event)")
    try:
        inv = StubInv(num_particles=P, num_chains=1)
        res = bx.run_batched(inv, [("e1", make_prep(seed=8))],
                             progress_callback=QUIET)["e1"]
        check("C6 gamma is 1-D and num_chains==1",
              res.gamma.ndim == 1 and res.num_chains == 1,
              f"gamma{res.gamma.shape}")
    except Exception as exc:
        check("C6 num_chains=1", False, f"{type(exc).__name__}: {exc}")

    nfail = sum(1 for _n, ok in OK if not ok)
    print(f"\n{len(OK) - nfail}/{len(OK)} checks passed")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
