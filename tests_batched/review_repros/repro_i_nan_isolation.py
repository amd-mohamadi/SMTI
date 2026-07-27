"""REPRO I: does one bad entry poison the rest of the bucket?

Injects NaN into one event's ``amp_ratio_obs`` (the sort of thing a corrupt
Step-3 input produces) and checks
  (a) the bucket still completes (no RuntimeError from the non-finite-lambda
      guard),
  (b) the healthy events in the same bucket are bit-identical to a run without
      the poisoned neighbour.
"""
from stub import StubInv, make_prep, QUIET  # noqa: E402

import numpy as np
import src.inversion_blackjax_batched as bx

OK = []


def check(name, ok, detail=""):
    OK.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def run(events):
    inv = StubInv(num_particles=60, num_chains=2)
    return bx.run_batched(inv, events, progress_callback=QUIET)


def main():
    good1 = make_prep(seed=31, n_pol=8, n_ar=8)
    good2 = make_prep(seed=32, n_pol=8, n_ar=8)
    bad = make_prep(seed=33, n_pol=8, n_ar=8)
    bad["amp_ratio_obs"] = bad["amp_ratio_obs"].copy()
    bad["amp_ratio_obs"][2] = np.nan

    print("\nI1: clean reference run (2 good events)")
    ref = run([("g1", good1), ("g2", good2)])

    print("\nI2: same bucket with a NaN event inserted FIRST")
    try:
        got = run([("bad", bad), ("g1", good1), ("g2", good2)])
        check("I2 bucket completes despite a NaN entry", True,
              f"keys={sorted(got)}")
    except Exception as exc:
        check("I2 bucket completes despite a NaN entry", False,
              f"{type(exc).__name__}: {exc}")
        return 1

    print(f"    bad event mt6 finite : "
          f"{bool(np.all(np.isfinite(got['bad'].mt6)))} "
          f"(loglik clipped to -1e3 by nan_to_num, so it samples the prior)")

    worst = {}
    for eid in ("g1", "g2"):
        worst[eid] = max(
            float(np.max(np.abs(np.asarray(getattr(ref[eid], f), dtype=float)
                                - np.asarray(getattr(got[eid], f), dtype=float))))
            for f in ("gamma", "delta", "kappa", "h", "sigma", "weights", "mt6")
        )
    check("I3 healthy neighbours are unaffected by the NaN entry "
          "(vmap keeps lanes independent)",
          all(v < 1e-12 for v in worst.values()),
          f"max|diff| {worst}")

    nfail = sum(1 for _n, ok in OK if not ok)
    print(f"\n{len(OK) - nfail}/{len(OK)} checks passed")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
