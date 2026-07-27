"""REPRO G: device-buffer / jit-cache hygiene across buckets and calls.

Counts live device arrays (``jax.live_arrays()``) and jit cache size before and
after multi-bucket runs, to see whether ``run_batched`` leaks buffers or
executables from bucket to bucket.
"""
from stub import StubInv, make_prep, QUIET  # noqa: E402

import gc

import jax
import src.inversion_blackjax_batched as bx


def live():
    gc.collect()
    arrs = jax.live_arrays()
    return len(arrs), sum(a.nbytes for a in arrs) / 1e6


def main():
    n0, mb0 = live()
    print(f"baseline live arrays: {n0} ({mb0:.2f} MB)")

    # 6 buckets in ONE call (distinct shapes)
    inv = StubInv(num_particles=60, num_chains=2)
    events = [(f"ev{i}", make_prep(seed=i, n_pol=6 + 8 * i, n_ar=8))
              for i in range(6)]
    bx.run_batched(inv, events, progress_callback=QUIET)
    n1, mb1 = live()
    print(f"after 6-bucket call : {n1} ({mb1:.2f} MB)   delta={n1 - n0} arrays, "
          f"{mb1 - mb0:+.2f} MB")

    # repeat the same call three more times
    for k in range(3):
        inv = StubInv(num_particles=60, num_chains=2)
        bx.run_batched(inv, events, progress_callback=QUIET)
        n, mb = live()
        print(f"after repeat {k + 1}      : {n} ({mb:.2f} MB)   "
              f"delta vs first={n - n1} arrays, {mb - mb1:+.2f} MB")

    nfin, mbfin = live()
    leaked = nfin - n1
    print(f"\nbuffers retained across repeated calls: {leaked} "
          f"({mbfin - mb1:+.2f} MB)  -> {'LEAK' if leaked > 5 else 'no leak'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
