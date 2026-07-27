"""REPRO A: two events sharing an event_id are silently merged into one result.

``run_batched`` groups batch entries with ``per_entry.setdefault(entry.event_id, ...)``
and then hands the whole list to ``_stack_chain_results``, which stacks *all* of
them and labels the result ``num_chains=self.num_chains``.  With a duplicated
id the returned InversionResult has ``2 * num_chains`` rows but claims
``num_chains``, the two events' chains are interleaved by chain index, and one
event vanishes from the result dict.

Expected (correct) behaviour: either an explicit error, or two separate results.
"""
from stub import StubInv, make_prep, QUIET  # noqa: E402
import numpy as np
import src.inversion_blackjax_batched as bx


def main():
    P, C = 40, 2
    inv = StubInv(num_particles=P, num_chains=C)
    evA = make_prep(seed=1)
    evB = make_prep(seed=2)

    # exact same shapes -> same bucket; DIFFERENT data, SAME id
    res = bx.run_batched(inv, [("ev", evA), ("ev", evB)],
                         progress_callback=QUIET)

    print(f"result keys            : {sorted(res)}")
    r = res["ev"]
    print(f"mt6.shape              : {r.mt6.shape}      (expected ({C}, 6, {P}))")
    print(f"gamma.shape            : {r.gamma.shape}")
    print(f"weights.shape          : {r.weights.shape}")
    print(f"num_chains attribute   : {r.num_chains}     (expected {C})")

    bad_shape = r.mt6.shape[0] != r.num_chains
    lost = len(res) == 1
    print()
    print(f"BUG: n_chain_rows ({r.mt6.shape[0]}) != num_chains ({r.num_chains})  -> {bad_shape}")
    print(f"BUG: one of the two events is missing from the result dict -> {lost}")

    # what the driver does with it (4_run_inversion.py:837-846)
    mt6 = np.asarray(r.mt6, dtype=float)
    if mt6.ndim == 3:
        if mt6.shape[1] == 6:
            mt6 = np.transpose(mt6, (1, 0, 2))
        mt6 = mt6.reshape(6, -1)
    w = np.asarray(r.weights, dtype=float).reshape(-1)
    print(f"driver mt6 flat shape  : {mt6.shape}, weights {w.shape} "
          f"-> sizes {'match (silently mixes 2 events)' if mt6.shape[1] == w.size else 'MISMATCH'}")

    # And what ArviZ sees: 2*C 'chains' declared as C
    print(f"arviz posterior would get {r.gamma.shape[0]} chains for a "
          f"{C}-chain inversion")
    return 0 if (bad_shape and lost) else 1


if __name__ == "__main__":
    raise SystemExit(main())
