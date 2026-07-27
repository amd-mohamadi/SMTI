"""R4: (a) duplicate event_id silently merges chains; (b) InversionResult field parity."""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, "/s0/data/CAPE/smti_workflow/SMTI_dev")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_loading import load_event, make_inversion  # noqa: E402


def main():
    from src.inversion_blackjax_batched import run_batched

    data = load_event("eq00126")

    # ---- (a) duplicate entry (the M2 "repeat-pad the last group" pattern) ----
    inv = make_inversion(data, num_particles=100, num_chains=2, num_mcmc_steps=2,
                         chain_execution="sequential")
    inv.max_smc_iterations = 6      # keep it short; correctness of shapes is the point
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = run_batched(inv, [("eq00126", data), ("eq00126", data)],
                          smc_dtype="float64", progress_callback=lambda s: None)
    print("(a) duplicate event_id")
    print("    keys              :", list(res.keys()))
    r = res["eq00126"]
    print("    gamma shape       :", np.shape(r.gamma), " (expect (2, 100))")
    print("    mt6 shape         :", np.shape(r.mt6))
    print("    weights shape     :", np.shape(r.weights))
    print("    result.num_chains :", r.num_chains)
    print("    MISMATCH" if np.shape(r.gamma)[0] != r.num_chains else "    ok")

    # ---- (b) field-by-field parity vs the unbatched multi-chain result ----
    print("\n(b) field parity batched vs _invert_multi_chain")
    inv2 = make_inversion(data, num_particles=100, num_chains=2, num_mcmc_steps=2,
                          chain_execution="sequential")
    inv2.max_smc_iterations = 6
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rb = run_batched(inv2, [("eq00126", data)], smc_dtype="float64",
                         progress_callback=lambda s: None)["eq00126"]
    inv3 = make_inversion(data, num_particles=100, num_chains=2, num_mcmc_steps=2,
                          chain_execution="sequential")
    inv3.max_smc_iterations = 6
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ru = inv3._invert_multi_chain(inv3._filter_event_by_options(data))

    import dataclasses
    for f in dataclasses.fields(rb):
        a = getattr(rb, f.name)
        b = getattr(ru, f.name)
        def d(x):
            if x is None:
                return "None"
            if isinstance(x, np.ndarray):
                return f"ndarray{x.shape}/{x.dtype}"
            return f"{type(x).__name__}={x}"
        flag = "" if d(a) == d(b) else "   <-- DIFFERS"
        print(f"    {f.name:32s} batched={d(a):28s} unbatched={d(b):28s}{flag}")

    print("\n    weights row sums batched  :", np.round(rb.weights.sum(axis=1), 8))
    print("    weights row sums unbatched:", np.round(ru.weights.sum(axis=1), 8))


if __name__ == "__main__":
    main()
