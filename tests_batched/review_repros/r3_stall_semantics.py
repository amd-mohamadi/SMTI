"""R3: per-entry stall guard does NOT stop an entry's sampling.

Unbatched (_invert_single_event): when delta_beta < min_tempering_increment for
`tempering_stall_patience` consecutive stages the loop BREAKS and the particles
at that moment are returned.

Batched (_run_bucket): `stalled` only removes the entry from the `done` test;
the vmapped step keeps advancing that entry (its own tempering bisection keeps
running) until every other entry finishes.  So a stalled entry
  * returns particles from a later, more-tempered stage than its unbatched twin
  * can end at lambda == 1 while still being reported tempering_stalled=True.

Settings are chosen so different chains stall at different stages
(min_tempering_increment=0.02, patience=1).
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/s0/data/CAPE/smti_workflow/SMTI_dev")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_loading import (  # noqa: E402
    load_event, SEED, LOCATION_SAMPLES_N, AZIMUTH_ERROR, TAKEOFF_ERROR,
    INVERSION_OPTIONS,
)

MIN_INC = 0.013
PATIENCE = 5
P = 200
NC = 4


def build(data):
    from src.inversion_blackjax import InversionBlackJAX

    return InversionBlackJAX(
        data,
        inversion_options=INVERSION_OPTIONS,
        num_particles=P,
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
        num_mcmc_steps=3,
        mechanism_steps=3,
        smc_target_ess_ratio=0.9,
        max_smc_iterations=60,
        min_tempering_increment=MIN_INC,
        tempering_stall_patience=PATIENCE,
        num_chains=NC,
        chain_execution="sequential",
    )


def main():
    from src.inversion_blackjax_batched import prepare_batch, _run_bucket
    import jax.numpy as jnp

    data = load_event("eq00126")

    # ---- batched ----
    inv = build(data)
    buckets = prepare_batch(inv, [("eq00126", data)])
    lines = []
    out = _run_bucket(inv, buckets[0], jnp.float64, lines.append, "[b]")
    print("\n--- batched per-stage progress ---")
    for ln in lines:
        print("   ", ln)
    print("\nbatched final lambda per entry :", np.round(out["lambda"], 4))
    print("batched stalled flag per entry :", out["stalled"])
    print("batched stages                 :", out["stages"])

    gam_b = out["params"]["gamma"]
    w_b = out["weights"]
    mean_b = [float(np.average(gam_b[i], weights=w_b[i])) for i in range(NC)]

    # ---- unbatched reference (same seeds, same stall settings) ----
    inv2 = build(data)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ref = inv2._invert_multi_chain(inv2._filter_event_by_options(data))
    text = buf.getvalue()
    betas = re.findall(r"Stage (\d+): beta=([0-9.]+)", text)
    stops = [m for m in re.finditer(r"SMC completed in [0-9.]+s with beta=([0-9.]+)", text)]
    print("\nunbatched per-chain final beta  :",
          [round(float(m.group(1)), 4) for m in stops])
    n_stall_msgs = text.count("SMC tempering stalled")
    print("unbatched chains that stalled   :", n_stall_msgs, "/", NC)

    mean_u = [float(np.average(ref.gamma[i], weights=ref.weights[i])) for i in range(NC)]

    print("\nper-chain weighted gamma mean")
    print("  batched   :", np.round(mean_b, 5))
    print("  unbatched :", np.round(mean_u, 5))
    print("  |diff|    :", np.round(np.abs(np.array(mean_b) - np.array(mean_u)), 5))

    stalled = out["stalled"]
    lam = out["lambda"]
    bad = [i for i in range(NC) if stalled[i] and lam[i] >= 1.0 - 1e-6]
    print("\nentries flagged tempering_stalled but ending at lambda>=1:", bad)


if __name__ == "__main__":
    main()
