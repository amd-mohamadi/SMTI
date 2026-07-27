"""R2/R3: exercise the untested has_ar=False and has_pol=False bucket branches.

Stage-3 validation covered only events with both data types.  Run a tiny
batched SMC for a polarity-only and an amplitude-ratio-only configuration and
compare against the unbatched single-event path with the same chain seed.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "/s0/data/CAPE/smti_workflow/SMTI_dev")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_loading import load_event, SEED, LOCATION_SAMPLES_N, AZIMUTH_ERROR, TAKEOFF_ERROR  # noqa: E402

POL_ONLY = ["PPolarity", "SHPolarity"]
AR_ONLY = ["P/SHAmplitudeRatio", "P/SVAmplitudeRatio"]


def build(data, options, num_chains):
    from src.inversion_blackjax import InversionBlackJAX

    return InversionBlackJAX(
        data,
        inversion_options=options,
        num_particles=200,
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
        max_smc_iterations=40,
        num_chains=num_chains,
        chain_execution="sequential",
    )


def run(tag, options):
    from src.inversion_blackjax_batched import run_batched

    data = load_event("eq00126")
    inv = build(data, options, num_chains=2)
    print(f"\n===== {tag} =====")
    lines = []
    try:
        res = run_batched(inv, [("eq00126", data)], smc_dtype="float64",
                          progress_callback=lines.append)
    except Exception as exc:
        print(f"{tag}: FAILED with {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return None
    r = res["eq00126"]
    print(f"{tag}: OK  gamma shape={np.shape(r.gamma)} "
          f"mean={float(np.average(r.gamma, weights=r.weights)):.4f} "
          f"sigma_amp_ratio={'None' if r.sigma_amp_ratio is None else np.shape(r.sigma_amp_ratio)}")
    print("  last progress:", lines[-1] if lines else "(none)")

    # unbatched reference: same seed stream (num_chains=2 -> SeedSequence)
    inv2 = build(data, options, num_chains=2)
    ref = inv2._invert_multi_chain(inv2._filter_event_by_options(data))
    print(f"{tag}: unbatched gamma mean="
          f"{float(np.average(ref.gamma, weights=ref.weights)):.4f}")
    d = float(abs(np.average(r.gamma, weights=r.weights)
                  - np.average(ref.gamma, weights=ref.weights)))
    print(f"{tag}: |d gamma| = {d:.6f}")
    return r


if __name__ == "__main__":
    run("POL-ONLY (has_ar=False)", POL_ONLY)
    run("AR-ONLY (has_pol=False)", AR_ONLY)
