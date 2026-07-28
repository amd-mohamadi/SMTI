"""CPU batched smoke: mcmc_kernel=mwg_irmh on eq00124 (small P/chains)."""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")

SMTI = os.environ.get("SMTI_ROOT", "/s0/data/CAPE/smti_workflow/SMTI")
if SMTI not in sys.path:
    sys.path.insert(0, SMTI)

import numpy as np

from src.data_loader import read_data
from src.inversion_blackjax import InversionBlackJAX
from src.inversion_blackjax_batched import clear_program_cache, run_batched


DAT = "/s0/data/CAPE/smti_workflow/inversion_data/data/eq00124/data.dat"
VEL = "/s0/data/CAPE/smti_workflow/cape_surface_corrected.tvel"
OPTS = [
    "PPolarity",
    "SHPolarity",
    "P/SHAmplitudeRatio",
    "P/SVAmplitudeRatio",
]


def _load_event():
    data = read_data(
        DAT,
        velocity_model_path=VEL,
        min_p_phase_score=0.55,
        min_s_phase_score=0.55,
        min_ppl_score=0.6,
        min_shpl_score=0.7,
        min_svpl_score=0.7,
    )
    if isinstance(data, list):
        data = data[0]
    data = dict(data)
    data["UID"] = "eq00124"
    return data


def main():
    clear_program_cache()
    event = _load_event()
    inv = InversionBlackJAX(
        event,
        inversion_options=OPTS,
        num_particles=120,
        num_chains=2,
        num_mcmc_steps=3,
        mcmc_kernel="mwg_irmh",
        irmh_n_components=2,
        irmh_steps=1,
        irmh_em_iters=4,
        adapt_proposal=True,
        rmh_proposal_scale=0.02,
        max_smc_iterations=30,
        random_seed=421,
        mechanism_steps=2,
        smc_target_ess_ratio=0.9,
        amp_ratio_noise_mode="global",
        dc=False,
        chain_execution="sequential",
    )
    t0 = time.time()
    results = run_batched(inv, [("eq00124", event)], smc_dtype="float64")
    wall = time.time() - t0
    assert "eq00124" in results, list(results)
    r = results["eq00124"]
    assert r.ln_p is not None, "ln_p missing"
    lnp = np.asarray(r.ln_p)
    assert np.isfinite(lnp).mean() > 0.9, np.isfinite(lnp).mean()
    print(
        f"PASS batched mwg_irmh in {wall:.1f}s  "
        f"ln_p={lnp.shape} max={float(np.nanmax(lnp)):.2f}  "
        f"num_chains={r.num_chains} stalled={r.tempering_stalled}"
    )


if __name__ == "__main__":
    main()
