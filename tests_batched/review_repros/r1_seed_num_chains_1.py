"""R1: num_chains=1 -> batched path uses a DIFFERENT PRNG seed than forward().

forward() with num_chains == 1 calls _invert_single_event(filtered), which uses
    key = random.PRNGKey(self.random_seed)
prepare_batch() always derives chain seeds via
    np.random.SeedSequence(inv.random_seed).generate_state(num_chains)
so for num_chains == 1 the batched entry is seeded with
SeedSequence(seed).generate_state(1)[0] != seed.

Claimed in the run_batched/prepare_batch docstring: "a batched chain and its
unbatched twin start from the same particles".
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "/s0/data/CAPE/smti_workflow/SMTI_dev")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_loading import load_event, make_inversion, SEED  # noqa: E402


def main():
    ss_seed = int(np.random.SeedSequence(SEED).generate_state(1)[0])
    print(f"random_seed              = {SEED}")
    print(f"SeedSequence(seed)[0]    = {ss_seed}")
    print(f"equal? {ss_seed == SEED}")

    from src.inversion_blackjax_batched import prepare_batch, _make_init_particle
    import jax
    from jax import random

    data = load_event("eq00126")
    inv = make_inversion(data, num_particles=64, num_chains=1,
                         chain_execution="sequential")
    buckets = prepare_batch(inv, [("eq00126", data)])
    b = buckets[0]
    print("bucket:", b, "entry seeds:", [e.seed for e in b.entries])
    assert b.entries[0].seed == ss_seed

    # particles that each path would start from
    init_particle = _make_init_particle(
        has_ar=b.has_ar,
        gamma_beta_prior=inv.gamma_beta_prior,
        delta_beta_prior=inv.delta_beta_prior,
        amp_ratio_sigma_prior=inv.amp_ratio_sigma_prior,
    )

    def first_particles(seed):
        key = random.PRNGKey(int(seed))
        key, ik = random.split(key)
        keys = random.split(ik, 8)
        return np.asarray(jax.vmap(init_particle)(keys)["gamma"])

    p_batched = first_particles(b.entries[0].seed)
    p_unbatched = first_particles(SEED)  # what _invert_single_event uses
    print("batched   gamma[:4]:", p_batched[:4])
    print("unbatched gamma[:4]:", p_unbatched[:4])
    print("max|diff| =", float(np.max(np.abs(p_batched - p_unbatched))))
    print("VERDICT: divergent initial particles for num_chains=1"
          if not np.allclose(p_batched, p_unbatched) else "VERDICT: identical")


if __name__ == "__main__":
    main()
