"""R5: what smc_dtype actually delivers under the two x64 process states.

Case 1: x64 ON  (the default after importing InversionBlackJAX) + smc_dtype='float32'
        -> only a WARNING is printed; does the sampler actually run in float32?
Case 2: x64 OFF (the fp32 recipe) + smc_dtype='float64'
        -> no warning at all; does it silently run in float32?
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, "/s0/data/CAPE/smti_workflow/SMTI_dev")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODE = sys.argv[1] if len(sys.argv) > 1 else "x64on_f32"

from event_loading import load_event, make_inversion  # noqa: E402
import jax  # noqa: E402

if MODE == "x64off_f64":
    from src.inversion_blackjax import InversionBlackJAX  # forces x64 on  # noqa
    jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402


def main():
    from src.inversion_blackjax_batched import prepare_batch, _run_bucket

    print(f"MODE={MODE}  jax_enable_x64={jax.config.jax_enable_x64}")
    data = load_event("eq00126")
    inv = make_inversion(data, num_particles=64, num_chains=2, num_mcmc_steps=2,
                         chain_execution="sequential")
    inv.max_smc_iterations = 3
    buckets = prepare_batch(inv, [("eq00126", data)])
    dtype = jnp.float32 if MODE.endswith("f32") else jnp.float64
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = _run_bucket(inv, buckets[0], dtype, lambda s: None, "[x]")
        msgs = [str(x.message)[:90] for x in w]
    print("requested dtype :", np.dtype(dtype).name)
    print("actual particle dtype:", out["particle_dtype"])
    print("warnings:", msgs[:3])


if __name__ == "__main__":
    main()
