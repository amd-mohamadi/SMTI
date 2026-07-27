"""REPRO F2: the documented fp32 recipe is PROCESS-GLOBAL.

``inversion_blackjax_batched``'s module docstring tells the caller to do::

    from src.inversion_blackjax import InversionBlackJAX   # forces x64 on
    jax.config.update("jax_enable_x64", False)             # ... turn it off

M2 will do exactly that in the GPU driver process.  Any event that falls back to
the *unbatched* path in that same process (unsupported config, a bucket that is
too big, a retry) then silently runs the production float64 sampler in float32:
``src/tape_jax.py``'s hardcoded ``dtype=jnp.float64`` literals are downcast with
only a UserWarning, and nothing in ``_invert_single_event`` checks.
"""
import os
import sys
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
sys.path.insert(0, "/s0/data/CAPE/smti_workflow/SMTI_dev")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax

import src.inversion_blackjax as ib
from src.inversion_blackjax import InversionBlackJAX

# the documented recipe
jax.config.update("jax_enable_x64", False)

ib.build_location_samples_from_errors = lambda *a, **k: None

from stub import make_prep  # noqa: E402


class SynthInversion(InversionBlackJAX):
    def set_prep(self, prep):
        self._prep = prep

    def _prepare_event_arrays(self, event, location_samples=None):
        return dict(self._prep)

    def _filter_event_by_options(self, event):
        return event


def main():
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    prep = make_prep(seed=7, n_pol=8, n_ar=8)
    inv = SynthInversion(
        {}, inversion_options=None, num_particles=40, dc=False,
        gamma_beta_prior=(3.0, 3.0), delta_beta_prior=(3.0, 3.0),
        amp_ratio_sigma_prior=2.0, amp_ratio_noise_mode="global",
        random_seed=7, location_samples_n=3, azimuth_error=5, takeoff_error=10,
        mcmc_kernel="rmh", num_mcmc_steps=1, mechanism_steps=1,
        smc_target_ess_ratio=0.9, max_smc_iterations=60, num_chains=1,
        chain_execution="sequential",
    )
    inv.set_prep(prep)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = inv._invert_single_event(dict(prep), progress_callback=lambda _m: None)

    f64_warnings = [str(w.message)[:80] for w in caught
                    if "float64" in str(w.message) or "x64" in str(w.message)]
    print(f"\nunbatched fallback in the fp32 process:")
    print(f"  result gamma dtype   : {np.asarray(res.gamma).dtype}")
    print(f"  result mt6 dtype     : {np.asarray(res.mt6).dtype}   (numpy-cast, "
          f"so this hides it)")
    print(f"  distinct float64 downcast warnings: {len(set(f64_warnings))}")
    for w in sorted(set(f64_warnings))[:3]:
        print(f"    {w}")

    # the real evidence: the sampler's own arrays
    import jax.numpy as jnp
    from src.tape_jax import jax_Tape_MT6
    out = jax_Tape_MT6(*(jnp.asarray(v) for v in (0.1, 0.2, 1.0, 0.5, 0.3)))
    print(f"  src.tape_jax.jax_Tape_MT6 output dtype in this process: {out.dtype} "
          f"(float64 in the production process)")

    degraded = out.dtype == jnp.float32
    print(f"\nBUG: the production float64 path silently degrades to float32 "
          f"in the fp32 process -> {degraded}")
    return 0 if degraded else 1


if __name__ == "__main__":
    raise SystemExit(main())
