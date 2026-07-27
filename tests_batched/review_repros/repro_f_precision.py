"""REPRO F: precision-related integration questions.

  F1  the ``softplus_inv`` rewrite in ``inversion_blackjax.py`` changes the
      float64 initial particles of the *unbatched production path* for
      sigma_amp_ratio > ~4.8e8 ... does it?  (bit-equality vs the old form over
      the real init range)
  F2  the documented fp32 recipe is process-global: with x64 turned off, the
      unbatched fallback path silently runs in float32.
  F3  ``smc_dtype='float32'`` inside an x64-enabled process: does it stay fp32?
"""
from stub import StubInv, make_prep, QUIET  # noqa: E402

import warnings

import numpy as np
import jax
import jax.numpy as jnp

import src.inversion_blackjax_batched as bx
from src.inversion_blackjax import softplus_inv

OK = []


def check(name, ok, detail=""):
    OK.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def f1_softplus():
    print("\nF1: softplus_inv rewrite vs the original expression (float64)")
    eps = 1e-6

    def old(x):
        return jnp.log(jnp.expm1(jnp.maximum(x, eps)) + eps)

    # the actual init distribution: sigma_ar = 2 * tan(pi*u/2), clipped [1e-6, 100]
    u = np.linspace(1e-6, 1.0 - 1e-6, 200001)
    sigma_ar = np.clip(2.0 * np.tan(np.pi * u / 2.0), 1e-6, 100.0)
    x = jnp.asarray(sigma_ar, dtype=jnp.float64)
    a = np.asarray(old(x))
    b = np.asarray(softplus_inv(x, eps))
    exact = int(np.sum(a == b))
    big = sigma_ar > 20.0
    print(f"    inputs={a.size}, bit-identical={exact} ({100 * exact / a.size:.3f}%)")
    print(f"    max|diff| overall = {np.max(np.abs(a - b)):.3e}")
    print(f"    of the {int(big.sum())} particles on the x>20 branch: "
          f"bit-identical={int(np.sum(a[big] == b[big]))}, "
          f"max|diff|={np.max(np.abs(a[big] - b[big])):.3e}")
    check("F1 softplus_inv is bit-identical to the old form on every "
          "float64 init particle (unbatched production results unchanged)",
          exact == a.size,
          f"{a.size - exact} particles differ, max|diff|={np.max(np.abs(a - b)):.3e}")
    # and the bug it fixes
    print(f"    float32 overflow check: old(f32 100.0) = "
          f"{float(old(jnp.float32(100.0))):.4g}, "
          f"new = {float(softplus_inv(jnp.float32(100.0), eps)):.4g}")


def f3_fp32_in_x64_process():
    print("\nF3: smc_dtype='float32' inside an x64-enabled process")
    print(f"    jax_enable_x64 = {jax.config.jax_enable_x64}")
    inv = StubInv(num_particles=40, num_chains=1)
    captured = {}
    orig = bx._run_bucket

    def spy(*a, **k):
        out = orig(*a, **k)
        captured.update(out)
        return out

    bx._run_bucket = spy
    msgs = []
    try:
        bx.run_batched(inv, [("ev", make_prep(seed=5))],
                       smc_dtype="float32", progress_callback=msgs.append)
    finally:
        bx._run_bucket = orig
    warned = any("WARNING" in m for m in msgs)
    print(f"    warning emitted    : {warned}")
    print(f"    particle dtype     : {captured['particle_dtype']}")
    check("F3 fp32 request in an x64 process warns and still yields fp32 particles",
          warned and captured["particle_dtype"] == np.float32,
          f"warned={warned}, dtype={captured['particle_dtype']}")


def main():
    f1_softplus()
    f3_fp32_in_x64_process()
    print("\nF2 runs in a separate x64-disabled process: repro_f2_fp32_fallback.py")
    nfail = sum(1 for _n, ok in OK if not ok)
    print(f"\n{len(OK) - nfail}/{len(OK)} checks passed")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
