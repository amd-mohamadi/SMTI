"""REPRO B: every ``run_batched`` call recompiles, even for identical shapes.

``_run_bucket`` builds ``init_fn``/``step_fn`` as fresh Python closures and wraps
them in ``jax.jit`` on every call.  jit caches are keyed on the wrapped function
object, so two calls with byte-identical shapes/config produce two separate XLA
compilations.  Stage 3 measured ~21 s of compile per (shape, B) program in fp32
on the A40; an M2 driver that calls ``run_batched`` once per chunk of events
pays that on *every* chunk.

Measured by counting XLA compile events over a whole ``run_batched`` call and by
wall time, compared against a control that reuses one jitted callable.
"""
from stub import StubInv, make_prep, QUIET  # noqa: E402

import time

import jax
import jax.numpy as jnp
import src.inversion_blackjax_batched as bx

import logging

COMPILES = []


class _CompileCounter(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "Finished XLA compilation" in msg or "Compiling " in msg:
            COMPILES.append(msg.split("\n")[0][:90])


jax.config.update("jax_log_compiles", True)
_lg = logging.getLogger("jax")
_lg.setLevel(logging.DEBUG)
_lg.addHandler(_CompileCounter())


def main():
    ev = make_prep(seed=1)

    print("--- three run_batched calls, identical shapes/config ---")
    walls, counts = [], []
    for call in range(3):
        inv = StubInv(num_particles=40, num_chains=2)
        COMPILES.clear()
        t0 = time.time()
        bx.run_batched(inv, [("ev", ev)], progress_callback=QUIET)
        walls.append(time.time() - t0)
        counts.append(len(COMPILES))
        print(f"  call {call}: wall={walls[-1]:6.2f}s   xla compile events={counts[-1]}")

    print()
    print(f"compile events per call : {counts}")
    print(f"wall per call           : {[round(w, 2) for w in walls]}")
    # 6 log records == 3 programs x (Compiling..., Finished...) =
    # init_fn + step_fn + read_back, all recompiled from scratch.
    recompiled = counts[1] >= 6
    print(f"BUG: calls 2/3 recompile init_fn/step_fn/read_back from scratch "
          f"-> {recompiled}  (see show_last_call_compiles())")

    print()
    print("--- control: one jitted callable reused ---")
    COMPILES.clear()
    f = jax.jit(lambda x: jnp.sin(x) * 2 + 1)
    a = jnp.arange(10000.0)
    t0 = time.time(); f(a).block_until_ready(); t1 = time.time()
    n1 = len(COMPILES)
    f(a).block_until_ready(); t2 = time.time()
    print(f"  first  {t1 - t0:.4f}s (compiles={n1})")
    print(f"  second {t2 - t1:.4f}s (compiles={len(COMPILES) - n1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def show_last_call_compiles():
    """Print WHICH programs are recompiled on a repeat call."""
    ev = make_prep(seed=1)
    inv = StubInv(num_particles=40, num_chains=2)
    COMPILES.clear()
    bx.run_batched(inv, [("ev", ev)], progress_callback=QUIET)
    print("\n--- programs compiled on a repeat (warm-process) call ---")
    for m in COMPILES:
        print("   ", m)
