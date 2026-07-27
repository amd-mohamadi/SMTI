"""REPRO H: test_batched_core.py test 1b cannot fail on NaN.

``test_pad_mask_equivalence`` accumulates the error with

    worst[tag] = max(worst[tag], rel)

and ``rel`` is ``abs(got - ref) / max(1, abs(ref))``.  If the padded/masked
log-likelihood returns NaN, ``rel`` is NaN and ``max(0.0, nan)`` returns **0.0**
in Python, so the running maximum never moves and the check
"padded vs unpadded production loglik, rel < 1e-12" reports max_rel=0.000e+00
and PASSES.  The ``finite`` guard next to it only inspects the *reference*
value, not the batched one.

This repro replays the exact accumulator logic on a deliberately NaN-producing
"implementation" and shows the test would have passed.
"""
import numpy as np


def replay_test_1b(loglik_impl):
    """Byte-for-byte the accumulator from tests_batched/test_batched_core.py:272-296."""
    rng = np.random.default_rng(11)
    worst = 0.0
    finite = True
    for _ in range(20):
        pos = rng.normal(0, 1.5, size=6)
        ref = -123.456                       # a finite production reference
        finite = finite and np.isfinite(ref)
        got = loglik_impl(pos)
        rel = abs(got - ref) / max(1.0, abs(ref))
        worst = max(worst, rel)
    passed = worst < 1e-12 and finite
    return passed, worst


def good(pos):
    return -123.456


def broken_nan(pos):
    return float("nan")


def broken_wrong(pos):
    return -100.0


def main():
    for name, impl in (("correct", good),
                       ("returns NaN", broken_nan),
                       ("returns a wrong finite value", broken_wrong)):
        passed, worst = replay_test_1b(impl)
        print(f"  implementation {name:30s} -> test 1b "
              f"{'PASSES' if passed else 'fails '}  (max_rel={worst:.3e})")

    print()
    print(f"python max(0.0, nan) = {max(0.0, float('nan'))}   <- the mechanism")
    p_nan, _ = replay_test_1b(broken_nan)
    print(f"BUG: a NaN log-likelihood passes test 1b -> {p_nan}")
    return 0 if p_nan else 1


if __name__ == "__main__":
    raise SystemExit(main())
