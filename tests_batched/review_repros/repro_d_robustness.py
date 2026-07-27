"""REPRO D: robustness / resource-hygiene gaps from the M2 driver's point of view.

  D1  ``run_batched`` puts every same-shaped event in ONE bucket with
      B = n_events x num_chains and offers no way to cap B (plan 4 wants
      B = GPU_BATCH_EVENTS * NUM_CHAINS = 16).
  D2  a failure inside bucket k discards the already-finished buckets 0..k-1;
      today's driver isolates failures per event (4_run_inversion.py:1395).
  D3  an exception from ``progress_callback`` aborts the whole run.
  D4  an entry that trips the tempering-stall guard is NOT stopped at the stall
      point (the unbatched path breaks out of its loop there); it keeps being
      stepped until the rest of the batch finishes and can end at beta = 1
      while still being reported ``tempering_stalled=True``.
"""
from stub import StubInv, make_prep, QUIET  # noqa: E402

import numpy as np
import src.inversion_blackjax_batched as bx

OK = []


def check(name, ok, detail=""):
    OK.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


# ---------------------------------------------------------------- D1
def d1_unbounded_batch():
    print("\nD1: no cap on bucket size B")
    inv = StubInv(num_particles=40, num_chains=4)
    events = [(f"ev{i}", make_prep(seed=i)) for i in range(60)]
    buckets = bx.prepare_batch(inv, events)
    B = buckets[0].size
    print(f"  60 same-shaped events x 4 chains -> {len(buckets)} bucket(s), B={B}")
    import inspect
    sig = inspect.signature(bx.run_batched)
    knobs = [p for p in sig.parameters if p not in ("inv", "events")]
    print(f"  run_batched knobs: {knobs}")
    check("D1 B grows without bound and no max-batch parameter exists",
          B == 240 and not any("batch" in k or "max" in k for k in knobs),
          f"B={B}, params={knobs}")

    # what that costs on the GPU at production shapes
    P, n_ar, n_loc = 2000, 64, 20
    per_tensor = B * P * n_ar * n_loc * 4 / 1e9  # fp32
    print(f"  at production P={P}, N_ar={n_ar}, N_loc={n_loc}, fp32: ONE likelihood "
          f"intermediate of shape (B,P,N_ar,N_loc) = {per_tensor:.1f} GB "
          f"(amp1_pred/amp2_pred/log_ratio_pred/log_prob_samples are 4 of them, "
          f"x num_mcmc_steps inner evaluations)")


# ---------------------------------------------------------------- D2
def d2_bucket_failure_discards_everything():
    print("\nD2: a failure in bucket 2 discards bucket 1's completed work")
    inv = StubInv(num_particles=40, num_chains=2)
    events = [
        ("small", make_prep(seed=1, n_pol=6, n_ar=8)),    # bucket (8, 8)
        ("big", make_prep(seed=2, n_pol=20, n_ar=8)),     # bucket (24, 8)
    ]
    orig = bx._run_bucket
    seen = []

    def failing(inv_, bucket, dtype, progress, label):
        seen.append(bucket.key)
        if len(seen) == 2:
            # exactly what _run_bucket raises on a non-finite tempering param
            # (inversion_blackjax_batched.py:742-746), and what an OOM would do
            raise RuntimeError("Batched SMC produced a non-finite tempering parameter")
        return orig(inv_, bucket, dtype, progress, label)

    bx._run_bucket = failing
    try:
        bx.run_batched(inv, events, progress_callback=QUIET)
        check("D2 exception propagates", False, "no exception raised")
    except RuntimeError as exc:
        check("D2 one bad bucket aborts run_batched; the finished bucket's "
              "results are unrecoverable", True,
              f"{len(seen)} buckets attempted, 0 results returned ({exc})")
    finally:
        bx._run_bucket = orig


# ---------------------------------------------------------------- D3
def d3_progress_callback_exception():
    print("\nD3: an exception from progress_callback kills the run")
    inv = StubInv(num_particles=40, num_chains=1)
    calls = {"n": 0}

    def flaky(msg):
        calls["n"] += 1
        if calls["n"] == 3:          # e.g. a full disk / closed log file
            raise IOError("log write failed")

    try:
        bx.run_batched(inv, [("ev", make_prep(seed=3))], progress_callback=flaky)
        check("D3 progress_callback failure is contained", False,
              "run completed (callback exception swallowed)")
    except IOError as exc:
        check("D3 progress_callback exception aborts the whole bucket "
              "(no try/except around `progress(...)`)", True, str(exc))
    finally:
        pass


# ---------------------------------------------------------------- D4
def d4_stall_semantics():
    print("\nD4: stall guard does not stop the entry (unbatched breaks out)")
    import re

    inv = StubInv(num_particles=200, num_chains=1)
    # 'sharp' advances beta by ~0.005/stage, 'slow-but-progressing' by ~0.02.
    inv.min_tempering_increment = 0.012
    inv.tempering_stall_patience = 2

    captured = {}
    lines = []
    orig = bx._run_bucket

    def spy(*a, **k):
        out = orig(*a, **k)
        captured.update(out)
        return out

    bx._run_bucket = spy
    try:
        res = bx.run_batched(
            inv,
            [("sharp", make_prep(seed=21, n_pol=16, n_ar=16,
                                 error_pol_val=0.01, log_sigma_val=0.02)),
             ("normal", make_prep(seed=22, n_pol=16, n_ar=16,
                                  error_pol_val=0.5, log_sigma_val=1.0))],
            progress_callback=lines.append,
        )
    finally:
        bx._run_bucket = orig

    stage_lines = [l for l in lines if " stage " in l]
    pat = re.compile(r"stage\s+(\d+): lambda min=([0-9.]+).*finished=(\d+)/(\d+)")
    flagged_stage, flagged_lambda = None, None
    for l in stage_lines:
        m = pat.search(l)
        if m and int(m.group(3)) >= 1 and flagged_stage is None:
            flagged_stage, flagged_lambda = int(m.group(1)), float(m.group(2))
    last = pat.search(stage_lines[-1])
    final_min_lambda = float(last.group(2))

    lam = captured["lambda"]
    stalled = captured["stalled"]
    print(f"  stall guard fired at stage {flagged_stage} with beta={flagged_lambda:.4f} "
          f"(the unbatched loop RETURNS here)")
    print(f"  batch ran on to stage {captured['stages']}; that entry's final "
          f"beta={final_min_lambda:.4f}")
    print(f"  final lambda per entry : {np.array2string(lam, precision=4)}")
    print(f"  stalled flags          : {stalled.tolist()}")
    kept_going = (
        flagged_stage is not None
        and captured["stages"] > flagged_stage
        and final_min_lambda > flagged_lambda + 1e-3
    )
    check("D4 a stalled entry keeps being tempered/rejuvenated after the guard "
          "fires, so its posterior differs from the unbatched result",
          kept_going,
          f"beta {flagged_lambda:.4f} @stage{flagged_stage} -> {final_min_lambda:.4f} "
          f"@stage{captured['stages']}")
    for eid, r in sorted(res.items()):
        print(f"  {eid}: tempering_stalled={r.tempering_stalled}")


def main():
    d1_unbounded_batch()
    d2_bucket_failure_discards_everything()
    d3_progress_callback_exception()
    d4_stall_semantics()
    nfail = sum(1 for _n, ok in OK if not ok)
    print(f"\n{len(OK) - nfail}/{len(OK)} repro assertions confirmed")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
