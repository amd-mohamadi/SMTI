# M1 status — batched-event SMC on GPU

*Branch `m1-batched-gpu` in the worktree `/s0/data/CAPE/smti_workflow/SMTI_dev`.
Nothing is committed; `/s0/data/CAPE/smti_workflow/SMTI` (production) is
untouched. Written 2026-07-27 after the stage-5 fix + gate pass.*

Implements sections 3.1–3.6 of `smti_inversion/202607_gpu_batched_smc_plan.md`.

---

## 1. What M1 delivers

`run_batched(inv, [(event_id, event), ...])` inverts many events at once as a
handful of vmapped XLA programs and returns `{event_id: InversionResult}` whose
structure is byte-for-byte what `forward()` returns today, so
`4_run_inversion.py`'s post-processing (ArviZ, beachballs, Q metrics, CSV)
needs no change.

* **One program per shape bucket.** Events are grouped by
  `(ceil(N_pol/8)*8, ceil(N_ar/8)*8, N_loc, has_pol, has_ar)`; inside a bucket
  every event is padded to the bucket shape and masked, so a bucket is a single
  `jax.vmap`ed SMC over `B = events x chains` independent entries. Chains stop
  being a special case — 4 events x 4 chains is just `B = 16`.
* **Batched == unbatched.** Entry `b` uses the same seed, the same key stream,
  the same station-angle realisation and the same math as its unbatched twin,
  and is *frozen* the moment it finishes, so it agrees with
  `_invert_single_event` particle-for-particle (measured: 5e-18 … 2e-13 on CPU,
  5.7e-14 on the A40 in fp64).
* **fp32 support** (`smc_dtype='float32'`) via a dtype-agnostic tape
  (`src/tape_jax_dtype.py`) and dtype-agnostic proposal scales — the decisive
  lever from M0 (9.3x/stage). Measured in stage 3: 107.3 ms/stage at B=16,
  P=2000 vs M0 spike4's 107.47 (1.00x — the productionised code costs nothing).
* **Bounded, restartable, isolated:** `max_batch` caps the width of one XLA
  program (chunks share one compiled program, the last chunk is repeat-padded
  and its pad rows discarded); jitted programs are cached across calls so a
  second `run_batched` with the same configuration compiles nothing; a bucket
  that raises does not destroy the buckets that already finished; a stalled or
  non-finite entry does not hold or poison the rest of its batch.
* **Scope guard.** `persistent` SMC, NUTS, `adapt_proposal=False`,
  `amp_ratio_noise_mode='station_smooth'` and `dc=True` raise
  `NotImplementedError` from `prepare_batch` before anything is traced; the
  caller falls back to the unbatched path.

---

## 2. Files changed

### New

| file | lines | what |
|------|-------|------|
| `src/inversion_blackjax_batched.py` | 1–1203 | the whole batched path |
| `src/tape_jax_dtype.py` | 1–111 | `jax_Tape_MT6` with `dtype = jnp.result_type(inputs)` on every literal; no import-time x64 side effect |
| `tests_batched/` | — | test suite (below) |

Map of `src/inversion_blackjax_batched.py`:

| lines | contents |
|-------|----------|
| 1–58 | module docstring: scope, fp32 process recipe, **process-global x64 hazard** |
| 102–106 | `PAD_MULTIPLE = 8`, particle field order |
| 112–120 | `BatchEntry(event_id, event_index, chain_index, seed)` |
| 122–175 | `Bucket` (+ `n_active` for repeat-padded chunks) |
| 178–208 | `_check_supported` — the five `NotImplementedError` guards |
| 211–285 | `_round_up`, `_pad_event_arrays` (pad values chosen so padded rows are finite garbage) |
| 288–407 | `prepare_batch` — station angles, `_prepare_event_arrays`, padding, bucketing, duplicate-id check, `forward()`-identical chain seeds |
| 410–437 | `_make_unconstrained_to_params` |
| 440–576 | `_make_model` — per-entry logprior/loglikelihood, `sum(x)` → `sum(mask*x)` |
| 579–611 | `_make_init_particle` (production `softplus_inv`) |
| 614–671 | `_build_smc` — MWG + `inner_kernel_tuning`, per-entry **traced** init stds |
| 682–709 | `_initial_particles` — key stream identical to `_invert_single_event` |
| 712–764 | **program cache** (`_PROGRAM_CACHE`, `clear_program_cache`, `_batched_programs`) |
| 767–785 | **`_freeze_entries`** — per-entry `where(done, old, new)` over the SMC state |
| 788–938 | `_run_bucket` — host loop, freeze-on-finish, per-entry stall + failure handling, progress |
| 941–979 | `_entry_result` |
| 982–1014 | `_chunk_bucket` — fixed-`B` chunking with repeat padding |
| 1017–1037 | `_safe_callback` |
| 1040–1174 | `run_batched` — dtype guards, chunking, per-bucket failure isolation |
| 1177–1203 | `_assemble` — regroup by `(event_index, event_id)`, chain-count assertion |

### Modified (production file, all behaviour-preserving or additive)

| file:lines | change |
|------------|--------|
| `src/blockwise_rmh.py:86–90` | `2.38/sqrt(d)` evaluated in Python (weakly typed) instead of `jnp.float64` — fp32-safe, bit-identical in fp64 |
| `src/inversion_blackjax.py:23` | `import warnings` |
| `src/inversion_blackjax.py:62–105` | `softplus_inv` — overflow-free (M0 blocker for fp32); docstring now states the ~0.2% / 1-ulp non-reproducibility |
| `src/inversion_blackjax.py:201` | `InversionResult.tempering_stalled: bool = False` |
| `src/inversion_blackjax.py:439`, `474` | `tempering_stalled` carried through `_serialize_chain_result` / `_deserialize_chain_result` (process chain execution) |
| `src/inversion_blackjax.py:719` | `_invert_multi_chain` ends in `return self._stack_chain_results(...)` |
| `src/inversion_blackjax.py:721–824` | `_stack_chain_results` — verbatim move of the stacking tail, reused by the batched path |
| `src/inversion_blackjax.py:905–1015` | `_prepare_event_arrays` — pure extraction from `_invert_single_event` (byte-identical block; verified by `test_refactor_equivalence.py`) |
| `src/inversion_blackjax.py:1047–1064` | warn when `_invert_single_event` runs with `jax_enable_x64=False` (process-global fp32 hazard) |
| `src/inversion_blackjax.py:1651`, `1720`, `1803` | `_invert_single_event` now reports `tempering_stalled` |

### Tests

`tests_batched/test_foundations.py` (stage 1), `test_refactor_equivalence.py`,
`test_batched_core.py` (stage 2 + stage 5 additions), `validate_gpu.py`
(stage 3), plus helpers `event_loading.py`, `survey_events.py`.
`tests_batched/review_repros/` holds the two reviewers' repro scripts (kept for
reference; the behaviours they demonstrated are now covered by checks 2h, 6a–6e,
7a–7d and 8a–8b of `test_batched_core.py`).

---

## 3. Test matrix (stage-5 re-run, 2026-07-27)

### CPU (`JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run -n smti python ...`)

| suite | checks | result |
|-------|--------|--------|
| `tests_batched/test_foundations.py` | 28 | **28/28 PASS** |
| `tests_batched/test_batched_core.py` (fp64 process) | 36 | **36/36 PASS** |
| `tests_batched/test_batched_core.py --fp32` (auto-spawned subprocess, x64 off) | 8 | **8/8 PASS** |

Highlights:

| check | result |
|-------|--------|
| 0a/0b tape dtype-agnostic | bit-identical in fp64, float32 in/out |
| 1b masked padded loglik vs production | max rel 3.3e-16 (accumulator now NaN-propagating) |
| 2a–2g bucketing + scope guards | pass |
| 2h duplicate `event_id` | raises `ValueError` |
| 2i `num_chains=1` seed | `20240216` = `random_seed`, matches `_invert_single_event` |
| 2j/2k `max_batch` chunking | B=6 → 2 chunks of 4, `active=[4,2]` |
| 3a–3f mini end-to-end (2 *different* events x 2 chains) | one bucket, distinct posteriors, structure = `_invert_multi_chain` |
| 6a–6e stall freeze | mixed batch (chains 0,1 stall at λ=0.013; 2,3 reach 1); every chain matches its unbatched twin to ≤ 1.6e-13; flag ⇔ λ<1 |
| 7a failure isolation | failed bucket reported, surviving bucket returned |
| 7b `max_batch=1` vs `B=2` | max abs diff 1.6e-14 |
| 7d raising `progress_callback` | run completes |
| 8a program cache | 49 XLA compilations on call 1, **0 on call 2** |

### GPU (A40, verified idle before the run: 0 MiB, no foreign compute apps)

`XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
conda run -n smti python tests_batched/validate_gpu.py --stage fp32 --quick`

**9/9 PASS** (`tests_batched/gpu_results_stage5/fp32.json`):

| check | result |
|-------|--------|
| 0. x64 disabled in the fp32 process | PASS |
| 2a/2b eq00124 x 4 chains fp32, P=2000 | PASS — particles float32, all 4 entries λ=1 in 34 stages (same schedule as stage 3), 28.9 ms/stage |
| — regression values | gamma **+0.269** (production 0.247 ± 0.06), delta **−0.178** (−0.145 ± 0.08), dominant kappa-mode weight **0.889** (band 0.68–0.92) |
| 3c-1/3c-2 eq00126 + eq00124 x 4 chains, 2 buckets | PASS — every event present, all finite, none stalled |
| 3d-1 wild garbage in every masked-out row | PASS — element-wise difference **exactly 0.0** for both events |
| 3d-2/3d-3 re-pad to a multiple of 32 | PASS — loglik moves only at fp32 round-off (rel ≤ 2.0e-7), posterior within MC noise (Δgamma ≤ 0.005) |
| 4a perf snapshot | PASS — single B=8 bucket, 54.8 ms/stage (≈ M0 spike4's 107.5 ms at B=16, i.e. the same per-entry rate), peak 328 MB device memory |

The program cache is visible in this log: the second and third `run_batched`
calls that reuse the `(48,64) B=4` program report `first stage 0.0s (incl. step
compile)` and finish the whole 34-stage run in 0.98 s, against 19.4 s for the
first call — ~19 s of XLA compile eliminated per repeated call.

Stage-3's full GPU matrix (25/25, incl. the fp64 batched==unbatched bar and the
B=16 perf snapshot) is in `tests_batched/gpu_results/`; the stage-5 gate above
re-ran the fp32 half after the fixes. The A40 was verified idle (0 MiB, no
foreign compute apps) before the run and is idle again after it.

---

## 4. Review findings — fixed / skipped

Two independent reviewers filed 18 findings (6 + 12, several overlapping).

### Fixed

| # | severity | finding | fix |
|---|----------|---------|-----|
| 1 | **major** (both reviewers) | stall guard only exempted an entry from the loop-exit test; a stalled entry kept being tempered to λ=1 and still reported `tempering_stalled` | `_freeze_entries` (batched:767–785) reverts finished/stalled/failed rows to their previous state every stage, so an entry stops evolving the moment it finishes. `tempering_stalled` is now derived from the **final λ** (batched:908–912), not latched. Verified by checks 6a–6e: the two stalled chains match their unbatched twins to 4.9e-15 after sitting out 29 further stages. Side benefit: entries that reach λ=1 early are now bit-identical to their unbatched twins regardless of bucket composition. |
| 2 | **major** (both) | duplicate `event_id` merged two events' chains into one result with a 2x chain axis while `num_chains` still reported the configured value | `prepare_batch` rejects duplicate ids with `ValueError` (batched:329–338); `_assemble` regroups by `(event_index, event_id)` and refuses to stack a group whose chain count ≠ `inv.num_chains` (batched:1177–1203). Check 2h. |
| 3 | **major** (integration) | every `run_batched` call rebuilt the jit closures ⇒ full XLA recompile (~21 s/program on the A40) even for identical shapes | module-level `_PROGRAM_CACHE` keyed on the closure's configuration (batched:712–764) + `clear_program_cache()`. Check 8a: 49 compilations on call 1, 0 on call 2. |
| 4 | **major** (integration) | bucket width `B` unbounded (`n_events x n_chains`) with no chunking knob ⇒ a 60-event bucket becomes one B=240 program and OOMs | `run_batched(max_batch=...)` + `_chunk_bucket` (batched:982–1014): fixed-width chunks, last chunk repeat-padded so all chunks share one compiled program, pad rows discarded. Checks 2j/2k/7b. |
| 5 | **major** (integration) | no failure isolation: one bucket's exception discarded every finished bucket | per-bucket `try/except` with `on_bucket_error='skip'` (default) or `'raise'` (attaches `exc.partial_results` / `exc.failures`), plus `return_failures=True` for the `{event_id: reason}` map (batched:1129–1145, 1167–1174). Check 7a. |
| 6 | minor | a single non-finite λ aborted the whole bucket | that entry is frozen at its last good state, marked `failed`, excluded from the results and reported in `failures`; the rest of the batch finishes (batched:870–885). |
| 7 | minor | `num_chains == 1` used a `SeedSequence`-derived seed while `forward()` uses `PRNGKey(random_seed)` | `prepare_batch` now mirrors `forward()`'s branch (batched:339–345). Check 2i. |
| 8 | minor | `smc_dtype='float64'` silently demoted to float32 in an x64-disabled process, while the only warning fired on the harmless opposite case | `run_batched` raises `RuntimeError` for `float64` without x64, downgrades the float32+x64 message to a NOTE, and warns if the realised particle dtype ≠ the requested one (batched:1099–1108, 1146–1151). Check 7c. |
| 9 | minor | an exception from `progress_callback` aborted the in-flight bucket | `_safe_callback` (batched:1017–1037). Check 7d. |
| 10 | minor | `tempering_stalled` was dropped by `_serialize_chain_result`/`_deserialize_chain_result` and never set on the unbatched path | both serialisers carry it (`inversion_blackjax.py:439`, `474`) and `_invert_single_event` sets it (`1651`, `1720`, `1803`). Check 6e. |
| 11 | minor | process-global fp32 recipe silently downgrades the unbatched fallback | `_invert_single_event` warns when x64 is off (`inversion_blackjax.py:1047–1064`) and the batched module docstring now carries a `.. warning::` recommending a dedicated fp32 subprocess. |
| 12 | minor | `softplus_inv` docstring implied bit-reproducibility | docstring now states the ~0.2% / 1-ulp shift and that archived references need re-baselining (`inversion_blackjax.py:88–101`). |
| 13 | minor (test) | test 1b's `max(worst, rel)` accumulator could not fail on NaN | NaN-propagating accumulator + explicit `isfinite(got)` (`test_batched_core.py:288–298`). |
| 14 | minor (test) | the mini end-to-end passed the *same* event under two ids; dead `bad` variable | two differently truncated events in one bucket + a new check that their posteriors differ (3f); dead variable deleted. |

### Skipped (with reasons)

| finding | why not fixed |
|---------|----------------|
| per-event data replicated `num_chains` times in the stacked arrays (4x device-resident data) | real but not a defect: `vmap` needs a true batch axis, and with `max_batch` the device footprint is now bounded by design. Revisit only if profiling shows the data copies matter (they are ~10 MB at production shapes vs GB-scale likelihood intermediates). |
| no direct unit test of the `station_smooth_ar` branch inside `prepare_batch` (unreachable because `_check_supported`'s `amp_ratio_noise_mode` check fires first) | dead-by-construction defence-in-depth; adding a test would require faking `_prepare_event_arrays`' output. Left as is, noted here. |
| an event whose likelihood is NaN silently returns prior samples with no diagnostic | pre-existing behaviour of the **unbatched** production path; changing it is out of M1 scope and would alter production results. |

---

## 5. Known limitations

1. **Scope**: `adaptive_tempered` + MWG-RMH + `adapt_proposal=True` +
   `amp_ratio_noise_mode='global'` + `dc=False` only. Everything else must use
   the unbatched path (guards raise `NotImplementedError`).
2. **fp32 needs a process-global switch.** `jax_enable_x64` is per-process, so a
   process that runs the batched sampler in fp32 cannot also produce trustworthy
   fp64 unbatched results. `run_batched(smc_dtype='float64')` now refuses in
   such a process and `_invert_single_event` warns — but the clean answer is a
   dedicated subprocess (see M2 below).
3. **Finished entries still consume compute.** Freezing is a *result* mask, not
   a work mask: the vmapped step still evaluates a finished entry, its output is
   just discarded. Plan section 6 ("efficiency masking of finished entries")
   remains open; the cost is bounded by the spread of stage counts inside a
   bucket (34 vs 30 stages on eq00124-like events).
4. **`max_batch` defaults to `None`** (one program per bucket, however wide).
   Drivers must pass `max_batch = gpu_batch_events * inv.num_chains`.
5. **Wider padding is not bit-neutral in fp32.** Masking is exact (garbage in
   padded rows changes nothing, 0.0e+00), but a different pad width regroups the
   XLA reduction tree, which fp32 round-off then amplifies chaotically through
   accept/reject. Results stay within MC noise (< 0.03 on gamma/delta); fp64 is
   element-wise stable. Consequence: **the pad multiple is part of the
   reproducibility contract of an fp32 run** — do not change `PAD_MULTIPLE`
   between a run and its reference.
6. **Not bit-reproducible against pre-M1 archives** because of the
   `softplus_inv` fix (~0.2% of initial particles shift by 1 ulp).
7. `InversionResult` has no per-entry final λ field; a stalled event is
   identifiable only through `tempering_stalled` (which now, at least, describes
   the samples that were actually returned).

---

## 6. Remaining steps for M2 (driver integration, `4_run_inversion.py`)

1. **Process layout — decide first.** `jax_enable_x64` is global and fp32 is
   what makes the GPU worth using (9.3x/stage), so the recommended layout is:
   * the driver process stays x64-enabled and does all CPU work
     (loading, filtering, post-processing, plots, Q, CSV);
   * a **dedicated fp32 GPU worker subprocess** imports
     `src.inversion_blackjax`, immediately does
     `jax.config.update("jax_enable_x64", False)`, then serves buckets;
   * events that hit the scope guards fall back to `_invert_single_event` **in
     the driver process** (fp64, correct), never inside the fp32 worker.

   `tests_batched/test_batched_core.py` (`--fp32` re-exec) and
   `validate_gpu.py --stage fp32` are working templates for the subprocess.
   The single-process alternative (flip x64 off once, batch everything) is only
   acceptable if no event ever needs the fallback — the new guard/warning makes
   the violation loud rather than silent, but it is still a footgun.
2. **Wire the new mode**: `--worker-device gpu` ⇒ single GPU worker, bucket the
   event list, call
   `run_batched(inv, events, smc_dtype='float32',
   max_batch=GPU_BATCH_EVENTS * NUM_CHAINS, return_failures=True,
   progress_callback=<log sink>)`. Add `--gpu-batch` (default 4 events x 4
   chains = 16). Keep `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
   `MEM_FRACTION≈0.85` (`4_run_inversion.py:44–45`).
3. **Feed events in shape order** (sort by `(N_pol, N_ar)`) so chunks of a
   bucket are contiguous and the program cache is hit; with the cache in place,
   compile cost is paid once per *distinct shape*, not per call.
4. **Handle `failures`**: events in the returned failure map must be recorded as
   `{'ok': False, 'error': ...}` exactly like today's per-event exception path
   (`4_run_inversion.py:1395`) and, optionally, retried once on the unbatched
   CPU path.
5. **Overlap post-processing with sampling**: submit bucket *i*'s
   post-processing to the existing bounded pool while bucket *i+1* samples. At
   fp32 GPU speeds post-processing is the end-to-end bottleneck (M0 note), so
   this is where the wall-clock win is won or lost.
6. **Propagate `tempering_stalled`** into `mt_results.csv` / the QC filter — it
   is now meaningful on both paths and is the cheapest available "this posterior
   is not converged" flag.
7. **Memory sanity check before the first full run**: at P=2000, padded
   `N_ar=64`, `N_loc=20`, fp32, the live likelihood intermediates are
   ~10 MB x B per tensor; B=16 is comfortable on a 48 GB A40 shared with a
   colleague, B=240 is not. Confirm with `nvidia-smi` during the first bucket
   and keep the GPU-occupancy guard (`validate_gpu.gpu_guard`) in the driver.
8. **M3 validation** then runs against this driver: eq00124 regression band,
   a 12-event batch vs 12 CPU runs, fp32 vs fp64, and the events/hour table.
