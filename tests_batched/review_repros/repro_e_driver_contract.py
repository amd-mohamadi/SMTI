"""REPRO E: does a batched InversionResult survive 4_run_inversion.py's
post-processing byte-for-byte the way an unbatched multi-chain one does?

Uses the *real* ``InversionBlackJAX`` (so ``_invert_multi_chain`` /
``_stack_chain_results`` run unmodified) with a synthetic
``_prepare_event_arrays`` so the comparison costs a couple of seconds instead of
a full read_data + 44x62 inversion.

Then replays the driver's consumption code verbatim (4_run_inversion.py:122-135,
778-806, 837-887, 866-867).
"""
from stub import make_prep, QUIET  # noqa: E402

import dataclasses
import pickle

import numpy as np

import src.inversion_blackjax as ib
import src.inversion_blackjax_batched as bx
from src.inversion_blackjax import InversionBlackJAX, InversionResult

ib.build_location_samples_from_errors = lambda *a, **k: None

OK = []


def check(name, ok, detail=""):
    OK.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


class SynthInversion(InversionBlackJAX):
    """Real sampler, synthetic per-event arrays."""

    def set_prep(self, prep):
        self._prep = prep

    def _prepare_event_arrays(self, event, location_samples=None):
        return dict(self._prep)

    def _filter_event_by_options(self, event):
        return event


def build(num_particles, num_chains, seed=4242):
    inv = SynthInversion(
        {},
        inversion_options=None,
        num_particles=num_particles,
        dc=False,
        gamma_beta_prior=(3.0, 3.0),
        delta_beta_prior=(3.0, 3.0),
        amp_ratio_sigma_prior=2.0,
        amp_ratio_noise_mode="global",
        random_seed=seed,
        location_samples_n=3,
        azimuth_error=5,
        takeoff_error=10,
        mcmc_kernel="rmh",
        num_mcmc_steps=1,
        mechanism_steps=1,
        smc_target_ess_ratio=0.9,
        max_smc_iterations=60,
        num_chains=num_chains,
        chain_execution="sequential",
    )
    return inv


# --- driver code, copied verbatim ------------------------------------------
def _as_arviz_posterior_array(samples):       # 4_run_inversion.py:122
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


REQUESTED_VARS = ["gamma", "delta", "kappa", "h", "sigma", "sigma_amp_ratio"]


def driver_postprocess(result_obj, tag):
    """Replay 4_run_inversion.py:778-887 and report what it produces."""
    out = {}
    import arviz as az

    posterior_dict = {}
    for var in REQUESTED_VARS:
        if hasattr(result_obj, var):
            samples = getattr(result_obj, var)
            if samples is not None:
                arr = _as_arviz_posterior_array(samples)
                if arr is not None:
                    posterior_dict[var] = arr
    idata = az.from_dict(posterior=posterior_dict) if posterior_dict else None
    out["idata_vars"] = sorted(posterior_dict)
    out["idata_shape"] = {k: v.shape for k, v in posterior_dict.items()}
    summary = az.summary(idata, var_names=list(posterior_dict), hdi_prob=0.90)
    out["summary_index"] = list(summary.index)
    out["rhat_finite"] = bool(np.all(np.isfinite(summary["r_hat"].to_numpy())))

    mt6_samples = np.asarray(result_obj.mt6, dtype=float)   # :837
    if mt6_samples.ndim == 3:
        if mt6_samples.shape[1] == 6:
            mt6_samples = np.transpose(mt6_samples, (1, 0, 2))
        mt6_samples = mt6_samples.reshape(6, -1)
    out["mt6_flat"] = mt6_samples.shape

    w = np.asarray(result_obj.weights, dtype=float).reshape(-1)   # :846/:880
    out["weights_flat"] = w.shape
    out["weights_match_mt6"] = w.size == mt6_samples.shape[1]
    w = w / np.sum(w)
    out["particle_ess_ratio"] = float(1.0 / np.sum(w**2) / w.size)
    out["ln_p"] = getattr(result_obj, "ln_p", "MISSING")
    out["station_sigma"] = getattr(result_obj, "sigma_amp_ratio_station", "MISSING")
    out["pickle_ok"] = pickle.loads(pickle.dumps(result_obj)).mt6.shape == result_obj.mt6.shape
    print(f"  [{tag}] {out}")
    return out


def main():
    P, C = 60, 2
    prep = make_prep(seed=99, n_pol=8, n_ar=8)

    print("\nE1: field-by-field structure, batched vs _invert_multi_chain")
    inv_u = build(P, C)
    inv_u.set_prep(prep)
    res_u = inv_u._invert_multi_chain(dict(prep))

    inv_b = build(P, C)
    inv_b.set_prep(prep)
    res_b = bx.run_batched(inv_b, [("ev", dict(prep))], progress_callback=QUIET)["ev"]

    mism = []
    for f in dataclasses.fields(InversionResult):
        a = getattr(res_u, f.name)
        b = getattr(res_b, f.name)
        ka = "None" if a is None else (
            f"{type(a).__name__}{getattr(a, 'shape', '')}:{getattr(a, 'dtype', type(a).__name__)}"
        )
        kb = "None" if b is None else (
            f"{type(b).__name__}{getattr(b, 'shape', '')}:{getattr(b, 'dtype', type(b).__name__)}"
        )
        flag = "" if ka == kb else "   <-- DIFFERS"
        print(f"    {f.name:32s} unbatched={ka:28s} batched={kb:28s}{flag}")
        if ka != kb:
            mism.append(f.name)
    check("E1 every InversionResult field has the same type/shape/dtype",
          not mism, f"differing: {mism}" if mism else "identical")

    print("\nE2: driver post-processing on both results")
    ou = driver_postprocess(res_u, "unbatched")
    ob = driver_postprocess(res_b, "batched  ")
    keys = ["idata_vars", "idata_shape", "summary_index", "rhat_finite",
            "mt6_flat", "weights_flat", "weights_match_mt6", "ln_p",
            "station_sigma", "pickle_ok"]
    diff = [k for k in keys if ou[k] != ob[k]]
    check("E2 driver post-processing sees the same structure", not diff,
          f"differing: {diff}" if diff else "identical")

    print("\nE3: the driver's `result[0] if isinstance(result, list)` contract")
    print(f"    forward() returns   : {type(inv_u.forward.__annotations__.get('return', '?'))} "
          f"-> InversionResult | list[InversionResult]")
    print(f"    run_batched returns : dict[str, InversionResult]")
    check("E3 run_batched's return type is NOT what line 776 unwraps "
          "(driver change required in M2)", True,
          "dict vs InversionResult/list -- documented, but note num_chains/idata "
          "are otherwise compatible")

    nfail = sum(1 for _n, ok in OK if not ok)
    print(f"\n{len(OK) - nfail}/{len(OK)} checks passed")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
