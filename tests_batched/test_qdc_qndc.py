"""Unit tests for Qdc / Qndc quality primitives (utilities.py)."""
from __future__ import annotations

import numpy as np

import src.utilities as utilities
from src.tape import SDR_TNP
from src.utilities import (
    cluster_lune_params,
    cluster_orientation_params,
    kagan_angle_deg,
    kagan_angles_deg,
    lune_distance_deg,
    mt_quality_scores_dc_ndc,
    orientation_matrices,
)


def test_orientation_matrices_match_sdr_tnp():
    kappa = np.radians(243.0)
    h = np.cos(np.radians(79.0))
    sigma = np.radians(36.0)
    t, n, p = SDR_TNP(kappa, np.arccos(h), sigma)
    u = orientation_matrices(kappa, h, sigma)[0]
    assert np.allclose(u[:, 0], t, atol=1e-12)
    assert np.allclose(u[:, 1], n, atol=1e-12)
    assert np.allclose(u[:, 2], p, atol=1e-12)


def test_kagan_identity():
    k = np.radians(120.0)
    h = np.cos(np.radians(45.0))
    s = np.radians(-30.0)
    assert kagan_angle_deg(k, h, s, k, h, s) < 1e-5


def test_kagan_auxiliary_plane_near_zero():
    """Conjugate / auxiliary plane of a pure DC has Kagan angle ~ 0."""
    from pyrocko import moment_tensor as pmt

    strike, dip, rake = 243.0, 79.0, 36.0
    mt = pmt.MomentTensor(strike=strike, dip=dip, rake=rake)
    s2, d2, r2 = mt.both_strike_dip_rake()[1]
    ang = kagan_angle_deg(
        np.radians(strike),
        np.cos(np.radians(dip)),
        np.radians(rake),
        np.radians(s2),
        np.cos(np.radians(d2)),
        np.radians(r2),
    )
    assert ang < 1e-4


def test_kagan_strike_rotation_90():
    k0 = np.radians(10.0)
    h = np.cos(np.radians(60.0))
    s = np.radians(0.0)
    ang = kagan_angle_deg(k0, h, s, k0 + np.radians(90.0), h, s)
    assert abs(ang - 90.0) < 1e-4


def test_kagan_batch_matches_scalar():
    k_ref = np.radians(40.0)
    h_ref = 0.3
    s_ref = np.radians(20.0)
    k = np.radians([40.0, 50.0, 130.0])
    h = np.array([0.3, 0.35, 0.2])
    s = np.radians([20.0, 10.0, -5.0])
    batch = kagan_angles_deg(k_ref, h_ref, s_ref, k, h, s)
    for i in range(3):
        assert abs(batch[i] - kagan_angle_deg(k_ref, h_ref, s_ref, k[i], h[i], s[i])) < 1e-9


def test_lune_identity():
    assert float(lune_distance_deg(0.1, -0.2, 0.1, -0.2)) < 1e-9


def test_lune_monotonic_in_delta():
    d1 = float(lune_distance_deg(0.0, 0.0, 0.0, 0.1))
    d2 = float(lune_distance_deg(0.0, 0.0, 0.0, 0.3))
    assert d2 > d1 > 0.0


def test_cluster_bimodal():
    rng = np.random.default_rng(0)
    # 4 chains x 250 draws, each with the same ~80/20 orientation split.
    n_chains, n_draw = 4, 250
    n0_c, n1_c = 200, 50
    kaps, hhs, sigs = [], [], []
    for _ in range(n_chains):
        kaps.append(
            np.concatenate(
                [
                    np.radians(243.0) + rng.normal(0.0, np.radians(4.0), n0_c),
                    np.radians(50.0) + rng.normal(0.0, np.radians(4.0), n1_c),
                ]
            )
        )
        hhs.append(
            np.clip(
                np.concatenate(
                    [
                        np.cos(np.radians(79.0)) + rng.normal(0.0, 0.015, n0_c),
                        np.cos(np.radians(86.0)) + rng.normal(0.0, 0.015, n1_c),
                    ]
                ),
                0.0,
                1.0,
            )
        )
        sigs.append(
            np.concatenate(
                [
                    np.radians(36.0) + rng.normal(0.0, np.radians(4.0), n0_c),
                    np.radians(-17.0) + rng.normal(0.0, np.radians(4.0), n1_c),
                ]
            )
        )
    kap = np.stack(kaps, axis=0)
    hh = np.stack(hhs, axis=0)
    sig = np.stack(sigs, axis=0)
    assert kap.shape == (n_chains, n_draw)
    cl = cluster_orientation_params(kap, hh, sig)
    assert cl["n_modes_dc"] == 2
    assert 0.72 <= cl["s_mode_dc"] <= 0.88
    assert cl["dc_mode_weight_std"] < 0.1


def test_cluster_unimodal_collapses():
    rng = np.random.default_rng(1)
    n = 500
    kap = np.radians(100.0) + rng.normal(0.0, np.radians(3.0), n)
    hh = np.clip(0.5 + rng.normal(0.0, 0.02, n), 0.0, 1.0)
    sig = np.radians(10.0) + rng.normal(0.0, np.radians(3.0), n)
    cl = cluster_orientation_params(kap, hh, sig)
    assert cl["n_modes_dc"] == 1
    assert abs(cl["s_mode_dc"] - 1.0) < 1e-12


def test_cluster_trimodal():
    rng = np.random.default_rng(11)
    mechanisms = [(0.0, 60.0, 0.0), (120.0, 45.0, 90.0), (240.0, 80.0, -60.0)]
    for first in range(len(mechanisms)):
        for second in range(first + 1, len(mechanisms)):
            s0, d0, r0 = mechanisms[first]
            s1, d1, r1 = mechanisms[second]
            assert kagan_angle_deg(
                np.radians(s0),
                np.cos(np.radians(d0)),
                np.radians(r0),
                np.radians(s1),
                np.cos(np.radians(d1)),
                np.radians(r1),
            ) > 30.0

    counts = [125, 75, 50]
    chains = []
    for parameter in ("kappa", "h", "sigma"):
        parameter_chains = []
        for _ in range(4):
            pieces = []
            for count, (strike, dip, rake) in zip(counts, mechanisms):
                if parameter == "kappa":
                    values = np.radians(strike) + rng.normal(0.0, np.radians(2.0), count)
                elif parameter == "h":
                    values = np.cos(np.radians(dip)) + rng.normal(0.0, 0.008, count)
                else:
                    values = np.radians(rake) + rng.normal(0.0, np.radians(2.0), count)
                pieces.append(values)
            parameter_chains.append(np.concatenate(pieces))
        chains.append(np.stack(parameter_chains))

    result = cluster_orientation_params(*chains)
    assert result["n_modes_dc"] == 3
    assert np.allclose(
        [mode["weight"] for mode in result["dc_mode_list"]],
        [0.5, 0.3, 0.2],
        atol=0.06,
    )


def test_cluster_conjugate_merge():
    from pyrocko import moment_tensor as pmt

    strike, dip, rake = 243.0, 79.0, 36.0
    auxiliary = pmt.MomentTensor(
        strike=strike,
        dip=dip,
        rake=rake,
    ).both_strike_dip_rake()[1]
    kappa = np.radians([strike] * 200 + [auxiliary[0]] * 200)
    h = np.cos(np.radians([dip] * 200 + [auxiliary[1]] * 200))
    sigma = np.radians([rake] * 200 + [auxiliary[2]] * 200)
    result = cluster_orientation_params(kappa, h, sigma)
    assert result["n_modes_dc"] == 1
    assert result["s_mode_dc"] == 1.0


def test_cluster_deterministic():
    rng = np.random.default_rng(12)
    kappa = rng.uniform(0.0, 2.0 * np.pi, (4, 100))
    h = rng.uniform(0.0, 1.0, (4, 100))
    sigma = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, (4, 100))
    first = cluster_orientation_params(kappa, h, sigma, seed=7)
    second = cluster_orientation_params(kappa, h, sigma, seed=7)
    assert first["n_modes_dc"] == second["n_modes_dc"]
    assert first["dc_mode_list"] == second["dc_mode_list"]
    assert np.array_equal(first["labels"], second["labels"])


def test_cluster_weight_dust_reassigned():
    counts = [500, 405, 95]
    mechanisms = [(248.0, 73.0, 45.0), (49.0, 86.0, -10.0), (43.0, 59.0, 18.0)]
    kappa = np.concatenate(
        [np.full(count, np.radians(sdr[0])) for count, sdr in zip(counts, mechanisms)]
    )
    h = np.concatenate(
        [np.full(count, np.cos(np.radians(sdr[1]))) for count, sdr in zip(counts, mechanisms)]
    )
    sigma = np.concatenate(
        [np.full(count, np.radians(sdr[2])) for count, sdr in zip(counts, mechanisms)]
    )
    responsibilities = np.zeros((sum(counts), 3), dtype=float)
    start = 0
    for component, count in enumerate(counts):
        responsibilities[start : start + count, component] = 1.0
        start += count

    original = utilities._diag_gmm_bic
    utilities._diag_gmm_bic = lambda features, k_max, seed: (
        responsibilities,
        {1: 10.0, 2: 5.0, 3: 0.0},
    )
    try:
        result = cluster_orientation_params(kappa, h, sigma)
    finally:
        utilities._diag_gmm_bic = original

    assert result["n_modes_dc"] == 2
    assert np.allclose(
        [mode["weight"] for mode in result["dc_mode_list"]],
        [500.0 / 905.0, 405.0 / 905.0],
    )
    assert np.count_nonzero(result["labels"] == 0) == 500
    assert np.count_nonzero(result["labels"] == 1) == 500
    assert result["dc_mode1_weight"] != 0.5


def test_cluster_lune_merges_nearby_components():
    rng = np.random.default_rng(13)
    gamma = np.concatenate(
        [
            np.radians(5.0) + rng.normal(0.0, np.radians(1.0), 300),
            np.radians(11.0) + rng.normal(0.0, np.radians(1.0), 300),
        ]
    )
    delta = np.concatenate(
        [
            np.radians(-4.0) + rng.normal(0.0, np.radians(1.0), 300),
            np.radians(1.0) + rng.normal(0.0, np.radians(1.0), 300),
        ]
    )
    result = cluster_lune_params(gamma, delta, mode_min_lune_deg=15.0)
    assert result["n_modes_ndc"] == 1
    assert result["s_mode_ndc"] == 1.0


def test_scorer_shapes_and_ranges():
    rng = np.random.default_rng(2)
    n = 200
    g = 0.2 + rng.normal(0, 0.02, n)
    d = -0.1 + rng.normal(0, 0.03, n)
    k = np.radians(90.0) + rng.normal(0, np.radians(5), n)
    h = np.clip(0.4 + rng.normal(0, 0.02, n), 0, 1)
    s = rng.normal(0, np.radians(5), n)
    sc = mt_quality_scores_dc_ndc(
        gamma=g, delta=d, kappa=k, h=h, sigma=s, ref_idx=0, idata=None
    )
    for key in ("Qdc", "Qndc", "s_prec_dc", "s_mode_dc", "s_conv_dc", "s_prec_ndc", "s_conv_ndc"):
        assert key in sc
        assert 0.0 <= sc[key] <= 1.0 + 1e-9
    # missing idata -> neutral conv 0.5
    assert abs(sc["s_conv_dc"] - 0.5) < 1e-12
    assert abs(sc["s_conv_ndc"] - 0.5) < 1e-12
    assert abs(
        sc["Qdc"] - sc["s_mode_dc"] * (sc["s_prec_dc"] + sc["s_conv_dc"]) / 2.0
    ) < 1e-12
    assert abs(
        sc["Qndc"] - sc["s_mode_ndc"] * (sc["s_prec_ndc"] + sc["s_conv_ndc"]) / 2.0
    ) < 1e-12


def test_low_ess_warning():
    import arviz as az
    import warnings

    rng = np.random.default_rng(14)
    shape = (4, 400)

    def ar1(center, scale):
        values = np.empty(shape, dtype=float)
        values[:, 0] = center
        for draw in range(1, shape[1]):
            values[:, draw] = (
                center
                + 0.999 * (values[:, draw - 1] - center)
                + rng.normal(0.0, scale, shape[0])
            )
        return values

    gamma = ar1(0.1, 0.002)
    delta = ar1(-0.1, 0.002)
    kappa = ar1(1.0, 0.005)
    h = np.clip(ar1(0.4, 0.001), 0.0, 1.0)
    sigma = ar1(0.2, 0.005)
    idata = az.from_dict(
        posterior={
            "gamma": gamma,
            "delta": delta,
            "kappa": kappa,
            "h": h,
            "sigma": sigma,
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mt_quality_scores_dc_ndc(
            gamma,
            delta,
            kappa,
            h,
            sigma,
            idata=idata,
        )
    messages = [str(item.message) for item in caught if item.category is RuntimeWarning]
    assert any("orientation-block ess_min" in message for message in messages)
    assert any("source-type block ess_min" in message for message in messages)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mt_quality_scores_dc_ndc(gamma, delta, kappa, h, sigma, idata=None)
    assert not [item for item in caught if item.category is RuntimeWarning]


def test_scorer_empty_inputs():
    """Empty samples must return the NaN dict, not crash in ref resolution."""
    empty = np.array([])
    sc = mt_quality_scores_dc_ndc(
        gamma=empty, delta=empty, kappa=empty, h=empty, sigma=empty
    )
    assert np.isnan(sc["Qdc"])
    assert np.isnan(sc["Qndc"])
    assert np.isnan(sc["s_mode_dc"])


def test_dc_constraint_nan_qndc():
    n = 50
    sc = mt_quality_scores_dc_ndc(
        gamma=np.zeros(n),
        delta=np.zeros(n),
        kappa=np.linspace(0, 1, n),
        h=np.full(n, 0.5),
        sigma=np.zeros(n),
        ref_idx=0,
        dc=True,
    )
    assert np.isnan(sc["Qndc"])
    assert np.isnan(sc["s_prec_ndc"])
    assert np.isfinite(sc["Qdc"])


def test_vertical_dip_antipodal_split_not_spurious():
    """A tight posterior straddling dip=90 must not collapse s_prec_dc.

    (kappa, h, sigma) and (kappa+pi, -h, -sigma) are the same double couple,
    so the sampler's dip<=90 domain splits a near-vertical posterior into two
    antipodal strike clusters. The embedding-mean representative of the
    merged mode lands between the clusters on a mechanism no sample is close
    to, inflating the within-mode Kagan quantiles (regression: CAPE catalog
    events with q50 ~ 58 deg and s_prec_dc = 0 for ~8 deg posteriors).
    """
    rng = np.random.default_rng(7)
    n = 2000
    kappa_true = np.radians(75.0) + rng.normal(0.0, np.radians(2.0), n)
    h_true = rng.uniform(-0.06, 0.06, n)  # dip straddles 90 deg
    sigma_true = np.radians(10.0) + rng.normal(0.0, np.radians(3.0), n)
    # fold into the sampler's dip<=90 domain
    fold = h_true < 0.0
    kappa = np.where(fold, np.mod(kappa_true + np.pi, 2.0 * np.pi), kappa_true)
    h = np.abs(h_true)
    sigma = np.where(fold, -sigma_true, sigma_true)

    fixed = cluster_orientation_params(kappa, h, sigma)
    assert fixed["n_modes_dc"] == 1
    assert fixed["dc_q50_within_mode_deg"] < 10.0
    # the representative is a member mechanism, close to the true plane
    ref = kagan_angle_deg(
        np.radians(fixed["dc_mode0_strike_deg"]),
        np.cos(np.radians(fixed["dc_mode0_dip_deg"])),
        np.radians(fixed["dc_mode0_rake_deg"]),
        np.radians(75.0),
        0.0,
        np.radians(10.0),
    )
    assert ref < 10.0

    legacy = cluster_orientation_params(
        kappa, h, sigma, canonicalize=False, mode_representative="mean"
    )
    assert legacy["dc_q50_within_mode_deg"] > 30.0  # the documented bug


if __name__ == "__main__":
    import traceback

    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    _failed = 0
    for _fn in _tests:
        try:
            _fn()
            print(f"PASS {_fn.__name__}")
        except Exception as _exc:
            _failed += 1
            print(f"FAIL {_fn.__name__}: {_exc}")
            traceback.print_exc()
    print(f"\n{len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(_failed)
