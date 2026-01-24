#!/usr/bin/env
import os
import sys
sys.path.append("./SMCMTI")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# Import inversion
from src.inversion_pytensor import InversionPyTensor as Inversion
from src.moment_tesnor_conversion import (
    MT6_MT33,
    MT33_MT6,
    MT6_TNPE,
    E_tk,
    tk_uv,
    toa_vec,
    TNP_SDR,
    TP_FP,
    FP_SDR,
    SDR_SDR,
)
from src.utilities import (
    find_map_mt6_gmm,
    angular_uncertainty,
    angle_mode_analysis,
    mt_quality_score,
    generate_synthetic_event_data,
    project_to_dc_mt6,
    load_mtfit_reference_mt6,
)
from src.forward_model import forward_amplitude
from src.plot.plot_classes import (
    _AmplitudePlot, 
    _FaultPlanePlot,
    plot_kaverina_with_hdi,
    kaverina_hdi_mask,
    plot_hudson_with_hdi
)
from src.plot.spherical_projection import equal_area
from pyrocko import moment_tensor as pmt
from pyrocko.plot import beachball
from scipy import stats

# Get Data (ndarray-based copy)
from example_data import synthetic_event,forge_event




def run(
    draws: int = 2000,
    chains: int = 4,
    cores: int | None = None,
    enable_file_logging: bool = True,
    dc: bool = False,
    inversion_options: list[str] | None = None
    ) -> None:

    
    out_path = os.path.join(os.getcwd(), "synthetic_test")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    
    # ==============================================================================
    # GOOD COVERAGE
    # ==============================================================================

    good_azimuth = np.array([
        15, 45, 75, 110, 145, 175, 210, 240, 270, 300, 330, 355
    ], dtype=float)

    good_takeoff = np.array([
        35, 65, 25, 45, 80, 120, 140, 75, 50, 95, 110, 155
    ], dtype=float)
    
    # ==============================================================================
    # MODERATE COVERAGE
    # ==============================================================================

    moderate_azimuth = np.array([
        20, 65, 110, 160, 200, 250, 320
    ], dtype=float)

    moderate_takeoff = np.array([
        55, 75, 40, 95, 125, 65, 80
    ], dtype=float)
    
    # ==============================================================================
    # LOW COVERAGE
    # ==============================================================================

    low_azimuth = np.array([
        10, 45, 85, 115, 130
    ], dtype=float)

    low_takeoff = np.array([
        70, 65, 80, 55, 90
    ], dtype=float)

    
    #==============================================================================
    # Select which coverage scenario to use
    #==============================================================================
    
    azimuth = good_azimuth
    takeoff = good_takeoff
    # azimuth = moderate_azimuth
    # takeoff = moderate_takeoff
    # azimuth = low_azimuth
    # takeoff = low_takeoff


    inversion_options = [
        'PPolarity',
        # 'SHPolarity',
        # 'SVPolarity',
        'P/SHAmplitudeRatio',
        'P/SVAmplitudeRatio',
        # 'SH/SVAmplitudeRatio'
    ]
    true_mt6 = np.array([-0.10833004, 0.03885163, 0.06936917, 0.00594627, 0.86124002, 0.48919666], dtype=float)
    
    amp_bias_P = 0.7
    amp_bias_SH = 1.0
    amp_bias_SV = 0.7

    data = generate_synthetic_event_data(
        inversion_options=inversion_options,
        true_mt6=true_mt6,
        azimuth=azimuth,
        takeoff=takeoff,
        azimuth_err=2.0,
        takeoff_error=3.0,
        polarity_error_range=(0.05, 0.1),
        amplitude_error_range=(0.1, 0.3),
        amplitude_bias_lognormal_sigma=0.5,
        amplitude_outlier_fraction=0.5,
        amplitude_outlier_scale=4.0,
        amplitude_bias_P = amp_bias_P,
        amplitude_bias_SH = amp_bias_SH,
        amplitude_bias_SV = amp_bias_SV,
        uid="moderate_",
        rng=np.random.default_rng(123),
    )
    
    data['UID'] += '_'.join(inversion_options)
    data['UID'] += "_dc" if dc else "_mt"
    data['UID'] = data['UID'].replace('/', '')
    print(f"Data UID: {data['UID']}")
    test_output_dir = os.path.join(out_path, data['UID'])
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    # Setup logging to file and console
    if enable_file_logging:
        class Logger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, 'w')

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)

            def flush(self):
                self.terminal.flush()
                self.log.flush()

            def close(self):
                self.log.close()

        log_path = f"{test_output_dir}/run.log"
        logger = Logger(log_path)
        sys.stdout = logger
        sys.stderr = logger
        print(f"Logging output to {log_path}")


    inversion_object = Inversion(
        data,
        inversion_options=inversion_options,
        dc=dc,
        draws=draws,
        chains=chains,
        cores=cores,
        random_seed=None,
        compute_convergence_checks=False,
        gamma_beta_prior=(10.0, 10.0),   # custom prior for gamma
        delta_beta_prior=(10.0, 10.0),   # custom prior for delta
        smc_kernel_kwargs={
            "threshold": 0.95,
            "correlation_threshold": 0.02,
        },
        amp_ratio_sigma_prior=2.0,        # loosen ratio likelihood
        location_samples_n=20,
        azimuth_error=1,
        takeoff_error=2,
        safe_sampling=False, # Custom flag to bypass ProcessPoolExecutor hangs
    )
    result = inversion_object.forward()
    

    # ------------------------------------------------------------------
    # Summarise and report moment tensor
    # ------------------------------------------------------------------
    result_obj = result[0] if isinstance(result, list) else result

    # Save inversion result as pickle file
    import pickle
    pickle_path = f"{test_output_dir}/inversion_result.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(result_obj, f)
    print(f"\nInversion result saved to \n{pickle_path}")

    import arviz as az
    post = result_obj.idata.posterior
    requested_vars = ["gamma", "delta", "kappa", "h", "sigma"]
    var_names = [v for v in requested_vars if v in post.data_vars]
    if var_names:
        summary_df = az.summary(result_obj.idata, var_names=var_names, hdi_prob=0.90)
        print(summary_df)

        summary_path = f"{test_output_dir}/arviz_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_df.to_string(max_cols=None, max_rows=None))
        print(f"\nArviZ summary saved to \n{summary_path}")

        summary_csv_path = f"{test_output_dir}/arviz_summary.csv"
        summary_df.to_csv(summary_csv_path)
        print(f"ArviZ summary (CSV) saved to \n{summary_csv_path}")
    else:
        print("No matching continuous parameters found in posterior for ArviZ summary.")

    non_dc_vars = [v for v in ["gamma", "delta"] if v in post.data_vars]
    if non_dc_vars:
        az.plot_posterior(result_obj.idata, var_names=non_dc_vars, hdi_prob=0.90)
        plt.savefig(f"{test_output_dir}/posterior_non_dc.png", dpi=150, bbox_inches="tight")
        plt.close()

    az.plot_posterior(result_obj.idata, var_names=["kappa", "h", "sigma"], hdi_prob=0.90)
    plt.savefig(f"{test_output_dir}/posterior_dc.png", dpi=150, bbox_inches="tight")
    plt.close()

    pair_vars = [v for v in requested_vars if v in post.data_vars]
    if pair_vars:
        az.plot_pair(result_obj.idata, var_names=pair_vars, kind="kde", marginals=True)
        plt.savefig(f"{test_output_dir}/posterior_pairs.png", dpi=150, bbox_inches="tight")
        plt.close()

    mt6_samples = np.asarray(result_obj.mt6, dtype=float)
    if mt6_samples.ndim != 2 or mt6_samples.shape[0] != 6:
        raise RuntimeError(f"Unexpected mt6 shape: {mt6_samples.shape}")

    mt6_median = np.median(mt6_samples, axis=1)

    mt6_mode, idx_map = find_map_mt6_gmm(mt6_samples)
    print(f"\nMAP estimate found at sample index: {idx_map}")

    mt33_median = MT6_MT33(mt6_median)
    mt33_mode = MT6_MT33(mt6_mode)
    mt33_true = MT6_MT33(true_mt6)
    
    print("\nTrue MT6:")
    print(true_mt6)

    print("\nPyTensor inversion – posterior median MT6 (normalised):")
    print(mt6_median)

    print("\nPyTensor inversion – MAP (mode) MT6 from KDE (normalised):")
    print(mt6_mode)

    print("\nCorresponding true 3x3 moment tensor (N, E, D):")
    print(mt33_true)
    print("\nCorresponding median 3x3 moment tensor (N, E, D):")
    print(mt33_median)
    print("\nCorresponding MAP (mode) 3x3 moment tensor (N, E, D):")
    print(mt33_mode)

    # ------------------------------------------------------------------
    # MT quality score
    # ------------------------------------------------------------------
    quality = mt_quality_score(result_obj.idata, mt6_samples, mt6_mode)
    print("\nMT quality score (0–1):")
    print(f"  Q              = {quality['Q']:.3f}  ({quality['quality_label']})")
    print(f"  s_conv         = {quality['s_conv']:.3f}")
    print(f"  s_angle        = {quality['s_angle']:.3f}")
    print(f"  s_modes        = {quality['s_modes']:.3f}")
    print(f"  rhat_max       = {quality['rhat_max']:.3f}")
    print(f"  ess_min        = {quality['ess_min']:.1f}")
    print(f"  q50 (deg)      = {quality['q50']:.2f}")
    print(f"  q90 (deg)      = {quality['q90']:.2f}")
    print(f"  top_mode_weight= {quality['top_mode_weight']:.3f}")

    # ------------------------------------------------------------------
    # Save MT quality scores and MT6 solutions to CSV
    # ------------------------------------------------------------------
    import pandas as pd

    results_dict = {
        # Quality scores
        'Q': quality['Q'],
        'quality_label': quality['quality_label'],
        's_conv': quality['s_conv'],
        's_angle': quality['s_angle'],
        's_modes': quality['s_modes'],
        'rhat_max': quality['rhat_max'],
        'ess_min': quality['ess_min'],
        'q50_deg': quality['q50'],
        'q90_deg': quality['q90'],
        'top_mode_weight': quality['top_mode_weight'],
        # True MT6
        'true_mt6_0': true_mt6[0],
        'true_mt6_1': true_mt6[1],
        'true_mt6_2': true_mt6[2],
        'true_mt6_3': true_mt6[3],
        'true_mt6_4': true_mt6[4],
        'true_mt6_5': true_mt6[5],
        # Median MT6
        'median_mt6_0': mt6_median[0],
        'median_mt6_1': mt6_median[1],
        'median_mt6_2': mt6_median[2],
        'median_mt6_3': mt6_median[3],
        'median_mt6_4': mt6_median[4],
        'median_mt6_5': mt6_median[5],
        # Mode MT6
        'mode_mt6_0': mt6_mode[0],
        'mode_mt6_1': mt6_mode[1],
        'mode_mt6_2': mt6_mode[2],
        'mode_mt6_3': mt6_mode[3],
        'mode_mt6_4': mt6_mode[4],
        'mode_mt6_5': mt6_mode[5],
    }

    results_df = pd.DataFrame([results_dict])
    results_csv_path = f"{test_output_dir}/mt_results.csv"
    results_df.to_csv(results_csv_path, index=False,float_format="%.3f")
    print(f"\nMT quality scores and solutions saved to \n{results_csv_path}")
    
    
    # ------------------------------------------------------------------

    fig = plt.figure(figsize=(2.5, 2.5))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    mt_for_plot = project_to_dc_mt6(mt33_median) if dc else mt6_median.reshape(6, 1)
    amp_plot = _AmplitudePlot(
        None,
        fig,
        mt_for_plot,
        phase="P",
        projection="equalarea",
        lower=True,
        full_sphere=False,
        colormap="bwr",
        axis_lines=False,
        fault_plane=True,
        nodal_line=False,
        TNP=False,
        text=False,
        show=False,
        resolution=720,
    )
    amp_plot.plot()
    ax = amp_plot.ax
    ax.set_axis_off()
    ax.set_aspect("equal")

    ppol = data.get("PPolarity")
    if ppol is not None:
        st = ppol["Stations"]
        az = np.asarray(st["Azimuth"], float).ravel()
        to = np.asarray(st["TakeOffAngle"], float).ravel()
        pol = np.asarray(ppol["Measured"], float).ravel()  # ±1

        # Use the same projection logic as MTfit's _FocalSpherePlot._plot_stations
        v = toa_vec(az, to, radians=False)
        x = np.squeeze(np.array(v[0, :]))
        y = np.squeeze(np.array(v[1, :]))
        z = np.squeeze(np.array(v[2, :]))

        # Duplicate antipodal points so that the chosen lower-hemisphere
        # projection always shows a station, regardless of original takeoff.
        x_all = np.append(x, -x)
        y_all = np.append(y, -y)
        z_all = np.append(z, -z)
        pol_all = np.append(pol, pol)

        # switched so north is up and east right (match _projection_fn usage)
        y_proj, x_proj = equal_area(
            x_all, y_all, z_all, lower=True, full_sphere=False
        )
        idx = ~np.isnan(x_proj)
        Xp = x_proj[idx]
        Yp = y_proj[idx]
        pol_plot = pol_all[idx]

        up = pol_plot > 0
        down = pol_plot < 0
        if np.any(up):
            ax.scatter(Xp[up], Yp[up], c="r", marker="o", edgecolors="k", s=40, label="P up")
        if np.any(down):
            ax.scatter(Xp[down], Yp[down], c="b", marker="o", edgecolors="k", s=40, label="P down")
        if np.any(up) or np.any(down):
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.06, 1.04),
                fontsize=8,
                frameon=False,
                borderaxespad=0.01,
                handletextpad=0.01,
                labelspacing=0.2,
                )

    fig.tight_layout()
    fig.savefig(f"{test_output_dir}/bb_median.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # ------------------------------------------------------------------
    # Beachball plot for posterior MT modes with SH-polarity stations
    # ------------------------------------------------------------------
    if "SHPolarity" in data:
        fig = plt.figure(figsize=(2.5, 2.5))
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

        mt_for_plot = project_to_dc_mt6(mt33_median) if dc else mt6_median.reshape(6, 1)
        amp_plot = _AmplitudePlot(
            None,
            fig,
            mt_for_plot,
            phase="SH",
            projection="equalarea",
            lower=True,
            full_sphere=False,
            colormap="bwr",
            axis_lines=False,
            fault_plane=True,
            nodal_line=False,
            TNP=False,
            text=False,
            show=False,
            resolution=720,
        )
        amp_plot.plot()
        ax = amp_plot.ax
        ax.set_axis_off()
        ax.set_aspect("equal")

        ppol = data.get("SHPolarity")
        if ppol is not None:
            st = ppol["Stations"]
            az = np.asarray(st["Azimuth"], float).ravel()
            to = np.asarray(st["TakeOffAngle"], float).ravel()
            pol = np.asarray(ppol["Measured"], float).ravel()  # ±1

            # Same station projection as for the P-wave plot
            v = toa_vec(az, to, radians=False)
            x = np.squeeze(np.array(v[0, :]))
            y = np.squeeze(np.array(v[1, :]))
            z = np.squeeze(np.array(v[2, :]))

            x_all = np.append(x, -x)
            y_all = np.append(y, -y)
            z_all = np.append(z, -z)
            pol_all = np.append(pol, pol)

            y_proj, x_proj = equal_area(
                x_all, y_all, z_all, lower=True, full_sphere=False
            )
            idx = ~np.isnan(x_proj)
            Xp = x_proj[idx]
            Yp = y_proj[idx]
            pol_plot = pol_all[idx]

            up = pol_plot > 0
            down = pol_plot < 0
            if np.any(up):
                ax.scatter(Xp[up], Yp[up], c="r", marker="s", edgecolors="k", s=30, label="SH up")
            if np.any(down):
                ax.scatter(Xp[down], Yp[down], c="b", marker="s", edgecolors="k", s=30, label="SH down")
            if np.any(up) or np.any(down):
                ax.legend(
                    loc="upper right",
                    bbox_to_anchor=(1.06, 1.04),
                    fontsize=8,
                    frameon=False,
                    borderaxespad=0.01,
                    handletextpad=0.01,
                    labelspacing=0.2,
                )
        fig.tight_layout()
        fig.savefig(f"{test_output_dir}/bb_median_sh.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    # ------------------------------------------------------------------
    # Beachball plot for posterior MT mode with P-polarity stations
    # ------------------------------------------------------------------

    fig = plt.figure(figsize=(2.5, 2.5))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    mt_for_plot = project_to_dc_mt6(mt33_mode) if dc else mt6_mode.reshape(6, 1)
    amp_plot = _AmplitudePlot(
        None,
        fig,
        mt_for_plot,
        phase="P",
        projection="equalarea",
        lower=True,
        full_sphere=False,
        colormap="bwr",
        axis_lines=False,
        fault_plane=True,
        nodal_line=False,
        TNP=False,
        text=False,
        show=False,
        resolution=720,
    )
    amp_plot.plot()
    ax = amp_plot.ax
    ax.set_axis_off()
    ax.set_aspect("equal")

    ppol = data.get("PPolarity")
    if ppol is not None:
        st = ppol["Stations"]
        az = np.asarray(st["Azimuth"], float).ravel()
        to = np.asarray(st["TakeOffAngle"], float).ravel()
        pol = np.asarray(ppol["Measured"], float).ravel()  # ±1

        v = toa_vec(az, to, radians=False)
        x = np.squeeze(np.array(v[0, :]))
        y = np.squeeze(np.array(v[1, :]))
        z = np.squeeze(np.array(v[2, :]))

        x_all = np.append(x, -x)
        y_all = np.append(y, -y)
        z_all = np.append(z, -z)
        pol_all = np.append(pol, pol)

        y_proj, x_proj = equal_area(
            x_all, y_all, z_all, lower=True, full_sphere=False
        )
        idx = ~np.isnan(x_proj)
        Xp = x_proj[idx]
        Yp = y_proj[idx]
        pol_plot = pol_all[idx]

        up = pol_plot > 0
        down = pol_plot < 0
        if np.any(up):
            ax.scatter(Xp[up], Yp[up], c="r", marker="o", edgecolors="k", s=40, label="P up")
        if np.any(down):
            ax.scatter(Xp[down], Yp[down], c="b", marker="o", edgecolors="k", s=40, label="P down")
        if np.any(up) or np.any(down):
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.06, 1.04),
                fontsize=8,
                frameon=False,
                borderaxespad=0.01,
                handletextpad=0.01,
                labelspacing=0.2,
                )

    fig.tight_layout()
    fig.savefig(f"{test_output_dir}/bb_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fault-plane distribution plot using 90% Kaverina HDI samples
    # ------------------------------------------------------------------
    print("\nCreating fault-plane distribution plot (90% Kaverina HDI)...")
    fig_fp = plt.figure(figsize=(3.0, 3.0))
    fig_fp.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    hdi_mask, kav_density, kav_thr = kaverina_hdi_mask(mt6_samples, hdi_prob=0.90)
    mt6_hdi = mt6_samples[:, hdi_mask]
    prob_hdi = kav_density[hdi_mask]

    max_fp = mt6_hdi.shape[1]#4000
    if mt6_hdi.shape[1] > max_fp:
        idx_sorted = np.argsort(prob_hdi)[::-1]
        idx_keep = idx_sorted[:max_fp]
        mt6_hdi = mt6_hdi[:, idx_keep]
        prob_hdi = prob_hdi[idx_keep]

    stations_fp = {}
    ppol_fp = data.get("PPolarity")
    if ppol_fp is not None:
        st = ppol_fp["Stations"]
        names = list(st["Name"])
        az = np.asarray(st["Azimuth"], float).reshape(-1)
        to = np.asarray(st["TakeOffAngle"], float).reshape(-1)
        pol = np.asarray(ppol_fp["Measured"], int).reshape(-1)
        stations_fp = {
            "names": names,
            "azimuth": az,
            "takeoff_angle": to,
            "polarity": pol,
        }

    fault_plot = _FaultPlanePlot(
        None,
        fig_fp,
        mt6_hdi,
        stations=stations_fp,
        probability=prob_hdi,
        phase="P",
        projection="equalarea",
        lower=True,
        full_sphere=False,
        fault_plane=True,
        nodal_line=False,
        axis_lines=False,
        TNP=False,
        text=False,
        show_max_likelihood=True,
        show_mean=True,
        probability_cutoff=0.0,
        max_number_planes=max_fp,
        linewidth=0.6,
    )
    fault_plot.plot()

    fault_plot.ax.set_axis_off()
    fault_plot.ax.set_aspect("equal")
    fig_fp.savefig(f"{test_output_dir}/fault_planes_hdi90.png", dpi=200, bbox_inches="tight")
    plt.close(fig_fp)
    
    # ------------------------------------------------------------------
    # Hudson plot with HDI contours
    # ------------------------------------------------------------------
    print("\nCreating Hudson plot with 90% HDI contours...")
    plot_hudson_with_hdi(
        mt6_samples,
        hdi_prob=0.90,
        output_path=f"{test_output_dir}/hudson_hdi_90.png",
        grid_size=300,
        n_contour_levels=20,
        min_threshold=0.0,
        cmap='jet',
        show_samples=False,
        # mt6_median=mt6_median,
        # mt6_mode=mt6_mode,
        mt6_true=true_mt6
    )

    # ------------------------------------------------------------------
    # Kaverina plot with HDI contours
    # ------------------------------------------------------------------
    print("\nCreating Kaverina plot with 90% HDI contours...")
    plot_kaverina_with_hdi(
        mt6_samples,
        hdi_prob=0.90,
        output_path=f"{test_output_dir}/kaverina_hdi_90.png",
        grid_size=300,
        n_contour_levels=20,
        cmap='jet',
        show_samples=False,
        # mt6_median=mt6_median,
        # mt6_mode=mt6_mode,
        mt6_true=true_mt6
    )


    # Close logger and restore stdout/stderr
    print(f"\nAll outputs saved to {test_output_dir}")
    if enable_file_logging:
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()
        print(f"Log file saved to {log_path}")


if __name__ == "__main__":
    import sys
    draws = 5000
    chains = 4
    cores = None
    run(draws=draws, chains=chains, cores=cores, enable_file_logging=False, dc=False, inversion_options=None)
