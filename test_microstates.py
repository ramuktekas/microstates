"""
test_microstates.py - VAR microstate analysis

Workflow:
1. Load full recording (NO filtering before VAR)
2. Scale data by 10e6
3. Fit VAR with statsmodels (default trend='c')
4. Simulate with simulate_var()
5. Scale back by /10e6
6. Filter + avg-ref for microstate extraction
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from statsmodels.tsa.api import VAR as StatsmodelsVAR

# Add micro-VAR-states to path
sys.path.insert(0, "/home/kumarsak/micro-VAR-states/python")

from microstates import (
    get_gfp_peaks,
    run_modified_k_means,
    KoenigTemplates,
    match_topomaps_per_template,
)

# =========================
# USER CONFIG (matching eeg_latent_space)
# =========================
LEMON_DIR = Path("/store/projects/kumarsak/LEMON_data")
OUT_DIR = Path("./microstate_results_eeg_latent_style")

N_SUBJECTS = 1
N_STATES = 4
RANDOM_SEED = 42
SUBJECT_ID = "sub-010006_EC"

# VAR parameters
VAR_SIMULATION_LENGTH = 3600.0  # seconds to simulate
DATA_FILTER = [2.0, 20.0]  # Filter applied AFTER VAR, for microstate extraction

np.random.seed(RANDOM_SEED)
OUT_DIR.mkdir(exist_ok=True)


# =========================
# UTILITIES
# =========================

def subject_maps_to_raw(km_maps, info):
    """Convert microstate maps (states x channels) into MNE RawArray."""
    return mne.io.RawArray(km_maps.T, info, verbose=False)


def plot_microstate_maps(km_maps, info, subject_id, out_dir):
    """Plot and save microstate topographies."""
    fig, axes = plt.subplots(1, km_maps.shape[0], figsize=(3 * km_maps.shape[0], 3))

    if km_maps.shape[0] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        mne.viz.plot_topomap(
            km_maps[i],
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap="RdBu_r",
        )
        ax.set_title(f"State {chr(ord('A') + i)}")

    fig.suptitle(f"{subject_id} - Microstate Topographies")

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    outpath = out_dir / f"{subject_id}_microstates_ABCD.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved microstate figure -> {outpath}")


def process_subject_eeg_latent_style(set_path, out_dir_base, n_states, p_fixed):
    """
    Process subject using statsmodels VAR approach.

    Workflow:
    1. Load full recording (NO filtering, NO cropping)
    2. Scale by 10e6
    3. Fit VAR with statsmodels (default trend='c')
    4. Simulate with simulate_var()
    5. Scale back by /10e6
    6. Filter + avg-ref for microstate extraction
    """
    subj_id = set_path.stem
    print(f"\n{'='*60}")
    print(f"Processing {subj_id} (p = {p_fixed}) - Full recording VAR")
    print(f"{'='*60}")

    # Initialize diagnostics struct
    diagnostics = {
        'subject_id': subj_id,
        'p_fixed': p_fixed,
        'n_states': n_states
    }

    # --- Load EEG (full recording, no cropping) ---
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

    diagnostics['original_data_size'] = raw.get_data().shape
    diagnostics['srate'] = raw.info['sfreq']

    print(f"\nFull recording shape: {raw.get_data().shape}")
    print(f"Duration: {raw.times[-1]:.1f} seconds")

    # --- STEP 1: Scale by 10e6 (like eeg_latent_space) ---
    # NO filtering before VAR, NO cropping - use full recording
    data_for_var = raw.get_data().T * 10e6  # time x channels

    diagnostics['data_shape_for_var'] = data_for_var.shape
    diagnostics['data_range_for_var'] = [float(np.min(data_for_var)), float(np.max(data_for_var))]

    print(f"Data for VAR shape: {data_for_var.shape} (time x channels)")
    print(f"Data range for VAR (10e6 scaled): [{np.min(data_for_var):.4f}, {np.max(data_for_var):.4f}]")

    # --- STEP 2: Fit VAR using statsmodels with default trend='c' ---
    pd_data = pd.DataFrame(
        data_for_var,
        columns=raw.info['ch_names'],
        index=raw.times,
    )

    print(f"\n=== Fitting VAR(p={p_fixed}) with trend='c' ===")
    model = StatsmodelsVAR(pd_data)
    fit_results = model.fit(p_fixed)  # default trend='c'

    # Check stability via companion matrix
    A = fit_results.coefs  # shape: (p, n_ch, n_ch)
    p_order, n_ch, _ = A.shape

    F = np.zeros((n_ch * p_order, n_ch * p_order))
    for k in range(p_order):
        F[:n_ch, k*n_ch:(k+1)*n_ch] = A[k]
    if p_order > 1:
        F[n_ch:, :-n_ch] = np.eye(n_ch * (p_order - 1))

    max_eig = np.max(np.abs(np.linalg.eigvals(F)))

    print(f"A[0] first element: {fit_results.coefs[0, 0, 0]:.6f}")
    print(f"A sum of abs: {np.sum(np.abs(fit_results.coefs)):.4f}")
    print(f"Max eigenvalue: {max_eig:.6f}")
    print(f"Stable (<1): {max_eig < 1.0}")
    print(f"Sigma condition: {np.linalg.cond(fit_results.sigma_u):.4e}")

    diagnostics['var_fit'] = {
        'max_eig': float(max_eig),
        'stabilized': max_eig < 1.0,
        'sigma_condition': float(np.linalg.cond(fit_results.sigma_u)),
        'A_sum_abs': float(np.sum(np.abs(fit_results.coefs)))
    }

    # --- STEP 3: Simulate using statsmodels simulate_var ---
    Tvar = int(VAR_SIMULATION_LENGTH * raw.info['sfreq'])
    print(f"\n=== Simulating VAR for {Tvar} samples ({VAR_SIMULATION_LENGTH}s) ===")

    simulated_df = fit_results.simulate_var(steps=Tvar)
    if hasattr(simulated_df, "values"):
        data_sim_scaled = simulated_df.values.T   # DataFrame case -> channels x time
    else:
        data_sim_scaled = simulated_df.T          # ndarray case

    print(f"Simulated data shape: {data_sim_scaled.shape}")
    print(f"Simulated data range (10e6 scaled): [{np.min(data_sim_scaled):.4f}, {np.max(data_sim_scaled):.4f}]")

    # --- STEP 4: Scale back by /10e6 ---
    data_sim = data_sim_scaled / 10e6  # Back to Volts

    print(f"Simulated data range (V): [{np.min(data_sim):.4e}, {np.max(data_sim):.4e}]")
    print(f"Simulated data range (uV): [{np.min(data_sim)*1e6:.4f}, {np.max(data_sim)*1e6:.4f}]")

    diagnostics['var_simulation'] = {
        'T': Tvar,
        'var_length_seconds': VAR_SIMULATION_LENGTH,
        'sigma_condition': float(np.linalg.cond(fit_results.sigma_u)),
        'data_range_scaled': [float(np.min(data_sim_scaled)), float(np.max(data_sim_scaled))],
        'data_range_V': [float(np.min(data_sim)), float(np.max(data_sim))],
        'contains_nan': bool(np.any(np.isnan(data_sim))),
        'contains_inf': bool(np.any(np.isinf(data_sim))),
        'mean_abs_V': float(np.mean(np.abs(data_sim))),
        'mean_std_V': float(np.mean(np.std(data_sim, axis=1)))
    }

    # Check for issues
    if (diagnostics['var_simulation']['contains_nan'] or
        diagnostics['var_simulation']['contains_inf']):
        raise RuntimeError("VAR simulation produced invalid data (NaN/Inf).")

    # --- STEP 5: Convert to MNE Raw, filter, then extract microstates ---

    # Create MNE Raw from simulated data
    simulated_raw = mne.io.RawArray(data_sim, raw.info, verbose=False)

    # Apply bandpass filter (like eeg_latent_space preprocess)
    simulated_raw_filtered = simulated_raw.copy().filter(
        l_freq=DATA_FILTER[0],
        h_freq=DATA_FILTER[1],
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )

    # Get filtered data in uV
    data_filtered_uV = simulated_raw_filtered.get_data() * 1e6  # channels x time

    print(f"\n=== After filtering {DATA_FILTER[0]}-{DATA_FILTER[1]} Hz ===")
    print(f"Filtered data range (uV): [{np.min(data_filtered_uV):.4f}, {np.max(data_filtered_uV):.4f}]")

    # Apply average reference for microstates
    data_sim_avgref = data_filtered_uV - data_filtered_uV.mean(axis=0, keepdims=True)
    print(f"After avg-ref range (uV): [{np.min(data_sim_avgref):.4f}, {np.max(data_sim_avgref):.4f}]")

    # --- GFP Analysis ---
    gfp_peaks_var, gfp_curve = get_gfp_peaks(data_sim_avgref)

    diagnostics['gfp'] = {
        'n_peaks': len(gfp_peaks_var),
        'peak_range': [int(np.min(gfp_peaks_var)), int(np.max(gfp_peaks_var))],
        'gfp_curve_range': [float(np.min(gfp_curve)), float(np.max(gfp_curve))],
        'gfp_curve_mean': float(np.mean(gfp_curve))
    }

    print(f"\n=== GFP Analysis ===")
    print(f"Number of GFP peaks: {len(gfp_peaks_var)}")
    print(f"GFP curve range: [{np.min(gfp_curve):.4e}, {np.max(gfp_curve):.4e}]")

    if len(gfp_peaks_var) < n_states:
        raise RuntimeError(f"Not enough GFP peaks ({len(gfp_peaks_var)}) for {n_states} states")

    peaks_data = data_sim_avgref[:, gfp_peaks_var]
    diagnostics['gfp']['peaks_data_range'] = [float(np.min(peaks_data)), float(np.max(peaks_data))]

    # --- Microstates ---
    km_maps = run_modified_k_means(data_sim_avgref[:, gfp_peaks_var], n_states=n_states)

    diagnostics['microstates'] = {
        'n_states': km_maps.shape[0],
        'n_channels': km_maps.shape[1],
        'maps_range': [float(np.min(km_maps)), float(np.max(km_maps))]
    }

    # --- Template matching ---
    subject_maps_raw = subject_maps_to_raw(km_maps, raw.info)
    templates = KoenigTemplates.load(n_states=n_states)

    attribution, corr = match_topomaps_per_template(
        maps=subject_maps_raw,
        template=templates
    )

    km_maps_labeled = km_maps[attribution, :]

    diagnostics['template_matching'] = {
        'attribution': list(attribution),
        'correlations': list(corr),
        'mean_correlation': float(np.mean(corr))
    }

    # --- Save everything ---
    out_dir = Path(out_dir_base) / subj_id
    out_dir.mkdir(exist_ok=True, parents=True)

    np.save(out_dir / "var_p_fixed.npy", p_fixed)
    np.save(out_dir / "var_km_maps.npy", km_maps)
    np.save(out_dir / "var_km_maps_labeled.npy", km_maps_labeled)
    np.save(out_dir / "var_permutation.npy", attribution)
    np.save(out_dir / "var_diagnostics_full.npy", diagnostics, allow_pickle=True)

    write_diagnostics_summary(out_dir / "var_diagnostics_summary.txt", diagnostics)

    return km_maps_labeled, raw.info, subj_id, out_dir, diagnostics


def write_diagnostics_summary(filename, diagnostics):
    """Write human-readable diagnostics summary."""
    with open(filename, 'w') as f:
        f.write("=== VAR-Microstate Analysis (eeg_latent_space style) ===\n\n")
        f.write(f"Subject: {diagnostics['subject_id']}\n")
        f.write(f"VAR Order: {diagnostics['p_fixed']}\n")
        f.write(f"Number of States: {diagnostics['n_states']}\n\n")

        f.write("--- Data Preparation ---\n")
        f.write(f"Original size: {diagnostics['original_data_size'][0]} x {diagnostics['original_data_size'][1]}\n")
        f.write(f"Sampling rate: {diagnostics['srate']:.1f} Hz\n")
        f.write(f"Data for VAR shape: {diagnostics['data_shape_for_var'][0]} x {diagnostics['data_shape_for_var'][1]}\n")
        f.write(f"Data range for VAR (10e6 scaled): [{diagnostics['data_range_for_var'][0]:.4f}, {diagnostics['data_range_for_var'][1]:.4f}]\n\n")

        if 'var_fit' in diagnostics:
            f.write("--- VAR Fitting (statsmodels, trend='c') ---\n")
            vf = diagnostics['var_fit']
            f.write(f"Max eigenvalue: {vf['max_eig']:.6f}\n")
            f.write(f"Stable (eigenvalue < 1): {vf.get('stabilized', False)}\n")
            f.write(f"Sigma condition: {vf['sigma_condition']:.4e}\n")
            f.write(f"A sum of abs: {vf['A_sum_abs']:.4f}\n\n")

        if 'var_simulation' in diagnostics:
            f.write("--- VAR Simulation ---\n")
            vs = diagnostics['var_simulation']
            f.write(f"Simulated length: {vs['T']} samples ({vs['var_length_seconds']:.0f}s)\n")
            f.write(f"Data range (10e6 scaled): [{vs['data_range_scaled'][0]:.4f}, {vs['data_range_scaled'][1]:.4f}]\n")
            f.write(f"Data range (V): [{vs['data_range_V'][0]:.4e}, {vs['data_range_V'][1]:.4e}]\n")
            f.write(f"Contains NaN: {vs['contains_nan']}\n")
            f.write(f"Contains Inf: {vs['contains_inf']}\n\n")

        if 'gfp' in diagnostics:
            f.write("--- GFP Analysis (after filter + avg-ref) ---\n")
            gfp = diagnostics['gfp']
            f.write(f"Number of peaks: {gfp['n_peaks']}\n")
            f.write(f"GFP curve range: [{gfp['gfp_curve_range'][0]:.4e}, {gfp['gfp_curve_range'][1]:.4e}]\n")
            f.write(f"Peaks data range (uV): [{gfp['peaks_data_range'][0]:.4f}, {gfp['peaks_data_range'][1]:.4f}]\n\n")

        if 'template_matching' in diagnostics:
            f.write("--- Template Matching ---\n")
            tm = diagnostics['template_matching']
            f.write(f"Attribution: {tm['attribution']}\n")
            f.write(f"Correlations: {[f'{c:.4f}' for c in tm['correlations']]}\n")
            f.write(f"Mean correlation: {tm['mean_correlation']:.4f}\n")

    print(f"Diagnostics summary saved to: {filename}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    set_path = LEMON_DIR / f"{SUBJECT_ID}.set"
    if not set_path.exists():
        raise FileNotFoundError(f"Subject not found: {set_path}")

    km_maps, info, subj_id, out_dir, diagnostics = process_subject_eeg_latent_style(
        set_path, OUT_DIR, N_STATES, p_fixed=10
    )

    plot_microstate_maps(km_maps, info, subj_id, out_dir)

    print("\nDone!")
