"""
run_microstates_lemon.py - Python port of run_microstates_lemon.m

Replicates the MATLAB implementation exactly with same diagnostics.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import mne

# Add micro-VAR-states to path
sys.path.insert(0, "/home/kumarsak/micro-VAR-states/python")

from VAR import VAR
from microstates import (
    get_gfp_peaks,
    run_modified_k_means,
    KoenigTemplates,
    match_topomaps_per_template,
)

# =========================
# USER CONFIG
# =========================
LEMON_DIR = Path("/store/projects/kumarsak/LEMON_data")
OUT_DIR = Path("./microstate_results_python_var")

N_SUBJECTS = 1
N_STATES = 4
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
OUT_DIR.mkdir(exist_ok=True)


# =========================
# UTILITIES
# =========================

def load_lemon_subjects(lemon_dir, n_subjects):
    """
    Load LEMON EEG .set files.

    Parameters
    ----------
    lemon_dir : Path
        Directory containing .set files
    n_subjects : int or str
        Number of subjects or 'all'

    Returns
    -------
    list of Path
        Sorted list of .set file paths
    """
    files = sorted(lemon_dir.glob("*_EC.set"))

    if isinstance(n_subjects, str) and n_subjects == 'all':
        return files

    if isinstance(n_subjects, int):
        return files[:min(n_subjects, len(files))]

    raise ValueError("n_subjects must be 'all' or an integer")


def subject_maps_to_raw(km_maps, info):
    """Convert microstate maps (states x channels) into MNE RawArray."""
    return mne.io.RawArray(km_maps.T, info, verbose=False)


def preprocess_for_gfp_peaks(raw, l_freq=1.0, h_freq=40.0, do_ica=False):
    """
    Classical EEGLAB-style preprocessing for microstates.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object
    l_freq : float
        Low cutoff frequency
    h_freq : float
        High cutoff frequency
    do_ica : bool
        Whether to run ICA cleaning

    Returns
    -------
    gfp_peaks : np.ndarray
        Indices of GFP peaks
    gfp_curve : np.ndarray
        GFP time series
    raw : mne.io.Raw
        Preprocessed Raw object
    """
    # Band-pass filter (zero-phase FIR)
    raw = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )

    # Average reference (MANDATORY for microstates)
    raw = raw.set_eeg_reference("average", projection=False, verbose=False)

    # ICA cleaning (OPTIONAL)
    if do_ica:
        ica = mne.preprocessing.ICA(
            n_components=None,
            method="fastica",
            random_state=RANDOM_SEED,
            max_iter="auto",
        )
        ica.fit(raw, verbose=False)
        raw = ica.apply(raw.copy(), verbose=False)

    # GFP peaks on voltage data
    data = raw.get_data()
    gfp_peaks, gfp_curve = get_gfp_peaks(data)

    return gfp_peaks, gfp_curve, raw


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


def estimate_var_order(set_path):
    """
    Estimate VAR order for a subject.

    Parameters
    ----------
    set_path : Path
        Path to .set file

    Returns
    -------
    p_opt : int
        Optimal VAR order
    aic_vals : np.ndarray
        AIC values
    diagnostics : dict
        Order selection diagnostics
    """
    subj_id = set_path.stem
    print(f"\n=== Estimating VAR order for {subj_id} ===")

    # Load EEG
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

    # Preprocess (same as main pipeline)
# --- VAR preprocessing (NO average reference) ---
    raw_var = raw.copy().filter(
        l_freq=1.0,
        h_freq=40.0,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )

    data = raw_var.get_data()   # ❗ full-rank data



    # Create VAR model
    var_model = VAR(verbose=True, save_diagnostics=True)

    # Select order with p_max=10, elbow=True
    p_max = 10
    p_opt, aic_vals, diagnostics = var_model.select_order(data, p_max, elbow=True)

    return p_opt, aic_vals, diagnostics


def process_subject_fixed_p(set_path, out_dir_base, n_states, p_fixed):
    """
    Process subject with VAR-based microstates using fixed p.

    Matches MATLAB process_subject_fixed_p exactly.

    Parameters
    ----------
    set_path : Path
        Path to .set file
    out_dir_base : Path
        Base output directory
    n_states : int
        Number of microstates
    p_fixed : int
        Fixed VAR order

    Returns
    -------
    km_maps_labeled : np.ndarray
        Labeled microstate maps
    info : mne.Info
        MNE info object
    subj_id : str
        Subject ID
    out_dir : Path
        Subject output directory
    diagnostics : dict
        Full diagnostics
    """
    subj_id = set_path.stem
    print(f"\n=== Processing {subj_id} (p = {p_fixed}) ===")

    # Initialize diagnostics struct
    diagnostics = {
        'subject_id': subj_id,
        'p_fixed': p_fixed,
        'n_states': n_states
    }

    # --- Load EEG ---
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

    diagnostics['original_data_size'] = raw.get_data().shape
    diagnostics['srate'] = raw.info['sfreq']

    # --- Preprocessing ---
    # --- VAR preprocessing (NO average reference) ---
    raw_var = raw.copy().filter(
        l_freq=1.0,
        h_freq=40.0,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )

    data_var = raw_var.get_data()   # ❗ used for VAR

    # --- Microstate preprocessing (WITH average reference) ---
    raw_ms = raw.copy().filter(
        l_freq=1.0,
        h_freq=40.0,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )
    raw_ms.set_eeg_reference("average", projection=False, verbose=False)


    diagnostics['preprocessed_data_size'] = data_var.shape
    diagnostics['data_range_original'] = [float(np.min(data_var)), float(np.max(data_var))]

    # --- VAR with fixed order ---
    var_model = VAR(verbose=True, save_diagnostics=True)
    model = var_model.fit(data_var, p_fixed)

    # Store fit diagnostics
    if 'diagnostics' in model:
        diagnostics['var_fit'] = model['diagnostics']

    # --- Simulate long VAR data ---
    Tvar = int(3600 * raw.info['sfreq'])  # 1 hour of data
    data_sim, sim_diag = var_model.simulate(model, Tvar)

    # Store simulation diagnostics
    diagnostics['var_simulation'] = sim_diag

    # Detailed checks
    diagnostics['simulation_checks'] = {
        'size': data_sim.shape,
        'range': [float(np.min(data_sim)), float(np.max(data_sim))],
        'contains_nan': bool(np.any(np.isnan(data_sim))),
        'contains_inf': bool(np.any(np.isinf(data_sim))),
        'mean_abs': float(np.mean(np.abs(data_sim))),
        'std_mean': float(np.mean(np.std(data_sim, axis=1)))
    }

    # ABORT if simulation failed
    if (diagnostics['simulation_checks']['contains_nan'] or
        diagnostics['simulation_checks']['contains_inf'] or
        diagnostics['simulation_checks']['mean_abs'] > 1e6):
        raise RuntimeError("VAR simulation produced invalid data. Model is still unstable.")
    # --- Apply average reference to simulated data (for microstates) ---
    data_sim = data_sim - data_sim.mean(axis=0, keepdims=True)

    # --- GFP Analysis ---
    gfp_peaks_var, gfp_curve = get_gfp_peaks(data_sim)

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

    peaks_data = data_sim[:, gfp_peaks_var]
    diagnostics['gfp']['peaks_data_range'] = [float(np.min(peaks_data)), float(np.max(peaks_data))]
    diagnostics['gfp']['peaks_contain_nan'] = bool(np.any(np.isnan(peaks_data)))

    # --- Microstates ---
    km_maps = run_modified_k_means(data_sim[:, gfp_peaks_var], n_states=n_states)

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

    # Save model and results
    np.save(out_dir / "var_model.npy", model, allow_pickle=True)
    np.save(out_dir / "var_p_fixed.npy", p_fixed)
    np.save(out_dir / "var_km_maps.npy", km_maps)
    np.save(out_dir / "var_km_maps_labeled.npy", km_maps_labeled)
    np.save(out_dir / "var_permutation.npy", attribution)

    # Save comprehensive diagnostics
    np.save(out_dir / "var_diagnostics_full.npy", diagnostics, allow_pickle=True)

    # Also save a summary text file
    write_diagnostics_summary(out_dir / "var_diagnostics_summary.txt", diagnostics)

    return km_maps_labeled, raw.info, subj_id, out_dir, diagnostics


def write_diagnostics_summary(filename, diagnostics):
    """Write human-readable diagnostics summary."""
    with open(filename, 'w') as f:
        f.write("=== VAR-Microstate Analysis Diagnostics ===\n\n")
        f.write(f"Subject: {diagnostics['subject_id']}\n")
        f.write(f"VAR Order: {diagnostics['p_fixed']}\n")
        f.write(f"Number of States: {diagnostics['n_states']}\n\n")

        f.write("--- Data ---\n")
        f.write(f"Original size: {diagnostics['original_data_size'][0]} x {diagnostics['original_data_size'][1]}\n")
        f.write(f"Sampling rate: {diagnostics['srate']:.1f} Hz\n")
        f.write(f"Data range: [{diagnostics['data_range_original'][0]:.4e}, {diagnostics['data_range_original'][1]:.4e}]\n\n")

        if 'var_fit' in diagnostics:
            f.write("--- VAR Fitting ---\n")
            vf = diagnostics['var_fit']
            f.write(f"Original max eigenvalue: {vf['max_eig_original']:.6f}\n")
            f.write(f"Final max eigenvalue: {vf['max_eig_final']:.6f}\n")
            f.write(f"Rescale iterations: {vf['rescale_iterations']}\n")
            f.write(f"Stabilized: {vf.get('stabilized', False)}\n\n")

        if 'var_simulation' in diagnostics:
            f.write("--- VAR Simulation ---\n")
            vs = diagnostics['var_simulation']
            f.write(f"Simulated length: {vs['T']} samples\n")
            f.write(f"Sigma condition number: {vs['sigma_condition']:.4e}\n")
            f.write(f"Data range: [{vs['data_range'][0]:.4e}, {vs['data_range'][1]:.4e}]\n")
            f.write(f"Contains NaN: {vs['contains_nan']}\n")
            f.write(f"Contains Inf: {vs['contains_inf']}\n")
            f.write(f"Mean absolute value: {vs['mean_abs']:.4e}\n")
            f.write(f"Mean std: {vs['mean_std']:.4e}\n\n")

        if 'gfp' in diagnostics:
            f.write("--- GFP Analysis ---\n")
            gfp = diagnostics['gfp']
            f.write(f"Number of peaks: {gfp['n_peaks']}\n")
            f.write(f"GFP curve range: [{gfp['gfp_curve_range'][0]:.4e}, {gfp['gfp_curve_range'][1]:.4e}]\n")
            f.write(f"Peaks data range: [{gfp['peaks_data_range'][0]:.4e}, {gfp['peaks_data_range'][1]:.4e}]\n\n")

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
    # Load subjects
    sets = load_lemon_subjects(LEMON_DIR, N_SUBJECTS)
    nSubj = len(sets)

    # ---------- PASS 1: estimate p for each subject ----------
    print("\n=== Estimating VAR order per subject ===")

    p_opts = []

    for i, set_path in enumerate(sets):
        # Get p_opt, AIC values, AND diagnostics
        p_opt, aic_vals, order_diagnostics = estimate_var_order(set_path)
        p_opts.append(p_opt)

        # Get subject ID
        subj_id = set_path.stem

        # Create output directory
        out_dir = OUT_DIR / subj_id
        out_dir.mkdir(exist_ok=True, parents=True)

        # Save AIC values, optimal p, AND diagnostics
        np.save(out_dir / "var_aic.npy", aic_vals)
        np.save(out_dir / "var_p_opt.npy", p_opt)
        np.save(out_dir / "var_order_selection_diagnostics.npy", order_diagnostics, allow_pickle=True)

        print(f"Subject {i + 1} ({subj_id}): p_opt = {p_opt}")

    p_med = int(np.median(p_opts))
    print(f"\n>>> Median VAR order across subjects: p = {p_med} <<<")

    np.save(OUT_DIR / "var_popts_all_subjects.npy", p_opts)
    np.save(OUT_DIR / "var_p_median.npy", p_med)

    # ---------- PASS 2: full pipeline using fixed p ----------
    print("\n=== Running full VAR + microstates with fixed p ===")

    all_diagnostics = []

    for i, set_path in enumerate(sets):
        km_maps, info, subj_id, out_dir, diagnostics = process_subject_fixed_p(
            set_path, OUT_DIR, N_STATES, p_med
        )

        plot_microstate_maps(km_maps, info, subj_id, out_dir)

        # Store diagnostics for aggregation
        all_diagnostics.append(diagnostics)

    # Save aggregated diagnostics across all subjects
    np.save(OUT_DIR / "all_subjects_diagnostics.npy", all_diagnostics, allow_pickle=True)

    print("\nAll subjects processed.")
