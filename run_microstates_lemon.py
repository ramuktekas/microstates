#!/usr/bin/env python3
import os
import random
import string
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import scipy.io as sio
import numpy as np
from itertools import permutations
import mne
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/home/kumarsak/micro-VAR-states/python")

import microstates
print("microstates file:", microstates.__file__)
print("available names:", dir(microstates))


from microstates import (
    get_gfp_peaks,
    run_modified_k_means,
    run_hierarchical_clustering,
    corr_vectors,
)

# =========================
# USER CONFIG
# =========================
LEMON_DIR = Path("/store/projects/kumarsak/LEMON_data")
TEMPLATE_FILE = Path("/home/kumarsak/micro-VAR-states/python/templates/Koenig2002.set")
OUT_DIR = Path("./microstate_results")

N_SUBJECTS = 1
N_STATES = 4
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
OUT_DIR.mkdir(exist_ok=True)


# =========================
# UTILITIES
# =========================

def load_lemon_subjects(n_subjects):
    """
    Load n_subjects random LEMON EEGLAB .set files.

    Parameters
    ----------
    n_subjects : int
        Number of subjects to sample.

    Returns
    -------
    list of Path
        Paths to *_EC.set files.
    """
    sets = sorted(LEMON_DIR.glob("*_EC.set"))

    if len(sets) < n_subjects:
        raise RuntimeError(
            f"Requested {n_subjects} subjects, but only {len(sets)} found"
        )

    return random.sample(sets, n_subjects)




def load_koenig_templates_from_set(path, n_states):
    """
    Load Koenig microstate templates from EEGLAB .set file
    using scipy (robust, MNE-independent).
    """
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    # Koenig templates are stored as:
    # data.shape == (channels, states, conditions)
    data = mat["data"]

    if data.ndim != 3:
        raise RuntimeError(f"Unexpected Koenig template shape {data.shape}")

    # Nikola's convention:from itertools import permutations

    # take the condition corresponding to n_states
    maps = data[:, :n_states, n_states - 1]  # (channels × states)
    maps = maps.T                             # (states × channels)

    # Channel labels
    chanlocs = mat["chanlocs"]
    ch_names = [str(ch.labels) for ch in chanlocs]

    return maps, ch_names

def preprocess_for_gfp_peaks(
    raw,
    l_freq=1.0,
    h_freq=40.0,
    use_hilbert=True,
):
    """
    Preprocess EEG only to compute GFP peaks.
    Does NOT alter data used for microstate maps.
    """

    raw_filt = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design="firwin",
        verbose=False,
    )

    if use_hilbert:
        raw_filt.apply_hilbert(envelope=True)

    data = raw_filt.get_data()  # channels × time

    gfp_peaks, gfp_curve = get_gfp_peaks(data)

    return gfp_peaks, gfp_curve

def match_reorder_topomaps(
    maps_input,
    maps_sortby,
    return_correlation=False,
    return_attribution_only=False,
):
    """
    Find the permutation of maps_input that best matches maps_sortby
    using absolute spatial correlation.
    """
    assert maps_input.shape == maps_sortby.shape, (
        maps_input.shape,
        maps_sortby.shape,
    )

    n_maps = maps_input.shape[0]
    best_corr_mean = -np.inf
    best_perm = None
    best_corr = None

    for perm in permutations(range(n_maps)):
        corr = np.abs(
            corr_vectors(
                maps_sortby,
                maps_input[list(perm), :],
                axis=1,
            )
        )
        if corr.mean() > best_corr_mean:
            best_corr_mean = corr.mean()
            best_perm = perm
            best_corr = corr

    if return_attribution_only:
        if return_correlation:
            return list(best_perm), best_corr
        return list(best_perm)

    reordered = maps_input[list(best_perm), :]
    if return_correlation:
        return reordered, best_corr
    return reordered


def match_reorder_maps_with_templates(
    maps,
    data_ch_names,
    template_maps,
    template_ch_names,
):
    """
    Reorder subject microstate maps using templates (Nikola-style).

    Parameters
    ----------
    maps : np.ndarray
        Subject microstate maps (n_states × n_channels)
    data_ch_names : list[str]
        Channel names of subject data
    template_maps : np.ndarray
        Template maps (n_states × n_template_channels)
    template_ch_names : list[str]
        Channel names of template maps

    Returns
    -------
    maps_reordered : np.ndarray
        Full-channel subject maps reordered to match templates
    attribution : list[int]
        Permutation corresponding to A/B/C/D
    """
    # --- channel intersection (TEMPORARY) ---
    common, idx_data, idx_template = np.intersect1d(
        data_ch_names,
        template_ch_names,
        return_indices=True,
    )

    if len(common) == 0:
        raise RuntimeError("No overlapping channels between data and templates")

    # reduced maps ONLY for matching
    maps_reduced = maps[:, idx_data]
    templates_reduced = template_maps[:, idx_template]

    # get permutation ONLY
    attribution = match_reorder_topomaps(
        maps_reduced,
        templates_reduced,
        return_attribution_only=True,
    )

    # apply permutation to FULL maps
    maps_reordered = maps[attribution, :]

    return maps_reordered, attribution



def plot_microstate_maps(maps, info, subject_id, out_dir):
    fig, axes = plt.subplots(1, maps.shape[0], figsize=(3 * maps.shape[0], 3))

    if maps.shape[0] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        mne.viz.plot_topomap(
            maps[i],
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap="RdBu_r",
        )
        ax.set_title(f"State {string.ascii_uppercase[i]}")

    fig.suptitle(f"{subject_id} – Microstate Topographies")

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    outpath = out_dir / f"{subject_id}_microstates_ABCD.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved microstate figure → {outpath}")


# =========================
# MAIN PIPELINE
# =========================
def process_subject(set_path):
    subj_id = set_path.stem
    print(f"\n=== Processing {subj_id} ===")

    # --- Load EEG ---
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    data = raw.get_data()  # channels × time

    # --- PREPROCESS ONLY FOR GFP ---
    gfp_peaks, gfp_curve = preprocess_for_gfp_peaks(
        raw,
        l_freq=1.0,
        h_freq=40.0,
        use_hilbert=True,
    )

    # --- MICROSTATE MAPS (UNCHANGED SENSOR SPACE) ---
    km_maps = run_modified_k_means(
        data[:, gfp_peaks],
        n_states=N_STATES,
    )  # (states × channels)

    # --- LOAD KOENIG TEMPLATES ---
    koenig_maps, koenig_chs = load_koenig_templates_from_set(
        TEMPLATE_FILE,
        N_STATES,
    )

    # --- MATCH & REORDER (NIKOLA-STYLE) ---
    km_maps_reordered, attribution = match_reorder_maps_with_templates(
        maps=km_maps,
        data_ch_names=raw.info["ch_names"],
        template_maps=koenig_maps,
        template_ch_names=koenig_chs,
    )

    # --- SAVE ---
    out_dir = OUT_DIR / subj_id
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / "km_maps.npy", km_maps)
    np.save(out_dir / "km_maps_reordered.npy", km_maps_reordered)
    np.save(out_dir / "microstate_permutation.npy", attribution)

    return km_maps_reordered, raw.info, subj_id, out_dir





# =========================
# RUN
# =========================
def load_lemon_subjects(n):
    sets = sorted(LEMON_DIR.glob("*_EC.set"))
    if len(sets) < n:
        raise RuntimeError("Not enough LEMON subjects")
    return random.sample(sets, n)


if __name__ == "__main__":
    subjects = load_lemon_subjects(N_SUBJECTS)

    for s in subjects:
        km_maps, info, subj_id, out_dir = process_subject(s)

        # 🔹 Plot here (explicitly in main)
        plot_microstate_maps(
            maps=km_maps,
            info=info,
            subject_id=subj_id,
            out_dir=out_dir,
        )

    print("\nAll subjects processed.")



