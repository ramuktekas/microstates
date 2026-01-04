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
from microstates import (
    MicrostateTemplates,
    KoenigTemplates,          # or CustoTemplates
    match_topomaps_per_template,
)
# =========================
# USER CONFIG
# =========================
LEMON_DIR = Path("/store/projects/kumarsak/LEMON_data")
TEMPLATE_FILE = Path("/home/kumarsak/micro-VAR-states/python/templates/Koenig2002.set")
OUT_DIR = Path("./microstate_results")

N_SUBJECTS = "all"
N_STATES = 4
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
OUT_DIR.mkdir(exist_ok=True)


# =========================
# UTILITIES
# =========================

def load_lemon_subjects(n_subjects):
    sets = sorted(LEMON_DIR.glob("*_EC.set"))

    if n_subjects == "all":
        return sets

    if isinstance(n_subjects, int):
        return sets[:n_subjects]

    raise ValueError(
        "n_subjects must be an integer or the string 'all'"
    )







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

import numpy as np
import mne

from microstates import get_gfp_peaks


def preprocess_for_gfp_peaks(
    raw,
    l_freq=1.0,
    h_freq=40.0,
    do_ica=False,
    random_state=42,
):
    """
    Classical EEGLAB-style preprocessing for microstates (Python version)

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG data
    l_freq : float
        Low cutoff frequency (Hz)
    h_freq : float
        High cutoff frequency (Hz)
    do_ica : bool
        Whether to run ICA cleaning
    random_state : int
        Random seed for ICA reproducibility

    Returns
    -------
    gfp_peaks : np.ndarray
        Indices of GFP peaks
    gfp_curve : np.ndarray
        GFP time series
    raw : mne.io.Raw
        Preprocessed EEG (important!)
    """

    # --------------------------------------------------
    # 1. Average reference (MANDATORY)
    # --------------------------------------------------
    raw = raw.copy().set_eeg_reference("average", projection=False)

    # --------------------------------------------------
    # 2. Band-pass filter (zero-phase FIR)
    # --------------------------------------------------
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )

    # --------------------------------------------------
    # 3. ICA cleaning (OPTIONAL)
    # --------------------------------------------------
    if do_ica:
        ica = mne.preprocessing.ICA(
            n_components=None,
            method="fastica",
            random_state=random_state,
            max_iter="auto",
        )
        ica.fit(raw)

        # Try ICLabel-style classification (if available)
        try:
            labels = ica.get_components()
            ic_labels = mne.preprocessing.iclabel.label_components(raw, ica)
            bad_idx = np.where(
                np.isin(
                    ic_labels["labels"],
                    ["eye", "muscle", "heart", "line_noise", "channel_noise"],
                )
            )[0]
            ica.exclude = list(bad_idx)
        except Exception:
            # fallback: no automatic rejection
            pass

        raw = ica.apply(raw.copy())

    # --------------------------------------------------
    # 4. GFP peaks on voltage data (NO Hilbert!)
    # --------------------------------------------------
    data = raw.get_data()  # channels × time

    gfp_peaks, gfp_curve = get_gfp_peaks(data)

    return gfp_peaks, gfp_curve, raw



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

def subject_maps_to_raw(km_maps, info):
    """
    Convert microstate maps (states × channels)
    into an MNE RawArray (channels × states),
    where each 'time point' is one microstate.
    """
    return mne.io.RawArray(km_maps.T, info, verbose=False)


# =========================
# MAIN PIPELINE
# =========================
def process_subject(set_path):
    subj_id = set_path.stem
    print(f"\n=== Processing {subj_id} ===")

    # --- Load EEG ---
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    

    # --- GFP peaks (preprocessing only for peak detection) ---
    gfp_peaks, gfp_curve, raw = preprocess_for_gfp_peaks(
        raw,
        l_freq=1,
        h_freq=40,
        do_ica=True,   # set False if you want speed
    )
    data = raw.get_data()
    # --- Microstate maps in SENSOR SPACE ---
    km_maps = run_modified_k_means(
        data[:, gfp_peaks],
        n_states=N_STATES,
    )  # (states × channels)

    # --- Wrap subject maps as Raw ---
    subject_maps_raw = subject_maps_to_raw(km_maps, raw.info)

    # --- Load templates ---
    templates = KoenigTemplates.load(n_states=N_STATES)
    # or: templates = CustoTemplates.load()

    # --- Match using Nikola’s function ---
    attribution, corr = match_topomaps_per_template(
        maps=subject_maps_raw,
        template=templates,
    )

    # --- Apply permutation to FULL maps ---
    km_maps_labeled = km_maps[attribution, :]

    # --- Save ---
    out_dir = OUT_DIR / subj_id
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / "km_maps.npy", km_maps)
    np.save(out_dir / "km_maps_labeled.npy", km_maps_labeled)
    np.save(out_dir / "microstate_permutation.npy", attribution)
    np.save(out_dir / "template_correlations.npy", corr)

    return km_maps_labeled, raw.info, subj_id, out_dir






# =========================
# RUN
# =========================

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







