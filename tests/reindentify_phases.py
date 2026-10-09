"""
File to re-run the phase identification with different parameters than those used during the scan.
"""
import numpy as np
import os

import model.config as config
from model.analysis import (
    count_degenerate_flavors,
)

# -------- Load Data File --------
filename = "phase_diagram_nD_170pts.npz"

with np.load(config.DATA_DIR / filename, allow_pickle=True) as f:
    data = {key: f[key] for key in f.files}

n_vals = data["n_vals"]
n_pts = len(n_vals)
D_vals = data["D_vals"]
density_map = data["density_map"]
phase_map = data["phase_map"]
phase_labels = data["phase_labels"]

new_phase_map = phase_map
new_phase_labels = phase_labels

# -------- Phase Identification Params --------
THRESHOLD = 1e6      # |n_alpha| above this counts as occupied [cm^-2]
PHASE_RTOL = 1e-3    # defining equal densities
PHASE_TO_INDEX = {label: i for i, label in enumerate(config.PHASE_ORDER)}

# -------- Loop Over the Grid Values ---------
for iD, D in enumerate(D_vals):
    for iN, n in enumerate(n_vals):
        partition_str, phase_idx = count_degenerate_flavors(density_map[iD, iN], PHASE_TO_INDEX, threshold=THRESHOLD, rtol=PHASE_RTOL)
        new_phase_map[iD, iN] = phase_idx
        new_phase_labels[iD, iN] = partition_str

# ------- Replace Phase Maps ---------
data["phase_map"] = new_phase_map
data["phase_labels"] = new_phase_labels

# ------- Save New Data --------
tmp_path = config.DATA_DIR / (filename + "tmp.npz")
np.savez_compressed(tmp_path, **data)
os.replace(tmp_path, config.DATA_DIR / filename)