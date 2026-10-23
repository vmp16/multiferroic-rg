"""
build_tables_scan.py
====================
Precompute and cache the mu / n / E_kin interpolation tables for a grid of displacement-field values D. One .npz file is written per D value.

Run this ONCE. Afterwards, scan_phase_diagram_PIP.py can call get_or_build_tables(...) with the SAME parameters and it will just load the files (milliseconds) instead of redoing the k-sums (minutes).

Features
--------
- Resumable: D values that already have a valid file are skipped, so you can stop the script at any time and re-run it to continue.
- Extendable: add more D values later (or widen D_MIN/D_MAX) and only the new ones are computed, as long as the D values coincide with old ones.
- Force a rebuild of everything with: python build_tables_scan.py --rebuild

IMPORTANT
---------
Every parameter in the "TABLE PARAMETERS" block below must be IDENTICAL to the one used later in the phase-diagram scan (k-mesh, mu range, number of table points). If any of them differs, the cache key differs and the phase-diagram scan will not find these files and will rebuild from scratch. The safest approach is to move this block into config.py and import it from both scripts.
"""

import argparse
import time

import numpy as np

import model.config as config
from model.analysis import get_kmesh, get_or_build_tables

# ===================================================
#                TABLE PARAMETERS
# ===================================================

# --------- D grid ---------
D_MIN, D_MAX, D_POINTS = 0.000, 0.070, 170     # displacement field [eV]
# TIP: the physics is symmetric in D, so only positive values are needed. The D grid must be built with np.linspace and the SAME numbers in the phase-diagram scan, otherwise the D values (and the keys) will not match.

# --------- k-mesh ---------
N_PTS_SCAN = 5000
K_LIM_SCAN = 0.20

# --------- mu grid of the tables ---------
MU_MIN, MU_MAX = -0.20, 0.20     # must cover mu_global - V_alpha everywhere
N_TABLE = 3000

# --------- Output ---------
CACHE_DIR = config.DATA_DIR / "tables_cache"

# ===================================================
#                   COMMAND LINE
# ===================================================

parser = argparse.ArgumentParser(description="Precompute mu/n/E_kin tables for a grid of D values.")
parser.add_argument("--rebuild", action="store_true",
                    help="ignore existing files and recompute everything")
args = parser.parse_args()

# ===================================================
#                      GRIDS
# ===================================================

D_vals = np.linspace(D_MIN, D_MAX, D_POINTS)

# The k-mesh is built ONCE and reused for every D (the bands depend on D, the mesh does not). NOTE: K_LIM_SCAN is used here, so the mesh and the cache key are consistent.
KX, KY = get_kmesh(K_LIM_SCAN, N_PTS_SCAN)

print("=" * 70)
print(f"  Building tables for {D_POINTS} values of D")
print(f"  D in [{D_MIN * 1e3:.2f}, {D_MAX * 1e3:.2f}] meV")
print(f"  k-mesh: {N_PTS_SCAN}^2, k_lim = {K_LIM_SCAN}")
print(f"  mu in [{MU_MIN}, {MU_MAX}] eV, {N_TABLE} points")
print(f"  cache directory: {CACHE_DIR}")
print(f"  rebuild existing files: {args.rebuild}")
print("=" * 70)

# ===================================================
#                   MAIN LOOP OVER D
# ===================================================

t_start = time.time()
failed = []          # D values whose table construction raised an error

for iD, D in enumerate(D_vals):
    t_row = time.time()
    print(f"[{iD + 1:3d}/{D_POINTS}]", end="")

    try:
        # Loads the file if it exists (and is valid), builds + saves it
        # otherwise. We do not need the returned arrays here: the point of
        # this script is just to make sure the file exists on disk.
        get_or_build_tables(
            D, config, KX, KY, cache_dir=CACHE_DIR,
            mu_min=MU_MIN, mu_max=MU_MAX, n_table_pts=N_TABLE,
            rebuild=args.rebuild, verbose=True,
        )
    except RuntimeError as exc:
        # Raised by the table builders when n(mu) is not strictly increasing.
        # Skip this D, report it at the end, and keep going with the others.
        print(f"  [D = {D * 1e3:.4f} meV] FAILED: {exc}")
        failed.append(D)
        continue

    # Rough time estimate for the remaining D values
    elapsed = time.time() - t_start
    done = iD + 1
    eta = elapsed / done * (D_POINTS - done)
    print(f"         {time.time() - t_row:6.1f} s   |   elapsed {elapsed / 60:6.1f} min   |   ETA {eta / 60:6.1f} min")

# ===================================================
#                      SUMMARY
# ===================================================

n_files = len(list(CACHE_DIR.glob("tables_D*.npz")))
print("\n" + "=" * 70)
print(f"  Done in {(time.time() - t_start) / 60:.1f} min")
print(f"  Table files currently in cache: {n_files}")
print(f"  Failed D values: {len(failed)}")
for D in failed:
    print(f"      D = {D * 1e3:.4f} meV")
print("=" * 70)