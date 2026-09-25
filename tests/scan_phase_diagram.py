"""
scan_phase_diagram.py
=====================
Build a phase diagram in the (n, D) plane.

Loop ordering
-------------
The bands depend on D only, so D is the OUTER loop: bands and the
mu-interpolation tables are rebuilt once per D value and reused for every
n in the inner loop.

Output
------
A compressed .npz in config.DATA_DIR containing the grids, the flavor
counts, the full converged densities and the internal energies.
"""

import numpy as np
import time

import model.config as config
from model.analysis import (
    get_kmesh,
    build_systems_with_displacement,
    precompute_flavor_bands,
    build_density_and_energy_tables,
    solve_scf_from_tables,
    internal_energy_from_tables,
    count_occupied_flavors,
)

# ===================================================
#                  SCAN PARAMETERS
# ===================================================

# --------- Diagram Axes ---------
N_MIN, N_MAX, N_POINTS = -5.0e12, -1.0e10, 150    # carrier density [cm^-2]
D_MIN, D_MAX, D_POINTS = 0.000, 0.050, 150    # displacement field [eV]
# TIP: symmetric in D, so we can just test positive values

# --------- Phase Identification ---------
THRESHOLD = 1e6      # |n_alpha| above this counts as occupied [cm^-2]

# --------- Grid (k-mesh) ---------
# Deliberately coarser than config.N_PTS: the scan runs this thousands of times, and the table construction cost scales as N_PTS^2.
N_PTS_SCAN = 2000
K_LIM_SCAN = 0.15
# Verify on a few isolated points that the phase boundaries are converged in N_PTS before trusting a low-resolution scan.

# --------- Interpolation Tables ---------
MU_MIN, MU_MAX = -0.30, 0.30    # must cover mu_global - V_alpha everywhere
N_TABLE = 3000

# --------- SCF Parameters ---------
TOLERANCE = 1e-6
MAX_ITER  = 300
MIXING    = 0.4

# --------- Seeding ---------
N_SEEDS   = 4        # independent perturbed seeds per point; best energy wins
AMPLITUDE = 2.5     # fractional perturbation around the symmetric point
WARM_START = True    # reuse previous n solution as an extra seed

rng = np.random.default_rng(2024)


def perturb_symmetric_seed(n_target, n_flavors, amplitude, rng):
    """Random seed off the symmetric point, summing exactly to n_target."""
    n_sym = n_target / n_flavors
    raw = n_sym + rng.uniform(-amplitude, amplitude, n_flavors) * abs(n_sym)
    return raw * (n_target / raw.sum())


def rescale_seed(n_prev, n_target):
    """Rescale a previous solution to satisfy the new density constraint."""
    s = n_prev.sum()
    if not np.isfinite(s) or s == 0.0:
        return None
    return n_prev * (n_target / s)


# ===================================================
#                       GRIDS
# ===================================================

# Phase diagram grid
n_vals = np.linspace(N_MIN, N_MAX, N_POINTS)
D_vals = np.linspace(D_MIN, D_MAX, D_POINTS)
n_flavors = 4

# k-space grid
KX, KY = get_kmesh(config.K_LIM, N_PTS_SCAN)
dk = KX[0, 1] - KX[0, 0]
prefactor = (dk**2) / (2.0 * np.pi)**2

# Result containers: rows = D, cols = n
phase_map  = np.full((D_POINTS, N_POINTS), -1, dtype=int)     # -1 = failed
energy_map = np.full((D_POINTS, N_POINTS), np.nan)
density_map = np.full((D_POINTS, N_POINTS, n_flavors), np.nan)
mu_map     = np.full((D_POINTS, N_POINTS), np.nan)
conv_map   = np.zeros((D_POINTS, N_POINTS), dtype=bool)

print("=" * 70)
print(f"  Phase diagram scan: {N_POINTS} x {D_POINTS} = {N_POINTS*D_POINTS} points")
print(f"  n in [{N_MIN:.2e}, {N_MAX:.2e}] cm^-2")
print(f"  D in [{D_MIN*1e3:.1f}, {D_MAX*1e3:.1f}] meV")
print(f"  k-mesh: {N_PTS_SCAN}^2   threshold: {THRESHOLD:.1e} cm^-2")
print("=" * 70)

t_start = time.time()

# ===================================================
#               OUTER LOOP OVER D
#       (bands + tables rebuilt here only)
# ===================================================

for iD, D in enumerate(D_vals):
    t_row = time.time()

    # Get the energy bands
    systems, base_deltas = build_systems_with_displacement(D, config)
    all_bands = precompute_flavor_bands(systems, KX, KY)

    # Build the tables for interpolation
    try:
        mu_table, n_table, e_table = build_density_and_energy_tables(
            all_bands, config.T_eff, prefactor, config.unit_cell_to_cm2,
            mu_min=MU_MIN, mu_max=MU_MAX, n_table_pts=N_TABLE,
        )
    except RuntimeError as exc:
        print(f"  [D = {D*1e3:+.2f} meV] table build failed: {exc}")
        continue

    del all_bands, systems      # free the band arrays before the inner loop

    # ===================================================
    #               INNER LOOP OVER n
    #       (pure interpolation, no k-sums)
    # ===================================================
    n_prev = None   # seed for the following iteration

    for iN, n_target in enumerate(n_vals):

        # Assemble the candidate seeds for this point
        seeds = [
            perturb_symmetric_seed(n_target, n_flavors, AMPLITUDE, rng)
            for _ in range(N_SEEDS)
        ]
        if WARM_START and n_prev is not None:
            warm = rescale_seed(n_prev, n_target)
            if warm is not None:
                seeds.append(warm)

        best = None         # contains the whole result
        best_E = np.inf

        for seed in seeds:
            res = solve_scf_from_tables(
                n_target, mu_table, n_table,
                config.U, config.area_uc, seed,
                max_iter=MAX_ITER, tolerance=TOLERANCE, mixing=MIXING,
            )
            if res is None or not res["converged"]:
                # Pass to the next seed if this one didn't converged, without calculating the internal energy
                continue

            E = internal_energy_from_tables(
                res["n_flavs"], res["mu_flavs"],
                mu_table, e_table, config.U, config.area_uc,
            )
            if E < best_E:
                best_E, best = E, res

        if best is None:    # only if no seed converged
            n_prev = None
            continue

        # Update the result containers
        phase_map[iD, iN]     = count_occupied_flavors(best["n_flavs"], THRESHOLD)
        energy_map[iD, iN]    = best_E
        density_map[iD, iN]   = best["n_flavs"]
        mu_map[iD, iN]        = best["mu_global"]
        conv_map[iD, iN]      = True
        n_prev = best["n_flavs"]

    # Print information for this row (one D value)
    # listing the different phases crossed on this row
    n_ok = np.count_nonzero(conv_map[iD])  # count the number of converged points on this row
    print(
        f"  D = {D*1e3:+7.2f} meV  |  {n_ok:3d}/{N_POINTS} converged  |  "
        f"phases: {np.unique(phase_map[iD][phase_map[iD] >= 0])}  |  "
        f"{time.time() - t_row:.1f} s"
    )

# ===================================================
#                       SAVE
# ===================================================

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
out_path = config.DATA_DIR / f"phase_diagram_nD_{N_POINTS}pts.npz"

np.savez_compressed(
    out_path,
    n_vals=n_vals,
    D_vals=D_vals,
    phase_map=phase_map,
    energy_map=energy_map,
    density_map=density_map,
    mu_map=mu_map,
    conv_map=conv_map,
    threshold=THRESHOLD,
    n_pts_scan=N_PTS_SCAN,
    k_lim_scan=K_LIM_SCAN,
    U=config.U,
    T_eff=config.T_eff,
)

n_failed = np.count_nonzero(phase_map < 0)
print("\n" + "=" * 70)
print(f"  Saved: {out_path}")
print(f"  Failed points: {n_failed}/{phase_map.size}")
print(f"  Phases found: {sorted(np.unique(phase_map[phase_map >= 0]).tolist())}")
print(f"  Total time: {time.time() - t_start:.1f} s")
print("=" * 70)