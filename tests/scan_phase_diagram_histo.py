"""
scan_phase_diagram.py
=====================
Build a phase diagram in the (n, D) plane.

Strategy
--------
At every grid point a single SCF is run with ALL FOUR flavors active.
The phase is then read off from the converged solution by counting how
many flavor densities exceed a threshold in magnitude.  Flavors that are
energetically inaccessible empty themselves out during the iteration, so
no separate phase-by-phase comparison is needed.

Loop ordering
-------------
The bands depend on D only, so D is the OUTER loop: bands and the
mu-interpolation tables are rebuilt once per D value and reused for every
n in the inner loop.  This is what makes the scan affordable.

Symmetry breaking
-----------------
The symmetric configuration n_alpha = n_target/4 is an exact fixed point
of the SU(4)-symmetric SCF: starting there, the solution never polarizes.
Every point is therefore seeded with a randomly perturbed configuration,
and optionally with several independent seeds whose lowest-energy result
is kept.

Output
------
A compressed .npz in config.DATA_DIR containing the grids, the flavor
counts, the full converged densities and the internal energies.
"""

import numpy as np
import time
from itertools import combinations

import model.config as config
from model.analysis import (
    get_kmesh,
    build_systems_with_displacement,
    precompute_flavor_bands,
    build_tables_histogram,
    solve_scf_from_tables,
    internal_energy_from_tables,
    count_occupied_flavors,
)

# ======================================================================
#  SCAN PARAMETERS
# ======================================================================

# --- diagram axes ---
N_MIN, N_MAX, N_POINTS = -5.0e11, -1.0e11, 30    # carrier density [cm^-2]
D_MIN, D_MAX, D_POINTS = -0.010,  0.010,   30    # displacement field [eV]

# --- phase identification ---
THRESHOLD = 1e6      # |n_alpha| above this counts as occupied [cm^-2]

# --- k-mesh ---
# Deliberately coarser than config.N_PTS: the scan runs this thousands of
# times, and the table construction cost scales as N_PTS^2.  Verify on a
# few isolated points that the phase boundaries are converged in N_PTS
# before trusting a low-resolution scan.
N_PTS_SCAN = 1500

# --- interpolation tables ---
MU_MIN, MU_MAX = -0.30, 0.30    # must cover mu_global - V_alpha everywhere
N_TABLE = 3000

# --- Histogram ---
N_BINS = 600000

# --- SCF ---
TOLERANCE = 1e-6
MAX_ITER  = 300
MIXING    = 0.4

# --- seeding ---
N_SEEDS   = 4        # independent perturbed seeds per point; best energy wins
AMPLITUDE = 2.5     # fractional perturbation around the symmetric point
WARM_START = True    # reuse previous n solution as an extra seed

rng = np.random.default_rng(2024)

def phase_template_seeds(n_target, n_flavors=4, epsilon=1e-3, rng=None):
    """
    Generate one seed per distinct flavor-occupation pattern.
 
    The SU(4) mean-field energy landscape is concave in the polarization
    direction, so a seed near the symmetric point flows either to a corner
    of the density simplex (full polarization) or stays symmetric.  Partial
    polarization lives in basins that near-symmetric seeds never enter.
    This function places a seed inside every basin explicitly: for each
    subset of k flavors (k = 1..n_flavors), the charge is shared equally
    among that subset and the rest start near zero.
 
    Empty flavors are given a small nonzero density (epsilon * n_target /
    n_flavors) rather than exactly zero, so that degenerate empty flavors
    are not perfectly tied and can separate if the solution wants them to.
 
    Parameters
    ----------
    n_target : float
        Total carrier density [cm^-2].
    n_flavors : int
        Total number of flavors (4 here).
    epsilon : float
        Relative density given to nominally empty flavors.
    rng : numpy.random.Generator or None
        If given, a small random jitter is added to break residual ties
        between equivalent flavors.
 
    Returns
    -------
    seeds : list of ndarray, each shape (n_flavors,)
        2**n_flavors - 1 = 15 seeds for n_flavors = 4, each summing
        exactly to n_target.
    """
    seeds = []
    n_sym = n_target / n_flavors
 
    for k in range(1, n_flavors + 1):
        for combo in combinations(range(n_flavors), k):
            s = np.full(n_flavors, epsilon * n_sym)
            s[list(combo)] = n_target / k
            if rng is not None:
                s = s * (1.0 + 0.02 * rng.uniform(-1.0, 1.0, n_flavors))
            seeds.append(s * (n_target / s.sum()))   # enforce the constraint
 
    return seeds


def rescale_seed(n_prev, n_target):
    """Rescale a previous solution to satisfy the new density constraint."""
    s = n_prev.sum()
    if not np.isfinite(s) or s == 0.0:
        return None
    return n_prev * (n_target / s)


# ======================================================================
#  GRIDS
# ======================================================================

n_vals = np.linspace(N_MIN, N_MAX, N_POINTS)
D_vals = np.linspace(D_MIN, D_MAX, D_POINTS)
n_flavors = 4

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

# ======================================================================
#  OUTER LOOP OVER D  (bands + tables rebuilt here only)
# ======================================================================

for iD, D in enumerate(D_vals):
    t_row = time.time()

    systems, base_deltas = build_systems_with_displacement(D, config)
    all_bands = precompute_flavor_bands(systems, KX, KY)

    try:
        mu_table, n_table, e_table = build_tables_histogram(
            all_bands, config.T_eff, prefactor, config.unit_cell_to_cm2,
            mu_min=MU_MIN, mu_max=MU_MAX, n_table_pts=N_TABLE, n_bins=N_BINS,
        )
    except RuntimeError as exc:
        print(f"  [D = {D*1e3:+.2f} meV] table build failed: {exc}")
        continue

    del all_bands, systems      # free the band arrays before the inner loop

    # ------------------------------------------------------------------
    #  INNER LOOP OVER n  (pure interpolation, no k-sums)
    # ------------------------------------------------------------------
    n_prev = None

    for iN, n_target in enumerate(n_vals):

        # Assemble the candidate seeds for this point
        seeds = [
            phase_template_seeds(n_target, n_flavors, AMPLITUDE, rng)
            for _ in range(N_SEEDS)
        ]
        if WARM_START and n_prev is not None:
            warm = rescale_seed(n_prev, n_target)
            if warm is not None:
                seeds.append(warm)

        best = None
        best_E = np.inf

        for seed in seeds:
            res = solve_scf_from_tables(
                n_target, mu_table, n_table,
                config.U, config.area_uc, seed,
                max_iter=MAX_ITER, tolerance=TOLERANCE, mixing=MIXING,
            )
            if res is None or not res["converged"]:
                continue

            E = internal_energy_from_tables(
                res["n_flavs"], res["mu_flavs"],
                mu_table, e_table, config.U, config.area_uc,
            )
            if E < best_E:
                best_E, best = E, res

        if best is None:
            n_prev = None
            continue

        phase_map[iD, iN]     = count_occupied_flavors(best["n_flavs"], THRESHOLD)
        energy_map[iD, iN]    = best_E
        density_map[iD, iN]   = best["n_flavs"]
        mu_map[iD, iN]        = best["mu_global"]
        conv_map[iD, iN]      = True
        n_prev = best["n_flavs"]

    n_ok = np.count_nonzero(conv_map[iD])
    print(
        f"  D = {D*1e3:+7.2f} meV  |  {n_ok:3d}/{N_POINTS} converged  |  "
        f"phases: {np.unique(phase_map[iD][phase_map[iD] >= 0])}  |  "
        f"{time.time() - t_row:.1f} s"
    )

# ======================================================================
#  SAVE
# ======================================================================

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
out_path = config.DATA_DIR / "phase_diagram_histo.npz"

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