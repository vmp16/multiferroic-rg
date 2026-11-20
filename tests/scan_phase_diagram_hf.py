"""
scan_phase_diagram_HF.py
========================
Phase diagram in the (n, D) plane from the self-consistent Hartree-Fock solver (Coulomb Fock + layer Hartree).

Loop ordering
-------------
D is the OUTER loop: the bare arrays and the HF model are built once per D. The Coulomb kernels depend only on
the k-mesh and are built once for the whole scan. For every (D, n) several SCF runs from different starting points
are made and the lowest-energy converged solution wins:
    - warm start   : converged Sigma of the previous n (same D)
    - symmetric    : Sigma = 0
    - random seeds : Sigma_alpha = eps_alpha * 1 with random per-flavor shifts eps_alpha (amplitudes cycle through
                     AMPLITUDES)

Cost: measured on a test machine, one SCF iteration takes ~0.02 s (N = 101), ~0.08 s (N = 201) and ~0.16 s
(N = 267), and a run needs ~30-70 iterations. With N_SEEDS = 8 that is 10 runs per point, i.e. roughly
10-60 s per point depending on the mesh. The default 12 x 12 grid at dk = 2e-3 (N = 201) is therefore a
survey of a few hours; scale N_POINTS, D_POINTS and DK_SCAN with the time per row printed during the run.

Resolution: the k-mesh must resolve the smallest Fermi pocket (dk << k_F) and contain the largest one (k_F < K_LIM).
A flavor holding the whole density |n| = 5e12 cm^-2 has k_F ~ 0.18 (units of 1/a), hence K_LIM_SCAN = 0.20.
Check both with test_fock_convergence.py before trusting a scan.

Output
------
A compressed .npz in config.DATA_DIR (rewritten after every D row, so an interrupted scan keeps its results).
"""

import time
import numpy as np

import model.config as config
from model.analysis import (
    get_kmesh,
    build_coulomb_kernels_fft,
    build_hf_model,
    solve_scf_hf,
    random_seed_shifts,
    pauli_components,
    count_degenerate_flavors,
)

# ===================================================
#                  SCAN PARAMETERS
# ===================================================

# --------- Diagram axes ---------
N_MIN, N_MAX, N_POINTS = -2.0e12, -5.0e10, 30     # total carrier density [cm^-2]
D_MIN, D_MAX, D_POINTS = 0.000, 0.020, 30         # displacement field [eV]
# N_MAX = -5e11 instead of -4e10: the pockets at very low density are only a few mesh points wide at this dk.
# Extend the range only together with a finer DK.

# --------- Phase identification ---------
THRESHOLD = 1e6          # |n_alpha| above this counts as occupied [cm^-2]
PHASE_RTOL = 5e-2
PHASE_TO_INDEX = {label: i for i, label in enumerate(config.PHASE_ORDER)}

# --------- k-mesh (odd N = 2M+1, dk = K_LIM_SCAN / M) ---------
K_LIM_SCAN = 0.25
DK_SCAN = 1e-3

# --------- HF model ---------
REFERENCE = "cnp"        # fixed background subtracted from rho: "cnp" | "valence" | "half"

# --------- SCF parameters ---------
TOLERANCE = 1e-5         # [eV] on max|Sigma_new - Sigma_in|
MAX_ITER = 300
MIXING = 0.6
MU_MIN, MU_MAX = -0.25, 0.25

# --------- Seeding ---------
N_SEEDS = 3                              # random seeds per point (plus symmetric and warm start)
AMPLITUDES = (3e-3, 1e-2, 3e-2)         # [eV] seed-shift amplitudes, cycled over the random seeds
WARM_START = True                        # reuse the previous n solution as an extra start
SAME_ENERGY_RTOL = 1e-5                  # energies closer than this are the same solution

rng = np.random.default_rng(2024)
n_flavors = 4


def distinct_energies(energies, rtol=SAME_ENERGY_RTOL):
    """Sorted list of the distinct values in `energies` (relative tolerance)."""
    e = np.sort(np.asarray(energies, dtype=float))
    out = [e[0]]
    for x in e[1:]:
        if abs(x - out[-1]) > rtol * abs(out[-1]) + 1e-18:
            out.append(x)
    return out


# ===================================================
#                       GRIDS
# ===================================================

n_vals = np.linspace(N_MIN, N_MAX, N_POINTS)
D_vals = np.linspace(D_MIN, D_MAX, D_POINTS)

M_HALF = int(round(K_LIM_SCAN / DK_SCAN))
N_PTS_SCAN = 2 * M_HALF + 1
KX, KY = get_kmesh(K_LIM_SCAN, N_PTS_SCAN)
dk = KX[0, 1] - KX[0, 0]
CENTER = M_HALF                                     # index of k = 0

kernels = build_coulomb_kernels_fft(N_PTS_SCAN, dk, config.e2_over_eps, config.d_layers)

# Result containers: rows = D, cols = n
shape = (D_POINTS, N_POINTS)
phase_map = np.full(shape, -1, dtype=int)           # -1 = failed
phase_labels = np.full(shape, "", dtype=object)
energy_map = np.full(shape, np.nan)                 # eV/a^2
dE_next_map = np.full(shape, np.nan)                    # energy to the next distinct solution [eV/a^2]
n_distinct_map = np.zeros(shape, dtype=int)         # distinct converged solutions found
n_conv_map = np.zeros(shape, dtype=int)             # converged runs
density_map = np.full(shape + (n_flavors,), np.nan)
mu_map = np.full(shape, np.nan)
szh_map = np.full(shape, np.nan)                    # Hartree sigma_z [eV]
sigma0_center = np.full(shape + (n_flavors,), np.nan)   # Sigma_0(k=0) per flavor [eV]
sigmaz_center = np.full(shape + (n_flavors,), np.nan)   # Sigma_z^F(k=0) per flavor [eV]
sigma_row = np.full(shape + (n_flavors, 4, N_PTS_SCAN), np.nan)   # (S0, Sz, Re S12, Im S12) along ky = 0 [eV]
winner_kind = np.full(shape, "", dtype=object)      # which start gave the winning solution
conv_map = np.zeros(shape, dtype=bool)

out_path = config.DATA_DIR / f"phase_diagram_nD_HF_{N_POINTS}x{D_POINTS}pts_opt.npz"


def save():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        n_vals=n_vals, D_vals=D_vals, phase_map=phase_map, phase_labels=phase_labels,
        energy_map=energy_map, dE_next_map=dE_next_map, n_distinct_map=n_distinct_map, n_conv_map=n_conv_map,
        density_map=density_map, mu_map=mu_map, szh_map=szh_map,
        sigma0_center=sigma0_center, sigmaz_center=sigmaz_center, sigma_row=sigma_row,
        winner_kind=winner_kind, conv_map=conv_map,
        threshold=THRESHOLD, n_pts_scan=N_PTS_SCAN, k_lim_scan=K_LIM_SCAN, dk_scan=dk,
        reference=REFERENCE, tolerance=TOLERANCE, n_seeds=N_SEEDS,
        e2_over_eps=config.e2_over_eps, d_layers=config.d_layers, T_eff=config.T_eff,
    )


print("=" * 70)
print(f"  HF phase diagram scan: {N_POINTS} x {D_POINTS} = {N_POINTS * D_POINTS} points")
print(f"  n in [{N_MIN:.2e}, {N_MAX:.2e}] cm^-2   D in [{D_MIN*1e3:.1f}, {D_MAX*1e3:.1f}] meV")
print(f"  k-mesh: {N_PTS_SCAN}^2, dk = {dk:.2e}, K_LIM = {K_LIM_SCAN}   reference = {REFERENCE}")
print(f"  starts per point: {N_SEEDS} random + symmetric" + (" + warm" if WARM_START else ""))
print("=" * 70)

t_start = time.time()

# ===================================================
#               OUTER LOOP OVER D
# ===================================================

for iD, D in enumerate(D_vals):
    t_row = time.time()
    model = build_hf_model(D, KX, KY, config, reference=REFERENCE, kernels=kernels)

    # ===================================================
    #               INNER LOOP OVER n
    # ===================================================
    prev = None          # (Sigma, SzH) of the previous n, warm start

    for iN, n_target in enumerate(n_vals):

        # Candidate starts: (label, kwargs for solve_scf_hf)
        starts = [("sym", {})]
        if WARM_START and prev is not None:
            starts.append(("warm", dict(Sigma_init=prev[0], SzH_init=prev[1])))
        for i in range(N_SEEDS):
            amp = AMPLITUDES[i % len(AMPLITUDES)]
            starts.append((f"rand{i}", dict(seed_shifts=random_seed_shifts(n_flavors, amp, rng))))

        solutions = []                                  # (energy, label, result)
        for label, kw in starts:
            res = solve_scf_hf(model, n_target, mu_min=MU_MIN, mu_max=MU_MAX, max_iter=MAX_ITER,
                               tolerance=TOLERANCE, mixing=MIXING, verbose=False, **kw)
            if res is None or not res["converged"]:
                continue
            solutions.append((res["energy"], label, res))

        if not solutions:                               # no start converged
            prev = None
            continue

        solutions.sort(key=lambda s: s[0])
        best_E, best_label, best = solutions[0]
        distinct = distinct_energies([s[0] for s in solutions])

        partition_str, phase_idx = count_degenerate_flavors(
            best["n_flavs"], PHASE_TO_INDEX, threshold=THRESHOLD, rtol=PHASE_RTOL)
        phase_map[iD, iN] = phase_idx
        phase_labels[iD, iN] = partition_str
        energy_map[iD, iN] = best_E
        dE_next_map[iD, iN] = distinct[1] - distinct[0] if len(distinct) > 1 else np.nan
        n_distinct_map[iD, iN] = len(distinct)
        n_conv_map[iD, iN] = len(solutions)
        density_map[iD, iN] = best["n_flavs"]
        mu_map[iD, iN] = best["mu"]
        szh_map[iD, iN] = best["SzH"]
        for a in range(n_flavors):
            S0, _, _, Sz = pauli_components(best["Sigma"][a])
            sigma0_center[iD, iN, a] = S0[CENTER, CENTER]
            sigmaz_center[iD, iN, a] = Sz[CENTER, CENTER]
        S = best["Sigma"][:, :, :, CENTER, :]                              # (4, 2, 2, N)
        S12 = S[:, 0, 1]
        sigma_row[iD, iN] = np.stack([0.5 * (S[:, 0, 0] + S[:, 1, 1]).real,
                                    0.5 * (S[:, 0, 0] - S[:, 1, 1]).real,
                                    S12.real, S12.imag], axis=1)
        winner_kind[iD, iN] = best_label
        conv_map[iD, iN] = True
        prev = (best["Sigma"], best["SzH"])

    n_ok = np.count_nonzero(conv_map[iD])
    print(
        f"  D = {D*1e3:+7.2f} meV  |  {n_ok:3d}/{N_POINTS} converged  |  "
        f"phases: {np.unique(phase_map[iD][phase_map[iD] >= 0])}  |  "
        f"{time.time() - t_row:.1f} s"
    )
    save()

# ===================================================
#                       SUMMARY
# ===================================================

n_failed = np.count_nonzero(phase_map < 0)
print("\n" + "=" * 70)
print(f"  Saved: {out_path}")
print(f"  Failed points: {n_failed}/{phase_map.size}")
print(f"  Phases found: {sorted(np.unique(phase_map[phase_map >= 0]).tolist())}")
print(f"  Points with several distinct solutions: {np.count_nonzero(n_distinct_map > 1)}/{phase_map.size}")
print(f"  Total time: {time.time() - t_start:.1f} s")
print("=" * 70)