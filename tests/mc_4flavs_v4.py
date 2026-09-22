"""
Calculates the 4 flavor configuration minimizing the total internal energy using a Monte-Carlo approach.

Version 4:
    - MODULAR: all MC logic lives in model.analysis.
    - Interaction function is injected as a pre-configured callable (functools.partial), so swapping interaction forms requires changing only one line.
"""
import numpy as np
from functools import partial
import time

from model.model import McCannCarts
import model.config as config
from model.analysis import (
    get_kmesh,
    precompute_flavor_bands,
    V_int_SU4,
    V_int_anisotropic,
    sample_flavor_densities,
    build_density_interp_tables,
    evaluate_realizations,
    find_minimum_energy_configuration,
)

# ----------- Reproducibility -----------
rng = np.random.default_rng(41)

# ----------- Physical Setup -----------
n_target = -3e11  # total charge carrier density  [cm^-2]
n_flavors = 2

# Order : (K, ↑) (K′, ↑) (K, ↓) (K′, ↓)
systems = [
    McCannCarts(
        N=config.N, valley_idx=xi, Delta=0, #delta,
        gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
        gamma3=config.GAMMA3, gamma4=config.GAMMA4
    )
    for xi in config.VALLEY_IDX
    # for xi, delta in zip(2*config.VALLEY_IDX, config.DELTAS.T.flatten())
]

# ----------- Grid -----------
KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
dk = KX[0, 1] - KX[0, 0]
prefactor = (dk**2) / (2 * np.pi)**2

# ----------- Precompute Energy Bands --------
print(f"Precomputing energy bands for {n_flavors} active flavors...")
all_bands = precompute_flavor_bands(systems, KX, KY)

# ----------- Interaction Potential -----------
v_int = partial(V_int_SU4, U=config.U, area_uc=config.area_uc)

# ----------- Monte-Carlo Params -----------
m_seed = 5.0
num_picks = 30000
mu_min = -1.0   # [eV]
mu_max = 1.0    # [eV]
N_TABLE = 3000

# ======================= MAIN =======================

t0 = time.time()

# 1. Sample random density configurations
print(f"\nBuilding {num_picks} random configurations...")
n_flavs_seed = sample_flavor_densities(n_target, n_flavors, num_picks, m_seed, rng=rng)
print(f"   Density range: [{np.min(n_flavs_seed):.4e}, {np.max(n_flavs_seed):.4e}] cm^-2")

# 2. Build per-flavor mu(n) interpolation tables
print(f"\nBuilding mu_flav(n) interpolation tables ({N_TABLE} points)...")

mu_table, n_table = build_density_interp_tables(
    all_bands,
    config.T_eff,
    prefactor,
    config.unit_cell_to_cm2,
    mu_min=mu_min,
    mu_max=mu_max,
    n_table_pts=N_TABLE
)

print(f"   Density range covered: [{np.min(n_table):.4e}, {np.max(n_table):.4e}] cm^-2")

t1 = time.time()
print(f"   Tables built in {t1-t0:.2f} s")

# 3. Evaluate all realizations
print(f"\nEvaluating {num_picks} realizations x {n_flavors} flavors...")
mu_flav_arr, E_flav_arr, V_int_vec, valid_mask = evaluate_realizations(
    n_flavs_seed,
    all_bands,
    mu_table,
    n_table,
    config.T_eff,
    prefactor,
    config.unit_cell_to_cm2,
    v_int
)

t2 = time.time()
print(f"   All realizations evaluated in {t2 - t1:.2f} s")

# 4. Find minimal energy configuration
n_sol, mu_sol, E_int_min = find_minimum_energy_configuration(
    n_flavs_seed,
    mu_flav_arr,
    E_flav_arr,
    V_int_vec,
    valid_mask
)

# ======================= RESULTS =======================
t3 = time.time()

print("\nRESULTS")
print(f"    Minimum internal energy: {E_int_min:.6e} eV·cm^-2")
print(f"    SOLUTION n_alpha (cm^-2): {np.array2string(n_sol, precision=4, floatmode='fixed')}")
print(f"    SOLUTION mu_alpha (meV):  {np.array2string(mu_sol*1e3, precision=4, floatmode='fixed')}")
print(f"\nTotal calculation time: {t3 - t0:.2f} s")