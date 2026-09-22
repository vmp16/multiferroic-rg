"""
Calculates the 4 flavor configuration minimizing the total grand potential using a Monte-Carlo approach.
VERSION 2:
    - All 4 densities in the same range
    - Loop over the number of realizations
    - Interpolation table to find the flavor Fermi levels mu_flavs
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
import time

from model.model import McCannCarts
import model.config as config
from model.analysis import get_kmesh, fermi_distrib, precompute_flavor_bands, V_int_anisotropic, get_flavor_density_from_bands, get_flavor_kinetic_energy_from_bands

# -------- System & Target Setup --------
np.random.seed(41)

n_target = -3e11  # total charge carrier density in /cm^2
n_flavors = 4

systems = [
    McCannCarts(
        N=config.N, valley_idx=xi, Delta=0, #delta,
        gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
        gamma3=config.GAMMA3, gamma4=config.GAMMA4
    )
    for xi, delta in zip(2*config.VALLEY_IDX, config.DELTAS.T.flatten())
]

# -------- Grid --------
KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
dk = KX[0, 1] - KX[0, 0]
prefactor = (dk**2) / (2 * np.pi)**2

# -------- Precompute Energy Bands --------
print("Precomputing energy bands for all active flavors...")
all_bands = precompute_flavor_bands(systems, KX, KY)

# -------- Params Initialization --------
m_seed = 0.1
num_picks = int(20)

# Pick 4 random densities around the symmetric distribution
raw = np.random.uniform(1 - m_seed, 1 + m_seed, (n_flavors, num_picks)) * (n_target / n_flavors)
# Normalize it to ensure sum(n_alpha) = n_target
n_flavs_seed = raw * (n_target / raw.sum(axis=0))

# -------- Bracket & Table Params for Interpolation --------
mu_min = -0.1
mu_max = 0.1

N_TABLE = 100      # increase for higher precision

# -------- Build per-flavor mu(n) Interpolation Tables --------
print(f"Building mu(n) interpolation tables for all flavors for {N_TABLE} points...")
t0 = time.time()
mu_table = np.linspace(mu_min, mu_max, N_TABLE)
n_table = np.zeros((n_flavors, N_TABLE))        # n_flav(mu) per flavor, shape (4, N_TABLE)

for a, flav_bands in enumerate(all_bands):
    for j, mu_j in enumerate(mu_table):
        n_table[a, j] = get_flavor_density_from_bands(
            flav_bands, mu_j, config.T_eff, prefactor, config.unit_cell_to_cm2
        )
    # Sanity check: n_flav(mu) must be strictly monotone for np.interp to be valid.
    # A non-monotone table would silently produce wrong results.
    if not np.all(np.diff(n_table[a]) > 0):
        raise RuntimeError(
            f"Flavor {a}: n_flav(mu) is not strictly increasing on [{mu_min}, {mu_max}]. "
            "Widen the bracket or check the band structure."
        )

print(f"  Density range covered per flavor:")
for a in range(n_flavors):
    print(f"    Flavor {a}: [{n_table[a, 0]:.3e}, {n_table[a, -1]:.3e}] cm^-2")

t1 = time.time()
print(f"Time for building the interpolation tables: {t1 - t0:.2f} s")


# -------- Lookup mu and E_kin for all realizations --------
# valid_mask -> tells us what values are not within the range of the interpolation tables, because np.interp() does not raise an error when this happens.

print(f"\nLooking up mu and E_kin for {num_picks} realizations x {n_flavors} flavors...")

mu_flav_arr = np.full((num_picks, n_flavors), np.nan)
E_flav_arr = np.full((num_picks, n_flavors), np.nan)
V_int_vec = np.full(num_picks, np.nan)
valid_mask = np.ones(num_picks, dtype=bool)     # all True

for a, flav_bands in enumerate(all_bands):
    n_flav_targets = n_flavs_seed[a]    # shape (num_picks,): all the realizations for one flavor

    # Identify realizations where n_flav_target is out of the table range
    out_of_range = (n_flav_targets < n_table[a, 0]) | (n_flav_targets > n_table[a, -1])
    valid_mask &= ~out_of_range

    # Invert n_flav(mu) through the interpolation
    mu_flav_arr[:, a] = np.interp(n_flav_targets, n_table[a], mu_table)

# Compute the kinetic energy per flavor for the valid realizations
for i in np.where(valid_mask)[0]:       # list of indices of the valid realizations
    for a, flav_bands in enumerate(all_bands):
        E_flav_arr[i, a] = get_flavor_kinetic_energy_from_bands(
            flav_bands, mu_flav_arr[i, a], config.T_eff, prefactor, config.unit_cell_to_cm2
        )

    V_int_vec[i] = V_int_anisotropic(
        n_flavs_seed[:, i], config.U, 0, config.area_uc
    )

n_discarded = num_picks - np.count_nonzero(valid_mask)      # total - valid realizations = number of realizations out of bounds
if n_discarded:
    print(f"Discarded {n_discarded}/{num_picks} realizations (target density out of range).")
else:
    print("All the realizations lie within the table range.")
if not np.any(valid_mask):      # if valid_mask is full of False
    raise RuntimeError("No valid realizations found: widen [mu_min, mu_max] or m_seed")

# -------- Internal Energy Calculation & Minimization --------
E_kin_vec = np.sum(E_flav_arr, axis=1)
E_int_vec = E_kin_vec + V_int_vec
E_int_vec[~valid_mask] = np.inf

E_int_min = np.min(E_int_vec)
E_int_min_idx = np.argmin(E_int_vec)

# -------- Global Fermi Level --------

def root_find_mu(mu, n_target, all_flavs_bands):
    n_flav_list = []
    for E0, E1 in all_flavs_bands:
        n_e = fermi_distrib(E0, mu, config.T_eff)
        n_h = 1.0 - fermi_distrib(E1, mu, config.T_eff)

        n_flav = prefactor * np.sum(n_e - n_h)
        n_flav_list.append(n_flav)

    n_total = np.sum(n_flav_list) * config.unit_cell_to_cm2

    return n_total - n_target

try:
    mu_global = brentq(root_find_mu, mu_min, mu_max, args=(n_target, all_bands), xtol=1e-5)
except ValueError:
    print(f"Error: The target density {n_target} is outside the bracket [{mu_min}, {mu_max}].")
    raise


# -------- Global Fermi level for shifted bands --------
print("Evaluating Fermi level for shifted bands...")
mu_shifted_vec = np.full(num_picks, np.nan)
for i in np.where(valid_mask)[0]:
    shifted_bands = [
        (E0 - mu_flav_arr[i, a], E1 - mu_flav_arr[i, a])
        for a, (E0, E1) in enumerate(all_bands)
    ]

    try:
        mu_shifted_vec[i] = brentq(root_find_mu, mu_min, mu_max, args=(n_target, shifted_bands), xtol=1e-5)
    except ValueError:
        print(f"Error: The target density {n_target} is outside the bracket [{mu_min}, {mu_max}].")
        raise

print(f"   Max mu_shifted: {np.max(mu_shifted_vec)}")
print(f"   Min mu_shifted: {np.min(mu_shifted_vec)}")


# -------- Consistency Check --------
potential_vec = E_int_vec - mu_shifted_vec * n_target
pot_min_idx = np.argmin(potential_vec)

if E_int_min_idx == pot_min_idx:
    print("Minimal internal energy gives the same solution as minimal potential.")
else:
    print("WARNING: The two minimizations don't agree -- check the implementation.")
 
print(f"Indices: E_min -> {E_int_min_idx}, Pot_min -> {pot_min_idx}")

n_sol  = n_flavs_seed[:, pot_min_idx]
mu_flavs_sol = mu_flav_arr[pot_min_idx]
mu_global_sol = mu_shifted_vec[pot_min_idx]

print("\nRESULTS")
print(f"    Global mu (for this configuration): {mu_global_sol:.4f} eV")
print(f"    SOLUTION n_alpha (cm^-2): {np.array2string(n_sol,  precision=4, floatmode='fixed')}")
print(f"    SOLUTION mu_alpha (meV):   {np.array2string(mu_flavs_sol*1e3, precision=4, floatmode='fixed')}")

t2 = time.time()
print(f"Time for {num_picks} realizations: {t2-t1:.2f} s")
print(f"Total calculation time: {t2 - t0:.2f} s")

# --------- Comparison plot ----------
plt.figure()
plt.plot(E_int_vec, 'o-', color='orange', label='Internal Energy')
plt.plot(E_int_min_idx, E_int_min, 'o-', color='red')

plt.plot(potential_vec, 's-', color='navy', label='Potential')
plt.plot(pot_min_idx, potential_vec[pot_min_idx], 's-', color='blue')

plt.xlabel("Realization")
plt.ylabel("Energy")

plt.legend()

plt.show()