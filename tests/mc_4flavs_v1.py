"""
Calculates the 4 flavor configuration minimizing the total grand potential using a Monte-Carlo approach.
VERSION 1:
    - Loop over the number of realizations
    - brentq to find the flavor Fermi levels mu_flavs
"""

import numpy as np
from scipy.optimize import brentq
import time

from model.model import McCannCarts
import model.config as config
from model.analysis import progress_bar, get_kmesh, fermi_distrib, precompute_flavor_bands, V_int_anisotropic, get_flavor_kinetic_energy_from_bands

# -------- System & Target Setup --------
np.random.seed(41)

n_target = -3e11  # total charge carrier density in /cm^2
n_flavors = 4

systems = [
    McCannCarts(
        N=config.N, valley_idx=xi, Delta=delta,
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
# Seed initialization / Density range

# 1. Pick N trios of numbers within [n_min, n_max]
m_seed = 0.05
num_picks = int(1e3)

n1, n2, n3 = np.random.uniform(1 - m_seed, 1 + m_seed, (3, num_picks)) * n_target

# 2. Get n_4 = n_target - (n_1 + n_2 + n_3)
n4 = n_target - (n1 + n2 + n3)
# forget those where n_4 not in the range ???

# Build the vector
n_flavs_seed = np.array((n1, n2, n3, n4))

# 3. Compute the interaction potential V_int
# 4. Get mu_alpha that yields n_alpha for every flavor and realization (look in find_mu_at_n.py)

def objective_function(mu_flav, bands, n_flav_target):
    E0, E1 = bands

    n_e = fermi_distrib(E0, mu_flav, config.T_eff)
    n_h = 1.0 - fermi_distrib(E1, mu_flav, config.T_eff)

    n_flav = prefactor * np.sum(n_e - n_h) * config.unit_cell_to_cm2

    return n_flav - n_flav_target

# Bracket for the root search
mu_min = -0.1
mu_max = 0.1

# mu_flav_arr[i, a]  -> Fermi level of flavor a in realization i
# E_flav_arr[i, a]   -> kinetic energy of flavor a in realization i
mu_flav_arr = np.full((num_picks, n_flavors), np.nan)
E_flav_arr = np.full((num_picks, n_flavors), np.nan)
V_int_vec = np.full(num_picks, np.nan)

# Realizations where at least one flavor's target density could not be reached
# within [mu_min, mu_max] are marked invalid and excluded from the minimization.
valid_mask = np.ones(num_picks, dtype=bool)

t0 = time.time()
print(f"Scanning {num_picks} realizations x {n_flavors} flavors ...")
for i in range(num_picks):
    progress_bar(i + 1, num_picks)
    for a, flav_bands in enumerate(all_bands):
        n_flav_target = n_flavs_seed[a, i]
 
        try:
            mu_flav_root = brentq(
                objective_function, mu_min, mu_max,
                args=(flav_bands, n_flav_target), xtol=1e-5
            )
        except ValueError:
            # n_flav_target not reachable in [mu_min, mu_max] for this flavor's bands -> discard this whole realization.
            valid_mask[i] = False
            break
 
        mu_flav_arr[i, a] = mu_flav_root

        # 5. Compute the kinetic energy E_alpha = E(mu_alpha) for this flavor
        E_flav_arr[i, a] = get_flavor_kinetic_energy_from_bands(
            flav_bands, mu_flav_root, config.T_eff, prefactor, config.unit_cell_to_cm2
        )
    else:
        # Only reached if the flavor loop completed without a `break`, i.e. every flavor's mu_alpha was found successfully.
        # Compute the interaction potential V_int for this realization.
        V_int_vec[i] = V_int_anisotropic(
            n_flavs_seed[:, i], config.U, 0, config.area_uc
        )

n_discarded = num_picks - np.count_nonzero(valid_mask)
if n_discarded:
    print(f"\nDiscarded {n_discarded}/{num_picks} realizations (target density outside [{mu_min}, {mu_max}] bracket for some flavor).")
else:
    print("\nAll flavor densities lie within the range [mu_min, mu_max] !")
if not np.any(valid_mask):
    raise RuntimeError("No valid realizations found: widen [mu_min, mu_max] or m_seed.")

# 6. Calculate the total internal energy sum(E_alpha) + V_int, per realization
E_kin_vec = np.sum(E_flav_arr, axis=1)    # shape (num_picks,)
E_int_vec = E_kin_vec + V_int_vec         # shape (num_picks,)
E_int_vec[~valid_mask] = np.inf           # invalid realizations never win the min

# 7. Compare them all and take the minimum --> SOLUTION
# We look for the minimum of E_int_vec and take the index of the configuration giving that minimum.

E_int_min = np.min(E_int_vec)
E_int_min_idx = np.argmin(E_int_vec)

# 8. Calculate the global Fermi level: the single mu that reproduces n_target when ALL flavors share that same mu (independent of any particular realization / flavor-polarization; it only depends on the band structure).

def root_find_mu(mu, n_target):
    n_flav_list = []

    for E0, E1 in all_bands:
        n_e = fermi_distrib(E0, mu, config.T_eff)
        n_h = 1.0 - fermi_distrib(E1, mu, config.T_eff)

        n_flav = prefactor * np.sum(n_e - n_h)
        n_flav_list.append(n_flav)

    n_total = np.sum(n_flav_list) * config.unit_cell_to_cm2

    return n_total - n_target

try:
    mu_global = brentq(root_find_mu, mu_min, mu_max, args=(n_target,), xtol=1e-5)
except ValueError:
    print(f"Error: The target density {n_target} is outside the bracket [{mu_min}, {mu_max}].")
    raise

# 9. Compare them all and check that the minimum corresponds to the minimum of the grand potential Phi = E_int - mu_global * n_tot. NOTE: every realization already satisfies sum_alpha n_alpha = n_target by construction (step 2), so mu_global * n_target is the SAME constant offset for every realization. Minimizing E_int_vec and minimizing potential_vec must therefore always pick the same realization -- this is a consistency check on the implementation, not an independent physical condition.

potential_vec = E_int_vec - mu_global * n_target
 
potential_min = np.min(potential_vec)
pot_min_idx = np.argmin(potential_vec)

# Compare and take the solution
if E_int_min_idx == pot_min_idx:
    print("Minimal internal energy gives the same solution as minimal potential.")
else:
    print("WARNING: The two minimizations don't agree -- check the implementation.")
 
print(f"Indices: E_min -> {E_int_min_idx}, Pot_min -> {pot_min_idx}")
 
n_sol = n_flavs_seed[:, pot_min_idx]
mu_sol = mu_flav_arr[pot_min_idx]

print("\nRESULTS")
print(f"    Global mu (from total density): {mu_global:.6f} eV")
print(f"    SOLUTION n_alpha (cm^-2): {np.array2string(n_sol, precision=4, floatmode='fixed')}")
print(f"    SOLUTION mu_alpha (meV):   {np.array2string(mu_sol*1e3, precision=6, floatmode='fixed')}")

t1 = time.time()
print(f"Time for {num_picks} realizations: {t1-t0:.2f} s")