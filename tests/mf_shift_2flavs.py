import numpy as np
from pathlib import Path
from scipy.optimize import brentq

from model.model import McCannCarts
from model.analysis import fermi_distrib, get_kmesh, get_part_density
import model.config as config

'''
We consider only two populated flavors, here (K, up) and (K', down), both with no gap and initially no energy shift.
We thus perform the self-consistent loop to determine their energy shifts E0.
'''

print(15*"="+" STARTING MEAN FIELD CALCULATION "+15*"=")

# -------- Iterative loop parameters --------
tolerance = 1e-6
max_iter = 200
mixing = 0.4

n_target = -3e11        # total charge carrier density in /cm^2
m_seed = 0.1 * n_target

# Initialize the flavored densities
# Since we consider only 2 flavors, we have: n_target = sum(n_flavs), m_seed = diff(n_flavs)
n_flavs = np.array([
    (n_target + m_seed) / 2.0,
    (n_target - m_seed) / 2.0
])

# -------- Precompute k-mesh and Energy bands --------
print("Precomputing the energy bands...")

KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
dk = KX[0, 1] - KX[0, 0]
prefactor = (dk**2) / (2 * np.pi)**2

# Define the unit cell area in cm^2
area_uc = (np.sqrt(3) / 2) * (config.a * 1e-8)**2
# Conversion factor from density per unit cell to /cm^2
unit_cell_to_cm2 = 1.0 / area_uc

# Build the systems
#   I SHOULD PROBABLY BUILD A FUNCTION PRECOMPUTE_ENERGY WITH A LIST OF FLAVORS TO BE CONSIDERED
system1 = McCannCarts(
    N=config.N, valley_idx=config.VALLEY_IDX[0], Delta=0.0,
    gamma0=config.GAMMA0, gamma1=config.GAMMA1,
    gamma2=config.GAMMA2, gamma3=config.GAMMA3,
    gamma4=config.GAMMA4, E0=0.0
)
system2 = McCannCarts(
    N=config.N, valley_idx=config.VALLEY_IDX[1], Delta=0.0,
    gamma0=config.GAMMA0, gamma1=config.GAMMA1,
    gamma2=config.GAMMA2, gamma3=config.GAMMA3,
    gamma4=config.GAMMA4, E0=0.0
)

# Get the energy bands
all_flavs_bands = [system1.get_energy(KX, KY), system2.get_energy(KX, KY)]

# -------- Helper Functions --------
def get_flavor_density(mu_alpha, flav_idx):   # Change to analysis.get_part_density() ?
    """Caclulates density n_alpha for flavor alpha at Fermi level mu_alpha."""
    E0, E1 = all_flavs_bands[flav_idx]

    # Count electrons in the conduction band and holes in the valence band
    n_e = fermi_distrib(E0, mu_alpha, config.T_eff)
    n_h = 1.0 - fermi_distrib(E1, mu_alpha, config.T_eff)

    return prefactor * np.sum(n_e - n_h) * unit_cell_to_cm2

def solve_mu_for_density(target_n_flav, flavor_idx, mu_search_min=-0.05, mu_search_max=0.05):
    """Inverts n_alpha -> mu_alpha for a specific single flavor."""
    residual = lambda mu: get_flavor_density(mu, flavor_idx) - target_n_flav
    return brentq(residual, mu_search_min, mu_search_max, xtol=1e-7)

def total_density_residual(mu_global, V_flavs):
    """
    Computes residual N_tot(mu) - n_target where each flavor sits 
    at effective energy shift mu_alpha = mu_global - V_alpha.
    """
    n_tot = 0.0
    for idx, V_alpha in enumerate(V_flavs):
        mu_alpha = mu_global - V_alpha
        n_tot += get_flavor_density(mu_alpha, idx)

    return n_tot - n_target

# Safe dynamic search range for global mu based on band structure
mu_min, mu_max = -0.10, 0.10  # eV

# -------- Self-Consistent Loop --------
print("Starting Self-Consistent Loop...")
for iteration in range(max_iter):

    # Get mean-field interactions V_alpha (Eq. S4)
    V_flavs = np.array([
        config.U * area_uc * (np.sum(n_flavs) - n_flav)
        for n_flav in n_flavs
    ])

    # Find global Fermi level mu that satisfies sum(n_alpha) = n_target
    try:
        mu_global = brentq(total_density_residual, mu_min, mu_max, args=(V_flavs,), xtol=1e-7)
    except ValueError:
        print(f"Error at iteration {iteration}: Target density outside mu bracket [{mu_min}, {mu_max}]")
        raise

    # Get new flavor densities at mu_alpha = mu_global - V_alpha
    n_flavs_new = np.array([
        get_flavor_density(mu_global - V_flavs[i], i)
        for i in range(len(V_flavs))
    ])

    # Evaluate the error and get the max
    rel_error = np.max(np.abs(n_flavs_new - n_flavs) / np.abs(n_target))
    
    print(f"Iter {iteration:02d} | mu_global: {mu_global*1e3:.3f} meV | Max Error: {rel_error:.3e} | n_flavs: {n_flavs}")

    if rel_error < tolerance:
        print(f"\n    Converged in {iteration+1} iterations!")
        print(f"Final Flavor Densities: {n_flavs_new}")
        print(f"Final Global Fermi Level: {mu_global} eV")
        break
    
    # Linear density mixing for numerical stability
    n_flavs = (1.0 - mixing) * n_flavs + mixing * n_flavs_new
else:
    print("   Warning: Reached maximum iterations without full convergence.")

# -------- Calculate Single-Flavor Fermi Levels --------
# Calculate final V_alpha for the converged values
V_flavs_final = np.array([
        config.U * area_uc * (np.sum(n_flavs) - n_flav)
        for n_flav in n_flavs
    ])

# Calculate the final Fermi levels mu_alpha
mu_flavs_final = mu_global - V_flavs_final

print("\n" + "="*15 + " FINAL FLAVOR RESULTS " + "="*15)
print(f"Global Fermi Level (mu_global): {mu_global * 1e3:.4f} meV")

# Verify for flavor 0
# mu_0_check = solve_mu_for_density(n_flavs_new[0], 0)
# print(f"Check mu_0 via direct inversion: {mu_0_check * 1e3:.4f} meV")
# print(f"Check mu_0 via (mu_global - V_0)  : {(mu_global - V_flavs_final[0]) * 1e3:.4f} meV")

# Check flavor 0 with integration from analysis.get_part_density()


for idx, (n_i, V_i, mu_i) in enumerate(zip(n_flavs, V_flavs_final, mu_flavs_final)):
    print(f"\nFlavor {idx}:")
    print(f"  Density (n_{idx})           : {n_i:.3e} cm^-2")
    print(f"  Interaction Potential (V_{idx}): {V_i * 1e3:.4f} meV")
    print(f"  Flavor Fermi Level (mu_{idx}) : {mu_i * 1e3:.4f} meV")