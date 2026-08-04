import numpy as np

from model.model import McCannCarts
import model.config as config
from model.analysis import get_kmesh, precompute_flavor_bands, solve_self_consistent_mean_field

print(15 * "=" + " STARTING MEAN FIELD CALCULATION " + 15 * "=")

# -------- System & Target Setup --------
n_target = -3e11  # total charge carrier density in /cm^2

# Define target flavor configuration dynamically
# (Easily scale to 4 flavors by adding valley/spin configurations here)
systems = [
    McCannCarts(
        N=config.N, valley_idx=config.VALLEY_IDX[0], Delta=0.0,
        gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
        gamma3=config.GAMMA3, gamma4=config.GAMMA4, E0=0.0
    ),
    McCannCarts(
        N=config.N, valley_idx=config.VALLEY_IDX[1], Delta=0.0,
        gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
        gamma3=config.GAMMA3, gamma4=config.GAMMA4, E0=0.0
    ),
]

# Seeds / initial flavor distribution
m_seed = 0.1 * n_target
initial_n_flavs = [
    (n_target + m_seed) / 2.0,
    (n_target - m_seed) / 2.0
]

# -------- Grid and Unit Conversions --------
KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
dk = KX[0, 1] - KX[0, 0]
prefactor = (dk**2) / (2 * np.pi)**2

area_uc = (np.sqrt(3) / 2) * (config.a * 1e-8)**2  # cm^2
unit_cell_to_cm2 = 1.0 / area_uc

# -------- Precompute Energy Bands --------
print("Precomputing energy bands for all active flavors...")
all_bands = precompute_flavor_bands(systems, KX, KY)

# -------- Execute Self-Consistent Mean Field Loop --------
n_flavs_final, V_flavs_final, mu_global = solve_self_consistent_mean_field(
    all_bands=all_bands,
    n_target=n_target,
    U=config.U,
    area_uc=area_uc,
    T_eff=config.T_eff,
    prefactor=prefactor,
    unit_cell_to_cm2=unit_cell_to_cm2,
    initial_n_flavs=initial_n_flavs,
    tolerance=1e-6,
    max_iter=200,
    mixing=0.4
)

# -------- Final Summary Output --------
mu_flavs_final = mu_global - V_flavs_final

print("\n" + "=" * 15 + " FINAL FLAVOR RESULTS " + "=" * 15)
print(f"Global Fermi Level (mu_global): {mu_global * 1e3:.4f} meV")

for idx, (n_i, V_i, mu_i) in enumerate(zip(n_flavs_final, V_flavs_final, mu_flavs_final)):
    print(f"\nFlavor {idx}:")
    print(f"  Density (n_{idx})           : {n_i:.3e} cm^-2")
    print(f"  Interaction Potential (V_{idx}): {V_i * 1e3:.4f} meV")
    print(f"  Flavor Fermi Level (mu_{idx}) : {mu_i * 1e3:.4f} meV")