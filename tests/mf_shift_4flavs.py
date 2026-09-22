import numpy as np

from model.model import McCannCarts
import model.config as config
from model.analysis import get_kmesh, precompute_flavor_bands, solve_self_consistent_mean_field

print(15 * "=" + " STARTING MEAN FIELD CALCULATION (4 FLAVORS) " + 15 * "=")

# -------- System & Target Setup --------
n_target = -3e11  # total charge carrier density in /cm^2

# Define target flavor configuration dynamically
# systems = [
#     McCannCarts(
#         N=config.N, valley_idx=config.VALLEY_IDX[0], Delta=0.0,
#         gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
#         gamma3=config.GAMMA3, gamma4=config.GAMMA4
#     ),
#     McCannCarts(
#         N=config.N, valley_idx=config.VALLEY_IDX[1], Delta=0.0,
#         gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
#         gamma3=config.GAMMA3, gamma4=config.GAMMA4
#     ),
#     McCannCarts(
#         N=config.N, valley_idx=config.VALLEY_IDX[0], Delta=0.0,
#         gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
#         gamma3=config.GAMMA3, gamma4=config.GAMMA4
#     ),
#     McCannCarts(
#         N=config.N, valley_idx=config.VALLEY_IDX[1], Delta=0.0,
#         gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
#         gamma3=config.GAMMA3, gamma4=config.GAMMA4
#     )
# ]

systems = [
    McCannCarts(
        N=config.N, valley_idx=xi, Delta=delta,
        gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
        gamma3=config.GAMMA3, gamma4=config.GAMMA4
    )
    for xi, delta in zip(2*config.VALLEY_IDX, config.DELTAS.T.flatten())
]

# -------- Params Initialization --------
# Seeds / initial flavor distribution
m_seed = 0.0 * n_target

# Initialization with 1 single seed
initial_n_flavs = [
    (n_target + m_seed) / 4.0,      # K, up
    (n_target + m_seed) / 4.0,      # K', up
    (n_target - m_seed) / 4.0,      # K, down
    (n_target - m_seed) / 4.0       # K', down
]

# # Initialization with 4 random values
# n_base = n_target / 4.0

# # Generate 4 independent random values
# raw_deltas = np.random.uniform(-1, 1, size=4)

# # Subtract the mean so they strictly sum to 0
# # Normalize by the theoretical maximum of 1.5 to constrain them to [-1.0, 1.0]
# centered_deltas = (raw_deltas - np.mean(raw_deltas)) / 1.5

# # Rescale as a percentage of m_seed
# deltas = centered_deltas * m_seed

# # Construct final populations
# initial_n_flavs = (n_base + deltas).tolist()

# Assuming 2 gapped flavors
# initial_n_flavs = np.array([
#     (n_target + m_seed) / 2,
#     0.0,
#     0.0,
#     (n_target - m_seed) / 2
# ])

# -------- Grid --------
KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
dk = KX[0, 1] - KX[0, 0]
prefactor = (dk**2) / (2 * np.pi)**2

# -------- Precompute Energy Bands --------
print("Precomputing energy bands for all active flavors...")
all_bands = precompute_flavor_bands(systems, KX, KY)

# -------- Execute Self-Consistent Mean Field Loop --------
n_flavs_final, V_flavs_final, mu_global = solve_self_consistent_mean_field(
    all_bands=all_bands,
    n_target=n_target,
    U=config.U,
    area_uc=config.area_uc,
    T_eff=config.T_eff,
    prefactor=prefactor,
    unit_cell_to_cm2=config.unit_cell_to_cm2,
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
    print(f"\nFlavor {idx+1}:")
    print(f"  Density (n_{idx+1})           : {n_i:.3e} cm^-2")
    print(f"  Interaction Potential (V_{idx+1}): {V_i * 1e3:.4f} meV")
    print(f"  Flavor Fermi Level (mu_{idx+1}) : {mu_i * 1e3:.4f} meV")