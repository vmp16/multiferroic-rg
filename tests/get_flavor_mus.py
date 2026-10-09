import numpy as np
import matplotlib.pyplot as plt

import model.config as config
from model.analysis import (
    get_kmesh,
    build_systems_with_displacement,
    precompute_flavor_bands,
    get_flavor_mu_from_densities,
)

# =========== Parameters ===========
D = 2.07e-3               # fixed displacement field [eV]
n_flavs = np.array([       # flavor densities [cm^-2]
    6.72712480e+11,
    -8.24783306e+11,
    -8.24783652e+11,
    6.72712480e+11
])
n_tot = np.sum(n_flavs)

print(f"Getting mu_alpha for D = {D*1e3:.2f} meV and n = {n_tot:.3e} cm^-2...")

# =========== k-mesh grid (for the density integral) ===========
KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
dk = KX[0, 1] - KX[0, 0]
prefactor = (dk**2) / (2 * np.pi)**2

# =========== Build Systems & Get Flavor Chemical Potentials ===========
systems, _ = build_systems_with_displacement(D, config)
all_bands = precompute_flavor_bands(systems, KX, KY)

mu_flavs = get_flavor_mu_from_densities(
    n_flavs, all_bands, config.T_eff, prefactor, config.unit_cell_to_cm2
)
print(f"Found mu_alpha (meV): {mu_flavs*1e3}")

# Free the big 2D arrays, only the 1D path is needed from here on
del all_bands, KX, KY

# =========== Bands along the k-path, shifted by mu_alpha ===========
k_pts = 500
kx = np.linspace(-config.K_LIM, config.K_LIM, k_pts)
ky = np.zeros_like(kx)

####### WARNING: THE ORDER MUST BE WRONG #######
# Flavor order from build_systems_with_displacement:
#   flavor f = valley_position + 2 * spin_index
#   0: (K, up)   1: (K', up)   2: (K, dn)   3: (K', dn)
bands_shifted = []
for f, system in enumerate(systems):
    E0, E1 = system.get_energy(kx, ky)
    bands_shifted.append((E0 - mu_flavs[f], E1 - mu_flavs[f]))   # Fermi level of flavor f -> 0

# =========== Plot (same format as bandplot.py) ===========
fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharex=True, sharey=True)
axes = np.atleast_1d(axes)

spin_labels = ['up', 'dn']
spin_colors = {'up': 'red', 'dn': 'blue'}
spin_ls = ['-', '--']

for ax, (v_idx, xi) in zip(axes, enumerate(config.VALLEY_IDX)):
    valley_name = r"$K$" if xi == 1 else r"$K'$"
    ax.set_title(f"Valley {valley_name}", fontsize=16)
    ax.axhline(0, linestyle=':', color='gray', alpha=0.7)   # flavor Fermi level

    for spin_idx, spin in enumerate(spin_labels):
        f = v_idx + 2 * spin_idx
        E0, E1 = bands_shifted[f]
        color = spin_colors[spin]

        # Plot both conduction and valence bands for this spin.
        ax.plot(kx, E0 * 1e3, color=color, lw=1.5, ls=spin_ls[spin_idx])
        ax.plot(kx, E1 * 1e3, color=color, lw=1.5, ls=spin_ls[spin_idx])

    ax.set_ylim(-70, 100)
    ax.set_xlim(-0.12, 0.12)
    ax.set_xticks([-0.1, 0, 0.1])
    ax.tick_params(axis='both', labelsize=17)

axes[0].set_ylabel(r'Energy (meV)', fontsize=18)
fig.text(0.53, 0.03, r'$k_x a$', ha='left', fontsize=18)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])

# Save the figure
output_path = config.FIGURES_DIR / f'energy_bands_shifted_D{D*1e3:.2f}meV.png'
# plt.savefig(output_path, dpi=300)
# print(f"Plot saved to {output_path}")
plt.show()