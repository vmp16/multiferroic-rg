import numpy as np
import matplotlib.pyplot as plt

import model.config as config
from model.analysis import (
    get_kmesh,
    build_systems_with_displacement,
    precompute_flavor_bands,
    get_flavor_mu_from_densities,
    deriv_fermi_distrib,
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

K_PLOT = 0.12              # half-width of the plotted k-window
SHARED_SCALE = False        # same color scale in all panels (empty flavors stay blank)


print(f"Fermi Surface for D = {D*1e3:.2f} meV and n = {n_tot:.3e} cm^-2...")

# =========== Sytems & Bands ===========
KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
dk = KX[0, 1] - KX[0, 0]
prefactor = (dk**2) / (2 * np.pi)**2

systems, _ = build_systems_with_displacement(D, config)
all_bands = precompute_flavor_bands(systems, KX, KY)

# ========== Flavor Chemical Potentials ===========
mu_flavs = get_flavor_mu_from_densities(
    n_flavs, all_bands, config.T_eff, prefactor, config.unit_cell_to_cm2
)
print(f"Found mu_alpha (meV): {mu_flavs*1e3}")

# =========== Thermal window -df/dE per flavor, cropped to the plot window ===========
kx_vals = KX[0, :]
ky_vals = KY[:, 0]
ix = np.where(np.abs(kx_vals) <= K_PLOT)[0]
iy = np.where(np.abs(ky_vals) <= K_PLOT)[0]
sl = np.ix_(iy, ix)
extent = [kx_vals[ix[0]], kx_vals[ix[-1]], ky_vals[iy[0]], ky_vals[iy[-1]]]

fs_maps = []
for f, (E0, E1) in enumerate(all_bands):
    # deriv_fermi_distrib returns df/dE (negative), so flip the sign
    w = -(deriv_fermi_distrib(E0[sl], mu_flavs[f], config.T_eff)
          + deriv_fermi_distrib(E1[sl], mu_flavs[f], config.T_eff))
    fs_maps.append(w)

del all_bands, KX, KY       # free the big 2D arrays

# =========== Plot: rows = spin, columns = valley (as in bandplot.py) ===========
fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)

spin_symbols = [r"\uparrow", r"\downarrow"]
valley_names = [r"K", r"K'"]

vmax_shared = max(m.max() for m in fs_maps) if SHARED_SCALE else None

for s_idx in range(2):
    for v_idx in range(2):
        ax = axes[s_idx, v_idx]
        f = v_idx + 2 * s_idx          # flavor order from build_systems_with_displacement
        im = ax.imshow(
            fs_maps[f], origin="lower", extent=extent, cmap="inferno",
            vmin=0.0, vmax=vmax_shared, interpolation="nearest", aspect="equal",
        )
        # ax.set_title(rf"${valley_names[v_idx]}\,{spin_symbols[s_idx]}$   ($\mu_\alpha$ = {mu_flavs[f]*1e3:.2f} meV)", fontsize=13)
        ax.set_xticks([-0.1, 0, 0.1])
        ax.set_yticks([-0.1, 0, 0.1])
        ax.tick_params(axis="both", labelsize=15)

for ax in axes[1, :]:
    ax.set_xlabel(r"$k_x a$", fontsize=16)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$k_y a$", fontsize=16)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02, shrink=0.9)
cbar.set_label(r"$-\partial f/\partial E$", fontsize=14) # [eV$^{-1}$]
cbar.set_ticklabels([])
# cbar.ax.tick_params(labelsize=12)

# Save the figure
output_path = config.FIGURES_DIR / f"fermi_surface_D{D*1e3:.2f}meV.png"
# fig.savefig(output_path, dpi=300, bbox_inches="tight")
# print(f"Plot saved to {output_path}")
plt.show()