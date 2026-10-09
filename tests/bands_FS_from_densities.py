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
D = 1.24e-3               # fixed displacement field [eV]
n_flavs = np.array([       # flavor densities [cm^-2]
    -9.08349958e+11,  6.09533180e+11, -9.08349629e+11,  6.09533271e+11
])
n_tot = np.sum(n_flavs)

K_PLOT = 0.12              # half-width of the plotted k-window (Fermi surface)
SHARED_SCALE = True        # same color scale in all Fermi-surface panels
K_PTS = 500                # points along the k-path for the band plot

print(f"Bands + Fermi surface for D = {D*1e3:.2f} meV and n = {n_tot:.3e} cm^-2...")

# =========== k-mesh, systems, bands (2D) ===========
KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
dk = KX[0, 1] - KX[0, 0]
prefactor = (dk**2) / (2 * np.pi)**2

systems, _ = build_systems_with_displacement(D, config)
all_bands = precompute_flavor_bands(systems, KX, KY)

# =========== Flavor chemical potentials ===========
mu_flavs = get_flavor_mu_from_densities(
    n_flavs, all_bands, config.T_eff, prefactor, config.unit_cell_to_cm2
)
print(f"Found mu_alpha (meV): {mu_flavs*1e3}")

# =========== Fermi-surface maps: -df/dE summed over both bands, cropped ===========
kx_vals = KX[0, :]
ky_vals = KY[:, 0]
ix = np.where(np.abs(kx_vals) <= K_PLOT)[0]
iy = np.where(np.abs(ky_vals) <= K_PLOT)[0]
sl = np.ix_(iy, ix)
extent = [kx_vals[ix[0]], kx_vals[ix[-1]], ky_vals[iy[0]], ky_vals[iy[-1]]]

fs_maps = []
for f, (E0, E1) in enumerate(all_bands):
    # deriv_fermi_distrib returns df/dE (negative), so flip the sign
    w = -(deriv_fermi_distrib(E0[sl], mu_flavs[f], config.T_eff) + deriv_fermi_distrib(E1[sl], mu_flavs[f], config.T_eff))
    fs_maps.append(w)

del all_bands, KX, KY       # free the big 2D arrays

# =========== Bands along the k-path, shifted by mu_alpha ===========
kx = np.linspace(-config.K_LIM, config.K_LIM, K_PTS)
ky = np.zeros_like(kx)

########  WARNING: THE ORDER MIGHT BE WRONG  ########
# Flavor order from build_systems_with_displacement: f = v_idx + 2 * spin_idx
bands_shifted = []
for f, system in enumerate(systems):
    E0, E1 = system.get_energy(kx, ky)
    bands_shifted.append((E0 - mu_flavs[f], E1 - mu_flavs[f]))   # flavor Fermi level -> 0


# =========== Figure 1: band structure ===========
def make_band_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharex=True, sharey=True,
                             num="Bands")
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
            ax.plot(kx, E0 * 1e3, color=color, lw=1.5, ls=spin_ls[spin_idx])
            ax.plot(kx, E1 * 1e3, color=color, lw=1.5, ls=spin_ls[spin_idx])

        ax.set_ylim(-50, 50)
        ax.set_xlim(-0.12, 0.12)
        ax.set_xticks([-0.1, 0, 0.1])
        ax.tick_params(axis='both', labelsize=17)

    axes[0].set_ylabel(r'Energy (meV)', fontsize=18)
    fig.text(0.53, 0.03, r'$k_x a$', ha='left', fontsize=18)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig


# =========== Figure 2: Fermi Surface ===========
def make_fermi_surface_figure():
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True,
                             num="Fermi surface")

    spin_symbols = [r"\uparrow", r"\downarrow"]
    valley_names = [r"K", r"K'"]
    vmax_shared = max(m.max() for m in fs_maps) if SHARED_SCALE else None

    im = None
    for s_idx in range(2):
        for v_idx in range(2):
            ax = axes[s_idx, v_idx]
            f = v_idx + 2 * s_idx
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

    fig.suptitle(f"Fermi Surfaces for (n, D) = ({n_tot:.2e}, {D*1e3:.1f})")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02, shrink=0.9)
    cbar.set_label(r"$-\partial f/\partial E$", fontsize=14) # [eV$^{-1}$]
    cbar.set_ticklabels([])
    # cbar.ax.tick_params(labelsize=12)
    return fig


fig_bands = make_band_figure()
fig_fs = make_fermi_surface_figure()

# Save the figures
# fig_bands.savefig(config.FIGURES_DIR / f"energy_bands_shifted_D{D*1e3:.2f}meV.png", dpi=300)
# fig_fs.savefig(config.FIGURES_DIR / f"fermi_surface_D{D*1e3:.2f}meV.png", dpi=300, bbox_inches="tight")

# A single show() call after both figures exist opens both windows at once
plt.show()