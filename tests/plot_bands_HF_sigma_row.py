"""
plot_bands_HF_sigma_row.py
==========================
HF band structure of the 4 flavors at one point (iD, iN) of the phase diagram saved by scan_phase_diagram_HF.py,
along the ky = 0 row of the scan mesh, using the stored self-energy `sigma_row` (no SCF re-run).

H_alpha(kx) = h0 + Sigma_0 + [[Delta_eff, X_eff], [X_eff*, -Delta_eff]],
    Delta_eff = Delta + Sigma_z + Sigma_z^H ,   X_eff = X + Sigma_12 ,
with Sigma_0, Sigma_z, Sigma_12 from `sigma_row`, Sigma_z^H from `szh_map`, and the bare h0, Delta, X rebuilt from D.
Energies are shifted by the HF chemical potential `mu_map` (common to all flavors) -> E_F = 0.

Flavor convention (build_systems_with_displacement): f = v_idx + 2*spin_idx
    f = 0: K up,  f = 1: K' up,  f = 2: K down,  f = 3: K' down        (valley xi = [+1, -1, +1, -1])
"""

import numpy as np
import matplotlib.pyplot as plt

import model.config as config
from model.analysis import build_systems_with_displacement, get_bands_from_arrays

# =========== Parameters ===========
DATA_FILE = "phase_diagram_nD_HF_40x40pts.npz"   # file in config.DATA_DIR
iD, iN = 0, 14                                   # point of the phase diagram

K_PLOT = None                                    # half-width of the k window [1/a]; None -> whole scan mesh
E_WIN = 50.0                                    # half-height of the energy window [meV]

# =========== Load the phase-diagram point ===========
data = np.load(config.DATA_DIR / DATA_FILE, allow_pickle=True)
D = float(data["D_vals"][iD])
n_target = float(data["n_vals"][iN])
n_pts, k_lim = int(data["n_pts_scan"]), float(data["k_lim_scan"])

if not bool(data["conv_map"][iD, iN]):
    raise ValueError(f"Point (iD={iD}, iN={iN}) did not converge in the scan.")

mu = float(data["mu_map"][iD, iN])                 # eV
sigma0_center = data["sigma0_center"][iD, iN]
sigmaz_center = data["sigmaz_center"][iD, iN]
SzH = float(data["szh_map"][iD, iN])               # eV
row = data["sigma_row"][iD, iN]                    # (4, 4, N): (S0, Sz, Re S12, Im S12) along ky = 0 [eV]
S0, Sz, S12 = row[:, 0], row[:, 1], row[:, 2] + 1j * row[:, 3]

print(f"\nD = {D*1e3:.2f} meV | n = {n_target:.3e} cm^-2 | phase: {data['phase_labels'][iD, iN]}")
print(f"flavor densities [cm^-2]: {data['density_map'][iD, iN]}")
print(f"mu = {mu*1e3:.3f} meV\n")
# testing
print("Sz shape = ", Sz.shape)
print("Max Sz [meV] = ", np.max(np.abs(Sz)) * 1e3)
print("Max S0 [meV] = ", np.max(np.abs(S0)) * 1e3)

print("Sigma_0(k=0) [meV] =    ", sigma0_center * 1e3)
print("Sigma_z^F(k=0) [meV] =  ", sigmaz_center * 1e3)
print("Sigma_z^H [meV] =  ", SzH * 1e3)

# =========== HF bands along ky = 0, shifted by mu ===========
kx = np.linspace(-k_lim, k_lim, n_pts)             # same as KX[0, :] of get_kmesh(k_lim, n_pts)
ky = np.zeros_like(kx)
systems, _ = build_systems_with_displacement(D, config)

bands = []
for f, system in enumerate(systems):
    h0 = np.broadcast_to(system.get_h0_at_k(kx, ky), kx.shape)
    X = system.X_at_k(kx, ky)
    Ep, Em = get_bands_from_arrays(h0 + S0[f], system.Delta + Sz[f] + SzH, X + S12[f])
    bands.append(((Ep - mu) * 1e3, (Em - mu) * 1e3))   # meV, Fermi level -> 0

if K_PLOT is None:
    K_PLOT = k_lim


# =========== Figure ===========
def make_band_figure():
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True, num="HF bands")
    spin_colors, spin_ls, spin_names = ['red', 'blue'], ['-', '--'], ['up', 'dn']

    for ax, (v_idx, xi) in zip(axes, enumerate(config.VALLEY_IDX)):
        ax.set_title(r"Valley $K$" if xi == 1 else r"Valley $K'$", fontsize=16)
        ax.axhline(0, linestyle=':', color='gray', alpha=0.7)
        for s_idx in range(2):
            f = v_idx + 2 * s_idx                                  # flavor index, see module docstring
            for i, E in enumerate(bands[f]):
                ax.plot(kx, E, color=spin_colors[s_idx], ls=spin_ls[s_idx], lw=1.5,
                        label=spin_names[s_idx] if i == 0 else None)
        ax.set_xlim(-0.15, 0.15)        # before, used K_PLOT
        ax.set_ylim(-E_WIN, E_WIN)
        # ax.set_xticks(np.round(np.linspace(-K_PLOT, K_PLOT, 5), 3))
        ax.set_xticks([-0.1, 0.0, 0.1])
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel(r'$k_x a$', fontsize=16)

    axes[0].set_ylabel(r'$E-\mu$ (meV)', fontsize=16)
    axes[0].legend(fontsize=11, loc='upper center')
    fig.suptitle(f"HF bands, (n, D) = ({n_target:.2e} cm$^{{-2}}$, {D*1e3:.1f} meV)", fontsize=12)
    fig.tight_layout()
    return fig


fig = make_band_figure()
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(config.FIGURES_DIR / f"HF_bands_D{D*1e3:.2f}meV_n{n_target:.2e}.png", dpi=300, bbox_inches="tight")
plt.show()