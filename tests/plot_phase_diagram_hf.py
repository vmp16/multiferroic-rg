"""
plot_phase_diagram_HF.py
========================
Render the (n, D) Hartree-Fock phase diagram saved by scan_phase_diagram_HF.py.

Left : flavor-polarization phase (discrete colours, labels as in config.PHASE_ORDER).
Right: energy gap E_2 - E_1 between the lowest and the next distinct converged solution [eV/a^2], log scale.
       Large gap = the ground state is robust; small gap = competing solutions (the phase may depend on the
       seeds / mesh). White = only one distinct solution was found (or the point failed).

Kept separate from the scan so the figure can be restyled without recomputing anything.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm

import model.config as config

DATA_FILE = config.DATA_DIR / "phase_diagram_nD_HF_40x40pts.npz"     # match the scan's N_POINTS x D_POINTS
PRINT_POINT = (25, 5)               # (iD, iN) to print and highlight

data = np.load(DATA_FILE, allow_pickle=True)       # allow_pickle: phase_labels / winner_kind are object arrays

n_vals = data["n_vals"]
D_vals = data["D_vals"]
phase_map = data["phase_map"].astype(float)
dE_next_map = data["dE_next_map"]        # "gap_map" for old scans
density_map = data["density_map"]
threshold = float(data["threshold"])
k_lim_scan = float(data["k_lim_scan"])
n_pts_scan = int(data["n_pts_scan"])

# Failed points -> NaN so they render as blank rather than as a phase
phase_map[phase_map < 0] = np.nan
dE_next = np.where(dE_next_map > 0, dE_next_map, np.nan)

# TEST: sigma_row for the band structure plot
# sigma_row = data["sigma_row"]
# print("Sigma row check: shape = ", sigma_row.shape)

# ---------------------------------------------------------
# Printing one configuration
# ---------------------------------------------------------
iD, iN = PRINT_POINT
if 0 <= iD < len(D_vals) and 0 <= iN < len(n_vals) and np.isfinite(phase_map[iD, iN]):
    print(f"Ground state for (n, D) = ({n_vals[iN]:.4e} cm^-2, {D_vals[iD]*1e3:.2f} meV):")
    print("   Densities [cm^-2]:     ", density_map[iD, iN])
    print("   Found phase:           ", config.PHASE_ORDER[int(phase_map[iD, iN])])
    print("   Winning start:         ", data["winner_kind"][iD, iN])
    print("   Converged runs:        ", int(data["n_conv_map"][iD, iN]),
          "| distinct solutions:", int(data["n_distinct_map"][iD, iN]))
    print("   Sigma_0(k=0) [meV]:    ", data["sigma0_center"][iD, iN] * 1e3)
    print("   Sigma_z^F(k=0) [meV]:  ", data["sigmaz_center"][iD, iN] * 1e3)
    print("   Hartree Sigma_z [meV]: ", data["szh_map"][iD, iN] * 1e3)
else:
    print(f"PRINT_POINT {PRINT_POINT} is out of range or failed; nothing printed.")

# ---------------------------------------------------------
#  Discrete colour scale, one colour per phase
# ---------------------------------------------------------
n_phases = len(config.PHASE_ORDER)
cmap = mpl.colormaps["viridis"].resampled(n_phases)
cmap.set_bad("white")
norm = BoundaryNorm(np.arange(-0.5, n_phases, 1.0), ncolors=n_phases)


# pcolormesh wants cell EDGES, not centres
def edges(v):
    d = np.diff(v)
    return np.concatenate([[v[0] - d[0] / 2], v[:-1] + d / 2, [v[-1] + d[-1] / 2]])


n_edges = edges(n_vals)
D_edges = edges(D_vals) * 1e3

fig, axs = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)

# --------- Left: phases ---------
ax = axs[0]
mesh = ax.pcolormesh(n_edges, D_edges, phase_map, cmap=cmap, norm=norm, shading="flat")
if np.isfinite(phase_map[iD, iN]):
    ax.plot(n_vals[iN], D_vals[iD] * 1e3, marker="o", markersize=7,
            markerfacecolor="none", markeredgecolor="orange", markeredgewidth=2)
ax.set_xlabel(r"Carrier density $n$  [cm$^{-2}$]")
ax.set_ylabel(r"Displacement field $D$  [meV]")
ax.set_title(rf"HF flavor polarization phases  (threshold = {threshold:.0e} cm$^{{-2}}$)", fontsize=10)
cbar = fig.colorbar(mesh, ax=ax, ticks=range(n_phases), pad=0.02)
cbar.set_label("phase (occupation + degeneracy)")
cbar.ax.set_yticklabels(config.PHASE_ORDER)

# --------- Right: robustness ---------
ax = axs[1]
if np.any(np.isfinite(dE_next)):
    cmap_dE_next = mpl.colormaps["magma"].copy()
    cmap_dE_next.set_bad("white")
    mesh2 = ax.pcolormesh(n_edges, D_edges, dE_next, cmap=cmap_dE_next,
                          norm=LogNorm(vmin=np.nanmin(dE_next), vmax=np.nanmax(dE_next)), shading="flat")
    cbar2 = fig.colorbar(mesh2, ax=ax, pad=0.02)
    cbar2.set_label(r"$E_2 - E_1$  [eV / $a^2$]")
else:
    ax.text(0.5, 0.5, "only one distinct solution\nfound everywhere", ha="center", va="center",
            transform=ax.transAxes)
ax.set_xlabel(r"Carrier density $n$  [cm$^{-2}$]")
ax.set_title("energy gap to the next distinct solution (white: none found)", fontsize=10)

for ax in axs:
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
fig.tight_layout()

config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out = config.FIGURES_DIR / f"phase_diagram_nD_HF_{len(n_vals)}x{len(D_vals)}pts_N{n_pts_scan}_klim{k_lim_scan}.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved figure: {out}")

plt.show()