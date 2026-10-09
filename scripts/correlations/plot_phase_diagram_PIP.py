"""
plot_phase_diagram.py
=====================
Render the (n, D) phase diagram saved by scan_phase_diagram.py.

Kept separate from the scan so the figure can be restyled without
recomputing anything.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

import model.config as config

data = np.load(config.DATA_DIR / "phase_diagram_nD_PIP_170pts.npz")

n_vals = data["n_vals"]
n_pts = len(n_vals)
D_vals = data["D_vals"]
phase_map = data["phase_map"].astype(float)
density_map = data["density_map"]
threshold = float(data["threshold"])

# Failed points -> NaN so they render as blank rather than as a phase
phase_map[phase_map < 0] = np.nan

# ---------------------------------------------------------
# Printing some configurations
# ---------------------------------------------------------
iD = 3
iN = 150
dens_print = density_map[iD, iN]
print(f"Ground state for (n, D) = ({n_vals[iN]:.4e}, {D_vals[iD]*1e3:.2f}):")
print("   Densities: ", dens_print)
print("   Found phase: ", config.PHASE_ORDER[int(phase_map[iD, iN])])

# ---------------------------------------------------------
#  Discrete colour scale, one colour per occupation number
# ---------------------------------------------------------

n_phases = len(config.PHASE_ORDER)
cmap = mpl.colormaps["viridis"].resampled(n_phases)
cmap.set_bad("white")

# One bin per integer phase index: edges at -0.5, 0.5, ..., n_phases - 0.5
norm = BoundaryNorm(np.arange(-0.5, n_phases, 1.0), ncolors=n_phases)

# pcolormesh wants cell EDGES, not centres
def edges(v):
    d = np.diff(v)
    return np.concatenate([[v[0] - d[0] / 2],
                           v[:-1] + d / 2,
                           [v[-1] + d[-1] / 2]])

n_edges = edges(n_vals)
D_edges = edges(D_vals)

fig, ax = plt.subplots(figsize=(7.0, 5.2))

mesh = ax.pcolormesh(
    n_edges, D_edges * 1e3, phase_map,
    cmap=cmap, norm=norm, shading="flat",
)

# Highlight a point
ax.plot(
    n_vals[iN], D_vals[iD]*1e3,
    marker="o", markersize=7,
    color='black',
    markerfacecolor="none", markeredgecolor="orange", markeredgewidth=2,
    # zorder=5,
)

ax.set_xlabel(r"Carrier density $n$  [cm$^{-2}$]")
ax.set_ylabel(r"Displacement field $D$  [meV]")
ax.set_title(
    rf"Flavor polarization phase diagram  (threshold = {threshold:.0e} cm$^{{-2}}$)",
    fontsize=10,
)

cbar = fig.colorbar(mesh, ax=ax, ticks=range(n_phases), pad=0.02)
cbar.set_label("phase (occupation + degeneracy)")
cbar.ax.set_yticklabels(config.PHASE_ORDER)   # human-readable labels directly on the colorbar

# Phase boundaries as contours on the integer field
# try:
#     ax.contour(
#         n_vals, D_vals * 1e3, phase_map,
#         levels=np.arange(0.5, 4.5, 1.0),
#         colors="k", linewidths=0.6, alpha=0.5,
#     )
# except Exception:
#     pass    # contouring fails harmlessly if the map has no interior boundaries
 
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
fig.tight_layout()
 
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out = config.FIGURES_DIR / f"phase_diagram_nD_{n_pts}pts.png"
# fig.savefig(out, dpi=300, bbox_inches="tight")
# print(f"Saved figure: {out}")
 
plt.show()