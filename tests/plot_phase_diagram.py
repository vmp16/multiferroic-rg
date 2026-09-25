"""
plot_phase_diagram.py
=====================
Render the (n, D) phase diagram saved by scan_phase_diagram.py.

Kept separate from the scan so the figure can be restyled without
recomputing anything.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import model.config as config

data = np.load(config.DATA_DIR / "phase_diagram_nD_80pts.npz")

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
iD = 40
iN = 36
dens_print = density_map[iD, iN]
print(f"Ground state for (n, D) = ({n_vals[iN]:.4e}, {D_vals[iD]*1e3:.2f}):")
print("Densities: ", dens_print)

# ---------------------------------------------------------
#  Discrete colour scale, one colour per occupation number
# ---------------------------------------------------------

levels = [0, 1, 2, 3, 4]
colors = ["#f0f0f0", "#4575b4", "#91bfdb", "#fc8d59", "#d73027"]
cmap = ListedColormap(colors)
cmap.set_bad("white")
norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)

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

ax.plot(
    n_vals[iN], D_vals[iD]*1e3,
    marker="o", markersize=7,
    color='black',
    # markerfacecolor="none", markeredgecolor="black", markeredgewidth=2,
    # zorder=5,
)

ax.set_xlabel(r"Carrier density $n$  [cm$^{-2}$]")
ax.set_ylabel(r"Displacement field $D$  [meV]")
ax.set_title(
    rf"Flavor polarization phase diagram  (threshold = {threshold:.0e} cm$^{{-2}}$)",
    fontsize=10,
)

cbar = fig.colorbar(mesh, ax=ax, ticks=levels, pad=0.02)
cbar.set_label("Number of active flavors")
cbar.ax.set_yticklabels([str(l) for l in levels])

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
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved figure: {out}")

plt.show()