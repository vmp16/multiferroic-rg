"""
get_relative_dens.py
====================
Plot the relative flavor densities n_alpha / n along a row (fixed D) of the (n, D) phase diagram saved by scan_phase_diagram_PIP.py.

Points are coloured by phase, using the same discrete colormap as plot_phase_diagram_PIP.py, so both figures can be read side by side.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

import model.config as config

# ----------------------------------------------------------
#                       Options
# ----------------------------------------------------------
iD = 50                   # row of the diagram (fixed D)
SORT_FLAVORS = True       # sort flavors at each n (largest first) so curves are continuous
SHOW_BACKGROUND = True    # shade the background with the phase colours
SAVE_FIG = False

# ----------------------------------------------------------
#               Load the phase diagram data
# ----------------------------------------------------------
data = np.load(config.DATA_DIR / "phase_diagram_nD_PIP_170pts.npz")

n_vals = data["n_vals"]
D_vals = data["D_vals"]
phase_map = data["phase_map"].astype(float)
density_map = data["density_map"]

phase_map[phase_map < 0] = np.nan      # failed points -> blank

print(f"Calculating relative densities (frequencies) for D = {D_vals[iD]*1e3:.2f} meV ...")

# ----------------------------------------------------------
#           Calculate the relative densities
# ----------------------------------------------------------
row = density_map[iD, :]               # shape (n_pts, 4)
n_rels = row / n_vals[:, None]         # shape (n_pts, 4)
phase_row = phase_map[iD, :]           # shape (n_pts,)

if SORT_FLAVORS:
    # Flavor labels can swap between neighbouring points (random seeds),
    # sorting gives a clean, label-independent picture.
    n_rels = -np.sort(-n_rels, axis=1)

# Discard failed points
ok = np.isfinite(phase_row) & np.all(np.isfinite(n_rels), axis=1)

# ----------------------------------------------------------
#   Discrete colormap (identical to the phase diagram plot)
# ----------------------------------------------------------
n_phases = len(config.PHASE_ORDER)
cmap = mpl.colormaps["viridis"].resampled(n_phases)
norm = BoundaryNorm(np.arange(-0.5, n_phases, 1.0), ncolors=n_phases)

def edges(v):
    """Cell edges from cell centres (for pcolormesh)."""
    d = np.diff(v)
    return np.concatenate([[v[0] - d[0] / 2], v[:-1] + d / 2, [v[-1] + d[-1] / 2]])

# ----------------------------------------------------------
#                   Plot the result
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.8))

Y_LO, Y_HI = -0.04, 1.04

# Faint phase-coloured background strip
if SHOW_BACKGROUND:    
    ax.pcolormesh(
        edges(n_vals), [Y_LO, Y_HI], np.ma.masked_invalid(phase_row)[None, :],
        cmap=cmap, norm=norm, alpha=0.18, shading="flat", zorder=0,
    )

# Faint connecting lines, one per flavor, to guide the eye
if SORT_FLAVORS:
    for a in range(n_rels.shape[1]):
        ax.plot(n_vals[ok], n_rels[ok, a], '-', color='0.55', lw=0.5, alpha=0.5, zorder=1)

# Points coloured by phase
for a in range(n_rels.shape[1]):
    ax.scatter(
        n_vals[ok], n_rels[ok, a],
        c=phase_row[ok], cmap=cmap, norm=norm,
        s=14, edgecolors="k", linewidths=0.3, zorder=2,
    )

# Reference lines at the symmetric fractions (1, 1/2, 1/3, 1/4)
for frac in (1, 1/2, 1/3, 1/4):
    ax.axhline(frac, color='0.7', lw=0.5, ls=':', zorder=0.5)

ax.set_xlim(n_vals.min(), n_vals.max())
ax.set_ylim(Y_LO, Y_HI)

plt.yscale("log")
plt.ylim(bottom=1e-13)

ax.set_xlabel(r"Carrier density $n$  [cm$^{-2}$]", fontsize=12)
ax.set_ylabel(r"Relative density  $n_\alpha / n$", fontsize=12)
ax.set_title(rf"Flavor frequencies at $D = {D_vals[iD]*1e3:.2f}$ meV", fontsize=12)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
# ax.set_yticks([0, 1/4, 1/3, 1/2, 1])
# ax.set_yticklabels(["0", "1/4", "1/3", "1/2", "1"])

# Legend: only the phases that actually appear on this row
present = np.unique(phase_row[ok]).astype(int)
handles = [
    Line2D([0], [0], marker='o', ls='', markersize=7,
           markerfacecolor=cmap(norm(p)), markeredgecolor='k', markeredgewidth=0.3,
           label=config.PHASE_ORDER[p])
    for p in present
]
ax.legend(handles=handles, title="Phase", fontsize=12, title_fontsize=12,
          loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

fig.tight_layout()

if SAVE_FIG:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / f"relative_densities_D{D_vals[iD]*1e3:.2f}meV.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out}")

plt.show()