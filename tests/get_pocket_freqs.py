import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import model.config as config
from model.analysis import (
    get_kmesh,
    build_systems_with_displacement,
    precompute_flavor_bands,
    build_tables_unique,
    get_or_build_tables,
    get_orbit_areas,
    merge_orbits,
)

iD = 60
THRESHOLD = 1e6          # flavor counted as occupied [cm^-2]
SAVE_FIG = True

# ------------ Load the phase diagram ------------
filename = "phase_diagram_nD_150pts.npz"

print(f"  Reading phase diagram scan: {filename}...")

with np.load(config.DATA_DIR / filename, allow_pickle=True) as f:
    data = {key: f[key] for key in f.files}

n_vals, D_vals = data["n_vals"], data["D_vals"]
density_map = data["density_map"]
phase_row = data["phase_map"][iD].astype(float)
phase_row[phase_row < 0] = np.nan
D = D_vals[iD]

print(f"  Analysing cut at D = {D*1e3:.2f} meV...")

# ------------ Build k-mesh and tables ------------
print("  Loading the density tables...")
# Same mesh and tables as the scan
KX, KY = get_kmesh(float(data["k_lim_scan"]), int(data["n_pts_scan"]))
dk = KX[0, 1] - KX[0, 0]
prefactor = dk**2 / (2 * np.pi)**2

systems, _ = build_systems_with_displacement(D, config)
bands = precompute_flavor_bands(systems, KX, KY)

# mu_table, n_table, _ = build_tables_unique(
#     systems, KX, KY, config.T_eff, prefactor, config.unit_cell_to_cm2,
#     mu_min=-0.20, mu_max=0.20, n_table_pts=2000)

del systems

mu_table, n_table, _ = get_or_build_tables(
    D, config, KX, KY, cache_dir=config.DATA_DIR / "tables_cache",
    mu_min=-0.20, mu_max=0.20, n_table_pts=2000,
)

# ====================================================================
#                   Main Loop: Search for pockets
# ====================================================================

print("  MAIN: Looking for pockets along the cut...")
# Orbits along the row
pts = []   # (n, f/|n|, phase, inner, trunc, mult)
for iN, n in enumerate(n_vals):
    if not np.isfinite(phase_row[iN]):
        continue
    orbits = []
    for a in range(4):
        n_a = density_map[iD, iN, a]
        if abs(n_a) < THRESHOLD or not (n_table[a, 0] <= n_a <= n_table[a, -1]):
            continue
        mu_a = np.interp(n_a, n_table[a], mu_table)     # invert n_a(mu_a)
        orbits += get_orbit_areas(
            bands[a], mu_a, dk, config.unit_cell_to_cm2,
            min_pixels=15, connectivity=1,
        )
    for m in merge_orbits(orbits):
        pts.append((n, m["f_cm2"] / abs(n), phase_row[iN], m["inner"], m["trunc"], m["mult"]))

pts = np.array(pts, dtype=float)
n_p, f_p, ph, inner, trunc, mult = pts.T

# ----------- Diagnosis: no points near n~0 ------------
# from scipy import ndimage
# for iN in range(len(n_vals) - 3, len(n_vals)):
#     for a in range(4):
#         E0, E1 = bands[a]
#         print(f"mu = {mu_a*1e3:.3f} meV | max(E1) = {E1.max()*1e3:.3f} | min(E0) = {E0.min()*1e3:.3f} | kT = {config.T_eff*1e3:.3f}")
#         n_a = density_map[iD, iN, a]
#         if abs(n_a) < THRESHOLD: continue
#         mu_a = np.interp(n_a, n_table[a], mu_table)
#         for name, mask in (("cond", bands[a][0] < mu_a), ("val", bands[a][1] > mu_a)):
#             lab, nl = ndimage.label(mask)
#             sizes = np.bincount(lab.ravel())[1:]
#             print(f"n={n_vals[iN]:.3e} a={a} {name}: True px={mask.sum()}, "
#                   f"components={nl}, largest={sizes.max() if nl else 0}")

# -------------- Plot the Resulting Normalized Frequencies ---------------
print("  DONE. Building the plot...")
def edges(v):
    d = np.diff(v)
    return np.concatenate([[v[0] - d[0] / 2], v[:-1] + d / 2, [v[-1] + d[-1] / 2]])

# Phase = background colors
n_phases = len(config.PHASE_ORDER)
cmap_ph = mpl.colormaps["viridis"].resampled(n_phases)
norm_ph = mpl.colors.BoundaryNorm(np.arange(-0.5, n_phases, 1.0), n_phases)

# Multiplicity = discrete marker colors
mult_vals = np.unique(mult).astype(int)
m_idx = np.searchsorted(mult_vals, mult)
cmap_m = mpl.colormaps["plasma"].resampled(len(mult_vals))
norm_m = mpl.colors.BoundaryNorm(np.arange(-0.5, len(mult_vals), 1.0), len(mult_vals))

fig, ax = plt.subplots(figsize=(7, 4.8))

# Background strip: x in data coordinates, y in axes coordinates (works with log y)
ne = edges(n_vals)
for i in range(len(n_vals)):
    if np.isfinite(phase_row[i]):
        ax.axvspan(ne[i], ne[i + 1], color=cmap_ph(norm_ph(int(phase_row[i]))),
                   alpha=0.25, lw=0, zorder=0)

# Phase boundaries: where the phase index changes between neighbouring n points
valid = np.isfinite(phase_row)
change = np.where(valid[:-1] & valid[1:] & (phase_row[:-1] != phase_row[1:]))[0]
n_bound = 0.5 * (n_vals[change] + n_vals[change + 1])

for nb in n_bound:
    ax.axvline(nb, color="k", lw=0.6, ls="--", alpha=0.3, zorder=0.5)

# Plot the values
for sel, mk in ((inner == 0, "o"), (inner == 1, "^")):
    sc = ax.scatter(n_p[sel], f_p[sel], c=m_idx[sel], cmap=cmap_m, norm=norm_m,
                    marker=mk, s=22, zorder=3,
                    edgecolors=np.where(trunc[sel] > 0, "r", "k"), linewidths=0.5)

# Highlight integer fractions
for frac in (1, 1/2, 1/3, 1/4):
    ax.axhline(frac, color="0.7", lw=0.5, ls=":", zorder=0)

# Style the plot
# ax.set_yscale("log")
ax.set_xlim(n_vals.min(), n_vals.max())
# ax.set_ylim(f_p.min() / 1.5, 1.5)

ax.set_xlabel(r"Carrier density $n$ [cm$^{-2}$]", fontsize=13)
ax.set_ylabel(r"$f_v = n_{\rm pocket}/|n|$", fontsize=13)
ax.set_title(rf"Fermi-pocket frequencies, $D={D*1e3:.2f}$ meV", fontsize=13)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
ax.tick_params(labelsize=12)

cbar = fig.colorbar(sc, ax=ax, ticks=range(len(mult_vals)), pad=0.02)
cbar.set_ticklabels(mult_vals)
cbar.set_label("multiplicity", fontsize=12)

present = np.unique(ph).astype(int)
handles = [Patch(facecolor=cmap_ph(norm_ph(p)), alpha=0.4, label=config.PHASE_ORDER[p])
           for p in present]
handles += [
    Line2D([0], [0], marker="o", ls="", mfc="w", mec="k", label="outer"),
    Line2D([0], [0], marker="^", ls="", mfc="w", mec="k", label="inner"),
    # Line2D([0], [0], marker="o", ls="", mfc="w", mec="r", label="truncated"),
]
# ax.legend(handles=handles, title="Phase / orbit", loc="upper center", fontsize=12,
#           bbox_to_anchor=(0.5, -0.18), ncol=min(len(handles), 6), frameon=False)

fig.tight_layout()

if SAVE_FIG:
    out_path = config.FIGURES_DIR / f"pocket_freqs_D{D*1e3:.2f}meV.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Figure saved: {out_path}")
plt.show()