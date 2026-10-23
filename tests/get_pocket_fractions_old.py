"""
get_pocket_fractions.py
=======================
Plot the carrier density carried by every Fermi pocket, n_pocket / n_tot, as a
function of the total density n, along a row (fixed D) of the (n, D) phase
diagram saved by scan_phase_diagram_PIP.py.

There is ONE CURVE PER POCKET. For each n:
  1. take the converged flavor densities n_alpha from the phase diagram,
  2. invert n_alpha(mu) on the SAME tables the scan used -> mu_alpha,
  3. label the Fermi pockets of every flavor (get_fermi_pockets); the density
     of a pocket is the area enclosed by its contour / (2 pi)^2,
  4. divide by n_tot and rank the pockets by size (largest first, separately
     for each group: hole / electron, outer / nested).
The k-th curve of a group joins the k-th largest pocket at every n. Identical
pockets (e.g. four equal flavors, or the three lobes of a trigonally warped
pocket) give curves lying on top of each other; a Lifshitz transition shows up
as a curve appearing/disappearing or a jump in the values. Flavor labels are
not followed (they can swap between neighbouring points), only sizes.

Signs: n_pocket / n_tot > 0 for pockets of the same character as the total
doping (holes when n < 0), < 0 for the opposite character (SIGNED = False
plots absolute values). Summing all fractions at one n gives 1, up to the
thermal remainder. Nested pockets (a hole inside a pocket) are open markers
with dotted lines; their enclosed area is subtracted from the outer one in
the sum rule, which is why they carry the opposite sign.

The background is shaded with the phase colours of plot_phase_diagram_PIP.py.

The pocket calculation is slow (one k-sum table per row, like one row of the
scan), so results are cached in DATA_DIR. Set RECOMPUTE = False to restyle the
figure without recomputing.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import model.config as config
from model.analysis import (
    get_kmesh,
    build_systems_with_displacement,
    precompute_flavor_bands,
    build_tables_unique,
    build_density_and_energy_tables,
    get_flavor_mu_from_densities,
    get_fermi_pockets,
)

# ----------------------------------------------------------
#                       Options
# ----------------------------------------------------------
iD = 50                   # row of the diagram (fixed D)
RECOMPUTE = True          # False: reuse the cached pocket results
MIN_PIXELS = 20           # pockets enclosing fewer mesh points are discarded
YSCALE = "symlog"         # "symlog" or "linear"
LINTHRESH = 1e-2          # linear region of the symlog axis
SHOW_BACKGROUND = True    # shade the background with the phase colours
SHOW_RESIDUAL = False     # also plot (n_tot - sum of pocket densities) / n_tot
SIGNED = True             # False: plot |n_pocket| / |n_tot|
REMAINDER_FLAG = 0.10     # hatch the n where |n_tot - sum of pockets| / |n_tot| exceeds this
                          # (T = 0 pocket picture unreliable at the temperature T_eff); None = off
SAVE_FIG = False

DATA_FILE = "phase_diagram_nD_PIP_170pts.npz"

# Table settings: MUST match scan_phase_diagram_PIP.py, so that mu_alpha is the one the SCF used.
MU_MIN, MU_MAX = -0.20, 0.20
N_TABLE = 2000

# ----------------------------------------------------------
#               Load the phase diagram data
# ----------------------------------------------------------
data = np.load(config.DATA_DIR / DATA_FILE)

n_vals = data["n_vals"]
D_vals = data["D_vals"]
phase_map = data["phase_map"].astype(float)
density_map = data["density_map"]
N_PTS_SCAN = int(data["n_pts_scan"])
K_LIM_SCAN = float(data["k_lim_scan"])

if not (np.isclose(float(data["T_eff"]), config.T_eff) and np.isclose(float(data["U"]), config.U)):
    raise RuntimeError("T_eff or U in config differ from the ones used in the scan.")

phase_map[phase_map < 0] = np.nan      # failed points -> blank
phase_row = phase_map[iD, :]
D = D_vals[iD]
n_pts = len(n_vals)

cache_path = config.DATA_DIR / f"pocket_fractions_iD{iD}.npz"


# ----------------------------------------------------------
#                 Pocket calculation along the row
# ----------------------------------------------------------
def compute_pocket_row():
    print(f"Pocket analysis along the row D = {D*1e3:.2f} meV ({n_pts} densities)...")

    KX, KY = get_kmesh(K_LIM_SCAN, N_PTS_SCAN)
    dk = KX[0, 1] - KX[0, 0]
    prefactor = (dk**2) / (2 * np.pi)**2

    # Bands depend on D only: build them (and the tables) once for the whole row
    systems, _ = build_systems_with_displacement(D, config)
    # all_bands = precompute_flavor_bands(systems, KX, KY)
    # del systems

    print("   Building the n_alpha(mu) tables (same cost as one row of the scan)...")
    # mu_table, n_table, _ = build_density_and_energy_tables(
    #     all_bands, config.T_eff, prefactor, config.unit_cell_to_cm2,
    #     mu_min=MU_MIN, mu_max=MU_MAX, n_table_pts=N_TABLE,
    # )
    mu_table, n_table, _ = build_tables_unique(
        systems, KX, KY, config.T_eff, prefactor, config.unit_cell_to_cm2,
        mu_min=MU_MIN, mu_max=MU_MAX, n_table_pts=N_TABLE
    )

    rec_iN, rec_flavor, rec_electron, rec_inner, rec_ratio = [], [], [], [], []
    n_pockets = np.full(n_pts, np.nan)
    resid = np.full(n_pts, np.nan)
    n_edge = 0

    for iN, n_target in enumerate(n_vals):
        dens = density_map[iD, iN]
        if not np.all(np.isfinite(dens)):
            continue            # failed point of the scan

        try:
            mu_flavs = get_flavor_mu_from_densities(
                dens, all_bands, config.T_eff, prefactor, config.unit_cell_to_cm2,
                tables=(mu_table, n_table),
            )
        except ValueError as exc:
            print(f"   [n = {n_target:.3e}] skipped: {exc}")
            continue

        count, n_sum = 0, 0.0
        for f in range(4):
            pockets, _, _ = get_fermi_pockets(
                all_bands[f], mu_flavs[f], KX, KY, dk, config.unit_cell_to_cm2,
                min_pixels=MIN_PIXELS,
            )
            for p in pockets:
                rec_iN.append(iN)
                rec_flavor.append(f)
                rec_electron.append(p["kind"] == "electron")
                rec_inner.append(p["inner"])
                rec_ratio.append(p["n_cm2"] / n_target)
                n_sum += p["n_cm2"]
                n_edge += p["touches_edge"]
                count += 1

        n_pockets[iN] = count
        resid[iN] = (n_target - n_sum) / n_target

        if iN % 10 == 0:
            print(f"   {iN + 1}/{n_pts} done")

    if n_edge:
        print(f"   WARNING: {n_edge} pocket(s) touch the mesh edge (truncated areas).")

    out = dict(
        iD=iD, D=D,
        iN=np.array(rec_iN, dtype=int),
        flavor=np.array(rec_flavor, dtype=int),
        electron=np.array(rec_electron, dtype=bool),
        inner=np.array(rec_inner, dtype=bool),
        ratio=np.array(rec_ratio, dtype=float),
        n_pockets=n_pockets,
        resid=resid,
    )
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **out)
    print(f"   Saved: {cache_path}")
    return out


if RECOMPUTE or not cache_path.exists():
    res = compute_pocket_row()
else:
    cached = np.load(cache_path)
    if not np.isclose(float(cached["D"]), D):
        raise RuntimeError("Cached results belong to a different D, set RECOMPUTE = True.")
    res = {k: cached[k] for k in cached.files}
    print(f"Loaded cached pocket results from {cache_path}")

rec_iN = res["iN"]
rec_electron = res["electron"]
rec_inner = res["inner"]
rec_ratio = res["ratio"]
n_pockets = res["n_pockets"]
resid = res["resid"]

print(f"Max |remainder| / n_tot over the row: {np.nanmax(np.abs(resid)):.3e} "
      f"(thermal tails and the T = 0 pocket definition)")

flagged = np.zeros(n_pts, dtype=bool)
if REMAINDER_FLAG is not None:
    flagged = np.isfinite(resid) & (np.abs(resid) > REMAINDER_FLAG)
    if flagged.any():
        print(f"WARNING: {flagged.sum()}/{n_pts} densities have a remainder above "
              f"{REMAINDER_FLAG:.0%} (hatched in the plot): pockets are shallow compared "
              f"to T_eff = {config.T_eff*1e3:.2f} meV there.")

# ----------------------------------------------------------
#   Discrete colormap (identical to the phase diagram plot)
# ----------------------------------------------------------
n_phases = len(config.PHASE_ORDER)
cmap = mpl.colormaps["viridis"].resampled(n_phases)
norm = BoundaryNorm(np.arange(-0.5, n_phases, 1.0), ncolors=n_phases)


def edges(v):
    """Cell edges from cell centres."""
    d = np.diff(v)
    return np.concatenate([[v[0] - d[0] / 2], v[:-1] + d / 2, [v[-1] + d[-1] / 2]])


# ----------------------------------------------------------
#      One curve per pocket: rank pockets by size at each n
# ----------------------------------------------------------
def rank_curves(is_electron, inner):
    """
    y[k, iN] = k-th largest (in |fraction|) pocket of this group at density
    index iN, NaN where there are fewer than k+1 pockets. Returns None if the
    group never occurs on this row.
    """
    sel = (rec_electron == is_electron) & (rec_inner == inner)
    if not sel.any():
        return None
    counts = np.bincount(rec_iN[sel], minlength=n_pts)
    y = np.full((counts.max(), n_pts), np.nan)
    for iN in np.nonzero(counts)[0]:
        vals = rec_ratio[sel & (rec_iN == iN)]
        y[:len(vals), iN] = vals[np.argsort(-np.abs(vals), kind="stable")]
    return y if SIGNED else np.abs(y)


# ----------------------------------------------------------
#                   Plot the result
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))

# Faint phase-coloured background
if SHOW_BACKGROUND:
    n_edges = edges(n_vals)
    for i in range(n_pts):
        if np.isfinite(phase_row[i]):
            ax.axvspan(n_edges[i], n_edges[i + 1], color=cmap(norm(phase_row[i])),
                       alpha=0.18, lw=0, zorder=0)

# Hatch the densities where the T = 0 pocket picture does not reproduce n_tot
if flagged.any():
    n_edges = edges(n_vals)
    for i in np.nonzero(flagged)[0]:
        ax.axvspan(n_edges[i], n_edges[i + 1], facecolor="none", edgecolor="0.45",
                   hatch="///", alpha=0.6, lw=0, zorder=1)

if YSCALE == "symlog":
    ax.set_yscale("symlog", linthresh=LINTHRESH)
ax.margins(y=0.12)

rank_colors = mpl.colormaps["tab10"].colors
groups = [   # (is_electron, inner, marker)
    (False, False, "o"),
    (True,  False, "s"),
    (False, True,  "o"),
    (True,  True,  "s"),
]
for is_el, inner, marker in groups:
    y = rank_curves(is_el, inner)
    if y is None:
        continue
    for k in range(y.shape[0]):
        col = rank_colors[k % len(rank_colors)]
        # Larger markers for larger pockets, so identical (overlaid) curves stay visible
        size = max(10.0, 46.0 - 9.0 * k)
        ax.plot(n_vals, y[k], ls=":" if inner else "-", color=col, lw=1.0, alpha=0.8, zorder=2)
        if inner:
            ax.scatter(n_vals, y[k], s=size + 8, marker=marker, facecolors="none",
                       edgecolors=[col], linewidths=1.1, zorder=3)
        else:
            ax.scatter(n_vals, y[k], s=size, marker=marker, color=col,
                       edgecolors="k", linewidths=0.3, zorder=3)

if SHOW_RESIDUAL:
    ax.plot(n_vals, resid, '-', color='0.4', lw=0.8, zorder=1)

ax.axhline(0, color='0.6', lw=0.5, ls=':', zorder=0.5)
ax.axhline(1, color='0.6', lw=0.5, ls=':', zorder=0.5)
for frac in (1/2, 1/3, 1/4):
    ax.axhline(frac, color='0.75', lw=0.5, ls=':', zorder=0.5)

ax.set_xlim(n_vals.min(), n_vals.max())
ax.set_xlabel(r"Carrier density $n$  [cm$^{-2}$]", fontsize=12)
ax.set_ylabel(r"Pocket density fraction  $n_{\rm pocket} / n$", fontsize=12)
ax.set_title(rf"Pocket fractions at $D = {D*1e3:.2f}$ meV", fontsize=12)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

# Legends: phases present on this row, and the marker code
present = np.unique(phase_row[np.isfinite(phase_row)]).astype(int)
phase_handles = [
    Patch(facecolor=cmap(norm(p)), alpha=0.35, label=config.PHASE_ORDER[p]) for p in present
]
marker_handles = [
    Line2D([0], [0], marker='o', ls='-', color='0.5', markersize=7, markeredgecolor='k',
           markeredgewidth=0.3, label="hole pocket"),
    Line2D([0], [0], marker='s', ls='-', color='0.5', markersize=7, markeredgecolor='k',
           markeredgewidth=0.3, label="electron pocket"),
    Line2D([0], [0], marker='o', ls=':', color='0.3', markersize=7, markerfacecolor='none',
           markeredgewidth=1.2, label="nested (inner)"),
]
if flagged.any():
    marker_handles.append(Patch(facecolor="none", edgecolor="0.45", hatch="///",
                                label=f"remainder > {REMAINDER_FLAG:.0%}"))
leg1 = ax.legend(handles=phase_handles, title="Phase", fontsize=11, title_fontsize=11,
                 loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
ax.add_artist(leg1)
ax.legend(handles=marker_handles, title="Pockets\n(colour = size rank)", fontsize=10,
          title_fontsize=10, loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=False)

fig.tight_layout()

if SAVE_FIG:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / f"pocket_fractions_D{D*1e3:.2f}meV.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out}")

plt.show()