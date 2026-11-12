"""
NOTE: made by Claude.ai

Visual/sanity test of the direct Fock self-energy for ONE flavor.

Path: p = (p_x, 0), p_x in [0, P_MAX]  [units of 1/a].
Output: a table of delta_f, m_f at a few p, a figure (delta_f and m_f along the path,
        for the "half" and "valence" references) and a K_LIM cutoff check.

Assumes get_fock_self_energy_direct(rho, KX, KY, e2_over_eps, p_points) is importable
from model.analysis (adjust the import if you placed it elsewhere).

Run:  python test_fock_direct.py
"""

import numpy as np
import matplotlib.pyplot as plt

import model.config as config
from model.analysis import (
    get_kmesh,
    build_systems_with_displacement,
    get_density_matrix,
    get_flavor_mu_from_densities,
    get_fock_self_energy_direct,
)

# ---------------- parameters ----------------
FLAVOR   = 0
D        = 0.02            # displacement field [eV]
N_TARGET = -1.0e12         # density of this flavor [cm^-2] (negative = hole doping)
EPS_R    = 10.0            # dielectric constant
E2_OVER_EPS = config.e2_over_eps / EPS_R

K_LIM  = 0.15
DK     = 6e-4              # mesh spacing [1/a] (kept fixed when K_LIM changes)
P_MAX  = 0.13
N_PATH = 60

system = build_systems_with_displacement(D, config)[0][FLAVOR]


def make_mesh(k_lim):
    return get_kmesh(k_lim, 2 * int(round(k_lim / DK)))      # even number of points


def pauli(S):
    """(M0, Mx, My, Mz) of a (2,2,...) array."""
    return (0.5 * (S[0, 0] + S[1, 1]).real, S[0, 1].real,
            -S[0, 1].imag, 0.5 * (S[0, 0] - S[1, 1]).real)


def fock(KX, KY, mu, reference, p_points, circular=False):
    rho = get_density_matrix(system, KX, KY, mu, config.T_eff, reference=reference)
    if circular:                                             # round cutoff |k| <= K_LIM
        rho = rho * (np.hypot(KX, KY) <= KX.max())
    return pauli(get_fock_self_energy_direct(rho, KX, KY, E2_OVER_EPS, p_points))


# ---------------- chemical potential for the target density ----------------
KX, KY = make_mesh(K_LIM)
dk = KX[0, 1] - KX[0, 0]
prefactor = dk**2 / (2 * np.pi) ** 2
bands = system.get_energy(KX, KY)
mu = get_flavor_mu_from_densities([N_TARGET], [bands], config.T_eff, prefactor,
                                  config.unit_cell_to_cm2, mu_min=-0.2, mu_max=0.2)[0]
print(f"Flavor {FLAVOR}, D = {D*1e3:.1f} meV, n = {N_TARGET:.2e} cm^-2  ->  mu = {mu*1e3:.3f} meV")
print(f"Mesh: {KX.shape[0]}^2, dk = {dk:.2e}, K_LIM = {K_LIM},  e2/(eps a) = {E2_OVER_EPS:.3f} eV")

# ---------------- Fock along the path ----------------
px = np.linspace(0.0, P_MAX, N_PATH)
p_path = (px, np.zeros_like(px))
res = {ref: fock(KX, KY, mu, ref, p_path) for ref in ("half", "valence")}

print("\n p_x [1/a] | delta_f half | m_f half | delta_f valence | m_f valence   [meV]")
for i in np.linspace(0, N_PATH - 1, 7).astype(int):
    print(f" {px[i]:9.4f} | {res['half'][0][i]*1e3:12.3f} | {res['half'][3][i]*1e3:8.3f} |"
          f" {res['valence'][0][i]*1e3:15.3f} | {res['valence'][3][i]*1e3:11.3f}")

# ---------------- figure ----------------
E_up, E_dn = system.get_energy(*p_path)
kF = [px[j] for E in (E_up, E_dn) for j in np.where(np.diff(np.sign(E - mu)) != 0)[0]]

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for ax, idx, name in ((axs[0], 0, r"$\delta_f$"), (axs[1], 3, r"$m_f$")):
    for ref, style in (("half", "-"), ("valence", "--")):
        ax.plot(px, res[ref][idx] * 1e3, style, label=f'reference "{ref}"')
    for k in kF:
        ax.axvline(k, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$p_x$  [1/a]")
    ax.set_ylabel(name + "  [meV]")
    ax.legend()
axs[0].set_title(f"flavor {FLAVOR}, D = {D*1e3:.0f} meV, dotted = Fermi crossings")
fig.tight_layout()
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out = config.FIGURES_DIR / "fock_direct_path.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved figure: {out}")

# ---------------- cutoff check at one p (mu fixed) ----------------
p0 = (np.array([0.05]), np.array([0.0]))
print(f"\nK_LIM check at p = (0.05, 0), fixed dk and mu   [meV]")
print(" K_LIM | setting            | delta_f |   m_f   | |Sigma_xy|")
for k_lim in (0.13, 0.16, 0.19):
    KXc, KYc = make_mesh(k_lim)
    for ref, circ in (("half", False), ("half", True), ("valence", False)):
        d0, sx, sy, mz = (v[0] for v in fock(KXc, KYc, mu, ref, p0, circular=circ))
        tag = f"{ref}{' + circular' if circ else ''}"
        print(f" {k_lim:5.2f} | {tag:18s} | {d0*1e3:7.3f} | {mz*1e3:7.3f} | {np.hypot(sx, sy)*1e3:8.3f}")

plt.show()