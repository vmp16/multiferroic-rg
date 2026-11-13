"""
Convergence of the Fock self-energy with the k-mesh resolution dk (FFT version), at fixed K_LIM and fixed mu.

Meshes: odd N = 2M+1 on [-K_LIM, K_LIM], so dk = K_LIM/M and the meshes are NESTED: every point of a
coarse mesh is also a point of the finer ones. Sigma can therefore be compared point by point against the
finest mesh without interpolation. mu is fixed from the finest mesh, so only dk changes between runs.

Compared along the line p = (p_x, 0), 0 <= p_x <= P_MAX, for delta_f = Sigma_0, m_f = Sigma_z and |Sigma_xy|,
for the "cnp" and "half" references.

Memory: the finest mesh (M = 1000, N = 2001) needs about 1.5 GB. Drop 1000 from MS if that is too much.

Run:  python test_fock_convergence.py
"""

import time
import numpy as np

import model.config as config
from model.analysis import (
    get_kmesh,
    build_systems_with_displacement,
    get_density_matrix,
    get_flavor_density_from_bands,
    get_flavor_mu_from_densities,
    build_coulomb_kernels_fft,
    get_fock_self_energy_fft,
)

# ---------------- parameters ----------------
FLAVOR   = 0
D        = 0.02                   # displacement field [eV]
N_TARGET = -1.0e12                # density of this flavor [cm^-2] -> fixes mu on the finest mesh
K_LIM    = 0.15
MS       = [125, 250, 500, 1000]  # dk = K_LIM/M = 12, 6, 3, 1.5 (x1e-4); N = 2M+1
REFS     = ("cnp", "half")
P_MAX    = 0.13                   # path p = (p_x, 0), p_x in [0, P_MAX]  [1/a]
P_SHOW   = 0.06                   # p_x at which the values are printed (on every mesh: 0.06/dk integer)
TOL_MEV  = 0.1                    # verdict: second-finest vs finest mesh

E2       = config.e2_over_eps
D_LAYERS = config.d_layers

systems, base_deltas = build_systems_with_displacement(D, config)
system, delta_ref = systems[FLAVOR], float(base_deltas[FLAVOR])

# ---------------- run all meshes, finest first (mu is fixed there) ----------------
results, dens, mu = {}, {}, None
M_F = max(MS)

for M in sorted(MS, reverse=True):
    t0 = time.time()
    N = 2 * M + 1
    KX, KY = get_kmesh(K_LIM, N)
    dk = KX[0, 1] - KX[0, 0]
    prefactor = dk**2 / (2 * np.pi) ** 2

    bands = system.get_energy(KX, KY)
    if mu is None:
        mu = get_flavor_mu_from_densities([N_TARGET], [bands], config.T_eff, prefactor,
                                          config.unit_cell_to_cm2, mu_min=-0.2, mu_max=0.2)[0]
        print(f"Flavor {FLAVOR}, D = {D*1e3:.1f} meV, n = {N_TARGET:.2e} cm^-2 -> mu = {mu*1e3:.3f} meV "
              f"(from the finest mesh, then fixed)")
    dens[M] = get_flavor_density_from_bands(bands, mu, config.T_eff, prefactor, config.unit_cell_to_cm2)
    del bands

    ks_fft, kd_fft, L = build_coulomb_kernels_fft(N, dk, E2, D_LAYERS)
    for ref in REFS:
        rho = get_density_matrix(system, KX, KY, mu, config.T_eff, reference=ref, Delta_ref=delta_ref)
        S = get_fock_self_energy_fft(rho, ks_fft, kd_fft, L, dk)
        del rho
        row = S[:, :, M, :].copy()          # ky = 0 row (axis 0 is y), shape (2, 2, N)
        del S
        results[(M, ref)] = dict(
            dk=dk,
            d0=0.5 * (row[0, 0] + row[1, 1]).real,
            dz=0.5 * (row[0, 0] - row[1, 1]).real,
            dxy=np.abs(row[0, 1]),
        )
    del ks_fft, kd_fft, KX, KY
    print(f"  M = {M:5d} (N = {N:5d}, dk = {dk:.2e}) done in {time.time() - t0:.1f} s")

# ---------------- compare with the finest mesh ----------------
verdict_ok = True
for ref in REFS:
    print(f'\nReference "{ref}"   (errors vs finest mesh, along p_x in [0, {P_MAX}], all in meV)')
    print("    M |     dk    | n/n_target-1 | delta_f(p) | m_f(p)  | max|d delta| | max|d m| | max|d |Sxy||")
    rf = results[(M_F, ref)]
    for M in sorted(MS):
        r = results[(M, ref)]
        s = M_F // M
        n_pts = int(P_MAX / r["dk"]) + 1
        sl_i = slice(M, M + n_pts)                           # p_x = 0 ... on mesh M
        sl_f = slice(M_F, M_F + s * n_pts, s)                # same p_x points on the finest mesh
        errs = {}
        for key in ("d0", "dz", "dxy"):
            a, b = r[key][sl_i], rf[key][sl_f]
            assert a.shape == b.shape, "mesh alignment problem"
            errs[key] = np.max(np.abs(a - b)) * 1e3
        i_show = M + int(round(P_SHOW / r["dk"]))
        print(f" {M:5d} | {r['dk']:.2e} | {dens[M] / N_TARGET - 1:12.2e} | "
              f"{r['d0'][i_show] * 1e3:10.3f} | {r['dz'][i_show] * 1e3:7.3f} | "
              f"{errs['d0']:12.4f} | {errs['dz']:8.4f} | {errs['dxy']:14.4f}")
        if M == sorted(MS)[-2] and max(errs.values()) > TOL_MEV:
            verdict_ok = False
    print(f"   (delta_f, m_f printed at p_x = {P_SHOW})")

M2 = sorted(MS)[-2]
print(f"\nVerdict: second-finest mesh (dk = {K_LIM / M2:.1e}) vs finest, tolerance {TOL_MEV} meV -> "
      f"{'CONVERGED' if verdict_ok else 'NOT CONVERGED'}")
raise SystemExit(0 if verdict_ok else 1)