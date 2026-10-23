"""
debug_tables.py
Find out why solve_scf_from_tables returns None with the cached tables.
Uses the same parameters as scan_phase_diagram_PIP.py.
"""
import tempfile
import numpy as np
import model.config as config
from model.analysis import (
    get_kmesh, get_or_build_tables, solve_scf_from_tables,
    compute_mf_symmetric_potentials,
    build_systems_with_displacement, precompute_flavor_bands,
    build_density_and_energy_tables,
)

# ---- same parameters as the scan ----
N_PTS_SCAN, K_LIM_SCAN = 1500, 0.15
MU_MIN, MU_MAX, N_TABLE = -0.20, 0.20, 2000
D = np.linspace(0.0, 0.070, 170)[80]
n_target = -1.0e12

KX, KY = get_kmesh(K_LIM_SCAN, N_PTS_SCAN)
mu_t, n_t, e_t = get_or_build_tables(
    D, config, KX, KY, cache_dir=config.DATA_DIR / "tables_cache",
    mu_min=MU_MIN, mu_max=MU_MAX, n_table_pts=N_TABLE,
)

# ---- 1. Basic sanity of the loaded arrays ----
print("\n[1] SHAPES / CONTENT")
print("  mu_table:", mu_t.shape, mu_t.dtype, f"[{mu_t[0]:.3f}, {mu_t[-1]:.3f}]")
print("  n_table :", n_t.shape, n_t.dtype)
print("  e_table :", e_t.shape, e_t.dtype)
print("  all finite:", np.all(np.isfinite(n_t)), np.all(np.isfinite(e_t)))
print("  strictly increasing n(mu):", [bool(np.all(np.diff(r) > 0)) for r in n_t])
print("  n at mu_min per flavor:", n_t[:, 0])
print("  n at mu_max per flavor:", n_t[:, -1])
print("  n at mu=0   per flavor:", [np.interp(0.0, mu_t, r) for r in n_t])

# ---- 2. Why does the SCF bail out? One point, one seed ----
print("\n[2] BRACKET FOR ONE POINT (n_target = %.2e)" % n_target)
seed = np.full(4, n_target / 4) * np.array([1.3, 0.7, 1.2, 0.8])
seed *= n_target / seed.sum()
V = compute_mf_symmetric_potentials(seed, config.U, config.area_uc)
lo = mu_t[0] + np.max(V)
hi = mu_t[-1] + np.min(V)
print("  V_alpha [meV]:", V * 1e3)
print(f"  lo = {lo:.4f} eV, hi = {hi:.4f} eV, lo < hi: {lo < hi}")

def resid(mu):
    return sum(np.interp(mu - V[a], mu_t, n_t[a]) for a in range(4)) - n_target

print(f"  residual(lo) = {resid(lo):.3e}   residual(hi) = {resid(hi):.3e}")
print("  sign change (brentq can work):", resid(lo) * resid(hi) < 0)

res = solve_scf_from_tables(n_target, mu_t, n_t, config.U, config.area_uc, seed)
print("  solve_scf_from_tables ->", None if res is None else (res["converged"], res["n_iter"]))

# ---- 3. Cached vs freshly built (small mesh, temporary cache) ----
print("\n[3] CACHED vs FRESH (300^2 mesh)")
KXs, KYs = get_kmesh(K_LIM_SCAN, 300)
with tempfile.TemporaryDirectory() as tmp:
    mu_a, n_a, e_a = get_or_build_tables(
        D, config, KXs, KYs, cache_dir=tmp,
        mu_min=MU_MIN, mu_max=MU_MAX, n_table_pts=N_TABLE, verbose=False)   # builds + saves
    mu_b, n_b, e_b = get_or_build_tables(
        D, config, KXs, KYs, cache_dir=tmp,
        mu_min=MU_MIN, mu_max=MU_MAX, n_table_pts=N_TABLE, verbose=False)   # loads
systems, _ = build_systems_with_displacement(D, config)
bands = precompute_flavor_bands(systems, KXs, KYs)
dk = KXs[0, 1] - KXs[0, 0]
mu_c, n_c, e_c = build_density_and_energy_tables(
    bands, config.T_eff, (dk**2) / (2 * np.pi)**2, config.unit_cell_to_cm2,
    mu_min=MU_MIN, mu_max=MU_MAX, n_table_pts=N_TABLE)
print("  built == loaded :", np.allclose(n_a, n_b), np.allclose(e_a, e_b))
print("  built == direct :", np.allclose(n_a, n_c), np.allclose(e_a, e_c))