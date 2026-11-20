"""
NOTE: made by Claude.ai

Tests of the self-consistent Hartree-Fock loop (hf_scf.py), on a coarse mesh (speed, not accuracy).

1. non-interacting limit  : interaction switched off -> Sigma = 0, densities of the bare bands, sum = n_target
2. density constraint     : sum_alpha n_alpha = n_target for every reference ("cnp", "half", "valence")
3. fixed point            : restarting from the returned Sigma gives a residual ~ tolerance
4. flavor symmetry        : flavors with identical bare data have identical n and Sigma from a symmetric start
5. unbracketed density    : an impossible n_target makes solve_scf_hf return None
A short report of a multi-start run (not a test) is printed at the end.

Run:  python test_hf_scf.py
"""

import numpy as np

import model.config as config
from model.analysis import (
    get_kmesh,
    build_systems_with_displacement,
    get_flavor_density_from_bands,
    build_hf_model,
    solve_scf_hf,
    random_seed_shifts,
)

N, K_LIM = 501, 0.15              # odd coarse mesh
D, N_TOT = 0.02, -4.0e12          # displacement [eV], total density [cm^-2]
TOL = 1e-6                        # SCF tolerance [eV]
SHIFTS = np.array([4e-3, -2e-3, 1e-3, -3e-3])     # seed shifts [eV]

KX, KY = get_kmesh(K_LIM, N)
MODEL = build_hf_model(D, KX, KY, config, reference="cnp")


def run(model, n_target=N_TOT, **kw):
    return solve_scf_hf(model, n_target, tolerance=TOL, max_iter=400, verbose=False, **kw)


# ---------------------------------------------------------------
def test_noninteracting_limit():
    m0 = dict(MODEL)
    ks, kd, L = MODEL["kernels"]
    m0["kernels"] = (0 * ks, 0 * kd, L)
    m0["e2_over_eps"] = 0.0
    r = run(m0)
    assert r is not None and r["converged"]
    assert np.max(np.abs(r["Sigma"])) == 0.0 and r["SzH"] == 0.0, "Sigma should vanish without interaction"

    systems, _ = build_systems_with_displacement(D, config)
    n_bare = np.array([
        get_flavor_density_from_bands(s.get_energy(KX, KY), r["mu"], config.T_eff,
                                      MODEL["prefactor"], config.unit_cell_to_cm2)
        for s in systems
    ])
    assert np.allclose(r["n_flavs"], n_bare, rtol=1e-9), "densities differ from the bare bands"
    assert np.isclose(r["n_flavs"].sum(), N_TOT, rtol=1e-6), "total density != n_target"


def test_density_constraint_all_references():
    for ref in ("cnp", "half", "valence"):
        model = build_hf_model(D, KX, KY, config, reference=ref, kernels=MODEL["kernels"])
        r = run(model, seed_shifts=SHIFTS)
        assert r is not None and r["converged"], f"reference {ref}: SCF did not converge"
        assert np.isclose(r["n_flavs"].sum(), N_TOT, rtol=1e-6), \
            f"reference {ref}: sum n = {r['n_flavs'].sum():.6e} vs {N_TOT:.6e}"


def test_fixed_point():
    r = run(MODEL, seed_shifts=SHIFTS)
    assert r is not None and r["converged"], "SCF did not converge"
    r2 = solve_scf_hf(MODEL, N_TOT, Sigma_init=r["Sigma"], SzH_init=r["SzH"],
                      max_iter=1, tolerance=TOL, verbose=False)
    assert r2["history"][0] < 10 * TOL, f"not a fixed point: residual {r2['history'][0]:.2e} eV"


def test_flavor_symmetry():
    r = run(MODEL)                                           # symmetric start (Sigma = 0, no seed)
    assert r is not None and r["converged"]
    h0, X, Dl = MODEL["h0"], MODEL["X"], MODEL["Delta"]
    n_pairs = 0
    for a in range(MODEL["n_flavors"]):
        for b in range(a + 1, MODEL["n_flavors"]):
            if np.allclose(h0[a], h0[b]) and np.allclose(X[a], X[b]) and np.isclose(Dl[a], Dl[b]):
                n_pairs += 1
                assert np.isclose(r["n_flavs"][a], r["n_flavs"][b], rtol=1e-9), f"n differs for flavors {a}, {b}"
                assert np.allclose(r["Sigma"][a], r["Sigma"][b], atol=1e-10), f"Sigma differs for flavors {a}, {b}"
    print(f"      ({n_pairs} equivalent flavor pair(s) checked)")


def test_unbracketed_density_returns_none():
    assert run(MODEL, n_target=-1.0e15) is None, "expected None for an impossible density"


def multistart_report(n_seeds=4):
    """Not a test: different seeds can end in different flavor patterns; the lowest energy wins."""
    rng = np.random.default_rng(0)
    print("      multi-start (n_alpha / n_target, energy [eV/a^2]):")
    for i in range(n_seeds):
        r = run(MODEL, seed_shifts=random_seed_shifts(4, 5e-3, rng))
        print(f"        seed {i}: conv={r['converged']}  n/n_tot={np.round(r['n_flavs'] / N_TOT, 4)}  "
              f"E={r['energy']:.8e}")


if __name__ == "__main__":
    tests = [test_noninteracting_limit, test_density_constraint_all_references, test_fixed_point,
             test_flavor_symmetry, test_unbracketed_density_returns_none]
    n_failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            n_failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    multistart_report()
    raise SystemExit(1 if n_failed else 0)