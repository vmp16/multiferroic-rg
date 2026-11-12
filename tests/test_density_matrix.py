"""
NOTE: made by Claude.ai

Check that Tr rho_rel (reference = "half" and "valence") reproduces
get_flavor_density_from_bands for every flavor.

Run:  python test_density_matrix.py
Requires get_density_matrix to be defined in thermo.py and listed in its __all__.
"""

import itertools
import numpy as np

import model.config as config
from model.analysis import (
    get_kmesh,
    build_systems_with_displacement,
    get_flavor_density_from_bands,
    get_density_matrix,
    fermi_distrib,
)

N_PTS = 400          # even -> k = 0 is not on the mesh (avoids eps = 0 when Delta = 0)
K_LIM = 0.15

KX, KY = get_kmesh(K_LIM, N_PTS)
dk = KX[0, 1] - KX[0, 0]
PREFACTOR = dk**2 / (2 * np.pi) ** 2

D_LIST = [0.0, 0.02, 0.05]                    # eV
MU_LIST = [-0.05, -0.01, 0.0, 0.005, 0.03]    # eV


def test_trace_matches_flavor_density():
    n_fail = 0
    for D, mu, reference in itertools.product(D_LIST, MU_LIST, ["half", "valence"]):
        systems, _ = build_systems_with_displacement(D, config)
        for a, system in enumerate(systems):
            rho = get_density_matrix(system, KX, KY, mu, config.T_eff, reference=reference)
            n_rho = (PREFACTOR * np.sum(np.trace(rho, axis1=0, axis2=1).real)
                     * config.unit_cell_to_cm2)
            n_ref = get_flavor_density_from_bands(
                system.get_energy(KX, KY), mu, config.T_eff, PREFACTOR, config.unit_cell_to_cm2
            )
            # atol: densities near neutrality can be ~0
            if not np.isclose(n_rho, n_ref, rtol=1e-9, atol=1e-6 * abs(n_ref) + 1e3):
                n_fail += 1
                print(f"  FAIL flavor {a}, D={D*1e3:.1f} meV, mu={mu*1e3:.1f} meV, "
                      f"ref={reference}: {n_rho:.6e} vs {n_ref:.6e}")
    assert n_fail == 0, f"{n_fail} trace mismatches"


def test_hermitian():
    system = build_systems_with_displacement(0.02, config)[0][0]
    for reference in ["none", "half", "valence"]:
        rho = get_density_matrix(system, KX, KY, 0.01, config.T_eff, reference=reference)
        assert np.allclose(rho[0, 1], np.conj(rho[1, 0])), f"off-diagonals not conjugate ({reference})"
        assert np.allclose(rho[0, 0].imag, 0) and np.allclose(rho[1, 1].imag, 0), \
            f"complex diagonal ({reference})"


def test_valence_reference_is_projector():
    """rho("none") - rho("valence") must equal P-, with P-^2 = P-, Tr P- = 1."""
    system = build_systems_with_displacement(0.02, config)[0][0]
    mu = 0.01
    P = (get_density_matrix(system, KX, KY, mu, config.T_eff, reference="none")
         - get_density_matrix(system, KX, KY, mu, config.T_eff, reference="valence"))
    P2 = np.einsum("ab...,bc...->ac...", P, P)
    assert np.allclose(P2, P, atol=1e-10), "P- is not idempotent"
    assert np.allclose(P[0, 0] + P[1, 1], 1.0, atol=1e-10), "Tr P- != 1"


def test_against_numerical_diagonalization():
    """
    Build H(k) = h0*1 + [[Delta, X], [X*, -Delta]] explicitly, diagonalize it with eigh,
    and compare get_density_matrix with sum_n f(E_n)|psi_n><psi_n| (minus the reference).
    This fixes the off-diagonal convention (X vs X*), which the other tests cannot see.
    """
    rng = np.random.default_rng(0)
    nk = 300
    px = rng.uniform(-K_LIM, K_LIM, nk)
    py = rng.uniform(-K_LIM, K_LIM, nk)
    mu = 0.01
    eye = np.eye(2)

    for D in [0.0, 0.02]:
        systems, _ = build_systems_with_displacement(D, config)
        for a, system in enumerate(systems):
            X = system.X_at_k(px, py)
            h0 = np.broadcast_to(system.get_h0_at_k(px, py), px.shape)

            H = np.zeros((nk, 2, 2), dtype=complex)
            H[:, 0, 0] = system.Delta
            H[:, 1, 1] = -system.Delta
            H[:, 0, 1] = X
            H[:, 1, 0] = np.conj(X)
            H += h0[:, None, None] * eye

            E, V = np.linalg.eigh(H)            # E[:, 0] = lower band
            E_model = system.get_energy(px, py)
            assert np.allclose(E[:, 1], E_model[0]) and np.allclose(E[:, 0], E_model[1]), \
                f"eigenvalues differ from get_energy (flavor {a}, D={D})"

            f = fermi_distrib(E, mu, config.T_eff)
            rho_num = np.einsum("kan,kn,kbn->kab", V, f, V.conj())
            P_minus = np.einsum("ka,kb->kab", V[:, :, 0], V[:, :, 0].conj())

            expected = {
                "none":    rho_num,
                "half":    rho_num - 0.5 * eye,
                "valence": rho_num - P_minus,
            }
            for reference, rho_ref in expected.items():
                rho = get_density_matrix(system, px, py, mu, config.T_eff, reference=reference)
                rho = np.moveaxis(rho, (0, 1), (-2, -1))     # (2,2,nk) -> (nk,2,2)
                assert np.allclose(rho, rho_ref, atol=1e-10), \
                    f"mismatch with diagonalization: flavor {a}, D={D}, ref={reference}, " \
                    f"max |diff| = {np.max(np.abs(rho - rho_ref)):.2e}"


if __name__ == "__main__":
    tests = [test_trace_matches_flavor_density, test_hermitian, test_valence_reference_is_projector,
             test_against_numerical_diagonalization]
    n_failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            n_failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    raise SystemExit(1 if n_failed else 0)