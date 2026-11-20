"""
NOTE: made by Claude.ai

Check that the FFT Fock self-energy reproduces the direct summation (same discretized V).

Tests
-----
1. kernel symmetry          : Vs, Vd arrays are symmetric under (dx, dy) -> (-dx, -dy) and dx <-> dy
2. random rho, even/odd mesh: direct == FFT for an arbitrary Hermitian rho (independent of get_density_matrix)
3. physical rho             : direct == FFT for rho from get_density_matrix (all flavors, several references)
4. structure                : Sigma_21 = conj(Sigma_12), Sigma_11 and Sigma_22 real
Each is run with d = 0 (Vd = Vs) and d = config.d_layers (Vd != Vs).

Run:  python test_fock_fft.py
Requires: config.e2_over_eps, config.d_layers, and the hf.py / thermo.py functions re-exported by model.analysis.
"""

import time
import numpy as np

import model.config as config
from model.analysis import (
    get_kmesh,
    fermi_distrib,
    build_systems_with_displacement,
    density_matrix_from_arrays,
    get_density_matrix,
    coulomb_kernels,
    get_fock_self_energy_direct,
    build_coulomb_kernels_fft,
    get_fock_self_energy_fft,
)

K_LIM = 0.15
E2 = config.e2_over_eps
D_VALUES = [0.0, config.d_layers]      # d = 0 -> Vd = Vs ;  d = d_layers -> Vd != Vs


def fock_both(rho, KX, KY, d):
    """Return (Sigma_direct, Sigma_fft) on the k-mesh."""
    N = KX.shape[0]
    dk = KX[0, 1] - KX[0, 0]
    S_dir = get_fock_self_energy_direct(rho, KX, KY, E2, d=d)
    ks, kd, L = build_coulomb_kernels_fft(N, dk, E2, d)
    S_fft = get_fock_self_energy_fft(rho, ks, kd, L, dk)
    return S_dir, S_fft


def assert_close(S_dir, S_fft, label):
    scale = np.max(np.abs(S_dir))
    err = np.max(np.abs(S_dir - S_fft))
    assert err <= 1e-9 * scale, f"{label}: max |direct - fft| = {err:.3e} (scale {scale:.3e})"


def random_rho(N, rng):
    """Arbitrary Hermitian 2x2 field: real diagonal, complex off-diagonal."""
    rho = np.empty((2, 2, N, N), dtype=complex)
    rho[0, 0] = rng.normal(size=(N, N))
    rho[1, 1] = rng.normal(size=(N, N))
    rho[0, 1] = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    rho[1, 0] = np.conj(rho[0, 1])
    return rho


# ---------------------------------------------------------------
def test_kernel_symmetry():
    N, dk = 12, 1e-2
    q1d = np.arange(-(N - 1), N) * dk
    QX, QY = np.meshgrid(q1d, q1d)
    for d in D_VALUES:
        for name, V in zip(("Vs", "Vd"), coulomb_kernels(np.hypot(QX, QY), dk, E2, d)):
            assert np.allclose(V, V[::-1, ::-1]), f"{name} (d={d}) not symmetric under q -> -q"
            assert np.allclose(V, V.T), f"{name} (d={d}) not symmetric under qx <-> qy"


def test_random_rho_even_and_odd_mesh():
    rng = np.random.default_rng(1)
    for N in (24, 25):                                  # odd N puts k = 0 (q = 0 cell) on the mesh
        KX, KY = get_kmesh(K_LIM, N)
        rho = random_rho(N, rng)
        for d in D_VALUES:
            assert_close(*fock_both(rho, KX, KY, d), f"random rho, N={N}, d={d:.2f}")


def test_physical_rho():
    N = 24
    KX, KY = get_kmesh(K_LIM, N)
    D, mu = 0.02, 0.005
    systems, base_deltas = build_systems_with_displacement(D, config)
    for a, system in enumerate(systems):
        for ref in ("half", "valence", "cnp"):
            rho = get_density_matrix(system, KX, KY, mu, config.T_eff,
                                     reference=ref, Delta_ref=base_deltas[a])
            for d in D_VALUES:
                assert_close(*fock_both(rho, KX, KY, d), f"flavor {a}, ref={ref}, d={d:.2f}")


def test_structure():
    N = 24
    KX, KY = get_kmesh(K_LIM, N)
    rho = random_rho(N, np.random.default_rng(2))
    _, S = fock_both(rho, KX, KY, config.d_layers)
    assert np.allclose(S[1, 0], np.conj(S[0, 1])), "Sigma_21 != conj(Sigma_12)"
    assert np.max(np.abs(S[0, 0].imag)) < 1e-9 * np.max(np.abs(S[0, 0])), "Sigma_11 not real"
    assert np.max(np.abs(S[1, 1].imag)) < 1e-9 * np.max(np.abs(S[1, 1])), "Sigma_22 not real"

def test_array_version_with_synthetic_self_energy():
    system = build_systems_with_displacement(0.02, config)[0][0]
    rng = np.random.default_rng(5)
    nk, mu = 200, 0.01
    px, py = rng.uniform(-K_LIM, K_LIM, nk), rng.uniform(-K_LIM, K_LIM, nk)
    X = system.X_at_k(px, py)
    h0 = np.broadcast_to(system.get_h0_at_k(px, py), px.shape)

    S0 = 0.01 * rng.normal(size=nk)                       # fake Sigma_0, Sigma_z, Sigma_12 [eV]
    Sz = 0.01 * rng.normal(size=nk)
    S12 = 0.01 * (rng.normal(size=nk) + 1j * rng.normal(size=nk))
    h0e, De, Xe = h0 + S0, system.Delta + Sz, X + S12

    def H_of(h, D, Xk):
        H = np.zeros((nk, 2, 2), dtype=complex)
        H[:, 0, 0], H[:, 1, 1], H[:, 0, 1], H[:, 1, 0] = D, -D, Xk, np.conj(Xk)
        return H + h[:, None, None] * np.eye(2)

    E, V = np.linalg.eigh(H_of(h0e, De, Xe))
    rho_num = np.einsum("kan,kn,kbn->kab", V, fermi_distrib(E, mu, config.T_eff), V.conj())
    D_ref = 0.003                                         # any fixed reference gap
    _, Vr = np.linalg.eigh(H_of(np.zeros(nk), np.full(nk, D_ref), X))
    P_ref = np.einsum("ka,kb->kab", Vr[:, :, 0], Vr[:, :, 0].conj())

    rho = density_matrix_from_arrays(h0e, De, Xe, mu, config.T_eff,
                                     reference="projector", Delta_ref=D_ref, X_ref=X)
    rho = np.moveaxis(rho, (0, 1), (-2, -1))
    assert np.allclose(rho, rho_num - P_ref, atol=1e-10)


def timing_report(N=60):
    """Not a test: just prints the speed-up."""
    KX, KY = get_kmesh(K_LIM, N)
    dk = KX[0, 1] - KX[0, 0]
    rho = random_rho(N, np.random.default_rng(3))
    t0 = time.time(); get_fock_self_energy_direct(rho, KX, KY, E2, d=config.d_layers)
    t1 = time.time()
    ks, kd, L = build_coulomb_kernels_fft(N, dk, E2, config.d_layers)
    t2 = time.time(); get_fock_self_energy_fft(rho, ks, kd, L, dk)
    t3 = time.time()
    print(f"      N = {N}: direct {t1 - t0:.2f} s | FFT kernel {t2 - t1:.3f} s (once) + FFT sigma {t3 - t2:.4f} s")


if __name__ == "__main__":
    tests = [test_kernel_symmetry, test_random_rho_even_and_odd_mesh,
             test_physical_rho, test_structure, test_array_version_with_synthetic_self_energy]
    n_failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            n_failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    timing_report()
    raise SystemExit(1 if n_failed else 0)