"""
Hartree-Fock functions: energies

Functions
---------
get_fock_self_energy_direct
"""

import numpy as np

from .thermo import get_density_matrix

__all__ = [
    "get_fock_self_energy_direct",
]


def get_fock_self_energy_direct(rho, KX, KY, e2_over_eps, p_points=None):
    """
    Sigma_F(p) = - int d^2k/(2pi)^2 V(|p-k|) rho(k), by direct summation, for one flavor.

    rho      : (2, 2, N, N) from get_density_matrix on the mesh (KX, KY)
    e2_over_eps : e^2/epsilon [eV · a]
    p_points : None -> p runs over the k-mesh, or a tuple (px, py) of arrays in units of 1/a
    
    Returns Sigma with shape (2, 2, *p_shape) [eV].

    Usage:
        rho = get_density_matrix(system, KX, KY, mu, config.T_eff, reference="half")
        Sigma = get_fock_self_energy_direct(rho, KX, KY, config.e2_over_eps)
        delta_f = 0.5 * (Sigma[0, 0] + Sigma[1, 1]).real
        m_f     = 0.5 * (Sigma[0, 0] - Sigma[1, 1]).real
    """
    dk = KX[0, 1] - KX[0, 0]
    pref = dk ** 2 / (2 * np.pi) ** 2

    kx, ky = KX.ravel(), KY.ravel()
    r11, r22, r12 = rho[0, 0].ravel(), rho[1, 1].ravel(), rho[0, 1].ravel()

    if p_points is None:
        px, py, out_shape = kx, ky, KX.shape
    else:
        px = np.asarray(p_points[0], dtype=float).ravel()
        py = np.asarray(p_points[1], dtype=float).ravel()
        out_shape = np.shape(p_points[0])

    # V at q = 0: average of 2*pi*e2/q over one dk x dk cell (1/q is integrable in 2D)
    V0 = 2 * np.pi * e2_over_eps * 4 * np.log(1 + np.sqrt(2)) / (dk)

    S = np.zeros((2, 2, px.size), dtype=complex)
    for m in range(px.size):                            # outer loop: over p
        q = np.hypot(px[m] - kx, py[m] - ky)            # inner "loop": all k at once
        V = np.full_like(q, V0)
        nz = q > 1e-5 * dk                           # q = 0 only when p sits on a mesh point
        V[nz] = 2 * np.pi * e2_over_eps / q[nz]
        S[0, 0, m] = V @ r11
        S[1, 1, m] = V @ r22
        S[0, 1, m] = V @ r12

    S *= -pref
    S[1, 0] = np.conj(S[0, 1])                          # V is real, rho is Hermitian
    return S.reshape(2, 2, *out_shape)