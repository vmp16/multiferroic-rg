"""
Hartree-Fock functions: Coulomb kernels, Fock self-energy (direct and FFT), layer Hartree term.

Conventions: k and p in units of 1/a, e2_over_eps in eV·a, d in units of a, Sigma in eV.
Basis (A1, BN) = (top, bottom): rho11 - rho22 = n_top - n_bottom.

Functions
---------
coulomb_kernels
get_fock_self_energy_direct
build_coulomb_kernels_fft
get_fock_self_energy_fft
get_hartree_sigma_z
"""

import numpy as np
from scipy import fft as sfft

from .thermo import get_density_matrix

__all__ = [
    "coulomb_kernels",
    "get_fock_self_energy_direct",
    "build_coulomb_kernels_fft",
    "get_fock_self_energy_fft",
    "get_hartree_sigma_z",
]


def coulomb_kernels(q, dk, e2_over_eps, d=0.0):
    """
    Vs(q) = 2*pi*e2_over_eps / q            (same layer)
    Vd(q) = Vs(q) * exp(-q*d)               (different layers)
    q : array of |q| in 1/a.  At q = 0 both get the cell average of Vs over a dk x dk cell
    (1/q is integrable in 2D); fine as long as dk*d << 1.
    """
    zero = q < 1e-5 * dk
    Vs = np.empty_like(q)
    Vs[~zero] = 2 * np.pi * e2_over_eps / q[~zero]
    Vs[zero] = 2 * np.pi * e2_over_eps * 4 * np.log(1 + np.sqrt(2)) / dk
    return Vs, Vs * np.exp(-q * d)


def get_fock_self_energy_direct(rho, KX, KY, e2_over_eps, p_points=None, d=0.0):
    """
    Sigma_F(p) = - int d^2k/(2pi)^2 V(|p-k|) rho(k), by direct summation, for one flavor.
    Diagonal entries use Vs, the off-diagonal entry uses Vd. d = 0 gives Vd = Vs.

    rho      : (2, 2, N, N) from get_density_matrix on the mesh (KX, KY)
    p_points : None -> p runs over the k-mesh, or a tuple (px, py) of arrays in units of 1/a

    Returns Sigma with shape (2, 2, *p_shape) [eV].
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

    S = np.zeros((2, 2, px.size), dtype=complex)
    for m in range(px.size):                            # outer loop: over p
        q = np.hypot(px[m] - kx, py[m] - ky)            # inner "loop": all k at once
        Vs, Vd = coulomb_kernels(q, dk, e2_over_eps, d)
        S[0, 0, m] = Vs @ r11
        S[1, 1, m] = Vs @ r22
        S[0, 1, m] = Vd @ r12                           # interlayer exchange

    S *= -pref
    S[1, 0] = np.conj(S[0, 1])                          # V is real, rho is Hermitian
    return S.reshape(2, 2, *out_shape)

def build_coulomb_kernels_fft(n_pts, dk, e2_over_eps, d=0.0):
    """
    FFTs of Vs and Vd on the (2N-1)^2 grid of offsets (m-n)*dk, zero-padded to L x L.
    They depend only on the mesh: build once and reuse for all flavors and SCF iterations.
    
    Returns (ks_fft, kd_fft, L).
    """
    L = sfft.next_fast_len(2 * n_pts - 1)               # L >= 2N-1 avoids wrap-around in the kept block
    q1d = np.arange(-(n_pts - 1), n_pts) * dk           # all offsets, centre entry is exactly 0
    QX, QY = np.meshgrid(q1d, q1d)
    Vs, Vd = coulomb_kernels(np.hypot(QX, QY), dk, e2_over_eps, d)
    return sfft.fft2(Vs, s=(L, L)), sfft.fft2(Vd, s=(L, L)), L

def get_fock_self_energy_fft(rho, ks_fft, kd_fft, L, dk):
    """
    Same as get_fock_self_energy_direct with p = k-mesh, computed as a convolution via FFT.
    rho : (2, 2, N, N).
    
    Returns Sigma (2, 2, N, N) [eV].
    """
    N = rho.shape[-1]
    pref = dk ** 2 / (2 * np.pi) ** 2
    sl = slice(N - 1, 2 * N - 1)                 # entries c[m + N - 1] = Sigma(p_m)

    S = np.empty(rho.shape, dtype=complex)
    for (i, j), K in (((0, 0), ks_fft), ((1, 1), ks_fft), ((0, 1), kd_fft)):
        S[i, j] = -pref * sfft.ifft2(sfft.fft2(rho[i, j], s=(L, L)) * K)[sl, sl]
    S[1, 0] = np.conj(S[0, 1])
    return S

def get_hartree_sigma_z(rho_flavs, dk, e2_over_eps, d):
    """
    Layer Hartree term, Sigma_z^H = pi * e^2/eps * d * Delta_n   (notes, Eq. 29), in eV.
    Delta_n = n_top - n_bottom summed over ALL flavors, in 1/a^2.
    rho_flavs : list of (2, 2, N, N) density matrices (same reference as used for the Fock term).
    The same number is added to the gap (Delta) of every flavor. Sigma_0^H is dropped
    (divergent constant, absorbed by the background / mu).
    """
    pref = dk ** 2 / (2 * np.pi) ** 2
    delta_n = sum(pref * np.sum((r[0, 0] - r[1, 1]).real) for r in rho_flavs)
    return np.pi * e2_over_eps * d * delta_n