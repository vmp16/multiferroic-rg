"""
k-space mesh, Fermi distribution helpers, and system construction.

Functions
---------
get_kmesh
fermi_distrib
deriv_fermi_distrib
build_systems_with_displacement
"""

import numpy as np

__all__ = [
    "get_kmesh",
    "fermi_distrib",
    "deriv_fermi_distrib",
    "build_systems_with_displacement",
]


def get_kmesh(k_lim, n_pts):
    """
    Generate a square mesh in k-space in cartesian coordinates.
    """
    kx_vals = np.linspace(-k_lim, k_lim, n_pts)
    ky_vals = np.linspace(-k_lim, k_lim, n_pts)
    KX, KY = np.meshgrid(kx_vals, ky_vals)

    return KX, KY


def fermi_distrib(E, mu, T):
    """
    Returns the Fermi distribution value.

    Parameters
    ----------
    E : float
        Energy in eV
    mu : float
        Fermi level in eV
    T : float
        Temperature in eV (kB * T_real)

    Returns
    ----------
    f(E, T, mu) : float
        Corresponding value of the Fermi distribution
    """
    x = (E - mu) / T

    # Avoid overflow in exp by clipping x
    x_clipped = np.clip(x, -700, 700)
    return 1 / (1 + np.exp(x_clipped))


def deriv_fermi_distrib(E, mu, T):
    """
    Returns the derivative of the Fermi distribution with respect to the energy.

    Parameters
    ----------
    E : float
        Energy in eV
    mu : float
        Fermi level in eV
    T : float
        Temperature in eV (kB * T_real)

    Returns
    ----------
    df_dE(E, T, mu) : float
        Corresponding value of the Fermi distribution's derivative in units of 1/eV.
    """
    x = (E - mu) / T
    # Avoid overflow in exp by clipping x
    x_clipped = np.clip(x, -700, 700)

    return - (fermi_distrib(E, mu, T))**2 * np.exp(x_clipped) / T


def build_systems_with_displacement(D, config):
    """
    Construct the four McCannCarts systems with displacement field D applied
    to every flavor gap: Delta_alpha(D) = delta_alpha + D.
 
    Parameters
    ----------
    D : float
        Displacement field contribution to the gap [eV].
    config : module
        The project config module (provides VALLEY_IDX, DELTAS, N,
        GAMMA0..GAMMA4, E0_ARRAY).
 
    Returns
    -------
    systems : list of McCannCarts, length 4
        Systems in the canonical flavor order (K↑, K'↓, K↓, K'↑).
    base_deltas : ndarray, shape (4,)
        Intrinsic gap values delta_alpha (without D), in the same order.
    """
    from model.model import McCannCarts

    valley_indices = 2 * config.VALLEY_IDX           # [1,-1] → [1,-1,1,-1]
    base_deltas    = config.DELTAS.T.flatten()       # shape (4,)
    e0_vals        = config.E0_ARRAY.T.flatten()     # shape (4,)

    systems = [
        McCannCarts(
            N          = config.N,
            valley_idx = xi,
            Delta      = delta + D,
            gamma0     = config.GAMMA0,
            gamma1     = config.GAMMA1,
            gamma2     = config.GAMMA2,
            gamma3     = config.GAMMA3,
            gamma4     = config.GAMMA4,
        )
        for xi, delta, e0 in zip(valley_indices, base_deltas, e0_vals)
    ]
    return systems, base_deltas
