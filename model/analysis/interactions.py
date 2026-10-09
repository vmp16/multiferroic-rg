"""
Interaction potentials (SU(4) symmetric and Hund's anisotropic).

Functions
---------
V_int_SU4
V_int_anisotropic
compute_mf_symmetric_potentials
compute_mf_anisotropic_potentials
"""

import numpy as np
from itertools import combinations

__all__ = [
    "V_int_SU4",
    "V_int_anisotropic",
    "compute_mf_symmetric_potentials",
    "compute_mf_anisotropic_potentials",
]


def V_int_SU4(n_vec, U, area_uc):
    """Computes SU(4) symmetric interaction potential."""
    # Generalized for any n_flavors
    pair_sum = sum(n_vec[i] * n_vec[j] for i, j in combinations(range(len(n_vec)), 2))

    return U * area_uc * pair_sum


def V_int_anisotropic(n_vec, U, J, area_uc):
    """
    Computes SU(4) broken interaction potential with Hund's anisotropy.
    WARNING: It is only defined for 4 flavors.
    """
    n1, n2, n3, n4 = n_vec
    
    su4_part = V_int_SU4(n_vec, U, area_uc)
    hunds_part = J * area_uc * (n1 - n3) * (n2 - n4)
    
    return su4_part + hunds_part


def compute_mf_symmetric_potentials(n_flavs, U, area_uc):
    """
    Computes the derivative of the mean-field SU(4) symmetric interaction shifts V_alpha for N flavors (Eq. S4).
    
    V_alpha = U * area_uc * sum_{beta != alpha} n_beta
            = U * area_uc * (N_tot - n_alpha)
    """
    n_tot = np.sum(n_flavs)
    return U * area_uc * (n_tot - n_flavs)


def compute_mf_anisotropic_potentials(n_flavs, U, J, area_uc):
    """∂V_int_anisotropic/∂n_α for each flavor."""
    n1, n2, n3, n4 = n_flavs
    n_tot = np.sum(n_flavs)
    su4_shifts = U * area_uc * (n_tot - n_flavs)
    hunds_shifts = J * area_uc * np.array([
         (n2 - n4),   # ∂/∂n1
         (n1 - n3),   # ∂/∂n2
        -(n2 - n4),   # ∂/∂n3
        -(n1 - n3),   # ∂/∂n4
    ])
    return su4_shifts + hunds_shifts
