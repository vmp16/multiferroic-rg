"""
Quantum geometry and transport: velocity matrix elements, Quantum Geometry Tensor (QGT),
Quantum Metric (QM), Berry curvature, Anomalous Hall Effect (AHE), Quantum Metric Dipole (QMD).

Functions (element-wise)
------------------------
velocity_element
get_QG_tensor_component
get_quantum_metric_component
get_Berry_curv
get_QG_tensor
get_quantum_metric
get_G_tensor
get_qmd
get_ahe
sym_decomp_cond

Functions (chunk / memory-efficient)
-------------------------------------
get_QGT_chunk
get_qm_from_qgt
get_bc_from_qgt
get_G_tensor_from_qgt
get_qmd_from_qgt
integrate_qmd_chunk
"""

import numpy as np

from .kspace import fermi_distrib, deriv_fermi_distrib

__all__ = [
    # element-wise
    "velocity_element",
    "get_QG_tensor_component",
    "get_quantum_metric_component",
    "get_Berry_curv",
    "get_QG_tensor",
    "get_quantum_metric",
    "get_G_tensor",
    "get_qmd",
    "get_ahe",
    "sym_decomp_cond",
    # chunk / memory-efficient
    "get_QGT_chunk",
    "get_qm_from_qgt",
    "get_bc_from_qgt",
    "get_G_tensor_from_qgt",
    "get_qmd_from_qgt",
    "integrate_qmd_chunk",
]


# -------- ELEMENT-WISE FUNCTIONS --------

def velocity_element(system, px, py, idx1, idx2, axis):
    """
    Analytic scalar calculation of <psi1 | v_axis | psi2>.
    axis=1 for x, axis=0 for y.
    """
    # Get eigenstate components
    psi1_x, psi1_y = system.get_eigenstate_components(px, py, idx1)
    psi1_x = np.conj(psi1_x)
    psi1_y = np.conj(psi1_y)

    psi2_x, psi2_y = system.get_eigenstate_components(px, py, idx2)

    # Get Hamiltonian derivative components
    dh0 = system.derivate_h0_at_k(px, py, axis)
    dX = system.derivate_X_at_k(px, py, axis)

    # Calculate <psi1 | v | psi2> = psi1* · (v · psi2)
    # v = [[dh0, dX], [dX*, dh0]]
    # res = psi1_x* (dh0*psi2_x + dX*psi2_y) + psi1_y* (dX**psi2_x + dh0*psi2_y)
    
    # Intraband (idx1 == idx2) is real
    if idx1 == idx2:
        # <psi | v | psi> = dh0 * <psi|psi> + 2*Re(psi1* * dX * psi2)
        # since <psi|psi> = 1
        res = dh0 + 2 * np.real(psi1_x * dX * psi2_y)
    else:
        # Interband (idx1 != idx2): <psi1 | psi2> = 0
        # res = psi1_x* * dX * psi2_y + psi1_y* * dX* * psi2_x
        res = psi1_x * dX * psi2_y + psi1_y * np.conj(dX) * psi2_x

    return res


def get_QG_tensor_component(system, band_idx, px, py, i, j):
    """
    Calculate a single component (i, j) of the Quantum Geometry Tensor.
    i, j can be 0 (x) or 1 (y).
    Note: i, j mapping to McCannCarts axis: 0 -> 1 (x), 1 -> 0 (y)
    """
    if band_idx not in (0, 1):
        return None
    n_idx = 1 - band_idx

    e0, e1 = system.get_energy(px, py)
    dE_sq = (e0 - e1) ** 2
    del e0, e1

    # Vi_bn = <band_idx | v_i | n_idx>
    # Vj_bn = <band_idx | v_j | n_idx>
    # T_ij = Vi_bn * Vj_bn* / dE^2
    
    ax_i = 1 if i == 0 else 0
    ax_j = 1 if j == 0 else 0

    Vi_bn = velocity_element(system, px, py, band_idx, n_idx, ax_i)
    if i == j:
        res = np.abs(Vi_bn)**2 / dE_sq
    else:
        Vj_bn = velocity_element(system, px, py, band_idx, n_idx, ax_j)
        res = Vi_bn * np.conj(Vj_bn) / dE_sq
    
    return res


def get_quantum_metric_component(system, band_idx, px, py, i, j):
    """
    Calculate a single component (i, j) of the Quantum Metric Tensor.
    """
    T_ij = get_QG_tensor_component(system, band_idx, px, py, i, j)
    return np.real(T_ij)


def get_Berry_curv(system, band_idx, px, py):
    """
    Get the Berry curvature.
    """
    T_xy = get_QG_tensor_component(system, band_idx, px, py, 0, 1)
    Omega_z = -2 * np.imag(T_xy)
    return Omega_z


def get_QG_tensor(system, band_idx, px, py):
    """
    Build the Quantum Geometry Tensor for a given band.
    Warning: This creates large 4D tensors. For large N_PTS, use component-wise functions.
    """
    T_xx = get_QG_tensor_component(system, band_idx, px, py, 0, 0)
    T_xy = get_QG_tensor_component(system, band_idx, px, py, 0, 1)
    T_yx = get_QG_tensor_component(system, band_idx, px, py, 1, 0)
    T_yy = get_QG_tensor_component(system, band_idx, px, py, 1, 1)

    T_tensor = np.zeros((2, 2, *px.shape), dtype=complex)
    T_tensor[0, 0] = T_xx
    T_tensor[0, 1] = T_xy
    T_tensor[1, 0] = T_yx
    T_tensor[1, 1] = T_yy
    
    return T_tensor


def get_quantum_metric(system, band_idx, px, py):
    """
    Get the Quantum Metric tensor.
    """
    T_tensor = get_QG_tensor(system, band_idx, px, py)
    g_tensor = np.real(T_tensor)
    return g_tensor


def get_G_tensor(system, px, py, band_idx):
    """
    Calculate the normalized G tensor.
    """
    e0, e1 = system.get_energy(px, py)
    dE = e0 - e1 if band_idx == 0 else e1 - e0
    
    g_tensor = get_quantum_metric(system, band_idx, px, py)
    return 2 * g_tensor / dE


def get_qmd(system, px, py, band_idx):
    """
    Calculate the Quantum Metric Dipole tensor.
    """
    g_tensor = get_quantum_metric(system, band_idx, px, py)
    dk = px[0, 1] - px[0, 0]
    
    dg_dky, dg_dkx = np.gradient(g_tensor, dk, axis=(2, 3))

    qmd_tensor = np.zeros((2, 2, 2, *px.shape))
    qmd_tensor[0] = dg_dkx
    qmd_tensor[1] = dg_dky

    return qmd_tensor


def get_ahe(system, band_idx, px, py, T, mu):
    """
    Calculate the Linear Anomalous Hall Effect of the given band (Intrinsic contribution).
    Result is in units of e^2/h.
    """
    # Set factors for integration
    dk = px[0, 1] - px[0, 0]
    prefactor = (dk**2) / (2 * np.pi)

    # Get the energies locally
    e0, e1 = system.get_energy(px, py)
    band_E = e0 if band_idx == 0 else e1
    del e0, e1

    # Get the Berry curvature scalar directly
    omega_z = get_Berry_curv(system, band_idx, px, py)

    # Get the Fermi distribution
    f_E = fermi_distrib(band_E, mu, T)
    del band_E

    # Integrate over the occupied states
    sigma_xy = prefactor * np.sum(omega_z * f_E)
    del omega_z, f_E

    return sigma_xy


def sym_decomp_cond(sigma):
    """
    Decompose the conductivity tensor sigma into symmetric and asymmetric parts.
    """
    # sigma has shape (2, 2, 2)
    sigma_sym = (sigma + sigma.transpose(2, 0, 1) + sigma.transpose(1, 2, 0)) / 3
    sigma_asym = sigma - sigma_sym

    return sigma_sym, sigma_asym


# -------- CHUNK / MEMORY-EFFICIENT FUNCTIONS --------

def get_QGT_chunk(system, band_idx, kx, ky):
    """
    Builds the 2x2 Quantum Geometry Tensor for a specific k-mesh chunk.
    Returns: T_tensor of shape (2, 2, Ny_chunk, Nx).
    """
    n_idx = 1 - band_idx
    e0, e1 = system.get_energy(kx, ky)
    dE_sq = (e0 - e1) ** 2

    # Get velocity elements for this chunk
    vx_band = velocity_element(system, kx, ky, band_idx, n_idx, 0)
    vy_band = velocity_element(system, kx, ky, band_idx, n_idx, 1)

    # Build the 4D complex tensor just for this chunk
    T_tensor = np.zeros((2, 2, *kx.shape), dtype=complex)

    T_tensor[0, 0] = np.abs(vx_band)**2 / dE_sq
    T_tensor[0, 1] = (vx_band * np.conj(vy_band)) / dE_sq
    T_tensor[1, 0] = (vy_band * np.conj(vx_band)) / dE_sq
    T_tensor[1, 1] = np.abs(vy_band)**2 / dE_sq

    return T_tensor


def get_qm_from_qgt(T_tensor):
    """Extracts the Quantum Metric tensor (g) from the QGT"""
    return np.real(T_tensor)


def get_bc_from_qgt(T_tensor):
    """Extracts the out-of-plane Berry Curvature (Omega) from the QGT"""
    # Omega_tensor = -2*np.imag(T_tensor)
    return (np.imag(T_tensor[1, 0]) - np.imag(T_tensor[0, 1]))


def get_G_tensor_from_qgt(T_tensor, e0, e1, band_idx):
    """Extracts band normalized quantum metric tensor (G) from the QGT"""
    dE = e0 - e1 if band_idx == 0 else e1 - e0
    g_tensor = get_qm_from_qgt(T_tensor)
    return 2 * g_tensor / dE


def get_qmd_from_qgt(T_tensor, dk):
    """
    Calculates the Quantum Metric Dipole tensor by taking gradients of the QGT's real part.
    """
    g_tensor = get_qm_from_qgt(T_tensor)

    # g_tensor has shape (2, 2, Ny_chunk, Nx)
    dg_dky, dg_dkx = np.gradient(g_tensor, dk, axis=(2, 3))

    qmd_tensor = np.zeros((2, 2, 2, *g_tensor.shape[2:]))
    qmd_tensor[0] = dg_dkx
    qmd_tensor[1] = dg_dky

    return qmd_tensor


def integrate_qmd_chunk(qmd_tensor_accumulated, band_E, mu_vals, T_eff, prefactor):
    """
    Integrates a precalculated, accumulated QMD chunk over the Fermi Surface.
    """
    # df_dE shape: (n_mu, Ny_chunk, Nx)
    df_dE = deriv_fermi_distrib(band_E[None, :, :], mu_vals[:, None, None], T_eff)
    
    # Contract spatial indices and sum over the chunk
    return np.einsum('abcjk, mjk -> mabc', qmd_tensor_accumulated, df_dE) * prefactor
