import numpy as np

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


# ================ DOS & PARTICLE DENSITY ================

def get_dos(system, px, py, T, mu):
    """
    Calculate the Density of States.
    """
    dk = px[0, 1] - px[0, 0]
    prefactor = (dk**2) / (2 * np.pi)

    E0, E1 = system.get_energy(px, py)
    
    df_dE0 = deriv_fermi_distrib(E0, mu, T)
    df_dE1 = deriv_fermi_distrib(E1, mu, T)
    
    total_dos = np.sum(-df_dE0 - df_dE1)
        
    return total_dos * prefactor

def get_part_density(system, px, py, T, mu):
    """
    Calculate the particle density for the given flavor.
    Returns: n [states / unit cell]
    """
    dk = px[0, 1] - px[0, 0]
    prefactor = (dk**2) / (2 * np.pi)**2

    E0, E1 = system.get_energy(px, py)

    # Count electrons in the Conduction band
    n_e = fermi_distrib(E0, mu, T)
    # Count holes in the Valence band
    n_h = 1.0 - fermi_distrib(E1, mu, T)

    # Net density
    n_total = np.sum(n_e - n_h)

    return prefactor * n_total


# ================ WAVEFUNCTIONS PROPERTIES ================

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


# -------- GET FUNCTIONS FOR CHUNK SPACE --------

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