import numpy as np
from scipy.optimize import brentq
from itertools import combinations

# --------- Useful Tools ----------
def progress_bar(current, total, width=40):
    FILL  = "\u2588"  # █
    EMPTY = "\u2591"  # ░

    filled = int(width * current / total)
    bar = FILL * filled + EMPTY * (width - filled)
    pct = current / total * 100
    print(f"\r {bar}  {pct:.1f}%", end="", flush=True)

# --------- Usual Simple Functions ----------

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

def precompute_flavor_bands(systems, KX, KY):
    """
    Precomputes the energy band structures for a list of McCannCarts systems.
    
    Parameters
    ----------
    systems : list of McCannCarts
        List of initialized model systems representing different flavors.
    KX, KY : ndarray
        2D k-space coordinate meshes.

    Returns
    ----------
    all_bands : list of tuples
        List containing (E0, E1) energy band pairs for each flavor.
    """
    return [system.get_energy(KX, KY) for system in systems]

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

def get_flavor_density_from_bands(bands, mu_alpha, T_eff, prefactor, unit_cell_to_cm2):
    """
    Calculates the carrier density n_alpha (in cm^-2) for a given flavor band pair at Fermi level mu_alpha.
    """
    E0, E1 = bands
    n_e = fermi_distrib(E0, mu_alpha, T_eff)
    n_h = 1.0 - fermi_distrib(E1, mu_alpha, T_eff)
    
    return prefactor * np.sum(n_e - n_h) * unit_cell_to_cm2

def get_flavor_dos(bands, mu_alpha, T_eff, prefactor, unit_cell_to_cm2):
    """
    Get the Density of States (DOS) of a given flavor up to a given value of energy mu_alpha.
    """
    E0, E1 = bands
    e_vals = np.linspace(0.0, mu_alpha, 100)

    dos = np.zeros_like(e_vals, dtype=float)

    for i, e in enumerate(e_vals):
        df_dE0 = deriv_fermi_distrib(E0, e, T_eff)
        df_dE1 = deriv_fermi_distrib(E1, e, T_eff)

        dos[i] = np.sum(- df_dE0 - df_dE1) * prefactor

    return dos

def get_flavor_kinetic_energy_from_bands(bands, mu_alpha, T_eff, prefactor, unit_cell_to_cm2):
    """
    Calculates the kinetic energy E_alpha(mu_alpha) = integral_0^mu_alpha eps * rho(eps) d eps
    (in eV * cm^-2) for a given flavor band pair, at Fermi level mu_alpha.

    Follows the same construction as get_flavor_density_from_bands: at T -> 0,
    fermi_distrib(E, mu, T) becomes a Heaviside step Theta(mu - E), so

        n_e   = sum_k Theta(mu - E0_k)             -> integral_0^mu rho(eps) d eps
        E_kin = sum_k E0_k * Theta(mu - E0_k)       -> integral_0^mu eps * rho(eps) d eps

    At finite T, the sharp cutoff is replaced by the Fermi function (same substitution
    used for n_e), and each band energy weights its own occupation term directly -
    no explicit integration over eps is needed.
    """
    E0, E1 = bands
    e_e = E0 * fermi_distrib(E0, mu_alpha, T_eff)
    e_h = E1 * (1.0 - fermi_distrib(E1, mu_alpha, T_eff))

    return prefactor * np.sum(e_e - e_h) * unit_cell_to_cm2

def get_part_density(system, px, py, T, mu):
    """Calculate the particle density for a given flavor system [states / unit cell]."""
    dk = px[0, 1] - px[0, 0]
    prefactor = (dk**2) / (2 * np.pi)**2
    bands = system.get_energy(px, py)
    
    # Unit factor set to 1.0 to retain the original [states / unit cell] return value
    return get_flavor_density_from_bands(bands, mu, T, prefactor, unit_cell_to_cm2=1.0)

# ================ MEAN FIELD CALCULATIONS ================

def V_int_SU4(n_vec, U, area_uc):
    """Computes SU(4) symmetric interaction potential."""
    # n1, n2, n3, n4 = n_vec
    # Sum over distinct pairs alpha < beta
    # pair_sum = n1*n2 + n1*n3 + n1*n4 + n2*n3 + n2*n4 + n3*n4

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

def solve_self_consistent_mean_field(all_bands, n_target, U, area_uc, T_eff, prefactor, unit_cell_to_cm2, initial_n_flavs, max_iter=200, tolerance=1e-6, mixing=0.4, mu_min=-0.10, mu_max=0.10):
    """
    Executes the self-consistent mean-field iteration loop for arbitrary flavors.
    
    Returns
    -------
    n_flavs : ndarray
        Converged flavor densities.
    V_flavs : ndarray
        Converged interaction potential shifts.
    mu_global : float
        Converged global chemical potential.
    """
    n_flavs = np.array(initial_n_flavs, dtype=float)
    num_flavors = len(all_bands)

    # Define root function for the global Fermi level
    def total_density_residual(mu_global, V_flavs):
        n_tot = 0.0
        for idx in range(num_flavors):
            mu_alpha = mu_global - V_flavs[idx]
            n_tot += get_flavor_density_from_bands(
                all_bands[idx], mu_alpha, T_eff, prefactor, unit_cell_to_cm2
            )
        return n_tot - n_target

    print("Starting Self-Consistent Loop...")
    for iteration in range(max_iter):
        # Calculate interaction potentials
        V_flavs = compute_mf_symmetric_potentials(n_flavs, U, area_uc)

        # Find global Fermi level matching total charge density target
        try:
            mu_global = brentq(total_density_residual, mu_min, mu_max, args=(V_flavs,), xtol=1e-7)
        except ValueError:
            raise ValueError(
                f"Iteration {iteration}: Target density outside mu bracket [{mu_min}, {mu_max}] eV."
            )

        # Compute updated densities for each flavor
        n_flavs_new = np.array([
            get_flavor_density_from_bands(
                all_bands[i], mu_global - V_flavs[i], T_eff, prefactor, unit_cell_to_cm2
            )
            for i in range(num_flavors)
        ])

        # Evaluate the error & Check convergence
        rel_error = np.max(np.abs(n_flavs_new - n_flavs) / np.abs(n_target))
        
        print(f"Iter {iteration:02d} | mu_global: {mu_global*1e3:.3f} meV | Max Error: {rel_error:.3e} | n_flavs: ", np.array2string(n_flavs, formatter={'float': '{:.4e}'.format}))

        if rel_error < tolerance:
            print(f"\n---> Converged in {iteration + 1} iterations!")
            n_flavs = n_flavs_new
            # Recompute quantities with final densities
            V_flavs = compute_mf_symmetric_potentials(n_flavs, U, area_uc)
            mu_global = brentq(total_density_residual, mu_min, mu_max, args=(V_flavs,), xtol=1e-7)
            break

        # Apply linear mixing
        n_flavs = (1.0 - mixing) * n_flavs + mixing * n_flavs_new
    else:
        print("\n---> Warning: Reached maximum iterations without full convergence.")

    return n_flavs, V_flavs, mu_global


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

# ================ MONTE-CARLO GROUND STATE SEARCH ================
def sample_flavor_densities(n_target, n_flavors, num_picks, m_seed, rng=None):
    """
    Draw random flavor-density configurations around the symmetric point, all constrained to sum exactly to n_target.

    Each flavor density is drawn uniformly within
        [1 - m_seed, 1 + m_seed] * (n_target / n_flavors)
    and then rescaled so that they all sum to n_target.

    Parameters
    ----------
    n_target : float
        Total carrier density [cm^-2].
    n_flavors : int
        Number of flavors.
    num_picks : int
        Number of random realizations to generate.
    m_seed : float
        Fractional maximal deviation from the symmetric point, defining the random range.
    rng : numpy.random.Generator or None
        Optional seeded RNG (e.g. np.random.delfault_rng(41)). If None, the global numpy random state is used.

    Returns
    -------
    n_flavs_seed : ndarray, shape(n_flavors, num_picks)
        Sampled density configurations, where each column sums to n_target.
    """
    _uniform = rng.uniform if rng is not None else np.random.uniform
    raw = _uniform(1 - m_seed, 1 + m_seed, (n_flavors, num_picks)) * (n_target / n_flavors)
    # Normalize to enforce each configuration to sum to n_target
    n_flavs_seed = raw * (n_target / raw.sum(axis=0))

    return n_flavs_seed

def build_density_interp_tables(all_bands, T_eff, prefactor, unit_cell_to_cm2, mu_min=-0.1, mu_max=0.1, n_table_pts=2000):
    """
    Build a per-flavor look-up table that maps mu -> n_flav(mu), then used inverted (via np.interp) to map n_flav -> mu, avoiding repeated root finding.

    Parameters
    ----------
    all_bands : list of (E0, E1) tuples, len = n_flavors
        Precomputed band pairs for each flavor.
    T_eff : float
        Thermal energy kB*T [eV].
    prefactor : float
        k-space integration prefactor (dk^2 / (2*pi^2))
    unit_cell_to_cm2 : float
        Conversion factor from unit-cell units to cm^-2.
    mu_min, mu_max : float, optional
        Limits of the chemical potential bracket [eV].
    n_table_pts : int
        Number of points in the mu_grid, increase for higher interpolation accuracy.

    Returns
    -------
    mu_table : ndarray, shape (n_table_pts,)
        Uniformly spaced mu points.
    n_table : ndarray, shape (n_flavors, n_table_pts)
        Flavor densities evaluated at each mu point.

    Raises
    ------
    RunTimeError
        If n_flav(mu) is not strictly monotone for any flavor, which would make the inversion via np.interp invalid.
    """
    n_flavors = len(all_bands)
    mu_table = np.linspace(mu_min, mu_max, n_table_pts)
    n_table = np.zeros((n_flavors, n_table_pts))    # n_flav(mu) per flavor, shape (4, n_table_pts)

    for a, flav_bands in enumerate(all_bands):
        for j, mu_j in enumerate(mu_table):
            n_table[a, j] = get_flavor_density_from_bands(
                flav_bands, mu_j, T_eff, prefactor, unit_cell_to_cm2
            )
        # Sanity check: n_flav(mu) must be strictly monotone for np.interp to be valid.
        # A non-monotone table would silently produce wrong results.
        if not np.all(np.diff(n_table[a]) > 0):
            raise RuntimeError(
                f"Flavor {a}: n_flav(mu) is not strictly increasing on [{mu_min}, {mu_max}]. "
                "Widen the bracket or check the band structure."
            )

    return mu_table, n_table

def evaluate_realizations(n_flavs_seed, all_bands, mu_table, n_table,T_eff, prefactor, unit_cell_to_cm2, v_int_func):
    """
    For every randomly sample configuration, look up the flavor chemical potential via interpolation, compute kinetic energies and evaluate the interaction potential.

    The interaction potential is supplied as a **pre-configured callable** ``v_int_func(n_vec) -> float``.  All physical parameters of the interaction (U, J, area_uc, …) must be fixed in advance with functools.partial or a lambda, so that this function stays agnostic to the form of V_int. For example:
 
        from functools import partial
        v_int = partial(V_int_anisotropic, U=config.U, J=0, area_uc=config.area_uc)

    Parameters
    ----------
    n_flavs_seed : ndarray, shape(n_flavors, num_picks)
        Sampled density configurations, where each column sums to n_target.
    all_bands : list of (E0, E1) tuples, len = n_flavors
        Precomputed band pairs for each flavor.
    mu_table : ndarray, shape (n_table_pts,)
        Uniformly spaced mu points.
    n_table : ndarray, shape (n_flavors, n_table_pts)
        Flavor densities evaluated at each mu point.
    T_eff : float
        Thermal energy kB*T [eV].
    prefactor : float
        k-space integration prefactor (dk^2 / (2*pi^2)).
    unit_cell_to_cm2 : float
        Conversion factor from unit-cell units to cm^-2.
    v_int_func : callable
        Interaction energy function with signature ``v_int_func(n_vec) -> float``, where n_vec is a 1D array of length n_flavors.

    Returns
    -------
    mu_flav_arr : ndarray, shape (num_picks, n_flavors)
        Flavor chemical potential for each realization. NaN for inavlid ones.
    E_flav_arr : ndarray, shape (num_picks, n_flavors)
        Flavor kinetic energy for each realization [eV·cm^-2]. NaN for inavlid ones.
    V_int_vec : ndarray, shape (num_picks,)
        Interaction potential for each realization. NaN for invalid ones.
    valid_mask : ndarray of bool, shape (num_picks,)
        True for realizations where every flavor lies within the interpolation table range.
    """
    n_flavors, num_picks = n_flavs_seed.shape

    mu_flav_arr = np.full((num_picks, n_flavors), np.nan)
    E_flav_arr = np.full((num_picks, n_flavors), np.nan)
    V_int_vec = np.full(num_picks, np.nan)
    valid_mask = np.ones(num_picks, dtype=bool)

    # Step 1: invert n_flav(mu) for every flavor via interpolation
    for a, flav_bands in enumerate(all_bands):
        n_flav_targets = n_flavs_seed[a]    # shape (num_picks,): all the realizations for one flavor

        # Identify realizations where n_flav_target is out of the table range
        out_of_range = (n_flav_targets < n_table[a, 0]) | (n_flav_targets > n_table[a, -1])
        valid_mask &= ~out_of_range
        # WARNING: some values might have been pushed out of range by the rescaling after the random sampling to ensure the n_target constraint. Not a big deal since they are just not taken into account, but if they were an important percentage of the total points, the tables range should be increased.

        # Invert n_flav(mu) through the interpolation
        mu_flav_arr[:, a] = np.interp(n_flav_targets, n_table[a], mu_table)

    # Step 2: get the kinetic energies and interaction potentials (valid realizations only)
    for i in np.where(valid_mask)[0]:       # list of indices of the valid realizations
        for a, flav_bands in enumerate(all_bands):
            E_flav_arr[i, a] = get_flavor_kinetic_energy_from_bands(
                flav_bands, mu_flav_arr[i, a], T_eff, prefactor, unit_cell_to_cm2
            )

        V_int_vec[i] = v_int_func(n_flavs_seed[:, i])

    n_discarded = num_picks - np.count_nonzero(valid_mask)      # total - valid realizations = number of realizations out of bounds
    if n_discarded:
        print(f"Discarded {n_discarded}/{num_picks} realizations (target density out of range).")
    else:
        print("OK: All the realizations lie within the table range.")
    if not np.any(valid_mask):      # if valid_mask is full of False
        raise RuntimeError("No valid realizations found: widen [mu_min, mu_max] or m_seed")

    return mu_flav_arr, E_flav_arr, V_int_vec, valid_mask

def find_minimum_energy_configuration(n_flavs_seed, mu_flav_arr, E_flav_arr, V_int_vec, valid_mask):
    """
    Assemble the total internal energy E_int = E_kin + V_int for every valid realization and return the ground-state configuration (the one minimizing E_int).

    Parameters
    ----------
    n_flavs_seed : ndarray, shape(n_flavors, num_picks)
        Sampled density configurations, where each column sums to n_target.
    mu_flav_arr : ndarray, shape (num_picks, n_flavors)
        Flavor chemical potential for each realization. NaN for inavlid ones.
    E_flav_arr : ndarray, shape (num_picks, n_flavors)
        Flavor kinetic energy for each realization [eV·cm^-2]. NaN for inavlid ones.
    V_int_vec : ndarray, shape (num_picks,)
        Interaction potential for each realization. NaN for invalid ones.
    valid_mask : ndarray of bool, shape (num_picks,)
        True for realizations where every flavor lies within the interpolation table range.

    Returns
    -------
    n_sol : ndarray, shape (n_flavors,)
        Ground-state flavor densities [cm^-2].
    mu_flavs_sol : ndarray, shape (n_flavors,)
        Ground-state flavor chemical potentials [eV].
    E_int_min : float
        Minimal total internal energy [eV·cm^-2].
    """
    E_kin_vec = np.sum(E_flav_arr, axis=1)   # shape (num_picks,)
    E_int_vec = E_kin_vec + V_int_vec
    E_int_vec[~valid_mask] = np.inf     # invalid realizations never win

    E_int_min = np.min(E_int_vec)
    E_int_min_idx = np.argmin(E_int_vec)

    # Get the results
    n_sol  = n_flavs_seed[:, E_int_min_idx]
    mu_sol = mu_flav_arr[E_int_min_idx]

    return n_sol, mu_sol, E_int_min