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
 
    valley_indices = 2 * config.VALLEY_IDX          # [1,-1] → [1,-1,1,-1]
    base_deltas    = config.DELTAS.T.flatten()       # shape (4,)
    e0_vals        = config.E0_ARRAY.T.flatten()     # shape (4,)
 
    systems = [
        McCannCarts(
            N       = config.N,
            valley_idx = xi,
            Delta   = delta + D,
            gamma0  = config.GAMMA0,
            gamma1  = config.GAMMA1,
            gamma2  = config.GAMMA2,
            gamma3  = config.GAMMA3,
            gamma4  = config.GAMMA4,
        )
        for xi, delta, e0 in zip(valley_indices, base_deltas, e0_vals)
    ]
    return systems, base_deltas


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


def solve_phase_mean_field(n_active, n_target, D, KX, KY, config, *, initial_n_flavs=None, mu_min=-0.1, mu_max=0.1, max_iter=200, tolerance=1e-6, mixing=0.4, verbose=True):
    """
    Run a self-consistent mean-field calculation (SCF) for a given number of occupied (active) flavors under a displacement field D.

    The SCF minimizes the SU(4)-symmetric potential by iterating over the differential of the grand potential. Empty (inactive) flavors are kept at n = 0 trhoughout.

    Parameters
    ----------
    n_active : int
        Number of occupied flavors (1 to 4).
    n_target : float
        Total carrier density [cm^-2].
    D : float
        Displacement field [eV].
    KX, KY : ndarray, shape (N_PTS, N_PTS)
        Pre-built cartesian k-space grid.
    config : module
        Project configuration module.
    initial_n_flavs : array-like, length n_active
        Seed flavor densities for the active flavors. Default to the symmetric point n_target / n_active.
    mu_min, mu_max : float
        Bracket for the root finder [eV].
    max_iter : int
        Maximum SCF iterations.
    tolerance : float
        Convergence criterion for the SCF.
    mixing : float
        Linear mixing parameter.
    verbose : bool
        Print iteration log and convergence message.

    Returns
    -------
    result : dict
    
    Raises
    ------
    """
    # Check: valid number of occupied flavors
    if not (1 <= n_active <= 4):
        raise ValueError(f"n_active must be 1-4, got {n_active}")

    # ----------- Build systems & Precompute all 4 bands -----------
    systems, base_deltas = build_systems_with_displacement(D, config)

    dk = KX[0, 1] - KX[0, 0]
    prefactor = (dk**2) / (2 * np.pi)**2

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Phase: {n_active} active flavor(s) | n_target = {n_target:.3e} cm⁻² | D = {D*1e3:.2f} meV")
        print(f"  Gaps (delta + D) [meV]: " + ", ".join(f"{(d+D)*1e3:.2f}" for d in base_deltas))
        print(f"{'='*60}")
        print("  Precomputing bands for all 4 flavors...")

    all_bands = precompute_flavor_bands(systems, KX, KY)
    # Active subset of bands
    # WARINING: WE JUST TAKE THE FIRST BANDS
    active_bands = all_bands[:n_active]

    # ----------- Initial Guesses for Flavor Densities -----------
    if initial_n_flavs is None:
        # Symmetric seed
        n_flavs_active = np.full(n_active, n_target / n_active)
    else:
        n_flavs_active = np.array(initial_n_flavs, dtype=float)
        if len(n_flavs_active) != n_active:
            raise ValueError(f"initial_n_flavs has length {len(initial_n_flavs)}, expected {n_active}.")

    # ----------- Root Function for Fermi Level -----------
    def total_density_residual(mu_global, V_flavs):
        n_tot = 0.0
        for idx in range(n_active):
            mu_alpha = mu_global - V_flavs[idx]
            n_tot += get_flavor_density_from_bands(
                all_bands[idx], mu_alpha, config.T_eff, prefactor, config.unit_cell_to_cm2
            )
        return n_tot - n_target

    # ----------- Exception: n_active = 1 -----------
    if n_active == 1:
        mu_global = brentq(
            lambda mu: get_flavor_density_from_bands(
                active_bands[0], mu, config.T_eff, prefactor, config.unit_cell_to_cm2
            ) - n_target,
            mu_min, mu_max,
            xtol=1e-9,
        )
        n_flavs_active = np.array([n_target])
        V_active       = np.array([0.0])
        converged      = True

        if verbose:
            print(f"  n_active=1: no self-interaction, solved directly.")
            print(f"  mu_global = {mu_global*1e3:.4f} meV\n")

        # Assemble and return immediately, skipping the SCF loop
        n_flavs_full              = np.zeros(4)
        V_flavs_full              = np.zeros(4)
        mu_flavs_full             = np.full(4, np.nan)
        n_flavs_full[0]           = n_target
        mu_flavs_full[0]          = mu_global

        return {
            "n_active"  : n_active,
            "D"         : D,
            "n_flavs"   : n_flavs_full,
            "V_flavs"   : V_flavs_full,
            "mu_global" : mu_global,
            "mu_flavs"  : mu_flavs_full,
            "converged" : converged,
            "all_bands" : all_bands,
            "prefactor" : prefactor,
        }

    # ----------- SCF Iteration Loop -----------
    converged = False
    mu_global = 0.0
    V_active = np.zeros(n_active)

    if verbose:
        print("   Starting SCF loop...")

    for iteration in range(max_iter):
        # Calculate interaction potentials
        V_active = compute_mf_symmetric_potentials(n_flavs_active, config.U, config.area_uc)

        # ----- DEBUG -----
        # print(f"  [debug] n_flavs_active = {n_flavs_active}")
        # print(f"  [debug] V_active = {V_active}")
        # -----------------

        # Find mu_global satisfying the total density constraint
        try:
            mu_global = brentq(total_density_residual, mu_min, mu_max, args=(V_active,), xtol=1e-8)
        except ValueError:
            raise ValueError(
                f"Iteration {iteration}: Target density outside mu bracket [{mu_min}, {mu_max}] eV."
            )

        # Update densities for active flavors
        n_flavs_new = np.array([
            get_flavor_density_from_bands(
                active_bands[i], mu_global - V_active[i], config.T_eff, prefactor, config.unit_cell_to_cm2
            )
            for i in range(n_active)
        ])

        # Evaluate the error & Check convergence
        rel_error = np.max(np.abs(n_flavs_new - n_flavs_active) / np.abs(n_target))

        if verbose:
            print(
                f"Iter {iteration:02d} | mu_global: {mu_global*1e3:.3f} meV | Max Error: {rel_error:.3e} | n_flavs: ", np.array2string(n_flavs_active, formatter={'float': '{:.4e}'.format})
            )

        if rel_error < tolerance:
            n_flavs_active = n_flavs_new
            # Recompute V and mu with the final densities
            V_active = compute_mf_symmetric_potentials(n_flavs_active, config.U, config.area_uc)
            mu_global = brentq(total_density_residual, mu_min, mu_max, args=(V_active,), xtol=1e-7)
            
            converged = True
            if verbose:
                print(f"   ---> Converged in {iteration + 1} iterations!")
            break
        
        # Linear mixing
        n_flavs_active = (1.0 - mixing) * n_flavs_active + mixing * n_flavs_new
    
    else:
        if verbose:
            print("\n---> Warning: Reached maximum iterations without full convergence.")

    # Assemble full 4-flavor arrays
    n_flavs_full = np.zeros(4)
    V_flavs_full = np.zeros(4)
    n_flavs_full[:n_active] = n_flavs_active
    V_flavs_full[:n_active] = V_active
 
    mu_flavs_full = mu_global - V_flavs_full   # mu_alpha for all flavors
    # For inactive flavors, the flavor Fermi level is not physically
    # meaningful; set to NaN to make this explicit.
    mu_flavs_full[n_active:] = np.nan
 
    return {
        "n_active" : n_active,
        "D" : D,
        "n_flavs" : n_flavs_full,
        "V_flavs" : V_flavs_full,
        "mu_global" : mu_global,
        "mu_flavs" : mu_flavs_full,
        "converged" : converged,
        "all_bands" : all_bands,
        "prefactor" : prefactor,
    }


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

def evaluate_realizations(n_flavs_seed, all_bands, mu_table, n_table, T_eff, prefactor, unit_cell_to_cm2, v_int_func):
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

# ================== PHASE DIAGRAM SUPPORT ==================

def build_density_and_energy_tables(all_bands, T_eff, prefactor, unit_cell_to_cm2, mu_min=-0.1, mu_max=0.1, n_table_pts=2000):
    """
    Build tabulated tables of n_alpha(mu) and E_kin,alpha(mu) for every flavor on a common mu grid, to be used inverted through interpolation.

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
    e_table : ndarray, shape (n_flavors, n_table_pts)
        Flavor kinetic energies at each mu point [eV cm^-2].

    Raises
    ------
    RunTimeError
        If n_flav(mu) is not strictly monotone for any flavor, which would make the inversion via np.interp invalid.
    """
    n_flavors = len(all_bands)
    mu_table = np.linspace(mu_min, mu_max, n_table_pts)
    n_table = np.zeros((n_flavors, n_table_pts))    # n_flav(mu) per flavor, shape (4, n_table_pts)
    e_table = np.zeros((n_flavors, n_table_pts))

    for a, flav_bands in enumerate(all_bands):
        for j, mu_j in enumerate(mu_table):
            n_table[a, j] = get_flavor_density_from_bands(
                flav_bands, mu_j, T_eff, prefactor, unit_cell_to_cm2
            )
            e_table[a, j] = get_flavor_kinetic_energy_from_bands(
                flav_bands, mu_j, T_eff, prefactor, unit_cell_to_cm2
            )
        # Sanity check: n_flav(mu) must be strictly monotone for np.interp to be valid.
        # A non-monotone table would silently produce wrong results.
        if not np.all(np.diff(n_table[a]) > 0):
            raise RuntimeError(
                f"Flavor {a}: n_flav(mu) is not strictly increasing on [{mu_min}, {mu_max}]. "
                "Widen the bracket or check the band structure."
            )
    return mu_table, n_table, e_table

def build_tables_histogram(
    all_bands, T_eff, prefactor, unit_cell_to_cm2,
    mu_min=-0.30, mu_max=0.30, n_table_pts=3000, n_bins=60000,
):
    """
    Build n_alpha(mu) and E_kin,alpha(mu) tables by histogramming the band
    energies once and convolving with the Fermi function.
 
    The direct approach evaluates a full k-mesh sum at every mu, costing
    O(N_mu * N_k).  But the summand depends on k only through the band
    energy, so binning the energies into a density of states and then
    convolving costs O(N_k + N_mu * N_bins) instead - typically two orders
    of magnitude faster for a 1500^2 mesh.
 
    Accuracy is controlled by the bin width, which must be small compared
    to the thermal smearing: aim for dE <~ T_eff / 5.  The function warns
    if that is not satisfied.
 
    Parameters
    ----------
    all_bands : list of (E0, E1) tuples
        Precomputed band pairs, one per flavor.
    T_eff : float
        Thermal energy kB*T [eV].
    prefactor : float
        k-space integration prefactor dk^2 / (2*pi)^2.
    unit_cell_to_cm2 : float
        Conversion from unit-cell density to cm^-2.
    mu_min, mu_max : float
        Bounds of the mu grid [eV].
    n_table_pts : int
        Number of mu samples.
    n_bins : int
        Number of energy bins for the histograms.
 
    Returns
    -------
    mu_table : ndarray, shape (n_table_pts,)
    n_table  : ndarray, shape (n_flavors, n_table_pts)
    e_table  : ndarray, shape (n_flavors, n_table_pts)
    """
    n_flavors = len(all_bands)
    mu_table = np.linspace(mu_min, mu_max, n_table_pts)
    n_table = np.zeros((n_flavors, n_table_pts))
    e_table = np.zeros((n_flavors, n_table_pts))
 
    conv = prefactor * unit_cell_to_cm2
 
    # Common energy range across all flavors and both bands
    e_lo = min(min(b[0].min(), b[1].min()) for b in all_bands)
    e_hi = max(max(b[0].max(), b[1].max()) for b in all_bands)
    pad = 10.0 * T_eff
    edges = np.linspace(e_lo - pad, e_hi + pad, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dE = edges[1] - edges[0]
 
    if dE > T_eff / 5.0:
        print(
            f"  Warning: energy bin width {dE*1e3:.4f} meV exceeds T_eff/5 = "
            f"{T_eff/5*1e3:.4f} meV. Increase n_bins for accurate tables."
        )
 
    for a, (E0, E1) in enumerate(all_bands):
        # Density of states per band (counts per bin, unnormalized)
        H0, _ = np.histogram(E0.ravel(), bins=edges)
        H1, _ = np.histogram(E1.ravel(), bins=edges)
 
        # Drop empty bins: typically most of them, and this shrinks the
        # inner loop considerably
        nz0 = H0 > 0
        nz1 = H1 > 0
        c0, w0 = centers[nz0], H0[nz0].astype(float)
        c1, w1 = centers[nz1], H1[nz1].astype(float)
 
        for j, mu_j in enumerate(mu_table):
            f0 = fermi_distrib(c0, mu_j, T_eff)          # electron occupation
            g1 = 1.0 - fermi_distrib(c1, mu_j, T_eff)    # hole occupation
 
            n_e = np.dot(w0, f0)
            n_h = np.dot(w1, g1)
            n_table[a, j] = conv * (n_e - n_h)
 
            e_e = np.dot(w0, c0 * f0)
            e_h = np.dot(w1, c1 * g1)
            e_table[a, j] = conv * (e_e - e_h)
 
        if not np.all(np.diff(n_table[a]) > 0):
            raise RuntimeError(
                f"Flavor {a}: n(mu) not strictly increasing. "
                "Increase n_bins or widen [mu_min, mu_max]."
            )
 
    return mu_table, n_table, e_table

def solve_scf_from_tables(
    n_target, mu_table, n_table, U, area_uc, initial_n_flavs, max_iter=300, tolerance=1e-6, mixing=0.4
):
    """
    Self-consistent mean-field calculation using interpolation tables. All four flavors are always active, a flavor that empties out does so on its own by acquiring a vanishing density.

    The root finder bracket for the global Fermi Level is chosen so that mu_global - stays inside the tabulated mu range for every flavor. This avoids a plausible-looking but wrong solution produced by np.interp.

    Parameters
    ----------
    n_target : float
        Total carrier density [cm^-2].
    mu_table : ndarray, shape (n_table_pts,)
        Uniformly spaced mu points.
    n_table : ndarray, shape (n_flavors, n_table_pts)
        Tabulated flavor densities
    U : float
        Interaction amplitude [eV].
    area_uc : float
        Unit cell area [cm^2].
    initial_n_flavs : array-like, shape (n_flavors,)
        Seed densities. Must not be exactly the symmetric point to allow spontaneous polarization.
    max_iter : int
        Maximum SCF iterations.
    tolerance : float
        Convergence criterion for the SCF.
    mixing : float
        Linear mixing parameter.

    Returns
    -------
    dict with keys 'n_flavs', 'V_flavs', 'mu_global', 'mu_flavs', 'converged', 'n_iter', or None if the root finder failed to bracket.
    """
    n_flavors = n_table.shape[0]
    n_flavs = np.array(initial_n_flavs, dtype=float)

    def total_density_residual(mu_global, V):
        n_tot = 0.0
        for a in range(n_flavors):
            n_tot += np.interp(mu_global - V[a], mu_table, n_table[a])
        return n_tot - n_target

    converged = False
    mu_global = np.nan
    V_flavs = np.zeros(n_flavors)

    for it in range(max_iter):
        # Compute interaction potential
        V_flavs = compute_mf_symmetric_potentials(n_flavs, U, area_uc)

        # Keep mu_global - V_alpha inside the table for all alpha
        lo = mu_table[0] + np.max(V_flavs)
        hi = mu_table[-1] + np.min(V_flavs)
        if lo >= hi:
            return None

        # Get mu_global satisfying the total density constraint
        try:
            mu_global = brentq(total_density_residual, lo, hi, args=(V_flavs,), xtol=1e-9)
        except ValueError:
            return None

        # Update the densities
        # n_(mu_alpha), with mu_alpha = mu_global - V_alpha
        n_new = np.array([
            np.interp(mu_global - V_flavs[a], mu_table, n_table[a])
            for a in range(n_flavors)
        ])

        # Evaluate the error & Check convergence
        rel_error = np.max(np.abs(n_new - n_flavs) / np.abs(n_target))

        if rel_error < tolerance:
            n_flavs = n_new
            # Recompute V and mu with the final densities
            V_flavs = compute_mf_symmetric_potentials(n_flavs, U, area_uc)
            lo = mu_table[0] + np.max(V_flavs)
            hi = mu_table[-1] + np.min(V_flavs)
            try:
                mu_global = brentq(total_density_residual, lo, hi, args=(V_flavs,), xtol=1e-11)
            except ValueError:
                return None
            converged = True
            break

        # Linear mixing for next iteration
        n_flavs = (1.0 - mixing) * n_flavs + mixing * n_new

    return {
        "n_flavs":   n_flavs,
        "V_flavs":   V_flavs,
        "mu_global": mu_global,
        "mu_flavs":  mu_global - V_flavs,
        "converged": converged,
        "n_iter":    it + 1,
    }

def internal_energy_from_tables(n_flavs, mu_flavs, mu_table, e_table, U, area_uc):
    """
    Total internal energy E_int = E_kin + V_int for a converged configuration, evaluated by interpolation.

    Parameters
    ----------
    n_flavs : ndarray, shape (n_flavors,)
        Converged flavor densities.
    mu_flavs : ndarray, shape (n_flavors,)
        Converged flavor Fermi levels.
    mu_table : ndarray, shape (n_table_pts,)
        Uniformly spaced mu points.
    n_table : ndarray, shape (n_flavors, n_table_pts)
        Tabulated flavor densities
    U : float
        Interaction amplitude [eV].
    area_uc : float
        Unit cell area [cm^2].

    Returns
    -------
    E_int : float
        Total internal energy [eV cm^-2].
    """
    E_kin = sum(
        np.interp(mu_flavs[a], mu_table, e_table[a])
        for a in range(len(n_flavs))
    )
    V_int = V_int_SU4(n_flavs, U, area_uc)
    return E_kin + V_int

def count_occupied_flavors(n_flavs, threshold=1e6):
    """
    Count how many flavors carry a density above a magnitude threshold, separating actual occupation from thermal residue. The absolute value is used because carrier densities might be negative or positive.

    Parameters
    ----------
    n_flavs : ndarray, shape (n_flavors,)
        Converged flavor densities.
    threshold : float
        Magnitude below which a flavor counts as empty [cm^-2].

    Returns
    -------
    int
    """

    return int(np.count_nonzero(np.abs(n_flavs) > threshold))