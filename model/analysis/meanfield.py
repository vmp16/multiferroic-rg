"""
Self-consistent mean-field (SCF) solvers.

Functions
---------
solve_self_consistent_mean_field
solve_phase_mean_field
"""

import numpy as np
from scipy.optimize import brentq

from .kspace import build_systems_with_displacement
from .thermo import precompute_flavor_bands, get_flavor_density_from_bands
from .interactions import compute_mf_symmetric_potentials

__all__ = [
    "solve_self_consistent_mean_field",
    "solve_phase_mean_field",
]


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

    The SCF minimizes the SU(4)-symmetric potential by iterating over the differential of the grand potential. Empty (inactive) flavors are kept at n = 0 throughout.

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
    # WARNING: WE JUST TAKE THE FIRST BANDS
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
