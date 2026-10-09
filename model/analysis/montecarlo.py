"""
Monte Carlo ground-state search over flavor-density configurations.

Functions
---------
sample_flavor_densities
build_density_interp_tables
evaluate_realizations
find_minimum_energy_configuration
"""

import numpy as np

from .thermo import get_flavor_density_from_bands, get_flavor_kinetic_energy_from_bands

__all__ = [
    "sample_flavor_densities",
    "build_density_interp_tables",
    "evaluate_realizations",
    "find_minimum_energy_configuration",
]


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
        Optional seeded RNG (e.g. np.random.default_rng(41)). If None, the global numpy random state is used.

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
    RuntimeError
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
    For every randomly sampled configuration, look up the flavor chemical potential via interpolation, compute kinetic energies and evaluate the interaction potential.

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
        Flavor chemical potential for each realization. NaN for invalid ones.
    E_flav_arr : ndarray, shape (num_picks, n_flavors)
        Flavor kinetic energy for each realization [eV·cm^-2]. NaN for invalid ones.
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
        n_flav_targets = n_flavs_seed[a]    # shape (num_picks,): all the realizations for one flavor

        # Identify realizations where n_flav_target is out of the table range
        out_of_range = (n_flav_targets < n_table[a, 0]) | (n_flav_targets > n_table[a, -1])
        valid_mask &= ~out_of_range
        # WARNING: some values might have been pushed out of range by the rescaling after the random sampling
        # to ensure the n_target constraint. Not a big deal since they are just not taken into account,
        # but if they were an important percentage of the total points, the tables range should be increased.

        # Invert n_flav(mu) through the interpolation
        mu_flav_arr[:, a] = np.interp(n_flav_targets, n_table[a], mu_table)

    # Step 2: get the kinetic energies and interaction potentials (valid realizations only)
    for i in np.where(valid_mask)[0]:       # list of indices of the valid realizations
        for a, flav_bands in enumerate(all_bands):
            E_flav_arr[i, a] = get_flavor_kinetic_energy_from_bands(
                flav_bands, mu_flav_arr[i, a], T_eff, prefactor, unit_cell_to_cm2
            )

        V_int_vec[i] = v_int_func(n_flavs_seed[:, i])

    n_discarded = num_picks - np.count_nonzero(valid_mask)
    if n_discarded:
        print(f"Discarded {n_discarded}/{num_picks} realizations (target density out of range).")
    else:
        print("OK: All the realizations lie within the table range.")
    if not np.any(valid_mask):
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
        Flavor chemical potential for each realization. NaN for invalid ones.
    E_flav_arr : ndarray, shape (num_picks, n_flavors)
        Flavor kinetic energy for each realization [eV·cm^-2]. NaN for invalid ones.
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
