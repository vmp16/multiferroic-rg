"""
Thermodynamic quantities: DOS, flavor densities, kinetic energies.

Functions
---------
precompute_flavor_bands
get_dos
get_flavor_density_from_bands
get_flavor_mu_from_densities
get_flavor_dos
get_flavor_kinetic_energy_from_bands
get_part_density
"""

import numpy as np
from scipy.optimize import brentq

from .kspace import fermi_distrib, deriv_fermi_distrib

__all__ = [
    "precompute_flavor_bands",
    "get_dos",
    "get_flavor_density_from_bands",
    "get_flavor_mu_from_densities",
    "get_flavor_dos",
    "get_flavor_kinetic_energy_from_bands",
    "get_part_density",
]


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


def get_flavor_mu_from_densities(n_flavs, all_bands, T_eff, prefactor, unit_cell_to_cm2, mu_min=-0.1, mu_max=0.1, xtol=1e-7, tables=None):
    """
    Invert n_alpha(mu) for each flavor: find the chemical potential mu_alpha that
    yields the density n_alpha on the single-particle band structure of flavor alpha.

    Flavor i of `n_flavs` is paired with `all_bands[i]`, so if fewer than 4 densities
    are given, the first len(n_flavs) flavors are used (same convention as
    solve_phase_mean_field).

    Parameters
    ----------
    n_flavs : array-like, length 1 to 4
        Flavor densities [cm^-2].
    all_bands : list of (E0, E1) tuples
        Precomputed band pairs, one per flavor (from precompute_flavor_bands).
    T_eff : float
        Thermal energy kB*T [eV].
    prefactor : float
        k-space integration prefactor dk^2 / (2*pi)^2.
    unit_cell_to_cm2 : float
        Conversion from unit-cell density to cm^-2.
    mu_min, mu_max : float
        Root-finder bracket [eV] (direct mode only).
    xtol : float
        Root-finder tolerance on mu [eV] (direct mode only).
    tables : tuple (mu_table, n_table) or None
        If given, mu_alpha is obtained by interpolating the tabulated n_alpha(mu)
        (e.g. from build_density_and_energy_tables or build_tables_histogram).
        Much faster, since it needs no k-sums. If None, each mu_alpha is found
        with brentq on the exact k-mesh sum.

    Returns
    -------
    mu_flavs : ndarray, shape (len(n_flavs),)
        Flavor chemical potentials [eV].

    Raises
    ------
    ValueError
        If the input has more than 4 entries, more entries than available bands,
        or if a density lies outside the bracket / table range.
    """
    n_flavs = np.atleast_1d(np.asarray(n_flavs, dtype=float))
    if n_flavs.ndim != 1 or not (1 <= len(n_flavs) <= 4):
        raise ValueError(f"n_flavs must be 1D with 1 to 4 entries, got shape {n_flavs.shape}.")
    if len(n_flavs) > len(all_bands):
        raise ValueError(f"{len(n_flavs)} densities given but only {len(all_bands)} band pairs available.")

    mu_flavs = np.empty(len(n_flavs))

    for a, n_a in enumerate(n_flavs):
        if tables is not None:
            mu_table, n_table = tables
            if n_a < n_table[a, 0] or n_a > n_table[a, -1]:
                raise ValueError(
                    f"Flavor {a}: n = {n_a:.4e} cm^-2 outside the tabulated range "
                    f"[{n_table[a, 0]:.4e}, {n_table[a, -1]:.4e}]."
                )
            mu_flavs[a] = np.interp(n_a, n_table[a], mu_table)
        else:
            def residual(mu):
                return get_flavor_density_from_bands(
                    all_bands[a], mu, T_eff, prefactor, unit_cell_to_cm2
                ) - n_a
            try:
                mu_flavs[a] = brentq(residual, mu_min, mu_max, xtol=xtol)
            except ValueError:
                raise ValueError(
                    f"Flavor {a}: n = {n_a:.4e} cm^-2 not bracketed by "
                    f"mu in [{mu_min}, {mu_max}] eV."
                )

    return mu_flavs


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
