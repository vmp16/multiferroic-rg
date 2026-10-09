"""
Phase diagram support: tabulated n(mu) and E_kin(mu), table-based SCF, energy evaluation,
and flavor counting / classification.

Functions
---------
build_density_and_energy_tables
build_tables_histogram
solve_scf_from_tables
internal_energy_from_tables
count_occupied_flavors
count_degenerate_flavors
"""

import numpy as np
from scipy.optimize import brentq

from .kspace import fermi_distrib
from .thermo import get_flavor_density_from_bands, get_flavor_kinetic_energy_from_bands
from .interactions import compute_mf_symmetric_potentials, V_int_SU4

__all__ = [
    "build_density_and_energy_tables",
    "build_tables_histogram",
    "solve_scf_from_tables",
    "internal_energy_from_tables",
    "count_occupied_flavors",
    "count_degenerate_flavors",
]


def build_density_and_energy_tables(all_bands, T_eff, prefactor, unit_cell_to_cm2, mu_min=-0.1, mu_max=0.1, n_table_pts=2000):
    """
    Build tabulated tables of n_alpha(mu) and E_kin,alpha(mu) for every flavor on a common mu grid,
    to be used inverted through interpolation.

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
    RuntimeError
        If n_flav(mu) is not strictly monotone for any flavor, which would make the inversion via np.interp invalid.
    """
    n_flavors = len(all_bands)
    mu_table = np.linspace(mu_min, mu_max, n_table_pts)
    n_table = np.zeros((n_flavors, n_table_pts))
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
 
        # Drop empty bins: typically most of them, and this shrinks the inner loop considerably
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
    Self-consistent mean-field calculation using interpolation tables. All four flavors are always active;
    a flavor that empties out does so on its own by acquiring a vanishing density.

    The root finder bracket for the global Fermi Level is chosen so that mu_global stays inside
    the tabulated mu range for every flavor. This avoids a plausible-looking but wrong solution
    produced by np.interp.

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
    e_table : ndarray, shape (n_flavors, n_table_pts)
        Tabulated flavor kinetic energies.
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
    Count how many flavors carry a density above a magnitude threshold, separating actual occupation
    from thermal residue. The absolute value is used because carrier densities might be negative or positive.

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


def count_degenerate_flavors(n_flavs, phase_list, threshold=1e6, tol=1e5, rtol=None):
    """
    Classify a converged flavor configuration by the partition of its occupied densities into clusters of equal value.

    Step 1: identify occupied flavors as |n_alpha| > threshold (same criterion as count_occupied_flavors).
    Step 2: union-find the occupied densities only, merging two flavors whenever they are within tol (or rtol)
            of each other. This is transitive: if a~b and b~c, a and c end up in the same cluster even if
            not directly within tolerance of one another.
    Step 3: read off the cluster sizes with np.unique_counts and sort them descending to build a canonical
            partition string, e.g. "2-1" for a 3-active configuration where two flavors share a density
            and the third is distinct.

    Parameters
    ----------
    n_flavs : array-like, shape (n_flavors,)
        Converged flavor densities [cm^-2].
    phase_list : dict
        Mapping from partition string to phase index (e.g. config.PHASE_ORDER as a dict).
    threshold : float
        Magnitude below which a flavor counts as empty [cm^-2].
    tol : float
        Absolute tolerance for two active densities to be called equal.
    rtol : float or None
        If given, used instead of tol as a relative tolerance.
 
    Returns
    -------
    partition_str : str
        Canonical partition label, one of PHASE_ORDER. "0" if no flavor is occupied.
    phase_index : int
        Index of partition_str in PHASE_ORDER, ready to store in a numeric map for colormap lookup.
        -1 if partition_str == "0".
    """
    n_flavs = np.asarray(n_flavs, dtype=float)
 
    active_idx = np.where(np.abs(n_flavs) > threshold)[0]
    n_active = len(active_idx)

    if n_active == 0:
        return "0", -1
 
    active_n = n_flavs[active_idx]
 
    def close(i, j):
        if rtol is not None:
            scale = max(abs(active_n[i]), abs(active_n[j]))
            return abs(active_n[i] - active_n[j]) <= rtol * scale
        return abs(active_n[i] - active_n[j]) <= tol
 
    parent = list(range(n_active))
 
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
 
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
 
    for i in range(n_active):
        for j in range(i + 1, n_active):
            if close(i, j):
                union(i, j)
 
    parent_arr = np.array([find(i) for i in range(n_active)])
 
    # np.unique_counts requires NumPy >= 2.0; fall back to the classic np.unique(..., return_counts=True) if unavailable.
    if hasattr(np, "unique_counts"):
        counts = np.unique_counts(parent_arr).counts
    else:
        _, counts = np.unique(parent_arr, return_counts=True)
 
    counts = np.sort(counts)[::-1]     # descending, e.g. [2, 1]
    partition_str = "-".join(str(c) for c in counts)
 
    return partition_str, phase_list[partition_str]
