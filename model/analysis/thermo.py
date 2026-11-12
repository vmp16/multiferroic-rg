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
get_density_matrix
get_fermi_pockets
"""

import numpy as np
from scipy.optimize import brentq
from scipy import ndimage

from .kspace import fermi_distrib, deriv_fermi_distrib

__all__ = [
    "precompute_flavor_bands",
    "get_dos",
    "get_flavor_density_from_bands",
    "get_flavor_mu_from_densities",
    "get_flavor_dos",
    "get_flavor_kinetic_energy_from_bands",
    "get_part_density",
    "get_density_matrix",
    "get_fermi_pockets",
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

def get_density_matrix(system, px, py, mu, T, reference="half"):
    """
    Build the density matrix for a given flavor, i.e. 2x2 matrix, using band-projectors.

    The density is calculated with respect to the given reference point:
        reference = "none"    : full rho              (Tr = f+ + f-)
                    "valence" : rho - P-              (Tr = f+ + f- - 1)
                    "half"    : rho - 1/2 * identity  (Tr = f+ + f- - 1)
    """
    # Build the bands because we also need abs(eigenvalue) = epsilon
    X, D = system.X_at_k(px, py), system.Delta
    eps = np.sqrt(D**2 + np.abs(X)**2)
    h0 = system.get_h0_at_k(px, py)

    E0, E1 = h0 + eps, h0 - eps

    fp = fermi_distrib(E0, mu, T)
    fm = fermi_distrib(E1, mu, T)

    s = 0.5 * (fp + fm) - (0.0 if reference == "none" else 0.5)
    num = 0.5 * (fp - fm) + (0.5 if reference == "valence" else 0.0)
    w = np.divide(num, eps, out=np.zeros_like(eps), where= eps > 1e-14)     # where -> avoid division by 0
                                                                            # out -> 0 whenever "where" is False

    rho = np.empty((2, 2, *np.shape(eps)), dtype=complex)
    rho[0, 0] = s + w * D
    rho[1, 1] = s - w * D
    rho[0, 1] = w * X
    rho[1, 0] = w * np.conj(X)
    
    return rho


def get_fermi_pockets(bands, mu, KX, KY, dk, unit_cell_to_cm2,
                      min_pixels=20, connectivity=1):
    """
    Identify and measure the Fermi pockets of one flavor on the 2D k-mesh.

    One pocket = one closed Fermi contour. Each contour encloses a region of k-space and has a character:
      - Outer contour of a region E0 < mu            -> electron pocket (+)
      - Outer contour of a region E1 > mu            -> hole pocket (-)
      - Contour of a hole inside such a region       -> pocket of the OPPOSITE
        character (e.g. a ring-shaped hole region gives an outer hole pocket
        and an inner electron-like pocket).
    The inner pocket is "electron-like" in the sense that it encloses occupied states of the valence band, not conduction-band electrons.

    Each pocket's area is the area ENCLOSED by its contour, so nested pockets overlap (the outer one includes the inner one). The net carrier area of a ring is area_outer - area_inner, which is exactly what the signed densities n_cm2 add up to. T -> 0 definition (sharp Fermi step).

    Connectivity: foreground regions use `connectivity` (1 = 4-neighbour, 2 = 8-neighbour) and holes use the dual one (3 - connectivity), so that closed contours are topologically consistent.

    Parameters
    ----------
    bands : tuple (E0, E1)
        Upper and lower band energies on the 2D mesh [eV].
    mu : float
        Flavor chemical potential mu_alpha [eV].
    KX, KY : ndarray
        2D k-mesh (same shape as the bands).
    dk : float
        Mesh spacing.
    unit_cell_to_cm2 : float
        Conversion from unit-cell density to cm^-2.
    min_pixels : int
        Pockets enclosing fewer mesh points than this are discarded as noise.
    connectivity : 1 or 2

    Returns
    -------
    pockets : list of dict, sorted by enclosed area (largest first)
        id, kind ('electron'/'hole'), inner (True if it is a hole inside another pocket), parent (id of the smallest enclosing pocket, or None), area [(1/a)^2] enclosed, frac (area / total_area), center (kx, ky) of the enclosed region, n_cm2 (signed density: electron > 0, hole < 0), touches_edge (outer contour reaches the mesh boundary: truncated).
    label_map : ndarray of int32
        0 = no pocket. Nested pockets are painted from large to small, so a ring has the outer id on the ring and the inner id on its hole.
    total_area : float
        Sum of the enclosed areas of all pockets (nested ones counted in full).
    """
    E0, E1 = bands
    st_fg = ndimage.generate_binary_structure(2, connectivity)
    st_bg = ndimage.generate_binary_structure(2, 3 - connectivity)

    contours = []
    for kind, mask, sign in (("electron", E0 < mu, +1.0), ("hole", E1 > mu, -1.0)):
        if not mask.any():
            continue
        lab, _ = ndimage.label(mask, structure=st_fg)
        border = set(np.unique(np.concatenate(
            [lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])).tolist()) - {0}

        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            comp = lab[sl] == i
            filled = ndimage.binary_fill_holes(comp, structure=st_bg)
            if filled.sum() < min_pixels:
                continue

            # Outer contour: encloses the component with its holes filled
            contours.append(dict(kind=kind, sign=sign, inner=False, sl=sl,
                                 mask=filled, touches_edge=(i in border)))

            # Inner contours: one per hole (islands inside a hole stay enclosed)
            holes, n_h = ndimage.label(filled & ~comp, structure=st_bg)
            for j in range(1, n_h + 1):
                hmask = holes == j
                if hmask.sum() < min_pixels:
                    continue
                contours.append(dict(
                    kind="hole" if kind == "electron" else "electron",
                    sign=-sign, inner=True, sl=sl, mask=hmask, touches_edge=False))
        del lab

    # Largest first: ids follow enclosed area, and painting nests correctly
    contours.sort(key=lambda c: int(c["mask"].sum()), reverse=True)
    label_map = np.zeros(E0.shape, dtype=np.int32)
    for idx, c in enumerate(contours, start=1):
        c["id"] = idx
        c["area_px"] = int(c["mask"].sum())
        label_map[c["sl"]][c["mask"]] = idx

    # Parent = smallest strictly larger pocket whose enclosed region contains it
    for c in contours:
        r, q = np.argwhere(c["mask"])[0]
        r += c["sl"][0].start
        q += c["sl"][1].start
        c["parent"] = None
        for d in reversed(contours):            # smallest first
            if d["id"] == c["id"] or d["area_px"] <= c["area_px"]:
                continue
            rr, qq = r - d["sl"][0].start, q - d["sl"][1].start
            if (0 <= rr < d["mask"].shape[0] and 0 <= qq < d["mask"].shape[1]
                    and d["mask"][rr, qq]):
                c["parent"] = d["id"]
                break

    pockets = []
    for c in contours:
        area = c["area_px"] * dk**2
        pockets.append({
            "id": c["id"],
            "kind": c["kind"],
            "inner": c["inner"],
            "parent": c["parent"],
            "area": area,
            "center": (float(KX[c["sl"]][c["mask"]].mean()),
                       float(KY[c["sl"]][c["mask"]].mean())),
            "n_cm2": c["sign"] * area / (2 * np.pi)**2 * unit_cell_to_cm2,
            "touches_edge": c["touches_edge"],
        })

    total_area = sum(p["area"] for p in pockets)
    for p in pockets:
        p["frac"] = p["area"] / total_area if total_area > 0 else np.nan

    return pockets, label_map, total_area