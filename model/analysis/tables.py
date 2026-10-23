"""
Building density and kinetic energy tables

Functions
---------
build_density_and_energy_tables
build_tables_unique
build_tables_histogram
"""

import numpy as np
import hashlib
import json
from pathlib import Path

from .kspace import fermi_distrib
from .thermo import get_flavor_density_from_bands, get_flavor_kinetic_energy_from_bands
from .kspace import build_systems_with_displacement


__all__ = [
    "build_density_and_energy_tables",
    "build_tables_unique",
    "build_tables_histogram",
    "table_cache_key",
    "get_or_build_tables",
]


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

def build_tables_unique(systems, KX, KY, T_eff, prefactor, unit_cell_to_cm2, mu_min=-0.1, mu_max=0.1, n_table_pts=2000, decimals=12):
    """
    Build tabulated tables of n_alpha(mu) and E_kin,alpha(mu) on a common mu grid, one per distinct (|Delta|, E0), to be used inverted through interpolation.

    Parameters
    ----------
    systems : list of McCannCarts
        List of initialized model systems representing different flavors.
    KX, KY : ndarray
        2D k-space coordinate meshes.
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
    decimals : int
        Number of decimals for the rounding of Delta and E0.

    Returns
    -------
    mu_table : ndarray, shape (n_table_pts,)
        Uniformly spaced mu points.
    n_table : ndarray, shape (n_flavors, n_table_pts)
        Flavor densities evaluated at each mu point.
    e_table : ndarray, shape (n_flavors, n_table_pts)
        Flavor kinetic energies at each mu point [eV cm^-2].
    """
    keys = np.array([
        [abs(syst.Delta), syst.E0] for syst in systems
    ])
    keys = np.round(np.array([[abs(s.Delta), s.E0] for s in systems]), decimals=decimals)
    _, rep_idx, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)     # shape (n_flavors,) in every NumPy version
    # rep_idx = indices of the first appearance of the elements in keys
    # inverse = positions of repeated values, to reconstruc the full list

    # Bands only for one representative per (|Delta|, E0) values
    rep_bands = [
        systems[i].get_energy(KX, KY) for i in rep_idx
    ]

    mu_table, n_uni, e_uni = build_density_and_energy_tables(
        rep_bands, T_eff, prefactor, unit_cell_to_cm2, mu_min=mu_min, mu_max=mu_max, n_table_pts=n_table_pts,
    )

    # Expand to all flavors
    n_full, e_full = n_uni[inverse], e_uni[inverse]
    assert n_full.shape == n_full.shape == (len(systems), n_table_pts), f"bad table shape {n_full.shape}"
    return mu_table, n_full, e_full


def build_tables_histogram(
    all_bands, T_eff, prefactor, unit_cell_to_cm2,
    mu_min=-0.30, mu_max=0.30, n_table_pts=3000, n_bins=60000,
):
    """
                WARNING: NOT WORKING PROPERLY

    Build n_alpha(mu) and E_kin,alpha(mu) tables by histogramming the band energies once and convolving with the Fermi function.
 
    The direct approach evaluates a full k-mesh sum at every mu, costing O(N_mu * N_k).  But the summand depends on k only through the band energy, so binning the energies into a density of states and then convolving costs O(N_k + N_mu * N_bins) instead - typically two orders of magnitude faster for a 1500^2 mesh.
 
    Accuracy is controlled by the bin width, which must be small compared to the thermal smearing: aim for dE <~ T_eff / 5. The function warns if that is not satisfied.
 
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

CACHE_VERSION = 1

def table_cache_key(D, config, KX, mu_min, mu_max, n_table_pts):
    """
    Build a short, reproducible fingerprint (16 hex characters) of every input that influences the mu / n / E_kin tables.
 
    Two calls with identical inputs always return the same key; changing ANY input (D, a hopping parameter, the k-mesh, T_eff, ...) returns a different key. The key is used as part of the cache filename, so a change in the inputs makes the old file "invisible" and forces a rebuild.

    Deliberately NOT in the key: U, J, n_target, the SCF settings and the random seeds.

    Parameters
    ----------
    D : float
        Displacement field [eV].
    config : module
        Project config module (N, GAMMA0..4, DELTAS, VALLEY_IDX, a, T_eff).
    KX : ndarray, shape (n_pts, n_pts)
        The k-mesh actually used. The mesh size and k_lim are read from it,
        so the key can never disagree with the mesh you really integrate on.
    mu_min, mu_max : float
        Limits of the chemical-potential grid [eV].
    n_table_pts : int
        Number of mu points in the table.
    method : {"direct", "histogram"}
        Which builder produces the tables. They give slightly different
        numbers, so they must never share a cache file.
    n_bins : int or None
        Number of energy bins (only relevant for method="histogram").
 
    Returns
    -------
    key : str
        16-character hexadecimal fingerprint.
    """
    
    params = {
        "version": CACHE_VERSION,
        "D": round(float(D), 12),   # rounding removes floating point noise
        # --- band structure ---
        "N": int(config.N),
        "gammas": [float(g) for g in (config.GAMMA0, config.GAMMA1,
                                      config.GAMMA2, config.GAMMA3,
                                      config.GAMMA4)],
        "valley_idx": [int(v) for v in config.VALLEY_IDX],
        "deltas": np.asarray(config.DELTAS, dtype=float).tolist(),
        # --- unit conversion (enters n and E_kin in cm^-2) ---
        "a": float(config.a),
        # --- thermal smearing ---
        "T_eff": float(config.T_eff),
        # --- k-mesh (read from the mesh itself) ---
        "n_pts": int(KX.shape[1]),
        "k_lim": round(float(np.max(KX)), 12),
        # --- table grid and method ---
        "mu_min": float(mu_min),
        "mu_max": float(mu_max),
        "n_table_pts": int(n_table_pts),
    }

    # 1) dict -> canonical text (sort_keys makes the order irrelevant)
    text = json.dumps(params, sort_keys=True)

    # 2) text -> bytes -> SHA-256 fingerprint -> first 16 hex characters
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def get_or_build_tables(D, config, KX, KY, cache_dir, mu_min, mu_max, n_table_pts, rebuild=False, verbose=True):
    """
    Return the tables (mu_table, n_table, e_table) for displacement field D, loading them from disk if a matching file exists, building and saving them otherwise.

    Flow
    ----
    1. Compute the cache key from all inputs that affect the tables.
    2. If  tables_D<...>_<key>.npz  exists (and rebuild is False): load it,
       check that the key stored inside matches, and return the arrays.
       If the file is unreadable or the key does not match, rebuild.
    3. Otherwise: build the bands, build the tables with the chosen method,
       save them (atomically, see below) and return them.
 
    Parameters
    ----------
    D : float
        Displacement field [eV].
    config : module
        Project config module.
    KX, KY : ndarray, shape (n_pts, n_pts)
        Cartesian k-mesh (from get_kmesh).
    cache_dir : str or pathlib.Path
        Folder where the .npz files live (created if missing).
    mu_min, mu_max : float
        Limits of the mu grid [eV]. Must cover mu_global - V_alpha for every
        point of the phase diagram you plan to compute.
    n_table_pts : int
        Number of mu points.
    rebuild : bool
        If True, ignore any existing file and overwrite it.
    verbose : bool
        Print whether the tables were loaded or built.
 
    Returns
    -------
    mu_table : ndarray, shape (n_table_pts,)
    n_table : ndarray, shape (4, n_table_pts)   [cm^-2]
    e_table : ndarray, shape (4, n_table_pts)   [eV cm^-2]
    """
    cache_dir = Path(cache_dir)

    # ---------- Name the computation ----------
    key = table_cache_key(
        D, config, KX, mu_min, mu_max, n_table_pts,
    )
    # Human-readable D in the filename (in meV) + hash for safety.
    path = cache_dir / f"tables_D{D * 1e3:09.4f}meV_{key}.npz"

    # ---------- Try to load ----------
    if path.exists() and not rebuild:
        try:
            with np.load(path) as d:            # 'with' closes the file for us
                stored_key = str(d["key"])
                mu_table = d["mu_table"]        # arrays are read from disk here
                n_table = d["n_table"]
                e_table = d["e_table"]
            if stored_key == key:
                if verbose:
                    print(f"  [cache] loaded  D = {D * 1e3:8.4f} meV  <- {path.name}")
                return mu_table, n_table, e_table
            print(f"  [cache] key mismatch inside {path.name}, rebuilding.")
        except Exception as exc:
            # Corrupted or incomplete file: do not crash, just rebuild it.
            print(f"  [cache] could not read {path.name} ({exc}), rebuilding.")

    # ---------- Build ----------
    if verbose:
        print(f"  [cache] building D = {D * 1e3:8.4f} meV")
 
    # k-space integration prefactor, derived from the mesh itself
    dk = KX[0, 1] - KX[0, 0]
    prefactor = (dk ** 2) / (2 * np.pi) ** 2

    systems, _ = build_systems_with_displacement(D, config)

    mu_table, n_table, e_table = build_tables_unique(
            systems, KX, KY, config.T_eff, prefactor, config.unit_cell_to_cm2,
            mu_min=mu_min, mu_max=mu_max, n_table_pts=n_table_pts,
        )
    del systems

    # ---------- Save (atomically) ----------
    # Write to a temporary file first, then rename. If the run is killed in the middle of writing, you never end up with a half-written file that looks valid. (np.savez_compressed adds ".npz" if the name lacks it, so the temporary name must also end in ".npz".)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_dir / f"{path.stem}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        key=key,
        D=float(D),
        mu_table=mu_table,
        n_table=n_table,
        e_table=e_table,
    )
    tmp_path.replace(path)   # rename = (almost) instantaneous and atomic

    return mu_table, n_table, e_table