"""
NOTE: made by Claude.ai

Self-consistent Hartree-Fock (SCF) for the two-band model of rhombohedral graphene.

Interaction content
-------------------
 - Fock: intra-flavor Coulomb exchange, Vs on the layer-diagonal entries and Vd = Vs*exp(-q d) on the layer-off-diagonal entry (see hf.py). Evaluated with FFT convolutions.
 - Hartree: only the layer-polarization term Sigma_z^H = pi e^2 d / eps * Delta_n, one number shared by all flavors. Sigma_0^H is dropped (divergent constant absorbed by the background).
 - No contact term. Flavors are coupled only through Sigma_z^H and through the total-density constraint.

Effective Hamiltonian of flavor alpha on the k-mesh, basis (A1, BN) = (top, bottom):
    h0_eff = h0 + Sigma_0
    Delta_eff = Delta + Sigma_z + Sigma_z^H
    X_eff = X + Sigma_12                       (H = h0_eff*1 + [[Delta_eff, X_eff], [X_eff*, -Delta_eff]])

Self-consistency: Sigma_in -> bands -> mu (common to all flavors, fixes the total density)
                  -> rho_alpha -> Sigma_new = F[rho]  ->  linear mixing on Sigma.
Linear mixing on Sigma or on rho gives the same iterates (F is linear in rho); the convergence criterion
is the residual max|Sigma_new - Sigma_in| [eV] over all flavors, entries and k, and |Sigma_z^H| change.

Units: k in 1/a, energies in eV, densities in cm^-2, energy per area in eV/a^2
(multiply by config.unit_cell_to_cm2 for the project's eV cm^-2 convention; compare only relative values).

Functions
---------
build_hf_model
random_seed_shifts
solve_scf_hf
hf_energy
pauli_components
"""

import numpy as np
from scipy.optimize import brentq

from .kspace import build_systems_with_displacement
from .thermo import (
    get_bands_from_arrays,
    get_flavor_density_from_bands,
    density_matrix_from_arrays,
)
from .hf import (
    build_coulomb_kernels_fft,
    get_fock_self_energy_fft,
    get_hartree_sigma_z,
)

__all__ = [
    "build_hf_model",
    "random_seed_shifts",
    "solve_scf_hf",
    "hf_energy",
    "pauli_components",
]


def build_hf_model(D, KX, KY, config, reference="cnp", kernels=None):
    """
    Collect everything that does not change during the SCF for a given displacement field D and k-mesh.

    Parameters
    ----------
    D : float
        Displacement field [eV].
    KX, KY : ndarray (N, N)
        k-mesh in units of 1/a (from get_kmesh).
    config : module
        Project config (N, GAMMA0..4, DELTAS, VALLEY_IDX, T_eff, unit_cell_to_cm2, e2_over_eps, d_layers).
    reference : "cnp" | "valence" | "half"
        Fixed background subtracted from rho (see density_matrix_from_arrays).
        "cnp": P- with gap = intrinsic delta_alpha (no D);  "valence": P- with the bare gap delta_alpha + D.
        The projector always uses the BARE X(k), so the background does not change during the SCF.
    kernels : tuple (ks_fft, kd_fft, L) or None
        Precomputed Coulomb kernels for this mesh (they depend only on the mesh): pass them to reuse across D.

    Returns
    -------
    model : dict
    """
    systems, base_deltas = build_systems_with_displacement(D, config)
    n_f = len(systems)
    N = KX.shape[0]
    dk = KX[0, 1] - KX[0, 0]

    # Bare arrays; h0 may be a scalar when gamma4 = 0, so broadcast it to the mesh
    h0 = np.array([np.broadcast_to(s.get_h0_at_k(KX, KY), KX.shape) for s in systems], dtype=float)
    X = np.array([s.X_at_k(KX, KY) for s in systems], dtype=complex)
    Delta = np.array([s.Delta for s in systems], dtype=float)

    if reference == "half":
        ref_mode, Delta_ref = "half", np.zeros(n_f)
    elif reference == "cnp":
        ref_mode, Delta_ref = "projector", np.array(base_deltas, dtype=float)
    elif reference == "valence":
        ref_mode, Delta_ref = "projector", Delta.copy()
    else:
        raise ValueError(f"unknown reference '{reference}'")

    if kernels is None:
        kernels = build_coulomb_kernels_fft(N, dk, config.e2_over_eps, config.d_layers)

    return {
        "D": D, "n_flavors": n_f, "N": N, "dk": dk, "prefactor": dk**2 / (2 * np.pi) ** 2,
        "h0": h0, "X": X, "Delta": Delta,
        "ref_mode": ref_mode, "Delta_ref": Delta_ref,
        "kernels": kernels,
        "T_eff": config.T_eff, "unit_cell_to_cm2": config.unit_cell_to_cm2,
        "e2_over_eps": config.e2_over_eps, "d_layers": config.d_layers,
    }


def random_seed_shifts(n_flavors, amplitude=5e-3, rng=None):
    """
    Random per-flavor energy shifts [eV] used as the initial self-energy Sigma_alpha = eps_alpha * 1.
    This is the minimal way to break the flavor symmetry of the starting point.
    """
    _uniform = rng.uniform if rng is not None else np.random.uniform
    return _uniform(-amplitude, amplitude, n_flavors)


def pauli_components(S):
    """(M0, Mx, My, Mz) of a (2, 2, ...) array, with M = M0 + Mx sx + My sy + Mz sz."""
    return (0.5 * (S[0, 0] + S[1, 1]).real, S[0, 1].real, -S[0, 1].imag, 0.5 * (S[0, 0] - S[1, 1]).real)


def hf_energy(model, rhos, Sigma, SzH):
    """
    Hartree-Fock energy per area relative to the subtracted reference, at fixed density:
        E = sum_alpha int d^2k/(2pi)^2 { Tr[H0 rho] + 1/2 Tr[(Sigma^F + Sigma_z^H sigma_z) rho] }
    with rho = rho_rel and Sigma = F[rho] the self-energy generated by the SAME final rho (not the mixed one).
    Finite-temperature entropy is neglected. Returns eV/a^2; use it to compare solutions at the same
    (n, D), e.g. different seeds.
    """
    E = 0.0
    for a, rho in enumerate(rhos):
        h0, X, Dl = model["h0"][a], model["X"][a], model["Delta"][a]
        r11, r22, r12 = rho[0, 0].real, rho[1, 1].real, rho[0, 1]
        S = Sigma[a]
        kin = h0 * (r11 + r22) + Dl * (r11 - r22) + 2 * np.real(np.conj(X) * r12)
        # Tr[Sigma rho] = S11 r11 + S22 r22 + 2 Re(S12 conj(r12));  Hartree: SzH * (r11 - r22)
        inter = (S[0, 0].real * r11 + S[1, 1].real * r22
                 + 2 * np.real(S[0, 1] * np.conj(r12)) + SzH * (r11 - r22))
        E += model["prefactor"] * np.sum(kin + 0.5 * inter)
    return float(E)


def solve_scf_hf(model, n_target, *, Sigma_init=None, SzH_init=0.0, seed_shifts=None,
                 mu_min=-0.2, mu_max=0.2, max_iter=300, tolerance=1e-6, mixing=0.4, verbose=True):
    """
    Self-consistent Hartree-Fock for fixed total density n_target [cm^-2].

    Parameters
    ----------
    model : dict from build_hf_model.
    n_target : float
        Total carrier density summed over the flavors [cm^-2].
    Sigma_init : ndarray (n_flavors, 2, 2, N, N) or None
        Starting self-energy (e.g. the converged one of a neighbouring point: warm start). Default: zeros.
    SzH_init : float
        Starting Hartree term [eV].
    seed_shifts : array (n_flavors,) or None
        Per-flavor constant shifts [eV] added to Sigma_init (diagonal), to break the flavor symmetry.
    mu_min, mu_max : float
        Bracket of the common chemical potential [eV]; must contain mu of every iteration.
    tolerance : float
        Convergence threshold on the residual [eV].
    mixing : float
        Linear mixing parameter, Sigma <- (1 - mixing) Sigma + mixing F[rho].

    Returns
    -------
    dict with n_flavs (cm^-2), mu (eV), Sigma (= F[rho] of the final rho, (n_flavors, 2, 2, N, N)), SzH (eV), energy (eV/a^2), converged, n_iter, history (residual per iteration, eV)
    None if mu is not bracketed.
    """
    n_f, N = model["n_flavors"], model["N"]
    T, conv, pref = model["T_eff"], model["unit_cell_to_cm2"], model["prefactor"]
    ks_fft, kd_fft, L = model["kernels"]
    dk = model["dk"]

    Sigma = np.zeros((n_f, 2, 2, N, N), dtype=complex) if Sigma_init is None else np.array(Sigma_init, dtype=complex)
    if seed_shifts is not None:
        for a in range(n_f):
            Sigma[a, 0, 0] += seed_shifts[a]
            Sigma[a, 1, 1] += seed_shifts[a]
    SzH = float(SzH_init)

    converged, history = False, []
    for it in range(max_iter):
        # --- effective model from the current self-energy ---
        S0 = 0.5 * (Sigma[:, 0, 0] + Sigma[:, 1, 1]).real
        Sz = 0.5 * (Sigma[:, 0, 0] - Sigma[:, 1, 1]).real
        h0e = model["h0"] + S0
        De = model["Delta"][:, None, None] + Sz + SzH
        Xe = model["X"] + Sigma[:, 0, 1]
        bands = [get_bands_from_arrays(h0e[a], De[a], Xe[a]) for a in range(n_f)]

        # --- common chemical potential fixing the total density ---
        def total_density_residual(mu):
            return sum(get_flavor_density_from_bands(b, mu, T, pref, conv) for b in bands) - n_target

        try:
            mu = brentq(total_density_residual, mu_min, mu_max, xtol=1e-10)
        except ValueError:
            if verbose:
                print(f"Iteration {it}: n_target = {n_target:.3e} cm^-2 not bracketed by mu in [{mu_min}, {mu_max}] eV.")
            return None

        # --- density matrices, new self-energy ---
        rhos = [density_matrix_from_arrays(h0e[a], De[a], Xe[a], mu, T, reference=model["ref_mode"],
                                           Delta_ref=model["Delta_ref"][a], X_ref=model["X"][a])
                for a in range(n_f)]
        Sigma_new = np.empty_like(Sigma)
        for a in range(n_f):
            Sigma_new[a] = get_fock_self_energy_fft(rhos[a], ks_fft, kd_fft, L, dk)
        SzH_new = get_hartree_sigma_z(rhos, dk, model["e2_over_eps"], model["d_layers"])

        # --- residual (before mixing) and convergence ---
        res = max(np.max(np.abs(Sigma_new - Sigma)), abs(SzH_new - SzH))
        history.append(res)
        if verbose:
            n_now = [get_flavor_density_from_bands(b, mu, T, pref, conv) for b in bands]
            print(f"Iter {it:03d} | mu = {mu*1e3:8.3f} meV | res = {res*1e3:.3e} meV | "
                  f"SzH = {SzH_new*1e3:7.3f} meV | n_flavs = "
                  + np.array2string(np.array(n_now), formatter={'float': '{:.3e}'.format}))
        if res < tolerance:
            converged = True
            break

        # --- linear mixing ---
        Sigma = (1.0 - mixing) * Sigma + mixing * Sigma_new
        SzH = (1.0 - mixing) * SzH + mixing * SzH_new

    n_flavs = np.array([get_flavor_density_from_bands(b, mu, T, pref, conv) for b in bands])
    if verbose:
        print(f"---> {'Converged' if converged else 'NOT converged'} after {len(history)} iterations.")

    return {
        "n_flavs": n_flavs, "mu": mu, "Sigma": Sigma_new, "SzH": SzH_new,
        "energy": hf_energy(model, rhos, Sigma_new, SzH_new),
        "converged": converged, "n_iter": len(history), "history": np.array(history),
    }