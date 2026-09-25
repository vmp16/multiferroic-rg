"""
Scan over all possible phase configurations (1, 2, 3 or 4 occupied flavors) for a given total density n_target and displacement field D, obtaining the self-consistent equilibrium solution for each phase.

For each equilibrium configuration, the total internal energy is computed to get the ground state.
"""

import numpy as np
import time

from model.model import McCannCarts
import model.config as config
from model.analysis import (
    get_kmesh,
    get_flavor_kinetic_energy_from_bands,
    V_int_SU4,
    solve_phase_mean_field,
)

def perturb_symmetric_seed(n_target, n_active, amplitude, rng=None):
    """
    Generate a single initial density configuration slightly off the
    symmetric point, with the total density constraint exactly satisfied.

    Each flavor gets n_target/n_active plus a random perturbation drawn
    uniformly from [-amplitude, +amplitude] * |n_target/n_active|,
    then the whole array is rescaled to enforce the sum = n_target exactly.

    Parameters
    ----------
    n_target : float
        Total carrier density [cm^-2].
    n_active : int
        Number of active flavors.
    amplitude : float
        Fractional size of the perturbation (e.g. 0.1 = ±10%).
    rng : numpy.random.Generator or None
        Optional seeded RNG for reproducibility.

    Returns
    -------
    n_init : ndarray, shape (n_active,)
        Perturbed initial densities summing exactly to n_target.
    """
    _uniform = rng.uniform if rng is not None else np.random.uniform
    n_sym = n_target / n_active
    raw   = n_sym + _uniform(-amplitude, amplitude, size=n_active) * abs(n_sym)
    return raw * (n_target / raw.sum())

# -------------- Parameters --------------
n_target = -3e11        # total density [cm^-2]
D = 0.0                 # displacement field [eV]

m_seed = 0.5

# Self-consistent params
MU_MIN = -0.1           # [eV]
MU_MAX = 0.1            # [eV]
TOLERANCE = 1e-6
MAX_ITER = 200
MIXING = 0.3

rng = np.random.default_rng(62)

# -------------- Grid --------------
KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
dk = KX[0, 1] - KX[0, 0]
prefactor = (dk**2) / (2 * np.pi)**2

# -------------- Phase Loop --------------
phases_to_run = [4]    # all occupation numbers
results = {}                    # key = n_active

print(f"\nRunning SCF for n_target = {n_target:.3e} cm^-2, D = {D*1e3:.2f} meV")
print(f"Phases to evaluate: {phases_to_run}\n")

t_start = time.time()

for n_active in phases_to_run:
    t0 = time.time()

    # initial_n_flavs = np.random.uniform(-2, 2, n_active) * n_target
    # print(f"Initial guesses: {initial_n_flavs}")

    # initial_n_flavs = perturb_symmetric_seed(n_target, n_active, amplitude=2.5, rng=rng)

    initial_n_flavs = np.array([
        (1 + m_seed) * n_target,
        (1 + m_seed) * n_target,
        (1 - m_seed) * n_target,
        (1 - m_seed) * n_target
    ])

    result = solve_phase_mean_field(
        n_active=n_active,
        n_target=n_target,
        D=D,
        KX=KX,
        KY=KY,
        config=config,
        initial_n_flavs=initial_n_flavs,
        mu_min=MU_MIN,
        mu_max=MU_MAX,
        max_iter=MAX_ITER,
        tolerance=TOLERANCE,
        mixing=MIXING,
        verbose=True,
    )

    t1 = time.time()
    result['wall_time'] = t1 - t0
    results[n_active] = result

# -------------- Internal Energy --------------
print("\n" + "="*60)
print("  POST-PROCESSING: total internal energies")
print("="*60)

energies = {}
for n_active, res in results.items():
    all_bands = res["all_bands"]
    n_flavs = res["n_flavs"]
    mu_flavs = res["mu_flavs"]
    prefactor = res["prefactor"]

    # Get total kinetic energy
    E_kin = 0.0
    for i in range(n_active):
        E_kin += get_flavor_kinetic_energy_from_bands(
            all_bands[i],
            mu_flavs[i],
            config.T_eff,
            prefactor,
            config.unit_cell_to_cm2,
        )

    # Interaction energy SU(4) symmetric
    V_int = V_int_SU4(n_flavs, config.U, config.area_uc)

    # Get total internal energy
    E_int = E_kin + V_int
    energies[n_active] = {"E_kin": E_kin, "V_int": V_int, "E_int": E_int}

# -------------- Summary Table --------------
# Sort phases by total internal energy (ascending)
sorted_phases = sorted(energies, key=lambda k: energies[k]["E_int"])
E_gs          = energies[sorted_phases[0]]["E_int"]   # ground-state energy
 
print(f"\n{'Phase':>8} {'Converged':>10} {'E_kin':>18} {'V_int':>18} {'E_int':>18} {'ΔE':>18} {'Time(s)':>9}")
print("-" * 105)
 
for n_active in phases_to_run:
    res  = results[n_active]
    enrg = energies[n_active]
    dE   = enrg["E_int"] - E_gs
 
    print(
        f"{n_active:>8d} "
        f"{'YES' if res['converged'] else 'NO':>10} "
        f"{enrg['E_kin']:>18.6e} "
        f"{enrg['V_int']:>18.6e} "
        f"{enrg['E_int']:>18.6e} "
        f"{dE:>18.6e} "
        f"{res['wall_time']:>9.2f}"
    )
 
print("-" * 105)
gs = sorted_phases[0]
print(f"\n  Ground-state phase: {gs} occupied flavor(s)")
print(f"  Ground-state energy: {E_gs:.6e} eV·cm⁻²")