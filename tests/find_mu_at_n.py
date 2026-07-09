import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import brentq
import gc

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, fermi_distrib, get_part_density
import model.config as config

def precompute_energy_bands(KX, KY):
    """
    Precomputes and stores energy bands for all 4 spin-valley flavors.
    """
    all_bands = []

    for v_idx, xi in enumerate(config.VALLEY_IDX):
        for s_idx in range(2):
            delta = config.DELTAS[v_idx, s_idx]
            e0 = config.E0_ARRAY[v_idx, s_idx]

            system = McCannCarts(
                N=config.N, valley_idx=xi, Delta=delta,
                gamma0=config.GAMMA0, gamma1=config.GAMMA1,
                gamma2=config.GAMMA2, gamma3=config.GAMMA3,
                gamma4=config.GAMMA4, E0=e0
            )

            # Get energies and sotre them
            E0, E1 = system.get_energy(KX, KY)
            all_bands.append((E0, E1))

    return all_bands

def find_mu(n_target, mu_min=-0.2, mu_max=0.2):
    """
    Finds the Fermi level mu (in eV) that yields the target particle density n_target.
    """
    # Setup kmesh and prefactor
    KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
    dk = KX[0, 1] - KX[0, 0]
    prefactor = (dk**2) / (2 * np.pi)**2

    # Conversion factor from density per unit cell to /cm^2
    unit_cell_to_cm2 = 1.0 / (config.a**2) * 1e16

    # Precompute the energy bands for all flavors
    print("Precomputing energy bands for all flavors...")
    all_flavs_bands = precompute_energy_bands(KX, KY)

    print("Bands ready. Starting root finder...")

    # Define the objective function: equals 0 when n_total = n_target
    def objective_function(mu):
        n_flav_list = []

        for E0, E1 in all_flavs_bands:
            n_e = fermi_distrib(E0, mu, config.T_eff)
            n_h = 1.0 - fermi_distrib(E1, mu, config.T_eff)

            n_flav = prefactor * np.sum(n_e - n_h)
            n_flav_list.append(n_flav)

        n_total = np.sum(n_flav_list) * unit_cell_to_cm2

        return n_total - n_target

    # Find the root -> mu
    try:
        mu_root = brentq(objective_function, mu_min, mu_max, xtol=1e-5)
        return mu_root
    except ValueError:
        print(f"Error: The target density {n_target} is outside the bracket [{mu_min}, {mu_max}].")
        raise

def main():
    n_target = -3e11        # in /cm^2

    # Bracket for the root finding (in eV)
    mu_min = -0.1
    mu_max = 0.1

    mu_solved = find_mu(n_target, mu_min=mu_min, mu_max=mu_max)
    print(f"\n   Target Density: {n_target:.4e} / cm2")
    print(f"   Calculated Fermi Level mu: {mu_solved * 1e3:.4f} meV")

if __name__ == "__main__":
    main()