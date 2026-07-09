import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, get_part_density
import model.config as config

def calculate_total_part_dens(mu):
    """
    Calculate the total particle density n_total at a fixed Fermi level.
    Returns: n_total (float), net particle density / cm^2
    """

    # Build the kmesh
    KX, KY = get_kmesh(config.K_LIM, config.N_PTS)    
    dk = KX[0, 1] - KX[0, 0]

    # Calculate the particle density per flavor
    n_flav_list = []
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

            # Get the particle density per unit cell
            n_flav = get_part_density(system, KX, KY, config.T_eff, mu)
            n_flav_list.append(n_flav)

    # Sum all the flavors and convert to /cm^2
    n_total = np.sum(n_flav_list) / (config.a**2) * 1e16

    return n_total

def main():
    mu = 0.0        # in eV
    print(f"Calculating total Particle Density at mu={mu*1e3} meV")
    n_total = calculate_total_part_dens(mu)

    print(f"   Net particle density for mu={mu*1e3} meV: {n_total:.5} / cm2")

if __name__ == "__main__":
    main()