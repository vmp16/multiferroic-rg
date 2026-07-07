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
    Calculate the total particle density n_total per unit cell at a fixed Fermi level.
    """

    print(f"Calculating total Particle Density at mu={mu*1e3} meV")

    KX, KY = get_kmesh(config.K_LIM, config.N_PTS)    
    dk = KX[0, 1] - KX[0, 0]

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

            n_flav = get_part_density(system, KX, KY, config.T_eff, mu)
            n_flav_list.append(n_flav)

    n_total = np.sum(n_flav_list)

    return n_total

def main():
    mu = 0.0        # in eV
    n_total = calculate_total_part_dens(mu)
    print(f"   Net particle density for mu={mu*1e3} meV: {n_total:.5} / unit cell")

    # Calculate the surfacic particle density
    n_tot_surf = n_total / (config.a**2) * 1e16
    print(f"   Net particle density for mu={mu*1e3} meV: {n_tot_surf:.5} / cm2")

if __name__ == "__main__":
    main()