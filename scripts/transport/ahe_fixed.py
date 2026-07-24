import numpy as np
import sys
from pathlib import Path

from model.model import McCannCarts
from model.analysis import get_kmesh, get_ahe
import model.config as config

def calculate_total_ahe():
    """
    Calculate the total Anomalous Hall Effect (AHE) conductivity
    by summing over all spin-valley flavors (2 valleys x 2 spins).
    """
    print("Initializing k-mesh...")
    # Use configuration parameters for the mesh
    KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
    p = np.stack((KX, KY), axis=-1)

    total_sigma_xy = 0.0
    flavors = []

    print(f"{'Valley':<8} | {'Spin':<6} | {'Delta':<8} | {'E0':<8} | {'sigma_xy':<12}")
    print("-" * 55)

    # Iterate over valley and spin indices
    for v_idx in range(2):      # Valley K (0) and K' (1)
        for s_idx in range(2):  # Spin Up (0) and Down (1)
            
            # Initialize the system for this specific flavor
            system = McCannCarts(
                gamma0=config.GAMMA0,
                gamma1=config.GAMMA1,
                valley_idx=config.VALLEY_IDX[v_idx],
                Delta=config.DELTAS[v_idx, s_idx],
                N=config.N,
                gamma2=config.GAMMA2,
                gamma3=config.GAMMA3,
                gamma4=config.GAMMA4,
                E0=config.E0_ARRAY[v_idx, s_idx]
            )

            # Calculate AHE for both bands (0 and 1) and sum them
            # Usually only one band contributes significantly near the Fermi level,
            # but we sum both for completeness.
            sigma_xy_flavor = 0.0
            for band_idx in [0, 1]:
                sigma_xy_flavor += get_ahe(system, band_idx, p, config.T_eff, config.mu_eff)
            
            total_sigma_xy += sigma_xy_flavor
            
            valley_name = "K" if config.VALLEY_IDX[v_idx] == 1 else "K'"
            spin_name = "Up" if s_idx == 0 else "Down"
            print(f"{valley_name:<8} | {spin_name:<6} | {config.DELTAS[v_idx, s_idx]:<8.3f} | {config.E0_ARRAY[v_idx, s_idx]:<8.3f} | {sigma_xy_flavor:<12.6e}")

    print("-" * 55)
    print(f"Total AHE Conductivity: {total_sigma_xy:.6e} [e^2/h]")

if __name__ == "__main__":
    calculate_total_ahe()
