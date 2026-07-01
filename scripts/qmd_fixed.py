import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, get_nlt_qmd, sym_decomp_cond
import model.config as config

def get_conductivity_qmd(mu):
    # Build the kmesh
    KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
    p = np.stack((KX, KY), axis=-1)

    sigma_list = []

    for xi, delta_K, E0_K in zip(config.VALLEY_IDX, config.DELTAS, config.E0_ARRAY):

        # Build one system per spin
        system_up = McCannCarts(
            N=config.N,
            valley_idx=xi,
            Delta=delta_K[0],
            gamma0=config.GAMMA0,
            gamma1=config.GAMMA1,
            gamma2=config.GAMMA2,
            gamma3=config.GAMMA3,
            gamma4=config.GAMMA4,
            E0=E0_K[0]
        )
        system_dn = McCannCarts(
            N=config.N,
            valley_idx=xi,
            Delta=delta_K[1],
            gamma0=config.GAMMA0,
            gamma1=config.GAMMA1,
            gamma2=config.GAMMA2,
            gamma3=config.GAMMA3,
            gamma4=config.GAMMA4,
            E0=E0_K[1]
        )

        # Calculate the conductivity for each band
        sigma_up = []
        sigma_dn = []
        for band_idx in range(2):
            sigma_up.append(get_nlt_qmd(system_up, band_idx, p, config.T_eff, mu))
            sigma_dn.append(get_nlt_qmd(system_dn, band_idx, p, config.T_eff, mu))
            
        # Sum contributions from all bands for each spin
        sigma_up_total = np.sum(sigma_up, axis=0)
        sigma_dn_total = np.sum(sigma_dn, axis=0)

        # Store the total conductivity for this valley (sum of both spins)
        sigma_list.append(sigma_up_total + sigma_dn_total)

    # Calculate the total conductivity by summing over valleys
    sigma_total = np.sum(sigma_list, axis=0)

    # Decompose in symmetry basis
    sigma_tot_sym, sigma_tot_asym = sym_decomp_cond(sigma_total)

    return sigma_tot_sym, sigma_tot_asym

def main():
    print(10*'=' + " CALCULATING THE CONDUCTIVITY FROM THE QUANTUM METRIC DIPOLE " + 10*'=')

    sigma_tot_sym, sigma_tot_asym = get_conductivity_qmd(config.mu_eff)

    print("\n" + 20*'=' + " FINAL RESULTS QMD " + 20*'=')
    print(f"Longitudinal Non-Linear Conductivity Tensor (QMD):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                print(f"sigma_{i}{j}{l}: {sigma_tot_sym[i, j, l]:.5}")

    print(f"\nTransverse Non-Linear Conductivity Tensor (QMD):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                print(f"sigma_{i}{j}{l}: {sigma_tot_asym[i, j, l]:.5}")


if __name__ == "__main__":
    main()
