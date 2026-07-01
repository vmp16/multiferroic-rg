import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, get_nlt_holder, sym_decomp_cond
import model.config as config

def test_holder():
    # Initialize the system with parameters from the configuration file
    system = McCannCarts(
        gamma0=config.GAMMA0,
        gamma1=config.GAMMA1,
        valley_idx=config.XI,
        Delta=config.DELTA,
        N=config.N,
        gamma2=config.GAMMA2,
        gamma3=config.GAMMA3,
        gamma4=0 # config.GAMMA4
    )

    # Build the kmesh
    KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
    p = np.stack((KX, KY), axis=-1)

    # Select the band
    band_idx = 0

    # Calculate NLT Holder
    print(f"Calculating NLT Holder for band {band_idx}...")
    sigma_holder = get_nlt_holder(system, band_idx, p, config.T_eff, config.mu_eff)

    # Decompose in symmetry basis
    sigma_holder_sym, sigma_holder_asym = sym_decomp_cond(sigma_holder)

    print(f"\nNLT Holder Tensor (Band {band_idx}):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                print(f"sigma_{i}{j}{l}: {sigma_holder[i, j, l]:.5}")
                
    print(f"\nSymmetric Part (Band {band_idx}):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                print(f"sigma_sym_{i}{j}{l}: {sigma_holder_sym[i, j, l]:.5}")

    print(f"\nAsymmetric Part (Band {band_idx}):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                print(f"sigma_asym_{i}{j}{l}: {sigma_holder_asym[i, j, l]:.5}")
                

    # Calculate NLT Holder for band 1
    # print("\nCalculating NLT Holder for band 1...")
    # sigma_holder_1 = get_nlt_holder(
    #     system, 
    #     band_idx=1, 
    #     k_lim=config.K_LIM, 
    #     n_pts=config.N_PTS, 
    #     T_eff=config.T_eff, 
    #     mu_eff=config.mu_eff
    # )

    # print("\nNLT Holder Tensor (Band 1):")
    # for i in range(2):
    #     for j in range(2):
    #         for l in range(2):
    #             print(f"sigma_{i}{j}{l}: {sigma_holder_1[i, j, l]}")

if __name__ == "__main__":
    test_holder()
