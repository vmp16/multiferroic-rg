import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, get_nlt_gao, sym_decomp_cond
import model.config as config

def test_gao():
    # Initialize the system with parameters from the configuration file
    system = McCannCarts(
        gamma0=config.GAMMA0,
        gamma1=config.GAMMA1,
        valley_idx=config.VALLEY_IDX[0],
        Delta=config.DELTAS[0, 0],
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

    # Calculate NLT Gao for band 0
    print(f"Calculating NLT Gao for band {band_idx} at mu=0.15...")
    sigma_gao_0 = get_nlt_gao(system, band_idx, p, config.T_eff, 0.15)

    # Decompose in symmetry basis
    sigma_gao_sym, sigma_gao_asym = sym_decomp_cond(sigma_gao_0)

    print(f"\nNLT Gao Tensor (Band {band_idx}):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                print(f"sigma_{i}{j}{l}: {sigma_gao_0[i, j, l]:.5}")

    print(f"\nSymmetric Part (Band {band_idx}):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                print(f"sigma_sym_{i}{j}{l}: {sigma_gao_sym[i, j, l]:.5}")

    print(f"\nAsymmetric Part (Band {band_idx}):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                print(f"sigma_asym_{i}{j}{l}: {sigma_gao_asym[i, j, l]:.5}")

    # Calculate NLT Gao for band 1
    # print("\nCalculating NLT Gao for band 1...")
    # sigma_gao_1 = get_nlt_gao(
    #     system, 
    #     band_idx=1, 
    #     k_lim=config.K_LIM, 
    #     n_pts=config.N_PTS, 
    #     T_eff=config.T_eff, 
    #     mu_eff=config.mu_eff
    # )

    # print("\nNLT Gao Tensor (Band 1):")
    # for i in range(2):
    #     for j in range(2):
    #         for l in range(2):
    #             print(f"sigma_{i}{j}{l}: {sigma_gao_1[i, j, l]}")

if __name__ == "__main__":
    test_gao()
