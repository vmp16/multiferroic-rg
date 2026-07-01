import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, get_nlt_qmd, sym_decomp_cond
import model.config as config

def test_qmd_conductivity():
    # Initialize the system with parameters from the configuration file
    system = McCannCarts(
        gamma0=config.GAMMA0,
        gamma1=config.GAMMA1,
        valley_idx=config.XI,
        Delta=config.DELTA,
        N=config.N,
        gamma2=config.GAMMA2,
        gamma3=config.GAMMA3,
        gamma4=0
    )

    # Build the kmesh
    KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
    p = np.stack((KX, KY), axis=-1)

    # Select the band
    band_idx = 0

    # Calculate QMD contribution to conductivity for band 0
    print(f"Calculating QMD conductivity for band {band_idx}...")
    sigma_qmd = get_nlt_qmd(system, band_idx, p, config.T_eff, config.mu_eff)

    # Decompose in symmetry basis
    sigma_qmd_sym, sigma_qmd_asym = sym_decomp_cond(sigma_qmd)

    print(f"\nQMD Conductivity Tensor (Band {band_idx}):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                print(f"sigma_{i}{j}{l}: {sigma_qmd[i, j, l]:.5}")

    print(f"\nSymmetric Part (Band {band_idx}):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                print(f"sigma_sym_{i}{j}{l}: {sigma_qmd_sym[i, j, l]:.5}")

    print(f"\nAsymmetric Part (Band {band_idx}):")
    for i in range(2):
        for j in range(2):
            for l in range(2):
                print(f"sigma_asym_{i}{j}{l}: {sigma_qmd_asym[i, j, l]:.5}")

    # Check symmetries: sigma_ijl should be symmetric in j, l because g_jl is symmetric
    sym_diff = sigma_qmd[:, 0, 1] - sigma_qmd[:, 1, 0]
    print(f"\nSymmetry check (sigma_i01 - sigma_i10): {sym_diff}")
    
    assert np.allclose(sigma_qmd[:, 0, 1], sigma_qmd[:, 1, 0]), "Sigma is not symmetric in j, l"

if __name__ == "__main__":
    test_qmd_conductivity()
