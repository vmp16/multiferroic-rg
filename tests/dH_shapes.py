import sys
import numpy as np
# import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh
from model.config import GAMMA0, GAMMA1, XI, N, DELTA, N_PTS, K_LIM

def main():
    # Build the system using McCannCarts and the imported parameters
    system = McCannCarts(
        gamma0=GAMMA0, 
        gamma1=GAMMA1, 
        valley_idx=XI, 
        Delta=DELTA, 
        N=N
    )

    # Create a k-mesh
    KX, KY = get_kmesh(K_LIM, N_PTS)
    p = np.stack((KX, KY), axis=-1)

    dX_dkx, dX_dky = system.derivate_X_at_k(p)
    # shape = (Ny, Nx)
    dH_dkx, dH_dky = system.derivate_hamiltonian(p)
    # shape = (2, 2, Ny, Nx) = (2x2 H matrix, Ny, Nx)

    print(f"dX_dkx shape: {dX_dkx.shape}")
    print(f"dX_dky shape: {dX_dky.shape}")
    print(f"dH_dkx shape: {dH_dkx.shape}")
    print(f"dH_dky shape: {dH_dky.shape}")

if __name__ == "__main__":
    main()