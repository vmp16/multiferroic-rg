import sys
import numpy as np
import matplotlib.pyplot as plt
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

    # Get eigenstates
    psi_p, psi_m = system.get_eigenstates(p)
    # psi shape = (Coord, Ny, Nx) = (2, Ny, Nx)

    # Stack the eigenstates
    U = np.array((psi_p, psi_m))
    U_dag = np.conj(np.transpose(U, (1, 0, 2, 3)))
    # U shape = (State, Coord, Ny, Nx)
    # U shape = (Coord, State, Ny, Nx)

    # Print shapes and sizes
    print(f"Input k-mesh shape: {KX.shape}")
    print(f"Eigenstate psi_p shape: {psi_p.shape}")
    print(f"Eigenstate psi_m shape: {psi_m.shape}")
    # print(f"Total elements in psi_p: {psi_p.size}")
    print(f"U shape: {U.shape}")
    print(f"U_dag shape: {U_dag.shape}")

if __name__ == "__main__":
    main()