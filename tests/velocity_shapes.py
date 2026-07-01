import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, velocity_element
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

    band_idx = 0
    n_idx = 1 - band_idx

    # Get the velocities
    Vx_01, Vy_01 = velocity_element(system, p, band_idx, n_idx)
    Vx_10, Vy_10 = velocity_element(system, p, n_idx, band_idx)

    # Check if Vy_01 = np.conj(Vy_10)
    check_vx = np.allclose(Vx_01, np.conj(Vx_10))
    check_vy = np.allclose(Vy_01, np.conj(Vy_10))

    print(f"Vx_01 shape: {Vx_01.shape}")
    print(f"Is Vx_01 equal to conj(Vx_10)? {check_vx}")
    print(f"Is Vy_01 equal to conj(Vy_10)? {check_vy}")



if __name__ == "__main__":
    main()