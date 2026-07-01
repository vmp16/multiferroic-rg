import sys
import os
import numpy as np

# Add the project root to sys.path to allow importing from the model package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import McCannCarts
from model.analysis import get_QG_tensor, get_quantum_metric, get_Berry_curv
import model.config as config

def test_shapes():
    # Initialize the system with parameters from config
    system = McCannCarts(
        gamma0=config.GAMMA0,
        gamma1=config.GAMMA1,
        valley_idx=config.XI,
        Delta=config.DELTA,
        N=config.N,
        gamma2=config.GAMMA2,
        gamma3=config.GAMMA3,
        gamma4=config.GAMMA4
    )

    n_pts = config.N_PTS
    k_lim = config.K_LIM
    band_idx = 0
    
    expected_shape = (2, 2, n_pts, n_pts)
    
    print(f"Testing shapes with N_PTS = {n_pts}...")
    
    # Check QG Tensor shape
    qg_tensor = get_QG_tensor(system, band_idx, k_lim, n_pts)
    print(f"QG Tensor shape: {qg_tensor.shape}")
    assert qg_tensor.shape == expected_shape, f"Expected {expected_shape}, got {qg_tensor.shape}"
    
    # Check Quantum Metric shape
    g_tensor = get_quantum_metric(system, band_idx, k_lim, n_pts)
    print(f"Quantum Metric shape: {g_tensor.shape}")
    assert g_tensor.shape == expected_shape, f"Expected {expected_shape}, got {g_tensor.shape}"
    
    # Check Berry Curvature shape
    omega_tensor = get_Berry_curv(system, band_idx, k_lim, n_pts)
    print(f"Berry Curvature shape: {omega_tensor.shape}")
    assert omega_tensor.shape == expected_shape, f"Expected {expected_shape}, got {omega_tensor.shape}"
    
    print("\nAll shapes are correct!")

if __name__ == "__main__":
    test_shapes()
