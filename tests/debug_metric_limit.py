import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.model import McCannCarts
from model.analysis import get_quantum_metric, velocity_element

def check_metric_limit(N=1, delta=0.0):
    print(f"--- Checking limit for N={N}, Delta={delta} ---")
    
    # Parameters
    gamma0 = 3.16
    gamma1 = 0.38
    valley_idx = 1
    
    system = McCannCarts(N=N, valley_idx=valley_idx, Delta=delta, 
                         gamma0=gamma0, gamma1=gamma1)
    
    # K-points approaching zero along x-axis (ky=0) and along diagonal (kx=ky)
    ks = np.logspace(-6, -1, 10)
    
    print(f"{'k':>10} | {'g_xx (ky=0)':>15} | {'g_xx (kx=ky)':>15} | {'|Vx_pm|^2':>15} | {'dE^2':>15}")
    print("-" * 85)
    
    for k in ks:
        # Along kx (ky=0) -> phi=0
        p_x = np.array([k, 0.0])
        # Add extra dimensions for analysis functions (expecting mesh)
        p_x_mesh = p_x[np.newaxis, np.newaxis, :]
        
        g_x = get_quantum_metric(system, 0, p_x_mesh)[0, 0, 0, 0]
        
        # Along diagonal (kx=ky) -> phi=pi/4
        p_d = np.array([k/np.sqrt(2), k/np.sqrt(2)])
        p_d_mesh = p_d[np.newaxis, np.newaxis, :]
        
        g_d = get_quantum_metric(system, 0, p_d_mesh)[0, 0, 0, 0]
        
        # Calculate components for diagonal case
        Vx, Vy = velocity_element(system, p_d_mesh, 0, 1)
        V2 = np.abs(Vx[0, 0])**2
        
        energies = system.get_energy(p_d_mesh)
        dE2 = (energies[0, 0, 0] - energies[1, 0, 0])**2
        
        print(f"{k:10.2e} | {g_x:15.2e} | {g_d:15.2e} | {V2:15.2e} | {dE2:15.2e}")

if __name__ == "__main__":
    # Monolayer
    check_metric_limit(N=1, delta=0.0)
    # Bilayer
    check_metric_limit(N=2, delta=0.0)
    # Gapped monolayer
    check_metric_limit(N=1, delta=0.1)
