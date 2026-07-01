import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, get_dos, deriv_fermi_distrib
import model.config as config

def diagnose():
    # Build k-mesh
    KX, KY = get_kmesh(config.K_LIM, 500)
    p = np.stack((KX, KY), axis=-1)
    
    # Check flavor 1 (Delta=0.1, E0=0)
    v_idx, s_idx = 0, 0
    delta = config.DELTAS[v_idx, s_idx]
    e0 = config.E0_ARRAY[v_idx, s_idx]
    
    system = McCannCarts(
        N=config.N,
        valley_idx=config.VALLEY_IDX[v_idx],
        Delta=delta,
        gamma0=config.GAMMA0,
        gamma1=config.GAMMA1,
        gamma2=config.GAMMA2,
        gamma3=config.GAMMA3,
        gamma4=config.GAMMA4,
        E0=e0
    )
    
    energies = system.get_energy(p)
    print(f"Flavor 1: Delta={delta}, E0={e0}")
    print(f"Min Energy Band 0: {np.min(energies[0])}")
    print(f"Max Energy Band 0: {np.max(energies[0])}")
    print(f"Min Energy Band 1: {np.min(energies[1])}")
    print(f"Max Energy Band 1: {np.max(energies[1])}")
    
    # Check config mu_eff and T_eff
    print(f"Config GAMMA0: {config.GAMMA0}")
    print(f"Config T_eff: {config.T_eff}")
    print(f"kB * T_real / GAMMA0: {(8.617e-5 * config.T_real) / config.GAMMA0}")
    
    # Probing DOS at E=0.1 eV (band edge)
    e_probe = 0.1
    mu_eff_norm = e_probe / config.GAMMA0
    dos_norm = get_dos(system, p, config.T_eff, mu_eff_norm)
    
    mu_eff_raw = e_probe
    T_eff_raw = 8.617e-5 * config.T_real
    dos_raw = get_dos(system, p, T_eff_raw, mu_eff_raw)
    
    print(f"DOS at 0.1 eV with normalized mu/T: {dos_norm}")
    print(f"DOS at 0.1 eV with raw eV mu/T: {dos_raw}")

if __name__ == "__main__":
    diagnose()
