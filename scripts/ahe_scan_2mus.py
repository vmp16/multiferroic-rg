import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from scripts.ahe_scan_U import scan_U, save_results
import model.config as config

def run_task1():
    # Common parameters
    config.T_real = 4
    config.T_eff = config.kB * config.T_real
    config.N_PTS = 6000
    U_lim = 0.1
    n_U = 50 # Standard number of points for U scan

    # Run 1: mu = 0
    print("\n" + 20*"-" + " RUNNING TASK 1.1: mu = 0 " + 20*"-")
    config.mu_eff = 0.0
    U_vals, s1, s2 = scan_U(U_lim=U_lim, n_U=n_U)
    save_results(U_vals, s1, s2, filename="scan_ahe_vs_U_mu0.npz")

    # Run 2: mu = -0.01
    print("\n" + 20*"-" + " RUNNING TASK 1.2: mu = -0.01 " + 20*"-")
    config.mu_eff = -0.01
    U_vals, s1, s2 = scan_U(U_lim=U_lim, n_U=n_U)
    save_results(U_vals, s1, s2, filename="scan_ahe_vs_U_mu_minus_001.npz")

if __name__ == "__main__":
    run_task1()
