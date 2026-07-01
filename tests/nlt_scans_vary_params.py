import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import model.config as config
from scripts.gao_scan_mu import scan_mu as scan_gao, save_results as save_gao
from scripts.holder_scan_mu import scan_mu as scan_holder, save_results as save_holder
from scripts.qmd_scan_mu import scan_mu as scan_qmd, save_results as save_qmd

def run_all_scans(suffix, n_mu=100):
    mu_lim = 0.5
    
    # GAO
    mu_vals, sigmas_sym, sigmas_asym = scan_gao(mu_lim=mu_lim, n_mu=n_mu)
    save_gao(mu_vals, sigmas_sym, sigmas_asym, filename=f"scan_nlt_gao_{suffix}.npz")
    
    # HOLDER
    mu_vals, sigmas_sym, sigmas_asym = scan_holder(mu_lim=mu_lim, n_mu=n_mu)
    save_holder(mu_vals, sigmas_sym, sigmas_asym, filename=f"scan_nlt_holder_{suffix}.npz")
    
    # QMD
    mu_vals, sigmas_sym, sigmas_asym = scan_qmd(mu_lim=mu_lim, n_mu=n_mu)
    save_qmd(mu_vals, sigmas_sym, sigmas_asym, filename=f"scan_nlt_qmd_{suffix}.npz")

def main():
    # Save original values
    orig_T_real = config.T_real
    orig_gamma4 = config.GAMMA4
    orig_gamma2 = config.GAMMA2
    orig_gamma3 = config.GAMMA3

    n_mu = 100 # Reduced from 500 to be faster for these tests

    # 1. Higher Temperatures
    print("\n--- RUNNING VARIATIONS: Temperature 150K ---")
    config.T_real = 150
    config.T_eff = config.kB * config.T_real
    run_all_scans("T150", n_mu=n_mu)

    print("\n--- RUNNING VARIATIONS: Temperature 300K ---")
    config.T_real = 300
    config.T_eff = config.kB * config.T_real
    run_all_scans("T300", n_mu=n_mu)
    
    # Reset T
    config.T_real = orig_T_real
    config.T_eff = config.kB * config.T_real

    # 2. Finite Gamma4
    print("\n--- RUNNING VARIATIONS: gamma4 = 0.005 ---")
    config.GAMMA4 = 0.005
    run_all_scans("g4_0005", n_mu=n_mu)

    print("\n--- RUNNING VARIATIONS: gamma4 = 0.05 ---")
    config.GAMMA4 = 0.05
    run_all_scans("g4_05", n_mu=n_mu)
    
    # Reset gamma4
    config.GAMMA4 = orig_gamma4

    # 3. Greater Trigonal Warping (gamma2 and gamma3)
    print("\n--- RUNNING VARIATIONS: Trigonal Warping (large) ---")
    # Let's double them
    config.GAMMA2 = -0.01
    config.GAMMA3 = 0.2
    run_all_scans("tw_large", n_mu=n_mu)
    
    # Reset all
    config.GAMMA2 = orig_gamma2
    config.GAMMA3 = orig_gamma3

    print("\nAll variations completed.")

if __name__ == "__main__":
    main()
