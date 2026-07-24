import sys
import numpy as np
from pathlib import Path
import gc

from model.model import McCannCarts
from model.analysis import fermi_distrib, get_QGT_chunk, get_bc_from_qgt
import model.config as config

def scan_U(U_lim=0.02, n_U=50, chunk_size=1000):
    """
    Scan the total Anomalous Hall conductivity as a function of the external
    interlayer potential U using chunking and scalar pipeline.
    """
    print(15*"=" + f" SCANNING TOTAL AHE VS U = [{-U_lim}, {U_lim}] (N_PTS={config.N_PTS}, T={config.T_real}K) " + 15*"=")

    U_vals = np.linspace(-U_lim, U_lim, n_U)
    kx_vals = np.linspace(-config.K_LIM, config.K_LIM, config.N_PTS)
    ky_vals = np.linspace(-config.K_LIM, config.K_LIM, config.N_PTS)
    dk = kx_vals[1] - kx_vals[0]
    
    sigmas_state1 = np.zeros(n_U)
    sigmas_state2 = np.zeros(n_U)

    e0_up_options = [config.E0_1UP, -config.E0_1UP]
    results = [sigmas_state1, sigmas_state2]

    for state_idx, e0_up in enumerate(e0_up_options):
        print(f"\nScanning Branch {state_idx+1} (E0_1UP = {e0_up:.3f})...", flush=True)
        
        # Re-build E0_ARRAY for this state
        e0_1up = e0_up
        e0_1dn = config.E0_1DN
        e0_2up = e0_1dn
        e0_2dn = -e0_up
        local_e0_array = np.array([[e0_1up, e0_1dn],
                                   [e0_2up, e0_2dn]])

        for i, U_val in enumerate(U_vals):
            print(f"\rProgress: {100 * (i + 1) / n_U:.1f}% (U={U_val:.4f})", end="", flush=True)
            
            total_sigma_xy = 0.0
            
            for j_start in range(0, config.N_PTS, chunk_size):
                j_end = min(j_start + chunk_size, config.N_PTS)
                KX_c, KY_c = np.meshgrid(kx_vals, ky_vals[j_start:j_end])
                
                for v_idx in range(2):
                    for s_idx in range(2):
                        xi = config.VALLEY_IDX[v_idx]
                        intrinsic_delta = config.DELTAS[v_idx, s_idx]
                        e0 = local_e0_array[v_idx, s_idx]
                        effective_delta = intrinsic_delta + U_val
                        
                        system = McCannCarts(
                            N=config.N, valley_idx=xi, Delta=effective_delta,
                            gamma0=config.GAMMA0, gamma1=config.GAMMA1,
                            gamma2=config.GAMMA2, gamma3=config.GAMMA3,
                            gamma4=config.GAMMA4, E0=e0
                        )
                        
                        # Calculate energies and Berry curvature once
                        E0, E1 = system.get_energy(KX_c, KY_c)
                        # Omega_0 = -Omega_1
                        T_chunk = get_QGT_chunk(system, 0, KX_c, KY_c)
                        Omega0 = get_bc_from_qgt(T_chunk)
                        
                        # Integration prefactors
                        dk_c = KX_c[0, 1] - KX_c[0, 0]
                        prefactor = (dk_c**2) / (2 * np.pi)

                        # Band 0 contribution
                        f0 = fermi_distrib(E0, config.mu_eff, config.T_eff)
                        total_sigma_xy += prefactor * np.sum(Omega0 * f0)
                        
                        # Band 1 contribution (Omega1 = -Omega0)
                        f1 = fermi_distrib(E1, config.mu_eff, config.T_eff)
                        total_sigma_xy -= prefactor * np.sum(Omega0 * f1)
                        
                        del E0, E1, Omega0, f0, f1, system
                
                del KX_c, KY_c
                gc.collect()
            
            results[state_idx][i] = total_sigma_xy


    print("\nScan completed.")
    return U_vals, sigmas_state1, sigmas_state2


def save_results(U_vals, sigmas_s1, sigmas_s2, filename="scan_ahe_vs_U.npz"):
    config.DATA_DIR.mkdir(exist_ok=True)
    
    save_path = data_dir / filename
    
    metadata = {
        "GAMMA0": config.GAMMA0,
        "mu_eff": config.mu_eff,
        "T_eff": config.T_eff,
        "N_PTS": config.N_PTS,
        "K_LIM": config.K_LIM,
    }

    np.savez(save_path, U_vals=U_vals, sigmas_s1=sigmas_s1, sigmas_s2=sigmas_s2, **metadata)
    print(f"Results saved to {save_path}")

def main():
    U_vals, sigmas_s1, sigmas_s2 = scan_U(U_lim=0.017, n_U=100)
    save_results(U_vals, sigmas_s1, sigmas_s2)

if __name__ == "__main__":
    main()
