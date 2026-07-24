import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import deriv_fermi_distrib
import model.config as config

def calculate_total_dos(e_vals, chunk_size=1000):
    """
    Calculate the total DOS for a range of energy values by summing over all flavors.
    """
    print(f"Calculating DOS with chunking (N_PTS={config.N_PTS}, CHUNK={chunk_size})...")
    
    kx_vals = np.linspace(-config.K_LIM, config.K_LIM, config.N_PTS)
    ky_vals = np.linspace(-config.K_LIM, config.K_LIM, config.N_PTS)
    
    dk = kx_vals[1] - kx_vals[0]
    prefactor = (dk**2) / (2 * np.pi)

    total_dos = np.zeros_like(e_vals)
    
    for j_start in range(0, config.N_PTS, chunk_size):
        j_end = min(j_start + chunk_size, config.N_PTS)
        print(f"Processing chunk {j_start//chunk_size + 1}/{int(np.ceil(config.N_PTS/chunk_size))}...", end='\r')
        
        KX_c, KY_c = np.meshgrid(kx_vals, ky_vals[j_start:j_end])
        
        for v_idx, xi in enumerate(config.VALLEY_IDX):
            for s_idx in range(2):
                delta = config.DELTAS[v_idx, s_idx]
                e0 = config.E0_ARRAY[v_idx, s_idx]
                system = McCannCarts(
                    N=config.N, valley_idx=xi, Delta=delta,
                    gamma0=config.GAMMA0, gamma1=config.GAMMA1,
                    gamma2=config.GAMMA2, gamma3=config.GAMMA3,
                    gamma4=config.GAMMA4, E0=e0
                )
                E0, E1 = system.get_energy(KX_c, KY_c)
                
                for i, e in enumerate(e_vals):
                    df_dE0 = deriv_fermi_distrib(E0, e, config.T_eff)
                    df_dE1 = deriv_fermi_distrib(E1, e, config.T_eff)
                    total_dos[i] += np.sum(-df_dE0 - df_dE1) * prefactor
                    
        del KX_c, KY_c
        gc.collect()
        
    print("\nCalculation completed.")
    return total_dos

def main():
    # Range of energy in eV to scan
    # Increased resolution and range to see peaks at band edges (+/- 0.05, +/- 0.1)
    e_lim = 0.15
    n_pts_e = 600
    e_vals = np.linspace(-e_lim, e_lim, n_pts_e)
    
    dos_vals = calculate_total_dos(e_vals)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(e_vals*1e3, dos_vals, color='navy', lw=2)
    
    # Mark expected peak positions from config.py
    # Flavor 1 & 4: +/- 0.1
    # Flavor 2 & 3: +/- 0.05
    # for peak in [-0.1, -0.05, 0.05, 0.1]:
    #     plt.axvline(peak, color='green', linestyle=':', alpha=0.5, label='Expected Band Edge' if peak == 0.1 else "")

    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Energy (meV)', fontsize=16)
    plt.ylabel('Density of States (a. u.)', fontsize=16)
    # plt.title(f'Total Density of States (N={config.N} layers)', fontsize=16)
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    plt.xticks([-0.1, 0.0, 0.1])
    plt.tick_params(axis='x', labelsize=15)
    plt.yticks([])
    
    output_path = config.FIGURES_DIR / 'dos_plot.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    
    # Also save data for future use
    np.savez(config.DATA_DIR / 'dos_data.npz', e_vals=e_vals, dos_vals=dos_vals)
    print(f"Data saved to {data_dir / 'dos_data.npz'}")

if __name__ == "__main__":
    main()
