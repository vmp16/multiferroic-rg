import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model.model import McCannCarts
from model.analysis import get_kmesh, get_dos
import model.config as config
from dos_plot import calculate_total_dos

def main():
    # Define data path
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    data_path = data_dir / 'bands_dos_data.npz'

    # Check if data exists to avoid re-calculation
    if data_path.exists():
        print(f"Loading pre-calculated data from {data_path}...")
        data = np.load(data_path)
        kx_1d = data['kx_1d']
        all_energies = [
            ((data['energies_K_up_b0'], data['energies_K_up_b1']), 
             (data['energies_K_dn_b0'], data['energies_K_dn_b1'])),
            ((data['energies_Kp_up_b0'], data['energies_Kp_up_b1']), 
             (data['energies_Kp_dn_b0'], data['energies_Kp_dn_b1']))
        ]
        e_vals = data['e_vals']
        dos_vals = data['dos_vals']
    else:
        print("No pre-calculated data found. Running calculations...")
        kx_1d = np.linspace(-config.K_LIM, config.K_LIM, 300)
        ky_1d = np.zeros_like(kx_1d)
        
        all_energies = []
        for i, (xi, delta_K, E0_K) in enumerate(zip(config.VALLEY_IDX, config.DELTAS, config.E0_ARRAY)):
            system_up = McCannCarts(
                N=config.N, valley_idx=xi, Delta=delta_K[0],
                gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
                gamma3=config.GAMMA3, gamma4=config.GAMMA4, E0=E0_K[0]
            )
            system_dn = McCannCarts(
                N=config.N, valley_idx=xi, Delta=delta_K[1],
                gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
                gamma3=config.GAMMA3, gamma4=config.GAMMA4, E0=E0_K[1]
            )

            # Get the energies
            energies_up = system_up.get_energy(kx_1d, ky_1d)
            energies_dn = system_dn.get_energy(kx_1d, ky_1d)
            all_energies.append((energies_up, energies_dn))

        # Calculate DOS
        print("Calculating DOS...")
        e_lim = 0.018
        e_vals = np.linspace(-e_lim, e_lim, 300)
        dos_vals = calculate_total_dos(e_vals)
        
        # Save the calculated data
        print(f"Saving calculated data to {data_path}...")
        np.savez(
            data_path,
            kx_1d=kx_1d,
            energies_K_up_b0=all_energies[0][0][0], energies_K_up_b1=all_energies[0][0][1],
            energies_K_dn_b0=all_energies[0][1][0], energies_K_dn_b1=all_energies[0][1][1],
            energies_Kp_up_b0=all_energies[1][0][0], energies_Kp_up_b1=all_energies[1][0][1],
            energies_Kp_dn_b0=all_energies[1][1][0], energies_Kp_dn_b1=all_energies[1][1][1],
            e_vals=e_vals,
            dos_vals=dos_vals
        )
    
    # --- Plotting ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    for i, (xi, (energies_up, energies_dn)) in enumerate(zip(config.VALLEY_IDX, all_energies)):
        # Plot the energies
        axs[i].plot(kx_1d, np.array(energies_up).T*1e3, color='red')
        axs[i].plot(kx_1d, np.array(energies_dn).T*1e3, color='blue', linestyle='--')
        
        valley_label = "K" if xi == 1 else "K'"
        axs[i].set_title(f"Valley {valley_label}", fontsize=17)
        axs[i].set_xlabel(r'$k_x a$', fontsize=17)
        axs[i].axhline(0, color='gray', linestyle=':', alpha=0.5)
        axs[i].set_ylim(-18, 18)
        axs[i].set_xlim(-0.14, 0.14)
        axs[i].tick_params(axis='both', labelsize=16)
        if i == 0:
            from matplotlib.lines import Line2D
            legend_lines = [Line2D([0], [0], color='red'), Line2D([0], [0], color='blue', linestyle='--')]
            axs[i].legend(legend_lines, [r'$\uparrow$', r'$\downarrow$'], fontsize=16, loc='upper left') # loc='lower right'

    axs[0].set_ylabel('Energy (meV)', fontsize=17)

    # Plot DOS
    axs[2].plot(dos_vals, e_vals*1e3, color='navy', lw=2)
    axs[2].set_title("Total Density of States", fontsize=17)
    axs[2].set_xlabel("DOS (a.u.)", fontsize=17)
    axs[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[2].tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    output_path = config.FIGURES_DIR / 'bands_dos_combined.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()