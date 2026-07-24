import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Set math font to Computer Modern to properly render \varepsilon
plt.rcParams['mathtext.fontset'] = 'cm'

def plot_combined_scans(files_list):
    """
    Load and plot the Anomalous Hall conductivity vs Interlayer Potential U for two different values of the Fermi level mu.
    """
    n_scans = len(files_list)

    fig, axs = plt.subplots(1, n_scans, figsize=(7*n_scans, 6), sharex=True, sharey=True)

    for i, filename in enumerate(files_list):
        data_path = config.DATA_DIR / filename
        if not data_path.exists():
            print(f"Error: Data file {data_path} not found. Run the scan script first.")
            return
        
        # Load data
        data = np.load(data_path)
        U_vals = data['U_vals']
        sigmas_s1 = data['sigmas_s1']
        sigmas_s2 = data['sigmas_s2']
        mu_eff = data['mu_eff']

        print(f"Plotting data from {filename}...")

        # Create plot
        axs[i].plot(U_vals, sigmas_s1, 'o-', color='darkorange', linewidth=2, markersize=5, label=r'$\epsilon_0>0$')
        axs[i].plot(U_vals, sigmas_s2, 's-', color='forestgreen', linewidth=2, markersize=5, label=r'$\epsilon_0<0$')

        # Add a text box with the mu_eff value at the top center
        text_str = rf'$\mu = {mu_eff:.2f}$ eV'
        axs[i].text(0.04, 0.95, text_str, transform=axs[i].transAxes, fontsize=17,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
         # Baseline for zero
        # plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axs[i].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Styling
        axs[i].set_xlabel(r'Interlayer Potential $U$ (eV)', fontsize=17)
        axs[i].set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
        axs[i].tick_params(axis='x', labelsize=16)

    
    axs[0].set_ylabel(r'$\sigma_{xy}$ ($e^2/h$)', fontsize=17)
    # axs[0].set_yticks([-0.3, 0.0, 0.3])
    axs[0].tick_params(axis='y', labelsize=16)
    axs[0].legend(fontsize=18)
    plt.tight_layout()

    # Save figure
    config.DATA_DIR.mkdir(exist_ok=True)
    plot_filename = "plot_combined_scans_ahe.png"
    plot_path = config.DATA_DIR / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Success! Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_combined_scans(["scan_ahe_vs_U_mu0.npz", "scan_ahe_vs_U_mu_minus_001.npz"])