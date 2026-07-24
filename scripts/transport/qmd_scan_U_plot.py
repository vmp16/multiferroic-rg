import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from model.analysis import sym_decomp_cond

# Set math font to Computer Modern
plt.rcParams['mathtext.fontset'] = 'cm'

def plot_qmd_vs_U(filename='qmd_scan_U.npz'):
    """
    Load and plot the QMD's Conductivity vs Interlayer Potential U.
    """
    data_path = config.DATA_DIR / filename
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found. Run the scan script first.")

    # Load data
    data = np.load(data_path)
    U_vals = data['U_vals']
    sigmas = data['sigmas_vals']
    mu_vals = data['mu_vals']
    n_fixed = data['n_fixed']

    # print("U values shape:", U_vals.shape)
    # print("Sigma_xxx shape:", sigmas[:, 0, 0, 0].shape)
    # print("Min sigma_xxx:", np.min(sigmas[:, 0, 0, 0]))
    # print("Max sigma_xxx:", np.max(sigmas[:, 0, 0, 0]))
    # print("Are there NaNs?:", np.isnan(sigmas[:, 0, 0, 0]).any())

    print(f"Plotting data from {filename}...")

    # Build the plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)


    # Baseline for zero
    # plt.axhline(0, color='gray', ls=':', lw=1, alpha=0.7)
    axs[0].axvline(0, color='gray', ls=':', lw=1, alpha=0.7)

    # Plot the conductivity
    axs[0].plot(U_vals*1e3, sigmas[:, 0, 0, 0], 'o-', color='teal', linewidth=1.5, markersize=5)

    axs[1].plot(U_vals*1e3, mu_vals*1e3, 'o-', color='black', markersize=2, linewidth=1, label=f"{n_fixed:.2e} /cm2")

    # Styling
    for ax in axs:
        ax.set_xlabel(r'$U~(meV)$', fontsize=18)
        ax.tick_params(axis='both', labelsize=17)

    axs[0].set_ylabel(r'$\sigma_{xxx}$ (($e^2/h)(a/V)$)', fontsize=18)
    axs[1].legend(fontsize=17)
    axs[1].set_ylabel(r'$\mu~(meV)$', fontsize=18)

    plt.tight_layout()

    # Save figure
    config.FIGURES_DIR.mkdir(exist_ok=True)
    plot_filename = f"plot_{Path(filename).stem}.png"
    plot_path = config.FIGURES_DIR / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    print(f"   Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_qmd_vs_U(filename="qmd_scan_U.npz")