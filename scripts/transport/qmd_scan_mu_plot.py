import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_qmd():
    data_path = config.DATA_DIR / "scan_nlt_qmd.npz"
    config.FIGURES_DIR.mkdir(exist_ok=True)
    
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run scan_nlt_qmd.py first.")
        return

    data = np.load(data_path)
    mu_vals = data['mu_vals']
    sigmas_sym = data['sigmas_sym']
    
    # Select the pertinent components to plot
    sigma_sym_xxx = sigmas_sym[:, 0, 0, 0]
    sigma_sym_xyy = sigmas_sym[:, 0, 1, 1]
    
    plt.figure(figsize=(8, 5))
    plt.axhline(0, linestyle=':', color='gray')
    plt.plot(mu_vals, sigma_sym_xxx, label=r'$\sigma^{L}_{xxx}$')
    plt.plot(mu_vals, sigma_sym_xyy, label=r'$\sigma^{L}_{xyy}$')
    plt.xlabel(r'$\mu$ (eV)')
    plt.ylabel(r'$\sigma$')
    # plt.ylim(-10, 10)
    plt.title("QMD's Nonlinear Conductivity")
    plt.legend()
    
    plot_path = config.FIGURES_DIR / "plot_scan_qmd.png"
    # plt.savefig(plot_path)
    # print(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_qmd()
