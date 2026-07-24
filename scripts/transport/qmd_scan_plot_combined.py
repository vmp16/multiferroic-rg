import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from qmd_scan_mu_flavs_plot import load_qmd_flavors
from plot_task3_results import load_task3_data

# Set math font to Computer Modern to properly render \varepsilon
plt.rcParams['mathtext.fontset'] = 'cm'

def main():
    config.FIGURES_DIR.mkdir(exist_ok=True)
    output_path = config.FIGURES_DIR / "combined_qmd.png"

    # Load the data
    mu_vals_contribs, contribs, _ = load_qmd_flavors(filename="scan_nlt_qmd_flavors.npz")

    data = load_task3_data(filename="task3_qmd_mu_scan.npz")
    mu_vals = data["mu_vals"]
    res1 = data["sigma_branch1"]
    res2 = data["sigma_branch2"]

    # contributions has shape (n_mu, 4, 2, 2, 2)
    contributions_xxx = contribs[:, :, 0, 0, 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    for i in range(len(axes)):
        axes[i].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[i].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    labels = [r"$K\uparrow$", r"$K\downarrow$", r"$K'\uparrow$", r"$K'\downarrow$"]
    colors = ['lightsalmon', 'slateblue', 'firebrick', 'lightsteelblue']
    for i in range(4):
        axes[0].plot(mu_vals_contribs, contributions_xxx[:, i], color=colors[i], label=labels[i])
    axes[0].set_xlabel(r'$\mu~(eV)$', fontsize=20)
    axes[0].set_ylabel(r'$\sigma_{xxx}$ ($(e^2/h)(a/V)$)', fontsize=20)
    axes[0].tick_params(axis='both', labelsize=18)
    axes[0].legend(fontsize=18)

    axes[1].plot(mu_vals, res1[:, 0, 0, 0], label=r"$\epsilon_0=0.05~eV$", linewidth=1.5, color='darkorange')
    axes[1].plot(mu_vals, res2[:, 0, 0, 0], label=r"$\epsilon_0=-0.05~eV$", linewidth=1.5, color='teal')

    axes[1].set_ylim(-280, 280)
    axes[1].set_xlim(-0.14, 0.14)

    axes[1].set_xlabel(r'$\mu~(eV)$', fontsize=20)
    # axes[1].set_ylabel(r'$\sigma_{xxx}$ ($(e^2/h)(a/V)$)', fontsize=19)
    # plt.title(r'Quantum Metric Dipole Conductivity vs $\mu$ ($U=0$, $T=%d$K)' % T, fontsize=15)
    axes[1].legend(fontsize=19)
    axes[1].tick_params(axis='both', labelsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Successfully re-plotted. Figure saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()