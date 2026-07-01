import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path to allow imports from model/
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, get_quantum_metric, get_qmd, deriv_fermi_distrib
import model.config as config

def test_plot_metric_qmd(v_idx=0, s_idx=0):
    """
    Generate 2D colormaps for:
    1. Quantum metric g_xx
    2. Quantum metric dipole component xxx (G_xxx)
    3. Product of G_xxx and the derivative of the Fermi distribution (integrand)
    """
    print("Initializing system and calculating tensors...")
    
    # Initialize the system using configuration parameters
    system = McCannCarts(
        gamma0=config.GAMMA0,
        gamma1=config.GAMMA1,
        valley_idx=config.VALLEY_IDX[v_idx],
        Delta=config.DELTAS[v_idx, s_idx],
        N=config.N,
        gamma2=config.GAMMA2,
        gamma3=config.GAMMA3,
        gamma4=config.GAMMA4,
        E0=config.E0_ARRAY[v_idx, s_idx]
    )

    # Build the kmesh
    n_pts = 200
    k_lim = 0.2
    KX, KY = get_kmesh(k_lim, n_pts)
    p = np.stack((KX, KY), axis=-1)

    # Select the band
    band_idx = 0

    # Calculate Quantum Metric tensor g_jl
    g_tensor = get_quantum_metric(system, band_idx, p)
    g_xx = g_tensor[0, 0]

    # Calculate Quantum Metric Dipole tensor D_ijl = d(g_jl) / d(k_i)
    qmd_tensor = get_qmd(system, p, band_idx)
    qmd_xxx = qmd_tensor[0, 0, 0]

    # Calculate the derivative of the Fermi distribution
    energies = system.get_energy(p)
    band_E = energies[band_idx]
    df_dE = deriv_fermi_distrib(band_E, config.mu_eff, config.T_eff)

    # Calculate the product (integrand for the conductivity)
    integrand_xxx = qmd_xxx * df_dE

    print("Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Quantum Metric g_xx
    im0 = axes[0].pcolormesh(KX, KY, g_xx, cmap='viridis', shading='auto')
    axes[0].set_title(r'Quantum Metric $g_{xx}$', fontsize=14)
    axes[0].set_xlabel(r'$k_x$', fontsize=12)
    axes[0].set_ylabel(r'$k_y$', fontsize=12)
    axes[0].set_aspect('equal')
    fig.colorbar(im0, ax=axes[0], label=r'$g_{xx}$')

    # Plot 2: Quantum Metric Dipole xxx
    vmax1 = np.max(np.abs(qmd_xxx))
    im1 = axes[1].pcolormesh(KX, KY, qmd_xxx, cmap='RdBu_r', shading='auto', 
                             vmin=-vmax1/10, vmax=vmax1/10)
    axes[1].set_title(r'Quantum Metric Dipole $\partial_x g_{xx}$', fontsize=14)
    axes[1].set_xlabel(r'$k_x$', fontsize=12)
    axes[1].set_ylabel(r'$k_y$', fontsize=12)
    axes[1].set_aspect('equal')
    fig.colorbar(im1, ax=axes[1], label=r'$G_{xxx}$')

    # Plot 3: Integrand (G_xxx * df/dE)
    vmax2 = np.max(np.abs(integrand_xxx))
    im2 = axes[2].pcolormesh(KX, KY, integrand_xxx, cmap='RdBu_r', shading='auto',
                             vmin=-vmax2, vmax=vmax2)
    axes[2].set_title(r'$\partial_x g_{xx} \times \partial f / \partial E, \mu=$'+str(config.mu_eff)[:5], fontsize=14)
    axes[2].set_xlabel(r'$k_x$', fontsize=12)
    axes[2].set_ylabel(r'$k_y$', fontsize=12)
    axes[2].set_aspect('equal')
    fig.colorbar(im2, ax=axes[2], label=r'Integrand')

    plt.tight_layout()
    
    # Ensure figures directory exists
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    plot_path = figures_dir / "test_metric_qmd_integrand_2d.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Success! 2D colormaps saved to: {plot_path}")

    plt.show()

if __name__ == "__main__":
    test_plot_metric_qmd(v_idx=0, s_idx=0)
