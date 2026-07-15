import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
import model.config as config

def plot_band_2d(valley_idx=1, spin_idx=0, n_pts=200, n_levels=10, k_lim=None):
    """
    Plot the energy of the conduction and valence bands as a color map over the k-mesh.
    """
    if k_lim is None:
        k_lim = config.K_LIM
        
    # Define k-mesh
    kx = np.linspace(-k_lim, k_lim, n_pts)
    ky = np.linspace(-k_lim, k_lim, n_pts)
    KX, KY = np.meshgrid(kx, ky)
    # p = np.stack((KX, KY), axis=-1)

    # Get parameters for the selected flavor
    v_idx = 0 if valley_idx == 1 else 1
    delta = config.DELTAS[v_idx, spin_idx]
    e0 = config.E0_ARRAY[v_idx, spin_idx]
    spin_label = 'up' if spin_idx == 0 else 'dn'

    system = McCannCarts(
        N=config.N,
        valley_idx=valley_idx,
        Delta=delta,
        gamma0=config.GAMMA0,
        gamma1=config.GAMMA1,
        gamma2=config.GAMMA2,
        gamma3=config.GAMMA3,
        gamma4=config.GAMMA4,
        E0=e0
    )

    # Get energies
    energies = system.get_energy(KX, KY)
    conduction = energies[0]
    valence = energies[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    
    # Adaptive color limits for better visibility
    vmax = max(np.abs(conduction).max(), np.abs(valence).max())
    if k_lim < 0.1: # Zoom mode
        vmax = 0.05 

    # Plot Conduction Band
    im0 = axes[0].pcolormesh(KX, KY, conduction, cmap='viridis', shading='auto', vmax=vmax)
    axes[0].contour(KX, KY, conduction, n_levels, colors='white', alpha=0.5, linewidths=0.5)
    axes[0].set_title(f'Conduction Band (Valley {valley_idx}, Spin {spin_label})')
    axes[0].set_aspect('equal')
    fig.colorbar(im0, ax=axes[0], label='Energy (eV)')

    # Plot Valence Band
    im1 = axes[1].pcolormesh(KX, KY, valence, cmap='plasma', shading='auto', vmin=-vmax)
    axes[1].contour(KX, KY, valence, n_levels, colors='gray', alpha=0.8, linewidths=0.5)
    axes[1].set_title(f'Valence Band (Valley {valley_idx}, Spin {spin_label})')
    axes[1].set_aspect('equal')
    fig.colorbar(im1, ax=axes[1], label='Energy (eV)')

    for ax in axes:
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')

    plt.tight_layout()
    
    # Save the figure
    suffix = f'_k{k_lim:.2f}' if k_lim != config.K_LIM else ''
    output_filename = f'energy_band_2d_v{valley_idx}_s{spin_label}{suffix}.png'
    output_path = project_root / 'figures' / output_filename
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    # To see trigonal warping pockets:
    # 1. Use a flavor with small or zero gap (e.g., spin_idx=1 for Valley 1)
    # 2. Zoom in by reducing k_lim
    plot_band_2d(valley_idx=1, spin_idx=1, k_lim=0.11, n_levels=30)
