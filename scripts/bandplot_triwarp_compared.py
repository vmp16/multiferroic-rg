import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
import model.config as config

def plot_report_comparison():
    # Parameters for the plot
    n_pts = 400
    k_lim = 0.1
    n_levels = 55
    
    # K down flavor (Valley 1, Spin 1)
    valley_idx = 1
    spin_idx = 1
    v_idx = 0 # Valley 1
    delta = config.DELTAS[v_idx, spin_idx]
    e0 = config.E0_ARRAY[v_idx, spin_idx]

    # Define k-mesh
    kx = np.linspace(-k_lim, k_lim, n_pts)
    ky = np.linspace(-k_lim, k_lim, n_pts)
    KX, KY = np.meshgrid(kx, ky)

    # 1. System WITH trigonal warping (modified)
    sys_trig = McCannCarts(
        N=config.N,
        valley_idx=valley_idx,
        Delta=delta,
        gamma0=config.GAMMA0,
        gamma1=config.GAMMA1,
        gamma2=config.GAMMA2,
        gamma3=config.GAMMA3,
        gamma4=config.GAMMA4,
        E0=0
    )
    valence_trig = sys_trig.get_energy(KX, KY)[1]

    # 2. System WITHOUT trigonal warping
    sys_no_trig = McCannCarts(
        N=config.N,
        valley_idx=valley_idx,
        Delta=delta,
        gamma0=config.GAMMA0,
        gamma1=config.GAMMA1,
        gamma2=0.0,
        gamma3=0.0,
        gamma4=config.GAMMA4,
        E0=0
    )
    valence_no_trig = sys_no_trig.get_energy(KX, KY)[1]

    # Build the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]}, sharex='row', sharey='row')
    
    # Common color scale for comparison
    vmin = min(valence_trig.min(), valence_no_trig.min())
    vmax = 0.0 # Valence band is below 0

    # Plot WITHOUT Trigonal Warping (2D)
    im0 = axes[0, 0].pcolormesh(KX, KY, valence_no_trig, cmap='viridis', shading='auto', vmin=0.9*vmin, vmax=vmax)
    axes[0, 0].contour(KX, KY, valence_no_trig, n_levels, colors='gray', alpha=0.9, linewidths=0.5)
    axes[0, 0].set_aspect('equal')
    # axes[0, 0].set_xlabel(r'$k_x a$')
    axes[0, 0].set_ylabel(r'$k_y a$')

    # Plot WITH Trigonal Warping (2D)
    im1 = axes[0, 1].pcolormesh(KX, KY, valence_trig, cmap='viridis', shading='auto', vmin=0.9*vmin, vmax=vmax)
    axes[0, 1].contour(KX, KY, valence_trig, n_levels, colors='gray', alpha=0.8, linewidths=0.5)
    for ax in [axes[0, 0], axes[0, 1]]:
        ax.set_xticks([-k_lim, 0.0, k_lim])
        ax.set_yticks([-k_lim, 0.0, k_lim])

    axes[0, 1].set_aspect('equal')
    # axes[0, 1].set_xlabel(r'$k_x a$')

    cbar = fig.colorbar(im1, ax=axes[0, 1], label='Energy', fraction=0.046, pad=0.04)
    cbar.set_ticks([])

    # 1D energy bands along kx direction (ky = 0)
    kx_1d = np.linspace(-k_lim, k_lim, n_pts)
    ky_1d = np.zeros_like(kx_1d)
    
    cond_no_trig, val_no_trig_1d = sys_no_trig.get_energy(kx_1d, ky_1d)
    cond_trig, val_trig_1d = sys_trig.get_energy(kx_1d, ky_1d)

    # Plot 1D WITHOUT Trigonal Warping
    axes[1, 0].plot(kx_1d, cond_no_trig, color='blue')
    axes[1, 0].plot(kx_1d, val_no_trig_1d, color='blue')
    axes[1, 0].set_xlabel(r'$k_x a$')
    axes[1, 0].set_ylabel(r'Energy (eV)')

    # Plot 1D WITH Trigonal Warping
    axes[1, 1].plot(kx_1d, cond_trig, color='blue')
    axes[1, 1].plot(kx_1d, val_trig_1d, color='blue')
    axes[1, 1].set_xlabel(r'$k_x a$')

    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_xlim(-(k_lim+0.01), k_lim+0.01)
        ax.set_xticks([-k_lim, 0.0, k_lim])
        ax.set_ylim(-0.04, 0.04)
        ax.axhline(0, linestyle=':', color='gray', alpha=0.7)
    axes[1, 0].set_yticks([-0.02, 0.0, 0.02])

    # Set all text in the figure to the same size
    fontsize = 22
    for ax in axes.flatten():
        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)
        ax.title.set_size(fontsize)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(fontsize)
    cbar.ax.yaxis.label.set_size(fontsize)



    plt.tight_layout()
    
    # Save the figure
    output_path = project_root / 'figures' / 'report_trigonal_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Report figure saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_report_comparison()
