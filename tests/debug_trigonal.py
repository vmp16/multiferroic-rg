import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
import model.config as config

def plot_trigonal_warping(n_pts=300):
    # Zoom in significantly
    k_lim = 0.1 # Reduced from 0.3
    kx = np.linspace(-k_lim, k_lim, n_pts)
    ky = np.linspace(-k_lim, k_lim, n_pts)
    KX, KY = np.meshgrid(kx, ky)
    p = np.stack((KX, KY), axis=-1)

    # Use zero gap to see the pockets clearly
    valley_idx = 1
    delta = 0.0 # Force zero gap
    e0 = 0.0
    
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

    energies = system.get_energy(p)
    conduction = energies[0]

    plt.figure(figsize=(8, 8))
    # Use many contours near zero energy
    levels = np.linspace(0, 0.02, 30) 
    im = plt.pcolormesh(KX, KY, conduction, cmap='viridis', shading='auto', vmax=0.05)
    plt.contour(KX, KY, conduction, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(im, label='Energy (eV)')
    plt.title(f'Trigonal Warping Pockets (N={config.N}, Delta=0)')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.gca().set_aspect('equal')
    
    output_path = project_root / 'figures' / 'test_trigonal_warping.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_trigonal_warping()
