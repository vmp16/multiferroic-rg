import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
import model.config as config

def plot_bands(mu_plot = 0.0, U=0.0):
    # Define k-path (along kx, ky=0)
    k_pts = 500
    kx = np.linspace(-config.K_LIM, config.K_LIM, k_pts)
    ky = np.zeros_like(kx)

    fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    spin_labels = ['up', 'dn']
    spin_colors = {'up': 'red', 'dn': 'blue'}

    for ax, (v_idx, xi) in zip(axes, enumerate(config.VALLEY_IDX)):
        valley_name = r"$K$" if xi == 1 else r"$K'$"
        ax.set_title(f"Valley {valley_name}", fontsize=16)
        ax.axhline(0, linestyle=':', color='gray', alpha=0.7)

        if mu_plot != 0.0:
            ax.axhline(mu_plot*1e3, linestyle='--', color='gray')

        for spin_idx, spin in enumerate(spin_labels):
            delta_eff = U + config.DELTAS[v_idx, spin_idx]

            system = McCannCarts(
                N=config.N,
                valley_idx=xi,
                Delta=delta_eff,
                gamma0=config.GAMMA0,
                gamma1=config.GAMMA1,
                gamma2=config.GAMMA2,
                gamma3=config.GAMMA3,
                gamma4=config.GAMMA4,
                E0=config.E0_ARRAY[v_idx, spin_idx]
            )
            energies = system.get_energy(kx, ky)
            color = spin_colors[spin]

            # Plot both conduction and valence bands for this spin.
            ax.plot(
                kx, energies[0]*1e3,
                color=color, lw=1.5,
            )
            ax.plot(
                kx, energies[1]*1e3,
                color=color, lw=1.5, ls='--',
            )

        ax.set_ylim(-25, 25)
        ax.set_xlim(-0.12, 0.12)
        ax.set_xticks([-0.1, 0, 0.1])
        # ax.set_yticks([-0.1, 0.0, 0.1])
        ax.tick_params(axis='both', labelsize=17)
    axes[0].set_ylabel(r'Energy (meV)', fontsize=18)

    # fig.text(0.53, 0.95, f'Delta={config.DELTA1UP} eV', ha='center', fontsize=18)
    fig.text(0.53, 0.95, f"U={U*1e3} meV, mu={mu_plot*1e3:.2f} meV", ha='center', fontsize=17)
    fig.text(0.53, 0.03, r'$k_x a$', ha='left', fontsize=18)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    # Save the figure
    output_path = project_root / 'figures' / f'bandstructure_U{U}.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_bands(mu_plot = 0.0*1e-3, U=-0.02)
