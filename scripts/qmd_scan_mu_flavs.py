import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import gc

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, get_QGT_chunk, get_qmd_from_qgt, integrate_qmd_chunk, sym_decomp_cond
import model.config as config

def scan_mu_flavors(mu_vals, chunk_size=1000):
    mu_lim = mu_vals[-1]
    n_mu = len(mu_vals)
    print(12*"=" + f" SCANNING QMD'S NLT PER FLAVOR FOR MU = [{-mu_lim}, {mu_lim}] AND {n_mu} POINTS " + 12*"=")

    # Build the 1D k-arrays
    n_pts = config.N_PTS
    kx_vals = np.linspace(-config.K_LIM, config.K_LIM, n_pts)
    ky_vals = np.linspace(-config.K_LIM, config.K_LIM, n_pts)

    # Define integration prefactors
    dk = kx_vals[1] - kx_vals[0]
    prefactor = (dk**2) / (2 * np.pi)

    # Labels for tracking flavors
    flavor_labels = [
        r"$\xi=1, \uparrow$", r"$\xi=1, \downarrow$",
        r"$\xi=-1, \uparrow$", r"$\xi=-1, \downarrow$"
    ]

    # Initialize accumulator array: (n_mu, n_flavors, j, l, m)
    sigmas_flavors = np.zeros((len(mu_vals), 4, 2, 2, 2))

    # 1. Outer Loop: Spin-Valley Flavors (System built once per flavor)
    for f_idx, (v_idx, s_idx) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        xi = config.VALLEY_IDX[v_idx]
        delta = config.DELTAS[v_idx, s_idx]
        e0 = config.E0_ARRAY[v_idx, s_idx]
        
        print(f"\nProcessing Flavor {f_idx + 1}/4: {flavor_labels[f_idx]}...")
        
        system = McCannCarts(
            N=config.N, valley_idx=xi, Delta=delta,
            gamma0=config.GAMMA0, gamma1=config.GAMMA1,
            gamma2=config.GAMMA2, gamma3=config.GAMMA3,
            gamma4=config.GAMMA4, E0=e0
        )
        
        # 2. Inner Loop: Iterate over kmesh chunks along the y axis
        for j_start in range(0, n_pts, chunk_size):
            j_end = min(j_start + chunk_size, n_pts)
            print(f"  Processing Chunk {j_start//chunk_size + 1}/{int(np.ceil(n_pts/chunk_size))}...", flush=True)
            
            # Define the chunked kmesh
            KX_c, KY_c = np.meshgrid(kx_vals, ky_vals[j_start:j_end])
            
            # Get energy bands for this chunk
            E0, E1 = system.get_energy(KX_c, KY_c)
            band_energies = {0: E0, 1: E1}

            # 3. Process and integrate each band immediately for this flavor
            for band_idx in (0, 1):
                # Compute full QGT and extract QMD using vector methods
                T_chunk = get_QGT_chunk(system, band_idx, KX_c, KY_c)
                qmd_chunk = get_qmd_from_qgt(T_chunk, dk)
                
                # Accumulate directly into this flavor's slice
                sigmas_flavors[:, f_idx, :, :, :] += integrate_qmd_chunk(
                    qmd_chunk, 
                    band_energies[band_idx], 
                    mu_vals, 
                    config.T_eff, 
                    prefactor
                )
                
                del T_chunk, qmd_chunk
            
            # Clean up chunk variables
            del KX_c, KY_c, E0, E1, band_energies
            gc.collect()

    print("\nScan completed safely.")
    return sigmas_flavors, flavor_labels


def save_results(mu_vals, contributions, labels, filename="scan_nlt_qmd_flavors.npz"):
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    save_path = data_dir / filename
    
    metadata = {
        "GAMMA0": config.GAMMA0,
        "GAMMA1": config.GAMMA1,
        "GAMMA2": config.GAMMA2,
        "GAMMA3": config.GAMMA3,
        "GAMMA4": config.GAMMA4,
        "N": config.N,
        "N_PTS": config.N_PTS,
        "K_LIM": config.K_LIM,
        "Temp": config.T_real,
        "DELTAS": config.DELTAS,
        "E0_ARRAY": config.E0_ARRAY,
    }
    
    np.savez(save_path, mu_vals=mu_vals, contributions=contributions, labels=labels, **metadata)
    print(f"Results saved to {save_path}")


def plot_flavors(mu_vals, contributions, labels):
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Extract xxx component for all flavors: shape (n_mu, 4)
    contributions_xxx = contributions[:, :, 0, 0, 0]
    
    # Calculate the total tensor summed over flavors: shape (n_mu, 2, 2, 2)
    total_qmd_tensor = np.sum(contributions, axis=1)
    
    # Symmetrize total tensor for plotting
    total_xxx_sym = np.zeros(len(mu_vals))
    for i in range(len(mu_vals)):
        sigma_sym, _ = sym_decomp_cond(total_qmd_tensor[i])
        total_xxx_sym[i] = sigma_sym[0, 0, 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    
    # Left Panel: Individual Flavors
    for i in range(4):
        axes[0].plot(mu_vals*1e3, contributions_xxx[:, i], label=labels[i])
    axes[0].set_xlabel(r'$\mu$ (meV)', fontsize=15)
    axes[0].set_ylabel(r'$\sigma_{xxx}$', fontsize=15)
    axes[0].tick_params(axis='both', labelsize=12)
    axes[0].legend(fontsize=12)
    
    # Right Panel: Symmetrized Total
    axes[1].plot(mu_vals*1e3, total_xxx_sym, color='black', linewidth=2, label='Total (Sym)')
    axes[1].set_xlabel(r'$\mu$ (meV)', fontsize=15)
    axes[1].set_ylabel(r'$\sigma_{xxx}$ (Sym Total)', fontsize=15)
    axes[1].tick_params(axis='both', labelsize=12)
    axes[1].legend(fontsize=12)
    
    plt.tight_layout()
    plot_path = figures_dir / f"scan_nlt_qmd_flavs_delta{config.DELTA1UP}.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    plt.show()


def main():
    # Use a safe range for chemical potential
    mu_lim = 0.02
    n_mu = 300
    mu_vals = np.linspace(-mu_lim, mu_lim, n_mu)
    
    contributions, labels = scan_mu_flavors(mu_vals, chunk_size=1000)
    save_results(mu_vals, contributions, labels, filename=f"scan_nlt_qmd_flavs_delta{config.DELTA1UP}.npz")
    plot_flavors(mu_vals, contributions, labels)

if __name__ == "__main__":
    main()