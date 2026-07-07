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

def scan_mu(mu_vals, chunk_size=1000):
    mu_lim = mu_vals[-1]
    n_mu = len(mu_vals)
    print(12*"=" + F" SCANNING QMD'S NLT FOR MU = [{-mu_lim}, {mu_lim}] AND {n_mu} POINTS " + 12*"=")

    # Build the 1D k-arrays
    n_pts = config.N_PTS
    kx_vals = np.linspace(-config.K_LIM, config.K_LIM, n_pts)
    ky_vals = np.linspace(-config.K_LIM, config.K_LIM, n_pts)

    # Define integration prefactors
    dk = kx_vals[1] - kx_vals[0]
    prefactor = (dk**2) / (2 * np.pi)

    # Initialize the total conductivity accumulator array
    sigmas_total = np.zeros((len(mu_vals), 2, 2, 2))

    for v_idx in range(2):
        for s_idx in range(2):
            xi = config.VALLEY_IDX[v_idx]
            delta = config.DELTAS[v_idx, s_idx]
            e0 = config.E0_ARRAY[v_idx, s_idx]
            
            print(f"\nInitializing system for Flavor: valley={xi}, spin={s_idx}...")
            system = McCannCarts(
                N=config.N, valley_idx=xi, Delta=delta,
                gamma0=config.GAMMA0, gamma1=config.GAMMA1,
                gamma2=config.GAMMA2, gamma3=config.GAMMA3,
                gamma4=config.GAMMA4, E0=e0
            )

            # 2. Inner Loop: Iterate over kmesh chunks along the y-axis
            for j_start in range(0, config.N_PTS, chunk_size):
                j_end = min(j_start + chunk_size, config.N_PTS)
                print(f"  Processing Chunk {j_start//chunk_size + 1}/{int(np.ceil(n_pts/chunk_size))}...", flush=True)
                
                # Define the chunked kmesh
                KX_c, KY_c = np.meshgrid(kx_vals, ky_vals[j_start:j_end])

                # Get energy bands
                E0, E1 = system.get_energy(KX_c, KY_c)
                band_energies = {0: E0, 1: E1}

                # 3. Process and integrate each band
                for band_idx in (0, 1):
                    # Compute QGT and extract QMD for the given flavor, chunk and band
                    T_chunk = get_QGT_chunk(system, band_idx, KX_c, KY_c)
                    qmd_chunk = get_qmd_from_qgt(T_chunk, dk)

                    # Accumulate directly into the total conductivity tensor
                    sigmas_total += integrate_qmd_chunk(
                        qmd_chunk, 
                        band_energies[band_idx], 
                        mu_vals, 
                        config.T_eff, 
                        prefactor
                    )

                    # Clear memory for this band's chunk calculation
                    del T_chunk, qmd_chunk
                
                # Clean up chunk mesh and energies
                del KX_c, KY_c, E0, E1, band_energies
                gc.collect()

    return sigmas_total

def save_results(mu_vals, sigmas_sym, filename="scan_nlt_qmd.npz"):
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    save_path = data_dir / filename
    
    # Metadata from config
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
    
    np.savez(save_path, mu_vals=mu_vals, sigmas_sym=sigmas_sym, **metadata)
    print(f"Results saved to {save_path}")

def plot_results(mu_vals, sigma_tensor):
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot components like xxx, xyy, etc.
    # sigma_tensors shape: (n_mu, 2, 2, 2)
    # sigma_ijl where i is current, j, l are E-fields
    plt.plot(mu_vals, sigma_tensor[:, 0, 0, 0], label=r'$\sigma_{xxx}$')
    plt.plot(mu_vals, sigma_tensor[:, 0, 1, 1], label=r'$\sigma_{xyy}$')
    plt.plot(mu_vals, sigma_tensor[:, 1, 0, 0], label=r'$\sigma_{yxx}$')
    plt.plot(mu_vals, sigma_tensor[:, 1, 1, 1], label=r'$\sigma_{yyy}$')
    
    plt.xlabel(r'$\mu$ (eV)')
    plt.ylabel(r'$\sigma$ (units)')
    plt.title("QMD's Nonlinear Conductivity Scan")
    plt.legend()
    plt.grid(True)
    plot_path = figures_dir / "scan_nlt_qmd.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def main():
    mu_lim = 0.02
    n_mu = 200
    mu_vals = np.linspace(-mu_lim, mu_lim, n_mu)
    sigmas_sym = scan_mu(mu_vals, chunk_size=1000)
    save_results(mu_vals, sigmas_sym)
    plot_results(mu_vals, sigmas_sym)

if __name__ == "__main__":
    main()
