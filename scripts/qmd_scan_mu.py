import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, get_qmd, deriv_fermi_distrib, sym_decomp_cond
import model.config as config

def scan_mu(mu_lim=0.3, n_mu=100):
    print(12*"=" + F" SCANNING QMD'S NLT FOR MU = [{-mu_lim}, {mu_lim}] AND {n_mu} POINTS " + 12*"=")
    mu_vals = np.linspace(-mu_lim, mu_lim, n_mu)

    # Build the kmesh
    n_pts = config.N_PTS
    KX, KY = get_kmesh(config.K_LIM, n_pts)
    p = np.stack((KX, KY), axis=-1)

    # Define integration prefactors
    dk = KX[0, 1] - KX[0, 0]
    prefactor = (dk**2) / (2 * np.pi)

    # Pre-calculate QMD tensors for each flavor (2 valleys * 2 spins) and 2 bands
    print(f"Pre-calculating QMD tensors for all flavors (N_PTS={n_pts})...")
    flavor_tensors = [] # List of list of (qmd_tensor, energy)

    for v_idx, xi in enumerate(config.VALLEY_IDX):
        for s_idx in range(2):
            delta = config.DELTAS[v_idx, s_idx]
            e0 = config.E0_ARRAY[v_idx, s_idx]
            
            system = McCannCarts(
                N=config.N,
                valley_idx=xi,
                Delta=delta,
                gamma0=config.GAMMA0,
                gamma1=config.GAMMA1,
                gamma2=config.GAMMA2,
                gamma3=config.GAMMA3,
                gamma4=config.GAMMA4,
                E0=e0
            )
            
            energies = system.get_energy(p)
            band_info = []
            for band_idx in range(2):
                qmd_tensor = get_qmd(system, p, band_idx)
                band_info.append((qmd_tensor, energies[band_idx]))
            flavor_tensors.append(band_info)

    sigmas_sym = []
    sigmas_asym = []

    print(f"Starting mu scan for {len(mu_vals)} points...")
    for i, mu in enumerate(mu_vals):
        print(f"Progress: {100 * (i + 1) / len(mu_vals):.1f}%", end='\r')

        sigma_total = np.zeros((2, 2, 2))
        # Iterate over the spin-valley flavors
        for f_idx in range(4):
            for band_idx in range(2):
                qmd_tensor, band_E = flavor_tensors[f_idx][band_idx]
                df_dE = deriv_fermi_distrib(band_E, mu, config.T_eff)
                
                # qmd_tensor has shape (2, 2, 2, Ny, Nx)
                integral = np.sum(qmd_tensor * df_dE, axis=tuple(range(3, qmd_tensor.ndim)))
                sigma_total += integral * prefactor
        
        sigma_sym, sigma_asym = sym_decomp_cond(sigma_total)
        sigmas_sym.append(sigma_sym)
        sigmas_asym.append(sigma_asym)

    print("\nScan completed.")
    return mu_vals, np.array(sigmas_sym), np.array(sigmas_asym)

def save_results(mu_vals, sigmas_sym, sigmas_asym, filename="scan_nlt_qmd.npz"):
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
    
    np.savez(save_path, mu_vals=mu_vals, sigmas_sym=sigmas_sym, sigmas_asym=sigmas_asym, **metadata)
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
    mu_vals, sigmas_sym, sigmas_asym = scan_mu(mu_lim=0.5, n_mu=1000)
    save_results(mu_vals, sigmas_sym, sigmas_asym)
    # plot_results(mu_vals, sigma_tensors)

if __name__ == "__main__":
    main()
