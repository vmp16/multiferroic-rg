import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from model.model import McCannCarts
from model.analysis import get_kmesh, get_G_tensor, velocity_element, deriv_fermi_distrib, sym_decomp_cond
import model.config as config

def scan_mu(mu_lim=0.3, n_mu=100):
    print(12*"=" + F" SCANNING GAO'S NLT FOR MU = [{-mu_lim}, {mu_lim}] AND {n_mu} POINTS " + 12*"=")
    mu_vals = np.linspace(-mu_lim, mu_lim, n_mu)

    # Build the kmesh
    n_pts = config.N_PTS
    KX, KY = get_kmesh(config.K_LIM, n_pts)
    p = np.stack((KX, KY), axis=-1)
    dk = KX[0, 1] - KX[0, 0]
    prefactor = (dk**2) / (2 * np.pi)

    # Pre-calculate Gao tensors for each flavor (2 valleys * 2 spins) and 2 bands
    print(f"Pre-calculating Gao tensors for all flavors (N_PTS={n_pts})...")
    flavor_tensors = [] # List of list of (T_tensor, energy)

    for v_idx, xi in enumerate(config.VALLEY_IDX):
        for s_idx in range(2):
            delta = config.DELTAS[v_idx, s_idx]
            e0 = config.E0_ARRAY[v_idx, s_idx]
            
            system = McCannCarts(
                N=config.N, valley_idx=xi, Delta=delta,
                gamma0=config.GAMMA0, gamma1=config.GAMMA1,
                gamma2=config.GAMMA2, gamma3=config.GAMMA3,
                gamma4=config.GAMMA4, E0=e0
            )
            
            energies = system.get_energy(p)
            band_info = []
            for band_idx in range(2):
                G_tensor = get_G_tensor(system, p, band_idx)
                Vx, Vy = velocity_element(system, p, band_idx, band_idx)
                V_vector = np.array([Vx, Vy])
                
                # Gao's formula integrand
                term1 = np.einsum('i..., jl... -> ijl...', V_vector, G_tensor)
                term2 = np.einsum('j..., il... -> ijl...', V_vector, G_tensor)
                term3 = np.einsum('l..., ij... -> ijl...', V_vector, G_tensor)
                T_tensor = term1 - (term2 + term3) / 2
                
                band_info.append((T_tensor, energies[band_idx]))
            flavor_tensors.append(band_info)

    sigmas_sym = []
    sigmas_asym = []

    print(f"Starting mu scan for {len(mu_vals)} points...")
    for i, mu in enumerate(mu_vals):
        print(f"Progress: {100 * (i + 1) / len(mu_vals):.1f}%", end='\r')

        sigma_total = np.zeros((2, 2, 2))
        for f_idx in range(4):
            for band_idx in range(2):
                T_tensor, band_E = flavor_tensors[f_idx][band_idx]
                df_dE = deriv_fermi_distrib(band_E, mu, config.T_eff)
                
                integral = np.sum(T_tensor * df_dE, axis=tuple(range(3, T_tensor.ndim)))
                sigma_total += integral * prefactor
        
        sigma_sym, sigma_asym = sym_decomp_cond(sigma_total)
        sigmas_sym.append(sigma_sym)
        sigmas_asym.append(sigma_asym)

    print("\nScan completed.")
    return mu_vals, np.array(sigmas_sym), np.array(sigmas_asym)

def save_results(mu_vals, sigmas_sym, sigmas_asym, filename="scan_nlt_gao.npz"):
    config.DATA_DIR.mkdir(exist_ok=True)
    
    save_path = config.DATA_DIR / filename
    
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
    config.FIGURES_DIR.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot components like xxx, xyy, etc.
    # sigma_tensor shape: (n_mu, 2, 2, 2)
    # sigma_ijl where i is current, j, l are E-fields
    plt.plot(mu_vals, sigma_tensor[:, 0, 0, 0], label=r'$\sigma_{xxx}$')
    plt.plot(mu_vals, sigma_tensor[:, 0, 1, 1], label=r'$\sigma_{xyy}$')
    plt.plot(mu_vals, sigma_tensor[:, 1, 0, 0], label=r'$\sigma_{yxx}$')
    plt.plot(mu_vals, sigma_tensor[:, 1, 1, 1], label=r'$\sigma_{yyy}$')
    
    plt.xlabel(r'$\mu$ (eV)')
    plt.ylabel(r'$\sigma$ (units)')
    plt.title("Gao's Nonlinear Conductivity Scan")
    plt.legend()
    plt.grid(True)

    plot_path = config.FIGURES_DIR / "scan_nlt_gao.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def main():
    mu_vals, sigmas_sym, sigmas_asym = scan_mu(mu_lim=0.2, n_mu=400)
    save_results(mu_vals, sigmas_sym, sigmas_asym)
    # plot_results(mu_vals, sigma_tensors)

if __name__ == "__main__":
    main()
