import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis_old import get_kmesh, get_quantum_metric_component, deriv_fermi_distrib, sym_decomp_cond
import model.config as config
import gc

def calculate_flavor_contributions(mu_vals, n_pts=400, chunk_size=1000):
    """
    Calculate the full QMD conductivity tensor (2,2,2) for each spin-valley flavour using chunking.
    """
    print(f"Calculating full QMD tensors sequentially with chunking (N_PTS={n_pts}, CHUNK={chunk_size})...")
    
    # Generate 1D k-arrays
    kx_vals = np.linspace(-config.K_LIM, config.K_LIM, n_pts)
    ky_vals = np.linspace(-config.K_LIM, config.K_LIM, n_pts)
    dk = kx_vals[1] - kx_vals[0]
    prefactor = (dk**2) / (2 * np.pi)

    # Flavor labels
    flavor_labels = [
        r"$\xi=1, \uparrow$", r"$\xi=1, \downarrow$",
        r"$\xi=-1, \uparrow$", r"$\xi=-1, \downarrow$"
    ]
    
    # contributions shape: (n_mu, 4, 2, 2, 2)
    contributions = np.zeros((len(mu_vals), 4, 2, 2, 2))
    
    for f_idx, (v_idx, s_idx) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        xi = config.VALLEY_IDX[v_idx]
        delta = config.DELTAS[v_idx, s_idx]
        e0 = config.E0_ARRAY[v_idx, s_idx]
        
        print(f"Processing flavor {flavor_labels[f_idx]}...")
        
        system = McCannCarts(
            N=config.N, valley_idx=xi, Delta=delta,
            gamma0=config.GAMMA0, gamma1=config.GAMMA1,
            gamma2=config.GAMMA2, gamma3=config.GAMMA3,
            gamma4=config.GAMMA4, E0=e0
        )

        for i_start in range(0, n_pts, chunk_size):
            i_end = min(i_start + chunk_size, n_pts)
            print(f"  Chunk {i_start//chunk_size + 1}/{int(np.ceil(n_pts/chunk_size))}...")
            
            KX_c, KY_c = np.meshgrid(kx_vals, ky_vals[i_start:i_end])
            E_plus, E_minus = system.get_energy(KX_c, KY_c)
            
            for band_idx in range(2):
                band_E = E_plus if band_idx == 0 else E_minus
                
                # Calculate all 8 components of QMD tensor D_ijl = d(g_jl)/dk_i
                for i_comp in range(2):
                    for j_comp in range(2):
                        for l_comp in range(2):
                            g_jl = get_quantum_metric_component(system, band_idx, KX_c, KY_c, j_comp, l_comp)
                            grad_axis = 1 if i_comp == 0 else 0
                            dq_di = np.gradient(g_jl, dk, axis=grad_axis)
                            del g_jl
                            
                            mu_block_size = 10
                            for m_start in range(0, len(mu_vals), mu_block_size):
                                m_end = min(m_start + mu_block_size, len(mu_vals))
                                mu_chunk = mu_vals[m_start:m_end]
                                df_dE_block = deriv_fermi_distrib(band_E[None, :, :], mu_chunk[:, None, None], config.T_eff)
                                integrals = np.einsum('mjk, jk -> m', df_dE_block, dq_di)
                                contributions[m_start:m_end, f_idx, i_comp, j_comp, l_comp] += integrals * prefactor
                                del df_dE_block, integrals
                            
                            del dq_di
                
            del E_plus, E_minus, KX_c, KY_c
            gc.collect()
            
    print("\nScan completed.")
    return contributions, flavor_labels

def save_results(mu_vals, contributions, labels, filename="scan_nlt_qmd_flavors_old.npz"):
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
    
    # contributions has shape (n_mu, 4, 2, 2, 2)
    contributions_xxx = contributions[:, :, 0, 0, 0]
    total_qmd_tensor = np.sum(contributions, axis=1) # (n_mu, 2, 2, 2)
    
    total_xxx_sym = np.zeros(len(mu_vals))
    for i in range(len(mu_vals)):
        sigma_sym, _ = sym_decomp_cond(total_qmd_tensor[i])
        total_xxx_sym[i] = sigma_sym[0, 0, 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    
    for i in range(4):
        axes[0].plot(mu_vals, contributions_xxx[:, i], label=labels[i])
    axes[0].set_xlabel(r'$\mu$ (eV)', fontsize=17)
    axes[0].set_ylabel(r'$\sigma_{xxx}$', fontsize=17)
    axes[0].tick_params(axis='both', labelsize=16)
    axes[0].legend(fontsize=16)
    
    axes[1].plot(mu_vals, total_xxx_sym, color='black', linewidth=2, label='Total (Sym)')
    axes[1].set_xlabel(r'$\mu$ (eV)', fontsize=17)
    axes[1].set_ylabel(r'$\sigma_{xxx}$ (Sym)', fontsize=17)
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1].legend(fontsize=16)
    
    plt.tight_layout()
    plot_path = figures_dir / "scan_nlt_qmd_flavors_old.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    mu_vals = np.linspace(-0.02, 0.02, 200)
    contributions, labels = calculate_flavor_contributions(mu_vals, n_pts=config.N_PTS)
    save_results(mu_vals, contributions, labels)
    plot_flavors(mu_vals, contributions, labels)
