import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import deriv_fermi_distrib, sym_decomp_cond
import model.config as config

def get_optimized_qmd_sigma(system, KX, KY, dk, mu, T):
    # 1. Energies and gap
    E0, E1 = system.get_energy(KX, KY)
    dE = E0 - E1
    dE_sq = dE**2
    
    # 2. Eigenstates
    psi0_1, psi0_2 = system.get_eigenstate_components(KX, KY, 0)
    psi1_1, psi1_2 = system.get_eigenstate_components(KX, KY, 1)
    
    # 3. Hamiltonian derivatives
    dh0_dx = system.derivate_h0_at_k(KX, KY, 1)
    dh0_dy = system.derivate_h0_at_k(KX, KY, 0)
    dX_dx = system.derivate_X_at_k(KX, KY, 1)
    dX_dy = system.derivate_X_at_k(KX, KY, 0)
    
    # 4. Velocity elements <0|v_i|1>
    v_x_01 = np.conj(psi0_1) * dX_dx * psi1_2 + np.conj(psi0_2) * np.conj(dX_dx) * psi1_1
    v_y_01 = np.conj(psi0_1) * dX_dy * psi1_2 + np.conj(psi0_2) * np.conj(dX_dy) * psi1_1
    
    # 5. Metric components
    g_xx = np.abs(v_x_01)**2 / dE_sq
    g_yy = np.abs(v_y_01)**2 / dE_sq
    g_xy = np.real(v_x_01 * np.conj(v_y_01)) / dE_sq
    
    # 6. Fermi derivatives
    df_sum = deriv_fermi_distrib(E0, mu, T) + deriv_fermi_distrib(E1, mu, T)
    
    # 7. Integrals
    sigma = np.zeros((2, 2, 2))
    
    # dg/dkx (axis 1), dg/dky (axis 0)
    dgxx_dx = np.gradient(g_xx, dk, axis=1)
    dgxx_dy = np.gradient(g_xx, dk, axis=0)
    dgxy_dx = np.gradient(g_xy, dk, axis=1)
    dgxy_dy = np.gradient(g_xy, dk, axis=0)
    dgyy_dx = np.gradient(g_yy, dk, axis=1)
    dgyy_dy = np.gradient(g_yy, dk, axis=0)
    
    sigma[0, 0, 0] = np.sum(df_sum * dgxx_dx) # xxx
    sigma[1, 0, 0] = np.sum(df_sum * dgxx_dy) # yxx
    
    sigma[0, 0, 1] = np.sum(df_sum * dgxy_dx) # xxy
    sigma[0, 1, 0] = sigma[0, 0, 1]
    sigma[1, 0, 1] = np.sum(df_sum * dgxy_dy) # yxy
    sigma[1, 1, 0] = sigma[1, 0, 1]
    
    sigma[0, 1, 1] = np.sum(df_sum * dgyy_dx) # xyy
    sigma[1, 1, 1] = np.sum(df_sum * dgyy_dy) # yyy
    
    return sigma

def scan_qmd_vs_U(U_lim=0.1, n_U=50, chunk_size=1000):
    print(12*"=" + f" SCANNING QMD VS U (mu={config.mu_eff}, T={config.T_real}K, N_PTS={config.N_PTS}) " + 12*"=")
    U_vals = np.linspace(-U_lim, U_lim, n_U)
    kx_vals = np.linspace(-config.K_LIM, config.K_LIM, config.N_PTS)
    ky_vals = np.linspace(-config.K_LIM, config.K_LIM, config.N_PTS)
    dk = kx_vals[1] - kx_vals[0]
    prefactor = (dk**2) / (2 * np.pi)

    e0_dn_options = [config.E0_1DN, -config.E0_1DN]
    branch_results = [np.zeros((n_U, 2, 2, 2)), np.zeros((n_U, 2, 2, 2))]

    for b_idx, e0_dn in enumerate(e0_dn_options):
        print(f"\nProcessing Branch {b_idx+1} (E0_1DN = {e0_dn:.3f})...")
        e0_1up = config.E0_1UP
        e0_1dn = e0_dn
        e0_2up = -e0_dn
        e0_2dn = e0_1up
        local_e0_array = np.array([[e0_1up, e0_1dn],
                                   [e0_2up, e0_2dn]])

        for i, U_val in enumerate(U_vals):
            print(f"Progress: {100 * (i + 1) / n_U:.1f}% (U={U_val:.4f})", flush=True)
            sigma_total = np.zeros((2, 2, 2))
            
            for j_start in range(0, config.N_PTS, chunk_size):
                j_end = min(j_start + chunk_size, config.N_PTS)
                KX_c, KY_c = np.meshgrid(kx_vals, ky_vals[j_start:j_end])
                
                for v_idx in range(2):
                    for s_idx in range(2):
                        xi = config.VALLEY_IDX[v_idx]
                        intrinsic_delta = config.DELTAS[v_idx, s_idx]
                        e0 = local_e0_array[v_idx, s_idx]
                        effective_delta = intrinsic_delta + U_val
                        
                        system = McCannCarts(
                            N=config.N, valley_idx=xi, Delta=effective_delta,
                            gamma0=config.GAMMA0, gamma1=config.GAMMA1,
                            gamma2=config.GAMMA2, gamma3=config.GAMMA3,
                            gamma4=config.GAMMA4, E0=e0
                        )
                        
                        sigma_total += prefactor * get_optimized_qmd_sigma(system, KX_c, KY_c, dk, config.mu_eff, config.T_eff)
                
                gc.collect()
            
            sigma_sym, _ = sym_decomp_cond(sigma_total)
            branch_results[b_idx][i] = sigma_sym

    return U_vals, branch_results[0], branch_results[1]

def main():
    config.T_real = 4
    config.T_eff = config.kB * config.T_real
    config.N_PTS = 6000
    config.mu_eff = -0.01
    
    U_lim = 0.1
    n_U = 50
    
    U_vals, res1, res2 = scan_qmd_vs_U(U_lim=U_lim, n_U=n_U)
    
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    save_path = data_dir / "task2_qmd_scan_U.npz"
    np.savez(save_path, U_vals=U_vals, sigma_branch1=res1, sigma_branch2=res2, 
             mu=config.mu_eff, T=config.T_real, N_PTS=config.N_PTS)
    print(f"Results saved to {save_path}")
    
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(U_vals, res1[:, 0, 0, 0], label=r'Branch 1 ($E_{0, 1dn} = 0.05$)', linewidth=2)
    plt.plot(U_vals, res2[:, 0, 0, 0], label=r'Branch 2 ($E_{0, 1dn} = -0.05$)', linewidth=2)
    plt.xlabel(r'Interlayer Potential $U$ (eV)', fontsize=14)
    plt.ylabel(r'$\sigma_{xxx}$ (units)', fontsize=14)
    plt.title(f'QMD Conductivity vs U (mu={config.mu_eff} eV, T={config.T_real} K)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plot_path = figures_dir / "task2_qmd_scan_U.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
