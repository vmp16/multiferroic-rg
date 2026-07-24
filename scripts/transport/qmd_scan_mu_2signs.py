import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc

from model.model import McCannCarts
from model.analysis import deriv_fermi_distrib, sym_decomp_cond
import model.config as config

def calculate_qmd_mu_scan(mu_vals, e0_dn, n_pts=6000, chunk_size=1000):
    print(f"\nScanning mu for Branch (E0_1DN = {e0_dn:.3f}, N_PTS={n_pts})...")
    
    kx_vals = np.linspace(-config.K_LIM, config.K_LIM, n_pts)
    ky_vals = np.linspace(-config.K_LIM, config.K_LIM, n_pts)
    dk = kx_vals[1] - kx_vals[0]
    prefactor = (dk**2) / (2 * np.pi)

    # Build local E0 array
    e0_1up = config.E0_1UP
    e0_1dn = e0_dn
    e0_2up = -e0_dn
    e0_2dn = e0_1up
    local_e0_array = np.array([[e0_1up, e0_1dn],
                               [e0_2up, e0_2dn]])

    # Results: (n_mu, 2, 2, 2)
    sigmas_total = np.zeros((len(mu_vals), 2, 2, 2))

    for j_start in range(0, n_pts, chunk_size):
        j_end = min(j_start + chunk_size, n_pts)
        print(f"  Chunk {j_start//chunk_size + 1}/{int(np.ceil(n_pts/chunk_size))}...", flush=True)
        KX_c, KY_c = np.meshgrid(kx_vals, ky_vals[j_start:j_end])
        
        for v_idx in range(2):
            for s_idx in range(2):
                print(f"    Flavor {v_idx*2 + s_idx + 1}/4...", flush=True)
                xi = config.VALLEY_IDX[v_idx]
                delta = config.DELTAS[v_idx, s_idx] # U=0
                e0 = local_e0_array[v_idx, s_idx]
                
                system = McCannCarts(
                    N=config.N, valley_idx=xi, Delta=delta,
                    gamma0=config.GAMMA0, gamma1=config.GAMMA1,
                    gamma2=config.GAMMA2, gamma3=config.GAMMA3,
                    gamma4=config.GAMMA4, E0=e0
                )
                
                # 1. Energies
                E0, E1 = system.get_energy(KX_c, KY_c)
                dE_sq = (E0 - E1)**2
                
                # 2. Eigenstates
                psi0_1, psi0_2 = system.get_eigenstate_components(KX_c, KY_c, 0)
                psi1_1, psi1_2 = system.get_eigenstate_components(KX_c, KY_c, 1)
                
                # 3. Hamiltonian derivatives
                dX_dx = system.derivate_h0_at_k(KX_c, KY_c, 1) # dh0 part
                dX_dy = system.derivate_h0_at_k(KX_c, KY_c, 0) # dh0 part
                # Wait, I need velocity elements <0|v|1>. McCannCarts.derivate_h0_at_k is dh0/dk.
                # McCannCarts.derivate_X_at_k is dX/dk.
                dX_dx_full = system.derivate_X_at_k(KX_c, KY_c, 1)
                dX_dy_full = system.derivate_X_at_k(KX_c, KY_c, 0)
                
                # <0|v_x|1> = psi0* . [dh0/dx*I + dX/dx*sigma_x + ...] . psi1
                # But interband dh0 part is zero because <0|1>=0.
                # v_01 = psi0_1* * dX_dk * psi1_2 + psi0_2* * conj(dX_dk) * psi1_1
                v_x_01 = np.conj(psi0_1) * dX_dx_full * psi1_2 + np.conj(psi0_2) * np.conj(dX_dx_full) * psi1_1
                v_y_01 = np.conj(psi0_1) * dX_dy_full * psi1_2 + np.conj(psi0_2) * np.conj(dX_dy_full) * psi1_1
                
                # Metric components
                g_xx = np.abs(v_x_01)**2 / dE_sq
                g_yy = np.abs(v_y_01)**2 / dE_sq
                g_xy = np.real(v_x_01 * np.conj(v_y_01)) / dE_sq
                
                del v_x_01, v_y_01, dE_sq, psi0_1, psi0_2, psi1_1, psi1_2, dX_dx_full, dX_dy_full
                
                # Gradients
                dgxx_dx = np.gradient(g_xx, dk, axis=1)
                dgxx_dy = np.gradient(g_xx, dk, axis=0)
                dgxy_dx = np.gradient(g_xy, dk, axis=1)
                dgxy_dy = np.gradient(g_xy, dk, axis=0)
                dgyy_dx = np.gradient(g_yy, dk, axis=1)
                dgyy_dy = np.gradient(g_yy, dk, axis=0)
                
                del g_xx, g_yy, g_xy
                
                # Mu loop (vectorized)
                for m_idx, mu in enumerate(mu_vals):
                    df_sum = deriv_fermi_distrib(E0, mu, config.T_eff) + deriv_fermi_distrib(E1, mu, config.T_eff)
                    
                    sigmas_total[m_idx, 0, 0, 0] += prefactor * np.sum(df_sum * dgxx_dx) # xxx
                    sigmas_total[m_idx, 1, 0, 0] += prefactor * np.sum(df_sum * dgxx_dy) # yxx
                    sigmas_total[m_idx, 0, 0, 1] += prefactor * np.sum(df_sum * dgxy_dx) # xxy
                    sigmas_total[m_idx, 0, 1, 0] += prefactor * np.sum(df_sum * dgxy_dx) # xxy (sym)
                    sigmas_total[m_idx, 1, 0, 1] += prefactor * np.sum(df_sum * dgxy_dy) # yxy
                    sigmas_total[m_idx, 1, 1, 0] += prefactor * np.sum(df_sum * dgxy_dy) # yxy (sym)
                    sigmas_total[m_idx, 0, 1, 1] += prefactor * np.sum(df_sum * dgyy_dx) # xyy
                    sigmas_total[m_idx, 1, 1, 1] += prefactor * np.sum(df_sum * dgyy_dy) # yyy
                    
                del E0, E1, dgxx_dx, dgxx_dy, dgxy_dx, dgxy_dy, dgyy_dx, dgyy_dy
        
        del KX_c, KY_c
        gc.collect()
        
    # Return symmetric part
    sigmas_sym = np.zeros_like(sigmas_total)
    for m in range(len(mu_vals)):
        s_sym, _ = sym_decomp_cond(sigmas_total[m])
        sigmas_sym[m] = s_sym
        
    return sigmas_sym

def main():
    config.T_real = 4
    config.T_eff = config.kB * config.T_real
    config.N_PTS = 6000
    
    mu_vals = np.linspace(-0.17, 0.17, 300)
    
    # Branch 1: E0_DN = 0.05
    res1 = calculate_qmd_mu_scan(mu_vals, 0.05, n_pts=config.N_PTS)
    
    # Branch 2: E0_DN = -0.05
    res2 = calculate_qmd_mu_scan(mu_vals, -0.05, n_pts=config.N_PTS)
    
    # Save
    config.DATA_DIR.mkdir(exist_ok=True)
    save_path = config.DATA_DIR / "task3_qmd_mu_scan.npz"
    np.savez(save_path, mu_vals=mu_vals, sigma_branch1=res1, sigma_branch2=res2, 
             T=config.T_real, N_PTS=config.N_PTS)
    print(f"Results saved to {save_path}")
    
    # Plot xxx component
    config.FIGURES_DIR.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_vals, res1[:, 0, 0, 0], label=r'$E_{0, 1dn} = 0.05$ eV', linewidth=2)
    plt.plot(mu_vals, res2[:, 0, 0, 0], label=r'$E_{0, 1dn} = -0.05$ eV', linewidth=2, linestyle='--')
    plt.xlabel(r'$\mu$ (eV)', fontsize=14)
    plt.ylabel(r'$\sigma_{xxx}$ (units)', fontsize=14)
    plt.title(r'QMD Conductivity vs $\mu$ ($U=0$, $T=%dK$)' % config.T_real, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plot_path = config.FIGURES_DIR / "task3_qmd_mu_scan.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
