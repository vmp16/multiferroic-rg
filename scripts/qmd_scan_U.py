import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path
import gc

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, fermi_distrib, get_QGT_chunk, get_qmd_from_qgt, integrate_qmd_chunk
import model.config as config

def find_mu_for_fixed_n(U_val, n_target, n_pts_light=600, mu_min=-0.1, mu_max=0.1):
    # Build light kmesh
    KX, KY = get_kmesh(config.K_LIM, n_pts_light)
    dk = KX[0, 1] - KX[0, 0]
    prefactor = (dk**2) / (2 * np.pi)**2

    # Build system and precompute energies for this U value
    all_flavs_bands = []
    for v_idx, xi in enumerate(config.VALLEY_IDX):
        for s_idx in range(2):
            e0 = config.E0_ARRAY[v_idx, s_idx]
            intrinsic_delta = config.DELTAS[v_idx, s_idx]
            delta_eff = intrinsic_delta + U_val

            system = McCannCarts(
                N=config.N, valley_idx=xi, Delta=delta_eff,
                gamma0=config.GAMMA0, gamma1=config.GAMMA1,
                gamma2=config.GAMMA2, gamma3=config.GAMMA3,
                gamma4=config.GAMMA4, E0=e0
            )

            # Get energies and sotre them
            E0, E1 = system.get_energy(KX, KY)
            all_flavs_bands.append((E0, E1))

    # Root finder
    def objective_function(mu):
        n_flav = 0
        for E0, E1 in all_flavs_bands:
            n_e = fermi_distrib(E0, mu, config.T_eff)
            n_h = 1.0 - fermi_distrib(E1, mu, config.T_eff)

            n_flav += prefactor * np.sum(n_e - n_h)

        n_total = n_flav / (config.a**2) * 1e16

        return n_total - n_target

    # Find the root -> mu
    try:
        mu_root = brentq(objective_function, mu_min, mu_max, xtol=1e-5)
        return mu_root
    except ValueError:
        print(f"Error: The target density {n_target} is outside the bracket [{mu_min}, {mu_max}].")
        raise

def scan_U(U_vals, n_fixed, chunk_size=1000):
    U_lim = U_vals[-1]
    n_U = len(U_vals)
    print(15*"=" + F" SCANNING QMD'S NLT FOR U = [{-U_lim}, {U_lim}] AND {n_U} POINTS " + 15*"=")

    print(f"Resolving Fermi levels for constant particle density...")
    mu_vals = []
    for U in U_vals:
        mu_at_U = find_mu_for_fixed_n(U, n_fixed)
        mu_vals.append(mu_at_U)

    print(f"\nRunning QMD Conductivity calculation...")

    # Build the 1D k-arrays
    n_pts = config.N_PTS
    kx_vals = np.linspace(-config.K_LIM, config.K_LIM, n_pts)
    ky_vals = np.linspace(-config.K_LIM, config.K_LIM, n_pts)

    # Define integration prefactors
    dk = kx_vals[1] - kx_vals[0]
    prefactor = (dk**2) / (2 * np.pi)

    # Initialize the total conductivity accumulator array
    sigmas_total = np.zeros((len(U_vals), 2, 2, 2))

    for i, (U_val, mu_val) in enumerate(zip(U_vals, mu_vals)):
        print(f"\rProgress: {100 * (i + 1) / n_U:.1f}% (U={U_val:.4f})", end="", flush=True)

        for v_idx in range(2):
            for s_idx in range(2):
                xi = config.VALLEY_IDX[v_idx]
                e0 = config.E0_ARRAY[v_idx, s_idx]
                intrinsic_delta = config.DELTAS[v_idx, s_idx]
                delta_eff = intrinsic_delta + U_val
                
                # print(f"\nInitializing system for Flavor: valley={xi}, spin={s_idx}...")
                system = McCannCarts(
                    N=config.N, valley_idx=xi, Delta=delta_eff,
                    gamma0=config.GAMMA0, gamma1=config.GAMMA1,
                    gamma2=config.GAMMA2, gamma3=config.GAMMA3,
                    gamma4=config.GAMMA4, E0=e0
                )

                # Inner Loop: Iterate over kmesh chunks along the y-axis
                for j_start in range(0, config.N_PTS, chunk_size):
                    j_end = min(j_start + chunk_size, config.N_PTS)
                    # print(f"  Processing Chunk {j_start//chunk_size + 1}/{int(np.ceil(n_pts/chunk_size))}...", flush=True)
                    
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
                        sigmas_total[i] += integrate_qmd_chunk(
                            qmd_chunk, 
                            band_energies[band_idx], 
                            np.array([mu_val]), 
                            config.T_eff, 
                            prefactor
                        )[0]

                        # Clear memory for this band's chunk calculation
                        del T_chunk, qmd_chunk
                    
                    # Clean up chunk mesh and energies
                    del KX_c, KY_c, E0, E1, band_energies
                    gc.collect()

    return sigmas_total

def save_results(U_vals, sigmas_vals, n_fixed, filename="qmd_scan_U.npz"):
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
        "n_fixed": n_fixed
    }
    
    np.savez(save_path, U_vals=U_vals, sigmas_vals=sigmas_vals, **metadata)
    print(f"Results saved to {save_path}")

def main():
    U_lim = 0.02        # eV
    n_U = 300
    U_vals = np.linspace(-U_lim, U_lim, n_U)

    n_fixed = -3e11     # /cm^2

    sigmas_total = scan_U(U_vals, n_fixed, chunk_size=1000)
    save_results(U_vals, sigmas_total, n_fixed)


if __name__ == "__main__":
    main()