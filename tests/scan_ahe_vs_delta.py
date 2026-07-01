import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.model import McCannCarts
from model.analysis import get_kmesh, get_ahe
import model.config as config

def scan_delta(delta_lim=0.5, n_delta=100):
    """
    Scan the total Anomalous Hall conductivity as a function of the gap Delta.
    TRS is explicitly broken by setting Delta(K, up) = -Delta(K', down) = d_val.
    """
    print(12*"=" + f" SCANNING TOTAL AHE VS DELTA = [{-delta_lim}, {delta_lim}] " + 12*"=")

    delta_vals = np.linspace(-delta_lim, delta_lim, n_delta)

    # Build the kmesh
    KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
    p = np.stack((KX, KY), axis=-1)

    sigma_list = []

    print(f"Starting scan for {n_delta} points...")
    for i, d_val in enumerate(delta_vals):
        print(f"Progress: {100 * (i + 1) / n_delta:.1f}% (Delta={d_val:.4f})", end='\r')
        
        total_sigma_xy = 0.0
        
        # 4 Flavors: (Valley index xi, Spin index s_idx, Gap delta, On-site E0)
        # TRS is broken by setting Delta(K, up) = -Delta(K', down)
        # Other flavors (K, down) and (K', up) remain ungapped.
        flavor_configs = [
            (1,  0, d_val,  config.E0_ARRAY[0, 0]),   # K, Up
            (1,  1, 0.0,    config.E0_ARRAY[0, 1]),   # K, Down
            (-1, 0, 0.0,    config.E0_ARRAY[1, 0]),   # K', Up
            (-1, 1, -d_val, config.E0_ARRAY[1, 1])    # K', Down
        ]
        
        for xi, s_idx, delta, e0 in flavor_configs:
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
            
            # Sum intrinsic AHE contribution from both bands
            for band_idx in [0, 1]:
                total_sigma_xy += get_ahe(system, band_idx, p, config.T_eff, config.mu_eff)
                
        sigma_list.append(total_sigma_xy)

    print("\nScan completed.")
    return delta_vals, np.array(sigma_list)

def save_results(delta_vals, sigmas_xy, filename="scan_ahe_vs_delta.npz"):
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    save_path = data_dir / filename
    
    # Metadata for reference
    metadata = {
        "GAMMA0": config.GAMMA0,
        "GAMMA1": config.GAMMA1,
        "mu_eff": config.mu_eff,
        "T_eff": config.T_eff,
        "N": config.N,
        "N_PTS": config.N_PTS,
        "K_LIM": config.K_LIM,
    }

    np.savez(save_path, delta_vals=delta_vals, sigmas_xy=sigmas_xy, **metadata)
    print(f"Results saved to {save_path}")

def main():
    # Perform the scan. delta_lim=0.1 might be a good range to see the evolution near mu.
    delta_vals, sigmas_xy = scan_delta(delta_lim=-0.07, n_delta=50)
    save_results(delta_vals, sigmas_xy)

if __name__ == "__main__":
    main()
