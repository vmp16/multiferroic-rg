import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_ahe():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "scan_ahe_vs_delta.npz"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run scan_ahe_vs_delta.py first.")
        return

    data = np.load(data_path)
    delta_vals = data['delta_vals']
    sigmas_xy = data['sigmas_xy']
    
    plt.figure(figsize=(8, 5))
    plt.plot(delta_vals, sigmas_xy)
    plt.xlabel(r'$\Delta$ (eV)')
    plt.ylabel(r'$\sigma$')

    plt.show()

if __name__ == "__main__":
    plot_ahe()