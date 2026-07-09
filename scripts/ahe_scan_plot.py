import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Set math font to Computer Modern to properly render \varepsilon
plt.rcParams['mathtext.fontset'] = 'cm'

def plot_ahe_vs_U(filename="scan_ahe_vs_U.npz"):
    """
    Load and plot the Anomalous Hall conductivity vs Interlayer Potential U.
    """
    data_path = project_root / "data" / filename
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found. Run the scan script first.")
        return

    # Load data
    data = np.load(data_path)
    U_vals = data['U_vals']
    sigmas_s1 = data['sigmas_s1']
    sigmas_s2 = data['sigmas_s2']
    mu_eff = data['mu_eff']

    print(f"Plotting data from {filename}...")

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(U_vals*1e3, sigmas_s1, 'o-', color='darkorange', linewidth=1.5, markersize=7, label=r'$\epsilon_0=5~meV$')
    plt.plot(U_vals*1e3, sigmas_s2, 's-', color='teal', linewidth=1.5, markersize=7, label=r'$\epsilon_0=-5~meV$')
    
    # Baseline for zero
    # plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Styling
    # plt.title(f'Total Anomalous Hall Conductivity vs Interlayer Potential $U$\n($\\mu_{{eff}} = {mu_eff:.4f}$ eV)', fontsize=17)
    plt.xlabel(r'$U~(meV)$', fontsize=20)
    plt.ylabel(r'$\sigma_{xy}$ ($e^2/h$)', fontsize=21)
    plt.legend(fontsize=19)
    plt.xticks(fontsize=17) # [-0.01, -0.005, 0.0, 0.005, 0.01]
    plt.yticks(fontsize=17) # [-0.3, 0.0, 0.3]

    plt.tight_layout()

    # Save figure
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_filename = f"plot_{Path(filename).stem}.png"
    plot_path = figures_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"   Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_ahe_vs_U(filename="scan_ahe_vs_U.npz")
