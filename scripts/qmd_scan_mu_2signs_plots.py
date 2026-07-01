import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Set math font to Computer Modern to properly render \varepsilon
plt.rcParams['mathtext.fontset'] = 'cm'


def load_task3_data(filename="task3_qmd_mu_scan.npz"):
    """Load the task 3 QMD scan data from the NPZ file."""
    data_path = project_root / "data" / filename

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run scripts/task3_qmd_mu_scan.py first to generate the data.")
        return None

    print(f"Loading data from {data_path}...")
    with np.load(data_path) as data:
        return {
            "path": data_path,
            "mu_vals": data["mu_vals"],
            "sigma_branch1": data["sigma_branch1"],
            "sigma_branch2": data["sigma_branch2"],
            "T": data["T"] if "T" in data else 4,
        }


def build_task3_plot(task3_data, output_path=None):
    """Build the task 3 sigma_xxx vs mu plot from loaded data."""
    if task3_data is None:
        return None

    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)

    if output_path is None:
        output_path = figures_dir / "task3_qmd_mu_scan_replot.png"

    mu_vals = task3_data["mu_vals"]
    res1 = task3_data["sigma_branch1"]
    res2 = task3_data["sigma_branch2"]
    T = task3_data["T"]

    plt.figure(figsize=(8, 6))

    plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.plot(mu_vals, res1[:, 0, 0, 0], label=r'$\epsilon_0=0.05$', linewidth=2, color='darkorange')
    plt.plot(mu_vals, res2[:, 0, 0, 0], label=r'$\epsilon_0=-0.05$', linewidth=2, color='teal')

    plt.ylim(-280, 280)
    plt.xlim(-0.14, 0.14)

    plt.xlabel(r'$\mu$ ($eV$)', fontsize=19)
    plt.ylabel(r'$\sigma_{xxx}$ ($(e^2/h)(a/V)$)', fontsize=19)
    # plt.title(r'Quantum Metric Dipole Conductivity vs $\mu$ ($U=0$, $T=%d$K)' % T, fontsize=15)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', labelsize=17)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Successfully re-plotted. Figure saved to {output_path}")
    plt.show()
    return output_path


def replot_task3():
    data_path = project_root / "data" / "task3_qmd_mu_scan.npz"
    task3_data = load_task3_data(data_path)
    if task3_data is None:
        return
    build_task3_plot(task3_data)


if __name__ == "__main__":
    replot_task3()
