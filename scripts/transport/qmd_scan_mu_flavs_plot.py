import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import plot_flavors from the calculation script
from scripts.qmd_scan_mu_flavs import plot_flavors

def load_qmd_flavors(filename='scan_nlt_qmd_flavors.npz'):
    data_path = config.DATA_DIR / filename
    config.FIGURES_DIR.mkdir(exist_ok=True)
    
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run scan_nlt_qmd_flavors.py first.")
        return

    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    mu_vals = data['mu_vals']
    contribs = data['contributions']
    labels = [r"$K\uparrow$", r"$K\downarrow$", r"$K'\uparrow$", r"$K'\downarrow$"]   # data['labels']

    # contribs should have shape (n_mu, 4, 2, 2, 2)
    print(f"Data loaded. Contributions shape: {contribs.shape}")
    
    return mu_vals, contribs, labels

if __name__ == "__main__":
    mu_vals, contribs, labels = load_qmd_flavors()
    plot_flavors(mu_vals, contribs, labels)
