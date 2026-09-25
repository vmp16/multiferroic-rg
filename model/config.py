import numpy as np
from pathlib import Path

# Define data and figures paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / 'figures'
DATA_DIR = PROJECT_ROOT / 'data'

# Lattice constants
a = 2.46          # Lattice constants [Å]
area_uc = (np.sqrt(3) / 2) * (a * 1e-8)**2  # unit cell area [cm^2]
unit_cell_to_cm2 = 1.0 / area_uc

# Hopping parameters for ABC Graphene, in eV
# Values from Huang's Suppl. Mat.
GAMMA0 = 3.16       # In-layer hopping
GAMMA1 = 0.380      # Nearest-layer vertical hopping
GAMMA2 = -0.015    # Next-nearest-layer hopping
GAMMA3 = -0.290     # Trigonal warping hopping
GAMMA4 = -0.141     # Second-order hopping

# Number of layers
N = 5

# Numerical Parameters
N_PTS = 6000
K_LIM = 0.2

# Distribution function parameters
T_real = 4
kB = 8.617e-5
T_eff = kB * T_real
mu_eff = 0.0

# Spin-valley polarization parameters
# Valley indices
VALLEY_IDX = [1, -1]

# Gaps
DELTA1UP = 0.0
DELTA1DN = 0.0  # before used 0.01 = 10 meV
DELTA2UP = DELTA1DN
DELTA2DN = DELTA1UP
DELTAS = np.array([[DELTA1UP, DELTA1DN],
                   [DELTA2UP, DELTA2DN]])

# On-site energies [eV]
E0_1UP = 9.3248*1e-3
E0_1DN = 0.0*1e-3
E0_2UP = 0.0*1e-3
E0_2DN = -14.4855*1e-3
E0_ARRAY = np.array([[E0_1UP, E0_1DN],
                     [E0_2UP, E0_2DN]])

# Interaction energy amplitudes
U = 30.0              # eV
J = 0.0 # - 0.3 * U       # eV