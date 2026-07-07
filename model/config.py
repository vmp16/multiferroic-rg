import numpy as np

# Lattice constant
a = 2.46          # [Å]

# Hopping parameters for ABC Graphene, in eV
# GAMMA0 = 3.16
# GAMMA1 = 0.39
# GAMMA2 = -0.02
# GAMMA3 = 0.315
# GAMMA4 = 0.044

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
VALLEY_IDX = [1,-1]

# Gaps
DELTA1UP = 0.0
DELTA1DN = 0.01
DELTA2UP = DELTA1DN
DELTA2DN = DELTA1UP
DELTAS = np.array([[DELTA1UP, DELTA1DN],
                   [DELTA2UP, DELTA2DN]])

# On-site energies
E0_1UP = -0.005
E0_1DN = 0.0
E0_2UP = E0_1DN
E0_2DN = -E0_1UP
E0_ARRAY = np.array([[E0_1UP, E0_1DN],
                     [E0_2UP, E0_2DN]])