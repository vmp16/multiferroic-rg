import numpy as np

# Physical parameters for the McCann Model in eV
GAMMA0 = 3.16     # In-layer hopping
GAMMA1 = 0.39     # Nearest-layer vertical hopping
GAMMA2 = -0.02    # Next-nearest-layer hopping
GAMMA3 = 0.315    # Trigonal warping hopping
GAMMA4 = 0.044    # Second-order hopping
N = 5             # Number of layers

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
DELTA1UP = 0.1
DELTA1DN = 0.0
DELTA2UP = DELTA1DN
DELTA2DN = DELTA1UP
DELTAS = np.array([[DELTA1UP, DELTA1DN],
                   [DELTA2UP, DELTA2DN]])

# On-site energies
E0_1UP = 0.0
E0_1DN = 0.05
E0_2UP = -E0_1DN
E0_2DN = E0_1UP
E0_ARRAY = np.array([[E0_1UP, E0_1DN],
                     [E0_2UP, E0_2DN]])