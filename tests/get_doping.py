import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from model.analysis import get_kmesh, fermi_distrib
from model.model import McCannCarts
from model import config


def calculate_filling_factor(mu=0.0):
    """Calculate the filling factor at the given chemical potential using the Fermi distribution."""
    KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
    px, py = KX, KY

    system = McCannCarts(
        N=config.N,
        valley_idx=config.VALLEY_IDX[0],
        Delta=config.DELTAS[0, 0],
        gamma0=config.GAMMA0,
        gamma1=config.GAMMA1,
        gamma2=config.GAMMA2,
        gamma3=config.GAMMA3,
        gamma4=config.GAMMA4,
        E0=config.E0_ARRAY[0, 0],
    )

    E0, E1 = system.get_energy(px, py)
    f0 = fermi_distrib(E0, mu, config.T_eff)
    f1 = fermi_distrib(E1, mu, config.T_eff)

    dk = px[0, 1] - px[0, 0]
    prefactor = (dk**2) / (2 * np.pi)
    filling_factor = prefactor * np.sum(f0 + f1)

    return filling_factor


if __name__ == "__main__":
    filling_factor = calculate_filling_factor(mu=0.0)
    print(f"Filling factor at mu=0: {filling_factor:.12f}")