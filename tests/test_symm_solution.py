import numpy as np
from scipy.optimize import brentq

from model.model import McCannCarts
import model.config as config
from model.analysis import get_kmesh, fermi_distrib, get_flavor_kinetic_energy_from_bands

def find_E_kin_flav(n_flav_target, mu_min=-0.1, mu_max=0.1):
    KX, KY = get_kmesh(config.K_LIM, config.N_PTS)
    dk = KX[0, 1] - KX[0, 0]
    prefactor = (dk**2) / (2 * np.pi)**2

    system = McCannCarts(
                N=config.N, valley_idx=1, Delta=0,
                gamma0=config.GAMMA0, gamma1=config.GAMMA1,
                gamma2=config.GAMMA2, gamma3=config.GAMMA3,
                gamma4=config.GAMMA4, E0=0
            )

    E0, E1 = system.get_energy(KX, KY)

    def objective_function(mu_flav):
        n_e = fermi_distrib(E0, mu_flav, config.T_eff)
        n_h = 1.0 - fermi_distrib(E1, mu_flav, config.T_eff)

        n_flav = prefactor * np.sum(n_e - n_h) * config.unit_cell_to_cm2

        return n_flav - n_flav_target

    try:
        mu_flav_root = brentq(objective_function, mu_min, mu_max, xtol=1e-5)
    except:
        print(f"Error: The target density {n_flav_target} is outside the bracket [{mu_min}, {mu_max}].")
        raise

    E_kin = get_flavor_kinetic_energy_from_bands((E0, E1), mu_flav_root, config.T_eff, prefactor, config.unit_cell_to_cm2)

    return mu_flav_root, E_kin


def main():
    n_flav_target = -3.0e11 / 4

    mu_solved, E_kin = find_E_kin_flav(n_flav_target)
    print(f"\n   Target Flavor Density: {n_flav_target:.4e} / cm2")
    print(f"   Calculated Flavor Fermi level: {mu_solved*1e3:.4f} meV")
    print(f"   Calculated Flavor Kinetic Energy: {E_kin:.4e} eV / cm2")

if __name__ == "__main__":
    main()