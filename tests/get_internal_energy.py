"""
Calculate the total internal energy for a given configuration.
"""

import numpy as np

from model.model import McCannCarts
import model.config as config
from model.analysis import V_int_anisotropic

systems = [
    McCannCarts(
        N=config.N, valley_idx=xi, Delta=delta,
        gamma0=config.GAMMA0, gamma1=config.GAMMA1, gamma2=config.GAMMA2,
        gamma3=config.GAMMA3, gamma4=config.GAMMA4
    )
    for xi, delta in zip(2*config.VALLEY_IDX, config.DELTAS.T.flatten())
]

n_vec = ()

