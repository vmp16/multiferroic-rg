"""
model.analysis — physics analysis package for rhombohedral graphene.

Sub-modules
-----------
utils         : progress_bar
kspace        : k-mesh, Fermi distribution, system construction
thermo        : DOS, flavor densities, kinetic energies
interactions  : SU(4) and anisotropic interaction potentials
meanfield     : self-consistent mean-field (SCF) solvers
geometry      : velocity matrix elements, QGT, QM, Berry curvature, AHE, QMD
montecarlo    : Monte Carlo ground-state search
tables        : density and kinetic energy tables construction
phases        : SCF from tables, flavor counting/classification

All public symbols are re-exported here so that existing code using
``from model.analysis import <name>`` continues to work unchanged.
"""

from .utils        import *
from .kspace       import *
from .thermo       import *
from .interactions import *
from .meanfield    import *
from .geometry     import *
from .montecarlo   import *
from .tables       import *
from .phases       import *
from .pockets      import *
