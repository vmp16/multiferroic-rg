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
phases        : phase diagram tables, SCF from tables, flavor counting/classification

All public symbols are re-exported here so that existing code using
``from model.analysis import <name>`` continues to work unchanged.
"""

from .utils        import *   # noqa: F401,F403
from .kspace       import *   # noqa: F401,F403
from .thermo       import *   # noqa: F401,F403
from .interactions import *   # noqa: F401,F403
from .meanfield    import *   # noqa: F401,F403
from .geometry     import *   # noqa: F401,F403
from .montecarlo   import *   # noqa: F401,F403
from .phases       import *   # noqa: F401,F403
