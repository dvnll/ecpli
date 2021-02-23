import gammapy
if not gammapy.__version__ == "0.16":
    raise ImportError("Currently only supporting gammapy v-0.16!")

from ecpli.MCMC import UniformPriorEnsembleMCMC, WeakPriorEnsembleMCMC
from ecpli.Bootstrap import BestFitParametricBootstrap, PoissonParametricBootstrap, NonParametricBootstrap
from ecpli.ProfileLikelihoodRatio import ConstrainedLR, UnconstrainedLR, LRBase


VERSION = (0, 1, 0)

__version__ = ".".join([str(x) for x in VERSION])
