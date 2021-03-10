from gammapy.modeling.models import Models, SkyModel
from gammapy.modeling import Dataset

import astropy.units as u
from abc import ABC, abstractmethod
from copy import deepcopy
import logging


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def _mcrab():
    """Crab flux at 1 TeV defined as in the HESS Crab Paper,
       https://arxiv.org/abs/astro-ph/0607333, Table 6
    """

    _crab_unit = 3.84e-11 * u.Unit("cm-2 s-1 TeV-1")
    return u.def_unit("mCrab", _crab_unit / 1000.)


mCrab = _mcrab()


class LimitTarget(object):
    """Class to specify the parameter for which to derive a limit.

        Attributes:
            parameter_name: Name of the parameter for which to derive a limit.
            model: Model where the parameter is defined.
            parmin: Minimal value of the parameter.
            parmax: Maximal value of the parameter.
    """
    def __init__(self, model: SkyModel,
                 parameter_name: str,
                 parmin: float,
                 parmax: float):

        self.model = model
        self.parameter_name = parameter_name
        self.parmin = parmin
        self.parmax = parmax

        assert parmin < parmax

        if parameter_name not in model.parameters.names:
            info = "Target parameter " + parameter_name
            info += " is not in the model: " + str(model.parameters.names)
            raise RuntimeError(info)


class ECPLiBase(ABC):
    """All methods which implement a method to derive an upper limit on the
       ECPL parameter lambda (=inverse energy cutoff) are to be derived from
       this class.

       Attributes:
            limit_target: Description of the parameter for which a limit is to
                          be derived.
            data: Actuall gamma-ray data in form of a gammapy.modeling.Dataset.
                  This can in practice e.g. be a MapDataset (3d-analysis)
                  or a SpectrumDataset (1d-analysis).
            models: Collection of models used to describe the data.
            CL: Confidence level on which to work.
            ul: Upper limit on the parameter speficied by limit_target at the
                confidence level given in the class constructor.
    """
    def __init__(self,
                 limit_target: LimitTarget,
                 data: Dataset,
                 models: Models,
                 CL: float):

        self.limit_target = limit_target
        self.data = data
        self.data.models = None
        self.models = models
        self.CL = CL
        self._logger = logging.getLogger(__name__)

        if limit_target.model.name not in models.names:
            info = "Cannot find model " + limit_target.model.name
            info += " in model list: " + str(models.names)
            raise RuntimeError(info)

    @property
    @abstractmethod
    def ul(self) -> float:
        raise NotImplementedError("Must be implemented in derived classes.")

    def copy(self):
        """Deep copy of this instance.
        """
        return deepcopy(self)
