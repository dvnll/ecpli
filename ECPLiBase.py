from gammapy.modeling.models import Models, SkyModel
from gammapy.cube import MapDataset
import astropy.units as u
from copy import deepcopy


_crab_unit = 3.84e-11 * u.Unit("cm-2 s-1 TeV-1") # Crab flux at 1 TeV defined as in the HESS Crab Paper, Table 6
mCrab = u.def_unit("mCrab", _crab_unit / 1000.)


class LimitTarget(object):
    def __init__(self, model: SkyModel, parameter_name: str):
        self.model = model
        self.parameter_name = parameter_name


        """
        assert parameter_name in model.parameters ...
        """
class ECPLiBase(object):

    def __init__(self,
                 limit_target: LimitTarget,
                 data: MapDataset,
                 models: Models,
                 CL: float):

        self.limit_target = limit_target
        self.data = data
        self.models = models
        self._fit_backend = "minuit"
        self.CL = CL

        if limit_target.model.name not in models.names:
            info = "Cannot find model " + limit_target.model.name
            info += " in model list: " + models.names
            raise RuntimeError(info)

    @property
    def fit_backend(self):
        return self._fit_backend

    @property
    def ul(self):
        raise NotImplementedError("Must be implemented in derived classes")

    def copy(self):
        return deepcopy(self)
