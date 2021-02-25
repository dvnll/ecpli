import numpy as np

import gammapy.modeling as modeling
from gammapy.datasets import Dataset

from scipy.optimize import brentq, fsolve
import scipy.stats as stats

import random
import string

from typing import List

from ecpli.ECPLiBase import ECPLiBase, LimitTarget


class _ProfileLRPool(object):
    """Helper class to contain the profile likelihood values
       and their corresponding models.

       Attributes:
        max_size: Maximal size of the pool. Limited to limit used memory.
        full: True if the maximum size of the pool is reached.
        parameter_list: List of parameters for the profile.
        likelihood_list: List of maximum profile likelihood values.

       Todo:
        This class is useful to find a good start model for fits and
        to perform plots of the profile likelihood function. However,
        it is very memory inefficient. One way to improve it would be
        to not save the complete list of models but only their parameters.
    """

    def __init__(self, max_size: int):
        self._parameter_list: List[float] = []
        self._likelihood_list: List[float] = []
        self._model_list: List[Dataset] = []
        self.max_size: int = max_size

    @property
    def full(self) -> bool:
        return (len(self) >= self.max_size)

    def __len__(self) -> int:
        return len(self._parameter_list)

    def append(self, parameter: float, dataset: Dataset) -> None:
        """Appends the profiled fit model and the parameter to the pool.

           Args:
            parameter: Fixed profile parameter.
            dataset: Optimized dataset for the given parameter.
        """
        if parameter in self.parameter_list:
            return

        self._parameter_list.append(parameter)
        self._likelihood_list.append(dataset.stat_sum())
        self._model_list.append(dataset.models)

    @property
    def _index_list(self) -> List[int]:
        return np.argsort(self._parameter_list)

    @property
    def parameter_list(self) -> List[str]:
        return np.array(self._parameter_list)[self._index_list]

    @property
    def likelihood_list(self) -> List[float]:
        return np.array(self._likelihood_list)[self._index_list]

    def ml_model(self) -> modeling.models.Models:
        """Returns the list of Skymodels which maximize the likelihood
           function (=minimize the negative loglikeihood) along the
           available parameter profile.
        """

        ml_index = np.argmin(self._likelihood_list)
        ml_model = self._model_list[ml_index]
        temp_model = []
        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))

        for model in ml_model:
            name = model.name.split("_A_")[0]
            temp_model.append(model.copy(name=name + "_A_" + random_string))

        return temp_model

    def closest_model(self, parameter: float) -> modeling.models.Models:
        """Returns the list of SkyModels in the pool which is optimized for the
           closest available profile parameter value in the pool.
           This is very useful to initialize ML fits for ugly profile
           likelihood functions.

           Args:
            parameter: Profile parameter for which the closest model is
                     requested.
        """

        best_index = np.argmin(
                        np.abs(np.array(self._parameter_list) - parameter))
        closest_model = self._model_list[best_index]

        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))
        temp_model = []
        for model in closest_model:
            name = model.name.split("_A_")[0]
            temp_model.append(model.copy(name=name + "_A_" + random_string))

        closest_model = temp_model
        return closest_model


class LRBase(ECPLiBase):
    """Base class to invert the profile likelihood ratio test to derive
       limits on the limit_target methods to.

       Attributes:
        fit_config: Dictionary which describes the parameters of the
                    likelihood fit optimization.
        data: Dataset from which a limit is to be derived.
        model: Likelihood fit optimization start model.
        n_fits_performed: Number of fits performed in this instance.
    """

    def __init__(self, limit_target: LimitTarget,
                 data: Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 fit_config: dict,
                 max_pool_entries=20):

        super().__init__(limit_target, data, models, CL)

        self._n_fits = 0
        self.fit_config = fit_config
        self._fitstart_model = models

        def initialize_lr_pool() -> None:
            """Initializes the profile likelihood lookup pool and fill it.
            """
            self._ml_fit_pool = _ProfileLRPool(max_pool_entries)

            for parameter in np.linspace(self.limit_target.parmin,
                                         self.limit_target.parmax,
                                         max_pool_entries):

                info = "Filling ML fit pool: Current size: "
                info += str(len(self._ml_fit_pool)) + "/"
                info += str(max_pool_entries)
                self._logger.debug(info)

                self.fit(parameter_value=parameter)

        initialize_lr_pool()

    def _fitstart_model_copy(self) -> modeling.models.Model:
        """Returns a copy of self._fitstart_model with new random names for the
           components.
        """
        letters = string.ascii_uppercase + string.digits

        random_string = "".join(random.choice(letters) for _ in range(10))
        model_copy_list: List[modeling.SkyModel] = []
        for model in self._fitstart_model:
            original_model_name = model.name.split("_A")[0]
            new_model_name = original_model_name + "_A_" + random_string
            model_copy = model.copy(name=new_model_name)
            model_copy_list.append(model_copy)

        return modeling.models.Models(model_copy_list)

    def fit(self, parameter_value=None) -> (float, float):
        """Performs the likelihood optimization.

           Args:
            parameter_value: Limit target parameter value.
                             If None: Keep limit target parameter free in fit.
                             If fixed to a float: Fix limit target parameter to
                             the given value in the likelihood fit.
           Returns: (best_fit_parameter,
                     likelihood value at best fit parameter)
        """

        self._n_fits += 1

        def initialize_dataset_fit_model() -> None:
            """Puts the fit-start model in place in the dataset.
               If the ML fit pool is unfilled, the right fit-start model is
               a fresh copy of the fit-start model defined in the constructor.
               If no profile likelihood is requested, the right fit-start model
               is the ML fit model. If a profile likelihood model is requested,
               the right fit-start model is the model corresponding to the
               closest parameter in the ML fit pool.
            """
            dataset = self.data
            if len(self._ml_fit_pool) == 0:
                dataset.models = self._fitstart_model_copy()
            else:
                if parameter_value is None:
                    dataset.models = self._ml_fit_pool.ml_model()
                else:
                    dataset.models = self._ml_fit_pool.closest_model(
                                            parameter_value)
            return dataset

        def freeze_target_parameter(frozen: bool) -> None:
            """Controls whether the target parameter is frozen in the fit.

            Args:
                frozen: If true, the target parameter is frozen in the fit.
            """
            for model in dataset.models:
                original_model_name = model.name.split("_A")[0]
                if original_model_name != self.limit_target.model.name:
                    continue

                for parameter in model.parameters:
                    if parameter.name == self.limit_target.parameter_name:
                        if frozen:
                            parameter.frozen = True
                            parameter.value = parameter_value
                        else:
                            parameter.frozen = False
                        break

        def show_fit_dataset_debug_information() -> None:
            """Prints debug information if in corresponding log-level.
            """
            self._logger.debug("---------------------------------------------")
            self._logger.debug("Fit input dataset:")
            info = "parameter value input: " + str(parameter_value)
            if parameter_value is not None:
                info += "Target parameter " + self.limit_target.parameter_name
                info += ": -> should be frozen to this "
                info += "value in the following dataset"
            self._logger.debug(info)
            self._logger.debug(dataset)
            if hasattr(dataset, "background_model"):
                self._logger.debug("Background model:")
                self._logger.debug(dataset.background_model)

            self._logger.debug("Free parameters:")
            for par in dataset.models.parameters.free_parameters:
                info = par.name + ": value: " + str(par.value)
                info += ", min: " + str(par.min) + ", max: " + str(par.max)
                self._logger.debug(info)
            self._logger.debug("---------------------------------------------")

        def fit_dataset() -> modeling.fit.OptimizeResult:
            """Fits the dataset to its model.

               Returns: FitResult
            """

            fit = modeling.Fit([dataset])
            with np.errstate(all="ignore"):
                result = fit.run(
                    optimize_opts=self.fit_config["optimize_opts"])

                n_repeat_fit = 1
                while result.success is False:
                    if n_repeat_fit > self.fit_config["n_repeat_fit_max"]:
                        break
                    info = "Fit " + str(n_repeat_fit)
                    info += " didn't converge -> Repeating"
                    self._logger.debug(info)

                    n_repeat_fit += 1
                    fit = modeling.Fit([dataset])
                    optimize_opts = self.fit_config["optimize_opts"]
                    result = fit.run(
                        optimize_opts=optimize_opts)
                if result.success is False:
                    if parameter_value is None:
                        info = "Cannot fit full model with free target "
                        info += "parameter"
                        raise RuntimeError(info)
                    else:
                        info = "Fit failed for fixed parameter_value="
                        info += str(parameter_value)
                        info += " (this is typically no problem)"
                        self._logger.debug(info)
                return result

        def _best_fit_parameter(
                    fit_result: modeling.fit.OptimizeResult) -> float:
            """Extracts the best fit target parameter out of the fit result.
               Args:
                fit_result: Return value of fit_dataset()
               Returns: Best fitting parameter
            """

            best_fit_parameter = None

            if fit_result.success:
                self._logger.debug("Result:")
                self._logger.debug(fit_result)
                self._logger.debug("Parameters:")
                for name, value in zip(fit_result.parameters.names,
                                       fit_result.parameters.values):
                    self._logger.debug(str(name) + ": " + str(value))
                self._logger.debug("**++++++++++++++++++++++++++++++++++++**")

                for model in dataset.models:
                    if self.limit_target.model.name in model.name:
                        for parameter in model.parameters:
                            pname = parameter.name
                            if pname == self.limit_target.parameter_name:
                                best_fit_parameter = parameter.value
                                break

            return best_fit_parameter

        def print_afterfit_debug() -> None:
            """Prints debug information for the dataset state after the fit
               if in respective log-level.
            """

            self._logger.debug("+++++++++++++++++++++++++++++++++++++++++++++")
            self._logger.debug("Fit output dataset:")
            self._logger.debug(dataset)
            if hasattr(dataset, "background_model"):
                self._logger.debug("Background model:")
                self._logger.debug(dataset.background_model)
            self._logger.debug("---------------------------------------------")

            info = str(self._n_fits) + ": Fit result for input parameter="
            info += str(parameter_value) + " -> fitstat: " + str(fitstat)
            self._logger.debug(info)

        dataset = initialize_dataset_fit_model()

        if parameter_value is not None:
            freeze_target_parameter(frozen=True)
        else:
            freeze_target_parameter(frozen=False)

        show_fit_dataset_debug_information()

        fit_result = fit_dataset()
        best_fit_parameter = _best_fit_parameter(fit_result)
        if best_fit_parameter is not None and not self._ml_fit_pool.full:
            self._ml_fit_pool.append(best_fit_parameter, dataset)

        fitstat = dataset.stat_sum()
        print_afterfit_debug()

        dataset.models = None  # Clear models after each fit.
        return (best_fit_parameter, fitstat)

    def fitstat(self, parameter_value=None) -> float:
        """Returns optimized likelihood value.

           Args:
            parameter_value: Fix limit_target to this value when float.
                             Keep limit_target free in fit if None.
        """
        _, fitstat = self.fit(parameter_value=parameter_value)
        return fitstat

    @property
    def n_fits_performed(self) -> int:
        return self._n_fits

    def ts(self) -> float:
        """Returns the test statistic for the LR test of a PL against an ECPL.
        """

        if hasattr(self, "__ts"):
            return getattr(self, "__ts")

        ml_fit_parameter = self.ml_fit_parameter()
        fitstat0 = self.fitstat(parameter_value=ml_fit_parameter)

        fitstat_pl = self.fitstat(parameter_value=0.)
        info = "fitstat_pl=" + str(fitstat_pl)
        info += ", fitstat_ml=" + str(fitstat0)
        self._logger.debug(info)

        ts = fitstat_pl - fitstat0
        setattr(self, "__ts", ts)
        return ts

    @property
    def _ul(self) -> float:
        """Returns an upper limit. The method has two steps:
           First: Try to find a solution to
                  ProfileLikelihood(parameter)=critical_value(CL)
                  for the parameter in [ml_fit_parameter, test_value]
                  with the brentq method.
                  In an optimal world, one could just use
                  a very large test_value. But this leads to frequent
                  numerical problems. So, a range of test_values is tried
                  until a solution is found.
           Second: If no solution is found with brentq, fsolve is used.
        """

        critical_value = self.critical_value

        ml_fit_parameter = self.ml_fit_parameter()

        fitstat0 = self.fitstat(parameter_value=ml_fit_parameter)

        def helper_func(parameter_ul):

            fitstat = self.fitstat(parameter_ul)
            ts = fitstat - fitstat0

            return ts - critical_value

        parameter_max_list = np.linspace(ml_fit_parameter,
                                         self.limit_target.parmax,
                                         10)
        limit_found = False
        for parameter_max in parameter_max_list:
            if limit_found:
                break

            try:
                ul = brentq(helper_func, a=ml_fit_parameter, b=parameter_max)
                limit_found = True
            except ValueError as e:
                info = "Brentq solver for parameter_max="
                info += str(parameter_max) + " failed: "
                info += str(e) + " -> This is usually not a problem "
                info += "because other parameter_max from the list will work."
                self._logger.debug(info)
                continue
            except Exception as e:
                info = "Unexpected exception during brentq solver " + str(e)
                raise RuntimeError(info)

        if limit_found is False:
            """
            Couldn't do it with brentq -> Going into panic mode and try fsolve
            """
            try:
                self._logger.debug("All brentq solvers failed. Using fsolve.")
                ul, info, ier, mesg = fsolve(helper_func,
                                             x0=ml_fit_parameter,
                                             full_output=True)
                if ier == 1:
                    self._logger.debug("fsolve succeeded: " + str(ul))
                if ier != 1:
                    info = "Couldn't find upper limit. Using minimal parameter"
                    info += " as degenerate covering solution."
                    self._logger.debug(info)

                    return self.limit_target.parmin

            except Exception as e:
                info = "Couldn't find upper limit " + str(e)
                raise RuntimeError(info)
            ul = np.max(ul)

        return ul

    @property
    def critical_value(self) -> float:
        raise NotImplementedError

    def ml_fit_parameter(self) -> float:
        raise NotImplementedError


class ConstrainedLR(LRBase):
    """An inversion of the profile likelihood ratio test where the
       limit_target parameter is constrained to be non-negative.
    """

    def __init__(self,
                 limit_target: LimitTarget,
                 data: Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 fit_config: dict = {"optimize_opts": {"print_level": 0,
                                                       "tol": 3.0,
                                                       "strategy": 2,
                                                       "backend": "minuit"},
                                     "n_repeat_fit_max": 3},
                 max_pool_entries: int = 20):

        super().__init__(limit_target, data, models,
                         CL, fit_config, max_pool_entries)

    @property
    def critical_value(self) -> float:
        """Critical value for the constrained LR test.
        """
        return stats.chi2.ppf(2. * self.CL - 1, df=1)

    @property
    def ul(self) -> float:
        ul = self._ul
        if ul < self.limit_target.parmin:
            info = "Upper limit smaller than expected"
            raise RuntimeError(info)
        return ul

    def ml_fit_parameter(self) -> float:
        """Returns the ML fit parameter given the parameter constraint to
           non-negative.
        """
        if hasattr(self, "_ml_fit_parameter"):
            return self._ml_fit_parameter

        with np.errstate(all="ignore"):
            ml_fit_parameter, _ = self.fit()

        info = "ML fit parameter from unconstrained fit: "
        info += str(ml_fit_parameter)
        self._logger.debug(info)

        if ml_fit_parameter < 0.:
            self._logger.debug("Setting ML fit parameter=0")
            ml_fit_parameter = 0.

        if ml_fit_parameter > self.limit_target.parmax:
            info = "ml_fit_parameter=" + str(ml_fit_parameter)
            info += " out of range: Increase limit_target.parmax"
            raise RuntimeError(info)

        setattr(self, "_ml_fit_parameter", ml_fit_parameter)

        return ml_fit_parameter


class UnconstrainedLR(LRBase):
    """An inversion of the profile likelihood ratio test without any
       constraints on the limit_target parameter.
    """

    def __init__(self,
                 limit_target: LimitTarget,
                 data: Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 fit_config: dict = {"optimize_opts": {"print_level": 0,
                                                       "tol": 3.0,
                                                       "strategy": 2,
                                                       "backend": "minuit"},
                                     "n_repeat_fit_max": 3},
                 max_pool_entries: int = 20):

        super().__init__(limit_target, data, models, CL,
                         fit_config, max_pool_entries,)

    @property
    def critical_value(self) -> float:
        """Critical value for the unconstrained test.
        """
        return stats.chi2.ppf(self.CL, df=1)

    def ml_fit_parameter(self) -> float:
        """Returns the ML fit parameter without constraints.
        """
        if hasattr(self, "_ml_fit_parameter"):
            return self._ml_fit_parameter

        with np.errstate(all="ignore"):
            ml_fit_parameter, _ = self.fit()

        info = "ML fit parameter: "
        info += str(ml_fit_parameter)
        self._logger.debug(info)

        setattr(self, "_ml_fit_parameter", ml_fit_parameter)

        return ml_fit_parameter

    @property
    def ul(self) -> float:
        return self._ul
