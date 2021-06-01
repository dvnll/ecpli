import numpy as np

import gammapy.modeling as modeling
from gammapy.datasets import Dataset

from scipy.optimize import brentq, fsolve
import scipy.stats as stats

from ecpli.ECPLiBase import ECPLiBase, LimitTarget


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
                 CL: float,
                 fit_config: dict):

        super().__init__(limit_target, data, CL)

        self._n_fits = 0
        self.fit_config = fit_config

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

        def freeze_target_parameter(frozen: bool) -> None:
            """Controls whether the target parameter is frozen in the fit.

            Args:
                frozen: If true, the target parameter is frozen in the fit.
            """
            for model in dataset.models:
                if model.name != self.limit_target.model.name:
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

        dataset = self.data.copy()

        if parameter_value is not None:
            freeze_target_parameter(frozen=True)
        else:
            freeze_target_parameter(frozen=False)

        show_fit_dataset_debug_information()

        fit_result = fit_dataset()
        best_fit_parameter = _best_fit_parameter(fit_result)

        fitstat = dataset.stat_sum()
        print_afterfit_debug()

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
                 CL: float,
                 fit_config: dict = {"optimize_opts": {"print_level": 0,
                                                       "tol": 3.0,
                                                       "strategy": 2},
                                     "n_repeat_fit_max": 3}):

        super().__init__(limit_target,
                         data,
                         CL,
                         fit_config)

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
                 CL: float,
                 fit_config: dict = {"optimize_opts": {"print_level": 0,
                                                       "tol": 3.0,
                                                       "strategy": 2},
                                     "n_repeat_fit_max": 3}):

        super().__init__(limit_target, data, CL,
                         fit_config,)

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
