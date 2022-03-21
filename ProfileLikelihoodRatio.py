import numpy as np

import gammapy.modeling as modeling
from gammapy.datasets import Datasets

from scipy.optimize import brentq, fsolve
import scipy.stats as stats
import astropy.units as u

from ecpli.ECPLiBase import ECPLiBase, LimitTarget


class LRBase(ECPLiBase):
    """Base class to invert the profile likelihood ratio test to derive
       limits on the limit_target methods to.

       Attributes:
        fit_config: Dictionary which describes the parameters of the
                    likelihood fit optimization.
        datasets: Datasets from which a limit is to be derived.
        n_fits_performed: Number of fits performed in this instance.
    """

    def __init__(self, limit_target: LimitTarget,
                 datasets: Datasets,
                 CL: float,
                 fit_config: dict):

        super().__init__(limit_target, datasets, CL)

        self._n_fits = 0
        self.fit_config = fit_config

    def _fit_datasets(self, datasets: Datasets) -> modeling.fit.OptimizeResult:
        """Fits the datasets to its model.
        Can be monkey-patched for special fit routines.

               Returns: FitResult
        """

        fit = modeling.Fit(datasets)
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
                fit = modeling.Fit(datasets)
                optimize_opts = self.fit_config["optimize_opts"]
                result = fit.run(
                    optimize_opts=optimize_opts)

        return result

    def fit(self, parameter_value=None) -> (float, float):
        """Performs the likelihood optimization.

           Args:
            parameter_value: Limit target parameter value.
                             If None: Keep limit target parameter free in fit.
                             If fixed to a float: Fix limit target parameter to
                             the given value in the likelihood fit.
           Returns: (parameter_value,
                     2 log(likelihood value) modulo constant
                     at parameter_value)
        """

        self._n_fits += 1

        def freeze_target_parameter(frozen: bool) -> None:
            """Controls whether the target parameter is frozen in the fit.

            Args:
                frozen: If true, the target parameter is frozen in the fit.
            """
            for model in datasets.models:
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

        def show_fit_datasets_debug_information() -> None:
            """Prints debug information if in corresponding log-level.
            """
            self._logger.debug("---------------------------------------------")
            self._logger.debug("Fit input datasets:")
            info = "parameter value input: " + str(parameter_value)
            if parameter_value is not None:
                info += "Target parameter " + self.limit_target.parameter_name
                info += ": -> should be frozen to this "
                info += "value in the following datasets"
            self._logger.debug(info)
            self._logger.debug(datasets)
            for dataset in datasets:
                if hasattr(dataset, "background_model"):
                    self._logger.debug(f"Background model of dataset {dataset.name}:")
                    self._logger.debug(datasets.background_model)

            self._logger.debug("Free parameters:")
            for par in datasets.models.parameters.free_parameters:
                info = par.name + ": value: " + str(par.value)
                info += ", min: " + str(par.min) + ", max: " + str(par.max)
                self._logger.debug(info)
            self._logger.debug("---------------------------------------------")

        def _best_fit_parameter(
                    fit_result: modeling.fit.OptimizeResult) -> float:
            """Extracts the best fit target parameter out of the fit result.
               Args:
                fit_result: Return value of fit_datasets()
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

                for model in datasets.models:
                    if self.limit_target.model.name in model.name:
                        for parameter in model.parameters:
                            pname = parameter.name
                            if pname == self.limit_target.parameter_name:
                                best_fit_parameter = parameter.value
                                break

            return best_fit_parameter

        def print_afterfit_debug() -> None:
            """Prints debug information for the datasets state after the fit
               if in respective log-level.
            """

            self._logger.debug("+++++++++++++++++++++++++++++++++++++++++++++")
            self._logger.debug("Fit output datasets:")
            self._logger.debug(datasets)
            if hasattr(datasets, "background_model"):
                self._logger.debug("Background model:")
                self._logger.debug(datasets.background_model)
            self._logger.debug("---------------------------------------------")

            info = str(self._n_fits) + ": Fit result for input parameter="
            info += str(parameter_value) + " -> fitstat: " + str(fitstat)
            self._logger.debug(info)

        datasets = self.datasets.copy()

        if parameter_value is not None:
            freeze_target_parameter(frozen=True)
        else:
            freeze_target_parameter(frozen=False)

        show_fit_datasets_debug_information()

        fit_result = self._fit_datasets(datasets)
        best_fit_parameter = _best_fit_parameter(fit_result)

        fitstat = datasets.stat_sum()
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

    def cutoff_ts(self) -> float:
        """Returns the test statistic or significance for the LR test of a
        restricted vs full model.
        """

        ml_fit_parameter = self.ml_fit_parameter()
        fitstat0 = self.fitstat(parameter_value=ml_fit_parameter)

        fitstat_pl = self.fitstat(parameter_value=0.)
        info = "fitstat_pl=" + str(fitstat_pl)
        info += ", fitstat_ml=" + str(fitstat0)
        self._logger.debug(info)

        ts = fitstat_pl - fitstat0
        return np.max([0, ts])

    def cutoff_significance(self) -> float:
        """Returns the significance of the cutoff."""
        ts = self.cutoff_ts()
        s = np.sqrt(ts)
        if self.ml_fit_parameter() < 0:
            s *= -1.

        return s

    def pts(self, threshold_energy: u.Quantity) -> float:
        """Returns the PTS value.

            Args:
             threshold_energy: Threshold value for the PTS,
                               e.g. 1 PeV for a PeVatron
                               in the hadron spectrum or 100 TeV
                               for the gamma-spectrum.
        """
        punit = self.limit_target.parameter_unit()
        threshold_energy = threshold_energy.to(1 / punit)
        fitstat_threshold = self.fitstat(
                    parameter_value=1./threshold_energy.value)

        ml_fit_parameter = self.ml_fit_parameter()
        fitstat_ml = self.fitstat(ml_fit_parameter)

        info = "fitstat_threshold" + str(fitstat_threshold)
        info += ", fitstat_ml=" + str(fitstat_ml)
        self._logger.debug(info)

        pts = fitstat_threshold - fitstat_ml
        return np.max([pts, 0])

    def pts_significance(self, threshold_energy: u.Quantity) -> float:
        """Returns the significance of the PTS.

            Args:
             threshold_energy: Threshold value for the PTS,
                               e.g. 1 u.PeV for a PeVatron
                               in the hadron spectrum or
                               100 u.TeV
                               for the gamma-spectrum.
        """

        pts = self.pts(threshold_energy)
        pts_significance = np.sqrt(pts)

        punit = self.limit_target.parameter_unit()
        threshold_energy = threshold_energy.to(1 / punit)
        delta = 1. / threshold_energy.value - self.ml_fit_parameter()
        
        if delta < 0:
            pts_significance *= -1
        return pts_significance

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
        if ml_fit_parameter > self.limit_target.parmax:
            info = "LimitTarget parmax must be larger than the "
            info = "ML fit parameter (" + str(ml_fit_parameter)
            raise RuntimeError(info)

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
                info += "-- This usually means that the "
                info += "parmax paramter of the "
                info += "LimitTarget is ill defined."
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
                 datasets: Datasets,
                 CL: float,
                 fit_config: dict = {"optimize_opts": {"print_level": 0,
                                                       "tol": 3.0,
                                                       "strategy": 2},
                                     "n_repeat_fit_max": 3}):

        super().__init__(limit_target,
                         datasets,
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
           be non-negative.
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
                 datasets: Datasets,
                 CL: float,
                 fit_config: dict = {"optimize_opts": {"print_level": 0,
                                                       "tol": 3.0,
                                                       "strategy": 2},
                                     "n_repeat_fit_max": 3}):

        super().__init__(limit_target, datasets, CL,
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
