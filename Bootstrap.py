import string
import random
import numpy as np
from gammapy.modeling import Fit
from ecpli.ECPLiBase import ECPLiBase, LimitTarget
import gammapy.modeling as modeling

from typing import List


class _BootstrapBase(ECPLiBase):
    """Base class to derive a limit on a model parameter with bootstrap
       samples.

       Attributes:
        dataset: Gammapy dataset from which the model parameter constraint
                 is to be derived.
        count_data: Measured event data
        relative_ul_error_max: Maximal relative error on the limit.
        fit_backend: Fit backend to be used in case that fits are performed.
        n_ul: Number of bootstrap datasets to generate to control the
              relative_ul_error_max.
    """
    def __init__(self,
                 limit_target: LimitTarget,
                 data: modeling.Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 relative_ul_error_max: float):
        """Args:
            limit_target: Instance of LimitTarget to specify parameter and
                          model to be constrained.
            data: Gammapy dataset from which the model parameter constraint
                  is to be derived.
            models: Models which are to be fit to the data.
            CL: Confidence level (e.g. 0.95) for the limit.
            relative_ul_error_max: Relative error on the upper limit. Bootstrap
                                   samples are drawn until this relative
                                   error is achieved. The error is controled by
                                   comparing multiple bootstrap datasets.
        """

        super().__init__(limit_target, data, models, CL)

        self.dataset = data
        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))
        dataset_copy = self.dataset.copy("dscopy_GE_" + random_string)

        self.count_data = dataset_copy.counts.data
        self.relative_ul_error_max = relative_ul_error_max
        self.fit_backend = "minuit"
        self.n_ul = 10

    def resample(self) -> np.ndarray:
        """Returns bootstrap sample of event count data.
        """

        info = "Must be implemented in derived classes"
        raise NotImplementedError(info)

    def _one_ul(self, n_bootstrap: int, bootstrap_list=None):
        dataset = self.dataset.copy("dscopy")

        if bootstrap_list is None:
            bootstrap_list = []

        def fill_bootstrap_list(bootstrap_list, n_entries: int):
            """Draws n_entries bootstrap datataset samples and fills up
               the bootstrap_list.
            """
            n_fit_errors = 0
            while len(bootstrap_list) < n_entries:
                dataset.counts.data = self.resample()
                dataset.models = self.limit_target.model.copy()
                fit = Fit([dataset])
                with np.errstate(all="ignore"):
                    try:
                        result = fit.optimize(backend=self.fit_backend)
                    except Exception:

                        n_fit_errors += 1
                        continue

                par_name = self.limit_target.parameter_name
                bootstrap_nan = np.isnan(result.parameters[par_name].value)
                if result.success is False or bootstrap_nan:
                    n_fit_errors += 1
                    info = "Fit problem --> Success : "
                    info += str(result.success) + " - best fit nan?: "
                    info += str(np.isnan(result.parameters[par_name].value))
                    self._logger.debug(info)
                    continue

                bootstrap_ml = result.parameters[par_name].value
                if len(bootstrap_list) % 10 == 0:
                    self._logger.debug("%", end="")

                bootstrap_list.append(bootstrap_ml)

            self._logger.debug("")
            if n_fit_errors > 0:
                info = "Had " + str(n_fit_errors)
                info += " fitting errors -> resimulated each until gone"
                self._logger.debug(info)

            return bootstrap_list

        bootstrap_list = fill_bootstrap_list(bootstrap_list, n_bootstrap)

        def _ul_from_bootstrap_list(bootstrap_list: List[float]) -> float:
            """Calculates the upper limit as quantile of the bootstrap_list.

               Note: In the paper, it is claimed that [0, lambda_UL] is an asymptotic confidence interval although the quantile 
               is calculated as int_{-\infty}^\lambda_UL d\lambda* f(\lambda*), i.e. from -infty instead of from 0. The reason why this is 
               valid is that the true lambda is always positive. This means that whenever the true lambda is in (-infty, lambda_UL) it must 
               be in [0, lambda_UL].
            """

            bootstrap_list = np.array(bootstrap_list)
            ul = np.quantile(bootstrap_list, q=self.CL)

            parmin = self.limit_target.parmin
            if ul < parmin:
                """
                If not enough entries in the bootstrap_list are
                larger than the minimal parameter,
                the formal UL at this confidence level is zero. Consider the
                case where limit_target.parmin=0 as an example. Now,
                the interval (0, UL=0) is a point and not an interval. A point
                must have zero coverage, so that's a no brainer.
                Instead, take the smallest positive entry in the
                bootstrap_list as UL.
                This is in practice the same as increasing the
                confidence level and will lead to over-coverage with respect
                to the requested confidence level - which is fine.
                """
                positive_list = bootstrap_list[bootstrap_list > parmin]
                if len(positive_list) == 0:
                    ul = parmin
                else:
                    ul = np.min(bootstrap_list[bootstrap_list > parmin])

            return ul

        ul = _ul_from_bootstrap_list(bootstrap_list)

        return ul, bootstrap_list

    @property
    def ul(self) -> float:
        """Returns the upper limit on the limit target.
        """

        bootstrap_list_list = []

        def _relative_ul_error(bootstrap_list_list, n_bootstrap):

            ul_list = []
            self._logger.debug(str(n_bootstrap) + " bootstrap samples ..")

            for i in range(self.n_ul):
                if len(bootstrap_list_list) < self.n_ul:
                    bootstrap_list = None
                else:
                    bootstrap_list = bootstrap_list_list[i]

                ul, bootstrap_list = self._one_ul(
                                            n_bootstrap,
                                            bootstrap_list=bootstrap_list)
                ul_list.append(ul)

                if len(bootstrap_list_list) < self.n_ul:
                    bootstrap_list_list.append(bootstrap_list)
                else:
                    bootstrap_list_list[i] = bootstrap_list

            ul_error = np.std(ul_list, ddof=1)

            relative_error = ul_error / np.mean(ul_list)

            info = "UL list: " + str(ul_list) + ", absolute error: "
            info += str(ul_error)
            info += " (" + str(round(relative_error * 100, 3)) + "%)"
            self._logger.debug(info)

            return relative_error

        n_bootstrap = 300
        relative_ul_error = _relative_ul_error(bootstrap_list_list,
                                               n_bootstrap)
        while(relative_ul_error > self.relative_ul_error_max):
            n_bootstrap *= 2
            relative_ul_error = _relative_ul_error(bootstrap_list_list,
                                                   n_bootstrap)

        final_bootstrap_list = np.array(bootstrap_list_list)
        """
        The final bootstrap limit is taken from all n_ul
        bootstrap samples. For n_ul = 10,
        I expect the final bootstrap limit about a factor
        of 1/sqrt(10) more precise than the
        test bootstrap samples.
        """
        ul, bootstrap_list = self._one_ul(
                                n_bootstrap,
                                bootstrap_list=final_bootstrap_list.ravel())
        info = "Final UL: " + str(ul)
        info += " with bootstrap size: " + str(len(bootstrap_list))
        self._logger.debug(info)
        return ul


class BestFitParametricBootstrap(_BootstrapBase):
    """Bootstrap samples are Poisson samples from the best fitting model.
    """

    def __init__(self,
                 limit_target: LimitTarget,
                 data: modeling.Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 relative_ul_error_max: float):
        """Args see base class.
        """

        super().__init__(limit_target, data, models, CL, relative_ul_error_max)

    def resample(self) -> np.ndarray:
        if hasattr(self, "npred"):
            return np.random.poisson(self.npred)

        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))

        dataset = self.dataset.copy("dscopy_GE_" + random_string)

        dataset.models = self.limit_target.model.copy()
        fit = Fit([dataset])
        _ = fit.optimize(backend=self.fit_backend)
        self.npred = dataset.npred().data
        return np.random.poisson(self.npred)


class PoissonParametricBootstrap(_BootstrapBase):
    """Bootstrap samples are Poisson samples of the binned measured data.
    """

    def __init__(self,
                 limit_target: LimitTarget,
                 data: modeling.Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 relative_ul_error_max: float):
        """Args see base class.
        """

        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))

        dataset_copy = data.copy("dscopy_GE_" + random_string)

        self.count_data = dataset_copy.counts.data
        super().__init__(limit_target, data, models, CL, relative_ul_error_max)

    def resample(self) -> np.ndarray:
        return np.random.poisson(self.count_data)


class NonParametricBootstrap(_BootstrapBase):
    """Bootstraps are multinomial vectors of the binned measured data.
    """

    def __init__(self,
                 limit_target: LimitTarget,
                 data: modeling.Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 relative_ul_error_max: float):
        """Args see base class.
        """

        self.n_events = np.sum(data.counts.data.ravel())
        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))

        dataset_copy = data.copy("dscopy_GE_" + random_string)

        self.probability_vector = dataset_copy.counts.data / self.n_events

        super().__init__(limit_target, data, models, CL, relative_ul_error_max)

    def resample(self) -> np.ndarray:

        r = np.random.multinomial(self.n_events,
                                  self.probability_vector,
                                  size=1)

        return r[0]
