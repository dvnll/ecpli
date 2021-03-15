import string
import random
import numpy as np
from gammapy.modeling import Fit
from ecpli.ECPLiBase import ECPLiBase, LimitTarget
import gammapy.modeling as modeling

from typing import List


class _BootstrapBase(ECPLiBase):

    def __init__(self,
                 limit_target: LimitTarget,
                 data: modeling.Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 relative_ul_error_max: float):

        super().__init__(limit_target, data, models, CL)

        self.dataset = data
        self.model = models[0]
        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))
        dataset_copy = self.dataset.copy("dscopy_GE_" + random_string)

        self.count_data = dataset_copy.counts.data
        self.relative_ul_error_max = relative_ul_error_max
        self.fit_backend = "minuit"

    def resample(self) -> np.ndarray:
        """Returns bootstrap sample of event count data.
        """

        info = "Must be implemented in derived classes"
        raise NotImplementedError(info)

    def _one_ul(self, n_bootstrap: int, lambda_list=None):

        dataset = self.dataset.copy("dscopy")

        if lambda_list is None:
            lambda_list = []

        def fill_lambda_list(lambda_list, n_entries: int):

            n_fit_errors = 0
            while len(lambda_list) < n_entries:
                dataset.counts.data = self.resample()
                dataset.models = self.model.copy()
                fit = Fit([dataset])
                with np.errstate(all="ignore"):
                    try:
                        result = fit.optimize(backend=self.fit_backend)
                    except Exception:

                        n_fit_errors += 1
                        continue

                lambda_nan = np.isnan(result.parameters["lambda_"].value)
                if result.success is False or lambda_nan:
                    n_fit_errors += 1
                    info = "Fit problem --> Success : "
                    info += str(result.success) + " - Lambda Nan : "
                    info += str(np.isnan(result.parameters["lambda_"].value))
                    print(info)
                    continue

                lambda_ml = result.parameters["lambda_"].value
                if len(lambda_list) % 10 == 0:
                    print("%", end="")

                lambda_list.append(lambda_ml)

            print("")
            if n_fit_errors > 0:
                info = "Had " + str(n_fit_errors)
                info += " fitting errors -> resimulated each until gone"
                print(info)

            return lambda_list

        lambda_list = fill_lambda_list(lambda_list, n_bootstrap)

        def _ul_from_lambda_list(lambda_list: List[float]) -> float:
            lambda_list = np.array(lambda_list)
            ul = np.quantile(lambda_list, q=self.CL)

            if ul < 0.:
                """
                If not enough entries in the lambda_list are larger than zero,
                the formal UL at this confidence level is zero. However,
                the interval (0, UL=0) is a point and not an interval. A point
                must have zero coverage, so that's a no brainer.
                Instead, take the smallest positive entry in the lambda_list as
                UL. This is in practice the same as improving the
                confidence level and will lead to over-coverage with respect
                to the requested confidence level - which is fine.
                """
                positive_list = lambda_list[lambda_list > 0]
                if len(positive_list) == 0:
                    ul = 0.
                else:
                    ul = np.min(lambda_list[lambda_list > 0])

            return ul

        ul = _ul_from_lambda_list(lambda_list)

        return ul, lambda_list

    @property
    def ul(self) -> float:
        """Returns the upper limit on the limit target.
        """

        n_ul = 10

        lambda_list_list = []

        def _relative_ul_error(lambda_list_list, n_ul, n_bootstrap):

            ul_list = []
            print(str(n_bootstrap) + " bootstrap samples ..")

            for i in range(n_ul):
                if len(lambda_list_list) < n_ul:
                    lambda_list = None
                else:
                    lambda_list = lambda_list_list[i]

                ul, lambda_list = self._one_ul(n_bootstrap,
                                               lambda_list=lambda_list)
                ul_list.append(ul)

                if len(lambda_list_list) < n_ul:
                    lambda_list_list.append(lambda_list)
                else:
                    lambda_list_list[i] = lambda_list

            ul_error = np.std(ul_list, ddof=1)

            relative_error = ul_error / np.mean(ul_list)

            info = "UL list: " + str(ul_list) + ", absolute error: "
            info += str(ul_error)
            info += " (" + str(round(relative_error * 100, 3)) + "%)"
            print(info)

            return relative_error

        n_bootstrap = 300
        relative_ul_error = _relative_ul_error(lambda_list_list,
                                               n_ul, n_bootstrap)
        while(relative_ul_error > self.relative_ul_error_max):
            n_bootstrap *= 2
            relative_ul_error = _relative_ul_error(lambda_list_list,
                                                   n_ul, n_bootstrap)

        final_lambda_list = np.array(lambda_list_list)
        """
        The final bootstrap limit is taken from all n_ul
        bootstrap samples. For n_ul = 10,
        I expect the final bootstrap limit about a factor
        of 1/sqrt(10) more precise than the
        test bootstrap samples.
        """
        ul, lambda_list = self._one_ul(n_bootstrap,
                                       lambda_list=final_lambda_list.ravel())
        info = "Final UL: " + str(ul)
        info += " with bootstrap size: " + str(len(lambda_list))
        print(info)
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

        super().__init__(limit_target, data, models, CL, relative_ul_error_max)

    def resample(self) -> np.ndarray:
        if hasattr(self, "npred"):
            return np.random.poisson(self.npred)

        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))

        dataset = self.dataset.copy("dscopy_GE_" + random_string)

        dataset.models = self.model.copy()
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
                 models: modeling.models.Models,
                 CL: float,
                 relative_ul_error_max: float):

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
