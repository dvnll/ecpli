import numpy as np

from gammapy.modeling import Fit, models
import gammapy.modeling as modeling

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
        max_size: Maximal size of the pool.
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
        self._parameter_list = []
        self._likelihood_list = []
        self._model_list = []
        self.max_size = max_size

    @property
    def full(self) -> bool:
        return (len(self) >= self.max_size)

    def __len__(self) -> int:
        return len(self._parameter_list)

    def append(self, parameter_best: float, dataset: modeling.Dataset):
        if parameter_best in self.parameter_list:
            return

        self._parameter_list.append(parameter_best)
        self._likelihood_list.append(dataset.stat_sum())
        self._model_list.append(dataset.models)

    @property
    def _index_list(self) -> list:
        return np.argsort(self._parameter_list)

    @property
    def parameter_list(self) -> list:
        return np.array(self._parameter_list)[self._index_list]

    @property
    def likelihood_list(self) -> list:
        return np.array(self._likelihood_list)[self._index_list]

    def ml_model(self) -> List[models.SkyModel]:
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
            name = model.name.split("_CACHE_")[0]
            temp_model.append(model.copy(name=name + "_CACHE_" + random_string))

        return temp_model

    def closest_model(self, parameter: float) -> List[models.SkyModel]:
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
            name = model.name.split("_CACHE_")[0]
            temp_model.append(model.copy(name=name + "_CACHE_" + random_string))

        closest_model = temp_model
        return closest_model


class _LRBase(ECPLiBase):

    def __init__(self, limit_target: LimitTarget,
                 data: modeling.Dataset,
                 models: models.Models,
                 CL: float,
                 fit_backend="minuit"):

        super().__init__(limit_target, data, models, CL, fit_backend)
        """
        ecpl_model_name is the name of the model in models for which
        a limit on the lambda parameter is to be derived
        """
        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))
        self.dataset = data
        self.model = [model.copy(name=model.name + "_GE_" + random_string) for model in models]
        self.ecpl_model_name = limit_target.model.name

        self.verbose = False #True
        self._n_fits = 0

        max_ml_pool_entries = 20  # 100 for plotting
        self._ml_fit_pool = _ProfileLRPool(max_ml_pool_entries)

        while len(self._ml_fit_pool) < max_ml_pool_entries:
            info = "Filling ML fit pool: Current size: "
            info += str(len(self._ml_fit_pool)) + "/"
            info += str(max_ml_pool_entries)
            print(info)

            lambda_ = stats.gamma.rvs(a=1.1, scale=1./(5. * 1.1), size=1)[0]
            self.fit(lambda_=lambda_)

    def fit(self, lambda_=None):

        n_repeat_fit_max = 3
        print_level = 0
        tolerance = 3.0
        strategy = 2
        self._n_fits += 1

        letters = string.ascii_uppercase + string.digits

        dataset = self.dataset  # .copy("dscopy_GE_" + random_string)

        if len(self._ml_fit_pool) == 0:
            random_string = "".join(random.choice(letters) for _ in range(10))
            dataset.models = [model.copy(name=model.name + "_GE_" + random_string) for model in self.model]

        else:
            if lambda_ is None:
                dataset.models = self._ml_fit_pool.ml_model()
            else:
                dataset.models = self._ml_fit_pool.closest_model(lambda_)

        if lambda_ is not None:
            for model in dataset.models:
                if self.ecpl_model_name in model.name:
                    model.spectral_model.lambda_.frozen = True
                    model.spectral_model.lambda_.value = lambda_
        else:
            for model in dataset.models:
                if self.ecpl_model_name in model.name:
                    model.spectral_model.lambda_.frozen = False

        if self.verbose:
            print("---------------------------------------------")
            print("Fit input dataset:")
            info = "lambda_ input: " + str(lambda_)
            if lambda_ is not None:
                info += ": -> lambda_ should be frozen to this "
                info += "value in the following dataset"
            print(info)
            print(dataset)
            if hasattr(dataset, "background_model"):
                print("Background model:")
                print(dataset.background_model)

            print("Free parameters:")
            for par in dataset.models.parameters.free_parameters:
                info = par.name + ": value: " + str(par.value)
                info += ", min: " + str(par.min) + ", max: " + str(par.max)
                print(info)
            print("---------------------------------------------")

        fit = Fit([dataset])
        with np.errstate(all="ignore"):
            #result = fit.run()
            result = fit.run(optimize_opts={"print_level": print_level,
                                            "tol": tolerance,
                                            "strategy": strategy})
            n_repeat_fit = 1
            while result.success is False:
                if n_repeat_fit > n_repeat_fit_max:
                    break
                if self.verbose:
                    info = "Fit " + str(n_repeat_fit)
                    info += " didn't converge -> Repeating"
                    print(info)

                n_repeat_fit += 1
                tolerance += 1
                fit = Fit([dataset])
                #result = fit.run()
                result = fit.run(optimize_opts={"print_level": print_level,
                                                "tol": tolerance,
                                                "strategy": strategy})
            if result.success is False:
                if lambda_ is None:
                    raise RuntimeError("Fitting error in unconstrained model!")
                else:
                    if self.verbose:
                        info = "Fit failed for fixed lambda=" + str(lambda_)
                        info += " (this is typically no problem)"
                        print(info)

        lambda_best = None

        if result.success:
            if self.verbose:
                print("Result:")
                print(result)
                print("Parameters:")
                for name, value in zip(result.parameters.names,
                                       result.parameters.values):
                    print(str(name) + ": " + str(value))
                print("**+++++++++++++++++++++++++++++++++++++++++++++**")
            for model in dataset.models:
                if self.ecpl_model_name in model.name:
                    lambda_best = model.spectral_model.lambda_.value

            if lambda_best is not None and not self._ml_fit_pool.full:
                self._ml_fit_pool.append(lambda_best, dataset)

        if self.verbose:
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print("Fit output dataset:")
            print(dataset)
            if hasattr(dataset, "background_model"):
                print("Background model:")
                print(dataset.background_model)
            print("---------------------------------------------")

        fitstat = dataset.stat_sum()

        dataset.models = None
        info = str(self._n_fits) + ": Fit result for input lambda="
        info += str(lambda_) + " -> fitstat: " + str(fitstat)
        print(info)

        return lambda_best, fitstat

    def fitstat(self, lambda_=None):

        _, fitstat = self.fit(lambda_=lambda_)
        return fitstat

    @property
    def n_fits_performed(self):
        return self._n_fits

    def ts(self):

        if hasattr(self, "__ts"):
            return getattr(self, "__ts")

        ml_fit_lambda = self.ml_fit_lambda()
        fitstat0 = self.fitstat(lambda_=ml_fit_lambda)

        fitstat_pl = self.fitstat(lambda_=0.)
        if self.verbose:
            info = "fitstat_pl=" + str(fitstat_pl)
            info += ", fitstat_ml=" + str(fitstat0)
            print(info)

        ts = fitstat_pl - fitstat0
        setattr(self, "__ts", ts)
        return ts

    @property
    def _ul(self):
        """Calculate an upper limit. The method has two steps:
           First: Try to find a solution to
                  ProfileLikelihood(lambda)=critical_value(CL)
                  for the parameter in [ml_fit_parameter, test_value]
                  with the brentq method.
                  In an optimal world, one could just use
                  a very large test_value. But this leads to frequent
                  numerical problems. So, a range of test_values is tried
                  until a solution is found.
           Second: If no solution is found with brentq, fsolve is used.
        """

        critical_value = self.critical_value

        ml_fit_lambda = self.ml_fit_lambda()

        fitstat0 = self.fitstat(lambda_=ml_fit_lambda)

        def helper_func(lambda_ul):

            fitstat = self.fitstat(lambda_ul)
            ts = fitstat - fitstat0

            return ts - critical_value

        lambda_max_list = np.logspace(np.log10(ml_fit_lambda),
                                      np.log10(self.limit_target.parmax),
                                      10)

        limit_found = False
        for lambda_max in lambda_max_list:
            if ml_fit_lambda >= lambda_max:
                continue
            try:
                ul = brentq(helper_func, a=ml_fit_lambda, b=lambda_max)
                limit_found = True
            except ValueError as e:
                info = "Brentq solver for lambda_max="
                info += str(lambda_max) + " failed: "
                info += str(e) + " -> This is usually not a problem "
                info += "because other lambda_max from the list will work."
                if self.verbose:
                    print(info)
                continue
            except Exception as e:
                info = "Unexpected exception during brentq solver " + str(e)
                raise RuntimeError(info)

        if limit_found is False:
            """
            Couldn't do it with brentq -> Going into panic mode and try fsolve
            """
            try:
                if self.verbose:
                    print("All brentq solvers failed. Using fsolve ...")
                ul, info, ier, mesg = fsolve(helper_func,
                                             x0=ml_fit_lambda,
                                             full_output=True)
                if self.verbose and (ier == 1):
                    print("fsolve succeeded: " + str(ul))
                if ier != 1:
                    info = "Couldn't find upper limit. Using minimal parameter"
                    info += " as degenerate covering solution."
                    if self.verbose:
                        print(info)

                    return self.limit_target.parmin

            except Exception as e:
                info = "Couldn't find upper limit " + str(e)
                raise RuntimeError(info)
            ul = np.max(ul)

        return ul

    @property
    def critical_value(self):
        raise NotImplementedError

    def ml_fit_lambda(self):
        raise NotImplementedError

    def plot_ts(self, plt, lambda_true=None, lambda_ul=None):

        ml_lambda, max_likelihood = self.fit()

        likelihood_list = self._ml_fit_pool.likelihood_list
        f_list = likelihood_list - max_likelihood
        lambda_list = self._ml_fit_pool.lambda_list
        index_list = np.argsort(lambda_list)
        lambda_list = lambda_list[index_list]
        f_list = f_list[index_list]
        plt.plot(lambda_list, f_list, color="r")
        critical_value = self.critical_value
        plt.axvline(self.ml_fit_lambda(),
                    label=r"Maximum likelihood $\lambda$", color="g", ls="--")
        if lambda_true is not None:
            plt.axvline(lambda_true,
                        label=r"True $\lambda$", color="y", ls="--")
        plt.axhline(critical_value, label="Critical value", color="b", ls="--")
        if lambda_ul is not None:
            plt.axvline(lambda_ul, color="m", ls="--",
                        label=r"Upper limit on $\lambda$")
            plt.gca().axvspan(plt.xlim()[0], lambda_ul, alpha=0.1, color="m")

        plt.xscale("log")
        plt.yscale("log")

        plt.legend(fontsize="xx-large")
        plt.xlabel(r"$\lambda$ ($\mathrm{TeV}^{-1}$)", fontsize=30)
        plt.ylabel(r"$\Lambda(\lambda)$", fontsize=30)

    def plot_profile_likelihood(self, plt, lambda_true=None):

        ts = self.ts()
        plt.title("TS=" + str(round(ts, 1)), fontsize=30, loc="left")

        plt.plot(self._ml_fit_pool.lambda_list,
                 self._ml_fit_pool.likelihood_list, color="b")

        plt.axvline(self.ml_fit_lambda(), label=r"ML $\lambda$",
                    color="r", ls="--")
        if lambda_true is not None:
            plt.axvline(lambda_true, label=r"True $\lambda$",
                        color="g", ls="--")

        plt.legend(fontsize="xx-large")
        plt.xscale("log")
        plt.xlabel(r"$\lambda$ ($\mathrm{TeV^{-1}}$)", fontsize=30)
        plt.ylabel(r"-2$\ln\mathcal{L}$", fontsize=30)


class ConstrainedLR(_LRBase):
    def __init__(self, limit_target, data, models, CL):

        super().__init__(limit_target, data, models, CL)

    @property
    def critical_value(self):
        return stats.chi2.ppf(2. * self.CL - 1, df=1)

    @property
    def ul(self):
        ul = self._ul
        if ul < self.limit_target.parmin:
            info = "Upper limit smaller than expected"
            raise RuntimeError(info)
        return ul
    
    def ml_fit_lambda(self):
        if hasattr(self, "_ml_fit_lambda"):
            return self._ml_fit_lambda

        with np.errstate(all="ignore"):
            ml_fit_lambda, _ = self.fit()

        if self.verbose:
            info = "ML fit lambda from unconstrained fit: "
            info += str(ml_fit_lambda)
            print(info)

        if ml_fit_lambda < 0.:
            if self.verbose:
                print("Setting ML fit lambda=0")
            ml_fit_lambda = 0.

        if ml_fit_lambda > self.limit_target.parmax:
            if self.verbose:
                info = "ml_fit_lambda=" + str(ml_fit_lambda)
                info += " out of range: Increase lambda_max"
                raise RuntimeError(info)

        setattr(self, "_ml_fit_lambda", ml_fit_lambda)

        return ml_fit_lambda


class UnconstrainedLR(_LRBase):
    def __init__(self, limit_target, data, models, CL):
        super().__init__(limit_target, data, models, CL)

    @property
    def critical_value(self):
        return stats.chi2.ppf(self.CL, df=1)

    def ml_fit_lambda(self):
        if hasattr(self, "_ml_fit_lambda"):
            return self._ml_fit_lambda

        with np.errstate(all="ignore"):
            ml_fit_lambda, _ = self.fit()

        if self.verbose:
            info = "ML fit lambda: "
            info += str(ml_fit_lambda)
            print(info)

        setattr(self, "_ml_fit_lambda", ml_fit_lambda)

        return ml_fit_lambda

    def ul(self):
        return self._ul
