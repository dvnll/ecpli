import numpy as np

from gammapy.modeling import Fit

from scipy.optimize import brentq, fsolve

import scipy.stats as stats

import random
import string

import matplotlib.pyplot as plt

from ecpli.ECPLiBase import ECPLiBase



class _MLFitPool(object):

    def __init__(self, max_size):
        self._lambda_list = []
        self._likelihood_list = []
        self._model_list = []
        self._max_size = max_size

    @property
    def full(self):
        return (self.entries >= self._max_size)

    @property
    def entries(self):
        return len(self._lambda_list)

    def append(self, lambda_best, dataset):

        if lambda_best in self.lambda_list:
            return

        self._lambda_list.append(lambda_best)
        self._likelihood_list.append(dataset.stat_sum())
        self._model_list.append(dataset.models)

    @property
    def _index_list(self):
        return np.argsort(self._lambda_list)

    @property
    def lambda_list(self):
        return np.array(self._lambda_list)[self._index_list]

    @property
    def likelihood_list(self):
        return np.array(self._likelihood_list)[self._index_list]

    def ml_model(self):
        ml_index = np.argmin(self._likelihood_list)
        ml_model = self._model_list[ml_index]
        temp_model = []
        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))

        for model in ml_model:
            name = model.name.split("_GE_")[0]
            temp_model.append(model.copy(name=name + "_GE_" + random_string))

        return temp_model

    def closest_model(self, lambda_):
        best_index = np.argmin(np.abs(np.array(self._lambda_list) - lambda_))
        closest_model = self._model_list[best_index]

        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))
        temp_model = []
        for model in closest_model:
            name = model.name.split("_GE_")[0]
            temp_model.append(model.copy(name=name + "_GE_" + random_string))

        closest_model = temp_model

        return closest_model


class _LRBase(ECPLiBase):

    def __init__(self, limit_target, data, models, CL, 
                 naima=False,
                 lambda_min=1./1e8, lambda_max=1./0.05):

        super().__init__(limit_target, data, models, CL)
        """
        ecpl_model_name is the name of the model in models for which
        a limit on the lambda parameter is to be derived
        """
        self.naima = naima
        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))
        self.dataset = data
        self.model = [model.copy(name=model.name + "_GE_" + random_string) for model in models]
        self.ecpl_model_name = limit_target.model.name

        self.verbose = False #True
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self._n_fits = 0

        max_ml_pool_entries = 20  # 100 for plotting
        self._ml_fit_pool = _MLFitPool(max_ml_pool_entries)

        while self._ml_fit_pool.entries < max_ml_pool_entries:
            info = "Filling ML fit pool: Current size: "
            info += str(self._ml_fit_pool.entries) + "/"
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

        if self._ml_fit_pool.entries == 0:
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

        critical_value = self.critical_value

        ml_fit_lambda = self.ml_fit_lambda()

        fitstat0 = self.fitstat(lambda_=ml_fit_lambda)

        def helper_func(lambda_ul):

            fitstat = self.fitstat(lambda_ul)
            ts = fitstat - fitstat0

            return ts - critical_value

        lambda_max_list = [1./100., 1./50., 1./10., 1./5, 1./1., 1./0.5, 1./0.1, self.lambda_max]
        if self.naima:
            lambda_max_list = [lambda_max / 10. for lambda_max in lambda_max_list]

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
                info += str(e) + " -> This is usually not a problem"
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
                    info = "Couldn't find upper limit. Using lambda_min"
                    info += " as degenerate covering solution."
                    if self.verbose:
                        print(info)

                    return self.lambda_min

            except Exception as e:
                info = "Couldn't find upper limit " + str(e)
                raise RuntimeError(info)
            ul = np.max(ul)

            if self.verbose:
                print("helper_func at solution: " + str(helper_func(ul)))

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
    def __init__(self, limit_target, data, models, CL,
                 naima=False,
                 lambda_min=1./1e8, lambda_max=1./0.05):

        super().__init__(limit_target, data, models, CL, naima)

    @property
    def critical_value(self):
        return stats.chi2.ppf(2. * self.CL - 1, df=1)

    @property
    def ul(self):
        ul = self._ul
        if ul < self.lambda_min:
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

        if ml_fit_lambda > self.lambda_max:
            if self.verbose:
                info = "ml_fit_lambda=" + str(ml_fit_lambda)
                info += " out of range: Increase lambda_max"
                raise RuntimeError(info)

        setattr(self, "_ml_fit_lambda", ml_fit_lambda)

        return ml_fit_lambda


class UnconstrainedLR(_LRBase):
    def __init__(self, limit_target, data, models, CL, naima=False):
        super().__init__(limit_target, data, models, CL, naima=False)

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
