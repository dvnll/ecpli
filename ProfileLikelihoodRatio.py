import numpy as np

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
        self._parameter_list : List[float] = []
        self._likelihood_list : List[float] = []
        self._model_list : List[modeling.Dataset] = []
        self.max_size : int = max_size

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


class _LRBase(ECPLiBase):

    def __init__(self, limit_target: LimitTarget,
                 data: modeling.Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 fit_backend="minuit",
                 max_pool_entries=20):

        super().__init__(limit_target, data, models, CL, fit_backend)
        
        letters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(letters) for _ in range(10))
        self.dataset = data
        def initialize_fitstart_model():
            letters = string.ascii_uppercase + string.digits
            
            random_string = "".join(random.choice(letters) for _ in range(10))
            model_copy_list : List[modeling.SkyModel] = []
            for model in models:
                original_model_name = model.name.split("_A")[0]
                new_model_name = original_model_name + "_A_" + random_string
                model_copy = model.copy(name=new_model_name)
                model_copy_list.append(model_copy)

            return modeling.models.Models(model_copy_list)

        self.model = initialize_fitstart_model()

        self._n_fits = 0
        self._ml_fit_pool = _ProfileLRPool(max_pool_entries)

        """ 
        for lambda_ in np.logspace(np.log10(self.limit_target.parmin),
                                   np.log10(self.limit_target.parmax),
                                   max_pool_entries):
        """
        while len(self._ml_fit_pool) < max_pool_entries:
            info = "Filling ML fit pool: Current size: "
            info += str(len(self._ml_fit_pool)) + "/"
            info += str(max_pool_entries)
            self._logger.debug(info)

            lambda_ = stats.gamma.rvs(a=1.1, scale=1./(5. * 1.1), size=1)[0]
            self.fit(lambda_=lambda_)

    def _fitstart_model_copy(self):
        letters = string.ascii_uppercase + string.digits
            
        random_string = "".join(random.choice(letters) for _ in range(10))
        model_copy_list : List[modeling.SkyModel] = []
        for model in self.model:
            original_model_name = model.name.split("_A")[0]
            new_model_name = original_model_name + "_A_" + random_string
            model_copy = model.copy(name=new_model_name)
            model_copy_list.append(model_copy)

        return modeling.models.Models(model_copy_list)

    def fit(self, lambda_=None):
        self._n_fits += 1

        fit_parameters = {"n_repeat_fit_max": 3,
                          "print_level": 0,
                          "tolerance": 3.0,
                          "strategy": 2,
                          "backend": self.fit_backend}
        

        def initialize_dataset_fit_model():
            """Puts the right fit start model in place in the dataset.
            """
            dataset = self.dataset
            if len(self._ml_fit_pool) == 0:
                dataset.models = self._fitstart_model_copy()
            else:
                if lambda_ is None:
                    dataset.models = self._ml_fit_pool.ml_model()
                else:
                    dataset.models = self._ml_fit_pool.closest_model(lambda_)
            return dataset

        def freeze_target_parameter(frozen: bool):
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
                            parameter.value = lambda_
                        else:
                            parameter.frozen = False
                        break

        def show_fit_dataset_debug_information():
            """Prints debug information if in corresponding log-level.
            """
            self._logger.debug("---------------------------------------------")
            self._logger.debug("Fit input dataset:")
            info = "lambda_ input: " + str(lambda_)
            if lambda_ is not None:
                info += ": -> lambda_ should be frozen to this "
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

        def fit_dataset():
            """Fits the dataset to its model.

               Returns: FitResult
            """
            fit = modeling.Fit([dataset])
            with np.errstate(all="ignore"):
                result = fit.run(optimize_opts={
                        "print_level": fit_parameters["print_level"],
                        "tol": fit_parameters["tolerance"],
                        "strategy": fit_parameters["strategy"],
                        "backend": fit_parameters["backend"]})

                n_repeat_fit = 1
                while result.success is False:
                    if n_repeat_fit > fit_parameters["n_repeat_fit_max"]:
                        break
                    info = "Fit " + str(n_repeat_fit)
                    info += " didn't converge -> Repeating"
                    self._logger.debug(info)

                    n_repeat_fit += 1
                    fit = modeling.Fit([dataset])
                    result = fit.run(optimize_opts={
                            "print_level": fit_parameters["print_level"],
                            "tol": fit_parameters["tolerance"] + n_repeat_fit,
                            "strategy": fit_parameters["strategy"],
                            "backend": fit_parameters["backend"]})
                if result.success is False:
                    if lambda_ is None:
                        info = "Cannot fit full model with free target "
                        info += "parameter"
                        raise RuntimeError(info)
                    else:
                        info = "Fit failed for fixed lambda=" + str(lambda_)
                        info += " (this is typically no problem)"
                        self._logger.debug(info)

                return result

        def _best_fit_parameter(fit_result):
            """Extracts the best fit target parameter out of the fit result.

               Returns:
                best_fit_parameter (float)
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

        def fill_ml_fit_pool(best_fit_parameter):
            """Fills the ML fit pool.
            """
            if best_fit_parameter is not None and not self._ml_fit_pool.full:
                self._ml_fit_pool.append(best_fit_parameter, dataset)


        def print_afterfit_debug():
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

            info = str(self._n_fits) + ": Fit result for input lambda="
            info += str(lambda_) + " -> fitstat: " + str(fitstat)
            self._logger.debug(info)

        dataset = initialize_dataset_fit_model()
       
        if lambda_ is not None:
            freeze_target_parameter(frozen=True)
        else:
            freeze_target_parameter(frozen=False)
        
        show_fit_dataset_debug_information()

        fit_result = fit_dataset()
        best_fit_parameter = _best_fit_parameter(fit_result)
        fill_ml_fit_pool(best_fit_parameter)

        fitstat = dataset.stat_sum()
        print_afterfit_debug()

        dataset.models = None # Clear models after each fit.
        
        return best_fit_parameter, fitstat

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
        info = "fitstat_pl=" + str(fitstat_pl)
        info += ", fitstat_ml=" + str(fitstat0)
        self._logger.debug(info)

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
                self._logger.debug("All brentq solvers failed. Using fsolve ...")
                ul, info, ier, mesg = fsolve(helper_func,
                                             x0=ml_fit_lambda,
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
    def __init__(self,
                 limit_target,
                 data,
                 models,
                 CL: float,
                 max_pool_entries : int =20,
                 fit_backend : str ="minuit"):

        super().__init__(limit_target, data, models,
                         CL, fit_backend, max_pool_entries)

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

        info = "ML fit lambda from unconstrained fit: "
        info += str(ml_fit_lambda)
        self._logger.debug(info)

        if ml_fit_lambda < 0.:
            self._logger.debug("Setting ML fit lambda=0")
            ml_fit_lambda = 0.

        if ml_fit_lambda > self.limit_target.parmax:
            info = "ml_fit_lambda=" + str(ml_fit_lambda)
            info += " out of range: Increase lambda_max"
            raise RuntimeError(info)

        setattr(self, "_ml_fit_lambda", ml_fit_lambda)

        return ml_fit_lambda


class UnconstrainedLR(_LRBase):
    def __init__(self,
                 limit_target,
                 data,
                 models,
                 CL : float,
                 max_pool_entries : int = 20,
                 fit_backend : str = "minuit"):

        super().__init__(limit_target, data, models, CL,
                         fit_backend, max_pool_entries)
    @property
    def critical_value(self):
        return stats.chi2.ppf(self.CL, df=1)

    def ml_fit_lambda(self):
        if hasattr(self, "_ml_fit_lambda"):
            return self._ml_fit_lambda

        with np.errstate(all="ignore"):
            ml_fit_lambda, _ = self.fit()

        info = "ML fit lambda: "
        info += str(ml_fit_lambda)
        self._logger.debug(info)

        setattr(self, "_ml_fit_lambda", ml_fit_lambda)

        return ml_fit_lambda

    def ul(self):
        return self._ul
