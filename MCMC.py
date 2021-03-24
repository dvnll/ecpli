import numpy as np

import gammapy.modeling as modeling

import emcee

from gammapy.modeling.sampling import (
    par_to_model,
    uniform_prior,
)
import scipy.stats as stats

from ecpli.ECPLiBase import ECPLiBase, LimitTarget


class _EnsembleMCMCBase(ECPLiBase):
    """Base class for all affine invariant ensemble MCMC methods.

       Attributes:
        dataset: Gammapy dataset from which the model parameter constraint
                 is to be derived.
        burn_in: Number of samples to be burned.
        walker_factor: The number of walkers is this factor times the number
                       of free parameters.
        n_threads: Number of CPU threads used for sampling.
        n_samples: Number of samples to be drawn.
    """

    def __init__(self,
                 limit_target: LimitTarget,
                 data: modeling.Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 n_samples: int,
                 n_burn: int):
        """Args:
            limit_target: Instance of LimitTarget to specify parameter and
                          model to be constrained.
            data: Gammapy dataset from which the model parameter constraint
                  is to be derived.
            models: Models which are to be fit to the data.
            CL: Confidence level (e.g. 0.95) for the limit.
            n_samples: Number of samples to draw.
            n_burn: Number of burn in samples.
        """

        super().__init__(limit_target, data, models, CL)

        self.dataset = data.copy()
        self.dataset.models = models
        self.burn_in = n_burn
        self.walker_factor = 20
        self.n_threads = 1
        self.n_samples = n_samples

        if self.limit_target.parameter_name != "lambda_":
            info = "Currently only supporting limits on lambda_"
            raise RuntimeError(info)

    @property
    def _lambda_index(self) -> int:
        """Returns the index of the target parameter in the
           self.dataset.models.parameters.free_parameters list.
        """

        index_counter = 0
        target_index = None
        for mod in self.dataset.models:
            for parameter in mod.parameters.free_parameters:
                if parameter.name == self.limit_target.parameter_name:
                    if mod.name == self.limit_target.model.name:
                        target_index = index_counter
                index_counter += 1

        if target_index is None:
            info = "Model " + self.target_model_name + " not found"
            raise RuntimeError(info)

        return target_index

    def _lambda_chain(self) -> np.ndarray:
        """Returns the list of samples of the target parameter.
        """

        burn_in = self.burn_in

        chain = self.sampler.chain

        sampler = np.zeros_like(chain)

        """
        chain.shape[0]: Number of walkers
        chain.shape[1]: Number of samples
        chain.shape[2]: Number of free parameters
        """
        free_parameter_list = self.dataset.models.parameters.free_parameters
        n_walkers = chain.shape[0]
        n_samples = chain.shape[1]

        for i_walker in range(n_walkers):
            for i_sample in range(n_samples):
                pars = chain[i_walker, i_sample, :]
                par_to_model(self.dataset, pars)
                par_values = [p.value for p in free_parameter_list]
                sampler[i_walker, i_sample, :] = par_values

        target_index = self._lambda_index
        samples_after_burn = sampler[:, burn_in:, target_index]

        result = samples_after_burn.flatten()

        negative_lambda = result[result < 0]
        if len(negative_lambda) > 0:
            info = "Have " + str(len(negative_lambda))
            info += " negative lambda samples"
            info += ": " + str(negative_lambda)
            self._logger.debug(info)
        else:
            info = "All " + str(len(result)) + " samples are positive"
            self._logger.debug(info)
        return result

    def _initial_samples(self) -> np.ndarray:
        """Runs the Markov chain initially for self.n_samples.
        """

        dataset = self.dataset

        self._logger.debug("Dataset before initial fit: ")
        self._logger.debug(dataset)

        self._logger.debug("Free parameters:")
        for par in dataset.models.parameters.free_parameters:
            info = par.name + ": value: " + str(par.value)
            info += ", min: " + str(par.min) + ", max: " + str(par.max)
            self._logger.debug(info)
        self._logger.debug("---------------------------------------------")

        fit_mcmc = modeling.Fit([dataset])
        _ = fit_mcmc.run()

        self._logger.debug("Dataset after initial fit:")
        self._logger.debug(dataset)
        self._logger.debug("Free parameters:")
        for par in dataset.models.parameters.free_parameters:
            info = par.name + ": value: " + str(par.value)
            info += ", min: " + str(par.min) + ", max: " + str(par.max)
            self._logger.debug(info)
        self._logger.debug("---------------------------------------------")

        nwalk = self.walker_factor
        nwalk *= len(dataset.models.parameters.free_parameters)

        walker_positions = self._sample(n_walkers=nwalk,
                                        n_samples=self.n_samples)

        return walker_positions

    def autocorrelation(self) -> float:
        """Estimates the autocorrelation time of the Markov chain.
        """
        tau = None
        walker_positions = self._initial_samples()

        finish = False
        n_trials = 0
        n_trials_max = 100
        while not finish:
            if n_trials > 0:
                n_samples = self.sampler.chain.shape[1]
                print("Getting " + str(n_samples) + " more samples")
                walker_positions = self._continue_sampling(walker_positions,
                                                           n_samples)

            try:
                with np.errstate(all="ignore"):
                    tau = self.sampler.get_autocorr_time()
                info = "Estimated autocorrelation time: " + str(tau)
                print(info)
                finish = True

            except Exception as e:
                info = "Couldn't estimate autocorrelation time: "
                info += str(e)
                self._logger.debug(info)

                n_trials += 1
                info = "Tried " + str(n_trials) + " times. Will retry ..."
                self._logger.debug(info)
                if n_trials > n_trials_max:
                    break

            continue
        return tau

    @property
    def ul(self) -> float:
        """Returns the upper limit on the LimitTarget.
        """

        info = "Burn in: " + str(self.burn_in)
        self._logger.debug(info)
        info = "Walker factor: " + str(self.walker_factor)
        self._logger.debug(info)
        info = "Number of samples/nrun: " + str(self.n_samples)
        self._logger.debug(info)

        _ = self._initial_samples()

        samp_lambda = self._lambda_chain()

        lambda_ul = np.quantile(samp_lambda, self.CL)

        info = "Mean acceptance fraction over all walkers: "
        info += str(np.mean(self.sampler.acceptance_fraction))
        self._logger.debug(info)

        return lambda_ul

    def ul_precision(self, n_chains: int = 10):
        """Estimate the precision of the limit given the configuration,
           e.g. with regards to the number of samples.
           The precision is estimated as standard deviation between
           the limit derived in different and fully independent
           Markov chains with equal configuration.
        """

        ul_list = []
        for i in range(n_chains):
            info = "Calculating limit " + str(i+1)
            info += "/" + str(n_chains)
            self._logger.debug(info)
            method_name = globals()[type(self).__name__]
            method = method_name(self.dataset,
                                 self.limit_target.model,
                                 self.limit_target.parameter_name,
                                 n_samples=self.n_samples,
                                 n_burn=self.burn_in)

            ul_list.append(method.ul)

        std = np.std(ul_list, ddof=1)
        relative_std = std / np.mean(ul_list) * 100.
        return std, relative_std, ul_list

    def _prior(self):
        return NotImplementedError

    def _lnposterior(self, pars: np.ndarray, dataset: modeling.Dataset):
        free_pars = dataset.models.parameters.free_parameters
        for factor, par in zip(pars, free_pars):
            par.factor = factor

        prior = self._prior()
        log_likelihood = -0.5 * dataset.stat_sum()
        """
        note the difference to the gammapy implementation (factor 0.5)!
        """
        return log_likelihood + prior

    def _walker_init(self, n_walkers: int):
        """
        Initialize walkers as ball around current values.
        """
        parameter_variance = []
        parameter = []
        spread = 0.5 / 100.
        spread_pos = 0.1  # in degrees
        for par in self.dataset.parameters.free_parameters:
            parameter.append(par.factor)
            if par.name in ["lon_0", "lat_0"]:
                parameter_variance.append(spread_pos / par.scale)
            else:
                parameter_variance.append(spread * par.factor)

        """
        Produce a ball of inital walkers around the inital values p0
        """
        p0 = emcee.utils.sample_ball(parameter, parameter_variance, n_walkers)
        ndim = len(parameter)
        return ndim, p0

    def _sample(self, n_walkers: int, n_samples: int) -> np.ndarray:
        """Draws n_samples from the markov chain.
        """
        self.dataset.parameters.autoscale()
        ndim, p0 = self._walker_init(n_walkers)

        sampler = emcee.EnsembleSampler(n_walkers, ndim, self._lnposterior,
                                        args=[self.dataset],
                                        threads=self.n_threads)

        info = "Starting MCMC sampling: "
        info += "# walkers=" + str(n_walkers)
        info += ", # samples=" + str(n_samples)
        self._logger.debug(info)

        for idx, result in enumerate(sampler.sample(p0, iterations=n_samples)):
            if idx % (n_samples / 10) == 0:
                info = "Sampled " + str(round(idx/n_samples * 100)) + " %"
                self._logger.debug(info)
            walker_positions = result[0]
        self._logger.debug("Sampling complete")

        self.sampler = sampler
        return walker_positions

    def _continue_sampling(self,
                           walker_positions: np.ndarray,
                           n_samples: int) -> np.ndarray:
        """Continues sampling for another n_samples.
        """

        info = "Continuing to get " + str(n_samples) + " samples"
        self._logger.debug(info)

        for idx, result in enumerate(self.sampler.sample(walker_positions,
                                                         iterations=n_samples)
                                     ):
            if idx % (n_samples / 10) == 0:
                info = "Sampled " + str(round(idx / n_samples * 100)) + " %"
                self._logger.debug(info)
            walker_positions = result[0]

        self._logger.debug("Sampling complete")

        return walker_positions

    def trace(self, burn: bool):

        def _tracedata():

            tracedata_list = []
            target_index_list = []

            i = 0
            for mod in self.dataset.models:
                for parameter in mod.parameters.free_parameters:
                    _tracedata = {"model_name": mod.name,
                                  "parameter_name": parameter.name,
                                  "burn": self.burn_in}

                    target_index_list.append(i)
                    tracedata_list.append(_tracedata)
                    i += 1

            chain = self.sampler.chain

            sampler = np.zeros_like(chain)

            models = self.dataset.models
            free_parameter_list = models.parameters.free_parameters
            n_walkers = chain.shape[0]
            n_samples = chain.shape[1]

            for i_walker in range(n_walkers):
                for i_sample in range(n_samples):
                    pars = chain[i_walker, i_sample, :]
                    par_to_model(self.dataset, pars)
                    par_values = [p.value for p in free_parameter_list]
                    sampler[i_walker, i_sample, :] = par_values

            for target_index in target_index_list:
                samples_before_burn = sampler[:, :, target_index]
                tracedata_list[target_index][
                                    "trace_unburned"] = samples_before_burn

            return tracedata_list

        _trace_list = []
        for tracedata in _tracedata():
            trace = tracedata["trace_unburned"].transpose()
            """
            After transpose: Now trace is an array of dimension
            (n_walkers, n_runs),
            so each row has the data for all walker positions in a sampler step
            """
            trace = trace.flatten()

            """
            After flattening: Now trace is of the form
            (walker1_step1, walker2_step1, ...,
             walker1_step2, walker2_step2, ...)
            -> Need to burn the first burn_in * n_walker entries
            """
            if burn:
                trace = trace[self.burn_in * self.n_walker:]

            _trace_list.append(trace)

        return _trace_list


class UniformPriorEnsembleMCMC(_EnsembleMCMCBase):
    """Ensemble MCMC with uniform priors.
    """

    def __init__(self,
                 limit_target: LimitTarget,
                 data: modeling.Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 n_samples: int,
                 n_burn: int):

        super().__init__(limit_target, data, models, CL,
                         n_samples, n_burn)

        for par in self.dataset.parameters.free_parameters:
            if par.name == self.limit_target.parameter_name:
                par.min = self.limit_target.parmin
                par.max = self.limit_target.parmax
            elif par.name == "index":
                par.min = 1.
                par.max = 4.
            elif par.name == "amplitude":
                par.max = 1.e-7
                par.min = 1.e-16
            else:
                print("Freezing " + par.name)
                par.frozen = True

    def _prior(self):
        logprob = 0.
        for par in self.dataset.models.parameters.free_parameters:
            logprob += uniform_prior(par.value, par.min, par.max)
        return logprob


class WeakPriorEnsembleMCMC(_EnsembleMCMCBase):
    """Ensemble MCMC with gamma distributed priors.
    """

    def __init__(self,
                 limit_target: LimitTarget,
                 data: modeling.Dataset,
                 models: modeling.models.Models,
                 CL: float,
                 n_samples: int,
                 n_burn: int):

        super().__init__(limit_target, data, models, CL,
                         n_samples, n_burn)

        # Fixing only for the ML fit
        for par in self.dataset.parameters.free_parameters:
            if par.name == self.limit_target.parameter_name:
                par.min = self.limit_target.parmin
                par.max = self.limit_target.parmax
            elif par.name == "index":
                par.min = 1.
                par.max = 4.
            elif par.name == "amplitude":
                par.max = 1.e-7
                par.min = 1.e-16
            else:
                print("Freezing " + par.name)
                par.frozen = True

    def _prior(self):
        logprob = 0
        for par in self.dataset.models.parameters.free_parameters:
            if par.name == "lambda_":
                shape = 1.1
                mean = 1. / 5.
                scale = mean / shape
                prior = stats.gamma
                prior_value = prior.pdf(x=par.value, a=shape, scale=scale)

            elif par.name == "index":
                shape = 7
                mean = 2.5
                scale = mean / shape
                prior = stats.gamma
                prior_value = prior.pdf(x=par.value, a=shape, scale=scale)

            elif par.name == "amplitude":
                shape = 1.3
                mean = 5.e-11
                shape = mean / shape
                prior = stats.gamma
                prior_value = prior.pdf(x=par.value, a=shape, scale=scale)

            else:
                raise RuntimeError("Unknown parameter " + str(par.value))

            if prior_value <= 0.:
                log_prior = -np.inf
            else:
                log_prior = np.log(prior_value)
            logprob += log_prior
        return logprob
