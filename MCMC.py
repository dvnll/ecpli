import numpy as np
#import seaborn as sns

from gammapy.modeling import Fit

import emcee

from gammapy.modeling.sampling import (
    par_to_model,
    uniform_prior,
)
import scipy.stats as stats

import random
import string

import matplotlib.pyplot as plt
from ecpli.ECPLiBase import ECPLiBase


class _EnsembleMCMCBase(ECPLiBase):

    def __init__(self, limit_target, data, models, CL,
                 verbose, n_correlation):
        super().__init__(limit_target, data, models, CL)

        self.dataset = data.copy()
        self.model = models[0]  # [m.copy() for m in model]
        self.verbose = verbose
        self.dataset.models = self.model
        self.n_correlation = n_correlation
        self.burn_in = 3 * n_correlation
        self.walker_factor = 20
        self.nrun = 20 * self.n_correlation

    @property
    def _lambda_index(self):

        index_counter = 0
        target_index = None
        for mod in self.dataset.models:
            for parameter in mod.parameters.free_parameters:
                if parameter.name == "lambda_":
                    if mod.name == self.limit_target.model.name:
                        target_index = index_counter
                index_counter += 1

        if target_index is None:
            info = "Model " + self.target_model_name + " not found"
            raise RuntimeError(info)

        return target_index

    def _lambda_chain(self, independent=False):

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
        if independent:
            """
            Return only every n_correlation'th element for every walker
            """
            result = samples_after_burn[:, ::self.n_correlation].flatten()


        if self.verbose:
            negative_lambda = result[result<0]
            if len(negative_lambda) > 0:
                info = "Have " + str(len(negative_lambda))
                info += " negative lambda samples"
                info += ": " + str(negative_lambda)
                print(info)
                #raise RuntimeError(info)
            else:
                info = "All " + str(len(result)) + " samples are positive"
                print(info)
        return result

    def _initial_samples(self):

        nrun = self.nrun
        threads = 1

        dataset = self.dataset#.copy()

        if self.verbose:
            print("Dataset before initial fit: ")
            print(dataset)

            print("Free parameters:")
            for par in dataset.models.parameters.free_parameters:
                info = par.name + ": value: " + str(par.value)
                info += ", min: " + str(par.min) + ", max: " + str(par.max)
                print(info)
            print("---------------------------------------------")

        fit_mcmc = Fit([dataset])
        _ = fit_mcmc.run()
        if self.verbose:
            print("Dataset after initial fit:")
            print(dataset)
            print("Free parameters:")
            for par in dataset.models.parameters.free_parameters:
                info = par.name + ": value: " + str(par.value)
                info += ", min: " + str(par.min) + ", max: " + str(par.max)
                print(info)
            print("---------------------------------------------")


        nwalk = self.walker_factor * len(dataset.models.parameters.free_parameters)

        walker_positions = self._sample(n_walkers=nwalk,
                                        n_samples=nrun,
                                        n_threads=threads)

        return walker_positions

    def autocorrelation(self):
        tau = None
        walker_positions = self._initial_samples()

        finish = False
        n_trials = 0
        n_trials_max = 100
        self.verbose = True
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
                if self.verbose:
                    info = "Couldn't estimate autocorrelation time: "
                    info += str(e)
                    print(info)

                n_trials += 1
                if self.verbose:
                    info = "Tried " + str(n_trials) + " times. Will retry ..."
                    print(info)
                if n_trials > n_trials_max:
                    break

            continue
        return tau

    @property
    def tracedata(self):
        if hasattr(self, "_tracedata"):
            return self._tracedata

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

        free_parameter_list = self.dataset.models.parameters.free_parameters
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
            tracedata_list[target_index]["trace_unburned"] = samples_before_burn

        self._tracedata = tracedata_list

        return tracedata_list
    
    @property                       
    def ul(self):
        if self.verbose:
            info = "Burn in: " + str(self.burn_in)
            print(info)
            info = "Walker factor: " + str(self.walker_factor)
            print(info)
            info = "Number of samples/nrun: " + str(self.nrun)
            print(info)
 
        save_lambda_posterior = False
        walker_positions = self._initial_samples()

        samp_lambda = self._lambda_chain()

        lambda_ul = np.quantile(samp_lambda, self.CL)

        info = "Mean acceptance fraction over all walkers: "
        info += str(np.mean(self.sampler.acceptance_fraction))

        if self.verbose:
            print(info)

        return lambda_ul

    def ul_precision(self, dataset, model, 
                     target_model_name, 
                     n_chains=10):

        ul_list = []
        for i in range(n_chains):
            info = "Calculating limit " + str(i+1)
            info += "/" + str(n_chains)
            print(info)
            method_name = globals()[type(self).__name__]
            method = method_name(dataset,
                                 model,
                                 target_model_name,
                                 verbose=False,
                                 n_correlation=self.n_correlation)

            ul_list.append(method.ul)

        std = np.std(ul_list, ddof=1)
        relative_std = std / np.mean(ul_list) * 100.
        return std, relative_std, ul_list

    def _prior(self):
        return NotImplementedError

    def _lnposterior(self, pars, dataset):

        free_pars = dataset.models.parameters.free_parameters
        for factor, par in zip(pars, free_pars):
            par.factor = factor

        prior = self._prior()
        log_likelihood = -0.5 * dataset.stat_sum()
        """
        note the difference to the gammapy implementation (factor 0.5)!
        """
        return log_likelihood + prior

    def _walker_init(self, n_walkers):
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
        if self.verbose:
            print(f"parameter = {parameter}")
            print(f"parameter_variance = {parameter_variance}")
            print("lenp0=" + str(len(p0)))
            print(p0)
        ndim = len(parameter)
        return ndim, p0

    def _sample(self, n_walkers, n_samples, n_threads):

        self.dataset.parameters.autoscale()
        ndim, p0 = self._walker_init(n_walkers)

        sampler = emcee.EnsembleSampler(n_walkers, ndim, self._lnposterior,
                                        args=[self.dataset], threads=n_threads)

        info = "Starting MCMC sampling: "
        info += "nwalkers=" + str(n_walkers)
        info += "n_samples=" + str(n_samples)
        print(info)

        for idx, result in enumerate(sampler.sample(p0, iterations=n_samples)):
            if idx % (n_samples / 10) == 0:
                print("{:5.0%}".format(idx / n_samples))
            walker_positions = result[0]
            # print(walker_positions)
            # print("----------------------")
        print("100% => sampling completed")

        self.sampler = sampler
        return walker_positions

    def _continue_sampling(self, walker_positions, n_samples):

        print(f"CONTINUE MCMC sampling: nrun={n_samples}")
        for idx, result in enumerate(self.sampler.sample(walker_positions,
                                                         iterations=n_samples)
                                     ):
            if idx % (n_samples / 10) == 0:
                print("{:5.0%}".format(idx / n_samples))
            walker_positions = result[0]
        print("100% => sampling completed")

        return walker_positions

    def trace_analysis(self, trace_outfile):

        tracedata_list = self.tracedata()
        method = type(self).__name__

        trace_analysis = TraceDataResult(method, tracedata_list)
        trace_analysis.plot_traces(trace_outfile)

    @property
    def parameter_names(self):
        parameter_name_list = []
        for tracedata in self.tracedata:
            parameter_name = tracedata["parameter_name"]
            parameter_name_list.append(parameter_name)
        
        return parameter_name_list

    @property
    def model_names(self):
        model_name_list = []
        for tracedata in self.tracedata:
            model_name = tracedata["model_name"]
            model_name_list.append(model_name)
        
        return model_name_list

    def trace(self, burn: bool):
        _trace_list = []
        for tracedata in self.tracedata:
            trace = tracedata["trace_unburned"].transpose()
            """
            After transpose: Now trace is an array of dimension (n_walkers, n_runs),
            so each row has the data for all walker positions in a sampler step
            """
            trace = trace.flatten()
            
            """
            After flattening: Now trace is of the form 
            (walker1_step1, walker2_step1, ..., walker1_step2, walker2_step2, ...)
            -> Need to burn the first burn_in * n_walker entries
            """
            if burn:
                trace = trace[self.burn_in * self.n_walker:]
    
            _trace_list.append(trace)
        
        return _trace_list
    
    def plot_traces(self, outname=None):
        plt.figure(figsize=(16, 9))
        parameter_list = self.parameter_names
        model_list = self.model_names
        trace_list = self.trace(burn=False)
        fig, axs = plt.subplots(nrows=len(trace_list),
                                ncols=1,
                                sharex=True,
                                figsize=(5 * len(trace_list), 16))

        fig.subplots_adjust(hspace=.01)
        i = 0

        for parameter, model, trace in zip(parameter_list, model_list, trace_list):

            burn = self.burn_in * self.n_walker
            axs[i].axvspan(0, burn, alpha=0.1, color="r")
            axs[i].plot(np.linspace(0, len(trace), len(trace)), trace)
            axs[i].set_title(self.method_name + ": " + model + "-" + parameter)
            i += 1
            
        if outname is not None:
            plt.savefig(outname)

    def scatterplot(self, model_parameter1, model_parameter2, outname=None, kind="scatter"):
        parameter_list = self.parameter_names
        model_list = self.model_names
        trace_list = self.trace(burn=True)

        trace1 = None
        trace2 = None
        burn = self.burn_in * self.n_walker
        
        for parameter, model, trace in zip(parameter_list, model_list, trace_list):
            if (model == model_parameter1[0]) and (parameter == model_parameter1[1]):
                trace1 = trace[burn:]
            if (model == model_parameter2[0]) and (parameter == model_parameter2[1]):
                trace2 = trace[burn:]
        
        if trace1 is None:
            info = str(model_parameter1) + " not found! "
            info += " Available models: " + str(self.model_names)
            info += ", Available parameters: " + str(self.parameter_names)
            raise RuntimeError(info)
        if trace2 is None:
            info = str(model_parameter2) + " not found! "
            info += " Available models: " + str(self.model_names)
            info += ", Available parameters: " + str(self.parameter_names)
            raise RuntimeError(info)
            
        xlabel = model_parameter1[0] + "-" + model_parameter1[1]
        ylabel = model_parameter2[0] + "-" + model_parameter2[1]

        data = {xlabel: trace1, ylabel: trace2}
        h = sns.jointplot(data=data, x=xlabel, y=ylabel, kind=kind, height=16)
        h.set_axis_labels(xlabel, ylabel, fontsize=20)
        h.ax_joint.plot(np.mean(trace1), np.mean(trace2), marker="o", color="r", label="Data mean")
        if outname is not None:
            h.savefig(outname)

        return h.ax_joint


class UniformPriorEnsembleMCMC(_EnsembleMCMCBase):

    def __init__(self, limit_target, data, models, CL,
                 verbose, n_correlation):
        super().__init__(limit_target, data, models, CL,
                         verbose, n_correlation)
        

        for par in self.dataset.parameters.free_parameters:
            if par.name == "lambda_":
                par.min = 0.
                par.max = 1.
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

    def __init__(self, limit_target, data, models, CL,
                 verbose, n_correlation):
        super().__init__(limit_target, data, models, CL,
                         verbose, n_correlation)        

        # Fixing only for the ML fit
        for par in self.dataset.parameters.free_parameters:
            if par.name == "lambda_":
                par.min = 0
                par.max = 1.
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
