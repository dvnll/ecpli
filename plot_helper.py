#import matplotlib.pyplot as plt
import ecpli
import numpy as np
try:
    import seaborn as sns
except ImportError as e:
    info = "Need seaborn for plotting on top "
    info = "of the gammapy-0.16 environment. "
    info = "install with conda install seaborn."
    raise ImportError(info + " - " + str(e))


def plot_profile_likelihood(plt,
                            lr_limit: ecpli.LRBase,
                            lambda_true=None):
    ts = lr_limit.ts()
    title = lr_limit.__class__.__name__ + ": TS=" + str(round(ts, 1))
    plt.title(title, fontsize=30, loc="left")
    
    plt.plot(lr_limit._ml_fit_pool.parameter_list,
             lr_limit._ml_fit_pool.likelihood_list, color="b")

    plt.axvline(lr_limit.ml_fit_parameter(), label=r"ML $\lambda$",
                color="r", ls="--")
   
    if lambda_true is not None:
        plt.axvline(lambda_true, label=r"True $\lambda$",
                    color="g", ls="--")

    plt.legend(fontsize="xx-large")
    plt.xscale("log")
    plt.xlabel(r"$\lambda$ ($\mathrm{TeV^{-1}}$)", fontsize=30)
    plt.ylabel(r"-2$\ln\mathcal{L}$", fontsize=30)


def plot_ts(plt,
            lr_limit: ecpli.LRBase,
            lambda_true=None,
            lambda_ul=None):
    
    ml_lambda, max_likelihood = lr_limit.fit()
    
    likelihood_list = lr_limit._ml_fit_pool.likelihood_list
    f_list = likelihood_list - max_likelihood
    lambda_list = lr_limit._ml_fit_pool.parameter_list
    index_list = np.argsort(lambda_list)
    lambda_list = lambda_list[index_list]
    f_list = f_list[index_list]
    plt.plot(lambda_list, f_list, color="r")
    critical_value = lr_limit.critical_value
    plt.axvline(lr_limit.ml_fit_parameter(),
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
