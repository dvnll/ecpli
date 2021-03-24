Derive limits on the energy cutoff of a spectral model in gammapy-0.16.

Three methods are supported:
- Profile likelihood
- Bootstrap
- Affine invariant Markov chain

Given a gammapy dataset of type gammapy.modeling.Dataset, the target model and 
variable needs to be defined:


```
import ecpli
limit_target = ecpli.LimitTarget(model=dataset["dataset"].fit_start_model[0],
                                 parameter_name="lambda_",
                                 parmin=0.,
                                 parmax=1./0.05)
```

Afterwards, a frequentist upper limit on this variable can be derived. Given a
confidence level CL and a set of gammapy models for the dataset (of type
modeling.models.Models), e.g. a profile likelihood limit is derived as:

```
method = ecpli.ConstrainedLR(limit_target, dataset, models, CL)
ul = method.ul
```
