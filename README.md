Derive limits on the energy cutoff parameter 
of a spectral model in gammapy-0.16.

Three methods are supported:
- Profile likelihood
- Bootstrap
- Affine invariant Markov chain

Given a gammapy dataset (of type gammapy.modeling.Dataset),
the target model (of type gammapy.modeling.models.SkyModel) 
and target pameter name needs to be defined:


```
import ecpli
limit_target = ecpli.LimitTarget(model=target_model,
                                 parameter_name="lambda_",
                                 parmin=0.,
                                 parmax=1./0.05)
```

Afterwards, a frequentist upper limit on this variable can be derived. Given a
confidence level CL and a set of gammapy models for the dataset (of type
gammapy.modeling.models.Models), e.g. a profile likelihood limit is derived as:

```
method = ecpli.ConstrainedLR(limit_target, dataset, models, CL)
ul = method.ul
```

As a complete example, consider the "3d analysis" notepad
[notepad](https://docs.gammapy.org/0.16/notebooks/analysis_3d.html) from 
the official gammapy-0.16 documentation. Let the notepad run. As final cell,
insert

'''
import ecpli
from gammapy.modeling.models import Models

limit_target = ecpli.LimitTarget(model=model,
                                 parameter_name="lambda_",
                                 parmin=0.,
                                 parmax=1./0.05)

method = ecpli.ConstrainedLR(limit_target, dataset,  Models([model,]), CL=0.95)
ul = method.ul

print("UL on lambda: " + str(ul))
print("LL on energy cutoff: " + str(1/ul))
'''

Other implemented methods are 

- UnconstrainedLR
- UniformPriorEnsembleMCMC, WeakPriorEnsembleMCMC
- BestFitParametricBootstrap, PoissonParametricBootstrap, NonParametricBootstrap

These methods have a very similar API as ConstrainedLR in the example above.

The frequentist coverage of all provided methods is tested for typical
gamma-ray point sources and confirmed at a confidence level of 95%.
