This is an ecpli branch for gammapy-0.18.2. Note that affine invariant Markov chain and Bootstrap methods are currently only implemented in the ecpli branch for gammapy-0.16.

Installation

1) If not already installed: Install [gammapy-0.18.2](https://docs.gammapy.org/0.18/install/index.html)

2) Clone the ecpli branch:
```
git clone -b v0.18.2 https://github.com/residualsilence/ecpli.git
```
Clone the gammapy dataset:
```
git clone https://github.com/gammapy/gammapy-data
```
and set the GAMMAPY_DATA environment variable, e.g.:
```
export GAMMAPY_DATA=$PWD/gammapy-data
```

For a small test case, check out the [gammapy 3d-analysis notepad](https://docs.gammapy.org/0.18/_static/notebooks/analysis_3d.ipynb).

At the bottom of the notepad, add the following cells:

1) Copy the initial models:
```
models_joint2 = Models()

model_joint2 = model.copy(name="source-joint2")
models_joint2.append(model_joint)

for dataset in analysis_joint.datasets:
    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    models_joint2.append(bkg_model)
```

2) Run a few ecpli functions:

```
import ecpli

limit_target = ecpli.LimitTarget(model=models_joint2[0],
                                 parameter_name="lambda_",
                                 parmin=0.,
                                 parmax=1./0.05)


method = ecpli.ConstrainedLR(limit_target=limit_target,
                             dataset=dataset,
                             CL=0.95)

ul = method.ul

print("UL on lambda: " + str(ul))
print("LL on energy cutoff: " + str(1/ul))

print("TS for cutoff: " + str(method.cutoff_ts()))
print("Significance of cutoff: " + str(method.cutoff_significance()))

print("PTS for 100 TeV gamma PeVatron: " + str(method.pts(threshold_energy=100. * u.TeV)))
print("PTS significance for 100 TeV gamma PeVatron: " + str(method.pts_significance(threshold_energy=100. * u.TeV)))
```

3) ecpli methods can also be applied to flux point data:

```
from astropy import units as u
from astropy.table import Table
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    PowerLawSpectralModel,
    SkyModel
)
from gammapy.datasets import (
    FluxPointsDataset,
)
import numpy as np
import os
import ecpli

# ECPLI for an ASCI table with flux points:

print(os.environ["GAMMAPY_DATA"])

table = Table.read(os.environ["GAMMAPY_DATA"] + '/tests/spectrum/flux_points/flux_points_ctb_37b.txt',
                   format='ascii.csv', delimiter=' ', comment='#')

table.meta['SED_TYPE'] = 'dnde'
table.rename_column('Differential_Flux', 'dnde')
table['dnde'].unit = 'cm-2 s-1 TeV-1'

table.rename_column('lower_error', 'dnde_errn')
table['dnde_errn'].unit = 'cm-2 s-1 TeV-1'

table.rename_column('upper_error', 'dnde_errp')
table['dnde_errp'].unit = 'cm-2 s-1 TeV-1'

table['dnde_err'] = 0.5 * (table['dnde_errp'] - table['dnde_errn'])
table.rename_column('E', 'e_ref')
table['e_ref'].unit = 'TeV'

flux_points = FluxPoints(table)
flux_points.plot()

f_spectral_model = ExpCutoffPowerLawSpectralModel(
        index=2.0,
        amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
        reference=1 * u.TeV,
        lambda_=1./20 * u.Unit("TeV-1"),
        alpha=1
    )

fmodel = SkyModel(spectral_model=f_spectral_model, name="ecpl_model")
fmodel.parameters["alpha"].frozen = True
print(fmodel)
dataset = FluxPointsDataset(fmodel, flux_points)

limit_target = ecpli.LimitTarget(model=fmodel,
                                 parameter_name="lambda_",
                                 parmin=0.,
                                 parmax=1./0.05)


method = ecpli.ConstrainedLR(limit_target=limit_target,
                             dataset=dataset,
                             CL=0.95)

ul = method.ul

print("UL on lambda: " + str(ul))
print("LL on energy cutoff: " + str(1/ul))

print("TS for cutoff: " + str(method.cutoff_ts()))
print("Significance of cutoff: " + str(method.cutoff_significance()))

print("PTS for 100 TeV gamma PeVatron: " + str(method.pts(threshold_energy=100. * u.TeV)))
print("PTS significance for 100 TeV gamma PeVatron: " + str(method.pts_significance(threshold_energy=100. * u.TeV)))

# Table with simulated true PeVatron flux points:

table = Table()
pwl = PowerLawSpectralModel()
e_ref = np.logspace(0, 2.5, 10) * u.TeV
table['e_ref'] = e_ref
true_pevatron_flux = pwl(e_ref)
table['dnde_true'] = true_pevatron_flux
table['dnde'] = table['dnde_true']
relative_flux_error = 0.3
table['dnde'] = np.random.normal(table['dnde_true'].data,
                                 relative_flux_error * table['dnde_true'].data) * true_pevatron_flux.unit
table['dnde_err'] = relative_flux_error * pwl(e_ref)

table.meta['SED_TYPE'] = 'dnde'

flux_points = FluxPoints(table)
flux_points.plot()

f_spectral_model = ExpCutoffPowerLawSpectralModel(
        index=2.0,
        amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
        reference=1 * u.TeV,
        lambda_=1./20 * u.Unit("TeV-1"),
        alpha=1
    )


fmodel = SkyModel(spectral_model=f_spectral_model, name="ecpl_model")
fmodel.parameters["alpha"].frozen = True
dataset = FluxPointsDataset(fmodel, flux_points)

limit_target = ecpli.LimitTarget(model=fmodel,
                                 parameter_name="lambda_",
                                 parmin=0.,
                                 parmax=1./0.05)


method = ecpli.ConstrainedLR(limit_target=limit_target,
                             dataset=dataset,
                             CL=0.95)

ul = method.ul

print("UL on lambda: " + str(ul))
print("LL on energy cutoff: " + str(1/ul))

print("TS for cutoff: " + str(method.cutoff_ts()))
print("Significance of cutoff: " + str(method.cutoff_significance()))

print("PTS for 100 TeV gamma PeVatron: " + str(method.pts(threshold_energy=100. * u.TeV)))
print("PTS significance for 100 TeV gamma PeVatron: " + str(method.pts_significance(threshold_energy=100. * u.TeV)))
```
