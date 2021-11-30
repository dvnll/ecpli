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
