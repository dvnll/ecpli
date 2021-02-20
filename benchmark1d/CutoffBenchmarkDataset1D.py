import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion

from gammapy.spectrum import SpectrumDataset, SpectrumDatasetMaker
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    SkyModel,
    Models
)
from gammapy.irf import load_cta_irfs
from gammapy.data import Observation
from ecpli.ECPLiBase import mCrab


class CutoffBenchmarkDataset1D(object):

    def __init__(self, lambda_true, index_true, normalization_true, irf_file):

        normalization_true = normalization_true.to("cm-2 s-1 TeV-1")
        self.lambda_true = lambda_true
        self.index_true = index_true
        self.normalization_true = normalization_true

        livetime = 10 * u.h
        pointing = SkyCoord(0, 0, unit="deg", frame="galactic")

        e_reco_min = 0.3
        e_reco_max = 150.

        self.energy_axis = np.logspace(np.log10(e_reco_min),
                                       np.log10(e_reco_max), 10) * u.TeV
        
        on_region_radius = Angle("0.11 deg")
        self.on_region = CircleSkyRegion(center=pointing,
                                         radius=on_region_radius)


        irfs = load_cta_irfs(irf_file)

        self.obs = Observation.create(pointing=pointing,
                                      livetime=livetime,
                                      irfs=irfs)

    @property
    def true_model(self):
        model_simu = ExpCutoffPowerLawSpectralModel(
            index=self.index_true,
            amplitude=self.normalization_true,
            lambda_=self.lambda_true,
            reference=1 * u.TeV,
        )
    
        model = SkyModel(spectral_model=model_simu,
                         name="source")
    
        return model

    @property
    def fit_start_model(self):

        model_fit = ExpCutoffPowerLawSpectralModel(
                index=2.2,
                amplitude=1.3e-12 * u.Unit("cm-2 s-1 TeV-1"),
                lambda_=1./40 * u.Unit("TeV-1"),
                reference=1 * u.TeV
            )

        fit_start_model = SkyModel(spectral_model=model_fit,
                                   name="source")

        return Models([fit_start_model, ])


    def data(self):

        dataset_empty = SpectrumDataset.create(
                e_reco=self.energy_axis,
                e_true=self.energy_axis,
                region=self.on_region
            )
        maker = SpectrumDatasetMaker(
                containment_correction=False,
                selection=["background", "aeff", "edisp"]
            )
        dataset = maker.run(dataset_empty, self.obs)
        dataset.models = self.true_model
        dataset.fake()

        return dataset


if __name__ == "__main__":

    import argparse
    import pickle
    import string
    import random
    import os

    letters = string.ascii_uppercase + string.digits

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_benchmarks",
                        type=int,
                        help="Number of benchmark datasets per true parameter",
                        dest="NBENCH",
                        required=True)

    parser.add_argument("--outdir",
                        type=str,
                        help="Output directory",
                        dest="OUTDIR",
                        required=True)

    parser.add_argument("--index",
                        type=float,
                        help="Powerlaw index",
                        dest="INDEX",
                        default=2.3)

    parser.add_argument("--normalization",
                        type=float,
                        help="Flux normalization in mCrab",
                        dest="NORM",
                        default=25.0)

    parser.add_argument("--irf_file",
                        type=str,
                        help="CTA IRF file",
                        dest="IRF",
                        required=True)

    parser.add_argument("--ecut_true_min",
                        type=float,
                        help="Minimal cutoff energy in TeV",
                        dest="ECUT_MIN",
                        default=5.)

    parser.add_argument("--ecut_true_max",
                        type=float,
                        help="Maximal cutoff energy in TeV",
                        dest="ECUT_MAX",
                        default=1000.)

    parser.add_argument("--n_ecut",
                        type=int,
                        help="Number of cutoff energies.",
                        dest="N_ECUT",
                        default=20)

    options = parser.parse_args()

    if options.ECUT_MAX < options.ECUT_MIN:
        info = "Maximal cutoff energy must be "
        info += "larger than minimal cutoff energy."
        raise RuntimeError(info)
    if options.N_ECUT <= 0:
        info = "Number of cutoff energies must be positive."
        raise RuntimeError(info)

    outdir = options.OUTDIR
    if outdir[-1] != "/":
        outdir += "/"

    if os.path.isdir(outdir):
        raise OSError(outdir + " already exists")

    os.mkdir(outdir)

    n_benchmarks = options.NBENCH

    ecut_true_list = np.logspace(np.log10(options.ECUT_MIN),
                                 np.log10(options.ECUT_MAX),
                                 options.N_ECUT) * u.Unit("TeV")
    index_true = options.INDEX
    normalization_true = options.NORM * mCrab

    for ecut_true in ecut_true_list:
        lambda_true = 1 / ecut_true
        dataset_parameter = {"name": "CutoffBenchmarkDataset1D",
                             "true_parameter": 
                                {"lambda_true": lambda_true,
                                 "index_true": index_true,
                                 "normalization_true": normalization_true}}

        for _ in range(n_benchmarks):
            dataset = CutoffBenchmarkDataset1D(irf_file=options.IRF,
                                    **dataset_parameter["true_parameter"])
            result = {"dataset": dataset}
            result["dataset_parameter"] = dataset_parameter
            result["data"] = dataset.data()

            random_string = "".join(random.choice(letters) for _ in range(10))

            _infile = outdir + "benchmark_" + random_string + ".pickle"
            with open(_infile, "wb") as fout:
                pickle.dump(result, fout)
