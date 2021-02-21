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
    """Dataset to benchmark 1-dimensional source analyses.

        Attributes:
            lambda_true: True inverse energy cutoff.
            index_true: True powerlaw index.
            normalization_true: True flux normalization.
            energy_axis: Energy axis used for reconstruction.
            on_region: Signal region.
            true_model: True source model.
            obs: Observation position. The telescope is assumed to
                 point directly towards the source.
            fit_start_model: Collection of start models (type Models) to
                             initialize the likelihood fit.
    """

    def __init__(self, lambda_true: u.quantity.Quantity,
                 index_true: float,
                 normalization_true: u.quantity.Quantity,
                 livetime: u.quantity.Quantity,
                 pointing_galactic: dict,
                 e_reco_binning: dict,
                 on_region_radius: str,
                 irf_file: str):

        normalization_true = normalization_true.to("cm-2 s-1 TeV-1")
        self.lambda_true = lambda_true
        self.index_true = index_true
        self.normalization_true = normalization_true

        pointing_l = pointing_galactic["pointing_l"]
        pointing_b = pointing_galactic["pointing_b"]
        pointing = SkyCoord(pointing_l, pointing_b,
                            unit="deg", frame="galactic")

        e_reco_min = e_reco_binning["e_reco_min"]
        e_reco_max = e_reco_binning["e_reco_max"]
        n_e_reco = e_reco_binning["n_e_reco"]
        self.energy_axis = np.logspace(np.log10(e_reco_min),
                                       np.log10(e_reco_max), n_e_reco) * u.TeV

        on_region_radius = Angle(on_region_radius)
        self.on_region = CircleSkyRegion(center=pointing,
                                         radius=on_region_radius)

        irfs = load_cta_irfs(irf_file)

        self.obs = Observation.create(pointing=pointing,
                                      livetime=livetime,
                                      irfs=irfs)

    @property
    def true_model(self) -> SkyModel:
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
    def fit_start_model(self) -> Models:

        model_fit = ExpCutoffPowerLawSpectralModel(
                index=2.2,
                amplitude=1.3e-12 * u.Unit("cm-2 s-1 TeV-1"),
                lambda_=1./40 * u.Unit("TeV-1"),
                reference=1 * u.TeV
            )

        fit_start_model = SkyModel(spectral_model=model_fit,
                                   name="source")

        return Models([fit_start_model, ])

    def data(self) -> SpectrumDataset:
        """Actual event data in form of a SpectrumDataset.
        """

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
    """Main method to create a set of 1-dimensional Monte-Carlo 
       data (SpectrumDataset) for a given set of true source parameters.
       Each Monte-Carlo source data is saved into a pickle-file. This file
       is to be analyzed with runecpli.py with regard to the limit on the
       exponential cutoff.
    """

    import argparse
    import pickle
    import string
    import random
    import os
    import json

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

    parser.add_argument("--config",
                        type=str,
                        help="Config file",
                        dest="CONFIG",
                        required=True)

    options = parser.parse_args()

    with open(options.CONFIG, "r") as fin:
        config = json.load(fin)

    ecut_binning = config["ecut_binning"]
    ecut_true_max = ecut_binning["ecut_true_max"]
    ecut_true_min = ecut_binning["ecut_true_min"]
    n_ecut = ecut_binning["n_ecut"]
    if ecut_true_max <= ecut_true_min:
        info = "Maximal cutoff energy must be "
        info += "larger than minimal cutoff energy."
        raise RuntimeError(info)

    if n_ecut <= 0:
        info = "Number of cutoff energies must be positive."
        raise RuntimeError(info)

    outdir = options.OUTDIR
    if outdir[-1] != "/":
        outdir += "/"

    if os.path.isdir(outdir):
        raise OSError(outdir + " already exists")

    os.mkdir(outdir)

    n_benchmarks = options.NBENCH

    ecut_true_list = np.logspace(
                np.log10(ecut_true_min),
                np.log10(ecut_true_max),
                n_ecut) * u.Unit("TeV")

    index_true = options.INDEX
    normalization_true = options.NORM * mCrab

    for ecut_true in ecut_true_list:
        lambda_true = 1 / ecut_true
        dataset_parameter = {"name": "CutoffBenchmarkDataset1D",
                             "true_parameter":
                             {"lambda_true": lambda_true,
                              "index_true": index_true,
                              "normalization_true": normalization_true,
                              "livetime": config["livetime"] * u.h,
                              "pointing_galactic": config["pointing_galactic"],
                              "e_reco_binning": config["e_reco_binning"],
                              "on_region_radius": config["on_region_radius"]}}

        for _ in range(n_benchmarks):
            dataset = CutoffBenchmarkDataset1D(
                                    irf_file=options.IRF,
                                    **dataset_parameter["true_parameter"])
            result = {"dataset": dataset}
            result["dataset_parameter"] = dataset_parameter
            result["data"] = dataset.data()

            random_string = "".join(random.choice(letters) for _ in range(10))
            dataset.fit_start_model
            _infile = outdir + "benchmark_" + random_string + ".pickle"
            with open(_infile, "wb") as fout:
                pickle.dump(result, fout)
