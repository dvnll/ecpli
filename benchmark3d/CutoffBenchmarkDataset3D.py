from astropy.coordinates import SkyCoord
from gammapy.modeling.models import SkyModel, Models
from gammapy.cube import MapDataset, MapDatasetMaker, SafeMaskMaker
from gammapy.modeling.models import PointSpatialModel
from gammapy.maps import WcsGeom, MapAxis
import astropy.units as u
from gammapy.modeling.models import (
        ExpCutoffPowerLawSpectralModel
    )
from gammapy.data import Observation

import time
import numpy as np
from scipy.optimize import fsolve
from gammapy.irf import load_cta_irfs
from ecpli.ECPLiBase import mCrab


class CutoffBenchmarkDataset3D(object):
    def __init__(self, *,
                 lambda_true: u.quantity.Quantity,
                 normalization_true: u.quantity.Quantity,
                 index_true: u.quantity.Quantity,
                 offset: u.quantity.Quantity,
                 source_pos_radec: dict,
                 e_reco_binning: dict,
                 map_geom: dict,
                 livetime: u.quantity.Quantity,
                 irf_file: str,
                 random_state: int):

        self.lambda_true = lambda_true
        self.index_true = index_true
        self.normalization_true = normalization_true.to("cm-2 s-1 TeV-1")
        self.source_pos_radec = source_pos_radec
        self.map_geom = map_geom
        self.e_reco_binning = e_reco_binning
        self.livetime = livetime
        self.offset = offset
        self.irf_file = irf_file

        self.random_state = random_state

    @property
    def true_model(self) -> SkyModel:

        def ecpl_model(amplitude):
            return ExpCutoffPowerLawSpectralModel(
                index=self.index_true,
                amplitude=amplitude / (u.cm**2 * u.s * u.TeV),
                reference=1 * u.TeV,
                lambda_=self.lambda_true,
                alpha=1.0)

        def flux_difference_at_reference(x):
            diff_norm = self.normalization_true.value
            flux = ecpl_model(x)(energy=1 * u.TeV)
            return flux.value - diff_norm

        amplitude = fsolve(flux_difference_at_reference,
                           self.normalization_true.value)[0]

        spectral_model_sim = ExpCutoffPowerLawSpectralModel(
            index=self.index_true,
            amplitude=amplitude / (u.cm**2 * u.s * u.TeV),
            lambda_=self.lambda_true,
            reference=1 * u.TeV)

        ra = self.source_pos_radec["ra"]
        dec = self.source_pos_radec["dec"]

        spatial_model_sim = PointSpatialModel(lon_0=ra, lat_0=dec)

        return SkyModel(spatial_model=spatial_model_sim,
                        spectral_model=spectral_model_sim, name="source")

    @property
    def fit_start_model(self) -> Models:
        ra = self.source_pos_radec["ra"]
        dec = self.source_pos_radec["dec"]

        spatial_model_fit = PointSpatialModel(lon_0=ra, lat_0=dec)

        spatial_model_fit.lon_0.frozen = False
        spatial_model_fit.lat_0.frozen = False

        spatial_model_fit.lon_0.min = spatial_model_fit.lon_0.value - 0.1
        spatial_model_fit.lon_0.max = spatial_model_fit.lon_0.value + 0.1
        spatial_model_fit.lat_0.min = spatial_model_fit.lat_0.value - 0.1
        spatial_model_fit.lat_0.max = spatial_model_fit.lat_0.value + 0.1

        spectral_model_ecpl_fit = ExpCutoffPowerLawSpectralModel(
                                    index=2.0,
                                    amplitude="1.3e-12 cm-2 s-1 TeV-1",
                                    reference="1 TeV",
                                    lambda_="0.1 TeV-1",
                                    alpha=1.0)

        spectral_model_ecpl_fit.amplitude.min = 1e-17
        spectral_model_ecpl_fit.amplitude.max = 1e-5
        spectral_model_ecpl_fit.index.min = 0.0
        spectral_model_ecpl_fit.index.max = 4.0
        spectral_model_ecpl_fit.lambda_.min = -1.0
        spectral_model_ecpl_fit.lambda_.max = 1.0
        spectral_model_ecpl_fit.reference.frozen = True
        spectral_model_ecpl_fit.alpha.frozen = True
        spectral_model_ecpl_fit.lambda_.frozen = False

        return Models(SkyModel(
                        spatial_model=spatial_model_fit,
                        spectral_model=spectral_model_ecpl_fit,
                        name="fitstartmodel"))

    def data(self) -> MapDataset:
        def empty_dataset(source_pos_radec, map_geom, e_reco_binning,
                          livetime, irf_file, offset):

            source_pos_ra = source_pos_radec["ra"]
            source_pos_dec = source_pos_radec["dec"]

            source = SkyCoord(source_pos_ra, source_pos_dec,
                              unit="deg", frame="icrs")

            e_reco_min = u.Quantity(e_reco_binning["e_reco_min"]).to("TeV")
            e_reco_min = e_reco_min.value
            e_reco_max = u.Quantity(e_reco_binning["e_reco_max"]).to("TeV")
            e_reco_max = e_reco_max.value
            n_e_reco = e_reco_binning["n_e_reco"]

            energy_axis = MapAxis.from_edges(
                np.logspace(np.log10(e_reco_min),
                            np.log10(e_reco_max),
                            n_e_reco),
                unit="TeV", name="energy", interp="log")

            geom = WcsGeom.create(
                skydir=source,
                binsz=u.Quantity(map_geom["binsize"]).to("deg").value,
                width=(u.Quantity(map_geom["width"]).to("deg").value,
                       u.Quantity(map_geom["width"]).to("deg").value),
                frame="icrs",
                axes=[energy_axis])

            energy_axis_true = MapAxis.from_edges(
                np.logspace(np.log10(e_reco_min),
                            np.log10(e_reco_max),
                            n_e_reco),
                unit="TeV", name="energy", interp="log")

            pointing = SkyCoord(u.Quantity(source_pos_ra).to("deg"),
                                u.Quantity(source_pos_dec).to("deg") + offset,
                                frame="icrs",
                                unit="deg")

            irfs = load_cta_irfs(irf_file)

            obs = Observation.create(pointing=pointing, livetime=livetime,
                                     irfs=irfs)

            empty = MapDataset.create(geom, energy_axis_true=energy_axis_true)
            maker = MapDatasetMaker(selection=["exposure",
                                               "background",
                                               "psf",
                                               "edisp"])
            maker_safe_mask = SafeMaskMaker(
                methods=["offset-max"],
                offset_max=u.quantity.Quantity(
                        map_geom["width"]) + 1.0 * u.deg)

            dataset = maker.run(empty, obs)
            dataset = maker_safe_mask.run(dataset, obs)

            return dataset

        _empty_dataset = empty_dataset(self.source_pos_radec,
                                       self.map_geom,
                                       self.e_reco_binning,
                                       self.livetime,
                                       self.irf_file,
                                       self.offset)

        dataset = MapDataset(
            models=self.true_model,
            exposure=_empty_dataset.exposure,
            background_model=_empty_dataset.background_model,
            psf=_empty_dataset.psf,
            edisp=_empty_dataset.edisp)

        dataset.background_model.norm.frozen = False
        dataset.background_model.tilt.frozen = True
        dataset.fake(random_state=self.random_state)

        return dataset


if __name__ == "__main__":
    """Routine used to create CutoffBencharkDataset3d datasets.
       Output datasets are to be analyzed with runecpli.py.
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

    parser.add_argument("--irf_file",
                        type=str,
                        help="CTA IRF file",
                        dest="IRF",
                        required=True)

    parser.add_argument("--config",
                        type=str,
                        help="CutoffBenchmarkDataset3D configuration file",
                        dest="CONFIG",
                        required=True)

    options = parser.parse_args()

    with open(options.CONFIG, "r") as fin:
        config = json.load(fin)

    normalization_true = config["normalization_mcrab"] * mCrab
    index_true = config["index"]

    ecut_binning = config["ecut_binning"]
    ecut_true_max = u.Quantity(ecut_binning["ecut_true_max"]).to("TeV").value
    ecut_true_min = u.Quantity(ecut_binning["ecut_true_min"]).to("TeV").value
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

    def get_random_seed():
        """Seed must be between 0 and 2**32 - 1.
        """
        _time = int(time.time() * 1000.0)
        seed = ((_time & 0xff000000) >> 24) + ((_time & 0x00ff0000) >> 8)
        seed += ((_time & 0x0000ff00) << 8) + ((_time & 0x000000ff) << 24)

        return seed

    for ecut_true in ecut_true_list:
        random_state = get_random_seed()
        time.sleep(1)
        lambda_true = 1 / ecut_true
        livetime = u.quantity.Quantity(config["livetime"])
        dataset_parameter = {"name": "CutoffBenchmarkDataset3D",
                             "true_parameter":
                             {"lambda_true": lambda_true,
                              "index_true": config["index"],
                              "normalization_true": normalization_true,
                              "livetime": livetime,
                              "source_pos_radec": config["source_pos_radec"],
                              "e_reco_binning": config["e_reco_binning"],
                              "map_geom": config["map_geom"],
                              "offset": u.quantity.Quantity(config["offset"]),
                              "random_state": random_state}}

        for _ in range(n_benchmarks):
            dataset = CutoffBenchmarkDataset3D(
                                    irf_file=options.IRF,
                                    **dataset_parameter["true_parameter"])
            print(dataset.__dict__)
            result = {"dataset": dataset}
            result["dataset_parameter"] = dataset_parameter

            random_string = "".join(random.choice(letters) for _ in range(10))
            dataset.fit_start_model
            _infile = outdir + "benchmark_" + random_string + ".pickle"
            with open(_infile, "wb") as fout:
                pickle.dump(result, fout)
