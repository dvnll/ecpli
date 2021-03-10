"""Performs the analysis of a CufoffBenchmarkDataset3D.
"""

import argparse
import pickle
import ecpli
from ecpli.benchmark3d.CutoffBenchmarkDataset3D import CutoffBenchmarkDataset3D
import json
from ecpli.ECPLiBase import LimitTarget
import logging
import copy

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()

parser.add_argument("--jobid",
                    type=str,
                    help="Job ID",
                    dest="JOBID",
                    required=False,
                    default="StandardJob")

parser.add_argument("--config",
                    type=str,
                    help="Config file",
                    dest="CONFIG",
                    required=True)

parser.add_argument("--infile",
                    type=str,
                    help="Input file",
                    dest="INFILE",
                    required=True)

parser.add_argument("--outfile",
                    type=str,
                    help="Output file",
                    dest="OUTFILE",
                    default=None)

parser.add_argument("--debug",
                    help="Enable debugging messages",
                    dest="DEBUG", action="store_true")

options = parser.parse_args()

if options.DEBUG:
    logging.getLogger().setLevel(level=logging.DEBUG)


with open(options.CONFIG, "r") as fin:
    config = json.load(fin)

method_name_list = list(config["method_dict"].keys())

infile = options.INFILE
outfile = options.OUTFILE
jobid = options.JOBID

if not infile.endswith(".pickle"):
    raise RuntimeError("Expect pickled infile")

if outfile is None:
    outfile = infile.replace(".pickle",
                             "_limit_" + "_".join(method_name_list) + ".pickle")

with open(infile, "rb") as fin:
    dataset = pickle.load(fin)

for method_name in method_name_list:
    optargs = config["method_dict"][method_name]

    limit_target = LimitTarget(model=dataset["dataset"].fit_start_model[0],
                               parameter_name="lambda_",
                               parmin=0.,
                               parmax=1./0.05)
    data_copy = copy.copy(dataset["dataset"])
    data = data_copy.data()
    arg_dict = {"limit_target": limit_target,
                "data": data,
                "models": dataset["dataset"].fit_start_model,
                "CL": config["CL"]}

    for arg_key in optargs.keys():
        arg_dict[arg_key] = optargs[arg_key]

    method = getattr(ecpli, method_name)(**arg_dict)

    ul = method.ul

    if "limit_list" not in dataset.keys():
        dataset["limit_list"] = []

    dataset["limit_list"].append((method_name, ul, config["CL"]))

if outfile is None:
    outfile = infile.replace(".pickle", "lr.pickle")

with open(outfile, "wb") as fout:
    pickle.dump(dataset, fout)

print("Finished")
