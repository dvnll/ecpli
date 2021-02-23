from setuptools import setup
import os
import sys


if sys.version_info < (3, 5):
    sys.exit("Python < 3.5 is not supported.")


def get_version(version_tuple):
    return ".".join(map(str, version_tuple))


init = os.path.join(
        os.path.dirname(__file__), ".", "", "__init__.py")

version_line = list(
        filter(lambda l: l.startswith("VERSION"), open(init))
)[0]

PKG_VERSION = get_version(eval(version_line.split("=")[-1]))

description = "Work in progress. Do not use!"

setup(name="ecpli",
      version=PKG_VERSION,
      description=description,
      license="MIT",
      install_requires=["numpy", "gammapy", "seaborn"])

import gammapy
if not gammapy.__version__ == "0.16":
    raise ImportError("Currently only supporting gammapy v-0.16!")
