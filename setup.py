# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Installation script for the 'bam_actuators' python package.

The package is both a regular pip-installable Python package and an Isaac Sim
extension, so the metadata lives in ``config/extension.toml`` and is read here
to keep the two in sync.
"""

import os

try:  # Python >= 3.11, stdlib
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_PATH = os.path.join(EXTENSION_PATH, "config", "extension.toml")
if tomllib is not None:
    with open(EXTENSION_TOML_PATH, "rb") as f:
        EXTENSION_TOML_DATA = tomllib.load(f)
else:
    import toml

    EXTENSION_TOML_DATA = toml.load(EXTENSION_TOML_PATH)

PACKAGE_DATA = EXTENSION_TOML_DATA["package"]

# Installation operation
setup(
    name="isaac_bam_actuators",
    packages=find_packages(include=["bam_actuators", "bam_actuators.*"]),
    # The identified models are plain JSON inside the package, so they have to be
    # declared here or a non-editable install would drop them and
    # params_file="sts3215/m5" would fail.
    package_data={"bam_actuators": ["params/*/*.json"]},
    author="Haris Husnain",
    maintainer="Haris Husnain",
    url=PACKAGE_DATA["repository"],
    version=PACKAGE_DATA["version"],
    description=PACKAGE_DATA["description"],
    keywords=PACKAGE_DATA["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)