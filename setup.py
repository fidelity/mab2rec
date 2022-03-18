# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    required = fh.read().splitlines()

with open(os.path.join('mab2rec', '_version.py')) as fp:
    exec(fp.read())

setuptools.setup(
    name="mab2rec",
    description="Mab2Rec: Multi-Armed Bandits Recommender",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    author=__author__,
    url="",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=required,
    python_requires=">=3.7",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/fidelity/mab2rec"
    }
)
