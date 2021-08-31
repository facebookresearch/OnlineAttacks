# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="online_attacks",
    version="0.1",
    author="Hugo Berard, Joey Bose, Andjela Mladenovic",
    author_email="berard.hugo@gmail.com",
    description="Online Adversarial Attacks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AndjelaMladenovic/online_adversarial_attacks",
    project_urls={
        "Bug Tracker": "https://github.com/AndjelaMladenovic/online_adversarial_attacks/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        'torch>=1.0.0',
        'advertorch @ git+https://github.com/hugobb/advertorch#egg=advertorch',
        'tqdm',
        'omegaconf',
        'torchvision',
      ],
)