#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

from setuptools import setup
from os import path


# Get the long description from the relevant file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()



setup(
    name='psfsim',
    version= "1.0",
    description='Simulation of high NA PSFs',
    long_description=long_description,
    url='',
    author='Aurélien Barbotin',
    author_email=', '.join([
        'aurelien.barbotin@dtc.ox.ac.uk']),
    license='to be defined',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    setup_requires=['numpy'],
    install_requires=['numpy','matplotlib','zernike','h5py','scipy'],
    package_data = {'psfsim': ['data/*.npy']}
)
