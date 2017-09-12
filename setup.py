# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='waveutils',
    version='0.1.3.1',
    description='Some utilities for data analysis and file IO',
    long_description=readme,
    author='Eoin O\'Farrell',
    author_email='eoin.ofarrell@nbi.ku.dk',
    url='https://github.com/ectof/WaveUtilities',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

