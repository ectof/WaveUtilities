# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='waveutils',
    version='0.1.2.0',
    description='Some utilities for data analysis',
    long_description=readme,
    author='Eoin',
    author_email='eoin@eoin.com',
    url='https://github.com/ectof/WaveUtilities',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

