# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='waveutils',
    version='0.1.1',
    description='Sample package for Python-Guide.org',
    long_description=readme,
    author='Eoin',
    author_email='eoin@eoin.com',
    url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

