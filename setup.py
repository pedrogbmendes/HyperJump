from setuptools import setup, find_packages

import sys, os, setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = [
    'george',
    'emcee',
    'pybnn',
    'cython',
    'scipy',
    'numpy==1.19.5',
    'scikit-learn==0.24.2',
    'ConfigSpace==0.4.19',
    'nose',
    'pyyaml',
    'jinja2',
    'Pyro4',
    'pybind11',
    'direct',
    'cma',
    'theano',
    'matplotlib',
    'lasagne',
    'mpmath',
    'statsmodels',
    'pandas==1.1.5',
    'Pillow',
    'netifaces',
    'termcolor',
    'statsmodels',
    'direct',
    'robo']


setuptools.setup(name='hyperjump',
                version='1',
                author='Pedro Mendes, Maria Casimiro, Paolo Romano, David Garlan',
                author_email='pgmendes@andrew.cmu.edu',
                url='https://arxiv.org/pdf/2108.02479.pdf',
                description='HyperJump: Accelerating HyperBand via Risk Modelling',
                long_description=long_description,
                keywords='Machine Learining, Bayesian Optimization, HyperBand, Hyper-parameter tuning',
                packages=setuptools.find_packages(),
                license='LICENSE.txt',
                install_requires=requires,
                python_requires='==3.6')
