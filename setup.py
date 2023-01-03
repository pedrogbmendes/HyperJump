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
    'numpy',
    'scikit-learn',
    'ConfigSpace==0.4.14',
    'nose',
    'pyyaml',
    'jinja2',
    'pybind11',
    'direct',
    'cma',
    'theano',
    'matplotlib',
    'lasagne',
    'mpmath',
    'statsmodels',
    'pandas',
    'Pillow',
    'netifaces',
    'termcolor',
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
