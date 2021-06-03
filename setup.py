from setuptools import setup, find_packages

import sys, os, setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = [
    'george',
    'emcee',
    'pyrfr',
    'pybnn',
    'cython==0.29.23',
    'scipy >= 0.13',
    'numpy == 1.16.4',
    'sklearn',
    'torch==1.6.0',
    'torchvision==0.7.0',
    'ConfigSpace==0.4.14',
    'nose',
    'pyyaml',
    'jinja2',
    'pybind11',
    'pybnn',
    'george',
    'direct',
    'cma',
    'theano',
    'matplotlib',
    'lasagne',
    #'sgmcmc',
    #'hpolib2',
    'mpmath',
    'pandas',
    'Pillow',
    'tensorflow==1.14.0',
    'keras==2.2.5',
    'Keras-Preprocessing==1.1.2',
    'termcolor',
    'robo']


setuptools.setup(name='hyperjump',
                version='1',
                author='-',
                author_email='-',
                url='-',
                description='HyperJump: Accelerating HyperBand via Risk Modelling',
                long_description=long_description,
                keywords='Machine Learining, Bayesian Optimization, HyperBand, Hyper-parameter tuning',
                packages=setuptools.find_packages(),
                license='LICENSE.txt',
                install_requires=requires,
                python_requires='==3.6')
