# HyperJump #


HyperJump: Accelerating HyperBand via Risk Modelling


This README would normally document whatever steps are necessary to get your application up and running.

### Install Requirements 

* install the following libraries 
```sudo apt-get install libeigen3-dev swig gfortran```
* install the dependencies 
```for req in $(cat requirements.txt); do pip3 install $req; done```
* install HyperJump package
```python3 setup.py develop --user```






### What is this repository for? ###

* Will support the development of the modified BOHB variant that my thesis explains
* Version 1.0
* [Original work](https://www.automl.org/automl/bohb/)
* [Quickstart guide](https://automl.github.io/HpBandSter/build/html/index.html)
### How do I get set up? ###

* You can follow the installation process of original work for hpbandster (I've posted it bellow for convenience) and then add the new files (also listed bellow) manualy

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact

# Original work:

## HpBandSter [![Build Status](https://travis-ci.org/automl/HpBandSter.svg?branch=master)](https://travis-ci.org/automl/HpBandSter)  [![codecov](https://codecov.io/gh/automl/HpBandSter/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/HpBandSter)
a distributed Hyperband implementation on Steroids

This python 3 package is a framework for distributed hyperparameter optimization.
It started out as a simple implementation of [Hyperband (Li et al. 2017)](http://jmlr.org/papers/v18/16-558.html), and contains
an implementation of [BOHB (Falkner et al. 2018)](http://proceedings.mlr.press/v80/falkner18a.html)

## How to install

Original authors keep the package on PyPI up to date. So you should be able to install it via:
```
pip install hpbandster
```
If you want to develop on the code you could install it via:


Then just add the files I have provided manualy:
**LIST FILES**

## Documentation

The documentation is hosted on github pages: [https://automl.github.io/HpBandSter/](https://automl.github.io/HpBandSter/)
It contains a quickstart guide with worked out examples to get you started in different circumstances.

There is also a written [blogpost](https://www.automl.org/blog_bohb/) showcasing the results from the original ICML paper.
