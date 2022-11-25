# HyperJump #


HyperJump: Accelerating HyperBand via Risk Modelling
[paper](https://arxiv.org/pdf/2108.02479.pdf)

The implementation present in this repo is based on [BOHB](https://github.com/automl/HpBandSter).

### Install Requirements
* Install python3.6 and pip3
* Clone this repository and enter in this directory
* Install the following libraries
```sudo apt-get install libeigen3-dev swig gfortran```
* Install the dependencies
```for req in $(cat requirements.txt); do pip3 install $req; done```
* Install HyperJump package
```python3 setup.py develop --user```
* Compile the script to compute the risk EAR
```cd test && gcc -shared -fPIC -o func.so func.c```


You should verified the python version (python3.6), and the version of packages (e.g. numpy=1.16.4, tensorflow=1.14.0, keras=2.2.5, configSpace=0.4.14) to avoid warnings. If you face some errors installing the dependencies, you should uninstall and re-install and verified the dependencies of the library being installed.


### Benchmarks
* Download the benchmarks used for a Neural Architectural Search scenario with Cifar10, Cifar100, and ImageNet from [Nats-bench](https://drive.google.com/drive/folders/1zjB6wMANiKwB2A1yil2hQ8H_qyeSe2yt) (pzb2 files) and then copy them to a folder called .torch  (see [Nats-bench-repo](https://github.com/D-X-Y/NATS-Bench) for more details or if you have problems).
* You can find some of the rest of benchmarks in this [link](https://drive.google.com/drive/folders/18FwyVbZHJSALkwaUceB6iXz4BmX5FmqZ?usp=sharing). These files should be copy to [test/files](https://github.com/pedrogbmendes/HyperJump/tree/main/test/files) directory. 



### Run HyperJump
You can deploy HyperJump using the script run.py that you can find in test folder. You can directly run ```python3 run.py``` (do not forget to set the arguments in this script) or run ```python3 fake_workload.py``` using the correct arguments (that can be found in this script).
The seeds used in each independent run are set in this script (run.py), and we generate one different seed for each run and give it as input to the function fake_workload.py that is launched in this script.


### Results/logs HyperJump
By default the results/logs of HyperJump are saved in logs folder that you can found in the same folder from where you launch HyperJump ([test/logs](https://github.com/pedrogbmendes/HyperJump/tree/main/test/logs) directory).


### Possible problem while running HyperJump
If you receive some warnings, please verified the version of python and also the versions of the python libraries used.

If while you are running HyperJump, you receive an error like "OSError: [Errno 24] Too many open files", you must check system-wide limits.
RUN ```ulimit -n```
If the result is small (e.g. 1024), you must perform the following steps:

* RUN: ```sudo nano /etc/security/limits.conf```
* Add to this file (usermane is the respective username of the machine (run ```whoami``` to find your username)):

```
usermane hard nofile 10000
username soft nofile 10000
```


* Reboot your machine ```sudo reboot```
