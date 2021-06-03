# HyperJump #


HyperJump: Accelerating HyperBand via Risk Modelling


### Install Requirements
* Install python3.6 and pip3
* Clone this repository and enter in this directory
* install the following libraries
```sudo apt-get install libeigen3-dev swig gfortran```
* install the dependencies
```for req in $(cat requirements.txt); do pip3 install $req; done```
* install HyperJump package
```python3 setup.py develop --user```
* compile the script to compute the risk
```cd test && gcc -shared -fPIC -o func.so func.c```


You should verified the python version (python3.6), and the version of packages (e.g. numpy=1.16.4, tensorflow=1.14.0, keras=2.2.5, configSpace=0.4.14) to avoid warnings. If you face some errors installing the dependencies, you should uninstall and re-install and verified the dependencies of the library being installed.


### Benchmarks
You can find some of the benchmarks in this [repository](https://github.com/pedrogbmendes/HyperJump_/tree/main/test/files) or in this  [link](https://drive.google.com/drive/folders/1LaQJrMygNqTYdFZERuwD08Um8t-3vp6s?usp=sharing)



### Run HyperJump
You can deploy HyperJump using the script in [test](https://github.com/pedrogbmendes/HyperJump_/tree/main/test). You can directly run ```python3 run.py``` (do not forget to set the arguments in this script) or run ```python3 fake_workload.py``` using the correct arguments (that can be found in this script).

### Results/logs HyperJump

By default the results/logs of HyperJump are saved in [logs](https://github.com/pedrogbmendes/HyperJump_/tree/main/test/logs).


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
