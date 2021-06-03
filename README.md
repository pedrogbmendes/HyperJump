# HyperJump #


HyperJump: Accelerating HyperBand via Risk Modelling


### Install Requirements 

* install the following libraries 
```sudo apt-get install libeigen3-dev swig gfortran```
* install the dependencies 
```for req in $(cat requirements.txt); do pip3 install $req; done```
* install HyperJump package
```python3 setup.py develop --user```


You should verified the python version (python3.6), and the version of packages (e.g. numpy=1.16.4, tensorflow=1.14.0, keras=2.2.5, configSpace==0.4.14) to avoid warnings. If you face some errors instllaing the dependencies, you should uninstall and re-install and verified the dependencies of the library being installed.


### Benchmarks
You can find some of the benchmarks in this [repository](https://github.com/pedrogbmendes/HyperJump_/tree/main/test/files) or in this  [link](https://drive.google.com/drive/folders/1LaQJrMygNqTYdFZERuwD08Um8t-3vp6s?usp=sharing)



### Run HyperJump
You can deploy HyperJump using the script in [test](https://github.com/pedrogbmendes/HyperJump_/tree/main/test). You can directly run ```python3 run.py``` (do not forget to set the arguments in this script) or run ```python3 fake_workload.py``` using the correct argument (that can be found in this script). 

### Results/logs HyperJump
By default the results/logs of HyperJump are saved in [logs](https://github.com/pedrogbmendes/HyperJump_/tree/main/test/logs). 
