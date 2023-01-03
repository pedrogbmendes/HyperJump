import os
import time
import numpy as np
#import signal
#from sklearn import svm
import multiprocessing
import subprocess
import pandas as pd
import copy

import numpy as np
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy

from hyperjump.core.worker import Worker

directory = ".."


def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


class SvmWorker(Worker):
    def __init__(self, pauseResume=True, checkpointing=True, dataset="covertype", **kwargs):
        super().__init__(**kwargs)
        self.no_cpus = int(multiprocessing.cpu_count() - 1)
        self.dataset = dataset
        
        if self.dataset == "cifar10":
            self.dataset_name = "svm_dataset.csv"
        else:
            self.dataset_name = "svm_covertype.csv"
        

        subprocess.run("export OMP_NUM_THREADS=" + str(self.no_cpus), shell=True)

        data_test = None
        data_train = None
        self.printFlag = False


        #verify the dataset
        if self.dataset == "cifar10":
            if not os.path.isfile(directory + "/libsvm-master/datasets/cifar10"):
                subprocess.run("wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/cifar10.bz2", shell=True)
                subprocess.run("bzip2 -d cifar10.bz2", shell=True)
                subprocess.run("cp  cifar10 " + directory + "/libsvm-master/datasets/ ", shell=True)
                print("Downloading cifar10 training")


            if not os.path.isfile(directory + "/libsvm-master/datasets/cifar10.t"):
                subprocess.run("wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/cifar10.t.bz2", shell=True)
                subprocess.run("bzip2 -d cifar10.t.bz2", shell=True)
                subprocess.run("cp  cifar10.t " + directory + "/libsvm-master/datasets ", shell=True)
                print("Downloading cifar10 testing")



            if not os.path.isfile(directory + "/libsvm-master/datasets/cifar10_size617"):
                #subprocess.run("head -n617 ~/libsvm-master/cifar10 >> ~/libsvm-master/cifar10_size617", shell=True)
                dataset_size = 617
                if data_train is None:
                    f = open(directory + "/libsvm-master/datasets/cifar10", "r")
                
                    #f = open("cifar10", "r")
                    Lines = f.readlines()
                    l_ = []
                    for line in Lines:
                        aux = line.split(" ")[:-1]
                        l = []
                        for i in aux:
                            l.append(i)
                        l_.append(copy.deepcopy(l))

                    data_train = pd.DataFrame(l_)

                data = stratified_sample_df(data_train, 0, int(dataset_size/10.0))
                size = len(data)
                if size < dataset_size:
                    samples = data_train.sample(n = int(dataset_size - size))
                    ct = 10000
                    for i in range(len(samples)):
                        l = []
                        sample = samples.iloc[i] 
                        for j in range(len(sample)):
                            l.append(sample[j])
                    
                        data.loc[ct+1] = copy.deepcopy(l)
                        ct +=1

                while size > dataset_size:
                    # Delete the last row in the DataFrame
                    data = data.drop(data.index[-1])
                    size = len(data)

                data.to_csv(directory + "/libsvm-master/datasets/cifar10_size617", sep=' ', index=False, header=False)
                print("created dataset cifar10_size617")


            if not os.path.isfile(directory + "/libsvm-master/datasets/cifar10_size1851"):
                #subprocess.run("head -n617 ~/libsvm-master/cifar10 >> ~/libsvm-master/cifar10_size617", shell=True)
                dataset_size = 1851
                if data_train is None:
                    f = open(directory + "/libsvm-master/datasets/cifar10", "r")
                
                    #f = open("cifar10", "r")
                    Lines = f.readlines()
                    l_ = []
                    for line in Lines:
                        aux = line.split(" ")[:-1]
                        l = []
                        for i in aux:
                            l.append(i)
                        l_.append(copy.deepcopy(l))

                    data_train = pd.DataFrame(l_)

                data = stratified_sample_df(data_train, 0, int(dataset_size/10.0))
                size = len(data)
                if size < dataset_size:
                    samples = data_train.sample(n = int(dataset_size - size))
                    ct = 10000
                    for i in range(len(samples)):
                        l = []
                        sample = samples.iloc[i] 
                        for j in range(len(sample)):
                            l.append(sample[j])
                    
                        data.loc[ct+1] = copy.deepcopy(l)
                        ct +=1

                while size > dataset_size:
                    # Delete the last row in the DataFrame
                    data = data.drop(data.index[-1])
                    size = len(data)

                data.to_csv(directory + "/libsvm-master/datasets/cifar10_size1851", sep=' ', index=False, header=False)
                print("created dataset cifar10_size1851")


            if not os.path.isfile(directory + "/libsvm-master/datasets/cifar10_size5555"):
                #subprocess.run("head -n617 ~/libsvm-master/cifar10 >> ~/libsvm-master/cifar10_size617", shell=True)
                dataset_size = 5555
                if data_train is None:
                    f = open(directory + "/libsvm-master/datasets/cifar10", "r")
                
                    #f = open("cifar10", "r")
                    Lines = f.readlines()
                    l_ = []
                    for line in Lines:
                        aux = line.split(" ")[:-1]
                        l = []
                        for i in aux:
                            l.append(i)
                        l_.append(copy.deepcopy(l))

                    data_train = pd.DataFrame(l_)

                data = stratified_sample_df(data_train, 0, int(dataset_size/10.0))
                size = len(data)
                if size < dataset_size:
                    samples = data_train.sample(n = int(dataset_size - size))
                    ct = 10000
                    for i in range(len(samples)):
                        l = []
                        sample = samples.iloc[i] 
                        for j in range(len(sample)):
                            l.append(sample[j])
                    
                        data.loc[ct+1] = copy.deepcopy(l)
                        ct +=1

                while size > dataset_size:
                    # Delete the last row in the DataFrame
                    data = data.drop(data.index[-1])
                    size = len(data)

                data.to_csv(directory + "/libsvm-master/datasets/cifar10_size5555", sep=' ', index=False, header=False)
                print("created dataset cifar10_size5555")


            if not os.path.isfile(directory + "/libsvm-master/datasets/cifar10_size16666"):
                #subprocess.run("head -n617 ~/libsvm-master/cifar10 >> ~/libsvm-master/cifar10_size617", shell=True)
                dataset_size = 16666
                if data_train is None:
                    f = open(directory + "/libsvm-master/datasets/cifar10", "r")
                
                    #f = open("cifar10", "r")
                    Lines = f.readlines()
                    l_ = []
                    for line in Lines:
                        aux = line.split(" ")[:-1]
                        l = []
                        for i in aux:
                            l.append(i)
                        l_.append(copy.deepcopy(l))

                    data_train = pd.DataFrame(l_)

                data = stratified_sample_df(data_train, 0, int(dataset_size/10.0))
                size = len(data)
                if size < dataset_size:
                    samples = data_train.sample(n = int(dataset_size - size))
                    ct = 10000
                    for i in range(len(samples)):
                        l = []
                        sample = samples.iloc[i] 
                        for j in range(len(sample)):
                            l.append(sample[j])
                    
                        data.loc[ct+1] = copy.deepcopy(l)
                        ct +=1

                while size > dataset_size:
                    # Delete the last row in the DataFrame
                    data = data.drop(data.index[-1])
                    size = len(data)

                data.to_csv(directory + "/libsvm-master/datasets/cifar10_size16666", sep=' ', index=False, header=False)
                print("created dataset cifar10_size16666")


            if not os.path.isfile(directory + "/libsvm-master/datasets/cifar10_size617.t"):
                dataset_size = int(10000.0/(50000.0/617.0))
                if data_test is None:
                    f = open(directory + "/libsvm-master/datasets/cifar10.t", "r")
                
                    #f = open("cifar10.t", "r")
                    Lines = f.readlines()
                    l_ = []
                    for line in Lines:
                        aux = line.split(" ")[:-1]
                        l = []
                        for i in aux:
                            l.append(i)
                        l_.append(copy.deepcopy(l))

                    data_test = pd.DataFrame(l_)


                data = stratified_sample_df(data_test, 0, int(dataset_size/10.0))
                size = len(data)
                if size < dataset_size:
                    samples = data_test.sample(n = int(dataset_size - size))
                    ct = 10000
                    for i in range(len(samples)):
                        l = []
                        sample = samples.iloc[i] 
                        for j in range(len(sample)):
                            l.append(sample[j])
                    
                        data.loc[ct+1] = copy.deepcopy(l)
                        ct +=1

                while size > dataset_size:
                    # Delete the last row in the DataFrame
                    data = data.drop(data.index[-1])
                    size = len(data)

                data.to_csv(directory + "/libsvm-master/datasets/cifar10_size617.t", sep=' ', index=False, header=False)
                print("created dataset cifar10_size617.t")


            if not os.path.isfile(directory + "/libsvm-master/datasets/cifar10_size1851.t"):
                dataset_size = int(10000.0/(50000.0/1851.0))
                if data_test is None:
                    f = open(directory + "/libsvm-master/datasets/cifar10.t", "r")
                
                    #f = open("cifar10.t", "r")
                    Lines = f.readlines()
                    l_ = []
                    for line in Lines:
                        aux = line.split(" ")[:-1]
                        l = []
                        for i in aux:
                            l.append(i)
                        l_.append(copy.deepcopy(l))

                    data_test = pd.DataFrame(l_)


                data = stratified_sample_df(data_test, 0, int(dataset_size/10.0))
                size = len(data)
                if size < dataset_size:
                    samples = data_test.sample(n = int(dataset_size - size))
                    ct = 10000
                    for i in range(len(samples)):
                        l = []
                        sample = samples.iloc[i] 
                        for j in range(len(sample)):
                            l.append(sample[j])
                    
                        data.loc[ct+1] = copy.deepcopy(l)
                        ct +=1

                while size > dataset_size:
                    # Delete the last row in the DataFrame
                    data = data.drop(data.index[-1])
                    size = len(data)

                data.to_csv(directory + "/libsvm-master/datasets/cifar10_size1851.t", sep=' ', index=False, header=False)
                print("created dataset cifar10_size1851.t")


            if not os.path.isfile(directory + "/libsvm-master/datasets/cifar10_size5555.t"):
                dataset_size = int(10000.0/(50000.0/5555.0))
                if data_test is None:
                    f = open(directory + "/libsvm-master/datasets/cifar10.t", "r")
                
                    #f = open("cifar10.t", "r")
                    Lines = f.readlines()
                    l_ = []
                    for line in Lines:
                        aux = line.split(" ")[:-1]
                        l = []
                        for i in aux:
                            l.append(i)
                        l_.append(copy.deepcopy(l))

                    data_test = pd.DataFrame(l_)


                data = stratified_sample_df(data_test, 0, int(dataset_size/10.0))
                size = len(data)
                if size < dataset_size:
                    samples = data_test.sample(n = int(dataset_size - size))
                    ct = 10000
                    for i in range(len(samples)):
                        l = []
                        sample = samples.iloc[i] 
                        for j in range(len(sample)):
                            l.append(sample[j])
                    
                        data.loc[ct+1] = copy.deepcopy(l)
                        ct +=1

                while size > dataset_size:
                    # Delete the last row in the DataFrame
                    data = data.drop(data.index[-1])
                    size = len(data)

                data.to_csv(directory + "/libsvm-master/datasets/cifar10_size5555.t", sep=' ', index=False, header=False)
                print("created dataset cifar10_size5555.t")


            if not os.path.isfile(directory + "/libsvm-master/datasets/cifar10_size16666.t"):
                dataset_size = int(10000.0/(50000.0/16666.0))
                if data_test is None:
                    f = open(directory + "/libsvm-master/datasets/cifar10.t", "r")
                
                    #f = open("cifar10.t", "r")
                    Lines = f.readlines()
                    l_ = []
                    for line in Lines:
                        aux = line.split(" ")[:-1]
                        l = []
                        for i in aux:
                            l.append(i)
                        l_.append(copy.deepcopy(l))

                    data_test = pd.DataFrame(l_)


                data = stratified_sample_df(data_test, 0, int(dataset_size/10.0))
                size = len(data)
                if size < dataset_size:
                    samples = data_test.sample(n = int(dataset_size - size))
                    ct = 10000
                    for i in range(len(samples)):
                        l = []
                        sample = samples.iloc[i] 
                        for j in range(len(sample)):
                            l.append(sample[j])
                    
                        data.loc[ct+1] = copy.deepcopy(l)
                        ct +=1

                while size > dataset_size:
                    # Delete the last row in the DataFrame
                    data = data.drop(data.index[-1])
                    size = len(data)

                data.to_csv(directory + "/libsvm-master/datasets/cifar10_size16666.t", sep=' ', index=False, header=False)
                print("created dataset cifar10_size16666.t")

        else:
            #covertype
            if not os.path.isfile(directory + "/libsvm-master/datasets/covtype"):
                subprocess.run("wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2", shell=True)
                subprocess.run("bzip2 -d covtype.libsvm.binary.scale.bz2", shell=True)
                subprocess.run("mv  covtype.libsvm.binary.scale covtype", shell=True)
                subprocess.run("cp  covtype " + directory + "/libsvm-master/datasets/ ", shell=True)
                print("Downloading covtype training")


            #for dataset_size in :
            for dataset_size in [27777, 9259, 3086, 1028, 342 ]:
            #for dataset_size in [250000, 83333, 27777, 9259, 3086]:

                if not os.path.isfile(directory + "/libsvm-master/datasets/covtype_" + str(dataset_size)):

                    if data_train is None:
                        f = open(directory + "/libsvm-master/datasets/covtype", "r")
                        Lines = f.readlines()
                        l_ = []
                        for line in Lines:
                            aux = line.split(" ")[:-1]
                            l = []
                            for i in aux:
                                l.append(i)
                            l_.append(copy.deepcopy(l))
                        data_train = pd.DataFrame(l_)

                    data = stratified_sample_df(data_train, 0, int(dataset_size/2.0))

                    size = len(data)
                    if size < dataset_size:
                        samples = data_train.sample(n = int(dataset_size - size))
                        ct = 100000
                        for i in range(len(samples)):
                            l = []
                            sample = samples.iloc[i] 
                            for j in range(len(sample)):
                                l.append(sample[j])
                        
                            data.loc[ct+1] = copy.deepcopy(l)
                            ct +=1

                    while size > dataset_size:
                        # Delete the last row in the DataFrame
                        data = data.drop(data.index[-1])
                        size = len(data)

                    data.to_csv(directory + "/libsvm-master/datasets/covtype_" + str(dataset_size), sep=' ', index=False, header=False)
                    print("created dataset covtype " + str(dataset_size))



                if not os.path.isfile(directory + "/libsvm-master/datasets/covtype_" + str(dataset_size) + ".t"):

                    if data_train is None:
                        f = open(directory + "/libsvm-master/datasets/covtype", "r")
                        Lines = f.readlines()
                        l_ = []
                        for line in Lines:
                            aux = line.split(" ")[:-1]
                            l = []
                            for i in aux:
                                l.append(i)
                            l_.append(copy.deepcopy(l))
                        data_train = pd.DataFrame(l_)

                    dataset_size_ = int(dataset_size * 0.1)
                    data = stratified_sample_df(data_train, 0, int(dataset_size_/2.0))

                    size = len(data)
                    if size < dataset_size_:
                        samples = data_train.sample(n = int(dataset_size_ - size))
                        ct = 100000
                        for i in range(len(samples)):
                            l = []
                            sample = samples.iloc[i] 
                            for j in range(len(sample)):
                                l.append(sample[j])
                        
                            data.loc[ct+1] = copy.deepcopy(l)
                            ct +=1

                    while size > dataset_size_:
                        # Delete the last row in the DataFrame
                        data = data.drop(data.index[-1])
                        size = len(data)

                    data.to_csv(directory + "/libsvm-master/datasets/covtype_" + str(dataset_size) + ".t", sep=' ', index=False, header=False)
                    print("created dataset covtype.t" + str(dataset_size))

        print("Datasets are ready!!!")

        self.trainingSet = dict()
        self.pauseResume = pauseResume
        self.checkpointing = checkpointing

        super().__init__(**kwargs)




    def compute(self, config, budget, *args, **kwargs):
        starttime = time.time()
        config_with_budgets = deepcopy(config)
        config_with_budgets["budget"] = budget

        subprocess.run("export OMP_NUM_THREADS=" + str(self.no_cpus), shell=True)

        budget = int(budget)
        kernel_name = config['kernel']
        gamma = config['gamma']
        c = config['C']

        command = directory + "/libsvm-master/svm-train "

        # Train the SVM on the subset set
        if self.dataset == "cifar10":
            if budget == 617:
                name_dataset = directory + "/libsvm-master/datasets/cifar10_size617"
                name_model = "cifar10_size617.model"

            elif budget == 1851:
                name_dataset = directory + "/libsvm-master/datasets/cifar10_size1851"
                name_model = "cifar10_size1851.model"

            elif budget == 5555:
                name_dataset = directory + "/libsvm-master/datasets/cifar10_size5555"
                name_model = "cifar10_size5555.model"

            elif budget == 16666:
                name_dataset = directory + "/libsvm-master/datasets/cifar10_size16666"
                name_model = "cifar10_size16666.model"

            else:
                name_dataset = directory + "/libsvm-master/datasets/cifar10"
                name_model = "cifar10.model"
        else:
            name_dataset = directory + "/libsvm-master/datasets/covtype_" + str(budget)
            name_model = "covtype_" + str(budget) + ".model"


        # -t kernel_type : set type of kernel function (default 2)
        # 	0 -- linear: u'*v
        # 	1 -- polynomial: (gamma*u'*v + coef0)^degree
        # 	2 -- radial basis function: exp(-gamma*|u-v|^2)
        # 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
        # -d degree : set degree in kernel function (default 3)
        # -g gamma : set gamma in kernel function (default 1/num_features)
        # -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)

          
        if  kernel_name == "linear":
            command += " -t 0 -c " + str(c) + " -g " + str(gamma)
        elif kernel_name == "polynomial2": 
            command += " -t 1 -d 2 -c " + str(c) + " -g " + str(gamma)
        elif kernel_name == "polynomial3":
            command += " -t 1 -d 3 -c " + str(c) + " -g " + str(gamma)
        elif kernel_name == "polynomial4":
            command += " -t 1 -d 4 -c " + str(c) + " -g " + str(gamma)
        elif kernel_name == "rbf":
            command += " -t 2 -c " + str(c) + " -g " + str(gamma)
        else: # sigmoid
            command += " -t 3 -c " + str(c) + " -g " + str(gamma)

        command += " -m 1000 -e 0.001 -q " + name_dataset 

        subprocess.call(command, shell=True)
        subprocess.call("mv " + name_model + " " + directory + "/libsvm-master/datasets/",  shell=True)

        self.logger.info("Training done")

        #command = "~/hyperjump/svm-predict " +  name_dataset + ".t " + name_dataset + ".model -q"
        command = directory + "/libsvm-master/svm-predict " +  name_dataset + ".t " + name_dataset + ".model -q"

        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        stdout = process.communicate()[0]
        acc = float((str(stdout).split("= ")[1]).split("%")[0])/100.0 

        #b'Accuracy = 80.8793% (25037/30956) (classification)'
        t_now = time.time() 
        training_time = t_now - starttime

        self.logger.info("Config " + str(config) + "ran in " + str(training_time) + "s and achieved an accuracy of " + str(acc))

        subprocess.call("rm " + directory + "/libsvm-master/datasets/" + name_model,  shell=True)

       
        return ({
            'loss': 1.0-acc,  # remember: hyperjump always minimizes!
            'info': {
                     'config': config_with_budgets,
                     'training_time': training_time,
                     'start_time': starttime,
                     'end_time': t_now,
                     'real_time': t_now-starttime,
                     'accuracy': acc,
                     'checkpoint': None,
                     }
            })


    def get_configspace(seed=100):
        """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
        cs = CS.ConfigurationSpace(seed=int(seed))


        # -t kernel_type : set type of kernel function (default 2)
        # 	0 -- linear: u'*v
        # 	1 -- polynomial: (gamma*u'*v + coef0)^degree
        # 	2 -- radial basis function: exp(-gamma*|u-v|^2)
        # 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
        # -d degree : set degree in kernel function (default 3)
        # -g gamma : set gamma in kernel function (default 1/num_features)
        # -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)

        # Total = 4
        # lr = CSH.UniformIntegerHyperparameter('lr', lower=1, upper=4, default_value=2)

        kernel = CSH.CategoricalHyperparameter('kernel', ["linear", "polynomial2", "polynomial3", "polynomial4", "sigmoid", "rbf"])
        gamma = CSH.CategoricalHyperparameter('gamma', [1e-6, 1e-5, 1e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        C = CSH.CategoricalHyperparameter('C', [1e-6, 1e-5, 1e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


        cs.add_hyperparameters([kernel, gamma, C])

        return cs

