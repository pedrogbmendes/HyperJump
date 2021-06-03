
import pandas as pd
import numpy as np
import math, csv, sys
from random import seed
from random import randint

from scipy.sparse import data
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hyperjump.core.worker import Worker

import logging

logging.basicConfig(level=logging.WARNING)

csv.field_size_limit(sys.maxsize)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# ['t2.small', 't2.medium', 't2.xlarge', 't2.2xlarge']
def get_num_workers(vm_flavor, num_cores):
    num_workers = float(num_cores)
    if str(vm_flavor) == 't2.medium':
        num_workers = float(num_cores) / 2
    elif str(vm_flavor) == 't2.xlarge':
        num_workers = float(num_cores) / 4
    elif str(vm_flavor) == 't2.2xlarge':
        num_workers = float(num_cores) / 8
    return int(num_workers)


# Price per hour	|	Vm flavour
# 0,023			|	t2.small
# 0,0464		|	t2.medium
# 0,1856		|	t2.xlarge
# 0,3712		|	t2.2xlarge

# Mudar para (price_per_hour(VM_type)/3600)+(price_per_hour(t2.2xlarge)/3600)*training_time
def get_cost(vm_flavor, num_workers, training_time):
    nr_ps = 1
    if vm_flavor == "t2.small":
        cost = (((num_workers+nr_ps) * 0.023/60.0) + (0.3712/60.0)) * (training_time/60.0)
    elif vm_flavor == "t2.medium":
        cost = (((num_workers+nr_ps) * 0.0464/60.0) + (0.3712/60.0)) * (training_time/60.0)
    elif vm_flavor == "t2.xlarge":
        cost = (((num_workers+nr_ps) * 0.1856/60.0) + (0.3712/60.0)) * (training_time/60.0)
    elif vm_flavor == "t2.2xlarge":
        cost = (((num_workers+nr_ps) * 0.3712/60.0) + (0.3712/60.0)) * (training_time/60.0)
    else:
        print("Tensorflow configuration - Unknown flavor" + vm_flavor)
        sys.exit(0)
    return cost

    #if vm_flavor == 't2.small':
    #    return (0.023 / 3600) * num_workers * training_time
    #elif vm_flavor == 't2.medium':
    #    return (0.0464 / 3600) * num_workers * training_time
    #elif vm_flavor == 't2.xlarge':
    #    return (0.1856 / 3600) * num_workers * training_time
    #else:
    #    return (0.3712 / 3600) * num_workers * training_time


class FakeWorker(Worker):
    def __init__(self, budget_type='accuracy', seed=0, dataset='all', **kwargs):
        print("DATASET IS NAS TIME" + dataset)
        self.budget_type = budget_type
        self.rng = np.random.RandomState(np.int64(seed))
        self.total_cost = 0
        self.seed = seed

        if "all_time" == dataset:
            print("Using nas dataset of time and eta = 2")
            self.df_table = pd.read_csv('files/all_time.csv')
            super().__init__(**kwargs)

            self.dictConfigs = []
            self.listConfigs = self.load_dataset('files/all_time.csv')

        else:
            print("Using nas dataset of time and eta = 3")
            self.df_table = pd.read_csv('files/all_time3.csv')
            super().__init__(**kwargs)

            self.dictConfigs = []
            self.listConfigs = self.load_dataset('files/all_time3.csv')


    #Read the dataset
    def load_dataset(self, CSV):
        print("Reading the dataset....")

        listConfig = []
        with open(CSV) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            headers = next(csv_reader, None)

            for row in csv_reader:
                network = row[0]
                nr_workers = int(row[1])
                learning_rate = float(row[3])
                batch_size = int(row[4])
                synchronism = 0 if row[6] == "async" else  1
                time = float(row[7])    # training time in seconds
                nr_ps = int(row[10])
                accuracy = float(row[11])
                flavor = row[13]
                size = int(row[7])

                cost = 0

                if flavor == "t2.small":
                    vm_flavor = 0
                    cost = (((nr_workers+nr_ps) * 0.023/60.0) + (0.3712/60.0)) * (time/60.0)
                    num_cores = nr_workers
                elif flavor == "t2.medium":
                    vm_flavor = 1
                    cost = (((nr_workers+nr_ps) * 0.0464/60.0) + (0.3712/60.0)) * (time/60.0)
                    num_cores = int(2*nr_workers)
                elif flavor == "t2.xlarge":
                    vm_flavor = 2
                    cost = (((nr_workers+nr_ps) * 0.1856/60.0) + (0.3712/60.0)) * (time/60.0)
                    num_cores = int(4*nr_workers)
                elif flavor == "t2.2xlarge":
                    vm_flavor = 3
                    cost = (((nr_workers+nr_ps) * 0.3712/60.0) + (0.3712/60.0)) * (time/60.0)
                    num_cores = int(8*nr_workers)
                else:
                    print("Tensorflow configuration - Unknown flavor" + flavor)
                    sys.exit(0)

                listConfig.append([network, nr_ps, nr_workers, learning_rate, batch_size, synchronism, vm_flavor, size])
                conf_dict = dict([
                    ('vm_flavor', flavor),
                    ('batch_size', int(batch_size)),
                    ('learning_rate', learning_rate),
                    ('num_cores', int(num_cores)),
                    ('synchronism', row[6]),
                    ('network', network),
                    ('budget', size)])
                self.dictConfigs.append([conf_dict, accuracy, cost, time])

            csv_file.close()

            print("Dataset read all")
            return listConfig

    def listConfigSpace(self):
        return self.listConfigs

    def dictConfigSpace(self):
        #print(self.dictConfigs)
        return self.dictConfigs


    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        """
        Get values from dataset
        example 
        df.loc[(df['learning_rate'] == 0.00001) & (df['batch_size'] == 256) & (df['vm_flavor'] == 't2.small') & (df['training_set_size'] == 1000)]
        
        """

        # seed(1)



        num_workers = get_num_workers(str(config['vm_flavor']), float(config['num_cores']))
        '''
        print(config)
        print('batch_size')
        print(config['batch_size'])
        print(type(config['batch_size']))
        print('learning_rate')
        print(config['learning_rate'])
        print(type(config['learning_rate']))
        print('vm_flavor')
        print(config['vm_flavor'])
        print(type(config['vm_flavor']))
        print('synchronism')
        print(config['synchronism'])
        print(type(config['synchronism']))
        print('num_cores')
        print(config['num_cores'])
        print(type(config['num_cores']))
        '''

        df = self.df_table
        sub_df = df.loc[
            (df['learning_rate'] == float(config['learning_rate'])) & (df['n_workers'] == int(num_workers)) &
            (df['network'] == str(config['network'])) & 
            (df['batch_size'] == int(config['batch_size'])) & (df['vm_flavor'] == str(config['vm_flavor'])) & 
            (df['synchronism'] == str(config['synchronism'])) &
            (df['training_time'] == int(budget))]

        '''
        num_workers = get_num_workers(str("t2.2xlarge"), float(1))
        sub_df = df.loc[(df['learning_rate'] == float("0.001")) &  (df['n_workers'] == int(num_workers)) &
            (df['batch_size'] == int(16)) & (df['vm_flavor'] == str("t2.2xlarge")) & (df['synchronism'] == str("sync")) &
            (df['training_set_size'] == int(1000))]
        '''
        # check if table is empty!

        if sub_df.empty:
            # print(config)
            # print(real_budget)
            return {
                'loss': 1,  # remember: hyperjump always minimizes!
                'info': {
                    'accuracy': 0,
                    'cost': 0,
                    'total cost': self.total_cost,
                    'accuracy loss': 1,
                    'error': "Invalid combination!"
                }
            }

        size = sub_df.shape[0]
        chosen = self.rng.randint(0, size)
        acc = sub_df.iloc[chosen]['acc']
        training_time = sub_df.iloc[chosen]['training_time']
        cost = get_cost(sub_df.iloc[chosen]['vm_flavor'], int(sub_df.iloc[chosen]['n_workers']), float(training_time))
        self.total_cost = self.total_cost + cost
        loss = 1 - float(acc)
        # print('\n\n\nBUDGET TYPE -> ' + self.budget_type )
        if self.budget_type == 'cost_accuracy':
            print("budget_type == 'cost_accuracy'")
            # sigmoid(accuracy*100/cost)
            # sigmoid is used to normalize the value
            # weighted losses? when the cost gets so litle, the accuracy almost doesn't matter..
            loss = (1 - float(acc)) * 0.7 + (1 - sigmoid((acc) / cost)) * 0.3
        # normalizar o custo !!

        # import IPython; IPython.embed()
        return ({
            'loss': loss,  # remember: hyperjump always minimizes!
            'info': {'accuracy': acc,
                     'cost': cost,
                     'total cost': self.total_cost,
                     'accuracy loss': 1 - float(acc),
                     'budget': int(budget),
                     'training_time': float(training_time),
                     'error': "None"
                     }

        })

    @staticmethod
    def get_configspace(seed):
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        '''
        
        Params
            Learning rate: 			0.00001; 0.0001; 0.001
            Batch size: 			256, 16
            synchronization: 		async; sync
            Vm Flavour:				t2.small; t2.medium; t2.xlarge; t2.2xlarge
            Budgets (Dataset size):	3750, 7500, 15000, 30000, 60000
        
        Config space
            3(lr)*2(bs)*2(sync)*6(cores)*4(vm_flavour) = 288
        '''
        cs = CS.ConfigurationSpace(seed=seed)
        batch_size = CSH.CategoricalHyperparameter('batch_size', [16, 256])
        learning_rate = CSH.CategoricalHyperparameter('learning_rate', [0.00001, 0.0001, 0.001])
        network = CSH.CategoricalHyperparameter('network', ['cnn', 'mlp', 'rnn'])
        num_cores = CSH.CategoricalHyperparameter('num_cores', [8, 16, 32, 48, 64, 80])
        synchronism = CSH.CategoricalHyperparameter('synchronism', ['async', 'sync'])
        vm_flavor = CSH.CategoricalHyperparameter('vm_flavor', ['t2.small', 't2.medium', 't2.xlarge', 't2.2xlarge'])

        cs.add_hyperparameters([batch_size, learning_rate, network, num_cores, synchronism, vm_flavor])

        return cs


if __name__ == "__main__":
    worker = FakeWorker(budget_type='accuracy', seed=0, dataset='rnn', run_id='0')
    cs = FakeWorker.get_configspace(12)

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=3750, working_directory='.')
    print(res)
