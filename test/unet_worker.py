
import pandas as pd
import numpy as np
import math, copy
from random import seed
from random import randint
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import csv

from hyperjump.core.worker import Worker

import logging
import sys
logging.basicConfig(level=logging.WARNING)

INTEL14_V1_COST = 0.9 / 3600
INTEL14_V2_COST = 1.14 / 3600


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_cost(vm_flavor, num_workers, training_time):
    if vm_flavor == "intel14_v2":
        if num_workers == 1:
            price_per_sec = 1.204/3600.0
        else:
            price_per_sec = 2.176/3600.0

    else:
        if num_workers == 1:
            price_per_sec = 1.14/3600.0
        else:
            price_per_sec = 2.28/3600.0

    return training_time * price_per_sec


    #if vm_flavor == 'intel14_v1':
    #    return num_workers * training_time * INTEL14_V1_COST
    #else:
    #    return num_workers * training_time * INTEL14_V2_COST


class UnetWorker(Worker):
    def __init__(self, seed=0, dataset='', **kwargs):
        #print(dataset)
        self.rng = np.random.RandomState(np.int64(seed))
        self.total_cost = 0
        self.seed = seed
        if dataset == "unet":
            print("ETA=2 dataset file unet.csv ")

            self.dictConfigs = []
            self.df_table = pd.read_csv('files/unet.csv', sep=";")
            self.listConfigs = self.load_dataset('files/unet.csv')

        elif dataset == "unet3":
             print("ETA=3 dataset file unet_eta3.csv ")
            
             self.dictConfigs = []
             self.df_table = pd.read_csv('files/unet_eta3.csv', sep=";")
             self.listConfigs = self.load_dataset('files/unet_eta3.csv')
             #print(self.df_table)

        # else:
        #     print("Wrong ETA")
        #     sys.exit()

        #print(self.listConfigs)
        super().__init__(**kwargs)


    #Read the dataset
    def load_dataset(self, CSV):
        print("Reading the dataset....")

        listConfig = []
        with open(CSV) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            headers = next(csv_reader, None)

            # {Flavor, batch, learningRate, momentum, nrWorker, sync, size}
            #lower = np.array([1, 1, 0.000001, 0.9, 1, 1, min_budget])
            #upper = np.array([2, 2, 0.0001, 0.99, 2, 2, max_budget])

            for row in csv_reader:
                flavor = 1 if row[0] == "intel14_v1" else 2
                nr_workers = int(row[1])
                batch_size = int(row[2])
                learning_rate = float(row[3])
                mommentum = float(row[4])
                synchronism = 1 if row[5] == "async" else  2
                size = int(row[6])

                acc = float(row[8])
                cost = get_cost(row[0], nr_workers, size)
                #time = float(row[9])
                #cost = float(row[8])

                listConfig.append([flavor, batch_size, learning_rate, mommentum, nr_workers, synchronism, size])
                conf_dict = dict([
                    ('vm_flavor', row[0]),
                    ('batch_size', int(batch_size)),
                    ('learning_rate', learning_rate),
                    ('momentum', float(mommentum)),
                    ('nrWorker', int(nr_workers)),
                    ('synchronism', row[5]),
                    ('budget', size)])

                self.dictConfigs.append([conf_dict, acc, cost, size])
                
            csv_file.close()
    
        print("Dataset read")
        return listConfig

    def listConfigSpace(self):
        return self.listConfigs

    def dictConfigSpace(self):
        return self.dictConfigs

    def compute(self, config, budget, working_directory, *args, **kwargs):

        """
        Get values from dataset
        example 
        df.loc[(df['learning_rate'] == 0.00001) & (df['batch_size'] == 256) & (df['vm_flavor'] == 't2.small') & (df['training_set_size'] == 1000)]
        """

        df = self.df_table

        if config['nrWorker'] == 1:
            sync = 'sync'
        else:
            sync = str(config['sync'])

        #print("olmaddksa")
        sub_df = df.loc[
            (df['learningRate'] == config['learningRate']) &
            (df['nrWorker'] == config['nrWorker']) &
            (df['batch'] == config['batch']) &
            (df['Flavor'] == str(config['Flavor'])) &
            (df['sync'] == sync) &
            (df['momentum'] == config['momentum']) &
            (df['sizeSeconds'] == int(budget))
        ]

        # check if table is empty!
        #print(sub_df)
        if sub_df.empty:
            raise BaseException(("Invalid configuration %s", config))

        size = sub_df.shape[0]
        chosen = self.rng.randint(0, size)
        acc = sub_df.iloc[chosen]['acc']
        training_time = budget
        cost = get_cost(sub_df.iloc[chosen]['Flavor'], sub_df.iloc[chosen]['nrWorker'], budget)
        self.total_cost = self.total_cost + cost
        loss = 1 - float(acc)

        return ({
            'loss': loss,  # remember: hyperjump always minimizes!
            'info': {'accuracy': acc,
                     'cost': cost,
                     'total cost': self.total_cost,
                     'accuracy loss': 1 - float(acc),
                     'budget': budget,
                     'training_time': training_time,
                     'error': "None"
                     }

        })

    @staticmethod
    def get_configspace(seed):

        '''
        Config space
            3(lr)*2(bs)*2(sync)*3(momentum)*2(vm_flavour)*2(workers) = 144
        '''
        cs = CS.ConfigurationSpace(seed=seed)
        batch = CSH.CategoricalHyperparameter('batch', [1, 2])
        learning_rate = CSH.CategoricalHyperparameter('learningRate', [0.000001, 0.00001, 0.0001])
        nr_worker = CSH.CategoricalHyperparameter('nrWorker', [1, 2])
        momentum = CSH.CategoricalHyperparameter('momentum', [0.9, 0.95, 0.99])
        sync = CSH.CategoricalHyperparameter('sync', ['async', 'sync'])
        flavor = CSH.CategoricalHyperparameter('Flavor', ['intel14_v1', 'intel14_v2'])

        cs.add_hyperparameters([batch, learning_rate, nr_worker, momentum, sync, flavor])

        return cs


if __name__ == "__main__":
    worker = UnetWorker(seed=0, run_id='0')
    cs = UnetWorker.get_configspace(12)

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=2250, working_directory='.')
