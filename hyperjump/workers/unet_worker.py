
import pandas as pd
import numpy as np
import math, copy
from random import seed
from random import randint
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import csv, pickle, time

from hyperjump.core.worker import Worker


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



class UnetWorker(Worker):
    def __init__(self, eta=2.0, pauseResume=True, checkpointing=True, factor=1.0, *args, **kwargs):
        if eta == 2.0:
            print("ETA=2 dataset file unet.csv ")
            name = './hyperjump/workers/data/unet_eta2.csv'
            self.Budgets = [1125, 2250, 4500, 9000, 18000]

        elif eta == 3.0:
            print("ETA=3 dataset file unet_eta3.csv ")
            name = './hyperjump/workers/data/unet_eta3.csv'
            self.Budgets = [222, 666, 2000, 6000, 18000]

        self.data  = self.load_dataset(name)
        self.trainingSet = dict()
        self.factor = factor
        self.pauseResume = pauseResume
        self.checkpointing = checkpointing

        super().__init__(**kwargs)


    #Read the dataset
    def load_dataset(self, CSV):
        _data = dict()

        print("Reading the dataset....")

        with open(CSV) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            next(csv_reader, None)

            for row in csv_reader:
                flavor = row[0] 
                nr_workers = int(row[1])
                batch_size = int(row[2])
                learning_rate = float(row[3])
                mommentum = float(row[4])
                synchronism = "async"
                size = float(row[6]) # budget is training time

                acc = float(row[8])
                cost = get_cost(row[0], nr_workers, size)

                #time = float(row[9])
                #cost = float(row[8])

                conf_dict = dict([
                    ('vm_flavor', flavor),
                    ('batch_size', int(batch_size)),
                    ('learning_rate', learning_rate),
                    ('momentum', float(mommentum)),
                    ('nrWorker', int(nr_workers)),
                    ('synchronism', synchronism),
                    ('budget', size)])

                key = pickle.dumps(conf_dict)
                _data[key] = [acc, size, cost]

            csv_file.close()
    
        print("Dataset read")
        return _data


    def getVal(self, config, budget , *args, **kwargs):
        config_with_budgets = {
            'vm_flavor': config['vm_flavor'],
            'batch_size': config['batch_size'],
            'learning_rate':config['learning_rate'],
            'momentum': config['momentum'],
            'nrWorker': config['nrWorker'],
            'synchronism': config['synchronism'],
            'budget': budget,
        }


        for kk in self.data.keys():
            conf = pickle.loads(kk)
            if conf['vm_flavor'] == config_with_budgets['vm_flavor'] \
                        and conf['batch_size'] == config_with_budgets['batch_size'] \
                        and conf['learning_rate'] == config_with_budgets['learning_rate'] \
                        and conf['momentum'] == config_with_budgets['momentum'] \
                        and conf['nrWorker'] == config_with_budgets['nrWorker']\
                        and conf['synchronism'] == config_with_budgets['synchronism']\
                        and conf['budget'] == config_with_budgets['budget']:

                        return config_with_budgets, self.data[kk][0], self.data[kk][1]
        return config_with_budgets, 0, budget



    def compute(self, config, budget, *args, **kwargs):
        budget = int(budget)
        starttime = time.time()

        config_with_budgets, acc, trainTime = self.getVal(config, budget) # budget is trainig time

        prev_time = 0
        prev_budget = 0 

        if self.pauseResume:
            for kk in self.trainingSet.keys():
                conf = pickle.loads(kk)
                if conf['vm_flavor'] == config_with_budgets['vm_flavor'] \
                                and conf['batch_size'] == config_with_budgets['batch_size'] \
                                and conf['learning_rate'] == config_with_budgets['learning_rate'] \
                                and conf['momentum'] == config_with_budgets['momentum'] \
                                and conf['nrWorker'] == config_with_budgets['nrWorker']\
                                and conf['synchronism'] == config_with_budgets['synchronism']:


                        if conf['budget'] > prev_budget: # restore the largest tested budget of that config
                            prev_time = self.trainingSet[kk][2]
                            prev_budget = conf['budget']


        working_time = (trainTime-prev_time)/(self.factor*1.0)
        if working_time > 0:
            time.sleep(working_time)
        
        key = pickle.dumps(config_with_budgets)
        self.trainingSet[key] = [config_with_budgets, acc, trainTime] # 

        checkpoint = []
        if self.checkpointing:
            for bb in self.Budgets:
                if bb < budget:
                    config_smaller_budget, _acc, _time = self.getVal(config, bb) # budget is trainig time
                    config['budget'] = bb
                    checkpoint.append([config, _acc, _time])

        if len(checkpoint) == 0:
            checkpoint = None

        t_now = time.time()

        self.logger.info("Running config " + str(config_with_budgets) + " that achieved a accuracy of " + str(acc) + "  during " + str(t_now - starttime) + " seconds " + str((t_now-starttime)*self.factor))

        return ({
            'loss': 1.0-acc,  # remember: hyperjump always minimizes!
            'info': {
                     'config': config_with_budgets,
                     'training_time': trainTime,
                     'start_time': starttime,
                     'end_time': t_now,
                     'real_time': (t_now-starttime)*self.factor,
                     'accuracy': acc,
                     'checkpoint': checkpoint,
                     }
            })


    @staticmethod
    def get_configspace(seed):
        cs = CS.ConfigurationSpace(seed=seed)

        flavor = CSH.CategoricalHyperparameter('vm_flavor', ['intel14_v1', 'intel14_v2'])
        batch = CSH.CategoricalHyperparameter('batch_size', [1, 2])
        learning_rate = CSH.CategoricalHyperparameter('learning_rate', [0.000001, 0.00001, 0.0001])
        momentum = CSH.CategoricalHyperparameter('momentum', [0.9, 0.95, 0.99])
        nr_worker = CSH.CategoricalHyperparameter('nrWorker', [1, 2])
        sync = CSH.CategoricalHyperparameter('synchronism', ['async', 'sync'])

        cs.add_hyperparameters([flavor, batch, learning_rate, momentum, nr_worker, sync])

        return cs




if __name__ == "__main__":
    worker = UnetWorker(seed=0, run_id='0')
    cs = UnetWorker.get_configspace(12)

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=2250, working_directory='.')
