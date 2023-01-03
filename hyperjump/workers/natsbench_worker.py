import numpy as np
import csv, sys
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import pickle, time
from copy import deepcopy

#from xautodl.config_utils import load_config
#from xautodl.datasets import get_datasets, SearchDataset
#from xautodl.procedures import prepare_seed, prepare_logger
#from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import CellStructure, get_search_spaces
from nats_bench import create

from hyperjump.core.worker import Worker
import logging

csv.field_size_limit(sys.maxsize)

SEARCH_SPACE = 'tss'


def get_topology_config_space(search_space, seed=10000, max_nodes=4):
    _cs = CS.ConfigurationSpace(seed=seed)
    #_cs = CS.ConfigurationSpace()
    # edge2index   = {}
    for i in range(1, max_nodes):
        for j in range(i):
            node_str = "{:}<-{:}".format(i, j)
            _cs.add_hyperparameter(
                CSH.CategoricalHyperparameter(node_str, search_space)
            )
    return _cs


def get_size_config_space(search_space,seed=10000):
    _cs = CS.ConfigurationSpace(seed=seed)
    #_cs = CS.ConfigurationSpace()
    for ilayer in range(search_space["numbers"]):
        node_str = "layer-{:}".format(ilayer)
        _cs.add_hyperparameter(
            CSH.CategoricalHyperparameter(node_str, search_space["candidates"])
        )
    return _cs


def config2topology_func(max_nodes=4):
    def config2structure(config):
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = config[node_str]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)
    return config2structure


def config2size_func(search_space):
    def config2structure(config):
        channels = []
        for ilayer in range(search_space["numbers"]):
            node_str = "layer-{:}".format(ilayer)
            channels.append(str(config[node_str]))
        return ":".join(channels)
    return config2structure



class NatsBenchWorker(Worker):
    def __init__(self, dataset, eta=2, pauseResume=True, checkpointing=True, factor=1.0, **kwargs):
        super().__init__(**kwargs)
        
        self.dataset = dataset

        self.trainingSet = dict()
        self.factor = factor
        self.pauseResume = pauseResume
        self.checkpointing = checkpointing
        self.eta = eta
        self.dataset_name ="ImageNet16-120" if "ImageNet16-120" in self.dataset else self.dataset.split('-')[-1]

        self.search_space = "tss" if "nats-tss" in self.dataset else "sss"
        global SEARCH_SPACE
        SEARCH_SPACE = self.search_space

        self._api  = create(None, self.search_space, fast_mode=True, verbose=False)

        self._api.reset_time()
        aux_search_space = get_search_spaces(self.search_space, "nats-bench")

        if self.search_space == "tss":
            #cs = get_topology_config_space(aux_search_space)
            config2structure = config2topology_func()
            self.maxBudget = "200"
            if self.eta==2:
                self.Budgets = [12, 25, 50, 100, 200]
            else:
                self.Budgets = [2, 7, 22, 66, 200]

        else:
            #cs = get_size_config_space(aux_search_space)
            config2structure = config2size_func(aux_search_space)
            self.maxBudget = "90"
            if self.eta==2:
                self.Budgets = [5, 11, 22, 45, 90]
            else:
                self.Budgets = [1, 3, 10, 30, 90]

        self.logger.info("search space " + SEARCH_SPACE)

        self.convert_func = config2structure


    def compare_dicts(self, dict1, dict2):
        if len(dict1) != len(dict2):
            return False

        for key in dict1:
            if key == "budget": continue
            if key not in dict2:
                return False
            if dict1[key] != dict2[key]:
                return False

        return True


    def compute(self, config, budget, *args, **kwargs):
        starttime = time.time()
        
        arch = self.convert_func(config)
        accuracy, _, trainTime, _ = self._api.simulate_train_eval(arch, self.dataset_name, iepoch=int(budget) - 1, hp=self.maxBudget)
        #this accuracy is in percentage
        acc = accuracy/100.0
        loss = 1.0 - acc

        config_with_budgets = deepcopy(config)
        config_with_budgets["budget"] = budget

        prev_time = 0
        prev_budget = 0 
        if self.pauseResume:
            for kk in self.trainingSet.keys():
                conf = pickle.loads(kk)
                if self.compare_dicts(conf, config_with_budgets): #same config (without budgets)
                    if conf['budget'] > prev_budget: # restore the largest tested budget of that config
                        prev_time = self.trainingSet[kk][2]
                        prev_budget = conf['budget']

        working_time = (trainTime-prev_time)/(self.factor*1.0)
        if working_time > 0:
            time.sleep(working_time)

        key = pickle.dumps(config_with_budgets)
        self.trainingSet[key] = [config_with_budgets, acc, trainTime] # 

        checkpoint = []
        if self.checkpointing: # and prev_time == 0:
            for bb in self.Budgets:
                if bb < budget:
                    _acc, _, _time, _ = self._api.simulate_train_eval(arch, self.dataset_name, iepoch=int(bb) - 1, hp=self.maxBudget)
                    #config_smaller_budget, _acc, _time = self.getVal(config, bb) # budget is epochs
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

    #@staticmethod
    def get_configspace(seed):
        aux_search_space = get_search_spaces(SEARCH_SPACE, "nats-bench")
        if SEARCH_SPACE == "tss":
            cs = get_topology_config_space(aux_search_space)
        else:
            cs = get_size_config_space(aux_search_space)
        return cs

