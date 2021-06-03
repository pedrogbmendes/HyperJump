import time, copy
import numpy as np

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util

from hyperjump.core.base_config_generator import base_config_generator
from hyperjump.optimizers.utils.tools import vector_to_conf
from hyperjump.optimizers.utils.tools import conf_to_vector
from hyperjump.optimizers.utils.tools import get_incumbent
from hyperjump.optimizers.utils.tools import print_progress_bar
from hyperjump.optimizers.utils.tools import print_custom_bar
from hyperjump.optimizers.maximizers.random_sampling import RandomSampling as rs


class RandomSampling(base_config_generator):
    """
        class to implement random sampling from a ConfigSpace
    """

    def __init__(self, configspace, incumbent=[], incumbent_value=-1, seed=0,
                 type_exp='fake', min_budget=1, max_budget=16, configspaceList=None, **kwargs):
        """

        Parameters:
        -----------

        configspace: ConfigSpace.ConfigurationSpace
            The configuration space to sample from. It contains the full
            specification of the Hyperparameters with their priors
        **kwargs:
            see  hyperband.config_generators.base.base_config_generator for additional arguments
        """

        if type_exp == 'fake':
            # [batch_size, learning_rate, num_cores, synchrony, vm_flavor, budget]
            lower = np.array([16, 0.00001, 8, 0, 0, min_budget])
            upper = np.array([256, 0.001, 80, 1, 3, max_budget])

            # Hyperparameter list without budgets
            hp_list = [[16, 256],
                       [0.00001, 0.0001, 0.001],
                       [8, 16, 32, 48, 64, 80],
                       [0, 1],
                       [0, 1, 2, 3]]
            # search_space_size = 1.440k #288*5

        elif type_exp == 'unet':
            # {Flavor, batch, learningRate, momentum, nrWorker, sync}
            lower = np.array([1, 1, 0.000001, 0.9, 1, 1, min_budget])
            upper = np.array([2, 2, 0.0001, 0.99, 2, 2, max_budget])

            # Hyperparameter list without budgets
            hp_list = [[1, 2],  # Flavor
                       [1, 2],  # batch
                       [0.000001, 0.00001, 0.0001],  # learningRate
                       [0.9, 0.95, 0.99],  # momentum
                       [1, 2],  # nrWorker
                       [1, 2]]  # sync
        # search_space_size = 720 #144*5

        elif type_exp == 'svm':
            # {Flavor, batch, learningRate, momentum, nrWorker, sync}
            lower = np.array([0, 0, 0.001, 0.001, min_budget])
            upper = np.array([3, 5, 100, 100, max_budget])

            # Hyperparameter list without budgets
            hp_list = [[0, 1, 2, 3],  # kernel
                       [0, 1, 2, 3, 4, 5],  # degree
                       [0.001, 0.01, 0.1, 1, 10, 100],  # gamma
                       [0.001, 0.01, 0.1, 1, 10, 100]]  # c
                       
        elif type_exp == 'mnist':
            '''
            dropout_rate',  ['0.0', '0.2', '0.4','0.6', '0.8'])
            learning_rate', ['0.000001', '0.00001', '0.0001', '0.001', '0.01'])
            num_fc_units',  ['8', '16', '32', '64', '128', '256'])
            num_filters_1', ['4', '8', '16', '32', '64'])
            num_filters_2', ['0', '4', '8', '16', '32', '64'])
            num_filters_3', ['0', '4', '8', '16', '32', '64'])
            sgd_momentum',  ['0.0', '0.2', '0.4','0.6', '0.8'])
            budget =,       [1, 2, 4, 8, 16])
            '''

            lower = np.array([0.0, 0.0000001, 8, 4, 0, 0, 0.0, 1])
            upper = np.array([0.8, 0.01, 256, 64, 64, 64, 0.8, 16])
            max_budget = 16
            min_budget = 1

            # CS = 5*5*6*5*6*6*5 = 135K
            # With budgets -> 135K*3 = 675k

            # Hyperparameter list without budgets:
            hp_list = [[0.0, 0.2, 0.4, 0.6, 0.8],
                       [1e-06, 1e-05, 0.0001, 0.001, 0.01],
                       [8, 16, 32, 64, 128, 256],
                       [4, 8, 16, 32, 64],
                       [0, 4, 8, 16, 32, 64],
                       [0, 4, 8, 16, 32, 64],
                       [0.0, 0.2, 0.4, 0.6, 0.8]]
        else:
            raise BaseException(("Invalid/unimplemented experiment %s", type_exp))
 
        super().__init__(**kwargs)
        self.listConfigSpace = configspaceList
        self.seed = seed
        self.rng = np.random.RandomState(np.int64(seed))
        self.configspace = configspace
        self.losses = []
        self.training_set = []
        self.incumbent = incumbent
        self.incumbent_value = -1
        self.incumbent_line = -1
        self.type_exp = type_exp
        self.lower = lower
        self.upper = upper
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.model_available = False

        self.maximize_func = rs(None, lower, upper, hp_list, rng=self.rng)

        # -----------------------------------------------------------------
        self.config_num = 0
        self.total_time = time.time()
        self.total_results = 128
        print_progress_bar(0, self.total_results, prefix='\nProgress:', suffix="Complete  Finished in:h???:m??:s??"
                           , length=50)

    def random_sample(self, budget, training_set):
        random_hps = self.maximize_func.get_random_sample(budget, training_set, self.lower[:-1], self.upper[:-1])
        rand_vector = vector_to_conf(random_hps, self.type_exp)
        return ConfigSpace.Configuration(self.configspace, values=rand_vector)

    def get_config(self, budget, no_total_configs=1):
        useless = self.rng.rand()
        overhead_time = time.time()

        if self.type_exp == 'unet' or self.type_exp == 'svm' :

            if isinstance(self.training_set, list):
                training_set_ = self.training_set
            else:
                training_set_ = self.training_set.tolist()

            listConfigsBudget = []
            for c in self.listConfigSpace:
                if int(c[-1]) == int(budget):
                    listConfigsBudget.append(c)

            
            rand_int_ = np.random.randint(0, len(listConfigsBudget))
            config = listConfigsBudget[rand_int_]
            rand_vector = vector_to_conf(config, self.type_exp)
            sample =  ConfigSpace.Configuration(self.configspace, values=rand_vector).get_dictionary()

            if config not in training_set_:
                #print("not in training set " + str(config))

                info_dict = {}
                info_dict["incumbent_line"] = self.incumbent_line
                info_dict["incumbent_value"] = self.incumbent_value

                info_dict['model_based_pick'] = False
                info_dict['overhead_time'] = time.time() - overhead_time
                return sample, info_dict
   
            #print("repeated confgig in training set " + str(config))

            return self.get_config(budget)

        else:
            sample = self.random_sample(budget, self.training_set).get_dictionary()

            info_dict = {}
            info_dict["incumbent_line"] = self.incumbent_line
            info_dict["incumbent_value"] = self.incumbent_value

            info_dict['model_based_pick'] = False
            info_dict['overhead_time'] = time.time() - overhead_time
        return sample, info_dict


    def new_result(self, job, update_model=True):
        """
        function to register finished runs

        Every time a run has finished, this function should be called
        to register it with the result logger. If overwritten, make
        sure to call this method from the base class to ensure proper
        logging.


        Parameters:
        -----------
        job: hyperjump.distributed.dispatcher.Job object
            contains all the info about the run
        """
        # print("__________________________________________________________\nI got a new result!!")
        super().new_result(job)
        loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf
        budget = job.kwargs["budget"]
        conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])
        conf_vec = conf_to_vector(conf, self.type_exp)
        # print(conf)
        conf_vec_b = np.append(conf_vec, budget)

        self.config_num += 1

        print_custom_bar((time.time() - self.total_time), self.config_num, self.total_results)

        if len(self.training_set) == 0:
            self.training_set = conf_vec_b
            self.losses.append(loss)

        else:
            self.training_set = np.vstack([self.training_set, conf_vec_b])
            self.losses = np.append(self.losses, loss)

        # New result!!!!

        # Estimate incumbent
        if self.training_set.shape == (len(self.upper),):
            if self.training_set[-1] == self.max_budget:
                self.incumbent_value, self.incumbent, self.incumbent_line = (self.losses[0], self.training_set[0], 0)
            return
        elif np.any(self.training_set[:, len(self.upper) - 1] == self.max_budget):
            self.incumbent_value, self.incumbent, self.incumbent_line = get_incumbent(self.training_set, self.losses,
                                                                                      self.max_budget)
