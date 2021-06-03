import logging
import time

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import george
import sys
import copy, random

from hyperjump.core.base_config_generator import base_config_generator
from hyperjump.optimizers.priors.default_priors import DefaultPrior
from hyperjump.optimizers.priors.env_priors import EnvPrior
from hyperjump.optimizers.models.gaussian_process import GaussianProcess
from hyperjump.optimizers.models.fabolas_gp import FabolasGP
from hyperjump.optimizers.models.trimtuner_dt import EnsembleDTs
from hyperjump.optimizers.acquisition_functions.ei import EI
from hyperjump.optimizers.maximizers.random_sampling import RandomSampling
from hyperjump.optimizers.initial_design import init_latin_hypercube_sampling
from hyperjump.optimizers.utils.tools import vector_to_conf as vector_to_conf
from hyperjump.optimizers.utils.tools import conf_to_vector as conf_to_vector
from hyperjump.optimizers.utils.tools import get_incumbent
from hyperjump.optimizers.utils.tools import print_progress_bar
from hyperjump.optimizers.utils.tools import print_custom_bar

logging.basicConfig(level=logging.WARNING)


class HYPERJUMP(base_config_generator):
    def __init__(self, configspace, min_points_in_model=None,
                 top_n_percent=15, num_samples=64, random_fraction=1/3,
                 bandwidth_factor=3, min_bandwidth=1e-3, min_budget=1, max_budget=16, incumbent=[],
                 incumbent_value=-1, seed=1, type_exp='fake', algorithm_variant='FBS', configspaceList=None, 
                 budgets=None, **kwargs):
        """
        Fits for each given budget a kernel density estimator on the best N percent of the
        evaluated configurations on this budget.

        Para todos os budgets, "encaixa" um KDE no top 15% das melhores configurações


        Parameters:
        -----------
        configspace: ConfigSpace
            Configuration space object
        top_n_percent: int
            Determines the percentile of configurations that will be used as training data
            for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
            for training.
        min_points_in_model: int
            minimum number of datapoints needed to fit a model
        num_samples: int
            number of samples drawn to optimize EI via sampling
        random_fraction: float
            fraction of random configurations returned
        bandwidth_factor: float
            widens the bandwidth for contiuous parameters for proposed points to optimize EI
        min_bandwidth: float
            to keep diversity, even when all (good) samples have the same value for one of the parameters,
            a minimum bandwidth (Default: 1e-3) is used instead of zero.

        """
        super().__init__(**kwargs)
        self.top_n_percent = top_n_percent
        self.configspace = configspace
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth
        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.configspace.get_hyperparameters()) + 1

        if self.min_points_in_model < len(self.configspace.get_hyperparameters()) + 1:
            self.logger.warning('Invalid min_points_in_model value. Setting it to %i' % (
                    len(self.configspace.get_hyperparameters()) + 1))
            self.min_points_in_model = len(self.configspace.get_hyperparameters()) + 1

        self.num_samples = num_samples
        self.random_fraction = random_fraction
        self.configs_tested_stage = 0

        hps = self.configspace.get_hyperparameters()

        self.kde_vartypes = ""
        self.vartypes = []

        # here, add the budget to the var type??
        for h in hps:
            if hasattr(h, 'sequence'):
                raise RuntimeError(
                    'This version on BOHB does not support ordinal hyperparameters. Please encode %s as an integer parameter!' % (
                        h.name))

            if hasattr(h, 'choices'):
                self.kde_vartypes += 'u'
                self.vartypes += [len(h.choices)]
            else:
                self.kde_vartypes += 'c'
                self.vartypes += [0]

        self.vartypes = np.array(self.vartypes, dtype=int)

        # store precomputed probs for the categorical parameters
        self.cat_probs = []

        # self.configs = dict()
        # self.losses = dict()
        # self.kde_models = dict()

        self.configs = []
        self.losses = []
        self.training_set = np.array([])
        self.configs2Test = np.array([])
        self.costs = []
        # ------------------------------------------------------------------------

        if type_exp == 'fake' or type_exp == 'fake_time':
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



        elif type_exp == 'fake_all'  or type_exp == 'fake_time_all':
            # [batch_size, learning_rate, network, num_cores, synchrony, vm_flavor, , budget]
            lower = np.array([16, 0.00001, 0, 8, 0, 0, min_budget])
            upper = np.array([256, 0.001, 2, 80, 1, 3, max_budget])

            # Hyperparameter list without budgets
            hp_list = [[16, 256],
                       [0.00001, 0.0001, 0.001],
                       [0, 1, 2],
                       [8, 16, 32, 48, 64, 80],
                       [0, 1],
                       [0, 1, 2, 3]]


        elif type_exp == 'mnist':
            # [dropout_rate, learning_rate, num_fc_units, num_filters_1, num_filters_2, sgd_momentum, budget]
            # lower = np.array([1, 1, 1, 1, 1, 1, 1])
            # upper = np.array([3, 4, 3, 3, 3, 4, 9])
            # search_space_size = 3.888k #1296*3

            '''
            dropout_rate', 	['0.0', '0.2', '0.4','0.6', '0.8'])
            learning_rate', ['0.000001', '0.00001', '0.0001', '0.001', '0.01'])
            num_fc_units', 	['8', '16', '32', '64', '128', '256'])
            num_filters_1', ['4', '8', '16', '32', '64'])
            num_filters_2', ['0', '4', '8', '16', '32', '64'])
            num_filters_3', ['0', '4', '8', '16', '32', '64'])
            sgd_momentum', 	['0.0', '0.2', '0.4','0.6', '0.8'])
            budget =, 		[1, 2, 4, 8, 16])
            '''
            lower = np.array([0.0, 0.0000001, 8, 4, 0, 0, 0.0, min_budget])
            upper = np.array([0.8, 0.01, 256, 64, 64, 64, 0.8, max_budget])
            # CS = 5*5*6*5*6*6*5 = 135K
            # With budgets -> 135K*5 = 675K

            # Hyperparameter list without budgets
            hp_list = [[0.0, 0.2, 0.4, 0.6, 0.8],
                       [1e-06, 1e-05, 0.0001, 0.001, 0.01],
                       [8, 16, 32, 64, 128, 256],
                       [4, 8, 16, 32, 64],
                       [0, 4, 8, 16, 32, 64],
                       [0, 4, 8, 16, 32, 64],
                       [0.0, 0.2, 0.4, 0.6, 0.8]]

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
            lower = np.array([1, 10e-6, 10e-6, min_budget])
            #upper = np.array([6, 100, 100, max_budget])
            upper = np.array([3, 100, 100, max_budget])

            # Hyperparameter list without budgets
            hp_list = [[1, 2, 3],  # kernel
                       [0.001, 0.01, 0.1, 1, 10, 100],  # gamma
                       [0.001, 0.01, 0.1, 1, 10, 100]]  # c

        else:
            raise BaseException(("Invalid/unimplemented experiment %s", type_exp))



        np.random.seed(seed)
        rng = np.random.RandomState(np.int64(seed))
        random.seed(seed)
        n_dims = lower.shape[0]

        ####            OLD MODEL
        #cov_amp = 2
        #initial_ls = np.ones([n_dims])
        #exp_kernel = george.kernels.Matern52Kernel(initial_ls, ndim=n_dims)
        #kernel = cov_amp * exp_kernel
        #prior = DefaultPrior(len(kernel) + 1)
        #######

        FlagCostModel = False
        if random_fraction == 1:
            acquisition_func = None
            model =None
            cost_model = None
        else:

            if 'DT' not in algorithm_variant:
                cov_amp = 1  # Covariance amplitude
                kernel = cov_amp

                # ARD Kernel for the configuration space
                for d in range(n_dims-1):
                    kernel *= george.kernels.Matern52Kernel(np.ones([1])*0.01, ndim=n_dims, axes=d)

                # Kernel for the environmental variable
                # We use (1-s)**2 as basis function for the Bayesian linear kernel
                env_kernel = george.kernels.BayesianLinearRegressionKernel(log_a=0.1,log_b=0.1,ndim=n_dims,axes=n_dims-1)
                kernel *= env_kernel
        
                prior = EnvPrior(len(kernel) + 1, n_ls=n_dims-1, n_lr=2, rng=rng)
                quadratic_bf = lambda x: (1 - x) ** 2

                model = FabolasGP(kernel,
                                basis_function=quadratic_bf,
                                prior=prior,
                                normalize_output=False,
                                lower=lower,
                                upper=upper,
                                rng=rng)

                #model = GaussianProcess(kernel, prior=prior, rng=rng, normalize_output=False, normalize_input=True, lower=lower, upper=upper)

                if FlagCostModel or "EI$" in algorithm_variant or "Hybrid" in algorithm_variant:
                    self.has_cost_model = True
                    
                    # Define model for the cost function
                    cost_cov_amp = 1
                    cost_kernel = cost_cov_amp

                    # ARD Kernel for the configuration space
                    for d in range(n_dims-1):
                        cost_kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01, ndim=n_dims , axes=d)

                    cost_env_kernel = george.kernels.BayesianLinearRegressionKernel(log_a=0.1, log_b=0.1, ndim=n_dims,axes=n_dims-1)
                    cost_kernel *= cost_env_kernel

                    cost_prior = EnvPrior(len(cost_kernel) + 1, n_ls=n_dims-1, n_lr=2, rng=rng)
                    linear_bf = lambda x: x

                    cost_model = FabolasGP(cost_kernel,
                                basis_function=linear_bf,
                                prior=cost_prior,
                                normalize_output=False,
                                lower=lower,
                                upper=upper,
                                rng=rng)

                    #cost_model = GaussianProcess(cost_kernel, prior=cost_prior, rng=rng, normalize_output=False, normalize_input=True, lower=lower, upper=upper)
                else:
                    self.has_cost_model = False
                    cost_model = None


            else:
                model = EnsembleDTs(30, rng)

                if "EI$" in algorithm_variant or "Hybrid" in algorithm_variant:
                    self.has_cost_model = True
                    cost_model = EnsembleDTs(30, rng)

                else:
                    self.has_cost_model = False
                    cost_model = None

            acquisition_func = EI(model)

        maximize_func = RandomSampling(acquisition_func, lower, upper, hp_list, seed, rng=rng, min_budget=min_budget,
                                       full_budget=max_budget, uses_budget=True)

        # ------------------------------------------------------------------------


        ########################################################################
        self.listConfigSpace = configspaceList
        self.rng = rng
        self.seed = seed
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.model = model
        self.cost_model = cost_model
        self.acquisition_func = acquisition_func
        self.maximize_func = maximize_func
        self.initial_design = init_latin_hypercube_sampling
        self.lower = lower
        self.upper = upper
        self.hp_list = hp_list
        self.incumbent = incumbent
        self.incumbent_value = -1
        self.incumbent_line = -1
        self.algorithm_variant = algorithm_variant
        self.type_exp = type_exp
        self.model_available = False
        self.budgets = budgets
        ########################################################################

        # self.random_fraction=1

        # -------------------------------------------------------------------------
        self.total_time = time.time()
        self.total_results = 128
        self.config_num = 0
        self.sampled = False
        print_progress_bar(0, self.total_results, prefix='\nProgress:', suffix="Complete  Finished in:h???:m??:s??", length=50)
        self.countDetph = 0 

        self.flag_full_budget = True
        self.actualSet2Test = []
        self.training_time = 0.0

        self.firstModel = True
        #self.listTrainingtimes = open("times.txt", "a")
        self.magicNumber2changeModel = 100 #min_points_in_model ^ 2
        self.snapshooting = True 
        self.SearchSpace = []

    def updateSearchSpace(self, searchSpace):
        self.SearchSpace = searchSpace

    def random_sample(self, budget, training_set):
        if self.type_exp ==  "unet" or self.type_exp ==  "svm":
            r_int = self.rng.randint(len(self.listConfigSpace))
            random_hps =  copy.deepcopy(self.listConfigSpace[r_int]) 
            #while random_hps not in self.listConfigSpace:
            #    r_int = self.rng.randint(len(self.listConfigSpace))
            #    random_hps = self.listConfigSpace[r_int]  

            random_hps_maxBud = copy.deepcopy(random_hps)
            random_hps_maxBud[-1] = self.max_budget
            random_hps[-1] = budget   


            count_it = 0
            while random_hps in self.actualSet2Test or random_hps_maxBud in training_set.tolist():
                r_int = self.rng.randint(len(self.listConfigSpace))
                random_hps =  copy.deepcopy(self.listConfigSpace[r_int]) 
                random_hps_maxBud = copy.deepcopy(random_hps)
                random_hps_maxBud[-1] = self.max_budget
                random_hps[-1] = budget  
                #print(random_hps)
                count_it +=1
                if count_it >100:
                    break

            #random_hps[-1] = budget

            self.actualSet2Test.append(copy.copy(random_hps))
            rand_vector = vector_to_conf(random_hps, self.type_exp)

        else:
            if self.flag_full_budget:
                random_hps = self.maximize_func.get_random_sample(self.max_budget, training_set, self.lower[:-1], self.upper[:-1])
            else:
                random_hps = self.maximize_func.get_random_sample(budget, training_set, self.lower[:-1], self.upper[:-1])
            rand_vector = vector_to_conf(random_hps, self.type_exp)

        print(rand_vector)
        return ConfigSpace.Configuration(self.configspace, values=rand_vector)

    def reset_testedConfig_counter(self):
        self.configs_tested_stage = 0

    def verifySets(self, conf, training_set_, configs2Test_):
        #conf = np.array(conf)

        for c in training_set_:
            c_ = np.array(c)
            if np.array_equal(c_, conf):
                return True

        for c in configs2Test_:
            c_ = np.array(c)
            if np.array_equal(c_, conf):
                return True

        return False


    def verifyConfig(self, sample, budget):
        v = conf_to_vector(sample, self.type_exp)
        conf_v = np.append(v, budget)

        if conf_v in self.training_set: 
            print("repeat " + str(conf_v))
            return True

        return False


    def get_listConfigs(self, budget, no_total_configs=1):  
        sample = None
        start_time = time.time()

     
        rand_or_Model = np.zeros(no_total_configs, dtype=int)
        for i in range(no_total_configs):
            if (random.uniform(0, 1) < self.random_fraction
                    or self.training_set.shape == (len(self.lower),) 
                    or len(self.training_set) <= self.min_points_in_model):   #model is not available
                
                rand_or_Model[i] = 1

        listConfigs = []
        listData = []
        #print(rand_or_Model)
        for i in range(no_total_configs):

        #if (self.configs_tested_stage < np.rint(self.random_fraction * no_total_configs)  #sample random
        #if (self.rng.rand() < self.random_fraction
        #if (random.uniform(0, 1) < self.random_fraction
        #        or self.training_set.shape == (len(self.lower),) 
        #        or len(self.training_set) <= self.min_points_in_model):   #model is not available
            if rand_or_Model[i] == 1:

                sample = self.random_sample(budget, self.configs2Test)
                sample = self.check_active_hps(sample, budget)
                #print(sample)

                self.logger.debug('done sampling a new configuration.')

                info_dict = {}
                info_dict['predicted_loss_mean'] = -1
                info_dict['predicted_loss_stdv'] = -1
                info_dict['model_based_pick'] = False
                info_dict["incumbent_line"] = self.incumbent_line
                info_dict["incumbent_value"] = self.incumbent_value
                info_dict['overhead_time'] = (time.time() - start_time)
        
                conf_vec = conf_to_vector(sample, self.type_exp)
                conf_vec_b = np.append(conf_vec, budget)

                if len(self.configs2Test) == 0:
                    self.configs2Test = conf_vec_b
                else:
                    self.configs2Test = np.vstack([self.configs2Test, conf_vec_b])

                self.configs_tested_stage += 1
            
                listConfigs.append([sample, info_dict])

            else:

                #try:
                        # Choose next point to evaluate
                t = time.time()
                if not listData:
                    self.training() # training models when we have the enough configs

                    target_budget = budget
                    target_budget_cost = budget
                    if 'CBS' not in self.algorithm_variant:
                        target_budget = self.max_budget
                        if 'FBS' in self.algorithm_variant:
                            target_budget_cost = self.max_budget

                    if self.type_exp == 'fake' or self.type_exp == 'fake_all' or self.type_exp == 'fake_time' or self.type_exp == 'fake_time_all':
                        if  "EI$" in self.algorithm_variant or "Hybrid" in self.algorithm_variant: #EI/$
                            if self.flag_full_budget:
                                listData = self.maximize_func.maximize_all_returnBests(self.configs2Test, target_budget,
                                                                            self.cost_model, target_budget_cost, self.max_budget, no_total_configs)
                            else:
                                listData = self.maximize_func.maximize_all_returnBests(self.configs2Test, target_budget,
                                                                            self.cost_model, target_budget_cost, budget, no_total_configs)
                        else: #EI
                            if self.flag_full_budget:
                                listData = self.maximize_func.maximize_all_returnBests(self.configs2Test, target_budget,
                                                                            None, target_budget_cost, self.max_budget, no_total_configs)
                            else:
                                listData = self.maximize_func.maximize_all_returnBests(self.configs2Test, target_budget,
                                                                            None, target_budget_cost, budget, no_total_configs)

                    elif self.type_exp == 'unet' or self.type_exp == 'svm':
                        if  "EI$" in self.algorithm_variant or "Hybrid" in self.algorithm_variant: #EI/$
                            if self.flag_full_budget:
                                listData= self.maximize_func.maximize_list_returnBests(self.configs2Test, target_budget,
                                                                            self.cost_model, target_budget_cost, self.max_budget, self.listConfigSpace, no_total_configs)

                            else:
                                listData= self.maximize_func.maximize_list_returnBests(self.configs2Test, target_budget,
                                                                            self.cost_model, target_budget_cost, budget, self.listConfigSpace, no_total_configs)

                        else: #EI
                            if self.flag_full_budget:
                                listData = self.maximize_func.maximize_list_returnBests(self.configs2Test, target_budget,
                                                                            None, target_budget_cost, self.max_budget, self.listConfigSpace, no_total_configs)  
                            else:                    
                                listData = self.maximize_func.maximize_list_returnBests(self.configs2Test, target_budget,
                                                                            None, target_budget_cost, budget, self.listConfigSpace, no_total_configs)  

                    else:
                        if  "EI$" in self.algorithm_variant or "Hybrid" in self.algorithm_variant: #EI/$
                            if self.flag_full_budget:
                                listData = self.maximize_func.maximize_ei_returnBests(self.configs2Test, self.max_budget, target_budget,
                                                                            target_budget_cost, no_total_configs, cost_model=self.cost_model)
                            else:
                                listData = self.maximize_func.maximize_ei_returnBests(self.configs2Test, budget, target_budget,
                                                                            target_budget_cost, no_total_configs, cost_model=self.cost_model)

                        else: #EI
                            if self.flag_full_budget:
                                listData = self.maximize_func.maximize_ei_returnBests(self.configs2Test, self.max_budget, target_budget,
                                                                            target_budget_cost, no_total_configs, cost_model=None)
                            else:
                                listData = self.maximize_func.maximize_ei_returnBests(self.configs2Test, budget, target_budget,
                                                                            target_budget_cost, no_total_configs, cost_model=None)
                    
                    
                if len(listData) == 0: #go random
                    sample = self.random_sample(budget, self.configs2Test)
                    sample = self.check_active_hps(sample, budget)
                    conf_vec = conf_to_vector(sample, self.type_exp)
                    conf_vec_b = np.append(conf_vec, budget)

                    if len(self.configs2Test) == 0: self.configs2Test = conf_vec_b
                    else: self.configs2Test = np.vstack([self.configs2Test, conf_vec_b])

                else:
                    confi = listData.pop(0) #pop the first element in the list with the highest ei value
                    best_hps = confi[0]
                    best_vector = vector_to_conf(best_hps, self.type_exp)
                    sample = self.transform_config(best_vector)
                    sample = self.check_active_hps(sample, budget)
                    conf_vec = conf_to_vector(sample, self.type_exp)
                    conf_vec_b = np.append(conf_vec, budget)
                    config_vec_b = list(conf_vec_b)

                    #print(config_vec_b)
                    #print(self.actualSet2Test)
                    #print()
                    while config_vec_b in self.actualSet2Test:
                        if len(listData) == 0: #go random
                            sample = self.random_sample(budget, self.configs2Test)
                            sample = self.check_active_hps(sample, budget)
                            conf_vec = conf_to_vector(sample, self.type_exp)
                            conf_vec_b = np.append(conf_vec, budget)

                            if len(self.configs2Test) == 0: self.configs2Test = conf_vec_b
                            else: self.configs2Test = np.vstack([self.configs2Test, conf_vec_b])
                            break
                        confi = listData.pop(0) #pop the first element in the list with the highest ei value
                        best_hps = confi[0]    
                        best_vector = vector_to_conf(best_hps, self.type_exp)
                        sample = self.transform_config(best_vector)
                        sample = self.check_active_hps(sample, budget)
                        conf_vec = conf_to_vector(sample, self.type_exp)
                        conf_vec_b = np.append(conf_vec, budget)
                        config_vec_b = list(conf_vec_b)

                        print(config_vec_b)
                        print(self.actualSet2Test)
                        print(" --")

                    #best_vector = vector_to_conf(best_hps, self.type_exp)
                    #sample = self.transform_config(best_vector)
                    #sample = self.check_active_hps(sample, budget)
                    #conf_vec = conf_to_vector(sample, self.type_exp)
                    #conf_vec_b = np.append(conf_vec, budget)

                    if len(self.configs2Test) == 0:
                        self.configs2Test = conf_vec_b
                    else:
                        self.configs2Test = np.vstack([self.configs2Test, conf_vec_b])

                    self.actualSet2Test.append(list(conf_vec_b))

                info_dict = {}    
                if len(listData) == 0:
                    info_dict['predicted_loss_mean'] = -1
                    info_dict['predicted_loss_stdv'] = -1
                    info_dict['model_based_pick'] = False
                else:            
                    info_dict['predicted_loss_mean'] = confi[2][0]
                    info_dict['predicted_loss_stdv'] = confi[3][0]
                    info_dict['model_based_pick'] = True

                info_dict["incumbent_line"] = self.incumbent_line
                info_dict["incumbent_value"] = self.incumbent_value
                info_dict['overhead_time'] = (time.time() - start_time)

                self.logger.debug('done sampling a new configuration.')

                self.configs_tested_stage += 1
                listConfigs.append([sample, info_dict])
        #for i in range(len(listConfigs)):
        #    print(listConfigs[i][0])
        self.actualSet2Test.clear()
        #print(len(listConfigs))
        return listConfigs


    def get_config(self, budget, no_total_configs=1):
        """
        Function to sample a new configuration

        This function is called inside Hyperband to query a new configuration


        Parameters:
        -----------
        budget: float
            the budget for which this configuration is scheduled

        returns: config
            should return a valid configuration

        """
        sample = None
        info_dict = {}
        start_time = time.time()
        info_dict['predicted_loss_mean'] = -1
        info_dict['predicted_loss_stdv'] = -1



        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        # If there is no model avalilable sample a config randomly. Also, even if there is a model available,
        # sample randomly if the random number falls in a certain range.
        aux_rand = random.uniform(0, 1)
        #print("rand " + str(aux_rand) + "  frac  " + str(self.random_fraction))
        if  aux_rand < self.random_fraction:
        #if self.rng.rand() < self.random_fraction:
        #if self.configs_tested_stage < np.rint(self.random_fraction * no_total_configs):
            #print("random selection -> " + str(self.configs_tested_stage) + "  " + str(np.rint(self.random_fraction * no_total_configs)) + "  " + str(no_total_configs))
            #sample = self.random_sample(budget, self.training_set)
            sample = self.random_sample(budget, self.configs2Test)
            info_dict['model_based_pick'] = False
            

        elif self.training_set.shape == (len(self.lower),) or len(self.training_set) <= self.min_points_in_model:
        #elif self.configs2Test.shape == (len(self.lower),) or len(self.training_set) <= self.min_points_in_model:
            #sample = self.random_sample(budget, self.training_set)
            sample = self.random_sample(budget, self.configs2Test)
            info_dict['model_based_pick'] = False

        
        if sample is None:
                self.training() # training models when we have the enough configs

            #try:
                # Choose next point to evaluate
                t = time.time()
                target_budget = budget
                target_budget_cost = budget
                if 'CBS' not in self.algorithm_variant:
                    target_budget = self.max_budget
                    if 'FBS' in self.algorithm_variant:
                        target_budget_cost = self.max_budget

                if self.type_exp == 'fake' or self.type_exp == 'fake_all' or self.type_exp == 'fake_time' or self.type_exp == 'fake_time_all':
                    #best_hps, loss, sigma = self.maximize_func.maximize_all(self.training_set, target_budget,
                    #                                                        self.cost_model, target_budget_cost, budget)
                    if  "EI$" in self.algorithm_variant or "Hybrid" in self.algorithm_variant:
                        #EI/$
                        best_hps, loss, sigma = self.maximize_func.maximize_all(self.configs2Test, target_budget,
                                                                            self.cost_model, target_budget_cost, budget)
                    else:
                        #EI
                        best_hps, loss, sigma = self.maximize_func.maximize_all(self.configs2Test, target_budget,
                                                                            None, target_budget_cost, budget)
                elif self.type_exp == 'unet' or self.type_exp == 'svm':
                    if  "EI$" in self.algorithm_variant or "Hybrid" in self.algorithm_variant:
                        #EI/$
                        best_hps, loss, sigma = self.maximize_func.maximize_list(self.configs2Test, target_budget,
                                                                            self.cost_model, target_budget_cost, budget, self.listConfigSpace)
                    else:
                        #EI
                        best_hps, loss, sigma = self.maximize_func.maximize_list(self.configs2Test, target_budget,
                                                                            None, target_budget_cost, budget, self.listConfigSpace)                                                                         
                else:
                    #best_hps, loss, sigma = self.maximize_func.maximize_ei(self.training_set, budget, target_budget,
                    #                                                       target_budget_cost,
                    #                                                       cost_model=self.cost_model)

                    if  "EI$" in self.algorithm_variant or "Hybrid" in self.algorithm_variant:
                        #EI/$
                        best_hps, loss, sigma = self.maximize_func.maximize_ei(self.configs2Test, budget, target_budget,
                                                                           target_budget_cost,
                                                                           cost_model=self.cost_model)
                    else:
                        #EI
                        best_hps, loss, sigma = self.maximize_func.maximize_ei(self.configs2Test, budget, target_budget,
                                                                           target_budget_cost,
                                                                           cost_model=None)
               
                # print(best_hps)
                info_dict['predicted_loss_mean'], info_dict['predicted_loss_stdv'] = loss[0], sigma[0]

                best_vector = vector_to_conf(best_hps, self.type_exp)

                sample = self.transform_config(best_vector)
                info_dict['model_based_pick'] = True

            #except Exception:
            #    self.logger.warning("Sampling based optimization failed\n", sys.exc_info())
            #    #sample = self.random_sample(budget, self.training_set)
            #    sample = self.random_sample(budget, self.configs2Test)
            #    info_dict['model_based_pick'] = False
            #    sys.exit(-1)

        sample = self.check_active_hps(sample, budget)

        self.logger.debug('done sampling a new configuration.')

        # Remove the config. from the models pool of choices
        # self.maximize_func.removeFromPool(conf_to_vector(sample, self.type_exp), budget)
        # self.sampled = True
        info_dict["incumbent_line"] = self.incumbent_line
        info_dict["incumbent_value"] = self.incumbent_value
        info_dict['overhead_time'] = (time.time() - start_time)
    
        conf_vec = conf_to_vector(sample, self.type_exp)
        conf_vec_b = np.append(conf_vec, budget)

        if len(self.configs2Test) == 0:
            self.configs2Test = conf_vec_b
        else:
            self.configs2Test = np.vstack([self.configs2Test, conf_vec_b])

        #if not self.configExist(conf_vec_b):
        #    self.countDetph += 1
        #    return self.get_config(budget, no_total_configs)

        self.configs_tested_stage += 1

        return sample, info_dict



    def configsToJump(self, jump):
        return
        #self.config_num += jump

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
        super().new_result(job)

        self.model_available = False ## new config. we need to retrain the model
        if job.result is None:
            # One could skip crashed results, but we decided to
            # assign a +inf loss and count them as bad configurations
            loss = np.inf
        else:
            # same for non numeric losses.
            # Note that this means losses of minus infinity will count as bad!
            loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf

        budget = int(job.kwargs["budget"])

        if self.type_exp == 'mnist':
            cost = float(job.result['info']['training_time'])
        else:
            cost = float(job.result['info']['cost'])

        self.config_num += 1
        print_custom_bar((time.time() - self.total_time), self.config_num, self.total_results)

        # We want to get a numerical representation of the configuration in the original space
        conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])
        #print("confff   "  +str(conf)  )
        conf_vec = conf_to_vector(conf, self.type_exp)
        #print("conf_vec   "  +str(conf_vec)  )

        # if not self.sampled:
        #	#Remove the config. from the models pool of choices
        #	self.maximize_func.removeFromPool(conf_vec, budget)
        # self.sampled = False


        conf_vec_b = np.append(conf_vec, budget)
        #print("conf_vec_b   "  +str(conf_vec_b)  )
        #print(self.training_set)

        if len(self.training_set) == 0:
            self.training_set = conf_vec_b
            self.losses = np.array([loss])
            self.costs = np.array([cost])

        else:
            #if conf_vec_b not in self.training_set:
                self.training_set = np.vstack([self.training_set, conf_vec_b])
                
                self.losses = np.append(self.losses, loss)
                self.costs = np.append(self.costs, cost)


        if self.snapshooting and self.budgets is not None and self.type_exp != "svm" and self.type_exp != "mnist":
            for bud in self.budgets:
                if bud < budget:
                    conf_vec_b_aux = copy.deepcopy(conf_vec_b)
                    conf_vec_b_aux[-1] = bud
                    #if conf_vec_b_aux in self.training_set: continue

                    c = vector_to_conf(conf_vec_b_aux, self.type_exp)

                    if self.type_exp == "unet":
                        conf_dict = dict([
                            ('vm_flavor', c["Flavor"]),
                            ('batch_size', int(c["batch"])),
                            ('learning_rate', c["learningRate"]),
                            ('momentum', float(c["momentum"])),
                            ('nrWorker', int(c["nrWorker"])),
                            ('synchronism', c["sync"]),
                            ('budget', int(bud))])

                    elif self.type_exp == "fake_all" or self.type_exp == "fake_time_all":
                        conf_dict = dict([
                            ('vm_flavor', c["vm_flavor"]),
                            ('batch_size', int(c["batch_size"])),
                            ('learning_rate', c["learning_rate"]),
                            ('num_cores', int(c["num_cores"])),
                            ('synchronism', c["synchronism"]),
                            ('network', c["network"]),
                            ('budget', int(bud))])	

                    else:
                        conf_dict = dict([
                            ('vm_flavor', c["vm_flavor"]),
                            ('batch_size', int(c["batch_size"])),
                            ('learning_rate', c["learning_rate"]),
                            ('num_cores', int(c["num_cores"])),
                            ('synchronism', c["synchronism"]),
                            ('budget', int(bud))])	

                    for c_i, c_acc, c_cost, c_time in self.SearchSpace:
                        #print(c_i)
                        #print(conf_dict)
                        #print()
                        if conf_dict == c_i:
                            #cost_b = c_time
                            loss_b = 1 - c_acc
                            cost_b = c_cost
                            break
                   
                    #print("SNAPSHOOTNG conf_vec_b   "  +str(conf_vec_b_aux)  + " loss " +str(loss_b) + " cost " + str(cost_b))
                    self.training_set = np.vstack([self.training_set, conf_vec_b_aux])
                    self.losses = np.append(self.losses, loss_b)
                    self.costs = np.append(self.costs, cost_b)

                else: 
                    break
  
        self.configs2Test = copy.deepcopy(self.training_set)

        #print("size " + str(len(self.training_set)))

        # New result!!!!

        # Estimate incumbent
        if self.training_set.shape == (len(self.upper),):
            if self.training_set[-1] == self.max_budget:
                self.incumbent_value, self.incumbent, self.incumbent_line = (self.losses[0], self.training_set[0], 0)
            return
        elif np.any(self.training_set[:, len(self.upper) - 1] == self.max_budget):
            self.incumbent_value, self.incumbent, self.incumbent_line = get_incumbent(self.training_set, self.losses,self.max_budget)

    def training(self):
        # to train the models
        if not self.model_available and len(self.training_set) > self.min_points_in_model:

            if self.firstModel and len(self.training_set) > self.magicNumber2changeModel:
                self.firstModel = False
                self.model = EnsembleDTs(50, self.rng)

                if "EI$" in self.algorithm_variant or "Hybrid" in self.algorithm_variant:
                    self.has_cost_model = True
                    self.cost_model = EnsembleDTs(50, self.rng)

                else:
                    self.has_cost_model = False
                    self.cost_model = None


            initTrainTime = time.time()
            print("training the model")
            self.train_models()
            self.model_available = True
            self.training_time = time.time() - initTrainTime
        #print(str_wt)
        #self.listTrainingtimes.write(str_wt)

    def returntrainingTime(self):
        return self.training_time

    def train_models(self):
        #try:
            self.logger.info("Train model...")
            t = time.time()
            self.model.train(self.training_set, self.losses, do_optimize=True)
            self.logger.info("Time to train the model: %f", (time.time() - t))
            self.logger.info("Train cost model...")
            self.model_available = True
            if self.has_cost_model:
                # print("Cost model is being trained(1)")
                t1 = time.time()
                self.cost_model.train(self.training_set, self.costs, do_optimize=True)
                self.logger.info("Time to train cost model: %f", (time.time() - t1))
        # else:
        #   print("Cost model was not trained(2)")
        #except Exception:
        #    self.logger.error("Model could not be trained!")
        #    raise
            self.acquisition_func.update(self.model)
        #self.acquisition_func.update(self.model)

    def transform_config(self, best_vector):

        self.logger.debug('best_vector: {}'.format(best_vector))
        sample = ConfigSpace.Configuration(self.configspace, values=best_vector)
        return sample

    def check_active_hps(self, sample, budget):
        #try:
            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                configuration_space=self.configspace,
                configuration=sample.get_dictionary())
            # All combination of configurations that been tested on full budget should not be considered anymore
            # self.maximize_func.removeFromPool(conf_to_vector(sample, self.type_exp), currBudget=budget)
            return sample.get_dictionary()

        #except Exception as e:
        #    self.logger.warning("Error (%s) converting configuration: %s -> "
        #                        "using random configuration!",
        #                        e,
        #                        sample)
        #    sample = self.random_sample(budget, self.training_set)
            # All combination of configurations that been tested on full budget should not be considered anymore
        #    return sample.get_dictionary()

    def get_predictions(self, budget, number_predict):

        predictions = []

        target_budget = budget
        target_budget_cost = budget
        if self.algorithm_variant != 'CBS':
            target_budget = self.max_budget
            if self.algorithm_variant == 'FBS':
                target_budget_cost = self.max_budget

        for i in range(number_predict):
            best_hps, loss, sigma = self.maximize_func.maximize_all(self.training_set, target_budget, self.cost_model,
                                                                    target_budget_cost, budget)
            predictions.append((best_hps, loss, sigma))

        clean_prediction = []

        for best_hps, loss, sigma in predictions:
            best_vector = vector_to_conf(best_hps, self.type_exp)
            config = ConfigSpace.Configuration(self.configspace, values=best_vector).get_dictionary()
            print("_" * 20)
            print(config)
            print("Loss: ", loss, "Sigma: ", sigma)
            print("_" * 20)
            clean_prediction.append([config, loss, sigma])

        return clean_prediction

    def make_predictions(self, prev_configs, budget):
        predictions = []

        for config in prev_configs:
            #t_ = time.time()
            loss, sigma = self.maximize_func.get_prediction_sample(
                np.append(conf_to_vector(config, self.type_exp), budget)
            )
            #print("queryin time = " +str(time.time()- t_))
            predictions.append([config, loss, sigma])
            #print("_" * 20)
            #print(config)
            #print("Loss: ", loss, "Sigma: ", sigma)
            #print("_" * 20)

        return predictions



    def make_predictions_Cost(self, prev_configs, budget):
        if self.cost_model is None:
            return []

        predictions = []
        for config in prev_configs:
            c = np.append(conf_to_vector(config, self.type_exp), budget) 
            cost, sigma = self.cost_model.predict(c.reshape(1, len(c)))                  
            if cost < 0:
                cost = 10^-1
                sigma = 1
            predictions.append([config, cost, sigma])

        return predictions

    def Flag_has_cost_model(self):
        return self.has_cost_model

    def returnModel(self):
        return self.model

    def returnLower(self):
        return self.lower 

    def returnUpper(self):
        return self.upper

    def returnCostModel(self):
        return self.cost_model

    def returnExpType(self):
        return self.type_exp

    def returnTrainingSet(self):
        return self.training_set
