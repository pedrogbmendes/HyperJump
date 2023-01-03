import logging
import time

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import george
import sys
import copy, random
from copy import deepcopy

from hyperjump.core.base_config_generator import base_config_generator
#from hyperjump.optimizers.priors.default_priors import DefaultPrior
from hyperjump.optimizers.priors.env_priors import EnvPrior
#from hyperjump.optimizers.models.gaussian_process import GaussianProcess
from hyperjump.optimizers.models.fabolas_gp import FabolasGP
from hyperjump.optimizers.models.decision_trees import EnsembleDTs
from hyperjump.optimizers.acquisition_functions.ei import EI


#logging.basicConfig(level=logging.WARNING)


class Hyperjump(base_config_generator):
    # config generator for HyperJump
    def __init__(self, configspace, min_points_in_model=None,
                 num_samples=64, random_fraction=1/3,
                 min_budget=1, max_budget=16, **kwargs):
        """
        Parameters:
        -----------
        configspace: ConfigSpace
            Configuration space object
        min_points_in_model: int
            minimum number of datapoints needed to fit a model
        num_samples: int
            number of samples drawn to optimize EI via sampling
        random_fraction: float
            fraction of random configurations returned
        """
        super().__init__(**kwargs)
        self.configspace = configspace
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

        self.configs = []
        self.losses = []
        self.costs = []

        self.training_set = list() #np.array([])
        self.set2train = np.array([])
        self.configs2Test = list() #np.array([])

        lower = []   
        upper = []
        hp_name_list = self.configspace.get_hyperparameter_names()
        for hp_name in hp_name_list:
            vals = self.configspace.get_hyperparameter(hp_name).choices
            if any(isinstance(val, str) for val in vals): # not numeric
                size_list = int(len(vals))
                vals = list(range(size_list))

            upper.append(max(vals))
            lower.append(min(vals))

        # add budget to the space
        upper.append(max_budget)
        lower.append(min_budget)

        self.lower =  np.array(lower, dtype=np.float32)   
        self.upper =  np.array(upper, dtype=np.float32)

        #modelOption = "DT"
        modelOption = "GP"
        FlagCostModel = False

        if random_fraction == 1:
            #no models
            model =None
            cost_model = None

        elif modelOption == 'GP':
            cov_amp = 1  # Covariance amplitude
            kernel = cov_amp
            n_dims = len(self.configspace.get_hyperparameter_names()) + 1 # add one for the budget dimension

            # ARD Kernel for the configuration space
            for d in range(n_dims-1):
                kernel *= george.kernels.Matern52Kernel(np.ones([1])*0.01, ndim=n_dims, axes=d)

            # Kernel for the environmental variable
            # We use (1-s)**2 as basis function for the Bayesian linear kernel
            env_kernel = george.kernels.BayesianLinearRegressionKernel(log_a=0.1,log_b=0.1,ndim=n_dims,axes=n_dims-1)
            kernel *= env_kernel
        
            prior = EnvPrior(len(kernel) + 1, n_ls=n_dims-1, n_lr=2)
            quadratic_bf = lambda x: (1 - x) ** 2

            model = FabolasGP(kernel,
                                basis_function=quadratic_bf,
                                prior=prior,
                                normalize_output=False,
                                lower=self.lower,
                                upper=self.upper,)


            if FlagCostModel:
                self.has_cost_model = True
                    
                # Define model for the cost function
                cost_cov_amp = 1
                cost_kernel = cost_cov_amp

                # ARD Kernel for the configuration space
                for d in range(n_dims-1):
                    cost_kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01, ndim=n_dims , axes=d)

                cost_env_kernel = george.kernels.BayesianLinearRegressionKernel(log_a=0.1, log_b=0.1, ndim=n_dims,axes=n_dims-1)
                cost_kernel *= cost_env_kernel

                cost_prior = EnvPrior(len(cost_kernel) + 1, n_ls=n_dims-1, n_lr=2)
                linear_bf = lambda x: x

                cost_model = FabolasGP(cost_kernel,
                                basis_function=linear_bf,
                                prior=cost_prior,
                                normalize_output=False,
                                lower=self.lower,
                                upper=self.upper,)

            else:
                self.has_cost_model = False
                cost_model = None

        else:
            no_trees = 30
            model = EnsembleDTs(no_trees)
            
            if FlagCostModel:
                self.has_cost_model = True
                cost_model = EnsembleDTs(no_trees)

            else:
                self.has_cost_model = False
                cost_model = None
            
        self.acquisition_func = EI(model)


        self.min_budget = min_budget
        self.max_budget = max_budget
        self.model = model
        self.cost_model = cost_model


        self.inc = None
        self.inc_id = None
        self.incTime = -1
        self.incAcc = -1
        
        self.model_available = False


    def random_sample(self, budget, configs2Test):
        sample =  dict(self.configspace.sample_configuration())
        sample_with_budget = deepcopy(sample)
        sample_with_maxBudget = deepcopy(sample)
        sample_with_budget['budget'] = budget
        sample_with_maxBudget['budget'] = self.max_budget


        count_it = 0
        while sample_with_budget in configs2Test or sample_with_maxBudget in self.training_set:
            # 1. config aleardy to be tested
            # 2. config was tested on full budget - so no need to test it again
            sample =  dict(self.configspace.sample_configuration())
            sample_with_budget = deepcopy(sample)
            sample_with_maxBudget = deepcopy(sample)
            sample_with_budget['budget'] = budget
            sample_with_maxBudget['budget'] = self.max_budget
            count_it +=1
            if count_it > 500: break

        return sample_with_budget


    def get_listConfigs(self, budget, no_total_configs=1):  
        sample = None

        rand_or_Model = np.zeros(no_total_configs, dtype=int)
        for i in range(no_total_configs):
            if random.uniform(0, 1) < self.random_fraction  or len(self.training_set) <= self.min_points_in_model:
                #model is not available
                rand_or_Model[i] = 1

        listConfigs = []
        listData = []
        configs2Test = []
        for i in range(no_total_configs):
            startTime = time.time()

            if rand_or_Model[i] == 1: # pick random
                sample = self.random_sample(budget, self.configs2Test)

            else:
                if not listData: # train the models and compute the EI just the first time
                    self.training() # training models when we have the enough configs

                    listData = self.maximize_ei(no_total_configs=no_total_configs)
                    #listData is a list that contains no_total_configs lists of type [config, ei_value, loss, loss_sigma]  

                if len(listData) == 0: #go random
                    sample = self.random_sample(budget, configs2Test)

                else:
                    confi = listData.pop(0) #pop the first element in the list with the highest ei value
                    sample = confi[0]
                    sample_with_maxBudget = deepcopy(sample)
                    sample_with_maxBudget['budget'] = self.max_budget


                    while sample in configs2Test or sample_with_maxBudget in self.training_set:
                    # 1. config aleardy to be tested
                    # 2. config was tested on full budget - so no need to test it again
                        if len(listData) == 0: #go random
                            sample = self.random_sample(budget, configs2Test)

                        else:
                            confi = listData.pop(0) #pop the first element in the list with the highest ei value
                            sample = confi[0]
                            sample_with_maxBudget = deepcopy(sample)
                            sample_with_maxBudget['budget'] = self.max_budget


            info_dict = {}
            info_dict['overhead_time'] = (time.time() - startTime)
        
            configs2Test.append(sample)
            self.configs_tested_stage += 1

            listConfigs.append([sample, info_dict])

            self.logger.debug('done sampling a new configuration.')

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


        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        # If there is no model avalilable sample a config randomly. Also, even if there is a model available,
        # sample randomly if the random number falls in a certain range.
        aux_rand = random.uniform(0, 1)
        #print("rand " + str(aux_rand) + "  frac  " + str(self.random_fraction))
        if  aux_rand < self.random_fraction  or len(self.training_set) <= self.min_points_in_model:
            sample = self.random_sample(budget, [])


        if sample is None:
            self.training() # training models when we have the enough configs
            t = time.time()
            #EI
            listData = self.maximize_ei()
            sample, loss, sigma,loss_sigma = listData[0]

        self.logger.debug('done sampling a new configuration.')

        info_dict['overhead_time'] = (time.time() - start_time)
    
        self.configs_tested_stage += 1
        return sample, info_dict


    def maximize_ei(self, no_total_configs=1):
        """
        given a list of configuration, it returns the configs with highest expected improvement on full budget
        if a cost model exists, divide its EI with the predicted cost of the same config with budget {cost_budget}

        Returns a list of size no_total_configs 
        """

        # Get Incumbent
        if self.inc is not None:
            incumbent_loss = 1.0-self.incAcc 
        else:
            incumbent_loss = 1.0


        list_Confgis_ei = []
        listConfgis = []
        count_it = 0
        while len(list_Confgis_ei) < self.num_samples:
            # Filter the configurations that were already sampled
            sample =  dict(self.configspace.sample_configuration())
            sample['budget'] = self.max_budget # if tested on full budget, we don´t want to insert
            if sample not in self.training_set:
                list_Confgis_ei.append(sample) # not tested, so we can add config

            count_it += 1
            if count_it == 1000: break

        for config in list_Confgis_ei:
            config_array = self.config_to_vector(config)
            #eta is the incumbet loss
            ei_value = self.acquisition_func(config_array.reshape(1, len(config_array)), eta=incumbent_loss)[0]

            if self.cost_model is not None:
                ei_value /= (self.cost_model.predict(config_array.reshape(1, len(config_array)))[0])[0]

            loss, loss_sigma = self.acquisition_func.model.predict(config_array.reshape(1, len(config_array)))  # loss model
            listConfgis.append([config, ei_value, loss, loss_sigma])

        if len(listConfgis) == 0:
            raise BaseException("No configuration was selected in the maximization process.")

        def sortFirst(val):
            return val[1]
        
        listConfgis.sort(key=sortFirst, reverse=True)

        return listConfgis[0:no_total_configs]


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

        budget = job.kwargs["budget"]
        trainTime = float(job.result['info']['training_time'])
        config = job.kwargs["config"]

        self.training_set.append(config)
        #self.losses.append(loss)
        #self.costs.append(trainTime)


        vectorConfig = self.config_to_vector(config)
        if len(self.set2train) == 0:
            self.set2train = vectorConfig
            self.losses = np.array([loss])
            self.costs = np.array([trainTime])
        else:
            self.set2train = np.vstack([self.set2train, vectorConfig])
            self.losses = np.append(self.losses, loss)
            self.costs = np.append(self.costs, trainTime)

        if job.result['info']['checkpoint'] is not None:
            # there was intermidate configs tested taht we need to add to the training set
            checkpoint_data = job.result['info']['checkpoint']
            for dd in checkpoint_data:
                if dd[0] not in self.training_set:
                    self.training_set.append(dd[0])

                    self.set2train = np.vstack([self.set2train, self.config_to_vector(dd[0])])
                    self.losses = np.append(self.losses, loss)
                    self.costs = np.append(self.costs, trainTime)

                #self.losses.append(dd[1])
                #self.costs.append( dd[2])

        self.configs2Test = copy.deepcopy(self.training_set)
        confAcc = 1 - loss
        
        if budget == self.max_budget and self.incAcc < confAcc:
            #new incumbern
            self.inc = config 
            self.inc_id = job.id
            self.incTime = trainTime
            self.incAcc = confAcc


    def training(self):
        # to train the models
        if not self.model_available and len(self.training_set) > self.min_points_in_model:
            self.train_models()
            self.model_available = True


    def train_models(self):
        #self.logger.info("Train model...")
        t = time.time()

        self.model.train(self.set2train, self.losses, do_optimize=True)
        #self.logger.info("Time to train the model: %f", (time.time() - t))
    
        if self.has_cost_model:
            self.logger.info("Train cost model...")
            t1 = time.time()
            self.cost_model.train(self.set2train, self.costs, do_optimize=True)
            self.logger.info("Time to train cost model: %f", (time.time() - t1))

        self.model_available = True
        self.acquisition_func.update(self.model)


    def make_predictions(self, prev_configs, budget):
        predictions = []

        for config in prev_configs:
            config['budget'] = budget # if tested on full budget, we don´t want to insert
            cc = self.config_to_vector(config)

            loss, sigma = self.model.predict(cc.reshape(1, len(cc)))
            for i in range(0, len(loss)):                    
                if loss[i] < 0: loss[i] = 0
                elif loss[i] > 1: loss[i] = 0.99

            predictions.append([config, loss, sigma])

        return predictions


    def make_predictions_Cost(self, prev_configs, budget):
        if self.cost_model is None:
            return []

        predictions = []
        for config in prev_configs:
            config['budget'] = budget # if tested on full budget, we don´t want to insert
            cc = self.config_to_vector(config)
            
            cost, sigma = self.cost_model.predict(cc.reshape(1, len(cc)))                  
            if cost < 0:
                cost = 0.1
                sigma = 1
            predictions.append([config, cost, sigma])

        return predictions


    def config_to_vector(self, config):
        # convert a config dict into a numpy array to be used in the models
        _list_config = []
        for hp_name, val in config.items():
            if isinstance(val, str): # not numeric
                it = 0
                for hp_val in self.configspace.get_hyperparameter(hp_name).choices:
                    if hp_val==val:
                        _list_config.append(it)
                        break
                    it += 1
            else:
                _list_config.append(val)

        return np.array(_list_config)
