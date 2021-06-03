import logging
import sys
import time, copy

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm
import random

from hyperjump.core.base_config_generator import base_config_generator
from hyperjump.optimizers.utils.tools import get_incumbent
from hyperjump.optimizers.acquisition_functions.ei import EI
from hyperjump.optimizers.maximizers.random_sampling import RandomSampling
from hyperjump.optimizers.utils.tools import vector_to_conf
from hyperjump.optimizers.utils.tools import conf_to_vector
from hyperjump.optimizers.utils.tools import print_progress_bar
from hyperjump.optimizers.utils.tools import print_custom_bar

logging.basicConfig(level=logging.WARNING)


class BOHB_TPE(base_config_generator):
    def __init__(self, configspace, min_points_in_model=None,
                 top_n_percent=15, num_samples=64, random_fraction=1 / 3,
                 bandwidth_factor=3, min_bandwidth=1e-3, incumbent=[],
                 incumbent_value=-1, seed=1, type_exp='fake',
                 max_budget=16, min_budget=1, configspaceList=None, **kwargs):
        """
			Fits for each given budget a kernel density estimator on the best N percent of the
			evaluated configurations on this budget.


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
        logging.getLogger('requests').setLevel(logging.WARNING)

        np.seterr(divide='ignore', invalid='ignore')
        self.top_n_percent = top_n_percent
        self.configspace = configspace
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth

        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        random.seed(seed)


        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.configspace.get_hyperparameters()) + 1

        if self.min_points_in_model < len(self.configspace.get_hyperparameters()) + 1:
            self.logger.warning('Invalid min_points_in_model value. Setting it to %i' % (
                    len(self.configspace.get_hyperparameters()) + 1))
            self.min_points_in_model = len(self.configspace.get_hyperparameters()) + 1

        self.num_samples = num_samples
        self.random_fraction = random_fraction

        hps = self.configspace.get_hyperparameters()

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

        elif type_exp == 'fake_all' or type_exp == 'fake_time_all':
            # [batch_size, learning_rate, num_cores, synchrony, vm_flavor, network, budget]
            lower = np.array([16, 0.00001, 8, 0, 0, 0, min_budget])
            upper = np.array([256, 0.001, 80, 1, 3, 2, max_budget])

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
            # {kernel, degree, gamma, c}
            lower = np.array([1, 10e-6, 10e-6, min_budget])
            upper = np.array([3, 100, 100, max_budget])

            # Hyperparameter list without budgets
            hp_list = [[1, 2, 3],  # kernel
                       [0.001, 0.01, 0.1, 1, 10, 100],  # gamma
                       [0.001, 0.01, 0.1, 1, 10, 100]]  # c

        else:
            raise BaseException(("Invalid/unimplemented experiment %s", type_exp))


        self.kde_vartypes = ""
        self.vartypes = []

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

        if type_exp == 'fake' or type_exp == 'fake_time':
            # [batch_size, learning_rate, num_cores, synchronism, vm_flavor, budget]
            lower = np.array([16, 0.00001, 8, 0, 0, min_budget])
            upper = np.array([256, 0.001, 80, 1, 3, max_budget])
        # search_space_size = 1.440k #288*5
        elif type_exp == 'fake_all' or type_exp == 'fake_time_all':
            # [batch_size, learning_rate, num_cores, synchrony, vm_flavor, network, budget]
            lower = np.array([16, 0.00001, 8, 0, 0, 0, min_budget])
            upper = np.array([256, 0.001, 80, 1, 3, 2, max_budget]) 
            
        elif type_exp == 'unet':
            # {Flavor, batch, learningRate, momentum, nrWorker, sync}
            lower = np.array([1, 1, 0.000001, 0.9, 1, 1, min_budget])
            upper = np.array([2, 2, 0.0001, 0.99, 2, 2, max_budget])
        # search_space_size = 720 #144*5       
        elif type_exp == 'svm':
            # {kernel, degree, gamma, c}
            lower = np.array([1, 10e-6, 10e-6, min_budget])
            upper = np.array([3, 100, 100, max_budget])

        elif type_exp == 'mnist':
            lower = np.array([0.0, 0.0000001, 8, 4, 0, 0, 0.0, min_budget])
            upper = np.array([0.8, 0.01, 256, 64, 64, 64, 0.8, max_budget])
            # CS = 5*5*6*5*6*6*5 = 135K
            # With budgets -> 135K*3 = 405K
        else:
            raise BaseException(("Invalid/unimplemented experiment %s", type_exp))


        self.configs = dict()
        self.losses = dict()
        self.good_config_rankings = dict()
        self.kde_models = dict()
        self.all_losses = []
        self.training_set = []
        self.incumbent = -1
        self.incumbent_value = -1
        self.incumbent_line = -1
        self.type_exp = type_exp
        self.lower = lower
        self.upper = upper
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.listConfigSpace = configspaceList
        self.model_available = False

        # -------------------------------------------------------------------------
        self.total_time = time.time()
        self.total_results = 500
        self.config_num = 0
        print_progress_bar(0, self.total_results, prefix='\nProgress:', suffix="Complete  Finished in:h???:m??:s??",
                           length=50)

        self.flag_full_budget = True
        self.first = True
        self.training_time = 0.0

        acquisition_func = EI(None)
        self.maximize_func = RandomSampling(acquisition_func, lower, upper, hp_list, seed, rng=self.rng, min_budget=min_budget,
                                       full_budget=max_budget, uses_budget=True)
        

    def largest_budget_with_model(self):
        if len(self.kde_models) == 0:
            return (-float('inf'))
        return (max(self.kde_models.keys()))


    def _getRandom_unet(self, budget):
        if self.type_exp != "unet":
            return self.configspace.sample_configuration()
            
        if isinstance(self.training_set, list):
            training_set_ = self.training_set
        else:
            training_set_ = self.training_set.tolist()

        listConfigsBudget = []
        for c in self.listConfigSpace:
            if int(c[-1]) == int(budget):
                listConfigsBudget.append(c)

        rand_int_ = self.rng.randint(0, len(listConfigsBudget))
        config = listConfigsBudget[rand_int_]

        count_rand = 0
        while config in training_set_:
            rand_int_ = self.rng.randint(0, len(listConfigsBudget))
            config = listConfigsBudget[rand_int_]
            count_rand += 1
            if count_rand > 100:
                break

        rand_vector = vector_to_conf(config, self.type_exp)
        sample =  ConfigSpace.Configuration(self.configspace, values=rand_vector).get_dictionary()

        return sample

    def _getRandom_svm(self, budget):
        if self.type_exp != "svm":
            return self.configspace.sample_configuration()
            
        if isinstance(self.training_set, list):
            training_set_ = self.training_set
        else:
            training_set_ = self.training_set.tolist()

        listConfigsBudget = []
        for c in self.listConfigSpace:
            if int(c[-1]) == int(budget):
                listConfigsBudget.append(c)

        rand_int_ = self.rng.randint(0, len(listConfigsBudget))
        config = listConfigsBudget[rand_int_]

        count_rand = 0
        while config in training_set_:
            rand_int_ = self.rng.randint(0, len(listConfigsBudget))
            config = listConfigsBudget[rand_int_]
            count_rand += 1
            if count_rand > 100:
                break

        rand_vector = vector_to_conf(config, self.type_exp)
        sample =  ConfigSpace.Configuration(self.configspace, values=rand_vector).get_dictionary()

        return sample

    def returntrainingTime(self):
        return self.training_time

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
        self.logger.debug('start sampling a new configuration.')
        budget = int(budget)
        overhead_time = time.time()

        sample = None
        info_dict = {}
        info_dict['predicted_loss_mean'] = -1
        info_dict['predicted_loss_stdv'] = -1

        if self.min_points_in_model >= len(self.training_set):
            self.model_available = True
            
        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        #if len(self.kde_models.keys()) == 0 or self.rng.rand() < self.random_fraction:
        if len(self.kde_models.keys()) == 0 or random.uniform(0, 1) < self.random_fraction:
            if self.first:
                self.first = False
                if self.type_exp ==  "unet" or self.type_exp ==  "svm":
                    r_int = self.rng.randint(len(self.listConfigSpace))
                    random_hps = self.listConfigSpace[r_int]
                    random_hps[-1] = self.max_budget if self.flag_full_budget else budget
                    
                    if  isinstance(self.training_set, list):
                        training_set_ = self.training_set
                    else:
                        training_set_ = self.training_set.tolist()

                    count_rand = 0
                    while random_hps in training_set_:
                        r_int = self.rng.randint(len(self.listConfigSpace))
                        random_hps = self.listConfigSpace[r_int]
                        random_hps[-1] = self.max_budget if self.flag_full_budget else budget
                        count_rand += 1
                        if count_rand > 100:
                            break
                else:
                    if self.flag_full_budget:
                        random_hps = self.maximize_func.get_random_sample(self.max_budget, self.training_set, self.lower[:-1], self.upper[:-1])
                    else:
                        random_hps = self.maximize_func.get_random_sample(budget, self.training_set, self.lower[:-1], self.upper[:-1])
                

                rand_vector = vector_to_conf(random_hps, self.type_exp)
                sample =  ConfigSpace.Configuration(self.configspace, values=rand_vector)


                sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                        configuration_space=self.configspace,
                        configuration=sample.get_dictionary()
                        ).get_dictionary()


                info_dict['model_based_pick'] = False
                info_dict["incumbent_line"] = self.incumbent_line
                info_dict["incumbent_value"] = self.incumbent_value
                info_dict['overhead_time'] = (time.time() - overhead_time)
        
                return sample, info_dict


            else:
                if self.type_exp == "unet":
                    sample = self._getRandom_unet(budget)

                    info_dict["incumbent_line"] = self.incumbent_line
                    info_dict["incumbent_value"] = self.incumbent_value

                    info_dict['overhead_time'] = time.time() - overhead_time    
                    info_dict['model_based_pick'] = False

                    return sample, info_dict

                elif self.type_exp == "svm":
                    sample = self._getRandom_svm(budget)

                    info_dict["incumbent_line"] = self.incumbent_line
                    info_dict["incumbent_value"] = self.incumbent_value

                    info_dict['overhead_time'] = time.time() - overhead_time    
                    info_dict['model_based_pick'] = False

                    return sample, info_dict                   
                else:
                    sample = self.configspace.sample_configuration()

            info_dict['model_based_pick'] = False

        best = np.inf
        best_vector = None

        if sample is None:
            try:
                best_budget = max(self.kde_models.keys())

                l = self.kde_models[best_budget]['good'].pdf
                g = self.kde_models[best_budget]['bad'].pdf

                minimize_me = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)

                kde_good = self.kde_models[best_budget]['good']
                kde_bad = self.kde_models[best_budget]['bad']

                for i in range(self.num_samples):
                    idx = self.rng.randint(0, len(kde_good.data))
                    datum = kde_good.data[idx]
                    vector = []

                    for m, bw, t in zip(datum, kde_good.bw, self.vartypes):

                        bw = max(bw, self.min_bandwidth)
                        if t == 0:
                            bw = self.bw_factor * bw
                            try:
                                vector.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
                            except:
                                self.logger.warning(
                                    "Truncated Normal failed for:\ndatum=%s\nbandwidth=%s\nfor entry with value %s" % (
                                        datum, kde_good.bw, m))
                                self.logger.warning("data in the KDE:\n%s" % kde_good.data)
                                pass
                        else:

                            if self.rng.rand() < (1 - bw):
                                vector.append(int(m))
                            else:
                                vector.append(self.rng.randint(t))
                    val = minimize_me(vector)

                    if not np.isfinite(val):
                        self.logger.debug('sampled vector: %s has EI value %s' % (vector, val))
                        self.logger.debug("data in the KDEs:\n%s\n%s" % (kde_good.data, kde_bad.data))
                        self.logger.debug("bandwidth of the KDEs:\n%s\n%s" % (kde_good.bw, kde_bad.bw))
                        self.logger.debug("l(x) = %s" % (l(vector)))
                        self.logger.debug("g(x) = %s" % (g(vector)))

                        # right now, this happens because a KDE does not contain all values for a categorical
                        # parameter this cannot be fixed with the statsmodels KDE, so for now, we are just going to
                        # evaluate this one if the good_kde has a finite value, i.e. there is no config with that
                        # value in the bad kde, so it shouldn't be terrible.
                        if np.isfinite(l(vector)):
                            best_vector = vector
                            break

                    if val < best:
                        if self.type_exp == "unet" or self.type_exp == "svm":
                            vector_aux = copy.deepcopy(vector)
                            for n, _ in enumerate(vector_aux):
                                if isinstance(
                                        self.configspace.get_hyperparameter(self.configspace.get_hyperparameter_by_idx(n)),
                                        ConfigSpace.hyperparameters.CategoricalHyperparameter
                                ):
                                    vector_aux[n] = int(np.rint(vector_aux[n]))

                            aux_sample = ConfigSpace.Configuration(self.configspace, vector=vector_aux).get_dictionary()
                            config = list(conf_to_vector(aux_sample, self.type_exp))
                            config.append(budget)
                            #print(self.listConfigSpace)
                            #print(config)
                            if config in self.listConfigSpace:
                                best = val
                                best_vector = vector

                        else:
                            best = val
                            best_vector = vector


                if best_vector is None:
                    self.logger.debug(
                        "Sampling based optimization with %i samples failed -> using random configuration" % self.num_samples)
                    if self.type_exp == "unet":
                        sample = self._getRandom_unet(budget)

                    if self.type_exp == "svm":
                        sample = self._getRandom_svm(budget)   
                    else:
                        sample = self.configspace.sample_configuration().get_dictionary()
                    info_dict['model_based_pick'] = False

                else:
                    self.logger.debug(
                        'best_vector: {}, {}, {}, {}'.format(best_vector, best, l(best_vector), g(best_vector)))
                    for i, hp_value in enumerate(best_vector):
                        if isinstance(
                                self.configspace.get_hyperparameter(
                                    self.configspace.get_hyperparameter_by_idx(i)
                                ),
                                ConfigSpace.hyperparameters.CategoricalHyperparameter
                        ):
                            best_vector[i] = int(np.rint(best_vector[i]))
                    sample = ConfigSpace.Configuration(self.configspace, vector=best_vector).get_dictionary()
                    
                    if self.type_exp == "unet":
                        config = list(conf_to_vector(sample, self.type_exp))
                        config.append(budget)
                        if config not in self.listConfigSpace:
                            sample = self._getRandom_unet(budget)

                    elif self.type_exp == "svm":
                        config = list(conf_to_vector(sample, self.type_exp))
                        config.append(budget)
                        if config not in self.listConfigSpace:
                            sample = self._getRandom_svm(budget)

                    try:
                        sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                            configuration_space=self.configspace,
                            configuration=sample
                        )
                        info_dict['model_based_pick'] = True
                        info_dict['predicted_loss_mean'] = val
                    except Exception as e:
                        raise (e)

            except:
                
                # self.logger.warning("Sampling based optimization with %i samples failed\n %s \nUsing random configuration"%(self.num_samples, traceback.format_exc()))
                if self.type_exp == "unet":
                    sample = self._getRandom_unet(budget)
                elif self.type_exp == "svm":
                    sample = self._getRandom_svm(budget)
                else:
                    sample = self.configspace.sample_configuration().get_dictionary()                
                #sample = self.configspace.sample_configuration()
                info_dict['model_based_pick'] = False
                print("ERROR: get config")
                #sys.exit()

        try:
            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                configuration_space=self.configspace,
                configuration=sample.get_dictionary()
            ).get_dictionary()
        except Exception as e:
            sample = self.configspace.sample_configuration().get_dictionary()

            if self.type_exp == "unet":
                sample = self._getRandom_unet(budget)
            elif self.type_exp == "svm":
                sample = self._getRandom_svm(budget)


        info_dict["incumbent_line"] = self.incumbent_line
        info_dict["incumbent_value"] = self.incumbent_value
        info_dict['overhead_time'] = time.time() - overhead_time
        


        return sample, info_dict



    def get_listConfigs(self, budget, no_total_configs=1):
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
        self.logger.debug('start sampling a new configuration.')

        overhead_time = time.time()

        sample = None
        info_dict = {}
        info_dict['predicted_loss_mean'] = -1
        info_dict['predicted_loss_stdv'] = -1

        if self.min_points_in_model >= len(self.training_set):
            self.model_available = True
            
        listConfigs = []
        
        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        #if len(self.kde_models.keys()) == 0 or self.rng.rand() < self.random_fraction:
        if len(self.kde_models.keys()) == 0 or random.uniform(0, 1) < self.random_fraction:
           
            listSamples = []
            for i in range(no_total_configs):
                if self.type_exp == "unet":
                    sample = self._getRandom_unet(budget)
                elif self.type_exp == "svm":
                    sample = self._getRandom_svm(budget)
                else:
                    sample = self.configspace.sample_configuration()
                
                random_hps = conf_to_vector(sample, self.type_exp)
                random_hps[-1] = self.max_budget if self.flag_full_budget else budget

                while sample in listSamples or random_hps in self.training_set.tolist():
                    if self.type_exp == "unet":
                        sample = self._getRandom_unet(budget)
                    elif self.type_exp == "svm":
                        sample = self._getRandom_svm(budget)
                    else:
                        sample = self.configspace.sample_configuration()
                    random_hps = conf_to_vector(sample, self.type_exp)
                    random_hps[-1] = self.max_budget if self.flag_full_budget else budget


                listSamples.append(sample)

                info_dict["incumbent_line"] = self.incumbent_line
                info_dict["incumbent_value"] = self.incumbent_value
                info_dict['overhead_time'] = time.time() - overhead_time    
                info_dict['model_based_pick'] = False

                listConfigs.append([sample, info_dict])
        else:
            best = np.inf
            best_vector = None
            list_configs = []
            
            for ct in range(no_total_configs):

                try:
                    best_budget = max(self.kde_models.keys())

                    l = self.kde_models[best_budget]['good'].pdf
                    g = self.kde_models[best_budget]['bad'].pdf

                    minimize_me = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)

                    kde_good = self.kde_models[best_budget]['good']
                    kde_bad = self.kde_models[best_budget]['bad']

                    for i in range(self.num_samples):
                        idx = self.rng.randint(0, len(kde_good.data))
                        datum = kde_good.data[idx]
                        vector = []

                        for m, bw, t in zip(datum, kde_good.bw, self.vartypes):

                            bw = max(bw, self.min_bandwidth)
                            if t == 0:
                                bw = self.bw_factor * bw
                                try:
                                    vector.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
                                except:
                                    self.logger.warning(
                                        "Truncated Normal failed for:\ndatum=%s\nbandwidth=%s\nfor entry with value %s" % (
                                            datum, kde_good.bw, m))
                                    self.logger.warning("data in the KDE:\n%s" % kde_good.data)
                                    pass
                            else:

                                if self.rng.rand() < (1 - bw):
                                    vector.append(int(m))
                                else:
                                    vector.append(self.rng.randint(t))
                        val = minimize_me(vector)

                        if not np.isfinite(val):
                            self.logger.debug('sampled vector: %s has EI value %s' % (vector, val))
                            self.logger.debug("data in the KDEs:\n%s\n%s" % (kde_good.data, kde_bad.data))
                            self.logger.debug("bandwidth of the KDEs:\n%s\n%s" % (kde_good.bw, kde_bad.bw))
                            self.logger.debug("l(x) = %s" % (l(vector)))
                            self.logger.debug("g(x) = %s" % (g(vector)))

                            # right now, this happens because a KDE does not contain all values for a categorical
                            # parameter this cannot be fixed with the statsmodels KDE, so for now, we are just going to
                            # evaluate this one if the good_kde has a finite value, i.e. there is no config with that
                            # value in the bad kde, so it shouldn't be terrible.
                            if np.isfinite(l(vector)):
                                best_vector = vector
                                break
                        
                        if self.type_exp == "unet" or self.type_exp == "svm":
                            vector_aux = copy.deepcopy(vector)
                            for n, _ in enumerate(vector_aux):
                                if isinstance(
                                        self.configspace.get_hyperparameter(self.configspace.get_hyperparameter_by_idx(n)),
                                        ConfigSpace.hyperparameters.CategoricalHyperparameter):
                                    vector_aux[n] = int(np.rint(vector_aux[n]))

                            aux_sample = ConfigSpace.Configuration(self.configspace, vector=vector_aux).get_dictionary()
                            config = list(conf_to_vector(aux_sample, self.type_exp))
                            config.append(budget)

                            if config in self.listConfigSpace:
                                best = val
                                best_vector = vector

                        else:
                            best = val
                            best_vector = vector
                        
                        aux_sample = ConfigSpace.Configuration(self.configspace, vector=vector).get_dictionary()
                        config = list(conf_to_vector(aux_sample, self.type_exp))
                        config.append(self.max_budget) if self.flag_full_budget else config.append(budget)

                        if config not in self.training_set.tolist():
                            list_configs.append([vector, val])


                    if len(list_configs) == 0:
                        self.logger.debug("Sampling based optimization with %i samples failed -> using random configuration" % self.num_samples)
                        listSamples = []
                        listConfigs = []
                        for i in range(no_total_configs):
                            if self.type_exp == "unet":
                                sample = self._getRandom_unet(budget)
                            elif self.type_exp == "svm":
                                sample = self._getRandom_unet(budget)
                            else:
                                sample = self.configspace.sample_configuration()

                            random_hps = conf_to_vector(sample, self.type_exp)
                            random_hps[-1] = self.max_budget if self.flag_full_budget else budget

                            while sample in listSamples or random_hps in self.training_set.tolist():
                                if self.type_exp == "unet":
                                    sample = self._getRandom_unet(budget)
                                elif self.type_exp == "svm":
                                    sample = self._getRandom_svm(budget)
                                else:
                                    sample = self.configspace.sample_configuration()
                                random_hps = conf_to_vector(sample, self.type_exp)
                                random_hps[-1] = self.max_budget if self.flag_full_budget else budget

                            listSamples.append(sample)

                            info_dict["incumbent_line"] = self.incumbent_line
                            info_dict["incumbent_value"] = self.incumbent_value
                            info_dict['overhead_time'] = time.time() - overhead_time    
                            info_dict['model_based_pick'] = False

                            listConfigs.append([sample, info_dict])                    
                    
                    else:
                        self.logger.debug('best_vector: {}, {}, {}, {}'.format(best_vector, best, l(best_vector), g(best_vector)))
                        
                        for ct_ in range(len(list_configs)):
                            best_vector = list_configs[ct][0]
                            info_dict = list_configs[ct][1]

                            for i, hp_value in enumerate(best_vector):
                                if isinstance(self.configspace.get_hyperparameter( self.configspace.get_hyperparameter_by_idx(i)),ConfigSpace.hyperparameters.CategoricalHyperparameter):
                                    best_vector[i] = int(np.rint(best_vector[i]))
                            sample = ConfigSpace.Configuration(self.configspace, vector=best_vector).get_dictionary()

                            if self.type_exp == "unet":
                                config = list(conf_to_vector(sample, self.type_exp))
                                config.append(budget)
                                if config not in self.listConfigSpace:
                                    sample = self._getRandom_unet(budget)
                            elif self.type_exp == "svm":
                                config = list(conf_to_vector(sample, self.type_exp))
                                config.append(budget)
                                if config not in self.listConfigSpace:
                                    sample = self._getRandom_svm(budget)
                            try:
                                sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                                    configuration_space=self.configspace,
                                    configuration=sample
                                )
                                info_dict['model_based_pick'] = True
                                info_dict['predicted_loss_mean'] = val
                            except Exception as e:
                                raise (e)

                            list_configs[ct][0] = sample
                            list_configs[ct][1] = info_dict

                        def sortFirst(val):
                            return val[1]
                        list_configs.sort(key=sortFirst)

                except:
                    print("ERROR: get config")
                    listSamples = []
                    listConfigs = []
                    for i in range(no_total_configs):
                        if self.type_exp == "unet":
                            sample = self._getRandom_unet(budget)
                        elif self.type_exp == "svm":
                            sample = self._getRandom_svm(budget)
                        else:
                            sample = self.configspace.sample_configuration()
                        
                        random_hps = conf_to_vector(sample, self.type_exp)
                        random_hps[-1] = self.max_budget if self.flag_full_budget else budget

                        while sample in listSamples or random_hps in self.training_set.tolist():
                            if self.type_exp == "unet":
                                sample = self._getRandom_unet(budget)
                            elif self.type_exp == "svm":
                                sample = self._getRandom_svm(budget)
                            else:
                                sample = self.configspace.sample_configuration()
                            random_hps = conf_to_vector(sample, self.type_exp)
                            random_hps[-1] = self.max_budget if self.flag_full_budget else budget
                            
                        listSamples.append(sample)

                        info_dict["incumbent_line"] = self.incumbent_line
                        info_dict["incumbent_value"] = self.incumbent_value
                        info_dict['overhead_time'] = time.time() - overhead_time    
                        info_dict['model_based_pick'] = False

                        listConfigs.append([sample, info_dict]) 


        for ct_ in range(len(list_configs)):
            sample = list_configs[ct][0]
            info_dict = list_configs[ct][1]
            try:
                sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                    configuration_space=self.configspace,
                    configuration=sample.get_dictionary() ).get_dictionary()
            except Exception as e:
                sample = self.configspace.sample_configuration().get_dictionary()

            info_dict["incumbent_line"] = self.incumbent_line
            info_dict["incumbent_value"] = self.incumbent_value
            info_dict['overhead_time'] = time.time() - overhead_time
            
            list_configs[ct][0] = sample
            list_configs[ct][1] = info_dict
        
        return list_configs


    def impute_conditional_data(self, array):

        return_array = np.empty_like(array)

        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()

            while (np.any(nan_indices)):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = self.rng.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = self.rng.rand()
                    else:
                        datum[nan_idx] = self.rng.randint(t)

                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return (return_array)

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

        if job.result is None:
            # One could skip crashed results, but we decided to
            # assign a +inf loss and count them as bad configurations
            loss = np.inf
        else:
            # same for non numeric losses.
            # Note that this means losses of minus infinity will count as bad!
            loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf

        budget = job.kwargs["budget"]

        self.config_num += 1
        print_custom_bar((time.time() - self.total_time), self.config_num, self.total_results)

        if budget not in self.configs.keys():
            self.configs[budget] = []
            self.losses[budget] = []

        conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])
        conf_vec = conf_to_vector(conf, self.type_exp)
        conf_vec_b = np.append(conf_vec, budget)

        if len(self.training_set) == 0:
            self.training_set = conf_vec_b
            self.all_losses.append(loss)

        else:
            self.training_set = np.vstack([self.training_set, conf_vec_b])
            self.all_losses = np.append(self.all_losses, loss)

        # New result!!!!

        # Estimate incumbent
        if self.training_set.shape == (len(self.upper),):
            if self.training_set[-1] == self.max_budget:
                self.incumbent_value, self.incumbent, self.incumbent_line = (
                    self.all_losses[0], self.training_set[0], 0)
            return
        elif np.any(self.training_set[:,-1] == self.max_budget):
            self.incumbent_value, self.incumbent, self.incumbent_line = get_incumbent(self.training_set,
                                                                                      self.all_losses, self.max_budget)

        # skip model building if we already have a bigger model
        if max(list(self.kde_models.keys()) + [-np.inf]) > budget:
            return

        # We want to get a numerical representation of the configuration in the original space
        self.configs[budget].append(conf.get_array())
        self.losses[budget].append(loss)

        # skip model building:
        #		a) if not enough points are available
        if len(self.configs[budget]) <= self.min_points_in_model - 1:
            self.logger.debug("Only %i run(s) for budget %f available, need more than %s -> can't build model!" % (
                len(self.configs[budget]), budget, self.min_points_in_model + 1))
            return

        #		b) during warnm starting when we feed previous results in and only update once
        if not update_model:
            return

        train_configs = np.array(self.configs[budget])
        train_losses = np.array(self.losses[budget])

        n_good = max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0]) // 100)
        # n_bad = min(max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100), 10)
        n_bad = max(self.min_points_in_model, ((100 - self.top_n_percent) * train_configs.shape[0]) // 100)

        # Refit KDE for the current budget
        idx = np.argsort(train_losses)

        train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
        train_data_bad = self.impute_conditional_data(train_configs[idx[n_good:n_good + n_bad]])

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return

        # more expensive crossvalidation method
        # bw_estimation = 'cv_ls'

        # quick rule of thumb
        bw_estimation = 'normal_reference'

        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=self.kde_vartypes, bw=bw_estimation)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde_vartypes, bw=bw_estimation)

        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        self.kde_models[budget] = {
            'good': good_kde,
            'bad': bad_kde
        }

        # update probs for the categorical parameters for later sampling
        self.logger.debug(
            'done building a new model for budget %f based on %i/%i split\nBest loss for this budget:%f\n\n\n\n\n' % (
                budget, n_good, n_bad, np.min(train_losses)))
