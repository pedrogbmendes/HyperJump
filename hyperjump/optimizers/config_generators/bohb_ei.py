import sys
import time

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import george
import copy

from hyperjump.core.base_config_generator import base_config_generator
from hyperjump.optimizers.models.trimtuner_dt import EnsembleDTs
from hyperjump.optimizers.priors.default_priors import DefaultPrior
from hyperjump.optimizers.models.gaussian_process import GaussianProcess
from hyperjump.optimizers.acquisition_functions.ei import EI
from hyperjump.optimizers.maximizers.random_sampling import RandomSampling
from hyperjump.optimizers.initial_design import init_latin_hypercube_sampling
from hyperjump.optimizers.utils.tools import vector_to_conf as vector_to_conf
from hyperjump.optimizers.utils.tools import conf_to_vector as conf_to_vector
from hyperjump.optimizers.utils.tools import get_incumbent
from hyperjump.optimizers.utils.tools import print_progress_bar
from hyperjump.optimizers.utils.tools import print_custom_bar


class BOHB_EI(base_config_generator):
    def __init__(self, configspace, min_points_in_model=None,
                 top_n_percent=15, num_samples=64, random_fraction=1 / 3,
                 bandwidth_factor=3, min_bandwidth=1e-3, min_budget=1, max_budget=16, incumbent=[],
                 incumbent_value=-1, seed=1, type_exp='fake', algorithm_variant='FBS',  configspaceList=None, **kwargs):
        """
        Fits for each given budget a GPs on the best N percent of the
        evaluated configurations on this budget.


        Parameters:
        -----------
        configspace: ConfigSpace
            Configuration space object
        top_n_percent: int
            Determines the percentile of configurations that will be used as training data
            for the Gps, e.g if set to 10 the 10% best configurations will be considered
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

        self.training_set_dict = dict()
        self.losses_dict = dict()
        self.costs_set_dict = dict()

        self.configs = []
        self.losses = []
        self.training_set = np.array([])
        self.costs = []
        # ------------------------------------------------------------------------
        print(type_exp)
        if type_exp == 'fake' or type_exp == 'fake_time':
            # [batch_size, learning_rate, num_cores, synchrony, vm_flavor, budget]
            lower = np.array([16, 0.00001, 8, 0, 0])
            upper = np.array([256, 0.001, 80, 1, 3])

            # Hyperparameter list without budgets
            hp_list = [[16, 256],
                       [0.00001, 0.0001, 0.001],
                       [8, 16, 32, 48, 64, 80],
                       [0, 1],
                       [0, 1, 2, 3]]

        elif type_exp == 'fake_all' or type_exp == 'fake_time_all':
            # [batch_size, learning_rate, num_cores, synchrony, vm_flavor, network, budget]
            lower = np.array([16, 0.00001, 8, 0, 0, 0])
            upper = np.array([256, 0.001, 80, 1, 3, 2])

            # Hyperparameter list without budgets
            hp_list = [[16, 256],
                       [0.00001, 0.0001, 0.001],
                       [0, 1, 2], 
                       [8, 16, 32, 48, 64, 80],
                       [0, 1],
                       [0, 1, 2, 3]]

        # search_space_size = 1.440k #288*5
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
            lower = np.array([0.0, 0.0000001, 8, 4, 0, 0, 0.0])
            upper = np.array([0.8, 0.01, 256, 64, 64, 64, 0.8])
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
            lower = np.array([1, 1, 0.000001, 0.9, 1, 1])
            upper = np.array([2, 2, 0.0001, 0.99, 2, 2])

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
            lower = np.array([1, 10e-6, 10e-6])
            upper = np.array([3, 100, 100])

            # Hyperparameter list without budgets
            hp_list = [[1, 2, 3],  # kernel
                       [0.001, 0.01, 0.1, 1, 10, 100],  # gamma
                       [0.001, 0.01, 0.1, 1, 10, 100]]  # c

        else:
            raise BaseException(("Invalid/unimplemented experiment %s", type_exp))



        np.random.seed(seed)
        rng = np.random.RandomState(np.int64(seed))

        cov_amp = 2
        n_dims = lower.shape[0]

        initial_ls = np.ones([n_dims])
        exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                                   ndim=n_dims)
        kernel = cov_amp * exp_kernel

        prior = DefaultPrior(len(kernel) + 1)

        if 'DT' not in algorithm_variant:
            model = GaussianProcess(kernel, prior=prior, rng=rng,
                                    normalize_output=False, normalize_input=True,
                                    lower=lower, upper=upper)
        else:
            model = EnsembleDTs(30, rng)

        acquisition_func = EI(model)

        maximize_func = RandomSampling(acquisition_func, lower, upper, hp_list, seed, rng=rng, min_budget=min_budget,
                                       full_budget=max_budget, uses_budget=False)

        # ------------------------------------------------------------------------

        ########################################################################
        self.rng = rng
        self.seed = seed
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.algorithm_variant = algorithm_variant
        self.model = model
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
        self.training_time = 0
        ########################################################################

        # -------------------------------------------------------------------------
        self.total_time = time.time()
        self.total_results = 128
        self.config_num = 0
        self.sampled = False
        print_progress_bar(0, self.total_results, prefix='\nProgress:', suffix="Complete  Finished in:h???:m??:s??",
                           length=50)

        self.actualSet2Test = []
        self.listConfigSpace = configspaceList

    def random_sample_old(self, budget, training_set):
        random_hps = self.maximize_func.get_random_sample(budget, training_set, self.lower, self.upper)
        rand_vector = vector_to_conf(random_hps, self.type_exp)
        return ConfigSpace.Configuration(self.configspace, values=rand_vector)


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
            #if self.flag_full_budget:
            #   random_hps = self.maximize_func.get_random_sample(self.max_budget, training_set, self.lower[:-1], self.upper[:-1])
            #else:
            random_hps = self.maximize_func.get_random_sample(budget, training_set, self.lower, self.upper)
            rand_vector = vector_to_conf(random_hps, self.type_exp)

        #print(rand_vector)
        return ConfigSpace.Configuration(self.configspace, values=rand_vector)



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
        if self.rng.rand() < self.random_fraction:
            sample = self.random_sample(budget, self.training_set)
            info_dict['model_based_pick'] = False

        elif self.training_set.shape == (len(self.lower),) or len(self.training_set) <= self.min_points_in_model:
            sample = self.random_sample(budget, self.training_set)
            info_dict['model_based_pick'] = False

        if sample is None:
            self.train_models()
            if self.getBudgetOfLargestBuiltModel() != 0:
                self.model_available = True
                
            try:
                # Choose next point to evaluate
                t = time.time()
                target_budget = budget
                target_budget_cost = budget
                if self.algorithm_variant != 'CBS':
                    target_budget = self.max_budget
                    if self.algorithm_variant == 'FBS':
                        target_budget_cost = self.max_budget

                #if self.type_exp == 'fake':
                #    best_hps, loss, sigma = self.maximize_func.maximize_all(self.training_set, 0, None, 0, budget)
                #    best_hps = np.append(best_hps, budget)
                #else:
                #    best_hps, loss, sigma = self.maximize_func.maximize_ei_no_budget(budget, self.training_set)
                # print(best_hps)

                if self.type_exp == 'fake' or self.type_exp == 'fake_all' or self.type_exp == 'fake_time' or self.type_exp == 'fake_time_all':
                    #best_hps, loss, sigma = self.maximize_func.maximize_all(self.training_set, target_budget,
                    #                                                        self.cost_model, target_budget_cost, budget)
                    #EI
                    best_hps, loss, sigma = self.maximize_func.maximize_all(self.training_set, 0,None, 0, budget)
                
                    best_hps = np.append(best_hps, budget)

                elif self.type_exp == 'unet' or self.type_exp == 'svm':
                    best_hps, loss, sigma = self.maximize_func.maximize_list(self.training_set, 0,None, 0, budget, self.listConfigSpace)                                                                         
                
                    best_hps = np.append(best_hps, budget)

                else:
                    #best_hps, loss, sigma = self.maximize_func.maximize_ei(self.training_set, budget, target_budget,
                    #                                                       target_budget_cost,
                    #                                                       cost_model=self.cost_model)
                    best_hps, loss, sigma = self.maximize_func.maximize_ei_no_budget(budget, self.training_set)


                info_dict['predicted_loss_mean'], info_dict['predicted_loss_stdv'] = loss[0], sigma[0]

                best_vector = vector_to_conf(best_hps, self.type_exp)

                sample = self.transform_config(best_vector)
                info_dict['model_based_pick'] = True

            except BaseException:
                self.logger.warning("Sampling based optimization failed\n", sys.exc_info())
                sample = self.random_sample(budget, self.training_set)
                info_dict['model_based_pick'] = False

        sample = self.check_active_hps(sample, budget)

        self.logger.debug('done sampling a new configuration.')

        # Remove the config. from the models pool of choices
        # self.maximize_func.removeFromPool(conf_to_vector(sample, self.type_exp), budget)
        # self.sampled = True
        info_dict["incumbent_line"] = self.incumbent_line
        info_dict["incumbent_value"] = self.incumbent_value
        info_dict['overhead_time'] = (time.time() - start_time)
        self.actualSet2Test.clear()

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

        if self.type_exp == 'mnist':
            cost = float(job.result['info']['training_time'])
        else:
            cost = float(job.result['info']['cost'])

        self.config_num += 1
        print_custom_bar((time.time() - self.total_time), self.config_num, self.total_results)

        # We want to get a numerical representation of the configuration in the original space
        conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])
        conf_vec = conf_to_vector(conf, self.type_exp)

        # if not self.sampled:
        #	#Remove the config. from the models pool of choices
        #	self.maximize_func.removeFromPool(conf_vec, budget)
        # self.sampled = False

        conf_vec_b = np.append(conf_vec, budget)

        # ---------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------
        if budget not in self.training_set_dict.keys():
            self.training_set_dict[budget] = conf_vec
            self.losses_dict[budget] = np.array([loss])
            self.costs_set_dict[budget] = np.array([cost])
        else:
            self.training_set_dict[budget] = np.vstack([self.training_set_dict[budget], conf_vec])
            self.losses_dict[budget] = np.append(self.losses_dict[budget], loss)
            self.costs_set_dict[budget] = np.append(self.costs_set_dict[budget], cost)

        # ---------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------

        if len(self.training_set) == 0:
            self.training_set = conf_vec_b
            self.losses = np.array([loss])

        else:
            self.training_set = np.vstack([self.training_set, conf_vec_b])
            self.losses = np.append(self.losses, loss)

        # New result!!!!

        # Estimate incumbent
        if self.training_set.shape == (len(self.upper) + 1,):
            if self.training_set[-1] == self.max_budget:
                self.incumbent_value, self.incumbent, self.incumbent_line = (self.losses[0], self.training_set[0], 0)
            return
        elif np.any(self.training_set[:, len(self.upper)] == self.max_budget):
            self.incumbent_value, self.incumbent, self.incumbent_line = get_incumbent(self.training_set, self.losses,
                                                                                      self.max_budget)

    def returntrainingTime(self):
        return self.training_time

    def train_models(self):
        time_i = time.time()
        try:
            t = time.time()
            best_budget = self.getBudgetOfLargestBuiltModel()
            self.model.train(self.training_set_dict[best_budget], self.losses_dict[best_budget], do_optimize=True)
            self.logger.info("Time to train the model: %f", (time.time() - t))
            self.logger.info("Train cost model...")
        except:
            self.logger.error("Model could not be trained!")
            raise
        self.acquisition_func.update(self.model)
        self.training_time  = time.time() - time_i

    def transform_config(self, best_vector):

        self.logger.debug('best_vector: {}'.format(best_vector))
        sample = ConfigSpace.Configuration(self.configspace, values=best_vector)
        return sample

    def check_active_hps(self, sample, budget):
        try:
            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                configuration_space=self.configspace,
                configuration=sample.get_dictionary())
            # All combination of configurations that been tested on full budget should not be considered anymore
            # self.maximize_func.removeFromPool(conf_to_vector(sample, self.type_exp), currBudget=budget)
            return sample.get_dictionary()

        except Exception as e:
            self.logger.warning("Error (%s) converting configuration: %s -> "
                                "using random configuration!",
                                e,
                                sample)
            sample = self.random_sample(budget, self.training_set)
            # All combination of configurations that been tested on full budget should not be considered anymore
            return sample.get_dictionary()

    def get_predictions(self, budget, number_predict):
        predictions = self.maximize_func.maximize_Tensorflow_EI(
            get_predictions=True,
            currBudget=budget,
            cost_model=None,
            algorithm_variant=self.algorithm_variant,
            get_all=True,
            number_returned=number_predict)

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
            loss, sigma = self.maximize_func.get_prediction_sample(
                np.append(conf_to_vector(config, self.type_exp), budget)
            )
            predictions.append([config, loss, sigma])
            print("_" * 20)
            print(config)
            print("Loss: ", loss, "Sigma: ", sigma)
            print("_" * 20)

        return predictions

    def getBudgetOfLargestBuiltModel(self):
        largestBudget = 0

        for i in range(len(self.training_set_dict.keys())):
            budget = list(self.training_set_dict.keys())[i]

            if len(self.training_set_dict[budget]) > self.min_points_in_model and budget > largestBudget:
                largestBudget = budget

        return largestBudget
