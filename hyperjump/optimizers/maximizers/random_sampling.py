import math
import sys
import time

import numpy as np
import copy

from hyperjump.optimizers.maximizers.base_maximizer import BaseMaximizer
from hyperjump.optimizers.initial_design import init_random_uniform
from hyperjump.optimizers.utils.tools import get_incumbent


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class RandomSampling(BaseMaximizer):

    def __init__(self, acq_function, lower, upper, hyperparameter_list, seed=0, n_samples=8000, rng=None, min_budget=3750,
                 full_budget=60000, uses_budget=True):
        """
        Samples candidates uniformly at random and returns the point with the highest objective value.
        Parameters
        ----------
        acq_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        n_samples: int
            Number of candidates that are samples
        """
        self.acq_function = acq_function
        self.seed = seed
        self.n_samples = n_samples
        self.resample_hits = 0
        self.full_budget = full_budget
        self.min_budget = min_budget
        self.upper = upper
        self.lower = lower
        self.hp_list = hyperparameter_list
        self.rng = rng
        np.random.seed(seed)
        super(RandomSampling, self).__init__(acq_function, lower, upper, rng)

        # self.v_configs = self.get_vconfigs(exp_type, upper)
        self.samples = 0
        self.uses_budget = uses_budget

        # np.set_printoptions(threshold=np.inf)

    def get_random_sample(self, curr_budget, training_set, lower, upper):

        rand = init_random_uniform(lower, upper, 1, rng=self.rng)

        random_indexes = self.get_nearest_valid_hp_array(rand.tolist()[0])
        random_indexes.append(curr_budget)

        # If training set is list then = []; if not and has only one row, compare with ==
        if isinstance(training_set, list) or (
                training_set.shape == (len(upper) + 1,) and training_set.tolist() != random_indexes):
            return random_indexes

        if random_indexes not in training_set.tolist():
            return random_indexes
        return self.get_random_sample(curr_budget, training_set, lower, upper)


    #maximize a set of configs choose at random and next to the incumbent
    # run in case of NMIST
    #return only an untested config with the best EI
    def maximize_ei(self, sampled_configs, currBudget, target_budget, target_budget_cost, cost_model=None):

        """
        if cost_model is None:
            print("Costmodel is none")
            return self.maximize_all(sampled_configs, target_budget, incumbent[1], None, target_budget_cost, currBudget)

        return
        self.maximize_all(sampled_configs, target_budget, incumbent[1], cost_model, target_budget_cost, currBudget)
        """

        # Try to get the incumbent value with the max budget
        incumbent = self.get_incumbent_by_budget(self.full_budget)

        # If inc_loss is -1 it means that we havent sampled in full budget yet
        if incumbent[1] == -1:
            # print("Error getting incumbent with", self.full_budget, "; Trying with ", currBudget)
            incumbent = self.get_incumbent_by_budget(currBudget)

        return self.maximize(currBudget, incumbent, sampled_configs, cost_model, target_budget, target_budget_cost)


    #maximize a set of configs choose at random and next to the incumbent
    # run in case of NMIST
    #return a list of untested configs with the best EI
    def maximize_ei_returnBests(self, sampled_configs, currBudget, target_budget, target_budget_cost, no_total_configs, cost_model=None):

        # Try to get the incumbent value with the max budget
        incumbent = self.get_incumbent_by_budget(self.full_budget)

        # If inc_loss is -1 it means that we havent sampled in full budget yet
        if incumbent[1] == -1:
            # print("Error getting incumbent with", self.full_budget, "; Trying with ", currBudget)
            incumbent = self.get_incumbent_by_budget(currBudget)

        return self.maximize_returnBests(currBudget, incumbent, sampled_configs, cost_model, target_budget, target_budget_cost, no_total_configs)

    def maximize_ei_no_budget(self,
                              budget,
                              sampled_configs):

        incumbent = self.acq_function.model.get_incumbent()

        return self.maximize_no_budget(budget, incumbent, sampled_configs)

    #OLD
    def maximize_all_deprecated(self, sampled_configs, target_budget, incumbent_loss, cost_model, cost_budget
                                , current_budget):
        """
        Returns the configuration that maximizes EI devided by the cost prediction across the available 
        configuration space for a specific budget for both base model prediction and another specific budget
        for the cost model. 
        While searching, the algorith also ignores already sampled points.

        Parameters
        ----------
        sampled_configs: np.ndarray (N, D)
            Vectors to ignore
        target_budget: int
            Budget used to predict on base model
        incumbent_loss: np.ndarray (D,)
            Current incumbent loss
        cost_model: BaseModel (can algo be a GaussianProcess)
            Model to estimate configuration cost
        cost_budget: int
            Budget used to predict on cost_model
        current_budget: int
            Budget that is being used to sample a configuration. 
            Used to prune aready sampled configs


        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,) or np.array(N, N) if full_cov == True
            predictive variance

        TODO: To make the algorithm more adaptable to variable sized nested loop, recursion would 
                be a solution - now you need to manually change the sampling method for each 
                diferent Configuration Space
              Handle and throw exceptions when something wrong happens (i.e. best config remains [])
        """
        current_max = -1
        best_config = []
        s_confis_list = sampled_configs.tolist()
        for dr in self.hp_list[0]:
            for lr in self.hp_list[1]:
                for n_fc in self.hp_list[2]:
                    for n_f1 in self.hp_list[3]:
                        for n_f2 in self.hp_list[4]:
                            for n_f3 in self.hp_list[5]:
                                for sgd_mm in self.hp_list[6]:

                                    config = np.array([dr, lr, n_fc, n_f1, n_f2, n_f3, sgd_mm, target_budget])
                                    config_cost = np.array([dr, lr, n_fc, n_f1, n_f2, n_f3, sgd_mm, cost_budget])
                                    if [dr, lr, n_fc, n_f1, n_f2, n_f3, sgd_mm, current_budget] in s_confis_list:
                                        continue
                                    ei_value = self.acq_function(config.reshape(1, len(config)), eta=incumbent_loss)[0]

                                    if cost_model is not None:
                                        ei_value /= (cost_model.predict(config_cost.reshape(1, len(config_cost)))[0])[0]

                                    if ei_value > current_max:
                                        current_max = ei_value
                                        best_config = config

        loss, loss_sigma = self.acq_function.model.predict(best_config.reshape(1, len(best_config)))  # loss model
        return best_config, loss, loss_sigma


    #maximize all configs (entire search space)
    # run in case of CNN, RNN, MLP
    #return only an untested config with the best EI
    def maximize_all(self, sampled_configs, target_budget, cost_model, cost_budget, current_budget):
        f"""
        For every possible configuration in the search space with budget {current_budget} that is not in
        {sampled_configs}, calculate the EI with budget {target_budget} and then, if a cost model {cost_model}
        exists, divide its EI with the predicted cost of the same config with budget {cost_budget}

        Parameters
        ----------
        sampled_configs: np.ndarray (N, D)
            Configurations that were sampled
        target_budget: float
            Budget appended to the configuration to get its EI
        cost_model: GP model/Ensemble of DTs
            Model used to predict the cost of a configuration
        cost_budget: float
            Budget appended to the configuration to get its predicted cost value
        current_budget: float
            Budget used to sample a configuration in the current bracket

        Returns
        ----------
        best_config: np.ndarray (N, )
            Configuration with the maximum EI value
        loss: float
            predictive mean
        loss_sigma: float
            predictive variance
            
        """
        # Get Incumbent
        if self.uses_budget:
            # Try to get the incumbent value with the max budget
            incumbent = self.get_incumbent_by_budget(self.full_budget)

            # If inc_loss is -1 it means that we havent sampled in full budget yet
            if incumbent[1] == -1:
                incumbent = self.get_incumbent_by_budget(current_budget)
        else:
            incumbent = self.acq_function.model.get_incumbent()

        current_max = -1
        best_config = []
        s_configs_list = sampled_configs.tolist()
        #print(s_configs_list)

        # Total Search Space Available
        total_sp = 1
        for i in range(0, len(self.hp_list)):
            total_sp *= len(self.hp_list[i])

        for i in range(1, int(total_sp) + 1):
            config = self.__get_config_rec(i, 0, [])

            # Filter the configurations that were already sampled
            if self.uses_budget:
                config.append(current_budget)
                aux_config = copy.deepcopy(config)

                # If the optimizer contains budget in hp_list
                config[-1] = target_budget
                config_array = np.array(config)
                config[-1] = cost_budget
                config_array_cost = np.array(config)

            else:
                aux_config = copy.deepcopy(config)

                config_array = np.array(config)
                config_array_cost = np.array(config)

            if aux_config in s_configs_list:
                #tested config
                #print(aux_config)
                continue
            
            ei_value = self.acq_function(config_array.reshape(1, len(config_array)), eta=incumbent[1])[0]

            if cost_model is not None:
                ei_value /= (cost_model.predict(config_array_cost.reshape(1, len(config_array_cost)))[0])[0]

            if ei_value > current_max:
                current_max = ei_value
                best_config = np.array(aux_config)

        # best_config == []
        if best_config is list:
            raise BaseException("No configuration was selected in the maximization process.")

        loss, loss_sigma = self.acq_function.model.predict(best_config.reshape(1, len(best_config)))  # loss model

        return best_config, loss, loss_sigma


    #maximize all configs (entire search space)
    # run in case of CNN, RNN, MLP
    #return a list of untested configs with the best EI
    def maximize_all_returnBests(self, sampled_configs, target_budget, cost_model, cost_budget, current_budget, no_total_configs):
        f"""
        For every possible configuration in the search space with budget {current_budget} that is not in
        {sampled_configs}, calculate the EI with budget {target_budget} and then, if a cost model {cost_model}
        exists, divide its EI with the predicted cost of the same config with budget {cost_budget}
            
        """
        # Get Incumbent
        if self.uses_budget:
            # Try to get the incumbent value with the max budget
            incumbent = self.get_incumbent_by_budget(self.full_budget)

            # If inc_loss is -1 it means that we havent sampled in full budget yet
            if incumbent[1] == -1:
                incumbent = self.get_incumbent_by_budget(current_budget)
        else:
            incumbent = self.acq_function.model.get_incumbent()


        s_configs_list = sampled_configs.tolist()
        #print(s_configs_list)

        # Total Search Space Available
        total_sp = 1
        for i in range(0, len(self.hp_list)):
            total_sp *= len(self.hp_list[i])


        listConfgis = []
        for i in range(1, int(total_sp) + 1):
            config = self.__get_config_rec(i, 0, [])

            # Filter the configurations that were already sampled
            if self.uses_budget:
                config.append(current_budget)
                aux_config = copy.deepcopy(config)

                # If the optimizer contains budget in hp_list
                config[-1] = target_budget
                config_array = np.array(config)
                config[-1] = cost_budget
                config_array_cost = np.array(config)

            else:
                aux_config = copy.deepcopy(config)

                config_array = np.array(config)
                config_array_cost = np.array(config)

            if aux_config in s_configs_list:
                #tested config
                #print(aux_config)
                continue
            
            ei_value = self.acq_function(config_array.reshape(1, len(config_array)), eta=incumbent[1])[0]

            if cost_model is not None:
                ei_value /= (cost_model.predict(config_array_cost.reshape(1, len(config_array_cost)))[0])[0]

            _config = np.array(aux_config)
            loss, loss_sigma = self.acq_function.model.predict(_config.reshape(1, len(_config)))  # loss model
            listConfgis.append([np.array(_config), ei_value, loss, loss_sigma])

        # best_config == []
        if len(listConfgis) == 0:
            raise BaseException("No configuration was selected in the maximization process.")

        def sortFirst(val):
            return val[1]
        
        listConfgis.sort(key=sortFirst, reverse=True)
        
        return listConfgis[0:no_total_configs]


    #maximize only a list of possible configs 
    # run in case of UNET
    #return a list of untested configs with the best EI   
    def maximize_list_returnBests(self, sampled_configs, target_budget, cost_model, cost_budget, current_budget, listConfigs, no_total_configs):
        f"""
        For every possible configuration in the search space with budget {current_budget} that is not in
        {sampled_configs}, calculate the EI with budget {target_budget} and then, if a cost model {cost_model}
        exists, divide its EI with the predicted cost of the same config with budget {cost_budget}            
        """
        #Selects only configs that are present in a list and not the entire search space
        #return the set of the configs with the higher EI
        #usefull for UNET

        # Get Incumbent
        if self.uses_budget:
            # Try to get the incumbent value with the max budget
            incumbent = self.get_incumbent_by_budget(self.full_budget)

            # If inc_loss is -1 it means that we havent sampled in full budget yet
            if incumbent[1] == -1:
                incumbent = self.get_incumbent_by_budget(current_budget)
        else:
            incumbent = self.acq_function.model.get_incumbent()

        current_max = -1
        best_config = []
        s_configs_list = sampled_configs.tolist()
        #print(s_configs_list)

        listConfigsBudget = []
        for c in listConfigs:
            if int(c[-1]) == int(current_budget):
                listConfigsBudget.append(c)
                
        #print(listConfigsBudget)
        listConfgis = []
        for conf in listConfigsBudget:
            config = copy.deepcopy(conf[:-1]) # drop budget

            # Filter the configurations that were already sampled
            if self.uses_budget:
                config.append(current_budget)
                aux_config = copy.deepcopy(config)

                # If the optimizer contains budget in hp_list
                config[-1] = target_budget
                config_array = np.array(config)
                config[-1] = cost_budget
                config_array_cost = np.array(config)

            else:
                aux_config = copy.deepcopy(config)

                config_array = np.array(config)
                config_array_cost = np.array(config)

            if aux_config in s_configs_list:
                #tested config
                continue

            ei_value = self.acq_function(config_array.reshape(1, len(config_array)), eta=incumbent[1])[0]

            if cost_model is not None:
                ei_value /= (cost_model.predict(config_array_cost.reshape(1, len(config_array_cost)))[0])[0]

            _config = np.array(aux_config)
            loss, loss_sigma = self.acq_function.model.predict(_config.reshape(1, len(_config)))  # loss model
            listConfgis.append([np.array(_config), ei_value, loss, loss_sigma])

        # best_config == []
        if len(listConfgis) == 0:
            raise BaseException("No configuration was selected in the maximization process.")

        def sortFirst(val):
            return val[1]
        
        listConfgis.sort(key=sortFirst, reverse=True)
        
        return listConfgis[0:no_total_configs]


    #maximize only a list of possible configs 
    # run in case of UNET
    #return only a untested configs with the best EI   
    def maximize_list(self, sampled_configs, target_budget, cost_model, cost_budget, current_budget, listConfigs):
        f"""
        For every possible configuration in the search space with budget {current_budget} that is not in
        {sampled_configs}, calculate the EI with budget {target_budget} and then, if a cost model {cost_model}
        exists, divide its EI with the predicted cost of the same config with budget {cost_budget}

        Parameters
        ----------
        sampled_configs: np.ndarray (N, D)
            Configurations that were sampled
        target_budget: float
            Budget appended to the configuration to get its EI
        cost_model: GP model/Ensemble of DTs
            Model used to predict the cost of a configuration
        cost_budget: float
            Budget appended to the configuration to get its predicted cost value
        current_budget: float
            Budget used to sample a configuration in the current bracket

        Returns
        ----------
        best_config: np.ndarray (N, )
            Configuration with the maximum EI value
        loss: float
            predictive mean
        loss_sigma: float
            predictive variance
            
        """

        #Selects only configs that are present in a lis and not the entire search space
        #returns only one config
        #usefull for UNET

        # Get Incumbent
        if self.uses_budget:
            # Try to get the incumbent value with the max budget
            incumbent = self.get_incumbent_by_budget(self.full_budget)

            # If inc_loss is -1 it means that we havent sampled in full budget yet
            if incumbent[1] == -1:
                incumbent = self.get_incumbent_by_budget(current_budget)
        else:
            incumbent = self.acq_function.model.get_incumbent()

        current_max = -1
        best_config = []
        s_configs_list = sampled_configs.tolist()
        #print(s_configs_list)

        listConfigsBudget = []
        for c in listConfigs:
            if int(c[-1]) == int(current_budget):
                listConfigsBudget.append(c)
                
        #print(listConfigsBudget)
        for conf in listConfigsBudget:
            config = copy.deepcopy(conf[:-1]) # drop budget

            # Filter the configurations that were already sampled
            if self.uses_budget:
                config.append(current_budget)
                aux_config = copy.deepcopy(config)

                # If the optimizer contains budget in hp_list
                config[-1] = target_budget
                config_array = np.array(config)
                config[-1] = cost_budget
                config_array_cost = np.array(config)

            else:
                aux_config = copy.deepcopy(config)

                config_array = np.array(config)
                config_array_cost = np.array(config)

            if aux_config in s_configs_list:
                #tested config
                continue

            ei_value = self.acq_function(config_array.reshape(1, len(config_array)), eta=incumbent[1])[0]

            if cost_model is not None:
                ei_value /= (cost_model.predict(config_array_cost.reshape(1, len(config_array_cost)))[0])[0]

            if ei_value > current_max:
                current_max = ei_value
                best_config = np.array(aux_config)

        # best_config == []
        if best_config is list:
            raise BaseException("No configuration was selected in the maximization process.")

        loss, loss_sigma = self.acq_function.model.predict(best_config.reshape(1, len(best_config)))  # loss model

        return best_config, loss, loss_sigma


    def __get_config_rec(self, config_num, hp_index, config_to_return):
        """
        Method that returns the configuration based on the hp number. This number is itself based on the
        total number of possible configs in the Search Space

        Parameters
        ----------
        config_num: int
            Number of configuration
        hp_index: int
            Hyperparameter index
        config_to_return: list
            List that will be built recursively until hp_index reaches the length of the Hyperparameter list

        Returns
        ----------
        list
            configuration got from the hp number
        """
        hp_len = len(self.hp_list[hp_index])

        # Calculate the hp value
        remainder = config_num % hp_len
        true_div = config_num // hp_len
        hp_value = self.hp_list[hp_index][remainder]

        # Fill first config
        config_to_return.append(hp_value)

        if hp_index < len(self.hp_list) - 1:
            return self.__get_config_rec(true_div, hp_index + 1, config_to_return)

        return config_to_return

    def get_incumbent_by_budget(self, budget):
        dummy_arr = np.array(self.lower)
        dummy_arr[-1] = budget
        return self.acq_function.model.get_incumbent_with_budget(dummy_arr)

    def get_prediction_sample(self, config):

        loss, sigma = self.acq_function.model.predict(config.reshape(1, len(config)))
        #print("prediction l=" +str(loss) +"  s= " + str(sigma))
        for i in range(0, len(loss)):                    
            if loss[i] < 0:
                #print("config wrong loss = " + str(loss[i]))
                loss[i] = 0
                #time.sleep(20)
            elif loss[i] > 1:
                #print("config wrong loss = " + str(loss[i]))
                loss[i] = 0.99
                #time.sleep(20)
        return loss, sigma

        # def __get_random(self, lower, upper, no_configs):
        #     for i in range(0,no_configs):
        #         for j in range(0, np.shape(lower)[0]):
        #             np.random.uniform()
        #     return


    def maximize(self, budget, incumbent, training_set, cost_model, target_budget, target_budget_cost):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        # Sample random points uniformly over the whole space
        rand = init_random_uniform(self.lower[:-1], self.upper[:-1],
                                   int(self.n_samples * .7), rng=self.rng)

        # Put a Gaussian on the incumbent and sample from that
        loc = (incumbent[0])[:-1],
        scale = np.ones([self.lower.shape[0] - 1]) * 0.1

        rand_incs = np.array([np.clip(self.rng.normal(loc, scale), self.lower[:-1], self.upper[:-1])[0]
                              for _ in range(int(self.n_samples * 0.3))])

        X = np.concatenate((rand, rand_incs), axis=0)
        y = self.process_and_predict(X, budget, training_set, incumbent[1], cost_model, target_budget,
                                     target_budget_cost)

        x_star = X[y.argmax()]

        approximated_arr = np.array(self.get_nearest_valid_hp_array(x_star.tolist()))
        approximated_arr = np.append(approximated_arr, budget)

        loss, sigma = self.get_prediction_sample(approximated_arr)

        return approximated_arr, loss, sigma

    def maximize_returnBests(self, budget, incumbent, training_set, cost_model, target_budget, target_budget_cost, no_total_configs):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        # Sample random points uniformly over the whole space
        rand = init_random_uniform(self.lower[:-1], self.upper[:-1],
                                   int(self.n_samples * .7), rng=self.rng)

        # Put a Gaussian on the incumbent and sample from that
        loc = (incumbent[0])[:-1],
        scale = np.ones([self.lower.shape[0] - 1]) * 0.1

        rand_incs = np.array([np.clip(self.rng.normal(loc, scale), self.lower[:-1], self.upper[:-1])[0]
                              for _ in range(int(self.n_samples * 0.3))])

        X = np.concatenate((rand, rand_incs), axis=0)
        y = self.process_and_predict(X, budget, training_set, incumbent[1], cost_model, target_budget,
                                     target_budget_cost)

        listConfgis = []
        aux_listConfs = []
        for i in range(len(X)):
            approximated_arr = np.array(self.get_nearest_valid_hp_array(X[i].tolist()))
            approximated_arr = np.append(approximated_arr, budget)

            if approximated_arr not in aux_listConfs:
                aux_listConfs.append(approximated_arr)
                ei_ = y[i]
                loss, sigma = self.get_prediction_sample(approximated_arr)
                listConfgis.append([approximated_arr, ei_, loss, sigma])

        # best_config == []
        if len(listConfgis) == 0:
            raise BaseException("No configuration was selected in the maximization process.")

        def sortFirst(val):
            return val[1]
        
        listConfgis.sort(key=sortFirst, reverse=True)
        
        return listConfgis[0:no_total_configs]


    def process_and_predict(self, predicted, budget, training_set, incumbent_loss,
                            cost_model, target_budget, target_budget_cost):
        result_set = np.array([])

        # This list of configs that do not contain budgets
        # They also might have values of configs that were already sampled
        #       which is not allowed
        for i in range(len(predicted)):
            x = predicted[i]
            # add the budget for the current configurations so that it will get filtered
            nearest = self.get_nearest_valid_hp_array(x)
            nearest.append(budget)

            # Then check if the aproximated value is already in the sampled list
            if nearest in training_set.tolist():
                # if it is, say that the EI is awful
                result_set = np.append(result_set, -1000000)
                continue

            # If it isnt, put the target budget on the config so that it will predict accordingly
            x = np.append(x, target_budget)
            ei_value = self.acq_function(x.reshape(1, len(x)), eta=incumbent_loss)[0]

            if cost_model is not None:
                # the same for the cost
                x[-1] = target_budget_cost
                ei_value /= (cost_model.predict(x.reshape(1, len(x)))[0])[0]

            result_set = np.append(result_set, ei_value)

        return result_set

    def get_nearest_valid_hp_array(self, x):
        nearest = []
        for j, hp_value in enumerate(x):
            # Approximate the random value to an allowed hp value
            near = find_nearest(self.hp_list[j], hp_value)
            nearest.append(near)
        return nearest

    def maximize_no_budget(self, budget, incumbent, training_set):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        # Sample random points uniformly over the whole space
        rand = init_random_uniform(self.lower, self.upper,
                                   int(self.n_samples * .7), rng=self.rng)

        # Put a Gaussian on the incumbent and sample from that
        loc = (incumbent[0]),
        scale = np.ones([self.lower.shape[0]]) * 0.1

        rand_incs = np.array([np.clip(self.rng.normal(loc, scale), self.lower, self.upper)[0]
                              for _ in range(int(self.n_samples * 0.3))])

        X = np.concatenate((rand, rand_incs), axis=0)
        y = self.process_and_predict_no_budget(X, budget, training_set, incumbent[1])

        x_star = X[y.argmax()]

        approximated_arr = np.array(self.get_nearest_valid_hp_array(x_star.tolist()))

        loss, sigma = self.get_prediction_sample(approximated_arr)
        approximated_arr = np.append(approximated_arr, budget)

        return approximated_arr, loss, sigma

    def process_and_predict_no_budget(self, predicted, budget, training_set, incumbent_loss):
        result_set = np.array([])

        # This list of configs that do not contain budgets
        # They also might have values of configs that were already sampled
        #       which is not allowed
        for i in range(len(predicted)):
            x = predicted[i]
            # add the budget for the current configurations so that it will get filtered
            nearest = self.get_nearest_valid_hp_array(x)
            nearest.append(budget)

            # Then check if the aproximated value is already in the sampled list
            if nearest in training_set.tolist():
                # if it is, say that the EI is awful
                result_set = np.append(result_set, -1000000)
                continue

            ei_value = self.acq_function(x.reshape(1, len(x)), eta=incumbent_loss)[0]
            result_set = np.append(result_set, ei_value)

        return result_set
