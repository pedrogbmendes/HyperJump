import sys
import os, ctypes
import logging
import numpy as np
import pdb
import math
from scipy.stats import norm
from scipy.integrate import quad, nquad
from scipy import LowLevelCallable
import itertools
import multiprocessing
from mpmath import mp
import time
import copy
import warnings, random

from hyperjump.core.dispatcher import Job
#from hyperjump.optimizers.config_generators.bohb_tpe import bohb_tpe
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# from hyperjump.optimizers.acquisition_functions.information_gain import InformationGain
# from hyperjump.optimizers.utils.tools import conf_to_vector as conf_to_vector




class Datum(object):
    def __init__(self, config, config_info, config_id, results=None, time_stamps=None, exceptions=None, status='QUEUED', budget=0):
        self.config = config
        self.config_info = config_info
        self.results = results if not results is None else {}
        self.time_stamps = time_stamps if not time_stamps is None else {}
        self.exceptions = exceptions if not exceptions is None else {}
        self.status = status
        self.budget = budget
        self.config_id = config_id

    def __repr__(self):
        return ( \
                    "\nconfig:{}\n".format(self.config) + \
                    "config_info:\n{}\n" % self.config_info + \
                    "losses:\n"
                    '\t'.join(["{}: {}\t".format(k, v['loss']) for k, v in self.results.items()]) + \
                    "time stamps: {}".format(self.time_stamps)
        )


class BaseIteration(object):

    def __init__(self, HPB_iter, num_configs, budgets, config_sampler, logger=None, result_logger=None, hyperjump=False, threshold=1.0):
        """
        Parameters
        ----------

        HPB_iter: int
            The current hyperjump iteration index.
        num_configs: list of ints
            the number of configurations in each stage of SH
        budgets: list of floats
            the budget associated with each stage
        config_sample: callable
            a function that returns a valid configuration. Its only
            argument should be the budget that this config is first
            scheduled for. This might be used to pick configurations
            that perform best after this particular budget is exhausted
            to build a better autoML system.
        logger: a logger
        result_logger: hyperjump.api.results.util.json_result_logger object
            a result logger that writes live results to disk
        """

        self.data = {}  # this holds all the configs and results of this iteration
        self.is_finished = False
        self.HPB_iter = HPB_iter
        self.stage = 0  # internal iteration, but different name for clarity
        self.budgets = budgets
        self.num_configs = num_configs
        self.actual_num_configs = [0] * len(num_configs)
        self.config_sampler = config_sampler
        self.num_running = 0
        self.logger = logger if not logger is None else logging.getLogger('hyperjump')
        self.result_logger = result_logger

        self.prints = True
        self.hyperjump = hyperjump

        self.thresholdRisk = threshold

        self.fullBudget_predictions = False
        self.SEL_set = None
        self.UNSEL_set = None
        self.printDAG = False
        self.time_start = time.time()

        if self.hyperjump:
            self.set2Test()

        self.randFrac = 0.7
        np.random.seed(1000)
        random.seed(1000)
        if hyperjump:
            self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)  

        self.eta = 2
        print ("The threshold lambd is " + str(self.thresholdRisk)) # + " and ETA=" + str(self.eta))

  
    def set2Test(self):
        no_confs = self.num_configs[self.stage]

        self.add_set_configurations()

        if self.prints:
            print("[NEXT BRACKET] ---> Creating config set to test....Loading " + str(no_confs) + " configurations!  \n")

    def add_set_configurations(self, config=None, config_info={}):
        """
        function to add a new configuration to the current iteration

        Parameters
        ----------

        config : valid configuration
            The configuration to add. If None, a configuration is sampled from the config_sampler
        config_info: dict
            Some information about the configuration that will be stored in the results
        """
        listConfigs_add =  self.config_sampler.get_listConfigs(self.budgets[self.stage], no_total_configs=self.num_configs[self.stage]) 

        listIDs = []
        listConfigs = []
        for i in range(len(listConfigs_add)):
            config, config_info = listConfigs_add[i]

            if self.is_finished:
                raise RuntimeError("This hyperjump iteration is finished, you can't add more configurations!")

            if self.actual_num_configs[self.stage] == self.num_configs[self.stage]:
                raise RuntimeError(
                    "Can't add another configuration to stage %i in hyperjump iteration %i." % (self.stage, self.HPB_iter))

            config_id = (self.HPB_iter, self.stage, self.actual_num_configs[self.stage])
            listIDs.append(config_id)
            listConfigs.append(config)
            self.data[config_id] = Datum(config=config, config_info=config_info, config_id=config_id, budget=self.budgets[self.stage])

            #if self.config_sampler.verifyConfig(config, self.budgets[self.stage]): self.data[config_id].status = 'REVIEW'

            self.actual_num_configs[self.stage] += 1

            if not self.result_logger is None:
                #overhead = time.time() - self.time_start
                self.result_logger.new_config(config_id, config, config_info)

        self.result_logger.bracketInfo(listConfigs, self.budgets[self.stage], self.budgets, self.num_configs) 
         
        return config_id


    def updateEta(self, eta):
        self.eta = eta
        print("ETA was set to " +str(self.eta))

    def add_configuration(self, config=None, config_info={}):
        """
        function to add a new configuration to the current iteration

        Parameters
        ----------

        config : valid configuration
            The configuration to add. If None, a configuration is sampled from the config_sampler
        config_info: dict
            Some information about the configuration that will be stored in the results
        """

        if config is None:
            config, config_info = self.config_sampler.get_config(self.budgets[self.stage], no_total_configs=self.num_configs[self.stage])

        if self.is_finished:
            raise RuntimeError("This hyperjump iteration is finished, you can't add more configurations!")

        if self.actual_num_configs[self.stage] == self.num_configs[self.stage]:
            raise RuntimeError(
                "Can't add another configuration to stage %i in hyperjump iteration %i." % (self.stage, self.HPB_iter))

        config_id = (self.HPB_iter, self.stage, self.actual_num_configs[self.stage])

        self.data[config_id] = Datum(config=config, config_info=config_info, config_id=config_id, budget=self.budgets[self.stage])

        self.actual_num_configs[self.stage] += 1

        if not self.result_logger is None:
            #overhead = time.time() - self.time_start
            self.result_logger.new_config(config_id, config, config_info, 0)

        #print("sampling " + str(config))
        return (config_id)

    def register_result(self, job, skip_sanity_checks=False):
        """
        function to register the result of a job

        This function is called from HB_master, don't call this from
        your script.
        """

        if self.is_finished:
            raise RuntimeError("This HB iteration is finished, you can't register more results!")

        config_id = job.id
        config = job.kwargs['config']
        budget = job.kwargs['budget']
        timestamps = job.timestamps
        result = job.result
        exception = job.exception

        d = self.data[config_id]

        if not skip_sanity_checks:
            assert d.config == config, 'Configurations differ!'
            assert d.status == 'RUNNING', "Configuration wasn't scheduled for a run."
            assert d.budget == budget, 'Budgets differ (%f != %f)!' % (self.data[config_id]['budget'], budget)

        d.time_stamps[budget] = timestamps
        d.results[budget] = result

        if (not job.result is None) and np.isfinite(result['loss']):
            d.status = 'REVIEW'
        else:
            d.status = 'CRASHED'

        d.exceptions[budget] = exception
        self.num_running -= 1


    def get_next_run(self):
        """
        function to return the next configuration and budget to run.

        """

        if self.is_finished:
            # print("[ITERATION] Finished.")
            return None

        eal = 0
        self.update_initTime()
        init_time_a = time.time()
        if self.hyperjump:
            self.config_sampler.training()
            eal = self.analyse_risk_new()
            self.result_logger.update_EAL_new(eal)

        time_risk = time.time() - init_time_a


        init_time_a = time.time()
        key_, config_, budget_ = self.runNextConfig(eal)
        time_testorder = time.time() - init_time_a

        training_time = self.config_sampler.returntrainingTime()
        self.result_logger.updateTimeOverhead(time_risk, time_testorder, training_time)

        if key_ is not None: 
            if self.prints:     
                print("[ITERATION] - Running config " + str(self.data[key_].config) + " on budget " + str(self.data[key_].budget) + "!!\n") # k,v.config,v.budget
  
            self.updateOverhead(key_)
            return (key_, config_, budget_)
        
    
        if not self.hyperjump and self.actual_num_configs[self.stage] < self.num_configs[self.stage]:
            # check if there are still slots to fill in the first stage and return that
            if self.prints:
                print("[ITERATION] - Sampling configuration -> " +  str(self.actual_num_configs[self.stage]) + " < " + str(self.num_configs[self.stage]) + "\n")
            self.add_configuration()
            return self.get_next_run()

        if self.num_running == 0:
            #print("END bracket")
            self.process_results()
            return self.get_next_run()

        return None


    def runNextConfig(self, eal):
        """
            funtion to select the next configuration to test
            
            There are several heuristics provided
            HJ uses the "newRisk_parallel"

        """

        ############################################
        # sort configs to test
        #############################################
        #run = "random"
        #run = "sortLoss"
        #run = "sortLCB"
        #run = "sortUCB"
        #run = "newRisk"
        run = "newRisk_parallel"
        #run = "ES"
        #run = "sortSTD"
        #run = "sortRisk"

        if not self.config_sampler.model_available or not self.hyperjump or run == "random":
            #run a random config
            for k, v in self.data.items():
                if v.status == 'QUEUED':
                    assert v.budget == self.budgets[self.stage], 'Configuration budget does not align with current stage!'
                    v.status = 'RUNNING'
                    self.num_running += 1

                    return (k, v.config, v.budget)

        elif run == "sortLoss":
            #run the config with lower loss
            listConf = []
            for k, v in self.data.items():
                if v.status == 'QUEUED':
                    assert v.budget == self.budgets[self.stage], 'Configuration budget does not align with current stage!'
                    
                    if self.fullBudget_predictions:
                        _, loss, _ = self.config_sampler.make_predictions([v.config], self.budgets[-1])[0]
                    else:
                        _, loss, _ = self.config_sampler.make_predictions([v.config], v.budget)[0]

                    listConf.append([k, loss])

            if len(listConf) != 0:
                def sortLoss(val): 
                    return val[1] 
                listConf.sort(key = sortLoss) 
                
                k, l = listConf[0]
                self.data[k].status = 'RUNNING'
                self.num_running += 1

                if self.hyperjump and self.result_logger is not None:
                    self.result_logger.update_EAL(self.data[k].config_id, eal)

                return (k, self.data[k].config, self.data[k].budget)

        elif run == "sortLCB":
            #run the config with lower loss
            listConf = []
            for k, v in self.data.items():
                if v.status == 'QUEUED':
                    assert v.budget == self.budgets[self.stage], 'Configuration budget does not align with current stage!'
                    
                    if self.fullBudget_predictions:
                        _, loss, s = self.config_sampler.make_predictions([v.config], self.budgets[-1])[0]
                    else:
                        _, loss, s = self.config_sampler.make_predictions([v.config], v.budget)[0]

                    acc_ = 1-loss
                    #lcb = acc_ - 1.282 * s # untested 80%
                    lcb = acc_ - 1.645 * s # untested 90%
                    #lcb = acc_ - 1.960 * s # untested 95%
                    #lcb = acc_ - 2.576 * s # untested 99%
                    listConf.append([k, lcb])

            #print(listConf)
            if len(listConf) != 0:
                def sortFirst(val): 
                    return val[1] 
                listConf.sort(key = sortFirst, reverse=True) 
                
                k, l = listConf[0]
                #print("runnung config " +str(self.data[k].config) + "with predicted acc = " + str(1-l)) 
                self.data[k].status = 'RUNNING'
                self.num_running += 1

                if self.hyperjump and self.result_logger is not None:
                    self.result_logger.update_EAL(self.data[k].config_id, eal)

                return (k, self.data[k].config, self.data[k].budget)                 

        elif run == "sortUCB":
            #run the config with lower loss
            listConf = []
            for k, v in self.data.items():
                if v.status == 'QUEUED':
                    assert v.budget == self.budgets[self.stage], 'Configuration budget does not align with current stage!'
                    
                    if self.fullBudget_predictions:
                        _, loss, s = self.config_sampler.make_predictions([v.config], self.budgets[-1])[0]
                    else:
                        _, loss, s = self.config_sampler.make_predictions([v.config], v.budget)[0]

                    acc_ = 1-loss
                    #ucb = acc_ +1.282 * s # untested 80%
                    ucb = acc_ + 1.645 * s # untested 90%
                    #ucb = acc_ + 1.960 * s # untested 95%
                    #ucb = acc_ + 2.576 * s # untested 99%
                    listConf.append([k, ucb])

            #print(listConf)
            if len(listConf) != 0:
                def sortFirst(val): 
                    return val[1] 
                listConf.sort(key = sortFirst, reverse=True) 
                
                k, l = listConf[0]
                self.data[k].status = 'RUNNING'
                self.num_running += 1

                if self.hyperjump and self.result_logger is not None:
                    self.result_logger.update_EAL(self.data[k].config_id, eal)

                return (k, self.data[k].config, self.data[k].budget) 

        elif run == "newRisk":
            def sortFirst_(val): 
                return val[1] 

            #run the config that is predicted the yield an higher risk reduction 
            untested_configs, untested_configs_id, untested_losses, untested_std = self.get_untested_configs()
            if not untested_configs:
                #all configs were tested
                return None, None, None

            if len(untested_configs) == 1:
                # there is only one configs
                k = untested_configs_id[0]
                self.data[k].status = 'RUNNING'
                self.num_running += 1
                return (k, self.data[k].config, self.data[k].budget)
                
            #if in the last stage
            if self.stage == len(self.num_configs)-1:
                #run the config with higher improvement
                listConf = []

                #untested configs in unsel
                aux_unsel = []
                for i in range(0, len(untested_configs)):
                    aux_unsel.append([untested_configs[i], untested_configs_id[i], untested_losses[i], untested_std[i]])

                inc, inc_id, incAcc, _, _ = self.result_logger.returnIncumbent()
                if inc is not None:
                    sel = [[inc, inc_id, 1-incAcc, None]]

                    for i in range(len(aux_unsel)): # only untested configs in unsel
                        unsel = copy.deepcopy(aux_unsel)
                        unsel[i][3] = None
                        risk_ = self.risk(sel, unsel, self.stage) 
                        listConf.append([unsel[i][1], risk_])

                else:
                    for i in range(len(aux_unsel)):
                        listConf.append([aux_unsel[i][1], aux_unsel[i][2]]) #id, loss


                #print(listConf)
                if len(listConf) != 0:
                    listConf.sort(key = sortFirst_) 
                    
                    k, _ = listConf[0]
                    self.data[k].status = 'RUNNING'
                    self.num_running += 1

                    return (k, self.data[k].config, self.data[k].budget)
                else:
                    return None, None, None


            ## not in last stage
            tested_configs, tested_configs_id, tested_losses = self.get_tested_configs()

            all_configs = []
            for i in range(0, len(tested_configs)):
                all_configs.append([tested_configs[i], tested_configs_id[i], tested_losses[i], None])

            for i in range(0, len(untested_configs)):
                all_configs.append([untested_configs[i], untested_configs_id[i], untested_losses[i], untested_std[i]])

            #print(all_configs) 
            m = self.num_configs[self.stage+1] #no of configs to test in next stage
            aux_sel , aux_unsel = self.createSEL(all_configs, m)
            # we have now sel and unsel 

            listConf = []
            #SIMULATE THE RISK
            for i in range(len(aux_sel)):
                if aux_sel[i][3] is None: continue
                
                sel = copy.deepcopy(aux_sel)
                sel[i][3] = None
                risk_ = self.risk(sel, aux_unsel, self.stage) 
                listConf.append([sel[i][1], risk_])

            for i in range(len(aux_unsel)):
                if aux_unsel[i][3] is None: continue

                unsel = copy.deepcopy(aux_unsel)
                unsel[i][3] = None
                risk_ = self.risk(aux_sel, unsel, self.stage) 
                listConf.append([unsel[i][1], risk_])


            if len(listConf) > 0:
                listConf.sort(key=sortFirst_) 
                k, _= listConf[0]

                self.data[k].status = 'RUNNING'
                self.num_running += 1

                return (k, self.data[k].config, self.data[k].budget)


        elif run == "newRisk_parallel":
            def sortFirst_(val): 
                return val[1] 

            #run the config that is predicted the yield an higher risk reduction 
            untested_configs, untested_configs_id, untested_losses, untested_std = self.get_untested_configs()
            if not untested_configs:
                #all configs were tested
                return None, None, None
            if len(untested_configs) == 1:
                 # there is only one configs
                k = untested_configs_id[0]
                self.data[k].status = 'RUNNING'
                self.num_running += 1
                return (k, self.data[k].config, self.data[k].budget)

            if len(untested_configs) == 0:
                return None, None, None
                
            #if in the last stage
            if self.stage == len(self.num_configs)-1:
                #run the config with higher improvement
                listConf = []

                #untested configs in unsel
                aux_unsel = []
                for i in range(0, len(untested_configs)):
                    aux_unsel.append([untested_configs[i], untested_configs_id[i], untested_losses[i], untested_std[i]])

                inc, inc_id, incAcc, _, _ = self.result_logger.returnIncumbent()
                if inc is not None:
                    sel = [[inc, inc_id, 1-incAcc, None]]
                    process_list = []
                    if __name__ == 'hyperjump.core.base_iteration':

                        for i in range(len(aux_unsel)): # only untested configs in unsel                                    
                            unsel = copy.deepcopy(aux_unsel)
                            unsel[i][3] = None

                            result = self.pool.apply_async(risk_parellel, (sel, unsel, self.stage, self.budgets, )) 
                            process_list.append(result)

                        for i in range(len(process_list)):
                            res, time_ = process_list[i].get()
                            self.result_logger.updateIntegralTime(time_)
                            if isinstance(res, list):
                            #res = [SEL, UNSEL, risk_]
                                listConf.append([res[1][i][1], res[2]])

                else:
                    for i in range(len(aux_unsel)):
                        listConf.append([aux_unsel[i][1], aux_unsel[i][2]]) #id, loss


                #print(listConf)
                if len(listConf) != 0:
                    listConf.sort(key = sortFirst_) 
                    
                    k, _ = listConf[0]
                    #print("runnung config " +str(self.data[k].config) + "with predicted acc = " + str(1-l)) 
                    self.data[k].status = 'RUNNING'
                    self.num_running += 1

                    return (k, self.data[k].config, self.data[k].budget)
                else:
                    print("no configg")
                    listConf = []
                    for i in range(0, len(untested_configs)):
                        listConf.append([untested_configs_id[i], untested_losses[i]])

                    listConf.sort(key=sortFirst_) 
                    k, _= listConf[0]
                    self.data[k].status = 'RUNNING'
                    self.num_running += 1

                    return (k, self.data[k].config, self.data[k].budget)            


            ## not in last stage
            if self.SEL_set is None or self.UNSEL_set is None:
                tested_configs, tested_configs_id, tested_losses = self.get_tested_configs()

                all_configs = []
                for i in range(0, len(tested_configs)):
                    all_configs.append([tested_configs[i], tested_configs_id[i], tested_losses[i], None])

                for i in range(0, len(untested_configs)):
                    all_configs.append([untested_configs[i], untested_configs_id[i], untested_losses[i], untested_std[i]])

                #print(all_configs) 
                
                m = self.num_configs[self.stage+1] #no of configs to test in next stage
                aux_sel , aux_unsel = self.createSEL(all_configs, m)

            else:
                aux_sel = self.SEL_set
                aux_unsel = self.UNSEL_set

                #untested_configs, untested_configs_id, untested_losses, untested_std = self.get_untested_configs()

                for i in range(len(aux_sel)):
                    _, c, s = self.config_sampler.make_predictions([aux_sel[i][0]], self.budgets[self.stage])[0]
                    aux_sel[i][2] = c
                    if aux_sel[i][3] is not None:
                        aux_sel[i][3] = s

                for i in range(len(aux_unsel)):
                    _, c, s = self.config_sampler.make_predictions([aux_unsel[i][0]], self.budgets[self.stage])[0]
                    aux_unsel[i][2] = c
                    if aux_unsel[i][3] is not None:
                        aux_unsel[i][3] = s


            listConf = []
            process_list = []
            if __name__ == 'hyperjump.core.base_iteration':
                #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  

                #queue = multiprocessing.SimpleQueue()
                #SIMULATE THE RISK
                l_aux = []
                for i in range(len(aux_sel)):
                    if aux_sel[i][3] is None: continue
                    
                    sel = copy.deepcopy(aux_sel)
                    sel[i][3] = None
                    
                    result = self.pool.apply_async(risk_parellel, (sel, aux_unsel, self.stage, self.budgets, )) 
                    process_list.append(result)
                    l_aux.append(i)

                for i in range(len(process_list)):
                    res, time_ = process_list[i].get()
                    self.result_logger.updateIntegralTime(time_)
                    if isinstance(res, list):
                    #res = [SEL, UNSEL, risk_]
                        listConf.append([res[0][l_aux[i]][1], res[2]])

                l_aux.clear()
                process_list.clear()
                for i in range(len(aux_unsel)):
                    if aux_unsel[i][3] is None: continue

                    unsel = copy.deepcopy(aux_unsel)
                    unsel[i][3] = None

                    result = self.pool.apply_async(risk_parellel, (aux_sel, unsel, self.stage, self.budgets, )) 
                    process_list.append(result)
                    l_aux.append(i)

                for i in range(len(process_list)):
                    res, time_ = process_list[i].get()
                    self.result_logger.updateIntegralTime(time_)
                    if isinstance(res, list):
                    #res = [SEL, UNSEL, risk_]
                        listConf.append([res[1][l_aux[i]][1], res[2]])

                self.SEL_set = None
                self.UNSEL_set = None

                #print(listConf)
                if len(listConf) > 0:
                    listConf.sort(key=sortFirst_) 
                    k, _= listConf[0]
                    self.data[k].status = 'RUNNING'
                    self.num_running += 1

                    return (k, self.data[k].config, self.data[k].budget)
                else:
                    #print("no configg")
                    listConf = []
                    for i in range(0, len(untested_configs)):
                        listConf.append([untested_configs_id[i], untested_losses[i]])

                    listConf.sort(key=sortFirst_) 
                    k, _= listConf[0]
                    self.data[k].status = 'RUNNING'
                    self.num_running += 1
                    #print(self.data[k].budget)
                    return (k, self.data[k].config, self.data[k].budget)


        elif run == "sortSTD":
            #run the config with higher std
            listConf = []
            for k, v in self.data.items():
                if v.status == 'QUEUED':
                    assert v.budget == self.budgets[self.stage], 'Configuration budget does not align with current stage!'
                    
                    _, _, std = self.config_sampler.make_predictions([v.config], v.budget)[0]
                
                    listConf.append([k, std])

            if len(listConf) != 0:
                def sortSTD(val): 
                    return val[1]

                listConf.sort(key = sortSTD, reverse=True) 
                
                k, _ = listConf[0]
                self.data[k].status = 'RUNNING'
                self.num_running += 1

                if self.hyperjump and self.result_logger is not None:
                    self.result_logger.update_EAL(self.data[k].config_id, eal)

                return (k, self.data[k].config, self.data[k].budget)

        else:
            #run the config that is predicted the yield an higher risk reduction 
            untested_configs, untested_configs_id, untested_losses, untested_std = self.get_untested_configs()
            if not untested_configs:
                #all configs were tested
                return None, None, None

            #if in the last stage -> not run hyperjump
            if self.stage == len(self.num_configs)-1:
                #run the config with higher improvement
                listConf = []
                inc, _, incAcc, _, _ = self.result_logger.returnIncumbent()
                for k, v in self.data.items():
                    if v.status == 'QUEUED':
                        assert v.budget == self.budgets[self.stage], 'Configuration budget does not align with current stage!'        
                        _, loss, std = self.config_sampler.make_predictions([v.config], v.budget)[0]

                        if inc is not None:
                            ei = self.ExpectedAccuracyLoss(incAcc, None, 1-loss, std)
                        else:
                            ei = 1 - loss

                        #listConf.append([k, loss])
                        listConf.append([k, ei])

                #print(listConf)
                if len(listConf) != 0:
                    def sortFirst_(val): 
                        return val[1] 
                    #listConf.sort(key = sortLoss) 
                    listConf.sort(key = sortFirst_, reverse=True) 
                    
                    k, l = listConf[0]
                    #print("runnung config " +str(self.data[k].config) + "with predicted acc = " + str(1-l)) 
                    self.data[k].status = 'RUNNING'
                    self.num_running += 1

                    #if self.hyperjump and self.result_logger is not None:
                    #    self.result_logger.update_EAL(self.data[k].config_id, eal)


                    return (k, self.data[k].config, self.data[k].budget)
                else:
                    return None, None, None


            if self.SEL_set is None or self.UNSEL_set is None:
                # order by risk
                tested_configs, tested_configs_id, tested_losses = self.get_tested_configs()

                all_configs = []
                for i in range(0, len(tested_configs)):
                    all_configs.append([tested_configs[i], tested_configs_id[i], tested_losses[i], None])

                for i in range(0, len(untested_configs)):
                    all_configs.append([untested_configs[i], untested_configs_id[i], untested_losses[i], untested_std[i]])

                #print(all_configs) 
                m = self.num_configs[self.stage+1] #no of configs to test in next stage

                SEL, UNSEL = self.createSEL(all_configs, m)

            else:
                SEL = self.SEL_set 
                UNSEL = self.UNSEL_set

            #current EAL matrix
            ealMatrix = np.array([[0 for _ in range(len(SEL)+1)] for _ in range(len(UNSEL)+1)], dtype='O') 

            cj = 1
            for c2, c2_id, l2, s2 in UNSEL:
                ci = 1
                for c1, c1_id, l1, s1 in SEL:
                    #expected accuracy loss
                    aux_eal =  self.ExpectedAccuracyLoss(1-l1, s1, 1-l2, s2)
                    if aux_eal < 0:
                        aux_eal = 0
                        #if self.prints:
                        #    print("ERROR: negative EAL --> eal= " + str(aux_eal) + "acc1= " +str(1-l1) +  " s1=" + str(s1) + " acc2=" + str(1-l2) + " s2=" + str(s2))

                    if isinstance(ealMatrix[0, ci], int):
                        ealMatrix[0, ci] = [c1, c1_id, l1, s1]        
                    if isinstance(ealMatrix[cj, 0], int):
                        ealMatrix[cj, 0] = [c2, c2_id, l2, s2]

                    ealMatrix[cj, ci] = aux_eal
                    ci += 1
                cj += 1

            #simulated EAL matrix if configs in SEL are tested
            auxSEL_ealMatrix = copy.deepcopy(ealMatrix)
            for i in range(1, np.shape(auxSEL_ealMatrix)[1]):
                a1 = 1-auxSEL_ealMatrix[0,i][2]
                for j in range(1, np.shape(auxSEL_ealMatrix)[0]):
                    if auxSEL_ealMatrix[0,i][3] is not None and auxSEL_ealMatrix[j,0][3] is not None:
                        a2 = 1-auxSEL_ealMatrix[j,0][2]
                        s2 = auxSEL_ealMatrix[j,0][3]
                        auxSEL_ealMatrix[j,i] =  self.ExpectedAccuracyLoss(a1,None, a2, s2)

            #simulated EAL matrix if configs in UNSEL are tested
            auxUNSEL_ealMatrix = copy.deepcopy(ealMatrix)
            for j in range(1, np.shape(auxUNSEL_ealMatrix)[0]):
                a2 = 1-auxUNSEL_ealMatrix[j,0][2]
                for i in range(1, np.shape(auxUNSEL_ealMatrix)[1]):
                    if auxUNSEL_ealMatrix[0,i][3] is not None and auxUNSEL_ealMatrix[j,0][3] is not None:
                        a1 = 1-auxUNSEL_ealMatrix[0,i][2]
                        s1 = auxUNSEL_ealMatrix[0,i][3]
                        auxUNSEL_ealMatrix[j,i] =  self.ExpectedAccuracyLoss(a1,s1, a2, None)

            listConf = []
            for i in range(1, np.shape(auxSEL_ealMatrix)[1]):
                if auxSEL_ealMatrix[0,i][3] is not None:
                    delta = np.sum(ealMatrix[1:,i]) - np.sum(auxSEL_ealMatrix[1:,i])
                    listConf.append([ealMatrix[0,i][1], ealMatrix[0,i][0], delta])

            for j in range(1, np.shape(auxUNSEL_ealMatrix)[0]):
                if auxUNSEL_ealMatrix[j,0][3] is not None:
                    delta = np.sum(ealMatrix[j,1:]) - np.sum(auxUNSEL_ealMatrix[j,1:])
                    listConf.append( [ealMatrix[j,0][1], ealMatrix[j,0][0], delta])
          
            if len(listConf) > 0:
                def sortSecond(val): 
                    return val[2] 

                listConf.sort(key=sortSecond, reverse=True) 
                k, c, _ = listConf[0]
                if self.data[k].config != c:
                    print("ERROR: configs different!!!!")

                self.data[k].status = 'RUNNING'
                self.num_running += 1

                return (k, self.data[k].config, self.data[k].budget)

        return None, None, None

    def update_initTime(self):
        self.time_start = time.time()

    def updateOverhead(self, confID):
        if not self.result_logger is None:
            overhead = time.time() - self.time_start
            self.result_logger.updateOverhead(confID, overhead)

    def _advance_to_next_stage(self, config_ids, losses):
        """
        Function that implements the strategy to advance configs within this iteration

        Overload this to implement different strategies, like
        SuccessiveHalving, SuccessiveResampling.

        Parameters
        ----------
            config_ids: list
                all config ids to be considered
            losses: list
                losses of the run on the current budget

        Returns
        -------
            list of bool
                A boolean for each entry in config_ids indicating whether to advance it or not


        """
        raise NotImplementedError('_advance_to_next_stage not implemented for %s' % type(self).__name__)

    def process_results(self):
        """
        function that is called when a stage is completed and
        needs to be analyzed befor further computations.

        The code here implements the original SH algorithms by
        advancing the k-best (lowest loss) configurations at the current
        budget. k is defined by the num_configs list (see __init__)
        and the current stage value.

        For more advanced methods like resampling after each stage,
        overload this function only.
        """
        self.stage += 1

        # print("[ITERATION] - Processing Results")

        # collect all config_ids that need to be compared
        config_ids = list(filter(lambda cid: self.data[cid].status == 'REVIEW', self.data.keys()))

        if (self.stage >= len(self.num_configs)):
            #print("              Need to finish up")
            #finish bracket
            
            self.finish_up()
            self.config_sampler.reset_testedConfig_counter()
            return

        budgets = [self.data[cid].budget for cid in config_ids]
        # print("              Budgets: ", budgets)

        if len(set(budgets)) > 1:
            raise RuntimeError('Not all configurations have the same budget!')

        budget = self.budgets[self.stage - 1]
        # print("              Evaluating budget: ", budget)

        losses = np.array([self.data[cid].results[budget]['loss'] for cid in config_ids])
        # print("              Losses: ", losses)

        advance = self._advance_to_next_stage(config_ids, losses)
        # print("              Advancing configs:")

        for i, a in enumerate(advance):
            if a:
                self.logger.debug(
                    'ITERATION: Advancing config %s to next budget %f' % (config_ids[i], self.budgets[self.stage]))

        for i, cid in enumerate(config_ids):
            if advance[i]:
                self.data[cid].status = 'QUEUED'
                self.data[cid].budget = self.budgets[self.stage]
                self.actual_num_configs[self.stage] += 1
            else:
                self.data[cid].status = 'TERMINATED'
        
        self.SEL_set = None
        self.UNSEL_set = None


    def finish_up(self):
        self.is_finished = True

        for k, v in self.data.items():
            assert v.status in ['TERMINATED', 'REVIEW', 'CRASHED'], 'Configuration has not finshed yet!'
            #if not (v.status == 'TERMINATED' or  v.status =='REVIEW'  or v.status == 'CRASHED'):
            #    print('Configuration has not finshed yet!')
            v.status = 'COMPLETED'

        if self.hyperjump:
            self.pool.close()


    def __repr__(self):
        raise NotImplementedError('This function needs to be overwritten in %s.' % (self.__class__.__name__))


    def get_tested_configs(self):
        # Gets sampled configs that are yet to be compared
        config_ids = list(filter(lambda cid: self.data[cid].status == 'REVIEW', self.data.keys()))
        budget = self.budgets[self.stage]
        list_cid = []
        list_configs = []
        list_losses = []
        #for i, cid in enumerate(self.data.keys()):
        #    if cid in config_ids:
        for cid in config_ids:
                #print(self.data[cid].config)
                #print(self.data[cid].results[budget]['loss'])
                list_cid.append(cid)
                list_configs.append(self.data[cid].config)
                list_losses.append(self.data[cid].results[budget]['loss'])

        return list_configs, list_cid, list_losses

    def get_untested_configs(self):

        budget = self.budgets[self.stage]

        list_configs = []
        list_config_ids = []
        for k, v in self.data.items():
            if v.status == 'QUEUED':
                list_configs.append(v.config)
                list_config_ids.append(v.config_id)
                #print(v.config)


        list_untested_configs = self.config_sampler.make_predictions(list_configs, budget)

        list_losses = []
        list_stds = []
        for _, untested_losses, untested_std in list_untested_configs:
            list_losses.append(untested_losses)
            list_stds.append(untested_std)

        return list_configs, list_config_ids, list_losses, list_stds


    def createSEL(self, all_configs, noConfSel):
        """
        Function tho create the the set to transist to the next stage and teh set to discard
        
        We provide different heuristics 
        However, HJ uses as default "newRisk3_parallel"
        """


        ################################################
        #       order configs to sel
        ################################################

        #sort = "acc"
        #sort = "lcb"
        #sort = "ucb"
        #sort = "random"
        #sort = "newRisk"
        #sort = "newRisk2"
        #sort = "newRisk2_parallel"
        sort = "newRisk3_parallel"
        #sort= "allSEL"
        #sort = "both"
        #sort = "risk"
        def sortSecond(val): 
            return val[2] 

        if sort == "acc": #sort by acc
            all_configs.sort(key = sortSecond) #sort configs by loss
            #all_configs.sort(key = sortSecond, reverse = True) #sort configs by accuracy        

            SEL = all_configs[0:noConfSel]  #predicted configs to transit to the next stage
            UNSEL = all_configs[noConfSel:] #predicted configs to be dropped in the current stage
        
        elif sort == "lcb": #sort by lcb
            
            listSet = []
            for i in range(len(all_configs)):
                acc_ = 1 - all_configs[i][2]
                s = all_configs[i][3]
                if s is None:
                    lcb = acc_
                else:
                    #lcb = acc_ - 1.282 * s # untested 80%
                    lcb = acc_ - 1.645 * s # untested 90%
                    #lcb = acc_ - 1.960 * s # untested 95%
                    #lcb = acc_ - 2.576 * s # untested 99%
                listSet.append([all_configs[i],  lcb])

            def sortFirst(val): 
                return val[1] 

            listSet.sort(key = sortFirst, reverse=True) #sort configs by loss
            listSet_ = next(zip(*listSet)) # list with the first element of  listSet

            SEL = listSet_[0:noConfSel]  #predicted configs to transit to the next stage
            UNSEL = listSet_[noConfSel:] #predicted configs to be dropped in the current stage

        elif sort == "ucb": #sort by ucb
            
            listSet = []
            for i in range(len(all_configs)):
                acc_ = 1 - all_configs[i][2]
                s = all_configs[i][3]
                if s is None:
                    ucb = acc_
                else:
                    #ucb = acc_ + 1.282 * s # untested 80%
                    ucb = acc_ + 1.645 * s # untested 90%
                    #ucb = acc_ + 1.960 * s # untested 95%
                    #ucb = acc_ + 2.576 * s # untested 99%
                listSet.append([all_configs[i],  ucb])

            def sortFirst(val): 
                return val[1] 

            listSet.sort(key = sortFirst, reverse=True) #sort configs by loss
            listSet_ = next(zip(*listSet)) # list with the first element of  listSet

            SEL = listSet_[0:noConfSel]  #predicted configs to transit to the next stage
            UNSEL = listSet_[noConfSel:] #predicted configs to be dropped in the current stage

        elif sort == "random":
            list_num = list(np.arange(len(all_configs)))
            random.shuffle(list_num)
            SEL = []
            UNSEL = []

            for i in range(len(all_configs)):
                if i < noConfSel:
                    SEL.append(all_configs[i])
                else:
                    UNSEL.append(all_configs[i])

            #print(SEL)
            #print(UNSEL)

        elif sort == "newRisk":
            ealMatrix = []

            for i in range(0, len(all_configs)):
                Sel = [all_configs[i]]
                Unsel = []

                for j in range(0, len(all_configs)):
                    if i == j: continue
                    Unsel.append(all_configs[i])

                eal = self.risk(Sel, Unsel, self.stage) 
                ealMatrix.append([all_configs[i], eal])

            def sortFirst(val): 
                return val[1] 
                
            ealMatrix.sort(key = sortFirst) #sort configs by loss
            #all_configs.sort(key = sortSecond, reverse = True) #sort configs by accuracy   
            listSet_ = next(zip(*ealMatrix))     

            SEL = listSet_[0:noConfSel]  #predicted configs to transit to the next stage
            UNSEL = listSet_[noConfSel:] #predicted configs to be dropped in the current stage

        elif sort == "newRisk2":
            ealMatrix = []

            all_configs.sort(key = sortSecond) #sort configs by loss
            #all_configs.sort(key = sortSecond, reverse = True) #sort configs by accuracy        

            sel = all_configs[0:noConfSel]  #predicted configs to transit to the next stage
            unsel = all_configs[noConfSel:] #predicted configs to be dropped in the current stage

            eal = self.risk(sel, unsel, self.stage) 
            ealMatrix.append([copy.deepcopy(sel), copy.deepcopy(unsel), eal])

            for i in range(len(sel)):
                for j in range(len(unsel)):
                    aux_sel = copy.deepcopy(sel)
                    aux_unsel = copy.deepcopy(unsel)

                    conf_sel = aux_sel.pop(i)
                    conf_unsel = aux_unsel.pop(j)

                    aux_sel.append(conf_unsel)
                    aux_unsel.append(conf_sel)

                    #expected accuracy loss
                    aux_eal =  self.risk(aux_sel, aux_unsel, self.stage)
                    ealMatrix.append([copy.deepcopy(aux_sel), copy.deepcopy(aux_unsel), copy.deepcopy(aux_eal)])

            ealMatrix.sort(key = sortSecond) #sort configs by loss

            SEL = ealMatrix[0][0]  #predicted configs to transit to the next stage
            UNSEL = ealMatrix[0][1] #predicted configs to be dropped in the current stage

        elif sort == "newRisk2_parallel":
            ealMatrix = []
            #queue = multiprocessing.SimpleQueue()

            all_configs.sort(key = sortSecond) #sort configs by loss
            #all_configs.sort(key = sortSecond, reverse = True) #sort configs by accuracy        

            sel = all_configs[0:noConfSel]  #predicted configs to transit to the next stage
            unsel = all_configs[noConfSel:] #predicted configs to be dropped in the current stage

            process_list = []

            if __name__ == 'hyperjump.core.base_iteration':
                #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                result = self.pool.apply_async(risk_parellel, (sel, unsel, self.stage, self.budgets,)) 
                process_list.append(result)


                for i in range(len(sel)):
                    for j in range(len(unsel)):
                        aux_sel = copy.deepcopy(sel)
                        aux_unsel = copy.deepcopy(unsel)

                        conf_sel = aux_sel.pop(i)
                        conf_unsel = aux_unsel.pop(j)

                        aux_sel.append(conf_unsel)
                        aux_unsel.append(conf_sel)
                        #result = pool.apply_async(self.risk_parellel, (aux_sel, aux_unsel, self.stage,)) 
                        result = self.pool.apply_async(risk_parellel, (aux_sel, aux_unsel, self.stage, self.budgets, )) 
                        process_list.append(result)


                for i in range(len(process_list)):
                    res, time_ = process_list[i].get()
                    self.result_logger.updateIntegralTime(time_)
                    if isinstance(res, list):
                        ealMatrix.append([res[0], res[1], res[2]])

  
            if len(ealMatrix) > 0:
                ealMatrix.sort(key = sortSecond) #sort configs by loss
                #print(ealMatrix[0])

                SEL = ealMatrix[0][0]  #predicted configs to transit to the next stage
                UNSEL = ealMatrix[0][1] #predicted configs to be dropped in the current stage
            else:

                SEL = all_configs[0:noConfSel]  #predicted configs to transit to the next stage
                UNSEL = all_configs[noConfSel:] #predicted configs to be dropped in the current stage
                

        elif sort == "newRisk3_parallel":

            ealMatrix = []
            process_list = []
            all_configs.sort(key = sortSecond) #sort configs by loss

            sel = all_configs[0:noConfSel]  #predicted configs to transit to the next stage
            unsel = all_configs[noConfSel:] #predicted configs to be dropped in the current stage

            if __name__ == 'hyperjump.core.base_iteration':
                result = self.pool.apply_async(risk_parellel, (sel, unsel, self.stage, self.budgets,)) 
                process_list.append(result)

                for i in range(1, 4):
                    k = int(noConfSel / (self.eta**i))
                    if k < 1: break
                    
                    sel_acc_per = copy.deepcopy(sel[:-k])
                    sel_acc_rem = copy.deepcopy(sel[-k:])

                    unsel_acc_rem = copy.deepcopy(unsel[:k])
                    unsel_acc_per = copy.deepcopy(unsel[k:])

                    sel_acc = sel_acc_per + unsel_acc_rem 
                    unsel_acc = sel_acc_rem + unsel_acc_per

                    result = self.pool.apply_async(risk_parellel, (sel_acc, unsel_acc, self.stage, self.budgets, )) 
                    process_list.append(result)


                    ## WE ARE USING LOSS not accuracy
                    sel_ucb = copy.deepcopy(sel)
                    unsel_lcb = copy.deepcopy(unsel)

                    for j in range(len(sel_ucb)):
                        if sel_ucb[j][3] is None: 
                            sel_ucb[j].append(sel_ucb[j][2])
                        else:
                            sel_ucb[j].append(sel_ucb[j][2] + 1.645 * sel_ucb[j][3])

                    for j in range(len(unsel_lcb)):
                        if unsel_lcb[j][3] is None: 
                            unsel_lcb[j].append(unsel_lcb[j][2])
                        else:
                            unsel_lcb[j].append(unsel_lcb[j][2] - 1.645 * unsel_lcb[j][3])

                    def sortForth(val): 
                        return val[4] 

                    sel_ucb.sort(key = sortForth) #sort configs by lcb
                    unsel_lcb.sort(key = sortForth) #sort configs by ucb

                    sel_ucb_per = copy.deepcopy(sel_ucb[:-k])
                    sel_ucb_rem = copy.deepcopy(sel_ucb[-k:])

                    unsel_lcb_rem = copy.deepcopy(unsel_lcb[:k])
                    unsel_lcb_per = copy.deepcopy(unsel_lcb[k:])

                    sel_ucb_ = [] 
                    for j in range(len(sel_ucb_per)):
                        sel_ucb_.append(sel_ucb_per[j][0:4])
                    for j in range(len(unsel_lcb_rem)):
                        sel_ucb_.append(unsel_lcb_rem[j][0:4])


                    unsel_lcb_ = [] 
                    for j in range(len(unsel_lcb_per)):
                        unsel_lcb_.append(unsel_lcb_per[j][0:4])
                    for j in range(len(sel_ucb_rem)):
                        unsel_lcb_.append(sel_ucb_rem[j][0:4])

                    result = self.pool.apply_async(risk_parellel, (sel_ucb_, unsel_lcb_, self.stage, self.budgets, )) 
                    process_list.append(result)


                for i in range(len(process_list)):
                    res, time_ = process_list[i].get()
                    self.result_logger.updateIntegralTime(time_)
                    if isinstance(res, list):
                        ealMatrix.append([res[0], res[1], res[2]])


            if len(ealMatrix) > 0:
                ealMatrix.sort(key = sortSecond) #sort configs by loss
                #print(ealMatrix[0])

                SEL = ealMatrix[0][0]  #predicted configs to transit to the next stage
                UNSEL = ealMatrix[0][1] #predicted configs to be dropped in the current stage

            else:
                all_configs.sort(key = sortSecond) #sort configs by loss
                #all_configs.sort(key = sortSecond, reverse = True) #sort configs by accuracy        

                SEL = all_configs[0:noConfSel]  #predicted configs to transit to the next stage
                UNSEL = all_configs[noConfSel:] #predicted configs to be dropped in the current stage


        elif sort == "allSEL": 
            # evaluate all SEL sets

            no_cpus = int(multiprocessing.cpu_count() / 1) -2
            _l = [i for i in range(len(all_configs))]
            all_top_sort = list(itertools.combinations(_l, noConfSel))

            print("All Sets created " + str(len(all_top_sort)))
            sorts = []
            for sel_ in all_top_sort:
                unsel_ = []
                for i in range(len(all_configs)):
                    if i not in sel_:
                        unsel_.append(i)
                sorts.append([sel_, unsel_])

            #print(no_cpus)
            if no_cpus > 1:
                queue = multiprocessing.SimpleQueue()
                process_list = []
                count = 0
                noConfigs = len(sorts)
                no_cpus_available = no_cpus

                for i in range(1, no_cpus+1):
                    size_v = int(np.rint(1.0 * noConfigs / no_cpus_available))

                    if i == 1:
                        vec = sorts[0:size_v]
                    elif i == no_cpus:
                        vec = sorts[count:]
                    else:
                        vec = sorts[count:count+size_v]

                    process = multiprocessing.Process(target=self.computeEAL_allSEL, args=(vec, all_configs, queue))

                    process.start()
                    process_list.append(process)

                    count += size_v
                    noConfigs -= size_v
                    no_cpus_available -= 1

                listSet = []
                for p in process_list:
                    listSet += queue.get()

                #wait termination
                for p in process_list:
                    p.join()

                #minimize eal 
                print("size list Set " + str(len(listSet)))
                listSet.sort(key = sortSecond)

                SEL = listSet[0][0]
                UNSEL = listSet[0][1]

            else:

                listSet = self.computeEAL_allSEL(sorts, all_configs)
 
                #maximize 1/EAL -> if there is no cost model
                #maximize cost_reduction/eal if there is cost model
                #listSet.sort(key = sortSecond, reverse=True)

                #minimize eal 
                listSet.sort(key = sortSecond)

                SEL = listSet[0][0]
                UNSEL = listSet[0][1]


        else: # risk
            #current EAL matrix
            ealMatrix = np.array([[0 for _ in range(len(all_configs)+1)] for _ in range(len(all_configs)+1)], dtype='O') 

            for i in range(1, len(all_configs)+1):
                ealMatrix[0,i] =  all_configs[i-1]
                ealMatrix[i,0] =  all_configs[i-1]


            for i in range(1, len(all_configs)+1):
                _, _, li, si = ealMatrix[i,0]
                for j in range(1, len(all_configs)+1):
                    if i == j: continue
                    
                    _, _, lj, sj = ealMatrix[0,i]
                    ealMatrix[i,j] =  self.ExpectedAccuracyLoss(1-li, si, 1-lj, sj)

            listSet = []
            for i in range(1, len(all_configs)+1):
                listSet.append([ealMatrix[i,0],  np.sum(ealMatrix[i, 1:])])
            
            def sortFirst(val): 
                return val[1] 
                
            listSet.sort(key = sortFirst) #sort configs by loss
            #all_configs.sort(key = sortSecond, reverse = True) #sort configs by accuracy   
            listSet_ = next(zip(*listSet))     

            SEL = listSet_[0:noConfSel]  #predicted configs to transit to the next stage
            UNSEL = listSet_[noConfSel:] #predicted configs to be dropped in the current stage

        return SEL, UNSEL


    def computeEAL_allSEL(self, sorts, all_configs, queue=None):
        listSet = []
        for set_ in sorts:
            _sel = []
            _unsel = []
            for i in set_[0]:
                _sel.append(all_configs[i])
            for i in set_[1]:
                _unsel.append(all_configs[i])
                    
            eal = 0.0
            #cost_reduction = 0
            for c, cid, l2, s2 in _unsel:


                for _, _, l1, s1 in _sel:
                #expected accuracy loss
                    aux_eal =  self.ExpectedAccuracyLoss(1-l1, s1, 1-l2, s2)
                    if aux_eal < 0: 
                        aux_eal = 0
                    eal += aux_eal


            listSet.append([copy.deepcopy(_sel), copy.deepcopy(_unsel), eal])

        if queue is None:
            return listSet
        else:   
            queue.put(listSet)

    def process_results_to_jump(self, targetStage, SEL, UNSEL):
        """
        function that is called when it is predicted that we should jump for a taget budget

        Terminate configs that were queded or not finished
        SEL configs to continue to the next stage
        """
        self.stage = targetStage # next stage jump

        #print("stopping configs:")
        for _,cid, _, _ in UNSEL:
            self.data[cid].status = 'TERMINATED'
            #print(self.data[cid].config)
 
        #print("continue configs:")
        for config,cid, _, _ in SEL:
            self.data[cid].status = 'QUEUED'
            self.data[cid].budget = self.budgets[self.stage]
            self.actual_num_configs[self.stage] += 1
            #print(self.data[cid].config)


            self.logger.debug(
                'ITERATION: Advancing config %s to next budget %f' % (config, self.budgets[self.stage]))


    def analyse_risk_new(self):
        if not self.config_sampler.model_available:
            if self.prints:
                print("[HYPERJUMP] ---> ERROR: no models\n")
            if self.result_logger is not None:
                self.result_logger.updateOption("c")     
            return -1

        firstSel = None
        firstUnsel = None

        #if in the last stage -> not run hyperjump
        if self.stage == len(self.num_configs)-1:
            #if self.actual_num_configs[self.stage] == self.num_configs[self.stage]-1:
            #    #last config to test in the current bracker
            #    if self.result_logger is not None:
            #        self.result_logger.updateOption("n")

            untested_configs, untested_configs_id , untested_losses, untested_std = self.get_untested_configs()
            if not untested_configs:
                #all configs were tested
                if self.result_logger is not None:
                    self.result_logger.updateOption("n")           
                return -1
            #print(untested_configs)
              
            inc, inc_id, incAcc, _, _ = self.result_logger.returnIncumbent()
            if inc is not None:
                inc_loss = 1-incAcc
                SEL = [[inc, inc_id, inc_loss, None]]

                aux_unsel = []
                for i in range(0, len(untested_configs)):
                    aux_unsel.append([untested_configs[i], untested_configs_id[i], untested_losses[i], untested_std[i]])
                
                UNSEL = []
                for i in range(0, len(aux_unsel)):
                    #UNSEL.append(aux_unsel[i])
                    UNSEL.append([aux_unsel[i][0], aux_unsel[i][1], aux_unsel[i][2], aux_unsel[i][3]])

                #if firstSel is None:
                #    firstSel = copy.deepcopy(SEL)
                #if firstUnsel is None:
                #    firstUnsel = copy.deepcopy(UNSEL)

                eal = self.risk(SEL, UNSEL, self.stage) 

                #T = self.thresholdRisk 
                T = self.thresholdRisk * (inc_loss)
                
                if self.prints:
                    print(" ---> EALl = " + str(eal) + " -- threshold = " + str(T) + " lenSEL " + str(len(SEL)) + " lenUNSEL " + str(len(UNSEL)))
                

                #risk_of_using_the_model = self.randFrac * inc_loss * eal
                risk_of_using_the_model = 1-self.randFrac #* inc_loss * eal
                rand_value = random.uniform(0, 1)
                if rand_value  < risk_of_using_the_model:
                    print("RISK OF USING THE MODEL HIGH rand" + str(rand_value) + " risk_model " + str(risk_of_using_the_model))

                if eal < T and  rand_value  > risk_of_using_the_model:
                    #jump to next bracker -> SEL = []  UNSEL = all_configs

                    self.result_logger.jumpInfo(SEL, UNSEL, self.budgets[self.stage], self.budgets[self.stage], self.budgets, self.num_configs)

                    no_confs_jump = len(UNSEL)
                    self.config_sampler.configsToJump(no_confs_jump)
                    self.process_results_to_jump(self.stage, [], UNSEL)

                    self.SEL_set = None
                    self.UNSEL_set = None

                    #self.result_logger.maxInSel([[inc,0,0,0]], self.budgets[self.stage])

                    if self.result_logger is not None:
                        self.result_logger.updateOption("n")
                else:

                    self.SEL_set = copy.deepcopy(SEL)
                    self.UNSEL_set = copy.deepcopy(UNSEL)
                    #not jump
                    if self.result_logger is not None:
                        self.result_logger.updateOption("c")    
                    
                return [(eal, T)]

            else:    #no jump continure
                self.SEL_set = None
                self.UNSEL_set = None   

            return -1

        # return the stage to jump or the current stage if not to jump
        #n = self.num_configs[self.stage] # no of config in current stage
        tested_configs, tested_configs_id, tested_losses = self.get_tested_configs()
        if not tested_configs:
            if self.result_logger is not None:
                self.result_logger.updateOption("c")
            #no test configs, SEL is empy
            return -1
     
        untested_configs, untested_configs_id, untested_losses, untested_std = self.get_untested_configs()
        if not untested_configs:
            if self.result_logger is not None:
                self.result_logger.updateOption("c")
            #all configs were tested
            return -1

        all_configs = []     
        if self.fullBudget_predictions:
            for i in range(0, len(tested_configs)):
                _, l, s = self.config_sampler.make_predictions([tested_configs[i]], self.budgets[-1])[0] #full budget
                all_configs.append([tested_configs[i], tested_configs_id[i], l, s])

            for i in range(0, len(untested_configs)):
                _, l, s = self.config_sampler.make_predictions([untested_configs[i]], self.budgets[-1])[0] #full budget
                all_configs.append([untested_configs[i], untested_configs_id[i], l, s])
                    
        else:
            for i in range(0, len(tested_configs)):
                all_configs.append([tested_configs[i], tested_configs_id[i], tested_losses[i], None])

            for i in range(0, len(untested_configs)):
                all_configs.append([untested_configs[i], untested_configs_id[i], untested_losses[i], untested_std[i]])
                    
        eal_list = [] 

        prevUNSEL = []
        prevSEL = []
        prevUNSEL1 = []
        EAL = 0

        for targetStage in range (self.stage + 1, len(self.num_configs)):

            m = self.num_configs[targetStage] #no of configs to test in next stage

            SEL, UNSEL = self.createSEL(all_configs, m)
            self.SEL_set = copy.deepcopy(SEL)
            self.UNSEL_set = copy.deepcopy(UNSEL)
            #if targetStage == self.stage + 1:
            #self.SEL_set = copy.deepcopy(SEL)
            #self.UNSEL_set = copy.deepcopy(UNSEL)

            if firstSel is None:
                firstSel = copy.deepcopy(SEL)
            if firstUnsel is None:
                firstUnsel = copy.deepcopy(UNSEL)

            EAL += self.risk(SEL, UNSEL, targetStage-1) 
            #self.result_logger.jumpInfo(SEL, UNSEL, self.budgets[targetStage-1])

            # THRESHOLD
            #T = self.thresholdRisk_d[self.budgets[targetStage-1]]
            #T = self.thresholdRisk

            inc, inc_id, incAcc, _, _ = self.result_logger.returnIncumbent()
            if inc is not None:
                inc_loss = 1.0-incAcc
            else:
                inc_loss = 1.0

            T = self.thresholdRisk * (inc_loss)

            eal_list.append((EAL, T))

            if self.prints:
                #print(" ---> EAL = " + str(EAL) + " -- threshold = " + str(self.thresholdRisk))
                print(" ---> EAL = " + str(EAL) + " -- threshold = " + str(T) + " lenSEL " + str(len(SEL)) + " lenUNSEL " + str(len(UNSEL)))


            if EAL >= T:
                #not to jump to this target stage -> return the previous stage
                #risk_of_using_the_model = self.randFrac * inc_loss * EAL
                risk_of_using_the_model = 1-self.randFrac #* inc_loss * EAL
                rand_value = random.uniform(0, 1)

                if rand_value  < risk_of_using_the_model:
                    print("RISK OF USING THE MODEL HIGH rand" + str(rand_value) + " risk_model " + str(risk_of_using_the_model))

                if targetStage-1 > self.stage and rand_value > risk_of_using_the_model: # if jump -> process results
                    if self.prints:
                        print("[HYPERJUMP]---> EVALUATING RISK ---> Jump to budget " + str(self.budgets[targetStage-1]) + "\n")
    
                    self.result_logger.jumpInfo(firstSel, firstUnsel, self.budgets[self.stage], self.budgets[targetStage-1], self.budgets, self.num_configs)

                    #only for ploting the progess on the bar 
                    no_confs_jump = len(untested_configs) # configs not tested in the current stage
                    for i in range(self.stage+1, targetStage-1): #count configs not tested because of jump
                        no_confs_jump += self.num_configs[i] 
                    self.config_sampler.configsToJump(no_confs_jump)
                
                    #self.result_logger.maxInSel(prevSEL, self.budgets[targetStage-1])

                    #m = self.num_configs[targetStage-1]
                    self.process_results_to_jump(targetStage-1, prevSEL, prevUNSEL)
                    #self.SEL_set = copy.deepcopy(SEL)
                    #self.UNSEL_set = copy.deepcopy(UNSEL)

                    if self.result_logger is not None:
                        self.result_logger.updateOption("j")

                else:
                    #self.SEL_set = copy.deepcopy(SEL)
                    #self.UNSEL_set = copy.deepcopy(UNSEL)

                    if self.result_logger is not None:
                        self.result_logger.updateOption("c")
                    if self.prints:
                        print("[HYPERJUMP]---> EVALUATING RISK ---> NOT JUMP\n")

                break 

            else:
                if targetStage ==  len(self.num_configs)-1:
                    # last stage

                    prevUNSEL += UNSEL
                    
                    #only for ploting the progess on the bar 
                    no_confs_jump = len(untested_configs)       # configs not tested in the current stage
                    for i in range(self.stage+1, targetStage):  #count configs not tested because of jump
                        no_confs_jump += self.num_configs[i] 

                    inc, inc_id, incAcc, _, _ = self.result_logger.returnIncumbent()
                    if inc is not None:
                        inc_loss = 1-incAcc
                        sel = [[inc, inc_id, inc_loss, None]]

                        unsel = []
                        for i in range(0, len(SEL)):
                            unsel.append([SEL[i][0], SEL[i][1], SEL[i][2], SEL[i][3]])
                            #unsel.append(SEL[i])

                        EAL += self.risk(sel, unsel, targetStage) 
                        #self.result_logger.jumpInfo(sel, unsel, self.budgets[targetStage])

                        #T = self.thresholdRisk
                        T = self.thresholdRisk * inc_loss
                        eal_list.append((EAL, T))

                        if self.prints:
                            #print(" ---> EAL = " + str(EAL) + " -- threshold = " + str(self.thresholdRisk))
                            print("l---> EAL = " + str(EAL) + " -- threshold = " + str(T) + " lenSEL " + str(len(sel)) + " lenUNSEL " + str(len(unsel)))
                        
                        
                        #risk_of_using_the_model = self.randFrac * inc_loss * EAL
                        risk_of_using_the_model = 1-self.randFrac #* inc_loss * EAL
                        rand_value = random.uniform(0, 1)
                        
                        if rand_value  < risk_of_using_the_model:
                            print("RISK OF USING THE MODEL HIGH rand" + str(rand_value) + " risk_model " + str(risk_of_using_the_model))
                            

                        if EAL < T and  rand_value  > risk_of_using_the_model: # skip last stage
                        #jump to next bracket -> SEL = []  UNSEL = all_configs

                            self.result_logger.jumpInfo(firstSel, firstUnsel, self.budgets[self.stage], self.budgets[targetStage], self.budgets, self.num_configs)

                            no_confs_jump += self.num_configs[targetStage] #len(UNSEL) + len(SEL)
                            
                            self.config_sampler.configsToJump(no_confs_jump)
                            prevUNSEL += SEL
                            self.process_results_to_jump(targetStage, [], prevUNSEL)

                            #self.result_logger.maxInSel([[inc,0,0,0]], self.budgets[targetStage])

                            self.SEL_set = None
                            self.UNSEL_set = None

                            if self.result_logger is not None:
                                self.result_logger.updateOption("n")
                            
                            if self.prints:
                                print("[HYPERJUMP]---> EVALUATING RISK ---> Jump to next bracket \n")
                        
                            return eal_list

                        self.SEL_set = copy.deepcopy(sel)
                        self.UNSEL_set = copy.deepcopy(unsel)
                    else:
                        self.SEL_set = None
                        self.UNSEL_set = None

                    # jump to the last stage
                    if self.prints:
                        print("[HYPERJUMP]---> EVALUATING RISK ---> Jump to last budget " + str(self.budgets[targetStage]) + "\n")
                        
                    self.result_logger.jumpInfo(firstSel, firstUnsel, self.budgets[self.stage], self.budgets[targetStage], self.budgets, self.num_configs)
                    
                    self.config_sampler.configsToJump(no_confs_jump)
                    self.process_results_to_jump(targetStage, SEL, prevUNSEL)

                    if self.result_logger is not None:
                        self.result_logger.updateOption("j")
                    
                    return eal_list
                    ################################################
                else:
                    #continue to the next stage/budget
                    prevSEL = copy.deepcopy(SEL)
                    prevUNSEL1 = copy.deepcopy(UNSEL)
                    prevUNSEL += UNSEL

                    _allConfigs  = []
                    _allConfigs_ids = []
                    for conf ,conf_id , _, _ in SEL:
                        _allConfigs.append(conf)
                        _allConfigs_ids.append(conf_id)

                    currentBudget = self.budgets[targetStage]
                    if self.fullBudget_predictions:
                        aux_allConfigs = self.config_sampler.make_predictions(_allConfigs, self.budgets[-1])
                    else:
                        aux_allConfigs = self.config_sampler.make_predictions(_allConfigs, currentBudget)

                    _allConfigs_losses = []
                    _allConfigs_std = []
                    for _, _losses, _std in aux_allConfigs:
                        _allConfigs_losses.append(_losses)
                        _allConfigs_std.append(_std)

                    all_configs = []
                    for i in range(0, len(_allConfigs)):
                        all_configs.append([_allConfigs[i], _allConfigs_ids[i], _allConfigs_losses[i], _allConfigs_std[i]])

        return eal_list

    def fx(self, x, S):
        # Not being used
        # function to integrate is implemented in C

        prod_ = 1
        sum_ = 0

        for _,_ ,u_c, s_c in S:
            if s_c is None: s_c = 10e-6

            prod_ *= mp.ncdf((x-u_c)/s_c)
            sum_ += (mp.npdf((x-u_c)/s_c) / (s_c * mp.ncdf((x-u_c)/s_c)))

        return sum_ * prod_

    def risk(self, SEL, UNSEL, stage):

        #not being used

        SEL_BudgetMax = copy.deepcopy(SEL)
        Max_tested_SEL = -1
        no_tested_SEL = 0
        no_untested_SEL = 0

        for i in range(len(SEL_BudgetMax)):
            c, _ , loss, std = SEL_BudgetMax[i]

            if std is None: #tested config
                no_tested_SEL += 1
                acc = 1 - loss
                SEL_BudgetMax[i][2] = acc
                SEL_BudgetMax[i][3] = 0.0
                if acc >  Max_tested_SEL:
                    Max_tested_SEL = acc

            else:
            #print(c)
                no_untested_SEL += 1
                if self.fullBudget_predictions:
                    _, l, s = self.config_sampler.make_predictions([c], self.budgets[-1])[0] #full budget
                else:
                    _, l, s = self.config_sampler.make_predictions([c], self.budgets[stage])[0] #current 
                    
                SEL_BudgetMax[i][2] = 1 - l[0] #acc
                if s < 0.05:
                    SEL_BudgetMax[i][3] = 0.05
                    self.result_logger.updateStdCounter()
                else:
                    SEL_BudgetMax[i][3] = s[0]


        UNSEL_BudgetMax = copy.deepcopy(UNSEL)
        Max_tested_UNSEL = -1
        no_tested_UNSEL = 0
        no_untested_UNSEL = 0

        for i in range(len(UNSEL_BudgetMax)):
            c, _ , loss, std = UNSEL_BudgetMax[i]
            #print(c)
            if std is None: #tested config
                no_tested_UNSEL += 1
                acc = 1 - loss 
                UNSEL_BudgetMax[i][2] = acc
                UNSEL_BudgetMax[i][3] = 0.0
                if acc >  Max_tested_UNSEL:
                    Max_tested_UNSEL = acc
            
            else:
                no_untested_UNSEL += 1
                if self.fullBudget_predictions: 
                    _, l, s = self.config_sampler.make_predictions([c], self.budgets[-1])[0] #full budget
                else:
                    _, l, s = self.config_sampler.make_predictions([c], self.budgets[stage])[0] #current budget

                UNSEL_BudgetMax[i][2] = 1 - l[0] #acc
                if s < 0.05:
                    UNSEL_BudgetMax[i][3] = 0.05
                    self.result_logger.updateStdCounter()
                else:
                    UNSEL_BudgetMax[i][3] = s[0]
        
        #if len(SEL_BudgetMax) == 1 and len(UNSEL_BudgetMax) == 1:
        #    area = self.ExpectedAccuracyLoss(1-SEL_BudgetMax[0][2], SEL_BudgetMax[0][3], 1-UNSEL_BudgetMax[0][2], UNSEL_BudgetMax[0][3] )
        #    return area

        if no_untested_SEL == 0:
            #only tested config in SEL

            if no_untested_UNSEL == 0 :
                #only tested config in UNSEL
                return 0
            else:
                #untested and tested configs in UNSEL
                #sel is a dirac

                lib = ctypes.CDLL(os.path.abspath('func.so'))
                lib.fd.restype = ctypes.c_double
                lib.fd.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),  ctypes.c_void_p)

                #array
                size = 3 + 1 + 2*len(UNSEL_BudgetMax)
                data = (ctypes.c_double * size)()
                #DATA  array 
                #DATA  array 
                # [total configs,  No SEL,   No UNSEL,  SEL   ,   UNSEL]
                # [total configs,  0,   No UNSEL,  SEL   ,   UNSEL]
                # [total configs, No SEL, No UNSEL, SEL_max, UNSEL[0][0], UNSEL[0][1], UNSEL[1][0], UNSEL[1][1]]

                data[0] = size
                data[1] = 0
                data[2] = len(UNSEL_BudgetMax)
                data[3] = Max_tested_SEL

                count = 4
                for i in range(len(UNSEL_BudgetMax)):
                    data[count] = UNSEL_BudgetMax[i][2]
                    data[count + 1] = UNSEL_BudgetMax[i][3]
                    count += 2


                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        data_= ctypes.cast(data, ctypes.c_void_p)
                        func = LowLevelCallable(lib.fd, data_ )
                        area = quad(func, 0, mp.inf, limit=1000) #, epsabs=1e-12)

                    except Warning as e:
                        print("ERROR in INTEGRAL1!!!!")
                        return 1
                    except Exception:
                        print("ERROR in INTEGRAL2!!!!")
                        return 1


        elif no_untested_UNSEL == 0: 
            #only tested config in SEL

            if no_untested_SEL == 0 :
                #only tested config in UNSEL
                return 0
            else:
                #untested and tested configs in SEL
                #unsel is a dirac
                lib = ctypes.CDLL(os.path.abspath('func.so'))
                lib.fd.restype = ctypes.c_double
                lib.fd.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),  ctypes.c_void_p)

                #array
                size = 3 + 2*len(SEL_BudgetMax) + 1
                data = (ctypes.c_double * size)()

                data[0] = size
                data[1] = len(SEL)
                data[2] = 0

                count = 3
                for i in range(len(SEL_BudgetMax)):
                    data[count] = SEL_BudgetMax[i][2]
                    data[count + 1] = SEL_BudgetMax[i][3]
                    count += 2
                
                data[-1] = Max_tested_UNSEL

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:

                        data_= ctypes.cast(data, ctypes.c_void_p)
                        func = LowLevelCallable(lib.fd, data_ )
                        area = quad(func, 0, mp.inf, limit=1000) #, epsabs=1e-12)

                    except Warning as e:
                        print("ERROR in INTEGRAL3!!!!")
                        return 1
                    except Exception:
                        print("ERROR in INTEGRAL4!!!!")
                        return 1

        else:
            #there tested and untested configs in both sets

            lib = ctypes.CDLL(os.path.abspath('func.so'))
            lib.f.restype = ctypes.c_double
            lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),  ctypes.c_void_p)

            #array
            size = 3 + 2*len(SEL_BudgetMax) + 2*len(UNSEL_BudgetMax)
            data = (ctypes.c_double * size)()
            #DATA  array 
            # [No SEL, No UNSEL,  SEL   ,   UNSEL]
            # [No SEL, No UNSEL, SEL[0][0], SEL[0][1], SEL[1][0], SEL[1][1], UNSEL[0][0], UNSEL[0][1], UNSEL[1][0], UNSEL[1][1]]

            data[0] = size
            data[1] = len(SEL_BudgetMax)
            data[2] = len(UNSEL_BudgetMax)

            count = 3
            for i in range(len(SEL_BudgetMax)):
                data[count] = SEL_BudgetMax[i][2]
                data[count + 1] = SEL_BudgetMax[i][3]
                count += 2

            for i in range(len(UNSEL_BudgetMax)):
                data[count] = UNSEL_BudgetMax[i][2]
                data[count + 1] = UNSEL_BudgetMax[i][3]
                count += 2
            
            data_= ctypes.cast(data, ctypes.c_void_p)
            func = LowLevelCallable(lib.f, data_ )
            opts = {"limit":2500, "epsabs": 1e-6}
            #opts = {"limit":1000} #, "epsabs":1e-12}

            if Max_tested_UNSEL != -1 and Max_tested_SEL == -1:
            #sel has only untested
            #unsel has untested and tested
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        area =  nquad(func, [[Max_tested_UNSEL, np.inf],[0, np.inf]], opts=opts)
                    except Warning as e:
                        print("ERROR in INTEGRAL5!!!!")
                        return 1
                    except Exception:
                        print("ERROR in INTEGRAL6!!!!")
                        return 1
            elif Max_tested_UNSEL == -1 and Max_tested_SEL != -1:
            #unsel has untested
            #sel has untested and tested   
                def bounds_k(x):
                    return [x-Max_tested_SEL, np.inf]

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        area =  nquad(func, [bounds_k ,[0, np.inf]], opts=opts)
                    except Warning as e:
                        print("ERROR in INTEGRAL7!!!!")
                        return 1
                    except Exception:
                        print("ERROR in INTEGRAL8!!!!")
                        return 1

            elif Max_tested_UNSEL != -1 and Max_tested_SEL != -1:
            #unsel has untested and tested
            #sel has untested and tested 
                def bounds_k1(x):
                    if x-Max_tested_SEL >  Max_tested_UNSEL:
                        return [x-Max_tested_SEL, np.inf]
                    else:
                        return [Max_tested_UNSEL, np.inf]
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        area =  nquad(func, [bounds_k1 , [0, np.inf]], opts=opts)
                    except Warning as e:
                        print("ERROR in INTEGRAL9!!!!")
                        return 1
                    except Exception:
                        print("ERROR in INTEGRAL10!!!!")
                        return 1
            else:

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        area =  nquad(func, [[-np.inf, np.inf],[0, np.inf]], opts=opts)        

                    except Warning as e:
                        print("ERROR in INTEGRAL11!!!!")
                        return 1
                    except Exception:
                        print("ERROR in INTEGRAL12!!!!")
                        return 1
        # else:
        #     f1 = lambda k, x: self.fx(k, UNSEL_BudgetMax) * self.fx(k-x, SEL_BudgetMax) * x
        #     opts = {"limit":500, "epsabs":1e-12}
        #     area =  nquad(f1, [[-mp.inf, mp.inf],[0, mp.inf]], opts=opts)
        #     #area =  nquad(f1, [[-mp.inf, mp.inf],[0, mp.inf]])

        return area[0]

    def risk1(self, SEL, UNSEL, stage):
        #not being used
        
        SEL_BudgetMax = copy.deepcopy(SEL)
        for i in range(len(SEL_BudgetMax)):
            c, _ , _, _ = SEL_BudgetMax[i]
            #print(c)
            if self.fullBudget_predictions:
                _, l, s = self.config_sampler.make_predictions([c], self.budgets[-1])[0] #full budget
            else:
                _, l, s = self.config_sampler.make_predictions([c], self.budgets[stage])[0] #current 
                
            SEL_BudgetMax[i][2] = 1 - l[0] #use acc
            if s < 0.05:
                SEL_BudgetMax[i][3] = 0.05
            else:
                SEL_BudgetMax[i][3] = s[0]

        UNSEL_BudgetMax = copy.deepcopy(UNSEL)
        #print("unsel")
        #print(UNSEL_BudgetMax)

        for i in range(len(UNSEL_BudgetMax)):
            c, _ , _, _ = UNSEL_BudgetMax[i]
            #print(c)
            if self.fullBudget_predictions: 
                _, l, s = self.config_sampler.make_predictions([c], self.budgets[-1])[0] #full budget
            else:
                _, l, s = self.config_sampler.make_predictions([c], self.budgets[stage])[0] #current budget
            
            UNSEL_BudgetMax[i][2] = 1 - l[0] #acc
            if s < 0.05:
                UNSEL_BudgetMax[i][3] = 0.05
            else:
                UNSEL_BudgetMax[i][3] = s[0]
        
        #if len(SEL_BudgetMax) == 1 and len(UNSEL_BudgetMax) == 1:
        #    area = self.ExpectedAccuracyLoss(1-SEL_BudgetMax[0][2], SEL_BudgetMax[0][3], 1-UNSEL_BudgetMax[0][2], UNSEL_BudgetMax[0][3] )
        #    return area
        LowLevel = True
        if LowLevel:
            lib = ctypes.CDLL(os.path.abspath('func.so'))
            lib.f.restype = ctypes.c_double
            lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),  ctypes.c_void_p)

            #array
            size = 3 + 2*len(UNSEL_BudgetMax) + 2*len(SEL_BudgetMax)
            data = (ctypes.c_double * size)()
            #DATA  array 
            # [No SEL, No UNSEL,  SEL   ,   UNSEL]
            # [No SEL, No UNSEL, SEL[0][0], SEL[0][1], SEL[1][0], SEL[1][1], UNSEL[0][0], UNSEL[0][1], UNSEL[1][0], UNSEL[1][1]]

            data[0] = size
            data[1] = len(SEL_BudgetMax)
            data[2] = len(UNSEL_BudgetMax)

            count = 3
            for i in range(len(SEL_BudgetMax)):
                data[count] = SEL_BudgetMax[i][2]
                data[count + 1] = SEL_BudgetMax[i][3]
                count += 2
            for i in range(len(UNSEL_BudgetMax)):
                data[count] = UNSEL_BudgetMax[i][2]
                data[count + 1] = UNSEL_BudgetMax[i][3]
                count += 2
            
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    data_   = ctypes.cast(data, ctypes.c_void_p)
                    func = LowLevelCallable(lib.f, data_ )
                    opts = {"limit":500, "epsabs":1e-12}
                    area =  nquad(func, [[-mp.inf, mp.inf],[0, mp.inf]], opts=opts)
                    #area =  nquad(func, [[-np.inf, np.inf],[0, np.inf]], opts=opts)

                except Warning as e:
                    print("ERROR in INTEGRAL1!!!!")
                    return 1
                except Exception:
                    print("ERROR in INTEGRAL2!!!!")
                    return 1
        else:
            f1 = lambda k, x: self.fx(k, UNSEL_BudgetMax) * self.fx(k-x, SEL_BudgetMax) * x
            opts = {"limit":500, "epsabs":1e-12}
            area =  nquad(f1, [[-mp.inf, mp.inf],[0, mp.inf]], opts=opts)
            #area =  nquad(f1, [[-mp.inf, mp.inf],[0, mp.inf]])

        return area[0]




class WarmStartIteration(BaseIteration):
    """
    iteration that imports a privious Result for warm starting
    """

    def __init__(self, Result, config_generator):

        self.is_finished = False
        self.stage = 0

        id2conf = Result.get_id2config_mapping()
        delta_t = - max(map(lambda r: r.time_stamps['finished'], Result.get_all_runs()))

        super().__init__(-1, [len(id2conf)], [None], None)

        for i, id in enumerate(id2conf):
            new_id = self.add_configuration(config=id2conf[id]['config'], config_info=id2conf[id]['config_info'])

            for r in Result.get_runs_by_id(id):

                j = Job(new_id, config=id2conf[id]['config'], budget=r.budget)

                j.result = {'loss': r.loss, 'info': r.info}
                j.error_logs = r.error_logs

                for k, v in r.time_stamps.items():
                    j.timestamps[k] = v + delta_t

                self.register_result(j, skip_sanity_checks=True)

                config_generator.new_result(j, update_model=(i == len(id2conf) - 1))

        # mark as finished, as no more runs should be executed from these runs
        self.is_finished = True

    def fix_timestamps(self, time_ref):
        """
            manipulates internal time stamps such that the last run ends at time 0
        """

        for k, v in self.data.items():
            for kk, vv in v.time_stamps.items():
                for kkk, vvv in vv.items():
                    self.data[k].time_stamps[kk][kkk] += time_ref


########################################################
#this function is being used to compute the risk
########################################################
def risk_parellel(SEL, UNSEL, stage, budgets):
        time_init_int = time.time()
        SEL_BudgetMax = copy.deepcopy(SEL)
        Max_tested_SEL = -1
        no_tested_SEL = 0
        no_untested_SEL = 0

        for i in range(len(SEL_BudgetMax)):
            c, _ , loss, std = SEL_BudgetMax[i]

            if std is None: #tested config
                no_tested_SEL += 1
                acc = 1 - loss
                SEL_BudgetMax[i][2] = acc
                SEL_BudgetMax[i][3] = 0.0
                if acc >  Max_tested_SEL:
                    Max_tested_SEL = acc

            else:
            #print(c)
                no_untested_SEL += 1
                SEL_BudgetMax[i][2] = 1 - loss #acc
                if std < 0.05:
                    SEL_BudgetMax[i][3] = 0.05
                    #self.result_logger.updateStdCounter()
                else:
                    SEL_BudgetMax[i][3] = std


        UNSEL_BudgetMax = copy.deepcopy(UNSEL)
        Max_tested_UNSEL = -1
        no_tested_UNSEL = 0
        no_untested_UNSEL = 0

        for i in range(len(UNSEL_BudgetMax)):
            c, _ , loss, std = UNSEL_BudgetMax[i]
            #print(c)
            if std is None: #tested config
                no_tested_UNSEL += 1
                acc = 1 - loss 
                UNSEL_BudgetMax[i][2] = acc
                UNSEL_BudgetMax[i][3] = 0.0
                if acc >  Max_tested_UNSEL:
                    Max_tested_UNSEL = acc
            
            else:
                no_untested_UNSEL += 1

                UNSEL_BudgetMax[i][2] = 1 - loss #acc
                if std < 0.05:
                    UNSEL_BudgetMax[i][3] = 0.05
                    #self.result_logger.updateStdCounter()
                else:
                    UNSEL_BudgetMax[i][3] = std
        
        #if len(SEL_BudgetMax) == 1 and len(UNSEL_BudgetMax) == 1:
        #    area = self.ExpectedAccuracyLoss(1-SEL_BudgetMax[0][2], SEL_BudgetMax[0][3], 1-UNSEL_BudgetMax[0][2], UNSEL_BudgetMax[0][3] )
        #    return area

        if no_untested_SEL == 0:
            #only tested config in SEL

            if no_untested_UNSEL == 0 :
                #only tested config in UNSEL
                time_final = time.time() - time_init_int
                return 0, time_final
            else:
                #untested and tested configs in UNSEL
                #sel is a dirac

                lib = ctypes.CDLL(os.path.abspath('func.so'))
                lib.fd.restype = ctypes.c_double
                lib.fd.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),  ctypes.c_void_p)

                #array
                size = 3 + 1 + 2*len(UNSEL_BudgetMax)
                data = (ctypes.c_double * size)()
                #DATA  array 
                #DATA  array 
                # [total configs,  No SEL,   No UNSEL,  SEL   ,   UNSEL]
                # [total configs,  0,   No UNSEL,  SEL   ,   UNSEL]
                # [total configs, No SEL, No UNSEL, SEL_max, UNSEL[0][0], UNSEL[0][1], UNSEL[1][0], UNSEL[1][1]]

                data[0] = size
                data[1] = 0
                data[2] = len(UNSEL_BudgetMax)
                data[3] = Max_tested_SEL

                count = 4
                for i in range(len(UNSEL_BudgetMax)):
                    data[count] = UNSEL_BudgetMax[i][2]
                    data[count + 1] = UNSEL_BudgetMax[i][3]
                    count += 2


                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        data_= ctypes.cast(data, ctypes.c_void_p)
                        func = LowLevelCallable(lib.fd, data_ )
                        area = quad(func, 0, mp.inf, limit=1000) #, epsabs=1e-12)

                    except Warning as e:
                        #print("ERROR in INTEGRAL1!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final
                    except Exception:
                        #print("ERROR in INTEGRAL2!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final


        elif no_untested_UNSEL == 0: 
            #only tested config in SEL

            if no_untested_SEL == 0 :
                #only tested config in UNSEL
                time_final = time.time() - time_init_int
                return 0, time_final
            else:
                #untested and tested configs in SEL
                #unsel is a dirac
                lib = ctypes.CDLL(os.path.abspath('func.so'))
                lib.fd.restype = ctypes.c_double
                lib.fd.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),  ctypes.c_void_p)

                #array
                size = 3 + 2*len(SEL_BudgetMax) + 1
                data = (ctypes.c_double * size)()

                data[0] = size
                data[1] = len(SEL)
                data[2] = 0

                count = 3
                for i in range(len(SEL_BudgetMax)):
                    data[count] = SEL_BudgetMax[i][2]
                    data[count + 1] = SEL_BudgetMax[i][3]
                    count += 2
                
                data[-1] = Max_tested_UNSEL

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:

                        data_= ctypes.cast(data, ctypes.c_void_p)
                        func = LowLevelCallable(lib.fd, data_ )
                        area = quad(func, 0, mp.inf, limit=1000) #, epsabs=1e-12)

                    except Warning as e:
                        #print("ERROR in INTEGRAL3!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final
                    except Exception:
                        #print("ERROR in INTEGRAL4!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final

        else:
            #there tested and untested configs in both sets

            lib = ctypes.CDLL(os.path.abspath('func.so'))
            lib.f.restype = ctypes.c_double
            lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double),  ctypes.c_void_p)

            #array
            size = 3 + 2*len(SEL_BudgetMax) + 2*len(UNSEL_BudgetMax)
            data = (ctypes.c_double * size)()
            #DATA  array 
            # [No SEL, No UNSEL,  SEL   ,   UNSEL]
            # [No SEL, No UNSEL, SEL[0][0], SEL[0][1], SEL[1][0], SEL[1][1], UNSEL[0][0], UNSEL[0][1], UNSEL[1][0], UNSEL[1][1]]

            data[0] = size
            data[1] = len(SEL_BudgetMax)
            data[2] = len(UNSEL_BudgetMax)

            count = 3
            for i in range(len(SEL_BudgetMax)):
                data[count] = SEL_BudgetMax[i][2]
                data[count + 1] = SEL_BudgetMax[i][3]
                count += 2

            for i in range(len(UNSEL_BudgetMax)):
                data[count] = UNSEL_BudgetMax[i][2]
                data[count + 1] = UNSEL_BudgetMax[i][3]
                count += 2
            
            data_= ctypes.cast(data, ctypes.c_void_p)
            func = LowLevelCallable(lib.f, data_ )
            opts = {"limit":2500, "epsabs": 1e-6}
            #opts = {"limit":1000} #, "epsabs":1e-12}

            if Max_tested_UNSEL != -1 and Max_tested_SEL == -1:
            #sel has only untested
            #unsel has untested and tested
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        area =  nquad(func, [[Max_tested_UNSEL, np.inf],[0, np.inf]], opts=opts)
                    except Warning as e:
                        #print("ERROR in INTEGRAL5!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final
                    except Exception:
                        #print("ERROR in INTEGRAL6!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final
            elif Max_tested_UNSEL == -1 and Max_tested_SEL != -1:
            #unsel has untested
            #sel has untested and tested   
                def bounds_k(x):
                    return [x-Max_tested_SEL, np.inf]

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        area =  nquad(func, [bounds_k ,[0, np.inf]], opts=opts)
                    except Warning as e:
                        #print("ERROR in INTEGRAL7!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final
                    except Exception:
                        #print("ERROR in INTEGRAL8!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final

            elif Max_tested_UNSEL != -1 and Max_tested_SEL != -1:
            #unsel has untested and tested
            #sel has untested and tested 
                def bounds_k1(x):
                    if x-Max_tested_SEL >  Max_tested_UNSEL:
                        return [x-Max_tested_SEL, np.inf]
                    else:
                        return [Max_tested_UNSEL, np.inf]
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        area =  nquad(func, [bounds_k1 , [0, np.inf]], opts=opts)
                    except Warning as e:
                        #print("ERROR in INTEGRAL9!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final
                    except Exception:
                        #print("ERROR in INTEGRAL10!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final
            else:

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        area =  nquad(func, [[-np.inf, np.inf],[0, np.inf]], opts=opts)        

                    except Warning as e:
                        #print("ERROR in INTEGRAL11!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final
                    except Exception:
                        #print("ERROR in INTEGRAL12!!!!")
                        time_final = time.time() - time_init_int
                        return 1, time_final
        # else:
        #     f1 = lambda k, x: self.fx(k, UNSEL_BudgetMax) * self.fx(k-x, SEL_BudgetMax) * x
        #     opts = {"limit":500, "epsabs":1e-12}
        #     area =  nquad(f1, [[-mp.inf, mp.inf],[0, mp.inf]], opts=opts)
        #     #area =  nquad(f1, [[-mp.inf, mp.inf],[0, mp.inf]])

        time_final = time.time() - time_init_int
        return [SEL, UNSEL, area[0]], time_final