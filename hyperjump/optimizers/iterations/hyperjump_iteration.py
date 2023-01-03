
#import sys
import os, ctypes
import logging
import numpy as np
#import pdb
import math
from scipy.stats import norm
from scipy.integrate import quad, nquad
from scipy import LowLevelCallable
#import itertools
#import multiprocessing
from mpmath import mp
import time
import copy
#import networkx as nx
import warnings, random

from hyperjump.core.dispatcher import Job
#from collections import defaultdict, deque



from hyperjump.optimizers.iterations import SuccessiveHalving
#from hyperjump.core.base_iteration import Datum


path_to_risk_c = 'hyperjump/optimizers/iterations/func.so'


class Datum(object):
    def __init__(self, config, config_info, config_id, results=None, time_stamps=None, exceptions=None, status='QUEUED', budget=0, stage=0):
        self.config = config
        self.config_info = config_info
        self.results = results if not results is None else {}
        self.time_stamps = time_stamps if not time_stamps is None else {}
        self.exceptions = exceptions if not exceptions is None else {}
        self.status = status
        self.budget = budget
        self.config_id = config_id
        self.stage = stage

    def __repr__(self):
        return ( \
                    "\nconfig:{}\n".format(self.config) + \
                    "config_info:\n{}\n" % self.config_info + \
                    "losses:\n"
                    '\t'.join(["{}: {}\t".format(k, v['loss']) for k, v in self.results.items()]) + \
                    "time stamps: {}".format(self.time_stamps)
        )


class BaseIteration_Hyperjump(SuccessiveHalving):
    def __init__(self, HPB_iter, num_configs, budgets, config_sampler, logger=None, result_logger=None, eta=3.0, not_randFrac=1.0, threshold=1.0, pool=None):
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

        self.done_num_configs = [0] * len(num_configs)

        self.prints = True
        self.configs2end = []

        self.thresholdRisk = threshold

        self.SEL_set = None
        self.UNSEL_set = None
        self.time_start = time.time()
        self.notMoreToRun = False

        self.setTestedConfigs = [] # tested configs in current stage
        self.setUntestedConfigs = [] # untested configs in current stage

        super().__init__(HPB_iter, num_configs, budgets, config_sampler, logger, result_logger)

        self.set2Test()

        self.not_randFrac = not_randFrac

        self.pool = pool
        self.eta = eta

        self.logger.info("Running Hyperjump using a threshold of " + str(self.thresholdRisk))


    def set2Test(self):
        no_confs = self.num_configs[self.stage]

        if self.prints:
            self.logger.info("[NEXT BRACKET] ---> Creating config set to test....Loading " + str(no_confs) + " configurations!  \n")

        self.add_set_configurations()


    def add_set_configurations(self):
        """
        function to add a new set configuration to the current iteration

        Parameters
        ----------

        config : valid configuration
            The configuration to add. If None, a configuration is sampled from the config_sampler
        config_info: dict
            Some information about the configuration that will be stored in the results
        """
        # get configs for current bracket
        listConfigs_add =  self.config_sampler.get_listConfigs(self.budgets[self.stage], no_total_configs=self.num_configs[self.stage]) 

        listIDs = []
        listConfigs = []
        for i in range(len(listConfigs_add)):
            config, config_info = listConfigs_add[i]

            if self.is_finished:
                raise RuntimeError("This hyperjump iteration is finished, you can't add more configurations!")

            if self.actual_num_configs[self.stage] == self.num_configs[self.stage]:
                raise RuntimeError("Can't add another configuration to stage %i in hyperjump iteration %i." % (self.stage, self.HPB_iter))

            config_id = (self.HPB_iter, self.stage, self.actual_num_configs[self.stage])
            listIDs.append(config_id)
            listConfigs.append(config)
            self.data[config_id] = Datum(config=config, config_info=config_info, config_id=config_id, budget=self.budgets[self.stage], stage=self.stage)
            
            self.setUntestedConfigs.append([config, config_id, int(self.budgets[self.stage])])

            self.actual_num_configs[self.stage] += 1

            if not self.result_logger is None:
                self.result_logger.new_config(config_id, config, config_info)

        return config_id


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
                raise RuntimeError("Can't add another configuration to stage %i in hyperjump iteration %i." % (self.stage, self.HPB_iter))

            config_id = (self.HPB_iter, self.stage, self.actual_num_configs[self.stage])

            self.data[config_id] = Datum(config=config, config_info=config_info, config_id=config_id, budget=self.budgets[self.stage], stage=self.stage)
            self.setUntestedConfigs.append([config, config_id, int(self.budgets[self.stage])])

            self.actual_num_configs[self.stage] += 1

            if not self.result_logger is None:
                self.result_logger.new_config(config_id, config, config_info)

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

        for ct in range(len(self.budgets)):
            if budget == self.budgets[ct]: 
                self.done_num_configs[ct] += 1
                break

        # add configuration to tested configs
        self.setTestedConfigs.append([config, config_id, budget, d.results[budget]['loss']])


    def get_next_run(self):
        """
        function to return the next configuration and budget to run.

        This function is called from HB_master, don't call this from
        your script.

        It returns None if this run of SH is finished or there are
        pending jobs that need to finish to progress to the next stage.

        If there are empty slots to be filled in the current SH stage
        (which never happens in the original SH version), a new
        configuration will be sampled and scheduled to run next.

        Here check try to evaluate the risk of jumping with the current configs.
        You have: List of sampled configs
        Need to have: List of predicted configs
        
        Start with Hyperband and sample of a SH iteration:
        1 - Fill a list with known configs and their values, and predicted configs and their predicted values
        2 - Sort them
        3 - Get a risk measure of top vs bottom configs (number of top configs = next iteration size)
        4 - Advance stage if risk is under a threshold value
        
        Question: 	How (and when) to advance multiple stages
        """

        if self.is_finished:
            self.logger.info("[ITERATION] Finished.")
            return None

        if self.actual_num_configs[self.stage]+1 == self.num_configs[self.stage] and self.stage == len(self.budgets)-1:
            self.notMoreToRun = True
            return None


        eal = 0
        self.configs2end = []
        self.time_start = time.time()

        self.logger.info("[ITERATION] - " + str(self.done_num_configs[self.stage]) + " configurations tested, " + str(self.actual_num_configs[self.stage]) \
                + " on queue to be tested in a total of " + str(self.num_configs[self.stage]) + " configs")

        # hyperjump 
        # risk model - determine if we jump (if jump, the configs are promoted and discarded automatically by this function)
        self.config_sampler.training()
        eal = self.analyse_risk_new()

        # determine next config to evaluate
        key_, config_, budget_ = self.runNextConfig(eal)

        overhead = time.time() - self.time_start

        if key_ is not None: 
            if self.prints:     
                self.logger.info("[ITERATION] - Running config " + str(self.data[key_].config) + " on budget " + str(self.data[key_].budget) + "!!\n")
           
            # remove configuration to untested configs
            for i in range(len(self.setUntestedConfigs)):
                conf, confId, budg =  self.setUntestedConfigs[i]
                if conf == config_ and confId == key_ and int(budg) == int(budget_):
                    self.setUntestedConfigs.pop(i)
                    break
			
            self.num_running += 1
            return (key_, config_, budget_)
        

        if self.num_running == 0:
            #print("END bracket")
            self.process_results()
            return self.get_next_run()

        return None


    def runNextConfig(self, eal):
        ############################################
        # sort configs to test
        #############################################
        if self.num_configs[self.stage] == self.done_num_configs[self.stage]: 
            # no more to run
            return None, None, None

        if not self.config_sampler.model_available:
            #run a random config
            for k, v in self.data.items():
                if v.status == 'QUEUED':
                    assert v.budget == self.budgets[self.stage], 'Configuration budget does not align with current stage!'
                    v.status = 'RUNNING'
                    #self.num_running += 1

                    return (k, v.config, v.budget)

        else:
        # run == "newRisk_parallel":
            if self.prints:
                self.logger.info("Selecting next config to test.")

            def sortFirst_(val): 
                return val[1] 

            #run the config that is predicted the yield an higher risk reduction 
            untested_configs, untested_configs_id, untested_losses, untested_std = self.get_untested_configs()
            if len(untested_configs) == 0:
                #all configs were tested
                return None, None, None

            if len(untested_configs) == 1:
                 # there is only one configs
                k = untested_configs_id[0]
                self.data[k].status = 'RUNNING'
                #self.num_running += 1
                return (k, self.data[k].config, self.data[k].budget)

                
            #if in the last stage
            if self.stage == len(self.num_configs)-1:
                #run the config with higher improvement
                listConf = []

                #untested configs in unsel
                aux_unsel = []
                for i in range(0, len(untested_configs)):
                    aux_unsel.append([untested_configs[i], untested_configs_id[i], untested_losses[i], untested_std[i]])

                inc, inc_id, incAcc, _= self.result_logger.returnIncumbent()
                if inc is not None:
                    sel = [[inc, inc_id, 1-incAcc, None]]
                    process_list = []
                    #if __name__ == 'hyperjump.core.base_iteration':
                    if __name__ == 'hyperjump.optimizers.iterations.hyperjump_iteration':
                        for i in range(len(aux_unsel)): # only untested configs in unsel                                    
                            unsel = copy.deepcopy(aux_unsel)
                            unsel[i][3] = None

                            result = self.pool.apply_async(risk_parellel, (sel, unsel, self.stage, self.budgets, )) 
                            process_list.append(result)

                        for i in range(len(process_list)):
                            res, time_ = process_list[i].get()
                            if isinstance(res, list):
                                listConf.append([res[1][i][1], res[2]])

                else:
                    for i in range(len(aux_unsel)):
                        listConf.append([aux_unsel[i][1], aux_unsel[i][2]]) #id, loss


                if len(listConf) != 0:
                    listConf.sort(key = sortFirst_) 
                    
                    k, _ = listConf[0]
                    #print("runnung config " +str(self.data[k].config) + "with predicted acc = " + str(1-l)) 
                    self.data[k].status = 'RUNNING'
                    #self.num_running += 1

                    return (k, self.data[k].config, self.data[k].budget)
                else:
                    self.logger.info("No configs to be tested next were selected using the risk")
                    listConf = []
                    for i in range(0, len(untested_configs)):
                        listConf.append([untested_configs_id[i], untested_losses[i]])

                    listConf.sort(key=sortFirst_) 
                    k, _= listConf[0]
                    self.data[k].status = 'RUNNING'
                    #self.num_running += 1

                    return (k, self.data[k].config, self.data[k].budget)            


            ## not in last stage
            if self.SEL_set is None or self.UNSEL_set is None:
                tested_configs, tested_configs_id, tested_losses = self.get_tested_configs()

                all_configs = []
                for i in range(0, len(tested_configs)):
                    all_configs.append([tested_configs[i], tested_configs_id[i], tested_losses[i], None])

                for i in range(0, len(untested_configs)):
                    all_configs.append([untested_configs[i], untested_configs_id[i], untested_losses[i], untested_std[i]])

                m = self.num_configs[self.stage+1] #no of configs to test in next stage
                aux_sel , aux_unsel = self.createSEL(all_configs, m)

            else:
                aux_sel = self.SEL_set
                aux_unsel = self.UNSEL_set

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

            # we have now sel and unsel 

            listConf = []
            process_list = []

            #if __name__ == 'hyperjump.core.base_iteration':
            if __name__ == 'hyperjump.optimizers.iterations.hyperjump_iteration':
                #SIMULATE THE RISK - set the models uncertainty of untested confgis to None
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
                    if isinstance(res, list):
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
                    if isinstance(res, list):
                        listConf.append([res[1][l_aux[i]][1], res[2]])

                self.SEL_set = None
                self.UNSEL_set = None

                if len(listConf) > 0:
                    listConf.sort(key=sortFirst_) 
                    k, _= listConf[0]
                    self.data[k].status = 'RUNNING'
                    #self.num_running += 1

                    return (k, self.data[k].config, self.data[k].budget)
                else:
                    self.logger.info("No configs to be tested next were selected using the risk")
                    listConf = []
                    for i in range(0, len(untested_configs)):
                        listConf.append([untested_configs_id[i], untested_losses[i]])

                    listConf.sort(key=sortFirst_) 
                    k, _= listConf[0]
                    self.data[k].status = 'RUNNING'
                    #self.num_running += 1
                    return (k, self.data[k].config, self.data[k].budget)

        return None, None, None


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

        # collect all config_ids that need to be compared
        currBud = self.budgets[self.stage-1]
        config_ids = list(filter(lambda cid: self.data[cid].status == 'REVIEW', self.data.keys()))

        size_list_confID = len(config_ids) 
        for io in reversed(range(size_list_confID)):
            dio = config_ids[io]
            if self.data[dio].budget < currBud: 
                self.data[dio].status = 'TERMINATED'
                config_ids.pop(io)
 

        if (self.stage >= len(self.num_configs)):
            #last stage - finish bracket
            self.finish_up()
            #self.config_sampler.reset_testedConfig_counter()
            return

        budgets = [self.data[cid].budget for cid in config_ids]

        if len(set(budgets)) > 1:
            raise RuntimeError('Not all configurations have the same budget!')

        budget = self.budgets[self.stage - 1]
        losses = np.array([self.data[cid].results[budget]['loss'] for cid in config_ids])
        advance = self._advance_to_next_stage(config_ids, losses)

        for i, a in enumerate(advance):
            if a:
                self.logger.debug('ITERATION: Advancing config %s to next budget %f' % (config_ids[i], self.budgets[self.stage]))


        for i, cid in enumerate(config_ids):
            if advance[i]:
                self.data[cid].status = 'QUEUED'
                self.data[cid].budget = self.budgets[self.stage]
                self.data[cid].stage = self.stage
                self.actual_num_configs[self.stage] += 1
                self.setUntestedConfigs.append([self.data[cid].config, cid, int(self.budgets[self.stage])])

            else:
                self.data[cid].status = 'TERMINATED'
        
        self.SEL_set = None
        self.UNSEL_set = None


    def finish_up(self):
        self.is_finished = True
        self.notMoreToRun = True

        for k, v in self.data.items():
            #assert v.status in ['TERMINATED', 'REVIEW', 'CRASHED'], 'Configuration has not finshed yet!'
            #if not (v.status == 'TERMINATED' or  v.status =='REVIEW'  or v.status == 'CRASHED'):
            #    print('Configuration has not finshed yet!')
            v.status = 'COMPLETED'


    def __repr__(self):
        raise NotImplementedError('This function needs to be overwritten in %s.' % (self.__class__.__name__))


    def get_tested_configs(self):
        # return a list with tested configs for current stage
        budget = self.budgets[self.stage]
       
        config_ids = list(filter(lambda cid: self.data[cid].status == 'REVIEW' and int(self.data[cid].budget) == int(budget), self.data.keys())) #only the ones tested in the current budget

        list_cid = []
        list_configs = []
        list_losses = []
        for cid in config_ids:
                list_cid.append(cid)
                list_configs.append(self.data[cid].config)
                list_losses.append(self.data[cid].results[budget]['loss'])

        return list_configs, list_cid, list_losses


    def get_untested_configs(self):
        # return a list with untested configs for current stage
        budget = self.budgets[self.stage]

        list_configs = []
        list_config_ids = []
        for k, v in self.data.items():
            if v.status == 'QUEUED':
                list_configs.append(v.config)
                list_config_ids.append(v.config_id)

        list_untested_configs = self.config_sampler.make_predictions(list_configs, budget)

        list_losses = []
        list_stds = []
        for _, untested_losses, untested_std in list_untested_configs:
            list_losses.append(untested_losses)
            list_stds.append(untested_std)

        return list_configs, list_config_ids, list_losses, list_stds


    def createSEL(self, all_configs, noConfSel):
        ################################################
        #       order configs to sel
        ################################################

        def sortSecond(val): 
            return val[2] 

        ealMatrix = []
        process_list = []
        all_configs.sort(key = sortSecond) #sort configs by loss

        sel = all_configs[0:noConfSel]  #predicted configs to transit to the next stage
        unsel = all_configs[noConfSel:] #predicted configs to be dropped in the current stage

        #if __name__ == 'hyperjump.core.base_iteration':
        if __name__ == 'hyperjump.optimizers.iterations.hyperjump_iteration':
            result = self.pool.apply_async(risk_parellel, (sel, unsel, self.stage, self.budgets,)) 
            process_list.append(result)

            no_stages = len(self.num_configs)
            for i in range(1, no_stages):
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
                if isinstance(res, list):
                    ealMatrix.append([res[0], res[1], res[2]])


        if len(ealMatrix) > 0:
            ealMatrix.sort(key = sortSecond) #sort configs by loss

            SEL = ealMatrix[0][0]  #predicted configs to transit to the next stage
            UNSEL = ealMatrix[0][1] #predicted configs to be dropped in the current stage

        else:
            all_configs.sort(key = sortSecond) #sort configs by loss

            SEL = all_configs[0:noConfSel]  #predicted configs to transit to the next stage
            UNSEL = all_configs[noConfSel:] #predicted configs to be dropped in the current stage

        return SEL, UNSEL


    def process_results_to_jump(self, targetStage, SEL, UNSEL):
        """
        function that is called when it is predicted that we should jump for a taget budget

        Terminate configs that were queded or not finished
        SEL configs to continue to the next stage
        """
        #print("stopping configs:")
        for config,cid, _, _ in UNSEL:
            self.data[cid].status = 'TERMINATED'
            self.configs2end.append([config, cid, self.budgets[self.stage]])
            
            for i in range(len(self.setUntestedConfigs)):
                conf, confId, budg =  self.setUntestedConfigs[i]
                if conf == config and confId == cid and int(budg) == int(self.budgets[self.stage]):
                    self.setUntestedConfigs.pop(i)
                    break

        self.stage = targetStage # next stage jump

        for config,cid, _, _ in SEL:
            self.data[cid].status = 'QUEUED'
            self.data[cid].budget = self.budgets[self.stage]
            self.data[cid].stage = self.stage
            self.actual_num_configs[self.stage] += 1

            self.setUntestedConfigs.append([self.data[cid].config, cid, int(self.budgets[self.stage])])


            self.logger.debug('ITERATION: Advancing config %s to next budget %f' % (config, self.budgets[self.stage]))


    def analyse_risk_new(self):
        # compute the expected risk/loss reduction 
        # and if above a given threshold it jumps 

        if not self.config_sampler.model_available:
            if self.prints:
                self.logger.info("[HYPERJUMP] ---> ERROR: no models\n")
            return -1

        firstSel = None
        firstUnsel = None

        #if in the last stage -> not run hyperjump
        if self.stage == len(self.num_configs)-1:

            untested_configs, untested_configs_id , untested_losses, untested_std = self.get_untested_configs()
            if not untested_configs:
                #all configs were tested
                return -1
              

            inc, inc_id, incAcc, _= self.result_logger.returnIncumbent()
            if inc is not None:
                inc_loss = 1-incAcc
                SEL = [[inc, inc_id, inc_loss, None]]

                aux_unsel = []
                for i in range(0, len(untested_configs)):
                    aux_unsel.append([untested_configs[i], untested_configs_id[i], untested_losses[i], untested_std[i]])
                
                UNSEL = []
                for i in range(0, len(aux_unsel)):
                    UNSEL.append([aux_unsel[i][0], aux_unsel[i][1], aux_unsel[i][2], aux_unsel[i][3]])

                eal = self.risk(SEL, UNSEL, self.stage) 

                T = self.thresholdRisk * (inc_loss)
                
                if self.prints:
                    self.logger.info(" ---> rEAR = " + str(eal) + ", threshold=" + str(T) + ", lenSEL=" + str(len(SEL)) + " lenUNSEL=" + str(len(UNSEL)))
                

                risk_of_using_the_model = 1-self.not_randFrac #* inc_loss * eal
                rand_value = random.uniform(0, 1)
                if rand_value  < risk_of_using_the_model:
                    self.logger.info("RISK OF USING THE MODEL HIGH rand" + str(rand_value) + " risk_model " + str(risk_of_using_the_model))

                if eal < T and  rand_value  > risk_of_using_the_model:
                    #jump to next bracker -> SEL = []  UNSEL = all_configs
                    no_confs_jump = len(UNSEL)
                    self.process_results_to_jump(self.stage, [], UNSEL)

                    self.SEL_set = None
                    self.UNSEL_set = None

                else:
                    #no jump
                    self.SEL_set = copy.deepcopy(SEL)
                    self.UNSEL_set = copy.deepcopy(UNSEL)

                return [(eal, T)]

            else:    #no jump continure
                self.SEL_set = None
                self.UNSEL_set = None   

            return -1


        # return the stage to jump or the current stage if not to jump
        tested_configs, tested_configs_id, tested_losses = self.get_tested_configs()
        if not tested_configs:
            return -1
     
        untested_configs, untested_configs_id, untested_losses, untested_std = self.get_untested_configs()
        if not untested_configs:
            return -1

        all_configs = []     
        for i in range(0, len(tested_configs)):
            all_configs.append([tested_configs[i], tested_configs_id[i], tested_losses[i], None])

        for i in range(0, len(untested_configs)):
            all_configs.append([untested_configs[i], untested_configs_id[i], untested_losses[i], untested_std[i]])
                    
        eal_list = [] 
        prevUNSEL = []
        prevSEL = []
        EAL = 0

        for targetStage in range (self.stage + 1, len(self.num_configs)):

            m = self.num_configs[targetStage] #no of configs to test in next stage

            SEL, UNSEL = self.createSEL(all_configs, m)
            self.SEL_set = copy.deepcopy(SEL)
            self.UNSEL_set = copy.deepcopy(UNSEL)


            # initial sel and unsel sets before jumping
            if firstSel is None:
                firstSel = copy.deepcopy(SEL)
            if firstUnsel is None:
                firstUnsel = copy.deepcopy(UNSEL)

            EAL += self.risk(SEL, UNSEL, targetStage-1) 

            inc, inc_id, incAcc, _= self.result_logger.returnIncumbent()
            if inc is not None:
                inc_loss = 1.0-incAcc
            else:
                inc_loss = 1.0

            T = self.thresholdRisk * (inc_loss)
            eal_list.append((EAL, T))

            if self.prints:
                self.logger.info(" ---> EAL = " + str(EAL) + ", threshold=" + str(T) + ", lenSEL=" + str(len(SEL)) + ", lenUNSEL=" + str(len(UNSEL)))

            if EAL >= T:
                #not to jump to this target stage -> return the previous stage
                risk_of_using_the_model = 1-self.not_randFrac #* inc_loss * EAL
                rand_value = random.uniform(0, 1)

                if rand_value  < risk_of_using_the_model:
                    self.logger.info("RISK OF USING THE MODEL HIGH rand" + str(rand_value) + " risk_model " + str(risk_of_using_the_model))

                if targetStage-1 > self.stage and rand_value > risk_of_using_the_model: # if jump -> process results
                    if self.prints:
                        self.logger.info("[HYPERJUMP]---> EVALUATING RISK ---> Jump to budget " + str(self.budgets[targetStage-1]) + "\n")
    
                    self.process_results_to_jump(targetStage-1, prevSEL, prevUNSEL)

                else:
                    if self.prints:
                        self.logger.info("[HYPERJUMP]---> EVALUATING RISK ---> NOT JUMP\n")

                break 

            else:
                if targetStage ==  len(self.num_configs)-1:
                    # last stage
                    prevUNSEL += UNSEL
                    
                    inc, inc_id, incAcc, _ = self.result_logger.returnIncumbent()
                    if inc is not None:
                        inc_loss = 1-incAcc
                        sel = [[inc, inc_id, inc_loss, None]]

                        unsel = []
                        for i in range(0, len(SEL)):
                            unsel.append([SEL[i][0], SEL[i][1], SEL[i][2], SEL[i][3]])

                        EAL += self.risk(sel, unsel, targetStage) 

                        T = self.thresholdRisk * inc_loss
                        eal_list.append((EAL, T))

                        if self.prints:
                            self.logger.info("---> EAL = " + str(EAL) + ", threshold=" + str(T) + ", lenSEL0" + str(len(sel)) + ", lenUNSEL0" + str(len(unsel)))
                        
                        risk_of_using_the_model = 1-self.not_randFrac #* inc_loss * EAL
                        rand_value = random.uniform(0, 1)
                        
                        if rand_value  < risk_of_using_the_model:
                            self.logger.info("RISK OF USING THE MODEL HIGH rand" + str(rand_value) + " risk_model " + str(risk_of_using_the_model))
                            

                        if EAL < T and  rand_value  > risk_of_using_the_model: # skip last stage
                        #jump to next bracket -> SEL = []  UNSEL = all_configs
                            prevUNSEL += SEL
                            self.process_results_to_jump(targetStage, [], prevUNSEL)

                            self.SEL_set = None
                            self.UNSEL_set = None
                            
                            if self.prints:
                                self.logger.info("[HYPERJUMP]---> EVALUATING RISK ---> Jump to next bracket \n")
                        
                            return eal_list

                        self.SEL_set = copy.deepcopy(sel)
                        self.UNSEL_set = copy.deepcopy(unsel)
                    else:
                        self.SEL_set = None
                        self.UNSEL_set = None

                    # jump to the last stage
                    if self.prints:
                        self.logger.info("[HYPERJUMP]---> EVALUATING RISK ---> Jump to last budget " + str(self.budgets[targetStage]) + "\n")
                        
                    self.process_results_to_jump(targetStage, SEL, prevUNSEL)
                    return eal_list

                else:
                    #continue to the next stage/budget
                    prevSEL = copy.deepcopy(SEL)
                    prevUNSEL += UNSEL

                    _allConfigs  = []
                    _allConfigs_ids = []
                    for conf ,conf_id , _, _ in SEL:
                        _allConfigs.append(conf)
                        _allConfigs_ids.append(conf_id)

                    currentBudget = self.budgets[targetStage]
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


    def risk(self, SEL, UNSEL, stage):

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
                no_untested_SEL += 1
                _, l, s = self.config_sampler.make_predictions([c], self.budgets[stage])[0] #current 
                    
                SEL_BudgetMax[i][2] = 1 - l[0] #acc
                if s < 0.05:
                    SEL_BudgetMax[i][3] = 0.05
                else:
                    SEL_BudgetMax[i][3] = s[0]

        UNSEL_BudgetMax = copy.deepcopy(UNSEL)
        Max_tested_UNSEL = -1
        no_tested_UNSEL = 0
        no_untested_UNSEL = 0

        for i in range(len(UNSEL_BudgetMax)):
            c, _ , loss, std = UNSEL_BudgetMax[i]
            if std is None: #tested config
                no_tested_UNSEL += 1
                acc = 1 - loss 
                UNSEL_BudgetMax[i][2] = acc
                UNSEL_BudgetMax[i][3] = 0.0
                if acc >  Max_tested_UNSEL:
                    Max_tested_UNSEL = acc
            
            else:
                no_untested_UNSEL += 1
                _, l, s = self.config_sampler.make_predictions([c], self.budgets[stage])[0] #current budget

                UNSEL_BudgetMax[i][2] = 1 - l[0] #acc
                if s < 0.05:
                    UNSEL_BudgetMax[i][3] = 0.05
                else:
                    UNSEL_BudgetMax[i][3] = s[0]
        


        if no_untested_SEL == 0:
            #only tested config in SEL

            if no_untested_UNSEL == 0 :
                #only tested config in UNSEL
                return 0
            else:
                #untested and tested configs in UNSEL
                #sel is a dirac

                lib = ctypes.CDLL(os.path.abspath(path_to_risk_c))
                #lib = ctypes.CDLL(os.path.abspath('func.so'))
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
                        self.logger.warning("ERROR in INTEGRAL1!!!!")
                        return 1
                    except Exception:
                        self.logger.warning("ERROR in INTEGRAL2!!!!")
                        return 1


        elif no_untested_UNSEL == 0: 
            #only tested config in SEL

            if no_untested_SEL == 0 :
                #only tested config in UNSEL
                return 0
            else:
                #untested and tested configs in SEL
                #unsel is a dirac
                lib = ctypes.CDLL(os.path.abspath(path_to_risk_c))
                #lib = ctypes.CDLL(os.path.abspath('func.so'))
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
                        self.logger.warning("ERROR in INTEGRAL3!!!!")
                        return 1
                    except Exception:
                        self.logger.warning("ERROR in INTEGRAL4!!!!")
                        return 1

        else:
            #there tested and untested configs in both sets

            lib = ctypes.CDLL(os.path.abspath(path_to_risk_c))
            #lib = ctypes.CDLL(os.path.abspath('func.so'))
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
                        self.logger.warning("ERROR in INTEGRAL5!!!!")
                        return 1
                    except Exception:
                        self.logger.warning("ERROR in INTEGRAL6!!!!")
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
                        self.logger.warning("ERROR in INTEGRAL7!!!!")
                        return 1
                    except Exception:
                        self.logger.warning("ERROR in INTEGRAL8!!!!")
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
                        self.logger.warning("ERROR in INTEGRAL9!!!!")
                        return 1
                    except Exception:
                        self.logger.warning("ERROR in INTEGRAL10!!!!")
                        return 1
            else:

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        area =  nquad(func, [[-np.inf, np.inf],[0, np.inf]], opts=opts)        

                    except Warning as e:
                        self.logger.warning("ERROR in INTEGRAL11!!!!")
                        return 1
                    except Exception:
                        self.logger.warning("ERROR in INTEGRAL12!!!!")
                        return 1

        return area[0]


def risk_parellel(SEL, UNSEL, stage, budgets):
    # this fucntion is used to compute the risk in parallel
    # but only difference for self.risk is that in this one the models predictions are given as input
    # while in  self.risk we used the model to predict untested configs inside the function
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
                else:
                    UNSEL_BudgetMax[i][3] = std
        

        if no_untested_SEL == 0:
            #only tested config in SEL
            if no_untested_UNSEL == 0 :
                #only tested config in UNSEL
                time_final = time.time() - time_init_int
                return 0, time_final
            else:
                #untested and tested configs in UNSEL
                #sel is a dirac

                lib = ctypes.CDLL(os.path.abspath(path_to_risk_c))
                #lib = ctypes.CDLL(os.path.abspath('func.so'))
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
                lib = ctypes.CDLL(os.path.abspath(path_to_risk_c))
                #lib = ctypes.CDLL(os.path.abspath('func.so'))
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

            lib = ctypes.CDLL(os.path.abspath(path_to_risk_c))
            #lib = ctypes.CDLL(os.path.abspath('func.so'))
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

        time_final = time.time() - time_init_int
        return [SEL, UNSEL, area[0]], time_final