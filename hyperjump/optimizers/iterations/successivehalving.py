from hyperjump.core.base_iteration import BaseIteration
import numpy as np


class SuccessiveHalving(BaseIteration):

	def _advance_to_next_stage(self, config_ids, losses):
		"""
			SuccessiveHalving simply continues the best based on the current loss.
		"""
		ranks = np.argsort(np.argsort(losses))
		return(ranks < self.num_configs[self.stage])


	def __repr__(self):

		return ("[SuccessiveHalving Iteration]"+
		"\n\tIs the run finished?: " + str(self.is_finished)+
		"\n\tHB Iteration: " + str(self.HPB_iter)+
		"\n\tStage: " + str(self.stage)+
		"\n\tBudgets: " + str(self.budgets)+
		"\n\tNumber of configs: " + str(self.num_configs)+
		"\n\tActual Number of cfgs: " + str(self.actual_num_configs)+
		"\n\tWorkers running: " + str(self.num_running)+
		"\n\tData: " + str(self.data))