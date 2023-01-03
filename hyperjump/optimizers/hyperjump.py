import os
import time
import math
import copy
import logging

import numpy as np
import multiprocessing

import ConfigSpace as CS

from hyperjump.core.master import Master
#from hyperjump.optimizers.iterations import SuccessiveHalving
from hyperjump.optimizers.iterations import BaseIteration_Hyperjump 
from hyperjump.optimizers.config_generators.hyperjump import Hyperjump as CG_HYPERJUMP


class Hyperjump(Master):
	def __init__(self, configspace = None,
					eta=3, min_budget=0.01, max_budget=1,
					min_points_in_model = None,
					num_samples = 64, random_fraction=1/3, threshold=1.0,
					**kwargs ):
		"""
		Parameters
		----------
		configspace: ConfigSpace object
			valid representation of the search space
		eta : float
			In each iteration, a complete run of sequential halving is executed. In it,
			after evaluating each configuration on the same subset size, only a fraction of
			1/eta of them 'advances' to the next round.
			Must be greater or equal to 2.
		min_budget : float
			The smallest budget to consider. Needs to be positive!
		max_budget : float
			The largest budget to consider. Needs to be larger than min_budget!
			The budgets will be geometrically distributed
                        :math:`a^2 + b^2 = c^2 \sim \eta^k` for :math:`k\in [0, 1, ... , num\_subsets - 1]`.
		min_points_in_model: int
			number of observations to start building a KDE. Default 'None' means
			dim+1, the bare minimum.
		num_samples: int
			number of samples to optimize EI (default 64)
		random_fraction: float
			fraction of purely random configurations that are sampled from the
			prior without the model.
		iteration_kwargs: dict
			kwargs to be added to the instantiation of each iteration
		"""



		# TODO: Propper check for ConfigSpace object!
		if configspace is None:
			raise ValueError("You have to provide a valid CofigSpace object")

		# Hyperband related stuff
		self.eta = eta
		self.min_budget = min_budget
		self.max_budget = max_budget
		self.threshold = threshold

		# precompute some HB stuff
		self.max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
		self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter-1, 0, self.max_SH_iter))

		cg = CG_HYPERJUMP( configspace = configspace,
					min_points_in_model = min_points_in_model,
					num_samples = num_samples,
					random_fraction=random_fraction,
					min_budget=min_budget, 
					max_budget=max_budget,

					)

		super().__init__(config_generator=cg, **kwargs)

		self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)  

		self.config.update({
						'eta'        : eta,
						'min_budget' : min_budget,
						'max_budget' : max_budget,
						'budgets'    : self.budgets,
						'max_SH_iter': self.max_SH_iter,
						'min_points_in_model' : min_points_in_model,
						'num_samples' : num_samples,
						'random_fraction' : random_fraction,
						'threshold' : threshold,
					})

	def get_next_iteration(self, iteration, iteration_kwargs={}, hyperjump=False):
		"""
		BO-HB uses (just like Hyperband) SuccessiveHalving for each iteration.
		See Li et al. (2016) for reference.
		
		Parameters
		----------
			iteration: int
				the index of the iteration to be instantiated

		Returns
		-------
			SuccessiveHalving: the SuccessiveHalving iteration with the
				corresponding number of configurations
		"""
		
		# number of 'SH runs'
		s = self.max_SH_iter - 1 - (iteration%self.max_SH_iter)
		# number of configurations in that bracket
		n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
		ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]

		res = (BaseIteration_Hyperjump(HPB_iter=iteration, num_configs=ns, budgets=self.budgets[(-s-1):], config_sampler=self.config_generator, eta=self.eta, threshold=self.threshold,  pool=self.pool, **iteration_kwargs))

		return res