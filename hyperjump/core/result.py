import copy
import os
import json
from re import search

from hyperjump.core.base_iteration import  Datum
#from hyperjump.optimizers.utils.tools import conf_to_vector as conf_to_vector


class csv_results(object):
	def __init__(self, directory, seed, max_budget,hyperjump=False, overwrite=False):
		os.makedirs(directory, exist_ok=True)

		self.file_logs = os.path.join(directory, 'logs.csv')
		try:
			with open(self.file_logs, 'x') as fh: pass
		except FileExistsError:
			if overwrite:
				with open(self.file_logs, 'w') as fh: pass
			else:
				raise FileExistsError('The file %s already exists.'%self.file_logs)
		except:
			print("ERROR: csv_results 1")
			raise
		
		with open(self.file_logs, 'a') as fh:
			fh.write("runID;it;incumbent;incTime;incAcc;incCost;budget;configTested;Time;Acc;Cost;Overhead;CumulativeCost;Option;EAL;simulatedCost;stdCounter;BestIfHB;BrackInfo;timeRisk;timeOrder;timeTrain;integralTime;infoJump\n")

		#print(directory)
		if "unet" in directory:
			self.type_exp = "unet"
		elif "mnist" in directory:
			self.type_exp = "mnist"
		elif "svm" in directory:
			self.type_exp = "svm"	
		elif "all" in directory:
			self.type_exp = "fake_all"	
		elif "time" in directory:
			if "all" in directory:
				self.type_exp = "fake_time_all"	
			else:
				self.type_exp = "fake_time"	
		else:
			self.type_exp = "fake"

		self.config_ids = set()
		self.seed = seed
		self.testedSet = []
		self.trainingSet = []
		self.hyperjump = hyperjump

		self.inc = None
		self.inc_id = None
		self.incTime = -1
		self.incAcc = -1
		self.incCost = -1
		self.bestAcc = 0.0
		self.threshold = -1

		self.max_budget = max_budget
		self.it = 1
		self.hyperjump = False
		self.EAL = dict()
		self.EAL_ = None
		self.option = "c" #c->continue , j->jump, n->next bracket
		self.counnter_appro_std = 0

		self.infoJump = ""
		self.maxSelJump =  ""
		self.BrackInfoConf = ""
		self.maxHB = ""
		#self.SearchSpace = dict()
		self.SearchSpace = []
		self.timeRisk = -1
		self.timeTestOrder = -1
		self.training_model = -1
		self.integralTime = []

	def updateSearchSpace(self, list_configs):

		for c, acc, cost, time  in list_configs:
			#self.SearchSpace[c] = (acc, cost, time)
			self.SearchSpace.append([c, acc, cost, time])

	def new_config(self, config_id, config, config_info, overhead=0):
		self.testedSet.append([config_id, config, config_info, overhead])
		self.config_ids.add(config_id)

	def updateStdCounter(self):
		self.counnter_appro_std += 1

	def updateBestseen_lastBracket(self, acc):
		self.bestAcc = acc

	def returnBestseen_lastBracket(self):
		return self.bestAcc

	def updateTheshold(self, t):
		self.threshold = t
	
	def returnThreshold(self):
		return self.threshold

	def update_EAL(self, config_id, eal):
		self.hyperjump = True
		if config_id in self.config_ids:
			self.EAL[config_id] = eal
		else:
			print("ERROR: write csv -> different configs")


	def update_EAL_new(self, eal):
		self.hyperjump = True

		if self.EAL_ is None:
			if isinstance(eal, list):
				eal_str = ""
				for i in range(0, len(eal)):
					tup = eal[i]
					eal_str += str(tup[0]) + ":" + str(tup[1]) + ","
				self.EAL_ =  eal_str
			else:
				self.EAL_ =  "-1"

		else:
			if isinstance(eal, list):
				eal_str = ""
				for i in range(0, len(eal)):
					tup = eal[i]
					eal_str += str(tup[0]) + ":" + str(tup[1]) + ","
				self.EAL_ += ( "|" + eal_str)
			else:
				self.EAL_ += ( "|-1")


	def updateOption(self, option):
		self.option = option

	def returnIncumbent(self):
		return self.inc, self.inc_id, self.incAcc, self.incCost, self.incTime 

	def returnSearchSpace(self):
		return self.SearchSpace


	def updateTimeOverhead(self, riskTime, testTime, training_model):
		self.timeRisk = riskTime
		self.timeTestOrder = testTime
		self.training_model = training_model

	def updateIntegralTime(self, int_time):
		self.integralTime.append(int_time)

	def updateOverhead(self, id, overhead):
		for i in range(len(self.testedSet)):
			con = self.testedSet[i]
			if con[0] == id:
				con[3] = overhead

	def jumpInfo(self, Sel, Unsel, budget, target_budget, all_budgets, numConfigs_stage):
		if self.type_exp == "nmist" or self.type_exp == "svm":
			return

		list_data = []
		list_data_sel = []

		str_sel = "["
		for c,_,_,_ in Sel:
			#print(c)
			if self.type_exp == "unet":
				conf_dict = dict([
					('vm_flavor', c["Flavor"]),
					('batch_size', int(c["batch"])),
					('learning_rate', c["learningRate"]),
					('momentum', float(c["momentum"])),
					('nrWorker', int(c["nrWorker"])),
					('synchronism', c["sync"]),
					('budget', int(budget))])

			elif self.type_exp == "fake_all" or self.type_exp == "fake_time_all":
				conf_dict = dict([
					('vm_flavor', c["vm_flavor"]),
					('batch_size', int(c["batch_size"])),
					('learning_rate', c["learning_rate"]),
					('num_cores', int(c["num_cores"])),
					('synchronism', c["synchronism"]),
					('network', c["network"]),
					('budget', int(budget))])			

			else:
				conf_dict = dict([
					('vm_flavor', c["vm_flavor"]),
					('batch_size', int(c["batch_size"])),
					('learning_rate', c["learning_rate"]),
					('num_cores', int(c["num_cores"])),
					('synchronism', c["synchronism"]),
					('budget', int(budget))])

			time_c = None
			acc_c = None
			cost_c = None

			for c_i, c_acc, c_cost, c_time in self.SearchSpace:
				#print(c_i)
				#print(conf_dict)
				#print()
				if conf_dict == c_i:
					time_c = c_time
					acc_c = c_acc
					cost_c = c_cost
					#print("dsadsadsad1")
					break
				
			
			if time_c is None or time_c is None or time_c is None:
				print("ERROR in the dictConfigs unsel")
				return

			list_data.append([conf_dict, time_c, acc_c, cost_c])
			list_data_sel.append([conf_dict, time_c, acc_c, cost_c])
			str_sel += (str(conf_dict) + "," + str(time_c) + "," + str(acc_c) + "," + str(cost_c) + ":")
		str_sel+= "]"

		str_unsel = "["
		for c,_,_,_ in Unsel:
			#print(c)
			if self.type_exp == "unet":
				conf_dict = dict([
					('vm_flavor', c["Flavor"]),
					('batch_size', int(c["batch"])),
					('learning_rate', c["learningRate"]),
					('momentum', float(c["momentum"])),
					('nrWorker', int(c["nrWorker"])),
					('synchronism', c["sync"]),
					('budget', int(budget))])

			elif self.type_exp == "fake_all" or self.type_exp == "fake_time_all":
				conf_dict = dict([
					('vm_flavor', c["vm_flavor"]),
					('batch_size', int(c["batch_size"])),
					('learning_rate', c["learning_rate"]),
					('num_cores', int(c["num_cores"])),
					('synchronism', c["synchronism"]),
					('network', c["network"]),
					('budget', int(budget))])	

			else:
				conf_dict = dict([
					('vm_flavor', c["vm_flavor"]),
					('batch_size', int(c["batch_size"])),
					('learning_rate', c["learning_rate"]),
					('num_cores', int(c["num_cores"])),
					('synchronism', c["synchronism"]),
					('budget', int(budget))])

			time_c = None
			acc_c = None
			cost_c = None

			for c_i, c_acc, c_cost, c_time in self.SearchSpace:
				#print(c_i)
				#print(conf_dict)
				#print()
				if conf_dict == c_i:
					time_c = c_time
					acc_c = c_acc
					cost_c = c_cost
					#print("dsadsadsad22")
					break
				
			if time_c is None or time_c is None or time_c is None:
				print("ERROR in the dictConfigs unsel")
				return
				
			list_data.append([conf_dict, time_c, acc_c, cost_c])
			str_unsel += (str(conf_dict) + "," + str(time_c) + "," + str(acc_c) + "," + str(cost_c) + ":")
		str_unsel+= "]"

		#self.infoJump += (str_sel + str_unsel + "|")

		def sortSecond(val): 
			return val[2] 

		maxHB_notJump = ""
		for ct_bud in range(len(all_budgets)):
			#print(len(list_data))		
			#print(all_budgets[ct_bud])		
			if all_budgets[ct_bud] < budget: #smaller budgets
				if all_budgets[ct_bud] == budget and len(list_data) != numConfigs_stage[ct_bud]: 
					print("ERROR: when verifying the budgets")
				continue
			
			elif all_budgets[ct_bud] == target_budget:
			#elif ct_bud == len(all_budgets)-1: # all_budgets[-1] == budget: #last budget

				list_data.sort(key = sortSecond, reverse=True) 
				maxHB_notJump += ("[" + str(list_data[0][0]) + "," + str(list_data[0][1]) + "," + str(list_data[0][2]) + "," + str(list_data[0][3]) + "," + str(target_budget) + "]")
				break

			elif all_budgets[ct_bud] > target_budget:
				print("It should not arrive to this point")
				break

			else:
				list_data.sort(key = sortSecond, reverse=True) 
				list_data = list_data[0:numConfigs_stage[ct_bud+1]]

				for i in range(len(list_data)):
					list_data[i][0]['budget'] = all_budgets[ct_bud+1]
					for c_i, c_acc, c_cost, c_time in self.SearchSpace:
						if list_data[i][0] == c_i:
							list_data[i][1] = c_time
							list_data[i][2] = c_acc
							list_data[i][3] = c_cost
							break

		maxSel_Jump = ""
		for ct_bud in range(len(all_budgets)):
			#print(len(list_data))		
			#print(all_budgets[ct_bud])		
			if all_budgets[ct_bud] < budget: #smaller budgets
				if all_budgets[ct_bud] == budget and len(list_data_sel) != numConfigs_stage[ct_bud]: 
					print("ERROR: when verifying the budgets")
				continue

			elif all_budgets[ct_bud] == target_budget:
			#elif ct_bud == len(all_budgets)-1: # all_budgets[-1] == budget: #last budget

				list_data_sel.sort(key = sortSecond, reverse=True) 
				maxSel_Jump += ("[" + str(list_data_sel[0][0]) + "," + str(list_data_sel[0][1]) + "," + str(list_data_sel[0][2]) + "," + str(list_data_sel[0][3]) + "]")


			elif all_budgets[ct_bud] > target_budget:
				print("It should not arrive to this point2")
				break

			else:
				list_data_sel.sort(key = sortSecond, reverse=True) 
				list_data_sel = list_data_sel[0:numConfigs_stage[ct_bud+1]]

				for i in range(len(list_data_sel)):
					list_data_sel[i][0]['budget'] = all_budgets[ct_bud+1]
					for c_i, c_acc, c_cost, c_time in self.SearchSpace:
						if list_data_sel[i][0] == c_i:
							list_data_sel[i][1] = c_time
							list_data_sel[i][2] = c_acc
							list_data_sel[i][3] = c_cost
							break

		self.infoJump += (str_sel + str_unsel + "|" + maxHB_notJump + "|" + maxSel_Jump)



	def bracketInfo(self, configs, budget, all_budgets, numConfigs_stage):
		if self.type_exp == "nmist" or self.type_exp == "svm":
			return

		str_sel = "["
		list_data = []
		for c in configs:
			#print(c)
			if self.type_exp == "unet":
				conf_dict = dict([
					('vm_flavor', c["Flavor"]),
					('batch_size', int(c["batch"])),
					('learning_rate', c["learningRate"]),
					('momentum', float(c["momentum"])),
					('nrWorker', int(c["nrWorker"])),
					('synchronism', c["sync"]),
					('budget', int(budget))])

			elif self.type_exp == "fake_all" or self.type_exp == "fake_time_all":
				conf_dict = dict([
					('vm_flavor', c["vm_flavor"]),
					('batch_size', int(c["batch_size"])),
					('learning_rate', c["learning_rate"]),
					('num_cores', int(c["num_cores"])),
					('synchronism', c["synchronism"]),
					('network', c["network"]),
					('budget', int(budget))])	

			else:
				conf_dict = dict([
					('vm_flavor', c["vm_flavor"]),
					('batch_size', int(c["batch_size"])),
					('learning_rate', c["learning_rate"]),
					('num_cores', int(c["num_cores"])),
					('synchronism', c["synchronism"]),
					('budget', int(budget))])

			time_c = None
			acc_c = None
			cost_c = None

			for c_i, c_acc, c_cost, c_time in self.SearchSpace:
				#print(c_i)
				#print(conf_dict)
				#print()
				if conf_dict == c_i:
					time_c = c_time
					acc_c = c_acc
					cost_c = c_cost
					#print("dsadsadsad1")
					break
			
			if time_c is None or time_c is None or time_c is None:
				print("ERROR in the dictConfigs unsel")
				return

			list_data.append([conf_dict, time_c, acc_c, cost_c])
			str_sel += (str(conf_dict) + "," + str(time_c) + "," + str(acc_c) + "," + str(cost_c) + ":")
		str_sel+= "]"

		self.BrackInfoConf += (str_sel + "|")
	
		def sortSecond(val): 
			return val[2] 

		for ct_bud in range(len(all_budgets)):
			#print(len(list_data))		
			#print(all_budgets[ct_bud])		
			if all_budgets[ct_bud] < budget: #smaller budgets
				if all_budgets[ct_bud] == budget and len(list_data) != numConfigs_stage[ct_bud]: 
					print("ERROR: when verifying the budgets")
				continue

			elif ct_bud == len(all_budgets)-1: # all_budgets[-1] == budget: #last budget

				list_data.sort(key = sortSecond, reverse=True) 
				self.maxHB += ("[" + str(list_data[0][0]) + "," + str(list_data[0][1]) + "," + str(list_data[0][2]) + "," + str(list_data[0][3]) + "]")

			else:

				list_data.sort(key = sortSecond, reverse=True) 
				list_data = list_data[0:numConfigs_stage[ct_bud+1]]

				for i in range(len(list_data)):
					list_data[i][0]['budget'] = all_budgets[ct_bud+1]
					for c_i, c_acc, c_cost, c_time in self.SearchSpace:
						if list_data[i][0] == c_i:
							list_data[i][1] = c_time
							list_data[i][2] = c_acc
							list_data[i][3] = c_cost
							break


	def maxInSel(self, sel, budget):
		if self.type_exp == "nmist" or self.type_exp == "svm":
			return
		
		max_c = None
		max_acc = 0
		max_time = 0
		max_cost = 0

		for c,_,_,_ in sel:
			if self.type_exp == "unet":
				conf_dict = dict([
					('vm_flavor', c["Flavor"]),
					('batch_size', int(c["batch"])),
					('learning_rate', c["learningRate"]),
					('momentum', float(c["momentum"])),
					('nrWorker', int(c["nrWorker"])),
					('synchronism', c["sync"]),
					('budget', int(budget))])

			elif self.type_exp == "fake_all" or self.type_exp == "fake_time_all":
				conf_dict = dict([
					('vm_flavor', c["vm_flavor"]),
					('batch_size', int(c["batch_size"])),
					('learning_rate', c["learning_rate"]),
					('num_cores', int(c["num_cores"])),
					('synchronism', c["synchronism"]),
					('network', c["network"]),
					('budget', int(budget))])	

			else:
				conf_dict = dict([
					('vm_flavor', c["vm_flavor"]),
					('batch_size', int(c["batch_size"])),
					('learning_rate', c["learning_rate"]),
					('num_cores', int(c["num_cores"])),
					('synchronism', c["synchronism"]),
					('budget', int(budget))])	


			for c_i, c_acc, c_cost, c_time in self.SearchSpace:
				#print(c_i)
				#print(conf_dict)
				#print()
				if conf_dict == c_i:
					if c_acc > max_acc:
						max_c = copy.deepcopy(conf_dict)
						max_time = c_time
						max_acc = c_acc
						max_cost = c_cost
					#print("dsadsadsad22")
					break

 				
		if max_c is None or max_time == 0  or max_acc == 0 or max_cost==0:
			print("ERROR in the dictConfigs unsel - max in sel")
			return

		self.maxSelJump = "[" + str(max_c) + "," + str(max_time) + "," + str(max_acc) + "," + str(max_cost) + "]"

	def __call__(self, job):
		cycle = len(self.testedSet)
		config_id = None

		for i in range(0, cycle):
			c = self.testedSet[i]
			c_id = c[0]
			if c_id == job.id:
				config_id = job.id
				break

		if config_id is None:
			print("ERROR: write csv -> different configs")
		
		config_data = self.testedSet[i]

		#job.id -> [1, 0, 5], 
		#job.kwargs['budget'] -> 60000.0, 
		#job.timestamps -> {"submitted": 1606633017.027284, "started": 1606633017.0274026, "finished": 1606633017.0394108}, 
		#job.result,{"loss": 0.022466659545898438, "info": {"accuracy": 0.9775333404541016, "cost": 0.37419374549526624, "total cost": 2.472803652821082, "accuracy loss": 0.022466659545898438, "budget": 60000.0, "training_time": 604.8390288390282789865, "error": "None"}}, 
		#job.exception -> null

		#config_id ->[0, 0, 0], 
		#config -> {"batch_size": 16.0, "learning_rate": 0.001, "num_cores": 32, "synchronism": "sync", "vm_flavor": "t2.medium"}, 
		#config_info -> {"predicted_loss_mean": -1, "predicted_loss_stdv": -1, "model_based_pick": false, "incumbent_line": -1, "incumbent_value": -1, "overhead_time": 0.0008485317230224609}]

		budget = job.kwargs['budget']

		if job.id != config_data[0]:
			print("ERROR: write csv -> running different configs")

		config = config_data[1]
		#print(config)
		#print(budget)
		#print(job.result)

		config_info = job.result["info"] #config_info[2]
		#print(config_info)

		confTime = config_info["training_time"]
		confAcc = config_info["accuracy"]
		confCost = config_info["cost"]
		cumulative_Cost = config_info["total cost"]

		simulatedCost = confCost
		aux_config = copy.deepcopy(config)
		aux_config["budget"] = budget
		aux_config["cost"] = confCost

		if self.hyperjump:
			if 	aux_config in self.trainingSet:
				# the config was tested in the current budget
				confCost = 0.0
				cumulative_Cost -= confCost
				confTime = 0.0
				simulatedCost = 0.0
			else:
				aux_list_tested = []
				if self.trainingSet:
					for i in range(len(self.trainingSet)):
						for k, v in self.trainingSet[i].items():
							if k == "budget":
								aux_list_tested.append(self.trainingSet[i])
								break
							if k == "cost":
								print("ERROR: training set results cost")
								break

							if aux_config[k] == v:
								continue
							else:
								break

					if aux_list_tested:
						aux_budget = -1
						aux_cost = -1
						for i in range(len(aux_list_tested)):
							if aux_list_tested[i]["budget"] > aux_budget:
								aux_budget = aux_list_tested[i]["budget"]
								aux_cost = aux_list_tested[i]["cost"]
						simulatedCost = confCost - aux_cost

		self.trainingSet.append(aux_config)
		#print(self.trainingSet)
		#print("real cost " + str(confCost) +  " incr cost " + str(simulatedCost))
		
		if budget == self.max_budget:
			if self.inc is None:
				self.inc = config 
				self.inc_id = config_id
				self.incTime = confTime
				self.incAcc = confAcc
				self.incCost = confCost
			else:
				if self.incAcc < confAcc:
					self.inc = config 
					self.inc_id = config_id
					self.incTime = confTime
					self.incAcc = confAcc
					self.incCost = confCost					 

		overhead = config_data[3]

		# eal_str = "-1"
		# if  self.hyperjump:
		# 	eal_data = self.EAL[config_id] 
		# 	if isinstance(eal_data, list):
		# 		eal_str = ""
		# 		for i in range(0, len(eal_data)):
		# 			tup = eal_data[i]
		# 			eal_str += str(tup[0]) + ":" + str(tup[1]) + ","
		
		if self.EAL_ is None:
			self.EAL_ = str(-1)
		if self.infoJump == "":
			self.infoJump = str(-1)
		if self.maxSelJump == "":
			self.maxSelJump = str(-1)
		if self.BrackInfoConf == "":
			self.BrackInfoConf = str(-1)
		if self.maxHB == "":
			self.maxHB = str(-1)

		if len(self.integralTime) == 0:
			str_integralTime = "-1"
		else:
			str_integralTime = ""
			for i in self.integralTime:
				str_integralTime += str(i) + ":"


		with open(self.file_logs, 'a') as fh:
			strWrite = str(self.seed) + ";" + str(self.it) + ";" + str(self.inc) + ";" + str(self.incTime) + ";" + str(self.incAcc) + ";" + str(self.incCost) + ";" + str(budget) + ";" + str(config) + ";" + str(confTime) + ";" + str(confAcc) + ";" + str(confCost) + ";" + str(overhead) + ";" + str(cumulative_Cost) + ";" + self.option + ";" +  self.EAL_ + ";" + str(simulatedCost) + ";" + str(self.counnter_appro_std) + ";" + self.maxHB + ";" + self.BrackInfoConf + ";" +  str(self.timeRisk) + ";" + str(self.timeTestOrder) + ";" + str(self.training_model) + ";" + str_integralTime + ";" + self.infoJump +  "\n"
			fh.write(strWrite)
		
		self.EAL_ = None
		self.infoJump = ""
		self.maxSelJump = ""
		self.BrackInfoConf = ""
		self.maxHB = ""
		self.it += 1
		self.integralTime.clear()




######################################
# this next classes are not being used
######################################

class Run(object):
	"""
		Not a proper class, more a 'struct' to bundle important
		information about a particular run
	"""
	def __init__(self, config_id, budget, loss, info, time_stamps, error_logs):
		self.config_id   = config_id
		self.budget      = budget
		self.error_logs  = error_logs
		self.loss        = loss
		self.info        = info
		self.time_stamps = time_stamps

	def __repr__(self):
		return(\
			"config_id: %s\t"%(self.config_id,) + \
			"budget: %f\t"%self.budget + \
			"loss: %s\n"%self.loss + \
			"time_stamps: {submitted} (submitted), {started} (started), {finished} (finished)\n".format(**self.time_stamps) + \
			"info: %s\n"%self.info
		)
	def __getitem__ (self, k):
		"""
			 in case somebody wants to use it like a dictionary
		"""
		return(getattr(self, k))


def extract_HBS_learning_curves(runs):
	"""
	function to get the hyperband learning curves

	This is an example function showing the interface to use the
	HB_result.get_learning_curves method.

	Parameters
	----------

	runs: list of HB_result.run objects
		the performed runs for an unspecified config

	Returns
	-------

	list of learning curves: list of lists of tuples
		An individual learning curve is a list of (t, x_t) tuples.
		This function must return a list of these. One could think
		of cases where one could extract multiple learning curves
		from these runs, e.g. if each run is an independent training
		run of a neural network on the data.
		
	"""
	sr = sorted(runs, key=lambda r: r.budget)
	lc = list(filter(lambda t: not t[1] is None, [(r.budget, r.loss) for r in sr]))
	return([lc,])
		

class json_result_logger(object):
	def __init__(self, directory, overwrite=False):
		"""
		convenience logger for 'semi-live-results'

		Logger that writes job results into two files (configs.json and results.json).
		Both files contain propper json objects in each line.

		This version opens and closes the files for each result.
		This might be very slow if individual runs are fast and the
		filesystem is rather slow (e.g. a NFS).

		Parameters
		----------

		directory: string
			the directory where the two files 'configs.json' and
			'results.json' are stored
		overwrite: bool
			In case the files already exist, this flag controls the
			behavior:
			
				* True:   The existing files will be overwritten. Potential risk of deleting previous results
				* False:  A FileExistsError is raised and the files are not modified.
		"""

		os.makedirs(directory, exist_ok=True)

		self.config_fn  = os.path.join(directory, 'configs.json')
		self.results_fn = os.path.join(directory, 'results.json')


		try:
			with open(self.config_fn, 'x') as fh: pass
		except FileExistsError:
			if overwrite:
				with open(self.config_fn, 'w') as fh: pass
			else:
				raise FileExistsError('The file %s already exists.'%self.config_fn)
		except:
			print("ERROR: resultjson_result_logger 1")
			raise

		try:
			with open(self.results_fn, 'x') as fh: pass
		except FileExistsError:
			if overwrite:
				with open(self.results_fn, 'w') as fh: pass
			else:
				raise FileExistsError('The file %s already exists.'%self.config_fn)

		except:
			print("ERROR: json_result_logger result2")
			raise

		self.config_ids = set()

	def new_config(self, config_id, config, config_info, overhead=0):
		if not config_id in self.config_ids:
			self.config_ids.add(config_id)
			with open(self.config_fn, 'a') as fh:
				fh.write(json.dumps([config_id, config, config_info, overhead]))
				fh.write('\n')

	def __call__(self, job):
		if not job.id in self.config_ids:
			#should never happen! TODO: log warning here!
			self.config_ids.add(job.id)
			with open(self.config_fn, 'a') as fh:
				fh.write(json.dumps([job.id, job.kwargs['config'], {}]))
				fh.write('\n')
		with open(self.results_fn, 'a') as fh:
			fh.write(json.dumps([job.id, job.kwargs['budget'], job.timestamps, job.result, job.exception]))
			fh.write("\n")


def logged_results_to_HBS_result(directory):
	"""
	function to import logged 'live-results' and return a HB_result object

	You can load live run results with this function and the returned
	HB_result object gives you access to the results the same way
	a finished run would.
	
	Parameters
	----------
	directory: str
		the directory containing the results.json and config.json files

	Returns
	-------
	hyperjump.core.result.Result: :object:
		TODO
	
	"""
	data = {}
	time_ref = float('inf')
	budget_set = set()
	
	with open(os.path.join(directory, 'configs.json')) as fh:
		for line in fh:
			
			line = json.loads(line)
			
			if len(line) == 3:
				config_id, config, config_info = line
			if len(line) == 2:
				config_id, config, = line
				config_info = 'N/A'

			data[tuple(config_id)] = Datum(config=config, config_info=config_info)

	with open(os.path.join(directory, 'results.json')) as fh:
		for line in fh:
			config_id, budget,time_stamps, result, exception = json.loads(line)

			id = tuple(config_id)
			
			data[id].time_stamps[budget] = time_stamps
			data[id].results[budget] = result
			data[id].exceptions[budget] = exception

			budget_set.add(budget)
			time_ref = min(time_ref, time_stamps['submitted'])


	# infer the hyperband configuration from the data
	budget_list = sorted(list(budget_set))
	
	HB_config = {
						'eta'        : None if len(budget_list) < 2 else budget_list[1]/budget_list[0],
						'min_budget' : min(budget_set),
						'max_budget' : max(budget_set),
						'budgets'    : budget_list,
						'max_SH_iter': len(budget_set),
						'time_ref'   : time_ref
				}
	return(Result([data], HB_config))


class Result(object):
	"""
	Object returned by the HB_master.run function

	This class offers a simple API to access the information from
	a Hyperband run.
	"""
	def __init__ (self, HB_iteration_data, HB_config):
		self.data = HB_iteration_data
		self.HB_config = HB_config
		self._merge_results()

	def __getitem__(self, k):
		return(self.data[k])


	def get_incumbent_id(self):
		"""
		Find the config_id of the incumbent.

		The incumbent here is the configuration with the smallest loss
		among all runs on the maximum budget! If no run finishes on the
		maximum budget, None is returned!
		"""
		tmp_list = []
		for k,v in self.data.items():
			try:
				# only things run for the max budget are considered
				res = v.results[self.HB_config['max_budget']]
				if not res is None:
					tmp_list.append((res['loss'], k))
			except KeyError as e:
				print("ERROR: Result 1")
				pass
			except:
				print("ERROR: Result 2")
				raise

		if len(tmp_list) > 0:
			return(min(tmp_list)[1])
		return(None)



	def get_incumbent_trajectory(self, all_budgets=True, bigger_is_better=True, non_decreasing_budget=True):
		"""
		Returns the best configurations over time
		
		
		Parameters
		----------
			all_budgets: bool
				If set to true all runs (even those not with the largest budget) can be the incumbent.
				Otherwise, only full budget runs are considered
			bigger_is_better:bool
				flag whether an evaluation on a larger budget is always considered better.
				If True, the incumbent might increase for the first evaluations on a bigger budget
			non_decreasing_budget: bool
				flag whether the budget of a new incumbent should be at least as big as the one for
				the current incumbent.
		Returns
		-------
			dict:
				dictionary with all the config IDs, the times the runs
				finished, their respective budgets, and corresponding losses
		"""
		all_runs = self.get_all_runs(only_largest_budget = not all_budgets)
		
		if not all_budgets:
			all_runs = list(filter(lambda r: r.budget==res.HB_config['max_budget'], all_runs))
		
		all_runs.sort(key=lambda r: r.time_stamps['finished'])
		
		return_dict = { 'config_ids' : [],
						'times_finished': [],
						'budgets'    : [],
						'losses'     : [],
		}
	
		current_incumbent = float('inf')
		incumbent_budget = self.HB_config['min_budget']
		
		for r in all_runs:
			if r.loss is None: continue
			
			new_incumbent = False
			
			if bigger_is_better and r.budget > incumbent_budget:
				new_incumbent = True
			
			
			if r.loss < current_incumbent:
				new_incumbent = True
			
			if non_decreasing_budget and r.budget < incumbent_budget:
				new_incumbent = False
			
			if new_incumbent:
				current_incumbent = r.loss
				incumbent_budget  = r.budget
				
				return_dict['config_ids'].append(r.config_id)
				return_dict['times_finished'].append(r.time_stamps['finished'])
				return_dict['budgets'].append(r.budget)
				return_dict['losses'].append(r.loss)

		if current_incumbent != r.loss:
			r = all_runs[-1]
		
			return_dict['config_ids'].append(return_dict['config_ids'][-1])
			return_dict['times_finished'].append(r.time_stamps['finished'])
			return_dict['budgets'].append(return_dict['budgets'][-1])
			return_dict['losses'].append(return_dict['losses'][-1])

			
		return (return_dict)


	def get_runs_by_id(self, config_id):
		"""
		returns a list of runs for a given config id

		The runs are sorted by ascending budget, so '-1' will give
		the longest run for this config.
		"""
		d = self.data[config_id]

		runs = []
		for b in d.results.keys():
			try:
				err_logs = d.exceptions.get(b, None)

				if d.results[b] is None:
					r = Run(config_id, b, None, None , d.time_stamps[b], err_logs)
				else:
					r = Run(config_id, b, d.results[b]['loss'], d.results[b]['info'] , d.time_stamps[b], err_logs)
				runs.append(r)
			except:
				print("ERROR: Result 3")
				raise
		runs.sort(key=lambda r: r.budget)
		return(runs)


	def get_learning_curves(self, lc_extractor=extract_HBS_learning_curves, config_ids=None):
		"""
		extracts all learning curves from all run configurations

		Parameters
		----------
			lc_extractor: callable
				a function to return a list of learning_curves.
				defaults to hyperjump.HB_result.extract_HP_learning_curves
			config_ids: list of valid config ids
				if only a subset of the config ids is wanted

		Returns
		-------
			dict
				a dictionary with the config_ids as keys and the
				learning curves as values
		"""

		config_ids = self.data.keys() if config_ids is None else config_ids
		
		lc_dict = {}
		
		for id in config_ids:
			runs = self.get_runs_by_id(id)
			lc_dict[id] = lc_extractor(runs)
			
		return(lc_dict)


	def get_all_runs(self, only_largest_budget=False):
		"""
		returns all runs performed

		Parameters
		----------
			only_largest_budget: boolean
				if True, only the largest budget for each configuration
				is returned. This makes sense if the runs are continued
				across budgets and the info field contains the information
				you care about. If False, all runs of a configuration
				are returned
		"""
		all_runs = []

		for k in self.data.keys():
			runs = self.get_runs_by_id(k)

			if len(runs) > 0:
				if only_largest_budget:
					all_runs.append(runs[-1])
				else:
					all_runs.extend(runs)

		return(all_runs)

	def get_id2config_mapping(self):
		"""
		returns a dict where the keys are the config_ids and the values
		are the actual configurations
		"""
		new_dict = {}
		for k, v in self.data.items():
			new_dict[k] = {}
			new_dict[k]['config'] = copy.deepcopy(v.config)
			try:
				new_dict[k]['config_info'] = copy.deepcopy(v.config_info)
			except:
				print("ERROR: Result 4")
				pass
		return(new_dict)

	def _merge_results(self):
		"""
		hidden function to merge the list of results into one
		dictionary and 'normalize' the time stamps
		"""
		new_dict = {}
		for it in self.data:
			new_dict.update(it)

		for k,v in new_dict.items():
			for kk, vv in v.time_stamps.items():
				for kkk,vvv in vv.items():
					new_dict[k].time_stamps[kk][kkk] = vvv - self.HB_config['time_ref']

		self.data = new_dict

	def num_iterations(self):
		return(max([k[0] for k in self.data.keys()]) + 1)
		

	def get_fANOVA_data(self, config_space, budgets=None, loss_fn=lambda r: r.loss, failed_loss=None):

		import numpy as np
		import ConfigSpace as CS

		id2conf = self.get_id2config_mapping()

		if budgets is None:
			budgets = self.HB_config['budgets']

		if len(budgets)>1:
			config_space.add_hyperparameter(CS.UniformFloatHyperparameter('budget', min(budgets), max(budgets), log=True))
		
		hp_names = config_space.get_hyperparameter_names()
		hps = config_space.get_hyperparameters()
		needs_transform = list(map(lambda h: isinstance(h, CS.CategoricalHyperparameter), hps))

		all_runs = self.get_all_runs(only_largest_budget=False)


		all_runs=list(filter( lambda r: r.budget in budgets, all_runs))

		X = []
		y = []

		for r in all_runs:
			if r.loss is None:
				if failed_loss is None: continue
				else: y.append(failed_loss)
			else:
				y.append(loss_fn(r))
				
			config = id2conf[r.config_id]['config']
			if len(budgets)>1:
				config['budget'] = r.budget

			config = CS.Configuration(config_space, config)
			
			x = []
			for (name, hp, transform) in zip(hp_names, hps, needs_transform):
				if transform:
					x.append(hp._inverse_transform(config[name]))
				else:
					x.append(config[name])
			
			X.append(x)

		return(np.array(X), np.array(y), config_space)


	def get_pandas_dataframe(self, budgets=None, loss_fn=lambda r: r.loss):

		import numpy as np
		import pandas as pd

		id2conf = self.get_id2config_mapping()

		df_x = pd.DataFrame()
		df_y = pd.DataFrame()


		if budgets is None:
			budgets = self.HB_config['budgets']

		all_runs = self.get_all_runs(only_largest_budget=False)
		all_runs=list(filter( lambda r: r.budget in budgets, all_runs))



		all_configs = []
		all_losses = []

		for r in all_runs:
			if r.loss is None: continue
			config = id2conf[r.config_id]['config']
			if len(budgets)>1:
				config['budget'] = r.budget

			all_configs.append(config)
			all_losses.append({'loss': r.loss})
			
			#df_x = df_x.append(config, ignore_index=True)
			#df_y = df_y.append({'loss': r.loss}, ignore_index=True)
		
		df_X = pd.DataFrame(all_configs)
		df_y = pd.DataFrame(all_losses)

		return(df_X, df_y)

