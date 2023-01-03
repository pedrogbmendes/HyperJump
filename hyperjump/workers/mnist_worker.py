import pickle
import math, csv, sys
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import time

from hyperjump.core.worker import Worker


csv.field_size_limit(sys.maxsize)


# ['t2.small', 't2.medium', 't2.xlarge', 't2.2xlarge']
def get_num_workers(vm_flavor, num_cores):
    num_workers = float(num_cores)
    if str(vm_flavor) == 't2.medium':
        num_workers = float(num_cores) / 2
    elif str(vm_flavor) == 't2.xlarge':
        num_workers = float(num_cores) / 4
    elif str(vm_flavor) == 't2.2xlarge':
        num_workers = float(num_cores) / 8
    return int(num_workers)


# Price per hour	|	Vm flavour
# 0,023			|	t2.small
# 0,0464		|	t2.medium
# 0,1856		|	t2.xlarge
# 0,3712		|	t2.2xlarge

# Mudar para (price_per_hour(VM_type)/3600)+(price_per_hour(t2.2xlarge)/3600)*training_time
def get_cost(vm_flavor, num_workers, training_time):
    nr_ps = 1
    if vm_flavor == "t2.small":
        cost = (((num_workers+nr_ps) * 0.023/60.0) + (0.3712/60.0)) * (training_time/60.0)
    elif vm_flavor == "t2.medium":
        cost = (((num_workers+nr_ps) * 0.0464/60.0) + (0.3712/60.0)) * (training_time/60.0)
    elif vm_flavor == "t2.xlarge":
        cost = (((num_workers+nr_ps) * 0.1856/60.0) + (0.3712/60.0)) * (training_time/60.0)
    elif vm_flavor == "t2.2xlarge":
        cost = (((num_workers+nr_ps) * 0.3712/60.0) + (0.3712/60.0)) * (training_time/60.0)
    else:
        print("Tensorflow configuration - Unknown flavor" + vm_flavor)
        sys.exit(0)
    return cost



class MnistWorker(Worker):
    def __init__(self, dataset='cnn', eta=2, pauseResume=True, checkpointing=True, factor=1.0, **kwargs):
        super().__init__(**kwargs)

        if "time" in dataset:
            if eta==2:
                self.dataset_name = dataset
            else:
                self.dataset_name = dataset + "3"

        else:
            self.dataset_name = dataset
        self.type_time = True if "time" in dataset else False


        self.data = self.load_dataset( "./hyperjump/workers/data/" +  self.dataset_name + '.csv')

        self.trainingSet = dict()
        self.factor = factor
        self.pauseResume = pauseResume
        self.checkpointing = checkpointing
        self.eta = eta

            
        if self.type_time:
            if eta==2:
                self.Budgets = [37, 75, 150, 300, 600]
            else:
                self.Budgets = [7, 22, 66, 200, 600]
        else:
            self.Budgets = [3750, 7500, 15000, 30000, 60000]
            self.eta = 2 # budget is dataset size only supports eta=2


    #Read the dataset
    def load_dataset(self, CSV):
        self.logger.info("Reading the dataset....")
        _data = dict()

        with open(CSV) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            headers = next(csv_reader, None)

            for row in csv_reader:
                nr_workers = int(row[1])
                learning_rate = float(row[3])
                batch_size = int(row[4])
                synchronism = row[6]
                _time = float(row[7])    # training time in seconds
                nr_ps = int(row[10])
                accuracy = float(row[11])
                flavor = row[13]

                if self.type_time:
                    size = _time  # dataset size is budget
                else:
                    size = float(row[15]) # dataset size is budget
                    if size == 1000: size = 3750 
                    elif size == 6000: size = 7500 


                if flavor == "t2.small":
                    cost = (((nr_workers+nr_ps) * 0.023/60.0) + (0.3712/60.0)) * (_time/60.0)
                    num_cores = nr_workers
                elif flavor == "t2.medium":
                    cost = (((nr_workers+nr_ps) * 0.0464/60.0) + (0.3712/60.0)) * (_time/60.0)
                    num_cores = int(2*nr_workers)
                elif flavor == "t2.xlarge":
                    cost = (((nr_workers+nr_ps) * 0.1856/60.0) + (0.3712/60.0)) * (_time/60.0)
                    num_cores = int(4*nr_workers)
                elif flavor == "t2.2xlarge":
                    cost = (((nr_workers+nr_ps) * 0.3712/60.0) + (0.3712/60.0)) * (_time/60.0)
                    num_cores = int(8*nr_workers)
                else:
                    self.logger.warning("configuration - Unknown flavor" + flavor)
                    sys.exit(0)

                config = dict([
                    ('vm_flavor', flavor),
                    ('batch_size', batch_size),
                    ('learning_rate', learning_rate),
                    ('num_cores', num_cores),
                    ('synchronism', synchronism),
                    ('budget', size)])

                key = pickle.dumps(config)
                _data[key] = [accuracy, _time, cost]

            csv_file.close()
            self.logger.info("Dataset read")
            return _data


    def getVal(self, config, budget):
        config_with_budgets = {
                        'vm_flavor': config['vm_flavor'],
                        'batch_size': config['batch_size'],
                        'learning_rate':  config['learning_rate'],
                        'num_cores': config['num_cores'], 
                        'synchronism': config['synchronism'],
                        'budget': budget, #'epoch': epoch,
                        }


        for kk in self.data.keys():
            conf = pickle.loads(kk)
            if conf['vm_flavor'] == config_with_budgets['vm_flavor'] \
                        and conf['batch_size'] == config_with_budgets['batch_size'] \
                        and conf['learning_rate'] == config_with_budgets['learning_rate'] \
                        and conf['num_cores'] == config_with_budgets['num_cores'] \
                        and conf['synchronism'] == config_with_budgets['synchronism']\
                        and conf['budget'] == config_with_budgets['budget']:

                        return config_with_budgets, self.data[kk][0], self.data[kk][1]

        return


    def compute(self, config, budget, *args, **kwargs):
        budget = int(budget)
        starttime = time.time()
        config_with_budgets, acc, trainTime = self.getVal(config, budget) # budget is dataset size

        prev_time = 0
        prev_budget = 0 
        if self.pauseResume:
            for kk in self.trainingSet.keys():
                conf = pickle.loads(kk)

                if conf['vm_flavor'] == config_with_budgets['vm_flavor'] \
                                and conf['batch_size'] == config_with_budgets['batch_size'] \
                                and conf['learning_rate'] == config_with_budgets['learning_rate'] \
                                and conf['num_cores'] == config_with_budgets['num_cores'] \
                                and conf['synchronism'] == config_with_budgets['synchronism']:
                    
                    if conf['budget'] > prev_budget: # restore the largest tested budget of that config
                        prev_time = self.trainingSet[kk][2]
                        prev_budget = conf['budget']

        working_time = (trainTime-prev_time)/(self.factor*1.0)
        if working_time > 0:
            time.sleep(working_time)

        key = pickle.dumps(config_with_budgets)
        self.trainingSet[key] = [config_with_budgets, acc, trainTime] # 

        checkpoint = []
        if self.checkpointing: # and prev_time == 0:
            for bb in self.Budgets:
                if bb < budget:
                    config_smaller_budget, _acc, _time = self.getVal(config, bb) # budget is dataset size
                    config['budget'] = bb
                    checkpoint.append([config, _acc, _time])

        if len(checkpoint) == 0:
            checkpoint = None

        t_now = time.time()

        self.logger.info("Running config " + str(config_with_budgets) + " that achieved a accuracy of " + str(acc) + "  during " + str(t_now - starttime) + " seconds " + str((t_now-starttime)*self.factor))
        
        return ({
            'loss': 1.0-acc,  # remember: hyperjump always minimizes!
            'info': {
                     'config': config_with_budgets,
                     'training_time': trainTime,
                     'start_time': starttime,
                     'end_time': t_now,
                     'real_time': (t_now-starttime)*self.factor,
                     'accuracy': acc,
                     'checkpoint': checkpoint,
                     }
            })


    @staticmethod
    def get_configspace(seed=1000):

        cs = CS.ConfigurationSpace(seed=seed)
        vm_flavor = CSH.CategoricalHyperparameter('vm_flavor', ['t2.small', 't2.medium', 't2.xlarge', 't2.2xlarge'])
        batch_size = CSH.CategoricalHyperparameter('batch_size', [16, 256])
        learning_rate = CSH.CategoricalHyperparameter('learning_rate', [0.00001, 0.0001, 0.001])
        num_cores = CSH.CategoricalHyperparameter('num_cores', [8, 16, 32, 48, 64, 80])
        synchronism = CSH.CategoricalHyperparameter('synchronism', ['async', 'sync'])

        cs.add_hyperparameters([vm_flavor, batch_size, learning_rate, num_cores, synchronism, ])

        return cs

