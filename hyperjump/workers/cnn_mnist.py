import time
import pickle

import ConfigSpace as CS
from hyperjump.core.worker import Worker
import ConfigSpace.hyperparameters as CSH

budget_option = 1 # epochs as budget
#budget_option = 2  # dataset size as budget


budget1 = [1, 2, 4, 8, 16]
budget2 = [0.0625, 0.125, 0.25, 0.5, 1.0]

if budget_option==1:
    # budget1 -> epcohs
    Budgets = budget1
else:
    Budgets = budget2



class CNN_Worker(Worker):

    def __init__(self, pauseResume=True, checkpointing=True, factor=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        datasetName = "./hyperjump/workers/data/data_mnist_epoch_datasetSize.csv"
        self.data = self.readData(datasetName)
        self.trainingSet = dict()
        self.factor = factor
        self.pauseResume = pauseResume
        self.checkpointing = checkpointing

        super().__init__(**kwargs)


    def readData(self, filename):
        _data = dict()
        with open(filename, "r") as fl:
            for line in fl:
                if "dataset" in line: continue # 1st line
            #dataset,size,epoch,lr,momentum,bacth,weightDecay,testError,testLoss,trainTime,testTime,stdTestError,stdTestLoss,stdTrainTime,stdTestTime

                aux = line.split(",")
                epoch = int(aux[2])

                _dataset = aux[0]
                size = float(aux[1])
                lr = float(aux[3])
                momentum = float(aux[4])
                bacth = int(aux[5])
                weightDecay = float(aux[6])
                testError = float(aux[7])
                trainTime = float(aux[9])

                if budget_option == 1 :# epochs as budget
                    config = {
                        'batch_size':  bacth,
                        'learning_rate':  lr,
                        'momentum': momentum, 
                        'weight_decay': weightDecay, 
                        'size': size,
                        'budget': epoch,
                        }
                else:
                    config = {
                        'batch_size':  bacth,
                        'learning_rate':  lr,
                        'momentum': momentum, 
                        'weight_decay': weightDecay, 
                        'epoch': epoch,
                        'budget': size,
                        }

                key = pickle.dumps(config)
                _data[key] = [testError, trainTime]
        return _data


    def getVal(self, config, epoch, dataset_size):
        if budget_option == 1 :# epochs as budget
            config_with_budgets = {
                            'batch_size': config['batch_size'],
                            'learning_rate':  config['learning_rate'],
                            'momentum': config['momentum'], 
                            'weight_decay': config['weight_decay'], 
                            'size': dataset_size,
                            'budget': epoch, #'epoch': epoch,
                            }


            for kk in self.data.keys():
                conf = pickle.loads(kk)
                if conf['batch_size'] == config_with_budgets['batch_size'] \
                            and conf['learning_rate'] == config_with_budgets['learning_rate'] \
                            and conf['momentum'] == config_with_budgets['momentum'] \
                            and conf['weight_decay'] == config_with_budgets['weight_decay']\
                            and conf['size'] == config_with_budgets['size']\
                            and conf['budget'] == config_with_budgets['budget']:

                            return config_with_budgets, self.data[kk][0], self.data[kk][1]
        else:
            config_with_budgets = {
                            'batch_size': config['batch_size'],
                            'learning_rate':  config['learning_rate'],
                            'momentum': config['momentum'], 
                            'weight_decay': config['weight_decay'], 
                            'epoch': epoch,
                            'budget': dataset_size,
                            }


            for kk in self.data.keys():
                conf = pickle.loads(kk)
                if conf['batch_size'] == config_with_budgets['batch_size'] \
                            and conf['learning_rate'] == config_with_budgets['learning_rate'] \
                            and conf['momentum'] == config_with_budgets['momentum'] \
                            and conf['weight_decay'] == config_with_budgets['weight_decay']\
                            and conf['epoch'] == config_with_budgets['epoch']\
                            and conf['budget'] == config_with_budgets['budget']:

                            return config_with_budgets, self.data[kk][0], self.data[kk][1]

        return


    def compute(self, config, budget, *args, **kwargs):
        starttime = time.time()

        if budget_option==1:
            config_with_budgets, loss, trainTime = self.getVal(config, budget, 1.0) # budget is epochs
        else:
            config_with_budgets, loss, trainTime = self.getVal(config, 16, budget)  # budget is dataset size

        #print(config_with_budgets)


        prev_time = 0
        prev_budget = 0 
        if self.pauseResume:
            for kk in self.trainingSet.keys():
                conf = pickle.loads(kk)

                if budget_option==1: # budget is epochs
                    if conf['batch_size'] == config_with_budgets['batch_size'] \
                                    and conf['learning_rate'] == config_with_budgets['learning_rate'] \
                                    and conf['momentum'] == config_with_budgets['momentum'] \
                                    and conf['weight_decay'] == config_with_budgets['weight_decay']\
                                    and conf['size'] == config_with_budgets['size']:
                        
                        if conf['budget'] > prev_budget: # restore the largest tested budget of that config
                            prev_time = self.trainingSet[kk][2]
                            prev_budget = conf['budget']


                else: # budget is dataset size
                    if conf['batch_size'] == config_with_budgets['batch_size'] \
                                    and conf['learning_rate'] == config_with_budgets['learning_rate'] \
                                    and conf['momentum'] == config_with_budgets['momentum'] \
                                    and conf['weight_decay'] == config_with_budgets['weight_decay']\
                                    and conf['epoch'] == config_with_budgets['epoch']:

                        if conf['budget'] > prev_budget: # restore the largest tested budget of that config
                            prev_time = self.trainingSet[kk][2]

        working_time = (trainTime-prev_time)/(self.factor*1.0)
        if working_time > 0:
            time.sleep(working_time)
        
        key = pickle.dumps(config_with_budgets)
        self.trainingSet[key] = [config_with_budgets, loss, trainTime] # 

        checkpoint = []
        if self.checkpointing: # and prev_time == 0:
            for bb in Budgets:
                if bb < budget:
                    if budget_option==1: # budget is epochs
                        config_smaller_budget, _loss, _time = self.getVal(config, bb, 1.0)
                    else:
                        config_smaller_budget, _loss, _time = self.getVal(config, 16, bb)  # budget is dataset size
                    
                    config['budget'] = bb
                    checkpoint.append([config, _loss, _time])

        if len(checkpoint) == 0:
            checkpoint = None

        t_now = time.time()

        self.logger.info("Running config " + str(config_with_budgets) + " that achieved a accuracy of " + str(1-loss) + "  during " + str(t_now - starttime) + " seconds " + str((t_now-starttime)*self.factor))

        return ({
            'loss': loss,  # remember: hyperjump always minimizes!
            'info': {
                     'config': config_with_budgets,
                     'training_time': trainTime,
                     'start_time': starttime,
                     'end_time': t_now,
                     'real_time': (t_now-starttime)*self.factor,
                     'accuracy': 1.0-loss,
                     'checkpoint': checkpoint,
                     }
            })



    @staticmethod
    def get_configspace(seed):
        cs = CS.ConfigurationSpace(seed=seed)
        batch_sizes = CSH.CategoricalHyperparameter('batch_size', [64, 128, 256, 512])
        learning_rates = CSH.CategoricalHyperparameter('learning_rate', [0.1, 0.01, 0.001, 0.0001])
        momentums = CSH.CategoricalHyperparameter('momentum', [0.0, 0.9, 0.95, 0.99])
        weight_decays = CSH.CategoricalHyperparameter('weight_decay', [0, 0.01, 0.001, 0.0001])

        cs.add_hyperparameters([batch_sizes, learning_rates, momentums, weight_decays])

        return cs
