"""
main worker

"""
import os
from os import listdir
from os.path import isdir, join
from termcolor import colored

import sys
import pickle
import argparse
import datetime
import time
import numpy as np

import hyperjump.core.nameserver as hpns
import hyperjump.core.result as hpres

from hyperjump.optimizers import BOHB_TPE
from hyperjump.optimizers import HYPERJUMP
from hyperjump.optimizers import BOHB_EI
from hyperjump.optimizers import HyperBand


import logging
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.WARNING)


def get_next_run_num(path):
    if not os.path.exists("./logs/" + path.split("/")[2]):
        os.mkdir("./logs/" + path.split("/")[2])
    if not os.path.exists("logs/" + path.split("/")[2] + "/" + path.split("/")[3]):
        os.mkdir("./logs/" + path.split("/")[2] + "/" + path.split("/")[3])

    l = [int(f.split("_")[-1]) for f in listdir(path) if isdir(join(path, f))]
    if len(l) != 0:
        return str(l[np.argmax(l)] + 1)
    else:
        return str(0)


def run(arguments):
    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(arguments.nic_name)
    if arguments.type == 'mnist':
        eta = arguments.eta
        from mnist_worker import MnistWorker as worker
        #from hyperjump.examples.fake.mnist_worker import MnistWorker as worker
        arguments.n_iterations = 10

    elif arguments.type == 'unet':
        eta = arguments.eta
        from unet_worker import UnetWorker as worker
        #arguments.dataset = "unet"

        # if arguments.dataset == "unet":
        #     eta=2
        # elif arguments.dataset == "unet3":
        #     eta=3
        # else:
        #     print("Wrong ETA")
        #     sys.exit()
    elif arguments.type == 'svm':
        eta = arguments.eta
        from svm_worker import SvmWorker as worker

    elif arguments.type == 'fake_time':
        eta =  arguments.eta
        from fake_worker_time import FakeWorker as worker

    elif arguments.type == 'fake_time_all':
        eta =  arguments.eta
        from fake_worker_time_all import FakeWorker as worker

    else:
        eta = 2
        if arguments.dataset == "all":
            from fake_worker_all import FakeWorker as worker
        else:
            from fake_worker import FakeWorker as worker
        #from hyperjump.examples.fake.fake_worker import FakeWorker as worker


    #print(arguments.shared_directory)

    #print("eta is " + str(eta))
    if arguments.worker:
        time.sleep(5)  # short artificial delay to make sure the nameserver is already running
        #w = worker(budget_type=arguments.budget_type, run_id=arguments.run_id, host=host, timeout=120,
        #           seedNum=arguments.seed, dataset=arguments.dataset)
        w = worker(budget_type=arguments.budget_type, run_id=arguments.run_id, host=host,
                   seedNum=arguments.seed, dataset=arguments.dataset)
        w.load_nameserver_credentials(working_directory=arguments.shared_directory)
        w.run(background=False)
        exit(0)

    np.random.seed(arguments.seed)
    if arguments.seed == -1:
        arguments.seed = np.random.randint(0, 100000)
        np.random.seed(arguments.seed)
    
    # This example shows how to log live results. This is most useful
    # for really long runs, where intermediate results could already be
    # interesting. The core.result submodule contains the functionality to
    # read the two generated files (results.json and configs.json) and
    # create a Result object.

    #result_logger = hpres.json_result_logger(directory=arguments.shared_directory, overwrite=False)
    result_logger = hpres.csv_results(directory=arguments.shared_directory, seed=arguments.seed, max_budget=arguments.max_budget, hyperjump=arguments.hyperjump, overwrite=False)

    # Start a nameserver:
    NS = hpns.NameServer(run_id=arguments.run_id, host=host, port=0, working_directory=arguments.shared_directory)
    ns_host, ns_port = NS.start()

    # Start local worker
    #w = worker(dataset=arguments.dataset,run_id=arguments.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120)
    if arguments.type == 'unet':
        if eta == 2:
            w = worker(dataset="unet",run_id=arguments.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)
        else:
            w = worker(dataset="unet3",run_id=arguments.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)

    elif "time" in arguments.type and eta == 3:
        dataset_ = arguments.dataset + "3"
        w = worker(dataset=dataset_,run_id=arguments.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)
    
    else:
        w = worker(dataset=arguments.dataset,run_id=arguments.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)
    
    w.run(background=True)

    t = time.time()
    # Run an optimizer
    if arguments.algorithm == 'HJ':
        #hyperjump

        if arguments.type == 'unet' or arguments.type == 'svm':
            listConfigSpace = w.listConfigSpace()
        else:
            listConfigSpace = None

        result_logger.updateSearchSpace(w.dictConfigSpace())

        alg = HYPERJUMP(configspace=worker.get_configspace(arguments.seed),
                       eta=eta,
                       run_id=arguments.run_id,
                       host=host,
                       nameserver=ns_host,
                       nameserver_port=ns_port,
                       result_logger=result_logger,
                       min_budget=arguments.min_budget, max_budget=arguments.max_budget,
                       seed=arguments.seed,
                       type_exp=arguments.type,
                       algorithm_variant=arguments.algorithm_variant,
                       hyperjump=arguments.hyperjump,
                       random_fraction=arguments.random_fraction,
                       threshold=arguments.threshold,
                       configspaceList=listConfigSpace
                       )

    elif arguments.algorithm == 'BOHB-EI':

        if arguments.type == 'unet' or arguments.type == 'svm':
            listConfigSpace = w.listConfigSpace()
        else:
            listConfigSpace = None

        result_logger.updateSearchSpace(w.dictConfigSpace())

        alg = BOHB_EI(configspace=worker.get_configspace(arguments.seed),
                      eta=eta,
                      run_id=arguments.run_id,
                      host=host,
                      nameserver=ns_host,
                      nameserver_port=ns_port,
                      result_logger=result_logger,
                      min_budget=arguments.min_budget, max_budget=arguments.max_budget,
                      seed=arguments.seed,
                      type_exp=arguments.type,
                      algorithm_variant=arguments.algorithm_variant,
                      configspaceList=listConfigSpace,
                      )

                      
    elif arguments.algorithm == 'BOHB-TPE':

        if arguments.type == 'unet' or arguments.type == 'svm':
            listConfigSpace = w.listConfigSpace()
        else:
            listConfigSpace = None

        result_logger.updateSearchSpace(w.dictConfigSpace())

        alg = BOHB_TPE(configspace=worker.get_configspace(arguments.seed),
                        eta=eta,
                        run_id=arguments.run_id,
                        host=host,
                        nameserver=ns_host,
                        nameserver_port=ns_port,
                        result_logger=result_logger,
                        min_budget=arguments.min_budget, max_budget=arguments.max_budget,
                        seed=arguments.seed,
                        type_exp=arguments.type,
                        configspaceList=listConfigSpace,
                       )

    elif arguments.algorithm == 'HB':
        if arguments.type == 'unet' or arguments.type == 'svm':
            listConfigSpace = w.listConfigSpace()
        else:
            listConfigSpace = None

        alg = HyperBand(configspace=worker.get_configspace(arguments.seed),
                        eta=eta,
                        run_id=arguments.run_id,
                        host=host,
                        nameserver=ns_host,
                        nameserver_port=ns_port,
                        result_logger=result_logger,
                        min_budget=arguments.min_budget, max_budget=arguments.max_budget,
                        seed=arguments.seed,
                        type_exp=arguments.type,
                        configspaceList=listConfigSpace
                        )
    else:
        print("There is no optimizer named " + arguments.algorithm)
        sys.exit()

    res = alg.run(n_iterations=arguments.n_iterations)
    # for c in alg.config_generator.training_set:
    #	if c[-1] == 60000:
    #		print(c)

    t_final = time.time() - t
    # store results
    #with open(os.path.join(arguments.shared_directory, 'results.pkl'), 'wb') as fh:
    #    pickle.dump(res, fh)

    # shutdown
    # print("Seed = %d"%(args.seed))
    # print("Exec time = %f"%(t_final))
    alg.shutdown(shutdown_workers=True)
    NS.shutdown()
    return arguments.seed, t_final

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('True', 'yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    now = datetime.datetime.now()
    # shared_dir = 'fake_run_results_' + str(now.day) + '.' + str(now.month) + '(' + str(now.hour) + ':' + str(now.minute) + ')'
    seed = 0
    cmd = sys.argv
    if '--algorithm' in cmd:
        alg = cmd[cmd.index('--algorithm') + 1]
    else:
        alg = 'BOHB++'

    if '--dataset' in cmd:
        ds = cmd[cmd.index('--dataset') + 1]
    else:
        ds = 'cnn'

    if '--algorithm_variant' in cmd:
        if cmd[cmd.index('--algorithm_variant') + 1] == "DT" and '++' in alg:
            print(colored('Warning:', 'red'), 'Only DT Chosen... Defaulting to ',
                  colored('FBS_DT - Full Budget Sampling', 'green'))
            var = 'FBS_DT'
        else:
            var = cmd[cmd.index('--algorithm_variant') + 1]
    else:
        if '++' in alg:
            print(colored('Warning:', 'red'), 'No variant chosen.. Defaulting to ',
                  colored('FBS - Full Budget Sampling', 'green'))
            var = 'FBS'
        else:
            var = ''

    if '--type' in cmd:
        ty = cmd[cmd.index('--type') + 1]
        bmin = 1
        bmax = 16
        if var != '':
            new_var = '_' + var
        else:
            new_var = var

        exp_name = ''
        if ty == 'mnist':
            exp_name = 'mnist'
        elif ty == 'unet':
            exp_name = 'unet'
        elif ty == 'svm':
            exp_name = 'svm'
        #elif ty == 'fake_time':
        #    exp_name = 'fake_time'
        else:
            raise BaseException("exp_type field incompatible. Please omit if 'fake' and only use with mnist or unet. Used: ", ty, '. Received ', cmd)

        shared_dir = './logs/' + exp_name + '/' + alg + new_var + '/run_' + get_next_run_num("./logs/" + exp_name + "/" + alg + new_var + '/')
        seed = get_next_run_num("./logs/" + exp_name + "/" + alg + new_var + '/')
   
    else:
        bmin = 3750
        bmax = 60000
        if var != '':
            new_var = '_' + var
        else:
            new_var = var
        shared_dir = './logs/' + ds + '/' + alg + new_var + '/run_' + get_next_run_num("./logs/" + ds + "/" + alg + new_var + '/')
        seed = get_next_run_num("./logs/" + ds + "/" + alg + new_var + '/')


    parser = argparse.ArgumentParser(description='Fake run of CNN on MNIST')
    # minimum budget
    parser.add_argument('--min_budget', type=float, help='Minimum dataset size evaluated:  3750 for fake, 1 for mnist', default=bmin)
    #maximum budget
    parser.add_argument('--max_budget', type=float, help='Maximum dataset size evaluated: 60000 for fake, 16 for mnist', default=bmax)
    #n_iterations -> number of times to run the algorithm
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer (Number of generated brackets)',default=10)
    #worker flag
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    # id
    parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
    
    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.', default='lo')
    #algorithm
    parser.add_argument('--algorithm', type=str, choices=['HJ', 'BOHB-TPE', 'BOHB-EI','HB'], default=alg,
                        help='Type of optimization algorithm you want to perform. Options are: HJ, BOHB-TPE, BOHB-EI, HB')
    #seed
    parser.add_argument('--seed', type=int, help='The seed for randomization', default=seed)
    #dataset
    parser.add_argument('--dataset', type=str, choices=['cnn', 'rnn', 'multilayer', 'all', 'cnn_time', 'rnn_time', 'multilayer_time', 'all_time'],default=ds, 
                        help='The dataset where the objective function values are. Options: cnn, multilayer, rnn',)
    #type
    parser.add_argument('--type', type=str, choices=['fake', 'mnist', 'unet', 'svm'], help='Options: fake, mnist', default='fake')

    parser.add_argument('--shared_directory', type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default=shared_dir)
   
    #algortihm variant
    parser.add_argument('--algorithm_variant', type=str, choices=['FBS', 'CBS', 'Hybrid', 'DT', 'FBS_EI$', 'CBS_EI$', 'FBS_DT', 'CBS_DT', 'FBS_EI$_DT', 'CBS_EI$_DT', 'Hybrid_DT'], default=var,
                        help='Type of the variant of the test that would change the sampling of configurations. ' +
                             'Examples:\n' +
                             '\tFBS for Full Budget Sampling\n' +
                             '\tCBS for Current Budget Sampling\n' +
                             '\tFBS_EI$ for Full Budget Sampling with a cost model that is also based on the FBS idea\n' +
                             '\tCBS_EI$ for Current Budget Sampling with a cost model that is also based on the CBS idea\n' +
                             '\tHybrid for Hybrid Budget Sampling, that predicts configuration accuracies according with FBS, with a cost model that predicts the cost with CBS\n')
   
    #hyperjump
    parser.add_argument('--hyperjump', type=str2bool, help='Hyperjump', default=False)
    #random fraction
    parser.add_argument('--random_fraction', type=float, help='random_fraction', default=1/3)
    parser.add_argument('--threshold', type=float, help='threshold', default=1.0)
    parser.add_argument('--eta', type=float, help='eta', default=2.0)

    args = parser.parse_args()

    if args.type == "unet":
        args.dataset = "unet"


    if args.type == "fake":
        if "time" in args.dataset:
            if "all" in args.dataset:
                args.type = "fake_time_all"
            else:
                args.type = "fake_time"

        else:
            if "all" in args.dataset:
                args.type = "fake_all"   
            #else-> fake


    maxIterations = 31
    if args.type == "svm":
        maxIterations = 21
        args.dataset = "svm"

    if args.seed == 0:
        for i in range (1,maxIterations):
        #for i in range (10, 16):
            args.seed = i
            args.shared_directory = './logs/' + args.dataset + '/' + alg + new_var + '/run_' + str(i) + '/'
            run(args)
    else:
        run(args)
