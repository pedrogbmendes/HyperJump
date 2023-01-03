import os
from os import listdir
from os.path import isdir, join
import math
import netifaces

import sys
import pickle
import argparse
import datetime
import time
import numpy as np

import hyperjump.core.nameserver as hpns
import hyperjump.core.result as hpres

from hyperjump.optimizers import Hyperjump
from hyperjump.optimizers import BOHB
from hyperjump.optimizers import HyperBand

import logging
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.ERROR)
#logging.basicConfig(level=logging.CRITICAL)

factor = 10.0


def run(arguments):

    logger = logging.getLogger('hyperjump')

    if args.n_workers != 1:
        logger.error("This implementation doesn't support several workers yet.")



    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(arguments.nic_name)

    if arguments.benchmark == 'cnn' or arguments.benchmark == 'rnn' or arguments.benchmark == 'multilayer' or \
            arguments.benchmark == 'cnn_time' or arguments.benchmark == 'rnn_time' or arguments.benchmark == 'multilayer_time':
        from hyperjump.workers.mnist_worker  import MnistWorker as worker
        factor = 10
        max_budget = 600 if "time" in arguments.benchmark else 60000

    elif arguments.benchmark == 'nas' or arguments.benchmark == 'nas_time':
        from hyperjump.workers.nas_worker  import MnistNasWorker as worker
        factor = 10
        max_budget = 600 if "time" in arguments.benchmark else 60000

    elif arguments.benchmark == 'cnn1':
        from hyperjump.workers.cnn_mnist  import CNN_Worker as worker
        factor = 10
        max_budget = 16

    elif "nats" in arguments.benchmark:
        from hyperjump.workers.natsbench_worker  import NatsBenchWorker as worker
        factor = 100
        max_budget = 200 if "tss" in arguments.benchmark else 90

    elif arguments.benchmark == 'unet':
        from hyperjump.workers.unet_worker  import UnetWorker as worker
        factor = 100
        max_budget = 18000 

    elif arguments.benchmark == 'svm':
        from hyperjump.workers.svm_worker  import SvmWorker as worker
        max_budget = 83333 

    else:
        logger.error("Error: experiment type is wrong!")
        sys.exit(0)


    no_stages = math.floor(np.log(args.max_b)/np.log(arguments.eta)) # number of stages 
    min_budget = max_budget/(arguments.eta**no_stages)


    if arguments.worker: # just a worker
        time.sleep(5)  # short artificial delay to make sure the nameserver is already running
        if arguments.benchmark == 'cnn' or arguments.benchmark == 'rnn' or arguments.benchmark == 'multilayer'\
                or arguments.benchmark == 'cnn_time' or arguments.benchmark == 'rnn_time' or arguments.benchmark == 'multilayer_time'\
                or arguments.benchmark == 'nas' or arguments.benchmark == 'nas_time'\
                or "nats" in arguments.benchmark:
           w = worker(dataset=arguments.benchmark, eta=arguments.eta, factor=factor, run_id=arguments.run_id, host=host, logger=logger)

        elif arguments.benchmark == 'cnn1':
           w = worker(factor=factor, run_id=arguments.run_id, host=host, logger=logger)

        elif arguments.benchmark == 'unet':
           w = worker(factor=factor, eta=arguments.eta,run_id=arguments.run_id, host=host, logger=logger)  

        elif arguments.benchmark == 'svm':
           w = worker(run_id=arguments.run_id, host=host, logger=logger)   

        else:
           w = worker( run_id=arguments.run_id, host=host, logger=logger)

        w.load_nameserver_credentials(working_directory=arguments.shared_dir)
        w.run(background=False)
        exit(0)

    result_logger = hpres.csv_results(directory=arguments.shared_dir, seed=arguments.seed, max_budget=max_budget, overwrite=True, logger=logger, factor=factor)

    # master and worker
    # Start a nameserver:
    NS = hpns.NameServer(run_id=arguments.run_id, host=host, port=0, working_directory=arguments.shared_dir)
    ns_host, ns_port = NS.start()

    # Start local worker
    if arguments.benchmark == 'cnn' or arguments.benchmark == 'rnn' or arguments.benchmark == 'multilayer'\
                or arguments.benchmark == 'cnn_time' or arguments.benchmark == 'rnn_time' or arguments.benchmark == 'multilayer_time'\
                or arguments.benchmark == 'nas' or arguments.benchmark == 'nas_time'\
                or "nats" in arguments.benchmark: 
        w = worker(dataset=arguments.benchmark, eta=arguments.eta, factor=factor, run_id=arguments.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, logger=logger)

    elif arguments.benchmark == 'cnn1':
        w = worker(factor=factor, run_id=arguments.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, logger=logger)        
    
    elif arguments.benchmark == 'unet':
        w = worker(factor=factor, eta=arguments.eta, run_id=arguments.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, logger=logger)
   
    elif arguments.benchmark == 'svm':
        w = worker(run_id=arguments.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, logger=logger)
    
    else:
        w = worker(run_id=arguments.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, logger=logger)
        
    w.run(background=True)


    t = time.time()
    # Run an optimizer
    if arguments.algorithm == 'HJ':
        alg = Hyperjump(configspace=worker.get_configspace(arguments.seed),
                       eta=arguments.eta,
                       run_id=arguments.run_id,
                       host=host,
                       nameserver=ns_host,
                       nameserver_port=ns_port,
                       result_logger=result_logger,
                       min_budget=min_budget, 
                       max_budget=max_budget,
                       random_fraction=arguments.random_fraction,
                       threshold=arguments.threshold,
                       logger=logger,
                      )

    elif arguments.algorithm == 'BOHB':

        alg = BOHB(configspace=worker.get_configspace(arguments.seed),
                      eta=arguments.eta,
                      run_id=arguments.run_id,
                      host=host,
                      nameserver=ns_host,
                      nameserver_port=ns_port,
                      result_logger=result_logger,
                      min_budget=min_budget, 
                      max_budget=max_budget,
                      logger=logger,
                      )

    elif arguments.algorithm == 'HB':
        alg = HyperBand(configspace=worker.get_configspace(arguments.seed),
                        eta=arguments.eta,
                        run_id=arguments.run_id,
                        host=host,
                        nameserver=ns_host,
                        nameserver_port=ns_port,
                        result_logger=result_logger,
                        min_budget=min_budget, 
                        max_budget=max_budget,
                        logger=logger,
                        )

    else:
        logger.error("Error: There is no optimizer named " + arguments.algorithm)
        sys.exit()

    res = alg.run(n_iterations=arguments.n_iterations, min_n_workers=args.n_workers)

    t_final = time.time() - t

    # shutdown
    alg.shutdown(shutdown_workers=True)
    NS.shutdown()

    result_logger.write_results()

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

    parser = argparse.ArgumentParser()

    parser.add_argument('--max_b', type=float, help='', default=81.0)
    parser.add_argument('--eta', type=float, help='eta', default=3.0)

    #n_iterations -> number of times to run the algorithm
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer (Number of generated brackets)',default=10)

    parser.add_argument('--seed', type=int, help='The seed for randomization', default=1)

    #type
    list_of_benchmarks = ["unet", "svm","cnn", "rnn", "multilayer", "nas", \
                            "cnn_time", "rnn_time", "multilayer_time", "nas_time", \
                            "nats-sss-cifar10", "nats-sss-cifar100", "nats-sss-ImageNet16-120", \
                            "nats-tss-cifar10", "nats-tss-cifar100", "nats-tss-ImageNet16-120"]
    parser.add_argument('--benchmark', type=str, choices=list_of_benchmarks, help='model to optimize', default='cnn')

    #algorithm
    parser.add_argument('--algorithm', type=str, choices=['HJ', 'BOHB','HB'], default='HB',help='Optimizer')

    #random fraction for Bohb
    parser.add_argument('--random_fraction', type=float, help='random_fraction', default=1/3)
    # threshold for hyperjump
    parser.add_argument('--threshold', type=float, help='threshold', default=0.1)

    # flag to set a worker
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')

    parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=1)
    #run_id
    parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
   

    args = parser.parse_args()

    np.random.seed(args.seed)

    now = datetime.datetime.now()

    if not os.path.exists("./logs/" ):
        os.mkdir("./logs/" )

    if not os.path.exists("./logs/" + args.benchmark):
        os.mkdir("./logs/" + args.benchmark)

    if not os.path.exists("logs/" + args.benchmark+ "/" + args.algorithm):
        os.mkdir("./logs/" + args.benchmark + "/" + args.algorithm)

    args.shared_dir = './logs/' + args.benchmark + '/' + args.algorithm + '/run_' + str(args.seed)+ '/'
    args.nic_name = netifaces.interfaces()[0]
    run(args)
