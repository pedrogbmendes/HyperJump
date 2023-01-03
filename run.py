

import subprocess
import sys

benchmarks = ["cnn","rnn_time","multilayer_time","unet","nas","nats-tss-cifar10","nats-tss-cifar100","nats-tss-ImageNet16-120"]
benchmarks = ["nats-tss-ImageNet16-120"]


optimizers = ['HB', 'BOHB', 'HJ']
n_iterations = 2
n_workers = 1

for benchmark in benchmarks:
    for algorithm in optimizers:

        if benchmark == 'cnn' or benchmark == 'nas':
            max_b = 16 # dataset size as budget
            eta = 2
        elif benchmark == 'rnn_time' or benchmark == 'multilayer_time':
            max_b = 81 # training time as budget
            eta = 3
        elif benchmark == 'unet':
            max_b = 16 # training time as budget
            eta = 2
        elif benchmark == "nats-tss-ImageNet16-120" or  benchmark == "nats-tss-cifar100":
            max_b = 81 # number of epochs as budget
            eta = 3
        elif benchmark == "nats-tss-cifar10":
            max_b = 16 # number of epochs as budget
            eta = 2
        else:
            print("Wrong benchmark " + benchmark)
            sys.exit(0)


        if algorithm == 'BOHB':
            random_fraction = 1.0/3.0
            threshold = 0
        elif algorithm == 'HB':
            random_fraction = 1.0
            threshold = 0
        elif algorithm == 'HJ':
            random_fraction = 0.0
            threshold = 0.1


        for seed in range(1, 2):

            command = "python3 launcher.py --max_b=" + str(max_b) + " --eta=" + str(eta) +  " --n_iterations=" + str(n_iterations) + \
                " --seed=" + str(seed) +  " --benchmark=" + str(benchmark) + " --algorithm=" + str(algorithm) +  " --random_fraction=" + str(random_fraction) + \
                " --threshold=" + str(threshold) + " --n_workers=" + str(n_workers) + " --run_id=" + str(seed)
            print("--------------------------")
            print("-")
            print(command)
            print("-")
            print("--------------------------")


            command_worker = command + " --worker"
            for i in range(1, n_workers):
                print("Start a new worker " + str(i))
                subprocess.Popen(command_worker, shell=True) # start the workers 


            print("Start the master and the first worker 0 ")
            subprocess.run(command, shell=True) # start the master 
