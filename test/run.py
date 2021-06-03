#from operator import ne
import subprocess
import os

hyperjump = "y"
algorithm = "BOHB++"
#algorithm = "BOHB-TPE"
#algorithm = "BOHB-EI"
algorithm_variant = "FBS"
#algorithm_variant = "FBS_DT" 
random_fraction = "0.0"

eta="2"


if algorithm == "BOHB-TPE":
    hyperjump = "n"
    threshold_list = [1.0]
    random_fraction = "0.3"
else:
    #threshold_list = [0.1, 0.01, 0.001, 1.0]
    #threshold_list = [0.001, 1.0]
    threshold_list = [0.1]

#for t in [ 0.1, 0.25, 0.01, 0.5]:  
#for t in [0.001, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:  
#for t in [0.001, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:  
for t in threshold_list:  
    #for network in ["cnn", "rnn", "multilayer", "unet"]:
    #for network in ["unet"]:
    #for network in ["svm"]:
    
    
    #for network in ["cnn_time", "multilayer_time", "rnn_time", "all_time"]: #["multilayer"]:
    #for network in ["cnn_time"]: #["multilayer"]:
    #for network in ["multilayer_time"]: 
    #for network in ["rnn_time"]: 
    #for network in ["all_time"]: #["multilayer"]:
    
    #for network in ["cnn"]:
    for network in ["unet"]:
    #for network in ["multilayer"]:
    #for network in ["all"]:
    #for t in [0,  0.001, 0.75, 1.5, 100]: #, 0.75, 1]:   
    #for t in [0.01, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50]:   

        if network == "unet": 
            if eta ==  "3":
                command = "python3 fake_workload.py"
                command += " --algorithm " + algorithm
                command += " --type " + network 
                command += " --eta " + eta 
                command += " --n_iterations 20 --min_budget 222 --max_budget 18000 --algorithm_variant " + algorithm_variant
                command += " --hyperjump " + hyperjump
                command += " --random_fraction " + random_fraction + " --threshold " + str(t)

            else:
                command = "python3 fake_workload.py"
                command += " --algorithm " + algorithm
                command += " --type " + network 
                command += " --eta " + eta 
                command += " --n_iterations 20 --min_budget 1125 --max_budget 18000 --algorithm_variant " + algorithm_variant
                command += " --hyperjump " + hyperjump
                command += " --random_fraction " + random_fraction + " --threshold " + str(t)


        elif network == "svm": 
            command = "python3 fake_workload.py"
            command += " --algorithm " + algorithm
            command += " --type " + network 
            #command += " --eta 2" 
            #command += " --n_iterations 20 --min_budget 1736 --max_budget 27777 --algorithm_variant " + algorithm_variant
            
            command += " --eta 3" 
            #command += " --n_iterations 20 --min_budget 342 --max_budget 27777 --algorithm_variant " + algorithm_variant
            #command += " --n_iterations 20 --min_budget 3086 --max_budget 250000 --algorithm_variant " + algorithm_variant
            command += " --n_iterations 20 --min_budget 1028 --max_budget 83333 --algorithm_variant " + algorithm_variant
            
            #command += " --n_iterations 20 --min_budget 617 --max_budget 50000 --algorithm_variant " + algorithm_variant
            #command += " --n_iterations 20 --min_budget 617 --max_budget 5555 --algorithm_variant " + algorithm_variant
            command += " --hyperjump " + hyperjump
            command += " --random_fraction " + random_fraction + " --threshold " + str(t)

        else:

            if "time" in network:
                if eta == 2:
                    command = "python3 fake_workload.py"
                    command += " --algorithm " + algorithm
                    command += " --dataset " + network
                    command += " --eta 2"  
                    command += " --n_iterations 20 --min_budget 37 --max_budget 600 --algorithm_variant " + algorithm_variant
                    command += " --hyperjump " + hyperjump
                    command += " --random_fraction " + random_fraction + " --threshold " + str(t)

                else:
                    command = "python3 fake_workload.py"
                    command += " --algorithm " + algorithm
                    command += " --dataset " + network
                    command += " --eta 3"  
                    command += " --n_iterations 20 --min_budget 7 --max_budget 600 --algorithm_variant " + algorithm_variant
                    command += " --hyperjump " + hyperjump
                    command += " --random_fraction " + random_fraction + " --threshold " + str(t)                 

            else:
                command = "python3 fake_workload.py"
                command += " --algorithm " + algorithm
                command += " --dataset " + network
                command += " --eta 2"  
                command += " --n_iterations 20 --min_budget 3750 --max_budget 60000 --algorithm_variant " + algorithm_variant
                command += " --hyperjump " + hyperjump
                command += " --random_fraction " + random_fraction + " --threshold " + str(t)

        print(command)
        subprocess.run(command, shell=True)
                
        name_dir = "logs/" + network + "/threshold_" + str(t)
        os.mkdir(name_dir) 

        command = "mv logs/" + network + "/" + algorithm + "_" + algorithm_variant +  " " + name_dir + "/GPs_fab_log_DT"
        subprocess.run(command, shell=True)

        #if hyperjump == "n":
        #    break


#python3 fake_workload.py --algorithm HB --type unet --n_iterations 20 --min_budget 1125 --max_budget 18000