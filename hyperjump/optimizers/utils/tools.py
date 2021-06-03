import traceback

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np


def vector_to_conf(vec, type_exp):
    if type_exp == 'fake' or type_exp == 'fake_time':
        return vector_to_conf_fake(vec)
    elif type_exp == 'fake_all' or type_exp == 'fake_time_all':
        return vector_to_conf_fake_all(vec)        
    elif type_exp == 'unet':
        return vector_to_conf_unet(vec)
    elif type_exp == 'svm':
        return vector_to_conf_svm(vec)
    elif type_exp == 'mnist':
        return vector_to_conf_mnist(vec)
    else:
        raise BaseException(("Invalid/unimplemented experiment %s", type_exp))
 

def conf_to_vector(vec, type_exp):
    if type_exp == 'fake' or type_exp == 'fake_time':
        return conf_to_vector_fake(vec)
    elif type_exp == 'fake_all' or type_exp == 'fake_time_all':
        return conf_to_vector_fake_all(vec)
    elif type_exp == 'unet':
        return conf_to_vector_unet(vec)
    elif type_exp == 'svm':
        return conf_to_vector_svm(vec)
    else:
        return conf_to_vector_mnist(vec)


def vector_to_conf_fake(vec):
    '''
    Order:
        batch_size, Type: Categorical, Choices: {16, 256}, Default: 16
        learning_rate, Type: Categorical, Choices: {0,00001, 0,0001, 0,001}, Default: 0,00001
        num_cores, Type: Categorical, Choices: {8, 16, 32, 48, 64, 80}, Default: 8
        synchronism, Type: Categorical, Choices: {async, sync}, Default: async
        vm_flavor, Type: Categorical, Choices: {t2.small, t2.medium, t2.xlarge, t2.2xlarge}, Default: t2.small


    '''
    if len(vec) != 6:
        raise ValueError('Need list of len 6, you have: %s' % vec)

    new_vec = vec
    if vec[3] == 0:
        sync = 'sync'
    elif vec[3] == 1:
        sync = 'async'
    else:
        raise ValueError('Sync value is 0 or 1, you have: %s' % vec[3])

    if new_vec[4] == 0:
        vm_type = 't2.small'
    elif new_vec[4] == 1:
        vm_type = 't2.medium'
    elif new_vec[4] == 2:
        vm_type = 't2.xlarge'
    elif new_vec[4] == 3:
        vm_type = 't2.2xlarge'
    else:
        raise ValueError('VM_Type value is 0 to 3, you have: %s' % vec[4])

    conf_dict = dict([
        ('batch_size', float(new_vec[0])),
        ('learning_rate', float(new_vec[1])),
        ('num_cores', int(new_vec[2])),
        ('synchronism', sync),
        ('vm_flavor', vm_type)
    ])
    return conf_dict

def vector_to_conf_fake_all(vec):
    '''
    Order:
        network: cnn, mlp, rnn
        batch_size, Type: Categorical, Choices: {16, 256}, Default: 16
        learning_rate, Type: Categorical, Choices: {0,00001, 0,0001, 0,001}, Default: 0,00001
        num_cores, Type: Categorical, Choices: {8, 16, 32, 48, 64, 80}, Default: 8
        synchronism, Type: Categorical, Choices: {async, sync}, Default: async
        vm_flavor, Type: Categorical, Choices: {t2.small, t2.medium, t2.xlarge, t2.2xlarge}, Default: t2.small


    '''
    if len(vec) != 7:
        raise ValueError('Need list of len 6, you have: %s' % vec)
#{'batch_size': 16.0, 'learning_rate': 0.001, 'network': 'cnn', 'num_cores': 8, 'synchronism': 'sync', 'vm_flavor': 't2.small'}

    new_vec = vec
    #print(new_vec)

    if vec[4] == 0:
        sync = 'sync'
    elif vec[4] == 1:
        sync = 'async'
    else:
        raise ValueError('Sync value is 0 or 1, you have: %s' % vec[4])

    if new_vec[5] == 0:
        vm_type = 't2.small'
    elif new_vec[5] == 1:
        vm_type = 't2.medium'
    elif new_vec[5] == 2:
        vm_type = 't2.xlarge'
    elif new_vec[5] == 3:
        vm_type = 't2.2xlarge'
    else:
        raise ValueError('VM_Type value is 0 to 3, you have: %s' % vec[5])

    if new_vec[2] == 0:
        nn = "cnn"
    elif new_vec[2] == 1:
        nn = "mlp"
    elif new_vec[2] == 2:    
        nn = "rnn"
    else:
        raise ValueError('Network value is 0 to 2, you have: %s' % vec[2])


    conf_dict = dict([
        ('batch_size', float(new_vec[0])),
        ('learning_rate', float(new_vec[1])),
        ('network', nn),
        ('num_cores', int(new_vec[3])),
        ('synchronism', sync),
        ('vm_flavor', vm_type),
    ])
    #print(conf_dict)
    return conf_dict


def conf_to_vector_fake(conf):
    """
    conf does not possess budget

    Order:
        batch_size, Type: Categorical, Choices: {16, 256}, Default: 16
        learning_rate, Type: Categorical, Choices: {0,00001, 0,0001, 0,001}, Default: 0,00001
        num_cores, Type: Categorical, Choices: {8, 16, 32, 48, 64, 80}, Default: 8
        synchronism, Type: Categorical, Choices: {async, sync}, Default: async
        vm_flavor, Type: Categorical, Choices: {t2.small, t2.medium, t2.xlarge, t2.2xlarge}, Default: t2.small
    """

    if isinstance(conf, dict):
        dic = conf

    else:
        dic = conf.get_dictionary()

    try:
        if len(dic) != 5:
            print("Bad configuration")
            raise ValueError('This configuration is invalid: ', dic)

        vec = []
        for value in dic.values():
            vec.append(value)
        new_vec = vec
        sync = vec[3]
        if sync == 'sync':
            new_vec[3] = 0
        elif sync == 'async':
            new_vec[3] = 1
        else:
            raise ValueError('Sync value has {async, sync}, you have: %s' % vec[3])

        vm_type = vec[4]
        if vm_type == 't2.small':
            new_vec[4] = 0
        elif vm_type == 't2.medium':
            new_vec[4] = 1
        elif vm_type == 't2.xlarge':
            new_vec[4] = 2
        elif vm_type == 't2.2xlarge':
            new_vec[4] = 3
        else:
            raise ValueError('VM_Type value has {t2.small, t2.medium, t2.xlarge, t2.2xlarge}, you have: %s' % vec[4])

    except Exception as e:
        print("Error converting vector to configuration:\n%s" % dic)
        raise e

    return np.array(new_vec)


def conf_to_vector_fake_all(conf):
    """
    conf does not possess budget

    Order:
        batch_size, Type: Categorical, Choices: {16, 256}, Default: 16
        learning_rate, Type: Categorical, Choices: {0,00001, 0,0001, 0,001}, Default: 0,00001
        num_cores, Type: Categorical, Choices: {8, 16, 32, 48, 64, 80}, Default: 8
        synchronism, Type: Categorical, Choices: {async, sync}, Default: async
        vm_flavor, Type: Categorical, Choices: {t2.small, t2.medium, t2.xlarge, t2.2xlarge}, Default: t2.small
    """

    if isinstance(conf, dict):
        dic = conf

    else:
        dic = conf.get_dictionary()

    try:
        if len(dic) != 6:
            print("Bad configuration")
            raise ValueError('This configuration is invalid: ', dic)

        #print("\n---")
        #print(dic)
        vec = []
        for value in dic.values():
            vec.append(value)
        #print(vec)
        #print("---\n")
#{'batch_size': 16.0, 'learning_rate': 0.001, 'network': 'cnn', 'num_cores': 8, 'synchronism': 'sync', 'vm_flavor': 't2.small'}

        new_vec = vec
        nn = vec[2]
        if nn == 'cnn':
            new_vec[2] = 0
        elif nn == 'mlp':
            new_vec[2] = 1
        elif nn == 'rnn':
            new_vec[2] = 2
        else:
            raise ValueError('NN value has {cnn, mlp, rnn}, you have: %s' % vec[3])


        sync = vec[4]
        if sync == 'sync':
            new_vec[4] = 0
        elif sync == 'async':
            new_vec[4] = 1
        else:
            raise ValueError('Sync value has {async, sync}, you have: %s' % vec[4])

        vm_type = vec[5]
        if vm_type == 't2.small':
            new_vec[5] = 0
        elif vm_type == 't2.medium':
            new_vec[5] = 1
        elif vm_type == 't2.xlarge':
            new_vec[5] = 2
        elif vm_type == 't2.2xlarge':
            new_vec[5] = 3
        else:
            raise ValueError('VM_Type value has {t2.small, t2.medium, t2.xlarge, t2.2xlarge}, you have: %s' % vec[5])

    except Exception as e:
        print("Error converting vector to configuration:\n%s" % dic)
        raise e
    #print(np.array(new_vec))
    return np.array(new_vec)


def vector_to_conf_svm(vec):
    if len(vec) != 4:
        raise ValueError('Need list of len 4, you have: %s' % vec)

    new_vec = vec

    conf_dict = dict([
        ('kernel', int(new_vec[0])),
        #('degree', int(new_vec[1])),
        ('gamma', float(new_vec[1])),
        ('C', float(new_vec[2])),
    ])
    return conf_dict


def conf_to_vector_svm(conf):
    if isinstance(conf, dict):
        dic = conf

    else:
        dic = conf.get_dictionary()

    try:
        if len(dic) != 3:
            print("Bad configuration")
            raise ValueError('This configuration is invalid: ', dic)

        vec = []
        for value in dic.values():
            vec.append(value)


    except Exception as e:
        print("Error converting vector to configuration:\n%s" % dic)
        raise e

    return np.array(vec)


def vector_to_conf_unet(vec):
    """
    Configuration space object:
      Hyperparameters:
        Flavor, Type: Categorical, Choices: {intel14_v1, intel14_v2}, Default: intel14_v1
        batch, Type: Categorical, Choices: {1, 2}, Default: 1
        learningRate, Type: Categorical, Choices: {1e-06, 1e-05, 0.0001}, Default: 1e-06
        momentum, Type: Categorical, Choices: {0.9, 0.95, 0.99}, Default: 0.9
        nrWorker, Type: Categorical, Choices: {1, 2}, Default: 1
        sync, Type: Categorical, Choices: {async, sync}, Default: async
    """

    if len(vec) != 7:
        raise ValueError('Need list of len 7, you have: %s' % vec)

    new_vec = vec
    if vec[5] == 1:
        sync = 'async'
    elif vec[5] == 2:
        sync = 'sync'
    else:
        raise ValueError('Sync value is 1 or 2, you have: %s' % vec[3])

    if new_vec[0] == 1:
        vm_type = 'intel14_v1'
    elif new_vec[0] == 2:
        vm_type = 'intel14_v2'
    else:
        raise ValueError('VM_Type value is 0 to 3, you have: %s' % vec[4])

    conf_dict = dict([
        ('Flavor', vm_type),
        ('batch', int(new_vec[1])),
        ('learningRate', float(new_vec[2])),
        ('momentum', float(new_vec[3])),
        ('nrWorker', int(new_vec[4])),
        ('sync', sync)
    ])
    return conf_dict



def vector_to_conf_mnist(vec):
    """
    Order:
        ------------[0]-------------
        dropout_rate  ->  dropout_rate, Type: UniformInteger, Range: [1, 10], Default: 6
        ------------[1]-------------
        learning_rate  ->  learning_rate, Type: UniformInteger, Range: [1, 16], Default: 8
        ------------[2]-------------
        num_fc_units  ->  num_fc_units, Type: UniformInteger, Range: [1, 6], Default: 4
        ------------[3]-------------
        num_filters_1  ->  num_filters_1, Type: UniformInteger, Range: [1, 5], Default: 3
        ------------[4]-------------
        num_filters_2  ->  num_filters_2, Type: UniformInteger, Range: [1, 5], Default: 3
        ------------[5]-------------
        num_filters_3  ->  num_filters_3, Type: UniformInteger, Range: [1, 5], Default: 3
        ------------[6]-------------
        sgd_momentum  ->  sgd_momentum, Type: UniformInteger, Range: [1, 100], Default: 90
    """

    try:
        conf_dict = dict([

            ('dropout_rate', float(vec[0])),
            ('learning_rate', float(vec[1])),
            ('num_fc_units', int(vec[2])),
            ('num_filters_1', int(vec[3])),
            ('num_filters_2', int(vec[4])),
            ('num_filters_3', int(vec[5])),
            ('sgd_momentum', float(vec[6]))
        ])
    except Exception as e:
        print("Error converting vector to dict: ", vec)
        raise e
    return conf_dict


def conf_to_vector_unet(conf):
    """
    Configuration space object:
      Hyperparameters:
        Flavor, Type: Categorical, Choices: {intel14_v1, intel14_v2}, Default: intel14_v1
        batch, Type: Categorical, Choices: {1, 2}, Default: 1
        learningRate, Type: Categorical, Choices: {1e-06, 1e-05, 0.0001}, Default: 1e-06
        momentum, Type: Categorical, Choices: {0.9, 0.95, 0.99}, Default: 0.9
        nrWorker, Type: Categorical, Choices: {1, 2}, Default: 1
        sync, Type: Categorical, Choices: {async, sync}, Default: async
    """

    if isinstance(conf, dict):
        dic = conf

    else:
        dic = conf.get_dictionary()

    try:
        if len(dic) != 6:
            print("Bad configuration")
            raise ValueError('This configuration is invalid: ', dic)

        vec = []
        for value in dic.values():
            vec.append(value)

        new_vec = vec
        sync = vec[5]
        if sync == 'sync':
            new_vec[5] = 2.0
        elif sync == 'async':
            new_vec[5] = 1.0
        else:
            raise ValueError('Sync value has {async, sync}, you have: %s' % vec[5])

        vm_type = vec[0]
        if vm_type == 'intel14_v1':
            new_vec[0] = 1.0
        elif vm_type == 'intel14_v2':
            new_vec[0] = 2.0
        else:
            raise ValueError('VM_Type value has {intel14_v1, intel14_v2}, you have: %s' % vec[4])

    except Exception as e:
        print("Error converting vector to configuration:\n%s" % dic)
        raise e

    return np.array(new_vec)



def conf_to_vector_mnist(conf):
    """
    conf does not possess budget
    """

    if isinstance(conf, dict):
        dic = conf

    else:
        dic = conf.get_dictionary()

    try:
        if len(dic) != 7:
            print("Bad configuration")
            raise ValueError('This configuration is invalid: ', dic)

        new_vec = []
        for value in dic.values():
            new_vec.append(value)

    except Exception as e:
        print("Error converting vector to configuration:\n%s" % dic)
        raise e

    return np.array(new_vec)


def get_incumbent(training_set, losses, budget):
    cur_min = 1
    cur_min_index = 0
    for i in range(len(training_set)):
        if training_set[i][-1] == budget:
            if losses[i] < cur_min:
                cur_min = losses[i]
                cur_min_index = i
    return cur_min, training_set[cur_min_index], cur_min_index + 1


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def neat_time_print(pred_time, ignore_sufix=False):
    s = pred_time % 60
    m = ((pred_time - s) / 60) % 60
    h = (pred_time - s - m * 60) / 3600
    s1 = s + 100
    m1 = m + 100
    h1 = h + 1000
    if ignore_sufix:
        return "h%s:m%s:s%s                   " % ((str(int(h1))[-3:]), str(int(m1))[-2:], str(round(s1))[-2:])
    return "Complete  Finished in: h%s:m%s:s%s" % ((str(int(h1))[-3:]), str(int(m1))[-2:], str(round(s1))[-2:])


def print_custom_bar(total_time, config_num, total_results):
    avg_time = total_time / config_num
    pred_time = avg_time * (total_results - config_num)
    suffix = neat_time_print(pred_time)
    if config_num != total_results:
        print_progress_bar(config_num, total_results, prefix='Progress:', suffix=suffix, length=50)
    else:
        print_progress_bar(config_num, total_results, prefix='Progress:', suffix=suffix, length=50, print_end="\n")
        print("Time passed: ", neat_time_print(total_time),
              "\nAvg time per run: ", neat_time_print(float(total_time) / float(total_results)))
