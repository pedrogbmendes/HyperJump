
import numpy as np
import random

def init_latin_hypercube_sampling(lower, upper, n_points, rng=None):
    """
    Returns as initial design a N data points sampled from a latin hypercube.

    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bound of the input space
    upper: np.ndarray (D)
        Upper bound of the input space
    n_points: int
        The number of initial data points
    rng: np.random.RandomState
            Random number generator

    Returns
    -------
    np.ndarray(N,D)
        The initial design data points
    """
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))
    n_dims = lower.shape[0]
    # Generate bounds for random number generator
    s_bounds = np.array([np.linspace(lower[i], upper[i], n_points + 1) for i in range(n_dims)])
    s_lower = s_bounds[:, :-1]
    s_upper = s_bounds[:, 1:]
    # Generate samples
    samples = s_lower + rng.uniform(0, 1, s_lower.shape) * (s_upper - s_lower)
    # Shuffle samples in each dimension
    for i in range(n_dims):
        rng.shuffle(samples[i, :])
    return samples.T




def init_latin_hypercube_sampling_Tensorflow(seed, lower, upper, n_points, rng=None):
    """
    Returns as initial design a N data points sampled from a latin hypercube.

    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bound of the input space
    upper: np.ndarray (D)
        Upper bound of the input space
    n_points: int
        The number of initial data points
    rng: np.random.RandomState
            Random number generator

    Returns
    -------
    np.ndarray(N,D)
        The initial design data points
    """
    random.seed(seed)


    n_dimension = 6
    samples = np.zeros((n_points, n_dimension))

    flavor = [0,1,2,3]
    learning_rate = [0.001, 0.0001, 0.00001]
    batch_size = [16, 256]
    synchronism = [0,1]
    nr_worker_cores = [8,16,32,48,64,80]

    random.shuffle(flavor)
    random.shuffle(learning_rate)
    random.shuffle(batch_size)
    random.shuffle(synchronism)
    random.shuffle(nr_worker_cores)


    c_fl = 0
    c_lr = 0
    c_bs = 0
    c_sy = 0
    c_cores = 0

    for i in range(0,n_points):
        fl = flavor[c_fl]
        c_fl +=1 if c_fl < len(flavor)-1 else -c_fl

        lr = learning_rate[c_lr]
        c_lr +=1 if c_lr < len(learning_rate)-1 else -c_lr

        bs = batch_size[c_bs]
        c_bs +=1 if c_bs < len(batch_size)-1 else -c_bs

        sync = synchronism[c_sy]
        c_sy +=1 if c_sy < len(synchronism)-1 else  -c_sy

        cores = nr_worker_cores[c_cores]
        c_cores +=1 if c_cores < len(nr_worker_cores)-1 else -c_cores

        #sample[0] = nr of workers
        #sample[1] = learning rate
        #sample[2] =  batch size
        #sample[3] = synchronism
        #sample[4] = flavor
        #sample[5] = budget


        #[learning_rate, batch_size, synchronism, num_cores, vm_flavor]
        samples[i,0] = bs
        samples[i,1] = lr
        samples[i,2] = cores
        samples[i,3] = sync
        samples[i,4] = fl
        samples[i,5] = 3750


    return samples
