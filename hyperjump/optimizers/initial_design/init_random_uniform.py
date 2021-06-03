
import numpy as np


def init_random_uniform(lower, upper, n_points, rng=None):
    """
    Samples N data points uniformly.

    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bounds of the input space
    upper: np.ndarray (D)
        Upper bounds of the input space
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

    return np.array([rng.uniform(lower, upper, n_dims) for _ in range(n_points)])





######################
##wrong - do not use
#####################

def init_random_uniform_Tensorflow(lower, upper, n_points, rng=None):
    """
    Samples N data points uniformly.

    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bounds of the input space
    upper: np.ndarray (D)
        Upper bounds of the input space
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

    #sample[0] = nr of workers
    #sample[1] = learning rate
    #sample[2] =  batch size
    #sample[3] = synchronism
    #sample[4] = flavor
    #sample[5] = size
    samples = np.array([rng.uniform(lower, upper, n_dims) for _ in range(n_points)])

    for i in range(n_points):
        #synchronism 0 to 2 -->   0 = async; 1= Sync
        samples[i,3] = int(samples[i,3]) if samples[i,2]!= 2 else 1

        # 0 = t2.small; 1 = t2.meduim; 2 = t2.xlarge; 3 = t2.2xlarge
        samples[i,4] = int(samples[i,4]) if samples[i,4]!= 4 else 3



        if samples[i,4] == 0:
            if samples[i,0] < 12:
                samples[i,0] = 8
            elif samples[i,0] >= 12 and samples[i,0] < 24:
                samples[i,0] = 16
            elif samples[i,0] >= 24 and samples[i,0] < 40:
                samples[i,0] = 32
            elif samples[i,0] >= 40 and samples[i,0] < 56:
                samples[i,0] = 48
            elif samples[i,0] >= 56 and samples[i,0] < 72:
                samples[i,0] = 64
            else:
                 samples[i,0] = 80

        elif samples[i,4] == 1:
            if samples[i,0] < 6:
                samples[i,0] = 4
            elif samples[i,0] >= 6 and samples[i,0] < 12:
                samples[i,0] = 8
            elif samples[i,0] >= 12 and samples[i,0] < 20:
                samples[i,0] = 16
            elif samples[i,0] >= 20 and samples[i,0] < 28:
                samples[i,0] = 24
            elif samples[i,0] >= 28 and samples[i,0] < 36:
                samples[i,0] = 32
            else:
                samples[i,0] = 40

        elif samples[i,4] == 2:
            if samples[i,0] < 3:
                samples[i,0] = 2
            elif samples[i,0] >= 3 and samples[i,0] < 6:
                samples[i,0] = 4
            elif samples[i,0] >= 6 and samples[i,0] < 10:
                samples[i,0] = 8
            elif samples[i,0] >= 10 and samples[i,0] < 14:
                samples[i,0] = 12
            elif samples[i,0] >= 14 and samples[i,0] < 18:
                samples[i,0] = 16
            else:
                samples[i,0] = 20

        else:
            if samples[i,0] < 1.5:
                samples[i,0] = 1
            elif samples[i,0] >= 1.5 and samples[i,0] < 3:
                 samples[i,0] = 2
            elif samples[i,0] >= 3 and samples[i,0] < 5:
                 samples[i,0] = 4
            elif samples[i,0] >= 5 and samples[i,0] < 7:
                 samples[i,0] = 6
            elif samples[i,0] >= 7 and samples[i,0] < 9:
                 samples[i,0] = 8
            else:
                samples[i,0] = 10


        if samples[i,1] < 0.000055:
            samples[i,1] = 0.00001
        elif samples[i,1] >= 0.000055 and samples[i,1] < 0.00055:
            samples[i,1] = 0.0001
        else:
            samples[i,1] = 0.001

        if samples[i,2] < 136:
            samples[i,2] = 16
        else:
            samples[i,2] = 256


        if samples[i,5] < 3.5/60.0:
            samples[i,5] = 10000
        elif samples[i,5] >= 3.5/60.0 and samples[i,5] < 1.05/6.0:
            samples[i,5] = 6000
        elif samples[i,5] >= 1.05/6.0 and samples[i,5] < 2.25/6.0:
            samples[i,5] = 15000
        elif samples[i,5] >= 2.25/6.0 and samples[i,5] < 4.5/6.0:
            samples[i,5] = 30000
        else:
            samples[i,5] = 60000


    return samples
