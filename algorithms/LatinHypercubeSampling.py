import numpy as np
from scipy.stats import qmc
from typing import List



def latin_hypercube_sampling(n_samples:int , dimension:int, boundaries:List[float]):
    """
    Generate a Latin Hypercube Sample of size n and dimension d.

    Parameters:
    - n_samples (int): The number of samples to generate.
    - dimension (int): The dimension of each sample.
    - lower_bounds: List[float] N-dimensional array with the lower bounds values
    - upper_bounds: List[float] N-dimensional array with the upper bounds values

    Returns:
    - numpy.ndarray: A n-by-d matrix of samples, where each row is a sample of length d.
    """

    boundaries = np.array(boundaries)

    samples = qmc.LatinHypercube(scramble=False,d=dimension)
    samples = samples.random(n_samples)
    samples = qmc.scale(samples, boundaries[0], boundaries[1])

    """
    samples = np.array([np.random.choice(range(1,n_samples+1), n_samples, replace=False) for _ in range(dimension)], dtype = float).reshape(2, -1)

    for i in range(dimension):
        for j in range(n_samples):
            samples[i,j] = (samples[i,j] - np.random.random())/n_samples
    """
    return samples

def latin_hypercube(fun, n_samples:int , dimension:int, boundaries:List[float]):

    X_log = latin_hypercube_sampling(n_samples=n_samples , dimension=dimension, boundaries=boundaries)
    Y_log = fun(X_log.T)

    idx = np.argmin(Y_log)
    X_best, Y_best  = X_log[idx], Y_log[idx]

    return X_best, Y_best, X_log, Y_log