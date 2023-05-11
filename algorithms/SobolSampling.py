import numpy as np
from scipy.stats import qmc
from typing import List
import warnings


def sobol_sampling(sampling_budget:int , dimension:int=2, boundaries=[[-5,-5],[5,5]]):
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

    samples = qmc.Sobol(scramble=False, d=dimension)
    samples = samples.random(sampling_budget, workers=-1)
    samples = qmc.scale(samples, boundaries[0], boundaries[1])

    return samples


def sobol(fun, sampling_budget:int=100 , dimension:int=2, boundaries=[[-5,-5],[5,5]]):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_log = sobol_sampling(sampling_budget=sampling_budget , dimension=dimension, boundaries=boundaries)
        
    Y_log = fun(X_log.T)

    idx = np.argmin(Y_log)
    X_best, Y_best  = X_log[idx], Y_log[idx]

    return X_best, Y_best, X_log, Y_log