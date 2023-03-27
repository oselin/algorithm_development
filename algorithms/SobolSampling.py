import numpy as np
from scipy.stats import qmc
from typing import List


def sobol_sampling(n_samples:int , dimension:int, lower_bounds:List[float], upper_bounds:List[float]):
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

    samples = qmc.Sobol(scramble=False, d=dimension)
    samples = samples.random(n_samples, workers=-1)
    samples = qmc.scale(samples, lower_bounds, upper_bounds)

    return samples


def sobol(fun, n_samples:int , dimension:int, lower_bounds:List[float], upper_bounds:List[float]):

    X_log = sobol_sampling(n_samples=n_samples , dimension=dimension, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    Y_log = fun(X_log.T)

    idx = np.argmin(Y_log)
    X_best, Y_best  = X_log[idx], Y_log[idx]

    return X_best, Y_best, X_log, Y_log