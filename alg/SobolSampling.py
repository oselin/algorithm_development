import numpy as np
from scipy.stats import qmc
from typing import List


def sobol(n_samples:int , dimension:int, lower_bounds:List[float], upper_bounds:List[float]):
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
    sample = samples.random(n_samples, workers=-1)
    qmc.scale(sample, lower_bounds, upper_bounds)

    return sample