import numpy as np
from scipy.stats import qmc
from typing import List



def latin_hypercube(n_samples:int , dimension:int, lower_bounds:List[float], upper_bounds:List[float]):
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

    samples = qmc.LatinHypercube(scramble=False,d=dimension)
    sample = samples.random(n_samples)
    qmc.scale(sample, lower_bounds, upper_bounds)

    """
    samples = np.array([np.random.choice(range(1,n_samples+1), n_samples, replace=False) for _ in range(dimension)], dtype = float).reshape(2, -1)

    for i in range(dimension):
        for j in range(n_samples):
            samples[i,j] = (samples[i,j] - np.random.random())/n_samples
    """
    return np.array(sample)
