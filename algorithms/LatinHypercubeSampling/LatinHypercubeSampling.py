import numpy as np

def latin_hypercube(n_samples:int , dimension:int):
    """
    Generate a Latin Hypercube Sample of size n and dimension d.

    Parameters:
    - n_samples (int): The number of samples to generate.
    - dimension (int): The dimension of each sample.

    Returns:
    - numpy.ndarray: A n-by-d matrix of samples, where each row is a sample of length d.
    """

    samples = np.array([np.random.choice(range(1,n_samples+1), n_samples, replace=False) for _ in range(dimension)], dtype = float).reshape(2, -1)

    for i in range(dimension):
        for j in range(n_samples):
            samples[i,j] = (samples[i,j] - np.random.random())/n_samples
    return samples
