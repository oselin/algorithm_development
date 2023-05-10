import numpy as np
from scipy.stats import qmc
import warnings

def sobol_sampling(n_samples:int , dimension:int, boundaries):
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
    samples = samples.random(n_samples, workers=-1)
    samples = qmc.scale(samples, boundaries[0], boundaries[1])

    return samples

def latin_hypercube_sampling(n_samples:int , dimension:int, boundaries):
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
    
    return samples


def particle_swarm(fun, dimension:int=2, boundaries=[[-5,-5],[5,5]], vel_boundaries=[[0,0],[0.1,0.1]], sampling_budget:int=100, 
                   sampling_method="sobol", n_particles=10, tollerance=10e-6, c1=0.1, c2=0.1, w=0.8):

    boundaries = np.array(boundaries)
    vel_boundaries = np.array(vel_boundaries)
    
    if (sampling_method == "sobol"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = sobol_sampling(n_samples=n_particles, dimension=dimension, boundaries=boundaries).T + np.random.uniform(low=-0.2, high=0.2, size=(n_particles, dimension)).T
    elif (sampling_method == "random"):
        X = np.random.uniform(low=boundaries[0], high=boundaries[1], size=(n_particles, dimension)).T
    elif (sampling_method == "latin_hypercube"):
        X = latin_hypercube_sampling(n_samples=n_particles, dimension=dimension, boundaries=boundaries).T + np.random.uniform(low=-0.2, high=0.2, size=(n_particles, dimension)).T
    else:
        # Default choice
        X = np.random.uniform(low=boundaries[0], high=boundaries[1], size=(n_particles, dimension)).T

    V = np.random.uniform(low=vel_boundaries[0], high=vel_boundaries[1], size=(n_particles, dimension)).T

    # Initialize data for the first iteration
    # Set the best coord for i-th particle + store f(x) values
    pbest = X
    pbest_obj = fun(X)

    # Find the global best and its associated function value
    gbest = pbest[:,pbest_obj.argmin()].reshape(-1,1)
    gbest_obj = pbest_obj.min() 

    # Initalize the log variables
    X_log, Y_log = gbest.copy(), np.array([gbest_obj])
    
    max_iter = (sampling_budget//n_particles) - 1

    for iteration in range(max_iter):
        # Set the random coefficients
        r1, r2 = np.random.rand(2)

        # Update the velocity
        V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest-X)
        # Update the position
        X = X + V

        # Evaluate the function for each particle
        obj = fun(X)

        # Find the right index at which update the values
        idx = (pbest_obj >= obj)

        pbest[:,idx] = X[:,idx]

        #print(np.array([pbest_obj, obj]))
        pbest_obj = np.vstack((pbest_obj, obj)).min(axis=0)

        gbest = pbest[:, pbest_obj.argmin()].reshape(-1,1)
        gbest_obj = pbest_obj.min()

        # Log the result
        X_log = np.hstack((X_log, gbest.copy()))
        Y_log = np.hstack((Y_log, gbest_obj.copy()))

        # if (np.abs(Y_log[-1] - Y_log[-2]) < tollerance):
        #     print(f"Tollearnce reacehed, {iteration} iterations is enough.")
        #     break

    X_best, Y_best = X_log[:, -1], Y_log[-1]

    return X_best, Y_best, X_log.T, Y_log.T