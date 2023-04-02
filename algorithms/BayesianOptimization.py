import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from typing import List
from scipy.stats import norm
import json, sys, argparse, ast


def surrogate_model(model: GaussianProcessRegressor, X):
    prediction = model.predict(X.T, return_std=True)
    return prediction

def acquisition(model, X, samples):
    yhat, _ = surrogate_model(model, X)
   
    best    = np.max(yhat)
    mu, std = surrogate_model(model, samples)

    probs = norm.cdf((mu - best) / (std + 1e-9))

    return probs



def bayesian_optimization(f, dimension:int=2, n_samples:int=15, sampling_budget:int=100, boundaries=List[float]):

    # Modify the function to obtain the minimization
    fun = lambda x: (-1)*f(x)

    boundaries = np.array(boundaries)
    # Decide new samples to analyze
    X =  np.random.uniform(low=boundaries[0], high=boundaries[1], size=[n_samples, dimension]).T

    # Evaluate the function
    Y = fun(X)

    # GaussianProcessRegressor(kernel=kernel)
    model = GaussianProcessRegressor()

    # Training of the model
    model.fit(X.T, Y)
    max_iter = sampling_budget - n_samples

    for iter in range(max_iter):

        # Define random sample on which test the model
        samples = np.random.uniform(low=boundaries[0], high=boundaries[1], size=[n_samples, dimension]).T

        # Rate the samples according to their score
        scores = acquisition(model, X, samples)

        # Find the best for the minimization (the smallest)
        idx = np.argmax(scores)

        # New point
        X_new = samples[:, idx].reshape(-1,1)

        # Find the actual value
        Y_new = fun(X_new)
        #print("STATUS:", model.predict(X_new.T)[0])
        #est, _ = surrogate_model(model, X_new)
        #print(f"Estimation error:{Y_new - est}")

        X = np.hstack((X, X_new))
        Y = np.hstack((Y, Y_new))

        model.fit(X.T, Y)
    # fun was inverted. Let's revert it
    Y = -1*Y

    idx = np.argmin(Y)
    X_best, Y_best = X[:, idx], Y[idx]

    return X_best, Y_best, X.T, Y.T