from scipy.optimize import minimize
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from typing import List


def gradient(model:Pipeline, X:List[float], y:float, delta_h:float=1e-5):
    """
    Gradient function via finite difference
    """
    # Create an array of zeros to allocate memory
    grad = np.zeros(len(X))

    for i in range(len(X)):
        Xh = X.copy() 
        # Add delta H in the i-th dimension
        Xh[i] += delta_h
        # Estimate the gradient via finite difference
        grad[i] = (model.predict([Xh]) - y) / delta_h

    return grad



def BFGS(fun, x0=None, low=-5, high=5, max_iter=100):


    if (x0 is None): x0 = np.random.uniform(low=low, high=high, size=[1,2])

    model = Pipeline([('poly',   PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))]) 


    # Log arrays
    X_log, Y_log = [], []


    #alpha = np.linspace(0.01, 0.005, 10)

    for iter in range(max_iter):

        # Evaluate the function for those samples
        Y = fun(x0)

        # Fit the model on those few samples
        model.fit(x0,Y)

