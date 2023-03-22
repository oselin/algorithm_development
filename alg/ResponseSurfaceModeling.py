from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
from typing import List




# Sampling methods
def box_behnken(X:np.array,low=0.1, high=0.1) -> np.array:

    points = [X]
    for i in range(len(X)):
        x_lower, x_higher = X.copy(), X.copy()
        x_lower[i] -= low
        x_higher[i] += high
        points.append(x_lower)
        points.append(x_higher)

    return np.array(points).T


def central_composite(X:np.array, low=0.1, high=0.1) -> np.array:
    return


# Gradient estimation via finite difference
def gradient(model:Pipeline, X:List[float], y:float, delta_h:float=1e-5):
    
    # Create an array of zeros to allocate memory
    grad = np.zeros(len(X))

    for i in range(len(X)):
        Xh = X.copy() 
        # Add delta H in the i-th dimension
        Xh[i] += delta_h

        # Estimate the gradient via finite difference
        grad[i] = (model.predict([Xh]) - y) / delta_h
    return grad


def next_step(model:Pipeline, X:List[float],Y:List[float], lr:float=0.01, method="gradient"):

    if (method == "gradient"):

        # Find the local minimum in the considered set
        min_idx = np.argmin(Y)
        # Apply the gradient descend to find the new point to analyze
        X_next = X[min_idx] - lr*gradient(model,X[min_idx], Y[min_idx])

    return X_next

def response_surface(fun, 
                     X_new:List[float]=None, Xmin=None, Xmax=None, dimension=2, 
                     iterations=100, 
                     sampling_method="box_behnken", sampling_lb=0.5, sampling_ub=0.5,
                     iteration_method="gradient", alpha=0.1
                     ):

    model = Pipeline([('poly',   PolynomialFeatures(degree=2)),
                    ('linear', LinearRegression(fit_intercept=False))]) 


    
    #if (X == None): X = np.random.uniform(low=Xmin, high=Xmax, size=[1,dimension])
    X_h, Y_h = [], []
    #alpha = np.linspace(0.01, 0.005, 10)

    for iter in range(iterations):

        # Find new samples to analyze
        #if (sampling_method == "box_behnken"): 
        samples = box_behnken(X_new, low=sampling_lb, high=sampling_ub)

        # Evaluate the function for those samples
        Y = fun(samples)

        # Fit the model on those few samples
        model.fit(samples.T,Y)

        # Find the new starting point for the next iteration
        X_new = next_step(model, samples.T,Y, lr=0.01)

        # Update the set of analyzed points with the ones of this iteration
        X_h.append(samples[:,np.argmin(Y)])
        Y_h.append(np.min(Y))
    
    return np.array(X_h), np.array(Y_h)