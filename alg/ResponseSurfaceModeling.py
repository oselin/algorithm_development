from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
from typing import List




def box_behnken(X:np.array, bound) -> np.array:
    """
    Box Behnken design function
    """
    points = [X]
    for i in range(len(X)):
        x_lower, x_higher = X.copy(), X.copy()
        x_lower[i]  -= bound
        x_higher[i] += bound
        points.append(x_lower)
        points.append(x_higher)

    return np.array(points)

def central_composite(X:np.array, bound) -> np.array:
    """
    Central composite design function
    """
    def new_row(idx):
        row = [0 for _ in range(len(X))]
        for arr_idx in range(len(row)):
            row[arr_idx] = (-1)**(idx//(2**arr_idx)) * bound
        return row
    
    permutations = []
    for idx in range(2**len(X)): permutations.append(new_row(idx))

    permutations = np.array(permutations)
    return np.vstack((X,X + permutations))



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


def expected_improvement(model:Pipeline, X:List[float]):
    """
    Expected improvement function from statistic
    """
    #y_hat = np.max(model.predict(X))

    mu, std = model.predict(X, return_std=True)
    print(mu)


def next_step(model:Pipeline, X:List[float],Y:List[float], lr:float=0.01, method="gradient"):
    """
    Next step function: find the best point to sample at next iteration
    """
    if (method == "gradient"):

        # Find the local minimum in the considered set
        min_idx = np.argmin(Y)
  
        X_best = X[min_idx]
        Y_best = Y[min_idx]
        # Apply the gradient descend to find the new point to analyze
        X_next = X[min_idx] - lr*gradient(model,X[min_idx], Y[min_idx])

    return X_next, X_best, Y_best
    #else:
    #    expected_improvement(model, X)


def response_surface(fun, X_new:List[float], iterations=100, tol = 1e-8, sampling_method="box_behnken", sampling_bound=0.5,
                     iteration_method="gradient", alpha=0.01):

    model = Pipeline([('poly',   PolynomialFeatures(degree=2)),
                    ('linear', LinearRegression(fit_intercept=False))]) 


    # Log arrays
    X_log, Y_log = [], []

    # If type(bounds) == list, iterative mode is requested
    if (type(sampling_bound)== list):
        sampling_bound = np.linspace(sampling_bound[0], sampling_bound[1], iterations)

    #alpha = np.linspace(0.01, 0.005, 10)

    for iter in range(iterations):

        # Find new samples to analyze
        if (sampling_method == "box_behnken"): 
            samples = box_behnken(X_new, sampling_bound[iter])
        elif (sampling_method == "central_composite"):
            samples = central_composite(X_new, sampling_bound[iter])


        # Evaluate the function for those samples
        Y = fun(samples)

        # Fit the model on those few samples
        model.fit(samples,Y)

        # Find the new starting point for the next iteration
        X_new, X_best, Y_best = next_step(model, samples,Y, lr=alpha, method=iteration_method)

        # Update the set of analyzed points with the ones of this iteration
        X_log.append(X_best)
        Y_log.append(Y_best)

        # Convergence condition
        if (len(Y_log) > 2 and np.abs(Y_log[-1] - Y_log[-2]) < tol): break
    print("Required iterations:", iter)
    return np.array(X_log), np.array(Y_log)

