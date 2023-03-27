from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
from typing import List




def box_behnken(X:np.array, bound) -> np.array:
    """
    Box Behnken design function
    """
    points = X.copy()
    for i in range(len(X)):
        x_lower, x_higher = X.copy(), X.copy()
        x_lower[i]  -= bound
        x_higher[i] += bound
        points = np.hstack((points, x_lower, x_higher))

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

    permutations = np.array(permutations).T

    return np.hstack((X,X + permutations))



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
        grad[i] = (model.predict(Xh.T) - y) / delta_h

    return grad.reshape(-1,1)


def expected_improvement(model:Pipeline, X:List[float]):
    """
    Expected improvement function from statistic
    """

    # TODO
    #y_hat = np.max(model.predict(X))
    # mu, std = model.predict(X, return_std=True)
    # print(mu)
    return



def next_step(model:Pipeline, X:List[float],Y:List[float], lr:float=0.01, method="gradient"):
    """
    Next step function: find the best point to sample at next iteration
    """
    if (method == "gradient"):

        # Find the local minimum in the considered set
        min_idx = np.argmin(Y)
  
        X_best = X[:, min_idx].reshape(-1,1)
        Y_best = Y[min_idx]
        # Apply the gradient descend to find the new point to analyze
        X_next = X_best - lr*gradient(model,X_best, Y_best)

    return X_next, X_best, Y_best
    #else:
    #    expected_improvement(model, X)


def response_surface(fun, x0:List[float], sampling_budget:int=100, tol = 1e-8, sampling_method="box_behnken", sampling_bound=0.5,
                     iteration_method="gradient", learning_rate=0.01):

    model = Pipeline([('poly',   PolynomialFeatures(degree=2)),
                      ('linear', LinearRegression(fit_intercept=False))]) 

    # Be sure the point is in the right format
    x0 = x0.reshape(-1, 1)
    
    # Log arrays
    X_log, Y_log = None, None

    dimensions = len(x0)

    if   (sampling_method == "box_behnken"):       coeff = 2*dimensions + 1  # box_behnken output: 2n + 1 samples
    elif (sampling_method == "central_composite"): coeff = 2**dimensions + 1 # central_composite output: 2^n + 1 samples
    max_iter = sampling_budget//coeff

    # If type(bounds) == list, iterative mode is requested
    if (type(sampling_bound)== list): sampling_bound = np.linspace(sampling_bound[0], sampling_bound[1], max_iter)
    else: sampling_bound = [sampling_bound]*max_iter

    # If type(bounds) == list, iterative mode is requested
    if (type(learning_rate)== list): learning_rate = np.linspace(learning_rate[0], learning_rate[1], max_iter)
    else: learning_rate = [learning_rate]*max_iter

    X_new = x0.copy()
    for iter in range(max_iter):

        # Find new samples to analyze
        if   (sampling_method == "box_behnken"):        samples = box_behnken(X_new, sampling_bound[iter])
        elif (sampling_method == "central_composite"):  samples = central_composite(X_new, sampling_bound[iter])

        # Evaluate the function for those samples
        Y = fun(samples)

        # Fit the model on those few samples
        model.fit(samples.T, Y)
        # Find the new starting point for the next iteration
        X_new, X_best, Y_best = next_step(model, samples, Y, lr=learning_rate, method=iteration_method)

        # Update the set of analyzed points with the ones of this iteration
        if (X_log is None): X_log = X_best.copy()
        else: X_log = np.hstack((X_log, X_best))

        if (Y_log is None): Y_log = Y_best.copy()
        else: Y_log = np.hstack((Y_log, Y_best))

        # Convergence condition
        if (iter > 2 and np.abs(Y_log[-1] - Y_log[-2]) < tol): break

    print("Required iterations:", iter)

    X_best, Y_best = X_log[:, -1], Y_log[-1]
    
    return X_best, Y_best, X_log.T, Y_log.T

