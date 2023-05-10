from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np


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


def gradient(model:Pipeline, X:np.array, y:float, delta_h:float=1e-5):
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


def expected_improvement(model:Pipeline, X:np.array):
    """
    Expected improvement function from statistic
    """

    # TODO
    #y_hat = np.max(model.predict(X))
    # mu, std = model.predict(X, return_std=True)
    # print(mu)
    return


def next_step(model:Pipeline, X:np.array,Y:np.array, lr:float=0.01, method="gradient"):
    """
    Next step function: find the best point to sample at next iteration
    """
    if (method == "gradient"):

        # Find the local minimum in the considered set
        min_idx = np.argmin(Y)
  
        X_best = X[:, min_idx].reshape(-1,1)
        Y_best = Y[min_idx]
        # Apply the gradient descend to find the new point to be investigated
        X_next = X_best - lr*gradient(model,X_best, Y_best)

    #elif (method=="expected_improvement"):
    #    expected_improvement(model, X)

    return X_next, X_best, Y_best


def response_surface(fun, boundaries=[[-5,-5],[5,5]], dimension:int=2,sampling_budget:int=100, tol = 1e-8, sampling_method="box_behnken", sampling_bound=0.5,
                     iteration_method="gradient", learning_rate=0.01, verbose=False):

    model = Pipeline([('poly',   PolynomialFeatures(degree=2)),
                      ('linear', LinearRegression(fit_intercept=False))]) 

    # Define the starting point
    boundaries = np.array(boundaries)
    x0 = np.random.uniform(low=boundaries[0], high=boundaries[1], size=[1,dimension]).T
    
    # Calculate the maximum number of iterations
    if   (sampling_method == "box_behnken"):       coeff = 2*dimension + 1  # box_behnken output: 2n + 1 samples
    elif (sampling_method == "central_composite"): coeff = 2**dimension + 1 # central_composite output: 2^n + 1 samples
    max_iter = sampling_budget//coeff

    # If type(bounds) == list, iterative mode is requested
    if (type(sampling_bound)== list): sampling_bound = np.linspace(sampling_bound[0], sampling_bound[1], max_iter)
    else: sampling_bound = [sampling_bound]*max_iter

    # If type(learning rate) == list, iterative mode is requested
    if (type(learning_rate)== list): learning_rate = np.linspace(learning_rate[0], learning_rate[1], max_iter)
    else: learning_rate = [learning_rate]*max_iter

    X_new = x0.copy()
    for iter in range(max_iter):

        # Find new samples to analyze
        if   (sampling_method == "box_behnken"):        samples = box_behnken(X_new, sampling_bound[iter])
        elif (sampling_method == "central_composite"):  samples = central_composite(X_new, sampling_bound[iter])

        # Evaluate the function for those samples
        Y = fun(samples)

        if (iter == 0 or np.min(Y) < Y_best):
            # Fit the model on those few samples
            model.fit(samples.T, Y)

            # Find the new starting point for the next iteration
            X_new, X_best, Y_best = next_step(model, samples, Y, lr=learning_rate, method=iteration_method)
        else:
            # Keep the same points and move to the next iteration
            pass

        # Update the set of analyzed points with the ones of this iteration
        if (iter == 0):
            X_log = X_best.copy().reshape(-1,1)
            Y_log = np.array([Y_best])
        else:
            X_log = np.hstack((X_log, X_best))
            Y_log = np.hstack((Y_log, Y_best))

        # Convergence condition
        if (iter > 2 and np.abs(Y_log[-1] - Y_log[-2]) < tol): break

    if (verbose): print("Required iterations:", iter)
    
    return X_best, Y_best, X_log.T, Y_log.T

