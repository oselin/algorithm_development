import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import  GaussianProcessRegressor

from scipy.optimize import line_search

def gradient(model:Pipeline, X:np.array, Y:float, delta_h:float=1e-3):
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
        grad[i] = (model.predict(Xh.T) - Y) / delta_h

    return grad.reshape(-1,1)

def subdomain_func(x, boundaries):
    """
    - The boundaries are 2-by-N-dimensional arrays, e.g.
      boundaries = [
                        [lower boundaries],
                        [upper boundaries]
                    ]
    - x is a point in Algebraic notation (column vector)
    """

    for i,coord in enumerate(x):
        if not (boundaries[0,i] <= coord <= boundaries[1,i]): return False
    return True

def backtracking_line_search(target_func, g, x, p, alpha=1, rho=0.5, c=1e-4):
    """
    Backtracking line search algorithm.
    
    Parameters
    ----------
    target_func : callable
        The target function to minimize.
    grad_func : callable
        The gradient function.
    x : array-like
        The current point.
    p : array-like
        The search direction.
    alpha : float, optional
        The initial step size.
    rho : float, optional
        The reduction factor for the step size.
    c : float, optional
        The sufficient decrease parameter.
    
    Returns
    -------
    alpha : float
        The step size that satisfies the Armijo-Goldstein condition.
    """
    
    f = target_func(x)
    dg = g.T @ p

    while target_func(x + alpha * p) > f + c * alpha * dg:
        alpha *= rho
    if (alpha < 0.01): return 0.01
    else: return alpha


def bfgs(fun, dimension:int=2, boundaries=None, sampling_budget:int=100, tol=10e-6, verbose=False):

    # Initialization of the surrogate model
    model =  GaussianProcessRegressor() #Pipeline([('poly',   PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))]) 


    # Define the two initial points to fit the model (Algebra notation)
    boundaries = np.array(boundaries)
    x0 = np.random.uniform(low=boundaries[0], high=boundaries[1], size=[2,dimension]).T

    B_inv = np.eye(dimension)

    x_prev    = x0[:,0].reshape(-1,1)
    x_k       = x0[:,1].reshape(-1,1)

    X_log = np.hstack((x_prev, x_k))
    Y_log = np.array(fun(X_log))

    max_iter = (sampling_budget  - 2)

    for iter in range(max_iter):
        # Fit the model on those few samples
        model.fit(X_log.T,Y_log)

        # Compute the two factors: differnce in the gradient, difference in the position
        x_delta = x_k - x_prev
        y       = gradient(model, x_k, Y_log[-1]) - gradient(model, x_prev, Y_log[-2])

        # Update the Inverse of the Hessian approximation
        B_inv = (np.eye(dimension) - (x_delta @ y.T)/ (y.T @ x_delta + 1e-9)) @ B_inv @ (np.eye(dimension) - (y @ x_delta.T)/(y.T @ x_delta + 1e-9)) + (x_delta @ x_delta.T)/(y.T @ x_delta + 1e-9)
       
        # Compute the update
        p = - B_inv @ gradient(model, x_k, Y_log[-1])

        # project the search direction onto the tangent space of the subdomain
        mask = np.logical_not(subdomain_func(x_k, boundaries))
        p[mask] = 0

        alpha = backtracking_line_search(fun, gradient(model,x_k, Y_log[-1]), x_k, p)

        # Update the point
        x_new = x_k + alpha*p

        # Log the new value
        X_log = np.hstack((X_log, x_new))
        Y_log = np.hstack((Y_log, fun(x_new)))

        if (np.linalg.norm(gradient(model, x_new, Y_log[-1])) < tol):
            if (verbose): print(f"Stopped at iteration {iter}")
            break
        
        # Update for next iteration
        x_prev = x_k.copy().reshape(-1,1)
        x_k = x_new.copy().reshape(-1,1)
        

    X_best, Y_best = X_log[:,-1], Y_log[-1]

    return X_best, Y_best, X_log.T, Y_log.T