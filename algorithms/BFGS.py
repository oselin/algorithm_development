from benchmark.benchmarkFunctions import StybliskiTang
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


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


def bfgs(fun, x0=np.array, x1=np.array, sampling_budget:int=100, tol=10e-6, verbose=False):

    # Initialization of the surrogate model
    model = Pipeline([('poly',   PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))]) 


    # Reshape of the initial points as column vector (Algebra notation)
    x0 = x0.reshape(-1, 1)
    x1 = x1.reshape(-1, 1)

    dim = len(x0)
    B_inv = np.eye(dim)

    x_k    = x0.copy()
    x_next = x1.copy()

    X_log = np.hstack((x_k, x_next))
    Y_log = np.array(fun(X_log))

    max_iter = (sampling_budget  - 2)

    for iter in range(max_iter):
        # Fit the model on those few samples

        model.fit(X_log.T,Y_log.T)

        # Compute the two factors: differnce in the gradient, difference in the position
        x_delta = x_next - x_k
        y       = gradient(model,x_next, Y_log[-1]) - gradient(model, x_k, Y_log[-2])

        # Update the Inverse of the Hessian approximation
        B_inv = (np.eye(dim) - (x_delta @ y.T)/ (y.T @ x_delta)) @ B_inv @ (np.eye(dim) - (y @ x_delta.T)/(y.T @ x_delta)) + (x_delta @ x_delta.T)/(y.T @ x_delta)
       
        # Update for next iteration
        x_next, x_k = x_next - B_inv @ gradient(model,x_next, Y_log[-1]), x_next.copy()

        X_log = np.hstack((X_log, x_next))
        Y_log = np.hstack((Y_log, fun(x_next)))

        if (np.linalg.norm(gradient(model, x_next, Y_log[-1])) < tol):
            if (verbose): print(f"Stopped at iteration {iter}")
            break
        

    X_best, Y_best = X_log[:,-1], Y_log[-1]

    return X_best, Y_best, X_log.T, Y_log.T