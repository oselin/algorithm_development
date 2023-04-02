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


def bfgs(fun, dimension:int=2, boundaries=None, sampling_budget:int=100, tol=10e-6, verbose=False):

    # Initialization of the surrogate model
    model = Pipeline([('poly',   PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))]) 


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
        B_inv = (np.eye(dimension) - (x_delta @ y.T)/ (y.T @ x_delta)) @ B_inv @ (np.eye(dimension) - (y @ x_delta.T)/(y.T @ x_delta)) + (x_delta @ x_delta.T)/(y.T @ x_delta)
       
        # Update the point
        x_new = x_k - B_inv @ gradient(model, x_k, Y_log[-1])

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