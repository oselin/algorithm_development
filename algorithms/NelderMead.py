import numpy as np


def nelder_mead(fun, x0 = None, low=-5, high=5, step=0.1, 
                no_improve_thr=10e-6, no_improv_break=10, 
                sampling_budget:int=10, alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    """
    Nelder mead optimization algorithm

    Parameters
    - f (function): function to optimize, must return a scalar score and operate over a numpy array of the same dimensions as x_start
    - x_start (numpy array): initial position
    - step (float): look-around radius in initial step
    - no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
    - max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
    - alpha, gamma, rho, sigma (floats): parameters of the algorithm (see Wikipedia page for reference)
    
    Returns: 
    - A tuple (best parameter array, best score)
    """

    if (x0 is None): x0 = np.random.uniform(low=low, high=high, size=(2,1))

    # Be sure the point is in the right format
    x0 = x0.reshape(-1, 1)
    
    dim, no_improv = len(x0), 0


    # Creation of the simplex
    simplex = np.zeros((dim, dim + 1))
    simplex[:, 0] = x0.T

    for i in range(dim):
        point = np.zeros(dim)
        point[i] = step
        simplex[:, i + 1] = x0.T.copy() + point
    fs = fun(simplex)
    
    # Log variables
    X_log = simplex.copy()
    Y_log = fs.copy()

    max_iter = (sampling_budget - (dim+1))//4
    
    for iteration in range(max_iter):
        # 1 - Sort the values
        inds = np.argsort(fs)
        fs = fs[inds]
        simplex = simplex[:, inds]

        # Log the best value
        X_log = np.hstack((X_log, simplex[:,0].reshape(-1,1)))
        Y_log = np.hstack((Y_log, fs[0]))

        
        # Break if no improvement has been reached
        if (Y_log[-1] < Y_log[-2] - no_improve_thr): no_improv = 0
        else: no_improv += 1

        if no_improv >= no_improv_break: break

        # 2 - centroid
        x0 = np.mean(simplex[:, :-1], axis=1).reshape(-1,1)

        # 3 - reflection
        xr = x0 + alpha*(x0 - simplex[:, -1].reshape(-1,1))
        fxr = fun(xr)

        if (fs[0] <= fxr < fs[-2]):
            fs[-1] = fxr
            simplex[:,-1] = xr.T
            continue

        # 4 - Expansion
        if (fxr < fs[0]):
            xe = x0 + gamma*(xr - x0)
            fxe = fun(xe)

            if (fxe < fxr):
                fs[-1] = fxe
                simplex[:,-1] = xe.T
            else:
                fs[-1] = fxr
                simplex[:,-1] = xr.T
            continue

        # 5 - contraction
        if (fxr < fs[-1]):
            xc = x0 + rho*(xr - x0)
            fxc = fun(xc)
            if (fxc < fxr):
                fs[-1] = fxc
                simplex[:,-1] = xc.T            
                continue
        else:
            xc = x0 + rho*(simplex[:,-1].reshape(-1,1) - x0)
            fxc = fun(xc)
            if (fxc < fs[-1]):
                fs[-1] = fxc
                simplex[:,-1] = xc.T 
                continue

        # 6 - Shrink
        simplex[:,1:] = simplex[:,0] + sigma*(simplex[:,1:] - simplex[:,0])

    X_best, Y_best = X_log[:, -1], Y_log[-1]

    return X_best, Y_best, X_log.T, Y_log.T