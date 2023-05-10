import numpy as np

def nelder_mead(fun, dimension:int=2, boundaries=[[-5,-5],[5,5]], step=0.1, 
                no_improve_thr=10e-6, no_improv_break=100, 
                sampling_budget:int=100, alpha=1., gamma=2., rho=-0.5, sigma=0.5):
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

    boundaries = np.array(boundaries)

    # Choose the initial point
    x0 = np.random.uniform(low=boundaries[0], high=boundaries[1], size=(1,dimension)).T
    
    no_improv = 0


    # Creation of the simplex
    simplex = np.zeros((dimension, dimension + 1))
    simplex[:, 0] = x0.T

    for i in range(dimension):
        point = np.zeros(dimension)
        point[i] = step
        simplex[:, i + 1] = x0.T.copy() + point
    fs = fun(simplex)
    
    max_iter = (sampling_budget - (dimension+1))//4

    for iteration in range(max_iter):
        # 1 - Sort the values
        inds = np.argsort(fs)
        fs = fs[inds]
        simplex = simplex[:, inds]

        # Log the best value
        if (iteration == 0):
            X_log = simplex[:,0].copy().reshape(-1,1)
            Y_log = np.array([fs[0]])
        else:
            X_log = np.hstack((X_log, simplex[:,0].reshape(-1,1)))
            Y_log = np.hstack((Y_log, fs[0]))

        
        # Break if no improvement has been reached
        if (iteration > 3 and Y_log[-1] < Y_log[-2] - no_improve_thr): no_improv = 0
        else: no_improv += 1

        if no_improv >= no_improv_break: break

        # 2 - centroid
        x0 = np.mean(simplex[:, :-1], axis=1).reshape(-1,1)

        # 3 - reflection
        xr = x0 + alpha*(x0 - simplex[:, -1].reshape(-1,1))
        fxr = fun(xr)

        if (fs[0] <= fxr < fs[-2]):
            fs[-1] = fxr.copy()
            simplex[:,-1] = xr.copy().T
            continue

        # 4 - Expansion
        if (fxr < fs[0]):
            xe = x0 + gamma*(xr - x0)
            fxe = fun(xe)

            if (fxe < fxr):
                fs[-1] = fxe.copy()
                simplex[:,-1] = xe.copy().T
            else:
                fs[-1] = fxr.copy()
                simplex[:,-1] = xr.copy().T
            continue

        # 5 - contraction
        if (fxr < fs[-1]):
            xc = x0 + rho*(xr - x0)
            fxc = fun(xc)
            if (fxc < fxr):
                fs[-1] = fxc.copy()
                simplex[:,-1] = xc.copy().T          
                continue  
        else:
            xc = x0 + rho*(simplex[:,-1].reshape(-1,1) - x0)
            fxc = fun(xc)
            if (fxc < fs[-1]):
                fs[-1] = fxc.copy()
                simplex[:,-1] = xc.T 
                continue


        # 6 - Shrink
        simplex[:,1:] = simplex[:,0].reshape(-1,1) + sigma*(simplex[:,1:] - simplex[:,0].reshape(-1,1))

    X_best, Y_best = X_log[:, -1], Y_log[-1]

    return X_best, Y_best, X_log.T, Y_log.T