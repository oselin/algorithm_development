import numpy as np

def nelder_mead(fun, x0 = None, step=0.1, low=-5, high=5, no_improve_thr=10e-6, no_improv_break=10, max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5):
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

    if(x0 is None): x0 = np.random.uniform(low=low, high=high, size=(1,2))

    #dim = len(x_start)
    X = x0.copy()
    Y = np.array(fun(x0)).reshape(-1,1)

    #no_improv = 0

    # Apply the displacement in each direction, one at a time
    for i in range(X.shape[1]):
        X = np.vstack((X, X[-1,i] + step))
        Y = np.vstack((Y, fun(X[-1])))

    print(X)
    print(Y)

    # for iteration in range(max_iter):
        
    #     # order
    #     res.sort(key=lambda x: x[1])
    #     best = res[0][1]

    #     # break after max_iter
    #     if max_iter and iters >= max_iter:
    #         return res[0]
    #     iters += 1

    #     # break after no_improv_break iterations with no improvement
    #     #print '...best so far:', best

    #     if best < prev_best - no_improve_thr:
    #         no_improv = 0
    #         prev_best = best
    #     else:
    #         no_improv += 1

    #     if no_improv >= no_improv_break:
    #         return res[0]

    #     # centroid
    #     x0 = [0.] * dim
    #     for tup in res[:-1]:
    #         for i, c in enumerate(tup[0]):
    #             x0[i] += c / (len(res)-1)

    #     # reflection
    #     xr = x0 + alpha*(x0 - res[-1][0])
    #     rscore = f(xr)
    #     if res[0][1] <= rscore < res[-2][1]:
    #         del res[-1]
    #         res.append([xr, rscore])
    #         continue

    #     # expansion
    #     if rscore < res[0][1]:
    #         xe = x0 + gamma*(x0 - res[-1][0])
    #         escore = f(xe)
    #         if escore < rscore:
    #             del res[-1]
    #             res.append([xe, escore])
    #             continue
    #         else:
    #             del res[-1]
    #             res.append([xr, rscore])
    #             continue

    #     # contraction
    #     xc = x0 + rho*(x0 - res[-1][0])
    #     cscore = f(xc)
    #     if cscore < res[-1][1]:
    #         del res[-1]
    #         res.append([xc, cscore])
    #         continue

    #     # reduction
    #     x1 = res[0][0]
    #     nres = []
    #     for tup in res:
    #         redx = x1 + sigma*(tup[0] - x1)
    #         score = f(redx)
    #         nres.append([redx, score])
    #     res = nres


