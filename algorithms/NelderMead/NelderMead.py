import numpy as np

def nelder_mead(fun, x0=None, low=-5, high=5, max_iter=200, alpha=1.0, gamma=2.0, rho=-0.5, sigma=0.5, tol=1e-6):
    """
    Nelder-Mead algorithm for nonlinear optimization.

    Parameters:
    - fun (callable): objective function to minimize.
    - x0 (array_like): initial guess for the minimum.
    - maxiter (int): maximum number of iterations.
    - alpha (float): reflection parameter.
    - gamma (float): expansion parameter.
    - rho (float): contraction parameter.
    - sigma (float): shrinkage parameter.
    - tol (float): tolerance for convergence.

    Returns:
    - x_min (ndarray): array of values at the minimum.
    - f_min (float): function value at the minimum.
    - n_iter (int): number of iterations performed.
    """

    if(x0 is None): x0 = np.random.uniform(low=low, high=high, size=(2,))

    history = []

    # Define the initial simplex.
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        xi = x0.copy()
        xi[i] += 1.0
        simplex[i + 1] = xi

    # Evaluate the function at each vertex of the simplex.
    f = np.zeros(n + 1)
    for i in range(n + 1):
        f[i] = fun(simplex[i])

    # Main loop.
    for iteration in range(max_iter):

        # Sort the vertices by function value.
        idx = np.argsort(f)
        simplex = simplex[idx]
        f = f[idx]

        # Compute the centroid of the best n vertices.
        x_bar = np.mean(simplex[:-1], axis=0)

        # Reflection.
        x_r = x_bar + alpha * (x_bar - simplex[-1])
        f_r = fun(x_r)
        if f[0] <= f_r < f[-2]:
            simplex[-1] = x_r
            f[-1] = f_r

        # Expansion.
        elif f_r < f[0]:
            x_e = x_bar + gamma * (x_r - x_bar)
            f_e = fun(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f[-1] = f_e
            else:
                simplex[-1] = x_r
                f[-1] = f_r

        # Contraction.
        elif f[-2] < f_r <= f[-1]:
            x_c = x_bar + rho * (simplex[-1] - x_bar)
            f_c = fun(x_c)
            if f_c < f[-1]:
                simplex[-1] = x_c
                f[-1] = f_c
            else:
                # Shrink the simplex.
                simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
                for i in range(1, n + 1):
                    f[i] = fun(simplex[i])
        
        history.append(f[0])

        # Check for convergence.
        if np.max(np.abs(f - f[0])) <= tol:
            break


    # Return the minimum.
    x_min = simplex[0]
    return x_min, history
