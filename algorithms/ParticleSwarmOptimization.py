import numpy as np
def particle_swarm(fun, X=None, V=None, dimension:int=2, low=-5, high=5, sampling_budget:int=1000, n_particles=50, tol=10e-6, c1=0.1, c2=0.1, w=0.8):

    
    if (X is None): X = np.random.uniform(low=low, high=high, size=(dimension, n_particles))
    if (V is None): V = np.random.randn(dimension, n_particles) * 0.1

    # Initialize data for the first iteration
    # Set the best coord for i-th particle + store f(x) values
    pbest = X
    pbest_obj = fun(X)

    # Find the global best and its associated function value
    gbest = pbest[:,pbest_obj.argmin()].reshape(-1,1)
    gbest_obj = pbest_obj.min() 

    # Initalize the log variables
    X_log, Y_log = gbest.copy(), np.array([gbest_obj])
    
    max_iter = (sampling_budget//n_particles) - 1

    for iteration in range(max_iter):
        # Set the random coefficients
        r1, r2 = np.random.rand(2)

        # Update the velocity
        V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest-X)
        # Update the position
        X = X + V

        # Evaluate the function for each particle
        obj = fun(X)

        # Find the right index at which update the values
        idx = (pbest_obj >= obj)

        pbest[:,idx] = X[:,idx]

        #print(np.array([pbest_obj, obj]))
        pbest_obj = np.vstack((pbest_obj, obj)).min(axis=0)

        gbest = pbest[:, pbest_obj.argmin()].reshape(-1,1)
        gbest_obj = pbest_obj.min()

        # Log the result
        X_log = np.hstack((X_log, gbest.copy()))
        Y_log = np.hstack((Y_log, gbest_obj.copy()))

        # if (np.abs(Y_log[-1] - Y_log[-2]) < tol):
        #     print(f"Tollearnce reacehed, {iteration} iterations is enough.")
        #     break

    X_best, Y_best = X_log[:, -1], Y_log[-1]

    return X_best, Y_best, X_log.T, Y_log.T