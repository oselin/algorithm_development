import numpy as np

def particle_swarm(fun, X=None, V=None, low=-5, high=5, max_iter=1000, n_particles=50, tol=10e-6, c1=0.1, c2=0.1, w=0.8):

    

    if (X == None or not X.any()): X = np.random.uniform(low=low, high=high, size=(2,n_particles))
    if (V == None or not V.any()): V = np.random.randn(2, n_particles) * 0.1
    
    history = []
    # Initialize data for the first iteration
    # Set the best coord for i-th particle + store f(x) values
    pbest = X
    pbest_obj = fun(X)

    # Find the global best and its associated function value
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()

    old_gbest_obj = 0
    
    for iteration in range(max_iter):
        old_gbest_obj = gbest_obj
        # Set the random coefficients
        r1, r2 = np.random.rand(2)

        # Update the velocity
        V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1,1)-X)

        # Update the position
        X = X + V

        # Evaluate the function for each particle
        obj = fun(X)

        pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
        pbest_obj = np.array([pbest_obj, obj]).min(axis=0)

        gbest = pbest[:, pbest_obj.argmin()]
        gbest_obj = pbest_obj.min()
        
        history.append(gbest_obj)
        #if (np.abs(gbest_obj - old_gbest_obj) < tol):
        #    print(f"Tollearnce reacehed, {iteration} iterations is enough.")
        #    break

    return gbest, history

    






