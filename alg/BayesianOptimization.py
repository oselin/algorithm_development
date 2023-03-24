import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from typing import List
from scipy.stats import norm


def surrogate_model(model: GaussianProcessRegressor, X):
    prediction = model.predict(X.T, return_std=True)
    return prediction

def acquisition(model, X, samples):
    yhat, _ = surrogate_model(model, X)
   
    best    = np.max(yhat)
    mu, std = surrogate_model(model, samples)

    probs = norm.cdf((mu - best) / (std + 1e-9))

    return probs



def bayesian_optimization(f, X:np.array=None, dimension:int=2, n_samples:int=15, max_iter:int=100, low=-5, high=5):

    # Modify the function to obtain the minimization
    fun = lambda x: (-1)*f(x)

    # Decide new samples to analyze
    if (X is None): X =  np.random.uniform(low=low, high=high, size=[dimension,n_samples])

    # Evaluate the function
    Y = fun(X)

    # GaussianProcessRegressor(kernel=kernel)
    model = GaussianProcessRegressor()

    # Training of the model
    model.fit(X.T, Y)
    for iter in range(max_iter):

        # Define random sample on which test the model
        samples = np.random.uniform(low=low, high=high, size=[dimension, 100])

        # Rate the samples according to their score
        scores = acquisition(model, X, samples)

        # Find the best for the minimization (the smallest)
        idx = np.argmax(scores)

        # New point
        X_new = samples[:, idx].reshape(-1,1)

        # Find the actual value
        Y_new = fun(X_new)

        #est, _ = surrogate_model(model, X_new)
        #print(f"Estimation error:{Y_new - est}")

        X = np.hstack((X, X_new))
        Y = np.hstack((Y, Y_new))

        model.fit(X.T, Y)

    # fun was inverted. Let's revert it
    Y = -1*Y
    return X.T, Y.T


####
# TEST FUNCTION

# Define the X1 and X2 span
# X1 = np.linspace(-5, 5, 101)
# X2 = np.linspace(-5, 5, 101)
# points = np.array([[x1,x2] for x1 in X1 for x2 in X2])

# # Compute the function
# # Fx = StybliskiTang(points.T)
# X = np.linspace(0,1, 101).reshape(1,-1)
# Fx = myfun(X)

# # Find minimum and its coordinates
# idx  = np.argmin(Fx)
# Xmin = points[idx]

# # Run the optimization algorithm
# X_alg, Y_alg = bayesian_optimization(myfun, dimension=1, n_samples=15, low=0, high=1, max_iter=300)

# print(f"[MIN function] Minimum in x={Xmin[0]},    y={Xmin[1]}    with f={Fx[idx]}")
# #print(f"[OPTIMIZATION] Minimum in x={X_alg[-1,0]}, y={X_alg[-1,1]} with f={Y_alg[-1]}")
# plt.plot(X[0], Fx.reshape(1, -1)[0])
# plt.scatter(X_alg, Y_alg)
# plt.show()
# fig, ax = plt.subplots(1, 2, figsize=(16,8))

# Fx = Fx.reshape(101,101)
# ax[0].contourf(X1,X2,Fx)
# ax[0].axis('scaled')
# ax[0].scatter(X_alg[:,0], X_alg[:,1], c="red", s=1)
# ax[0].scatter(X_alg[0,0], X_alg[0,1], c="blue",s=3)
# ax[0].scatter(X_alg[-1,0],X_alg[-1,1],c="orange",s=3)
# ax[1].plot(np.arange(0,len(Y_alg)), Y_alg)
# ax[1].set_title("Performances over time")



