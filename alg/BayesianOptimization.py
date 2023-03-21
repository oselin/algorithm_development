from bayes_opt import BayesianOptimization

def bayesian_optimization(fun, bounds, iterations, n_samples):

    minf = lambda X,Y: (-1)*fun([X,Y])
    optimizer = BayesianOptimization(
        f = minf,
        pbounds = bounds,
        random_state = None,
        allow_duplicate_points = True,
        verbose = 0
    )

    optimizer.maximize(
        init_points =n_samples,
        n_iter = iterations
    )

    return optimizer.max, optimizer.res

def history_wrapper(params):
    history = []

    for elem in params:
        history.append(-1 * elem['target'])

    return history

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from typing import List
from scipy.stats import norm

def StybliskiTang(x: List[float]) -> float:
    f, dimension = 0, len(x)

    for i in range(dimension):
        xi = x[i]
        f += xi**4 - 16*xi**2 + 5*xi
    return f


def surrogate_model(model: GaussianProcessRegressor, X):
    prediction = model.predict(X, return_std=True)
    return prediction

def acquisition(X, Xsamples, model):
    yhat, _ = surrogate_model(model, X)
   
    best    = np.max(yhat)
    print("best; ", best)
    mu, std = surrogate_model(model, Xsamples)
    print(yhat)
    print(mu-np.max(yhat))
    probs = norm.cdf((mu - best) / (std + 1e-9))
    print(probs)
    return probs

def opt_acquisition(X, model):
    # Decide new samples to analyze
    Xsamples =  np.random.uniform(low=-5, high=5, size=[15,2])

    scores = acquisition(X, Xsamples, model)

    ix = np.argmax(scores)
    return Xsamples[ix,0]

# Define the number of samples
n_samples = 15

# GaussianProcessRegressor(kernel=kernel)
model = GaussianProcessRegressor()
# Pick random sample values
X = np.random.uniform(low=-5, high=5, size=[n_samples,2])
print(X.shape)
# Compute the expensive objective function (in the wind tunnel)
Y = StybliskiTang(X.T)

# Apply the Bayesian optimization, namely refit the model
model.fit(X,Y)

# Estimate the surrogate mdel values
y, _ = surrogate_model(model, X)


# Perform the optimization process
for i in range(100):
    # select the next point to sample (random chosen point with the max likelihood)
    x = opt_acquisition(X, model)
    print(x.shape)
    # sample the point (EXPENSIVE FUNCTION)
    actual = StybliskiTang(x)
    print(actual.shape)
    # SURROGATE VALUES
    est, _ = surrogate_model(model, x)

    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))

    # add the data to the dataset FOR THE MODEL
    X = np.vstack((X, [[x]]))
    y = np.vstack((y, [[actual]]))

    # update the model
    model.fit(X, Y)
 

# best result
ix = np.argmax(Y)
print('Best Result: x=%.3f, y=%.3f. The estimated value is %.3f' % (X[ix], Y[ix], y[ix]))

fig = plt.figure(figsize=[8,8])
ax = fig.add_subplot(1,1,1)
#ax.scatter(X, Y)
#ax.scatter(X, y, c="red")

"""