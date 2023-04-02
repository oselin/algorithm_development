import numpy as np

def StybliskiTang(x: np.array) -> float:

    f, dimension = 0, len(x)

    for i in range(dimension):
        xi = x[i]
        f += xi**4 - 16*xi**2 + 5*xi
    return f/2

def bfgs(f, x0, maxiter=1000, tol=1e-6, h=1e-6):
    n = len(x0)
    x = x0
    B = np.eye(n)  # Initialize the Hessian approximation
    for k in range(maxiter):
        g = approx_gradient(f, x, h)
        if np.linalg.norm(g) < tol:
            break
        d = -np.dot(B, g)
        alpha = 1.0
        while f(x + alpha*d) > f(x) + alpha*0.1*np.dot(g, d):
            alpha *= 0.5
        s = alpha*d
        y = approx_gradient(f, x + s, h) - g
        if np.dot(y, s) > 0:
            B = B - np.outer(np.dot(B, s), np.dot(s, B)) / np.dot(s, np.dot(B, s)) + np.outer(y, y) / np.dot(y, s)
        x = x + s
    return x

def approx_gradient(f, x, h):
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += h
        x2[i] -= h
        g[i] = (f(x1) - f(x2)) / (2*h)
    return g

x0 = np.array([1.0, 2.0])

a = bfgs(StybliskiTang, x0)
print("MIN:", a)