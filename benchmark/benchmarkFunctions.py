
import numpy as np


"""

    Benchmark functions to test the best optimization algorithms
    
"""

def StybliskiTang(x: np.array) -> float:

    f, dimension = 0, len(x)

    for i in range(dimension):
        xi = x[i]
        f += xi**4 - 16*xi**2 + 5*xi
    return f/2
    
def Rastrigin(x: np.array) -> float:
    f, dimension = 0, len(x)
    f = 10*dimension

    for i in range(dimension):
        xi = x[i]
        f += xi**2 - 10*np.cos(2*np.pi*xi)

    return f


def Rosenbrock(x: np.array) -> float:
    f, dimension = 0, len(x)

    for i in range(dimension-1):
        xi, xii = x[i], x[i+1]
        f += 100*(xii - xi**2)**2 + (xi - 1)**2
    
    return f


def Beale(x: np.array) -> float:
    x1, x2 = x[0], x[1]

    term1 = (1.5 - x1 + x1*x2)**2
    term2 = (2.25 - x1 + x1*x2**2)**2
    term3 = (2.625 - x1 + x1*x2**3)**2

    return (term1 + term2 + term3)


def Sphere(x: np.array) -> float:
    f, dimension = 0, len(x)

    for i in range(dimension):
        xi = x[i]
        f += xi**2
    
    return f


def Perm(x: np.array, beta:float=0.5) -> float:
    f, dimension = 0, len(x)

    for i in range(1, dimension+1):
        inner = 0
        for j in range(1, dimension+1):
            xj = x[j-1]
            inner += (j**i+ beta) * ((xj/j)**i-1)
        f += inner**2

    return f


def GoldsteinPrice(x: np.array) -> float:
    x1, x2 = x[0], x[1]

    fact1a = (x1 + x2 + 1)**2
    fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
    fact1 = 1 + fact1a*fact1b

    fact2a = (2*x1 - 3*x2)**2
    fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    fact2 = 30 + fact2a*fact2b

    return (fact1*fact2)


def Hartmann(x: np.array) -> float:

    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array([
            [  10,   3,   17, 3.5, 1.7,  8],
            [0.05,  10,   17, 0.1,   8, 14],
            [   3, 3.5,  1.7,  10,  17,  8],
            [  17,   8, 0.05,  10, 0.1, 14]])
    
    P = 10**(-4) * np.array([
            [1312, 1696, 5569,  124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091,  381]])

    outer = 0
    for i in range(4):
        inner = 0
        for j in range(6):
            xj  = x[j]
            Aij = A[i,j]
            Pij = P[i,j]
            inner = inner + Aij*(xj-Pij)**2

        new = alpha[i] * np.exp(-inner)
        outer = outer + new

    return  (-(2.58 + outer) / 1.94)


def Ackley(x: np.array, a:float=20, b:float=0.2, c:float=2*np.pi) -> float:
    dimension = len(x)

    sum1 = 0
    sum2 = 0
    for i in range(dimension):
        xi = x[i]
        sum1 = sum1 + xi**2
        sum2 = sum2 + np.cos(c*xi)

    term1 = -a * np.exp(-b*np.sqrt(sum1/dimension))
    term2 = -np.exp(sum2/dimension)

    return (term1 + term2 + a + np.exp(1))


def Bohachevsky(x: np.array) -> float:
    x1, x2 = x[0], x[1]

    return (x1**2 +2*(x2**2)-0.3*np.cos(3*np.pi*x1)-0.4*np.cos(4*np.pi*x2)+0.7)


def getFunctionList():
    return [StybliskiTang,
            Rastrigin,
            Rosenbrock,
            Beale,
            Sphere,
            Perm,
            GoldsteinPrice,
            Hartmann,
            Ackley,
            Bohachevsky
    ]
