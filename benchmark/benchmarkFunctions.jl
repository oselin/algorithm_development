

function StybliskiTang(x)
    f, dimension = 0, length(x)
    for i in 1:dimension
        xi = x[i]
        f += xi^4 - 16*xi^2 + 5*xi
    end
    output =  f/2

    # Manually scale the function
    min = -78.3319
    max = 250.0
    return (output - min) * (0.15)/(max - min) + 0.20
end
    

function Rastrigin(x)
    f, dimension = 0, length(x)
    f = 10*dimension

    for i in 1:dimension
        xi = x[i]
        f += xi^2 - 10*cos(2*pi*xi)
    end

    output =  f

    # Manually scale the function
    min = 0.0
    max = 80.5
    return (output - min) * (0.15)/(max - min) + 0.20
end


function Rosenbrock(x)
    f, dimension = 0, length(x)

    for i in 1:dimension-1
        xi, xii = x[i], x[i+1]
        f += 100*(xii - xi^2)^2 + (xi - 1)^2
    end

    output =  f

    # Manually scale the function
    min = 0
    max = 90036.0
    return (output - min) * (0.15)/(max - min) + 0.20
end


function Beale(x)
    x1, x2 = x[1], x[2]

    term1 = (1.5 - x1 + x1*x2)^2
    term2 = (2.25 - x1 + x1*x2^2)^2
    term3 = (2.625 - x1 + x1*x2^3)^2

    output =  term1 + term2 + term3

    # Manually scale the function
    min = 0.0
    max = 415071.703125
    return (output - min) * (0.15)/(max - min) + 0.20
end


function Sphere(x)
    f, dimension = 0, length(x)

    for i in 1:dimension
        xi = x[i]
        f += xi^2
    end

    output =  f

    # Manually scale the function
    min = 0.0
    max = 50.0
    return (output - min) * (0.15)/(max - min) + 0.20
end


function Perm(x, beta::Float64=0.5)
    f, dimension = 0, length(x)

    for i in 1:dimension
        inner = 0
        for j in 1:dimension
            xj = x[j]
            inner += (j^i+ beta) * ((xj/j)^i-1)
        end
        f += inner^2
    end

    output =  f

    # Manually scale the function
    min = 0.0
    max = 3870.203125
    return (output - min) * (0.15)/(max - min) + 0.20
end


function GoldsteinPrice(x)
    x1, x2 = x[1], x[2]

    fact1a = (x1 + x2 + 1)^2
    fact1b = 19 - 14*x1 + 3*x1^2 - 14*x2 + 6*x1*x2 + 3*x2^2
    fact1 = 1 + fact1a*fact1b

    fact2a = (2*x1 - 3*x2)^2
    fact2b = 18 - 32*x1 + 12*x1^2 + 48*x2 - 36*x1*x2 + 27*x2^2
    fact2 = 30 + fact2a*fact2b

    output =  fact1*fact2

    # Manually scale the function
    min = 3.0
    max = 293509731.0
    return (output - min) * (0.15)/(max - min) + 0.20
end



function Hartmann(x::Array{T,1}) where T<:Real
    alpha = [1.0, 1.2, 3.0, 3.2]

    A = [
        10    3    17   3.5  1.7  8;
        0.05 10   17   0.1   8   14;
        3   3.5   1.7  10   17    8;
        17    8  0.05  10  0.1  14
    ]
    
    P = 10.0^(-4) .* [
        1312 1696 5569  124 8283 5886;
        2329 4135 8307 3736 1004 9991;
        2348 1451 3522 2883 3047 6650;
        4047 8828 8732 5743 1091  381
    ]

    outer = 0
    for i in 1:4
        inner = 0
        for j in 1:6
            xj  = x[j]
            Aij = A[i,j]
            Pij = P[i,j]
            inner = inner + Aij*(xj-Pij)^2
        end

        new = alpha[i] * exp(-inner)
        outer = outer + new
    end

    output =   (-(2.58 + outer) / 1.94)
end



function Ackley(x, a::Float64=20.0, b::Float64=0.2, c::Float64=2π)
    dimension = length(x)

    sum1 = sum(xi^2 for xi in x)
    sum2 = sum(cos(c*xi) for xi in x)

    term1 = -a * exp(-b*sqrt(sum1/dimension))
    term2 = -exp(sum2/dimension)

    output =  term1 + term2 + a + exp(1)

    # Manually scale the function
    min = 4.440892098500626e-16
    max = 14.302605427560742
    return (output - min) * (0.15)/(max - min) + 0.20
end
    


function Bohachevsky(x)
    x1, x2 = x[1], x[2]

    output =  x1^2 + 2*x2^2 - 0.3*cos(3π*x1) - 0.4*cos(4π*x2) + 0.7

    # Manually scale the function
    min = 0.0
    max = 75.6
    return (output - min) * (0.15)/(max - min) + 0.20
end
    


