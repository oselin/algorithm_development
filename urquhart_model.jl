using LatinHypercubeSampling
import LatinHypercubeSampling.Continuous, LatinHypercubeSampling.Categorical 
using ScatteredInterpolation
using BlackBoxOptim
import BlackBoxOptim: best_candidate, best_fitness
using Parameters
using Distances
using Statistics
using StatsBase
using LinearAlgebra
using Distributed
using StaticArrays
import NearestNeighbors: KDTree, knn
using Printf
using TimerOutputs
using DelimitedFiles

include("SurrogateModelOptim/src/types.jl")
include("SurrogateModelOptim/src/LHC_sampling_plan.jl")
include("SurrogateModelOptim/src/model_infill_utilities.jl")
include("SurrogateModelOptim/src/model_infill.jl")
include("SurrogateModelOptim/src/surrogate_model_utilities.jl")
include("SurrogateModelOptim/src/surrogate_model.jl")
include("SurrogateModelOptim/src/smoptimize_utilities.jl")
include("SurrogateModelOptim/src/smoptimize.jl")
include("benchmark/benchmarkFunctions.jl")




if abspath(PROGRAM_FILE) == @__FILE__
    benchmarkfunction_list = [StybliskiTang, Rastrigin, Rosenbrock, Beale, Sphere, Perm, GoldsteinPrice, Ackley, Bohachevsky] #Hartmann

    search_range=[(-5.0,5.0),(-5.0,5.0)]
    
    for bench in benchmarkfunction_list
        for i in range(1, stop=2)
            if i < 10
                filename = string(nameof(bench),"00$i")
            elseif i < 100
                filename = string(nameof(bench),"0$i")
            else
                filename = string(nameof(bench),"$i")
            end
            smoptimize(bench, search_range, filename; options=Options(paper_bench_opts(),iterations=95,num_start_samples=5))
        end
    end
end