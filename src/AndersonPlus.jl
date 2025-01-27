module AndersonPlus

import LinearAlgebra
using LinearAlgebra

import SparseArrays
using SparseArrays

export greet_your_package_name, AASolve

export p1_f!, p2_f!, p3_f!, p4_f!, p5_f!
export P4, P5
include("0_Structs.jl")
include("2_HelperFunctions.jl")
include("3_AndersonFunctions.jl")
include("4_AnalysisFunctions.jl")
include("5_ProblemsSimple.jl")
include("6_Problems.jl")
include("7_ProblemGrossPitvaeskiiEqn.jl")
include("8_CRHeat.jl")
include("1_Functions.jl")

end
