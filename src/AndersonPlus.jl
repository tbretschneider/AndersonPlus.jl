module AndersonPlus

import LinearAlgebra
using LinearAlgebra

import SparseArrays
using SparseArrays


using AbstractFFTs: AbstractFFTs, plan_fft, plan_fft!
using FFTW: FFTW

export greet_your_package_name, AASolve

export p1_f!, p2_f!, p3_f!, p4_f!, p5_f!
export P3, P4, P5, P6, P7
include("0_Structs.jl")
include("2_HelperFunctions.jl")
include("3_AnalysisFunctions.jl")
include("problems/0_ProblemsSimple.jl")
include("problems/1_ProblemsHard.jl")
include("problems/2_ProblemGrossPitvaeskiiEqn.jl")
include("problems/3_ProblemCRHeat.jl")
include("methods/0_Setup.jl")
include("methods/1_AndersonFunctions.jl")
include("methods/2_SupportFunctions.jl")
include("1_Functions.jl")

end
