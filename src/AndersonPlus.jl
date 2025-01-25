module AndersonPlus

import LinearAlgebra
using LinearAlgebra

export greet_your_package_name, AASolve

export p1_f!, p2_f!, p3_f!
include("0_Structs.jl")
include("2_HelperFunctions.jl")
include("3_AndersonFunctions.jl")
include("4_AnalysisFunctions.jl")
include("5_ProblemsSimple.jl")
include("6_Problems.jl")
include("1_Functions.jl")

end
