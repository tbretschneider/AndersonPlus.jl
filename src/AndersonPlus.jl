module AndersonPlus

import LinearAlgebra

export greet_your_package_name
include("0_Structs.jl")
include("2_HelperFunctions.jl")
include("3_AndersonFunctions.jl")
include("4_AnalysisFunctions.jl")
export p1_f!
include("5_ProblemsSimple.jl")
export AASolve
include("1_Functions.jl")

end
