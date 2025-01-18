# Structure for the output of the algorithm

export AAMethod
struct AAMethod
    methodname::Symbol
	methodparams::NamedTuple
end

export AAAlgorithm
struct AAAlgorithm
    method::AAMethod
    algorithmparams::NamedTuple
end

export AAConvParams
struct AAConvParams
	rtol::Float64
    atol::Float64
end

export AAProblem
struct AAProblem
    GFix!::Function                 # Fixed-point map (GFix!(G, x))
    x0::Vector{Float64}			# Initial iterate
    convparams::AAConvParams
end

export AAAnalysis
struct AAAnalysis
    liveanalysis::Vector{Symbol}
    midanalysis::Vector{Symbol} # Metrics to track (e.g., :residual, :iterations)
    postanalysis::Vector{Symbol} # Metrics to track (e.g., :residual, :iterations)
    updatefreq::Int
    summary::Bool
end

export AAInput
struct AAInput
    problem::AAProblem               # Fixed-point map (GFix!(G, x))
    algorithm::AAAlgorithm			# Initial iterate
    analyses::AAAnalysis
end

export AAAnalysisOutput
struct AAAnalysisOutput
    output::NamedTuple # Metrics to track (e.g., :residual, :iterations)
end

export AAOutput
struct AAOutput
    solution::Vector{Float64}        # Converged result
    input::AAInput   # G(solution)
    analysis::AAAnalysisOutput
end

abstract type HistoricalStuff end

mutable struct VanillaHistoricalStuff <: HistoricalStuff
    residual::Vector{Vector{Float64}}
    solhist::Vector{Vector{Float64}}
    iterations::Int
end
