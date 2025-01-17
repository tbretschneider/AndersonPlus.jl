# Structure for the output of the algorithm

struct AAMethod
    methodname::Symbol
	methodparams::NamedTuple
end

struct AAAlgorithm
    method::AAMethod
    algorithmparams::NamedTuple
end

struct AAConvParams
	rtol::Float64
    atol::Float64
end

struct AAProblem
    GFix!::Function                 # Fixed-point map (GFix!(G, x))
    x0::Vector{Float64}			# Initial iterate
    convparams::AAConvParams
end

struct AAAnalysis
    liveanalysis::Vector{Symbol}
    midanalysis::Vector{Symbol} # Metrics to track (e.g., :residual, :iterations)
    postanalysis::Vector{Symbol} # Metrics to track (e.g., :residual, :iterations)
    updatefreq::Int
end

struct AAInput
    problem::AAProblem               # Fixed-point map (GFix!(G, x))
    algorithm::AAAlgorithm			# Initial iterate
    analyses::AAAnalysis
end

struct AAAnalysisOutput
    postanalysis::NamedTuple # Metrics to track (e.g., :residual, :iterations)
end

struct AAOutput
    solution::Vector{Float64}        # Converged result
    input::AAInput   # G(solution)
    analysis::AAAnalysisOutput
end




