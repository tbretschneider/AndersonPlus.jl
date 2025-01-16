# Structure for the output of the algorithm

struct AAMethod
    methodname::NamedTuple
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
    metricnames::NamedTuple # Metrics to track (e.g., :residual, :iterations)
end

struct AAInput
    problem::AAProblem               # Fixed-point map (GFix!(G, x))
    algorithm::AAAlgorithm			# Initial iterate
    metrics::AAAnalysis
end

struct AAAnalysisOutput
    metrics::NamedTuple # Metrics to track (e.g., :residual, :iterations)
end

struct AAOutput
    solution::Vector{Float64}        # Converged result
    input::AAInput   # G(solution)
    analysis::AAAnalysisOutput
end




