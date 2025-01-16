# Structure for the output of the algorithm


struct AAHistory
	history::NamedTuple	
end

struct AAAlgorithm
	method::NamedTuple
	methodparams::NamedTuple
end

struct AAOutput
    solution::Vector{Float64}        # Converged result
    functionval::Vector{Float64}    # G(solution)
    residual::Vector{Float64}
    history::AAHistory       # Residual norms ||x - G(x)||
    conv::Bool                      # Success indicator
    errcode::Int                    # Error code (e.g., 0 for success, 10 for maxit exceeded)
    algorithm::AAAlgorithm
end



struct AAConvParams
	convparams::NamedTuple
end

struct AAInput
    GFix!::Function                 # Fixed-point map (GFix!(G, x))
    x0::Vector{Float64}			# Initial iterate
    algorithm::AAAlgorithm
end

struct AAAnalysis
    performance_metrics::NamedTuple # Metrics to track (e.g., :residual, :iterations)
end
