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

mutable struct PAQRHistoricalStuff <: HistoricalStuff
    residual::Vector{Vector{Float64}}
    solhist::Vector{Vector{Float64}}
    G::Vector{Vector{Float64}}
    F::Vector{Vector{Float64}}
    iterations::Int
end

mutable struct FAAHistoricalStuff <: HistoricalStuff
    G_k::Matrix{Float64}  # Matrix of Float64
    X_k::Matrix{Float64}  # Matrix of Float64
    g_km1::Vector{Float64}  # Vector of Float64
    iterations::Int  # Integer

    # Constructor with default empty values
    function FAAHistoricalStuff(numrows::Int)
        new(Matrix{Float64}(undef,numrows,0), Matrix{Float64}(undef,numrows,0), Vector{Float64}(undef,numrows), 0)
    end
end

mutable struct FFTAAHistoricalStuff <: HistoricalStuff
    FFTG_k::Matrix{Float64}  # Matrix of Float64
    FFTX_k::Matrix{Float64}  # Matrix of Float64
    g_km1::Vector{Float64}  # Vector of Float64
    iterations::Int  # Integer

    # Constructor with default empty values
    function FFTAAHistoricalStuff(numrows::Int,tf::Float64)
        new(Matrix{Float64}(undef,Int(ceil(tf*numrows)),0), Matrix{Float64}(undef,Int(ceil(numrows*tf)),0), Vector{Float64}(undef,numrows), 0)
    end
end