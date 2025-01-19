using LinearAlgebra

# Perform ridge regression
function ridge_regression(A::AbstractMatrix, b::AbstractVector, λ::Real=1e-10)::AbstractVector
    n, m = size(A)
    I_mat = λ * I(m)
    result = (A' * A + I_mat) \ (A' * b)
    return result
end

# Convert gamma coefficients to alpha coefficients
function gamma_to_alpha(gamma::AbstractVector)::AbstractVector
    m_k = length(gamma)
    alpha = similar(gamma, eltype(gamma), m_k + 1) # Create vector of appropriate type and size

    # Apply the transformation rules
    alpha[1] = gamma[1] # First element: alpha_0 = gamma_0
    if m_k > 1
        for j in 2:m_k # Elements 1 to m_k - 1: alpha_j = gamma_j - gamma_{j-1}
            alpha[j] = gamma[j] - gamma[j-1]
        end
    end
    alpha[m_k + 1] = 1 - gamma[m_k] # Last element: alpha_{m_k} = 1 - gamma_{m_k-1}

    return alpha
end

function gamma_to_alpha(x::Float64)
    return isnan(x) ? NaN : x
end


function checktolerances(x::Vector{Float64}, y::Vector{Float64}, tolparams::AAConvParams)

    atol = tolparams.atol
    rtol = tolparams.rtol

    # Check tolerances
    if atol == 0.0 && rtol == 0.0
        return false
    end

    if norm(x .- y) <= atol || 
       (norm(x .- y) / min(norm(x), norm(y))) <= rtol || 
       x == y
        return true
    end

    return false
end

function createAAMethod(method::Symbol; methodparams=nothing)::AAMethod
    # Define default parameters for each method
    defaults = Dict(
        :vanilla => (m = 2),
        :ipoptjumpvanilla => (m = 3, beta = 1.0),
        :picard => (beta = 1.0),
        :function_averaged => (beta = 1.0, m = 3, sample_size = 10),
        :runs_averaged => (beta = 1.0, m = 3, sample_size = 10),
        :runs_greedy => (beta = 1.0, m = 3, sample_size = 10),
        :probabilistic => (beta = 1.0, pdf = (i, n) -> 0.9 - 0.8 * (i + 1) / (n + 1)),
        :dynamic_probabilistic => (beta = 1.0, hardcutoff = 10, scaling = x -> 3, curprobs = []),
        :probabilistic_coordinate_importance => (beta = 1.0, coordsampleprop = 0.1, probdecrease = 0.1, hardcutoff = 10, replace = true, curprobs = []),
        :apci => (beta = 1.0, coordsampleprop = 0.1, probdecrease = 0.1, hardcutoff = 10, replace = true, curprobs = []),
        :ci => (beta = 1.0, coordsampleprop = 0.1, probdecrease = 0.0, hardcutoff = 10, replace = true, curprobs = []),
        :ap => (beta = 1.0, coordsampleprop = 1, probdecrease = 0.1, hardcutoff = 10, replace = false, curprobs = []),
        :thresh => (beta = 1.0, percentvariance = 0.9),
        :anglefilter => (beta = 1.0, anglethreshold = 0.1, hardcutoff = 20),
        :faa => (beta = 1.0, cs = 0.1, cond = 1, hardcutoff = 20),
        :hdexplicit => (beta = 1.0, threshold = 1e-5)
    )
    
    # Map method parameters to their expected structure
    param_mappings = Dict(
        :vanilla => [:m],
        :ipoptjumpvanilla => [:m, :beta],
        :picard => [:beta],
        :function_averaged => [:m, :beta, :sample_size],
        :runs_averaged => [:m, :beta, :sample_size],
        :runs_greedy => [:m, :beta, :sample_size],
        :probabilistic => [:beta, :pdf],
        :dynamic_probabilistic => [:beta, :hardcutoff, :scaling],
        :probabilistic_coordinate_importance => [:beta, :hardcutoff, :coordsampleprop, :probdecrease],
        :apci => [:beta, :hardcutoff, :coordsampleprop, :probdecrease],
        :ci => [:beta, :hardcutoff, :coordsampleprop],
        :ap => [:beta, :hardcutoff, :probdecrease],
        :thresh => [:beta, :percentvariance],
        :anglefilter => [:beta, :anglethreshold, :hardcutoff],
        :faa => [:beta, :cs, :cond, :hardcutoff],
        :hdexplicit => [:beta, :threshold]
    )

    # Handle the case where no parameters are provided
    if isnothing(methodparams)
        params = get(defaults, method, error("Unknown method: $method"))
    else
        # Map provided parameters to method-specific structure
        param_keys = get(param_mappings, method, error("Unknown method: $method"))
        params = NamedTuple{param_keys}(methodparams)
    end

    # Return an AAMethod object
    return AAMethod(method, params)
end

function initialise_historicalstuff(methodname::Symbol)
    if methodname == :vanilla
        return VanillaHistoricalStuff([], [], 0)
    else
        error("Unsupported AAMethod: $methodname")
    end
end

function AAAnalysisOutput(input::AAInput,fullmidanalysis::Vector{Any},iterations::Int)
    fields = keys(fullmidanalysis[1])
    
    # Transform the vector of NamedTuples into a NamedTuple of Vectors
    output = [(field => [i[field] for i in fullmidanalysis]) for field in fields]

    output = NamedTuple(output)

    output = merge(output,(methodname = input.algorithm.method.methodname, 
    methodparams = input.algorithm.method.methodparams,
    algorithmparams = input.algorithm.algorithmparams,
    convparams = input.problem.convparams,
    iterations = iterations))

    return AAAnalysisOutput(output)
end
