using LinearAlgebra

"""
Perform ridge regression.

This function performs ridge regression, a regularization technique 
that adds a penalty term to the least squares solution to prevent 
overfitting. The penalty term is controlled by the parameter `λ`.

# Arguments
- `A::AbstractMatrix`: The design matrix.
- `b::AbstractVector`: The response vector.
- `λ::Real=1e-10`: The regularization parameter. Defaults to a small value.

# Returns
- `AbstractVector`: The ridge regression solution vector.
"""
function ridge_regression(A::AbstractMatrix, b::AbstractVector, λ::Real=1e-10)::AbstractVector
    n, m = size(A)
    I_mat = λ * I(m)
    result = (A' * A + I_mat) \ (A' * b)
    return result
end

"""
Convert gamma coefficients to alpha coefficients.

This function transforms a vector of gamma coefficients into 
alpha coefficients based on the specified transformation rules.

# Arguments
- `gamma::AbstractVector`: The input vector of gamma coefficients.

# Returns
- `AbstractVector`: The resulting vector of alpha coefficients.
"""
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


"""
Convert gamma coefficients to alpha coefficients.

This function transforms a vector of gamma coefficients into 
alpha coefficients based on the specified transformation rules.

# Arguments
- `gamma::AbstractVector`: The input vector of gamma coefficients.

# Returns
- `AbstractVector`: The resulting vector of alpha coefficients.
"""
function gamma_to_alpha(x::Float64)
    return isnan(x) ? NaN : x
end


"""
Check tolerances between two vectors.

This function checks if two vectors `x` and `y` satisfy specified absolute 
or relative tolerances.

# Arguments
- `x::Vector{Float64}`: The first vector.
- `y::Vector{Float64}`: The second vector.
- `tolparams::AAConvParams`: A structure containing the tolerances `atol` (absolute) and `rtol` (relative).

# Returns
- `Bool`: `true` if the tolerances are satisfied, otherwise `false`.
"""
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

"""
Calculate the geometric condition number of a matrix.

This function normalizes the columns of a matrix and computes its 
condition number.

# Arguments
- `A::AbstractMatrix`: The input matrix.

# Returns
- `Real`: The geometric condition number of the matrix.
"""
function geometriccond(A::AbstractMatrix)
    
    # Calculate the norm of each column
    col_norms = norm.(eachcol(A))
    
    # Normalize the columns of A
    A_normalized = A ./ reshape(col_norms, (1, size(A, 2)))
    
    # Calculate and return the condition number
    return cond(A_normalized)
end

"""
Calculate the geometric condition number of a matrix.

This function normalizes the columns of a matrix and computes its 
condition number.

# Arguments
- `A::AbstractMatrix`: The input matrix.

# Returns
- `Real`: The geometric condition number of the matrix.
"""
function geometriccond(x::Float64)
    return isnan(x) ? NaN : x
end



"""
    AAAnalysisOutput(input::AAInput, fullmidanalysis::Vector{Any}, iterations::Int)

Transforms a vector of mid-analysis results into a structured `NamedTuple` for output, including method and algorithm parameters.

### Parameters:
- `input::AAInput`: Input object containing the algorithm and problem information.
- `fullmidanalysis::Vector{Any}`: Vector of NamedTuples representing the analysis data.
- `iterations::Int`: The number of iterations performed during the analysis.

### Returns:
- `AAAnalysisOutput`: A structured output with detailed results.
"""
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

import Base: !

function !(x::AbstractFloat)
	    isnan(x) ? NaN : !Bool(x)
end
