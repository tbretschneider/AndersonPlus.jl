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

function checktolerances(x::Vector{Float64}, y::Vector{Float64}, tolparams::NamedTuple)

    atol = tolparams[:atol]
    rtol = tolparams[:rtol]

    # Check tolerances
    if atol == 0.0 && rtol == 0.0
        return false
    end

    if norm(x .- y) < atol || 
       (norm(x .- y) / min(norm(x), norm(y))) < rtol || 
       x == y
        return true
    end

    return false
end
