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
Apply a non-linear reflector to a vector in-place.

This function applies a non-linear reflector operation to the given vector `x`. 
The function also modifies the vector in-place.

# Arguments
- `x::AbstractVector{T}`: The input vector.

# Returns
- `T`: The reflector scaling factor.
"""
@inline function nl_reflector!(x::AbstractVector{T}) where {T}
    n = length(x)
    n == 0 && return zero(eltype(x))
    @inbounds begin
        ξ1 = x[1]
        normu = norm(x)
        if iszero(normu)
            return zero(ξ1/normu)
        end

        safe_min = ((1/prevfloat(T(Inf))) * (T(1.0) + 4*eps(T)))/eps(T)
        inv_safe_min = inv(safe_min)
        count = 0
        while normu < safe_min && count < 20
            count += 1
            x *= inv_safe_min
            normu *= inv_safe_min
        end
        if count > 0
            ξ1 = x[1]
            normu = norm(x)
        end

        ν = T(copysign(normu, real(ξ1)))
        ξ1 += ν
        x[1] = -ν
        for i = 2:n
            x[i] /= ξ1
        end
    end
    safe_min^count * ξ1/ν
end

"""
Perform pivoted QR decomposition with tolerance-based filtering.

This function applies a pivoted QR decomposition to the input matrix `A`, 
with an optional tolerance-based filtering mechanism.

# Arguments
- `A::AbstractMatrix{T}`: The input matrix.
- `tol`: The tolerance for pivoting. Defaults to machine epsilon.

# Returns
- `Tuple`: A tuple containing the pivoted QR decomposition and a boolean array indicating deleted columns.
"""
function paqr_piv(A::AbstractMatrix{T}, args...; kwargs...) where {T}
    AA = copy(A)
    paqr_piv!(AA, args...; kwargs...)
end


"""
Perform pivoted QR decomposition with tolerance-based filtering.

This function applies a pivoted QR decomposition to the input matrix `A`, 
with an optional tolerance-based filtering mechanism.

# Arguments
- `A::AbstractMatrix{T}`: The input matrix.
- `tol`: The tolerance for pivoting. Defaults to machine epsilon.

# Returns
- `Tuple`: A tuple containing the pivoted QR decomposition and a boolean array indicating deleted columns.
"""
function paqr_piv!(A::AbstractMatrix{T}; tol=eps(T)) where {T}
    m, n = size(A)
    τ = zeros(T, min(m,n))
    piv = Vector(UnitRange{LinearAlgebra.BlasInt}(1,n))
    deleted = falses(n)  # Initialize the deleted vector

    A_copy = copy(A)

    k = 1
    keep_count = 0
    i = 1 # iterations in the first pass
    while i <= min(m, n)
        x = copy(A[k:m, k]) # copy to avoid mutating A on rejected columns
        τk = nl_reflector!(x)
        #println("--$i: $(norm(A[k:m, k])), $τk, $(abs(x[1]))")
        # This is doing a fixed tolerance each time! We want to divide by norm of original vector.
        #
        #if abs(x[1]) >= tol
        #
        #So we compare abs(x[1]) >= tol * norm(A[:,k])
        #println(abs(x[1]))
        #println(sqrt(norm(A[1:k-1,k])^2+abs(x[1])^2))
        #We also base it on history k
        #println(τk)
        if abs(x[1])/norm(A[:,k]) >= tol
            τ[k] = τk
            A[k:m, k] = x
            LinearAlgebra.reflectorApply!(x, τk, view(A, k:m, k + 1:n))
            k += 1
            keep_count += 1
        else
            # rotate to the end
            x = A[:, k]
            #p = piv[k]
            for j = k:n-1
                A[:, j] = A[:, j+1]
            #    piv[j] = piv[j+1]
            end
            A[:, n] = x
            #piv[n] = p
            deleted[i] = true
        end

        i += 1
    end

    # remaining columns are all rejects.  Don't pivot further, just compute valid R22
    #while k <= min(m - 1 + !(T<:Real), n)
    #    x = view(A, k:m, k)
    #    τk = nl_reflector!(x)
    #    τ[k] = τk
    #    LinearAlgebra.reflectorApply!(x, τk, view(A, k:m, k + 1:n))
    #    k += 1
    #end

    QRP = LinearAlgebra.QRPivoted{eltype(A[:,1:keep_count]), typeof(A[:,1:keep_count])}(A[:,1:keep_count], τ[1:keep_count], piv[1:keep_count])

    #if verbose
    #    println("--Relative reconstruction norm: $(opnorm(QRP.Q * QRP.R * QRP.P' - A_copy)/opnorm(A_copy))")
    #end

    return QRP, deleted
end

##################### Pollock Functions #################################

"""
Filter based on angular thresholds.

Filters columns of the input historical structure `HS` based on angular 
thresholding, determined by the parameter `cs`.

# Arguments
- `HS`: Historical structure containing matrices `X_k` and `G_k`.
- `cs`: Angular threshold for filtering.

# Returns
- `AbstractVector{Bool}`: A boolean array indicating filtered elements.
"""
function AngleFiltering!(HS, cs)
    X_k = HS.X_k
    G_k = HS.G_k
        # Perform QR decomposition of F
    Q, R = qr(G_k)
    
    # Compute sigma values
    sigma = zeros(Float64, size(G_k, 2))
    for i in 1:length(sigma)
        sigma[i] = abs(R[i,i]) / norm(G_k[:,i])
    end
    
    # Find indices where sigma > cs
    indicesKeep = findall(sigma .> cs)
    kept = falses(size(X_k,2))
    
    # Update E and F by keeping only the selected columns
    HS.X_k = X_k[:, indicesKeep]
    HS.G_k = G_k[:, indicesKeep]
    
    # Update m to the new size of E
    kept[indicesKeep] .= true

    return .!kept
end

"""
    LengthFiltering!(HS, cs, kappabar)

Performs length filtering on matrices `X_k` and `G_k` within the `HS` object by reducing dimensions based on a specified threshold `kappabar`.

### Parameters:
- `HS::Object`: A structure containing `X_k` (data matrix) and `G_k` (gradient matrix).
- `cs::Float64`: A scaling parameter used for transformation and dimensional reduction.
- `kappabar::Float64`: A threshold controlling the maximum allowed cumulative variance.

### Returns:
- `filtered::Vector{Bool}`: A boolean vector indicating which columns were filtered out.
"""
function LengthFiltering!(HS,cs,kappabar)
    X_k = HS.X_k
    G_k = HS.G_k
    ct = sqrt(1 - cs^2)
    ncol = size(G_k, 2)

    # Compute Fnormi
    Fnormi = [norm(G_k[:, i]) for i in 1:ncol]

    # Initialize b
    b = zeros(eltype(G_k), ncol)
    b[1] = 1 / Fnormi[1]^2
    b[2] = 1 / cs^2 * (ct^2 / Fnormi[1]^2 + 1 / Fnormi[2]^2)

    for j in 3:ncol
        term1 = ct^2 * (ct + cs)^(2 * (j - 2)) / (Fnormi[1]^2 * cs^(2 * (j - 2)))
        term2 = 0.0
        for i in 2:(j - 1)
            term2 += ct^2 * (ct + cs)^(2 * (j - i - 1)) / (Fnormi[i]^2 * cs^(2 * (j - i)))
        end
        term3 = 1 / Fnormi[j]^2
        b[j] = 1 / cs^2 * (term1 + term2 + term3)
    end

    # Reduce dimensions based on kappabar
    m = ncol
    for k in ncol:-1:1
        Cf = sum(Fnormi[1:k].^2) * sum(b[1:k])
        if Cf < kappabar^2
            m = k
            break
        end
    end
    filtered = trues(size(X_k,2))

    # Resize X_k and G_k in place by reassigning sliced views
    HS.X_k = X_k[:, 1:m]
    HS.G_k = G_k[:, 1:m]

    filtered[1:m] .= false
    return filtered
end

function pad_to_next_power_of_2(x::Vector{Float64})
    n = length(x)
    next_pow2 = 2 ^ ceil(Int, log2(n))
    padded_x = vcat(x, zeros(next_pow2 - n))  # Pad with zeros
    return padded_x, n  # Return padded signal and original length
end

# Wavelet compression function with padding
function wavelet_compress(x::Vector{Float64}, wt, ratio::Float64)
    padded_x, original_length = pad_to_next_power_of_2(x)  # Pad signal
    coeffs = dwt(padded_x, wt, 4)  # Compute wavelet transform on padded signal
    
    num_to_zero = Int(floor(length(coeffs) * (1 - 1 / ratio)))
    
    # Get indices of smallest coefficients
    sorted_indices = sortperm(abs.(coeffs))
    zero_indices = sorted_indices[1:num_to_zero]
    coeffs[zero_indices] .= 0.0  # Zero out the smallest coefficients
    
    return coeffs, original_length  # Return modified coefficients and original length
end

function wavelet_decompress(coeffs, wt, original_length::Int)
    # Inverse wavelet transform
    decompressed = idwt(coeffs, wt, 4) 
    
    # Trim the decompressed signal back to the original length
    return decompressed[1:original_length]
end

"""
    Computes an adaptive compression ratio based on the ratio of residuals and iteration number.

    Arguments:
    - residual_ratio: (r_new / r_old) ratio of new residual to old residual.
    - iteration: Current iteration number.

    Returns:
    - compression_ratio: Percentage of coefficients to zero out (between 5% and 30%).
"""
function compute_compression_ratio(residual_ratio::Float64, iteration::Int)
    base_compression = 20  # Initial compression level
    sensitivity = 0.1

    # Adjust compression: if residual improves fast, reduce compression
    compression_ratio = base_compression - sensitivity * residual_ratio

    # Clamp between 5% and 30%
    return clamp(compression_ratio, 1.0, 50.0)
end

##########################################
########## Method Specific Stuff #########
##########################################

"""
    createAAMethod(method::Symbol; methodparams=nothing)

Creates an `AAMethod` object with specified or default parameters for various Anderson Acceleration (AA) methods.

### Parameters:
- `method::Symbol`: A symbol representing the AA method (e.g., `:vanilla`, `:paqr`, `:faa`).
- `methodparams::NamedTuple`: (Optional) A named tuple of parameters for the method. If not provided, default parameters are used.

### Returns:
- `AAMethod`: An object encapsulating the method and its parameters.
"""
function createAAMethod(method::Symbol; methodparams=nothing)::AAMethod
    # Define default parameters for each method
    defaults = Dict(
        :vanilla => (m = 2),
        :paqr => (threshold = 1e-5),
        :faa => (cs = 0.1, kappabar = 1, m = 20),
        :fftaa => (m = 10, tf  = 0.9),
        :dwtaa => (m=10),
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
        :paqr => [:threshold],
        :faa => [:cs, :kappabar, :m],
        :fftaa => [:m, :tf],
        :dwtaa => [:m],
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
        :hdexplicit => [:beta, :threshold],
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

"""
    initialise_historicalstuff(methodname::Symbol, x_k::Vector)

Initializes a historical storage object for a specific AA method, based on the provided method name and input vector `x_k`.

### Parameters:
- `methodname::Symbol`: The name of the AA method (`:vanilla`, `:paqr`, `:faa`).
- `x_k::Vector{T}`: The initial vector `x_k` used for initializing historical data.

### Returns:
- `HistoricalStuff`: A structure for storing historical data specific to the AA method.
"""
function initialise_historicalstuff(method::AAMethod,x_0::Vector)
    methodname = method.methodname
    if methodname == :vanilla
        return VanillaHistoricalStuff([],[],0) # Carries Solhist and Residual and iterations...
    elseif methodname == :paqr
        return PAQRHistoricalStuff([],[],[],[],0)
    elseif methodname == :faa
        return FAAHistoricalStuff(length(x_0))
    elseif methodname == :fftaa
        return FFTAAHistoricalStuff(length(x_0),method.methodparams.tf)
    elseif methodname == :dwtaa
        return DWTAAHistoricalStuff(length(x_0))
    else
        error("Unsupported AAMethod: $methodname")
    end
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
