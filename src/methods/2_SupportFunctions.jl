

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

function sample_bool_vector(probabilities::Vector{Float64})
    return rand(length(probabilities)) .> probabilities
end

function RandomFilter!(HS,probfun)
    if HS.iterations in (0,1)
        return NaN
    end
    probabilities = probfun(HS.iterations,size(HS.G_k,2))
    filtered = sample_bool_vector(probabilities)
    HS.G_k = HS.G_k[:,.!filtered]
    HS.F_k = HS.F_k[:,.!filtered]
    return filtered
end

function RandomFilterLS!(HS,probfun)
    if HS.iterations in (0,1)
        return NaN
    end
    probabilities = probfun(HS.iterations,size(HS.Gcal_k,2)-1)
    filtered = vcat(true,sample_bool_vector(probabilities))
    HS.Gcal_k = HS.Gcal_k[:,.!filtered]
    HS.Xcal_k = HS.Xcal_k[:,.!filtered]
    return filtered
end

function AnglesUpdate!(HS, gtilde_k)
    for (i,position) in enumerate(HS.positions)
        if position != -1
            HS.sin_k[i] .= dot(gtilde_k, HS.Gtilde_k[:,i])
        end
    end
end

function Filtering!(HS,methodparams)
    filteredindices = filteringindices(HS,methodparams)
    filtersforview = filteredindices[HS.positions .!= -1]

    for x in reverse((1:length(filtersforview))[filtersforview])
        updateinverse!(@view HS.GtildeTGtildeinv[HS.positions .!= -1,HS.positions .!= -1]
		       ,x)
        (@view HS.positions[HS.positions .!= -1])[x] .= -1
    end

end

function filteringindices(HS, methodparams)
    threshold_func = methodparams[:threshold_func]  # Extract threshold function
    m = methodparams[:m]
    thresholds = threshold_func(HS.positions,HS.iterations)   # Compute thresholds for each index

    # Create a boolean vector indicating whether each entry should be filtered
    filteredindices = HS.sin_k .> thresholds

    return filteredindices
end

using LinearAlgebra.BLAS, Random

# Really nice as we can pass extra stuff - even for multiplication 
# as zeros are inserted in the rest of the places

function updateinverse!(inverse::Symmetric{T, Matrix{T}}, index::Int) where T
    A = inverse.data  # Access underlying matrix (modifies in place)
    α = inverse[index, index]

    # Define the update vector
    v = inverse[:, index]  # View to avoid allocations

    # Perform a symmetric rank-one update using BLAS
    #A -= inv(α)*v*v'

    BLAS.syr!('U', -inv(α), v, A)  # Only updates upper triangular part

end

function addinverse!(inverse::Symmetric{T, Matrix{T}}, index::Int,u1) where T
    A = inverse.data  # Extract the underlying matrix
    d = inverse[index,index]
    u2 = BLAS.symv('U', A, u1) #Calculates A*u1
    d = inv(1 - dot(u1,u2))
    u3 = d*u2
    u3[index] = d
    BLAS.syr!('U', d, u2, A)  # Only updates upper triangular part, calculates A + d*u2*u2'
    A[index,vcat(1:index-1,index+1:end)] = -u3'
    A[vcat(1:index-1,index+1:end),index] = -u3
    inverse[index,index] = d
end

function AddNew!(HS,n_kinv)
    #no -1 somewhere first. Need to find position
    index = findfirst(x -> x == -1, HS.positions)
    if isnothing(index)
        index = argmin(HS.positions)
        HS.positions[index] .= -1
        updateinverse!(HS.GtildeTGtildeinv,index)
        HS.sin_k[index] = 1.0
        addinverse!(HS.GtildeTGtildeinv,
			index,
			HS.sin_k)
        HS.Ninv.diag[index] = n_kinv
        HS.positions[index] .= HS.iterations
    else
        HS.positions[index] = HS.iterations
        HS.sin_k[index] = 1.0
        addinverse!(HS.GtildeTGtildeinv,
        index,
        HS.sin_k)
        HS.Ninv.diag[index] = n_kinv
    end
end

