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

function geometriccond(A::AbstractMatrix)
    
    # Calculate the norm of each column
    col_norms = norm.(eachcol(A))
    
    # Normalize the columns of A
    A_normalized = A ./ reshape(col_norms, (1, size(A, 2)))
    
    # Calculate and return the condition number
    return cond(A_normalized)
end

function geometriccond(x::Float64)
    return isnan(x) ? NaN : x
end

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

function paqr_piv(A::AbstractMatrix{T}, args...; kwargs...) where {T}
    AA = copy(A)
    paqr_piv!(AA, args...; kwargs...)
end
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


function createAAMethod(method::Symbol; methodparams=nothing)::AAMethod
    # Define default parameters for each method
    defaults = Dict(
        :vanilla => (m = 2),
        :paqr => (threshold = 1e-5),
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

function initialise_historicalstuff(methodname::Symbol)
    if methodname == :vanilla
        return VanillaHistoricalStuff([],[],0) # Carries Solhist and Residual and iterations...
    elseif methodname == :paqr
        return PAQRHistoricalStuff([],[],[],[],0)
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
