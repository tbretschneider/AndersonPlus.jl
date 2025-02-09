

###############################
# Method Specific Structs #######
###############################


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
    FFTG_k::Matrix{ComplexF64}  # Matrix of Float64
    FFTX_k::Matrix{ComplexF64}  # Matrix of Float64
    g_km1::Vector{Float64}  # Vector of Float64
    iterations::Int  # Integer

    # Constructor with default empty values
    function FFTAAHistoricalStuff(numrows::Int,tf::Float64)
        new(Matrix{ComplexF64}(undef,Int(ceil(tf*numrows)),0), Matrix{ComplexF64}(undef,Int(ceil(numrows*tf)),0), Vector{Float64}(undef,numrows), 0)
    end
end

mutable struct DWTAAHistoricalStuff <: HistoricalStuff
    DWTG_k::Matrix{Float64}  # Matrix of Float64
    DWTF_k::Matrix{Float64}  # Matrix of Float64
    residual::Float64
    iterations::Int  # Integer

    # Constructor with default empty values
    function DWTAAHistoricalStuff(numrows::Int)
        new(Matrix{Float64}(undef,2 ^ ceil(Int, log2(numrows)),0), Matrix{Float64}(undef,2 ^ ceil(Int, log2(numrows)),0), 1.0,0)
    end
end

mutable struct RFAAHistoricalStuff <: HistoricalStuff
    G_k::Matrix{Float64}  # Matrix of Float64
    F_k::Matrix{Float64}  # Matrix of Float64
    residual::Float64
    iterations::Int  # Integer

    # Constructor with default empty values
    function RFAAHistoricalStuff(numrows::Int)
        new(Matrix{Float64}(undef,numrows,0), Matrix{Float64}(undef,numrows,0), 1.0,0)
    end
end

mutable struct RFLSAAHistoricalStuff <: HistoricalStuff
    Gcal_k::Matrix{Float64}  # Matrix of Float64
    Xcal_k::Matrix{Float64}  # Matrix of Float64
    g_km1::Vector{Float64}  # Vector of Float64
    iterations::Int  # Integer

    # Constructor with default empty values
    function RFLSAAHistoricalStuff(numrows::Int)
        new(Matrix{Float64}(undef,numrows,0), Matrix{Float64}(undef,numrows,0), Vector{Float64}(undef,numrows), 0)
    end
end

mutable struct quickAAHistoricalStuff <: HistoricalStuff
    GtildeTGtildeinv::Symmetric{Float64, Matrix{Float64}}  # Symmetric matrix of size (m × m)
    Ninv::Diagonal{Float64, Vector{Float64}}  # Diagonal matrix of size (m × m)
    F_k::Matrix{Float64}  # Regular matrix of size (numrows × m)
    Gtilde_k::Matrix{Float64}  # Regular matrix of size (numrows × m)
    sin_k::Vector{Float64}  # Vector of size (m)
    positions::Vector{Int}  # Vector of size (m)
    iterations::Int  # Integer counter

    # Constructor
    function quickAAHistoricalStuff(numrows::Int, m::Int)
        GtildeTGtildeinv = Symmetric(zeros(m, m))  # Symmetric matrix of size (m × m)
        Ninv = Diagonal(zeros(m))  # Diagonal matrix of size (m × m)
        F_k = zeros(numrows, m)  # Regular matrix (numrows × m)
        Gtilde_k = zeros(numrows, m)  # Regular matrix (numrows × m)
        sin_k = zeros(m)  # Vector of size (m)
        positions = fill(-1, m)  # Vector of size (m), initialized to -1
        iterations = 0  # Start at zero iterations

        return new(GtildeTGtildeinv, Ninv, F_k, Gtilde_k, sin_k, positions, iterations)
    end
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
    elseif methodname == :rfaa
        return RFAAHistoricalStuff(length(x_0))
    elseif methodname == :rflsaa
        return RFLSAAHistoricalStuff(length(x_0))
    elseif methodname == :quickaa
        return quickAAHistoricalStuff(length(x_0),method.methodparams.m)
    else
        error("Unsupported AAMethod: $methodname")
    end
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
        :rfaa => (m=10, probfun = (it, len) -> [1-i/(2*len*len) for i in 1:len]),
        :rflsaa => (m=10, probfun = (it, len) -> [1-i/(2*len*len) for i in 1:len]),
        :quickaa => (m=10, threshold_func = (itnums, curr) -> [1.0 for i in 1:length(itnums)]),
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
        :rfaa => [:m, :probfun],
        :rflsaa => [:m, :probfun],
        :quickaa => [:m, :threshold_func],
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


SD = Dict(
    :methodname => "Method Name",
    :methodparams => "Method Parameters",
    :algorithmparams => "Algorithm Parameters",
    :convparams => "Convergence Parameters",
    :iterations => "Iterations",
    :vanilla => "Vanilla",
    :paqr => "Pivoting Avoiding QR",
    :faa => "Filtered (Pollock)",
    :fftaa => "Truncated Fourier Transformed History",
    :dwtaa => "Wavelet Transformed History",
    :rfaa => "Randomised Filtering",
    :rflsaa => "Randomised Filtering for Least Squares",
    :quickaa => "Rank one updates to inverse...",
)