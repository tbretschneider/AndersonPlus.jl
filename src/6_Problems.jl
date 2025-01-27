function p3_f_helper!(u_re_im::Vector{Float64}, k0::Float64, ε::Float64, N::Int)
    """
    Performs one Picard iteration for the Nonlinear Helmholtz equation with separate real
    and imaginary parts packed into a single vector.

    Parameters:
        u_re_im :: Vector{Float64} : Current solution (real and imaginary parts concatenated).
                                     The first N entries are the real part, the next N are the imaginary part.
        k0      :: Float64         : Linear wavenumber.
        ε       :: Float64         : Nonlinear coefficient.
        N       :: Int             : Number of grid points.
        dx      :: Float64         : Grid spacing.

    Modifies:
        u_re_im :: Vector{Float64} : Updates `u_re_im` in-place with the result of one Picard iteration.
    """
    # Split real and imaginary parts
    dx = 10/N

    u_re = @view u_re_im[1:N+1]
    u_im = @view u_re_im[N+2:end]
    
    # Compute the midpoints using linear interpolation
    u_re_midpoints = 0.5 .* (u_re[1:end-1] + u_re[2:end])
    u_im_midpoints = 0.5 .* (u_im[1:end-1] + u_im[2:end])

    # Compute the sum of squares of the midpoints
    sum2 = (norm(u_re_midpoints)^2+norm(u_im_midpoints)^2) * dx^2

    # Combine into complex representation for calculations
    u = complex(u_re)+ im*complex(u_im)

    # Construct finite difference matrix (second-order centered difference)
    main_diag = complex([(-2+dx^2*k0^2*(1+ε*(u_re[i]^2+u_im[i]^2))) for i in 1:N+1])
    off_diag = complex(ones(N))
    A = spdiagm(-1 => off_diag,0=>main_diag, 1=>off_diag)

    # Apply boundary conditions to the system matrix
    A[1, 1] = -3 + 2*dx*im * k0  # Left boundary, real part
    A[1, 2] = 4
    A[1,3] = -1
    A[end, end] = 3 - 2*dx*im*k0  # Right boundary, real part
    A[end, end - 1] = -4
    A[end,end-2] = 1

    # Boundary conditions source term (real and imaginary components separately)
    b_re = zeros(Float64, N+1)
    b_im = zeros(Float64, N+1)
    b_im[1] = 2*2.0 * k0 * dx  # Imaginary part of boundary condition at x = 0

    b = complex(b_re) +im * complex(b_im)    
    
    newu = A \ b

    return vcat(real(newu),imag(newu))
end


# Parameters
k0 = 8.0
ε = 0.2
N = 2000
x_start = 0.0
x_end = 10.0
x = range(x_start, x_end, length=N+1)

function p3_f!(G,u)
    G .= AndersonPlus.p3_f_helper!(u, k0, ε, N)
end

u_re = cos.(k0 * x)
u_im = sin.(k0 * x)
x_0 = vcat(u_re, u_im); # Concatenate into a single vector


"""
Hequation.jl

This file contains the function/Jacobian evaluations for
the Chandrasekhar H-equation examples and everything you should 
need to run them.

If you only want to run the examples, you should not have to look
at the code.
"""
### And now for the functions ...
"""
function heqJ!(FP,F,x,pdata)

The is the Jacobian evaluation playing by nsol rules. The
precomputed data is a big deal for this one. 
"""
function heqJ!(FP::Array{T,2}, F, x, pdata) where {T<:Real}
    pseed = pdata.pseed
    mu = pdata.mu
    n = length(x)
    #
    # Look at the formula in the notebook and you'll see what I did here.
    #
    pmu = pdata.pmu
    Gfix = pdata.gtmp
    @views Gfix .= x - F
    @views Gfix .= -(Gfix .* Gfix .* pmu)
    @views @inbounds for jfp = 1:n
        FP[:, jfp] .= Gfix[:, 1] .* pseed[jfp:jfp+n-1]
        FP[jfp, jfp] = 1.0 + FP[jfp, jfp]
    end
    return FP
end

"""
heqf!(F,x,pdata)

The function evaluation as per nsold rules.

The precomputed data is a big deal for this example. In particular,
the output pdata.FFB from plan_fft! goes to the fixed point map
computation. Things get very slow if you do not use plan_fft or plan_fft!
"""
function heqf!(F, x, pdata)
    HeqFix!(F, x, pdata)
    #
    # naked BLAS call to fix the allocation blues
    #
    # Using any variation of F.=x-F really hurts
    #
    axpby!(1.0, x, -1.0, F)
    return F
end


"""
function HeqFix!(Gfix,x,pdata)
The fixed point map. Gfix goes directly into the function and
Jacobian evaluations for the nonlinear equations formulation.

The precomputed data is a big deal for this example. In particular, 
the output pdata.FFA from plan_fft goes to the fixed point map
computation. Things get very slow if you do not use plan_fft. 
"""
function HeqFix!(Gfix, x, pdata)
    n = length(x)
    Gfix .= x
    heq_hankel!(Gfix, pdata)
    Gfix .*= pdata.pmu
    Gfix .= 1.0 ./ (1.0 .- Gfix)
end

"""
heqinit(x0::Array{T,1}, c) where T :< Real

Initialize H-equation precomputed data.
"""
function heqinit(x0::Array{T,1}, c) where {T<:Real}
    (c > 0) || error("You can't set c to zero.")
    n = length(x0)
    cval = ones(1)
    cval[1] = c
    vsize = (n)
    bsize = (2 * n,)
    ssize = (2 * n - 1,)
    FFA = plan_fft(ones(bsize))
    mu = collect(0.5:1:n-0.5)
    pmu = mu * c
    mu = mu / n
    hseed = zeros(ssize)
    for is = 1:2*n-1
        hseed[is] = 1.0 / is
    end
    hseed = (0.5 / n) * hseed
    pseed = hseed
    gtmp = zeros(vsize)
    rstore = zeros(bsize)
    zstore = zeros(bsize) * (1.0 + im)
    hankel = zeros(bsize) * (1.0 + im)
    FFB = plan_fft!(zstore)
    bigseed = zeros(bsize)
    @views bigseed .= [hseed[n:2*n-1]; 0; hseed[1:n-1]]
    @views hankel .= conj(FFA * bigseed)
    return (
        cval = cval,
        mu = mu,
        hseed = hseed,
        pseed = pseed,
        gtmp = gtmp,
        pmu = pmu,
        rstore = rstore,
        zstore = zstore,
        hankel = hankel,
        FFB = FFB,
    )
end


"""
setc!(pdata, cin)

If you are varying c in a computation, this function
lets you set it. 

But! You can't set c to zero.
"""
function setc!(pdata, cin)
    (cin > 0) || error("You can't set c to zero")
    c = pdata.cval[1]
    cfix = cin / c
    pdata.pmu .*= cfix
    pdata.cval[1] = cin
end


"""
heq_hankel!(b,pdata)
Multiply an nxn Hankel matrix with seed in R^(2N-1) by a vector b
FFA is what you get with plan_fft before you start computing
"""
function heq_hankel!(b, pdata)
    reverse!(b)
    heq_toeplitz!(b, pdata)
end


"""
heq_toeplitz!(b,pdata)
Multiply an nxn Toeplitz matrix with seed in R^(2n-1) by a vector b
"""
function heq_toeplitz!(b, pdata)
    n = length(b)
    y = pdata.rstore
    y .*= 0.0
    @views y[1:n] = b
    heq_cprod!(y, pdata)
    @views b .= y[1:n]
end

"""
heq_cprod!(b,pdata)
Circulant matrix-vector product with FFT
compute u = C b

Using in-place FFT
"""

function heq_cprod!(b, pdata)
    xb = pdata.zstore
    xb .*= 0.0
    xb .+= b
    pdata.FFB \ xb
    hankel = pdata.hankel
    xb .*= hankel
    pdata.FFB * xb
    b .= real.(xb)
end

"""
Alternative formulation for CI. Tuned to match the paper

author="P. B. Bosma and W. A. DeRooij",
title="Efficient Methods to Calculate Chandrasekhar's H-functions",
journal="Astron. Astrophys.",
volume=126,
year=1983,
pages=283

"""
function heqbos!(F, x, pdata)
    c = pdata
    n = length(x)
    mu = 0.5:1:n-0.5
    mu = mu / n
    h = 1.0 / n
    cval = sqrt(1.0 - c)
    A = zeros(n, n)
    for j = 1:n
        for i = 1:n
            A[i, j] = mu[j] / (mu[i] + mu[j])
        end
    end
    A = (c / 2) * h * A
    F .= (A * x)
    for ig = 1:n
        F[ig] = 1.0 / (cval + F[ig])
    end
    F .= x - F
end


function P6(n,c)
    u0 = ones(n)
    hdata = heqinit(u0, c)

    return AAProblem((G,u) -> HeqFix!(G,u,hdata),
                u0,
                AAConvParams(1e-10, 0))
end
           