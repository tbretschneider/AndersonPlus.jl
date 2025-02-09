"""
PDE_Tools

This file has the operators I need for the PDE example. They
live in a separate file to make the CI easier for me to organize.
"""
# Famous sparse matrices
"""
Dx2d(n)

returns x partial on n x n grid.
Unit square, homogeneous Dirichlet BC
"""
function Dx2d(n)
    h = 1 / (n + 1)
    ssdiag = ones(n^2 - 1) / (2 * h)
    for iz = n:n:n^2-1
        ssdiag[iz] = 0.0
    end
    updiag = Pair(1, ssdiag)
    lowdiag = Pair(-1, -ssdiag)
    Dx = spdiagm(lowdiag, updiag)
    return Dx
end

"""
Dy2d(n)

returns y partial on n x n grid.
Unit square, homogeneous Dirichlet BC
"""
function Dy2d(n)
    h = 1 / (n + 1)
    ssdiag = ones(n^2 - n) / (2 * h)
    updiag = Pair(n, ssdiag)
    lowdiag = Pair(-n, -ssdiag)
    Dy = spdiagm(lowdiag, updiag)
    return Dy
end

"""
Lap2d(n)

returns the negative Laplacian in two space dimensions
on n x n grid.

Unit square, homogeneous Dirichlet BC
"""
function Lap2d(n)
    # hm2=1/h^2
    hm2 = (n + 1.0)^2
    maindiag = fill(4 * hm2, (n^2,))
    sxdiag = fill(-hm2, (n^2 - 1,))
    sydiag = fill(-hm2, (n^2 - n,))
    for iz = n:n:n^2-1
        sxdiag[iz] = 0.0
    end
    D2 = spdiagm(-n => sydiag, -1 => sxdiag, 0 => maindiag, 1 => sxdiag, n => sydiag)
    return D2
end


"""
u=fish2d(f, fdata)

Fast Poisson solver in two space dimensions.
Same as the Matlab code.
Unit square + homogeneous Dirichlet BCs.

Grid is nx by nx

You give me f as a two-dimensional vector f(x,y).
I return the solution u.
"""
function fish2d(f, fdata)
    u = fdata.utmp
    v = fdata.uhat
    T = fdata.T
    ST = fdata.ST
    (nx, ny) = size(f)
    nx == ny || error("need a square grid in fish2d")
    u .= f
    u = ST * u
    u = u'
    u1 = reshape(u, (nx * nx,))
    v1 = reshape(v, (nx * nx,))
    v1 .= u1
    ldiv!(u1, T, v1)
    u = u'
    u .= ST * u
    u ./= (2 * nx + 2)
    return u
end

"""
fishinit(n)

Run FFTW.plan_r2r to set up the solver. Do not mess
with this function.
"""
function fishinit(n)
    #
    # Get the sine transform from FFTW. This is faster/better/cleaner
    # than what I did in the Matlab codes.
    #
    zstore = zeros(n, n)
    ST = FFTW.plan_r2r!(zstore, FFTW.RODFT00, 1)
    uhat = zeros(n, n)
    fishu = zeros(n, n)
    TD = newT(n)
    T = lu!(TD)
    fdata = (ST = ST, uhat = uhat, utmp = zstore, T = T, fishu = fishu)
    return fdata
end

"""
T = newT(n)

Builds the n^2 x n^2 sparse tridiagonal matrix for
the 2D fast Poisson solver.
"""
function newT(n)
    N = n * n
    h = 1 / (n + 1)
    x = h:h:1-h
    h2 = 1 / (h * h)
    LE = 2 * (2 .- cos.(pi * x)) * h2
    fn = ones(N - 1) * h2
    gn = ones(N - 1) * h2
    dx = zeros(N)
    for k = 1:n-1
        fn[k*n] = 0.0
        gn[k*n] = 0.0
        dx[(k-1)*n+1:n*k] = LE[k] * ones(n)
    end
    dx[(n-1)*n+1:n*n] = LE[n] * ones(n)
    T = Tridiagonal(-fn, dx, -gn)
    return T
end

"""
Use fish2d and reshape for preconditioning.
"""
function Pfish2d(v, fdata)
    n2 = length(v)
    n = Int(sqrt(n2))
    (n * n == n2) || error("input to Pfish2d not a square array")
    v2 = reshape(v, (n, n))
    u = fish2d(v2, fdata)
    u = reshape(u, (n2,))
    return u
end

"""
	Pvec2d(v, u, pdata)

Returns inverse Laplacian * v

u is a dummy argument to make nsoli happy

Preconditioner for nsoli
"""
function Pvec2d(v, u, pdata)
    fdata = pdata.fdata
    p = Pfish2d(v, fdata)
    return p
end


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
    G .= AndersonPlus.p3_f_helper!(copy(u), k0, ε, N)
end

function p3_f!(G,u,x0,ε, N)
    G .= AndersonPlus.p3_f_helper!(copy(u), k0, ε, N)
end

u_re = cos.(k0 * x)
u_im = sin.(k0 * x)
x_0 = vcat(u_re, u_im); # Concatenate into a single vector


function P3(k0, ε, N)
    
    x_start = 0.0
    x_end = 10.0
    x = range(x_start, x_end, length=N+1)

    u_re = cos.(k0 * x)
    u_im = sin.(k0 * x)
    x_0 = vcat(u_re, u_im); # Concatenate into a single vector
    
    return AAProblem((G,u) -> p3_f!(G,copy(u), k0, ε, N),
            x_0,
            AAConvParams(1e-10, 0))
end



####################################################################3

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



##################################################################################
"""
pdeF!.jl

This file contains everything you need to run the Ellptic PDE examples.  
This includes the version with an explict sparse matrix Jacobian and
the fixed point formulations using the fish2d.jl preconditioner.

I've also parked the exact solution in here so you can do the grid refinement
study.

Look at pdeinit for the construction of the precomputed data. There is
a lot of it.

If you only want to run the examples, you should not have to look
at the code.
"""
### And now for the functions ...
"""
pdeF!(FV, u, pdata)

Residual using sparse matrix-vector multiplication
"""
function pdeF!(FV, u, pdata)
    D2 = pdata.D2
    CV = pdata.CV
    rhs = pdata.RHS
    p1 = pdata.jvect1
    #    FV .= D2 * u + 20.0 * u .* (CV * u) - rhs
    #    FV .= D2*u
    #    p1 .= CV*u
    mul!(FV, D2, u)
    mul!(p1, CV, u)
    p1 .*= 20.0
    p1 .*= u
    FV .+= p1
    FV .-= rhs
end

"""
pdeJ!(FP, F, u, pdata)

Sparse matrix Jacobian. The package does not do its own sparse
differencing. The Jacobian for this problem is easy enough to 
compute analytically.
"""
function pdeJ!(FP, F, u, pdata)
    D2 = pdata.D2
    CV = pdata.CV
    CT = pdata.CT
    cu = CV * u
    #DC=spdiagm(0 => 20*cu); DU=spdiagm(0 => 20*u)
    DC = Diagonal(20 * cu)
    DU = Diagonal(20 * u)
    #
    # The easy way to compute the Jacobian is 
    #FP .= D2 + DU*CV + DC
    # but you allocate yourself silly with that one.
    # So we preallocate room for DU*CV in CT and sum the terms for FP
    # one at a time. I have to use Diagonal instead of spdiagm if I want
    # mul! to work fast.
    #
    FP .= D2
    FP .+= DC
    mul!(CT, DU, CV)
    #CT .= CV; lmul!(DU,CT); 
    FP .+= CT
    # I should be able to do mul!(FP,DU,CV), but it's 1000s of times slower.
end

"""
Jvec2d(v, FS, u, pdata)

Analytic Jacobian-vector product for PDE example
"""
function Jvec2d(v, FS, u, pdata)
    D2 = pdata.D2
    CV = pdata.CV
    CT = pdata.CT
    jvec = pdata.jvec
    p1 = pdata.jvect1
    #    jvec .= D2 * v
    #    p1 .= CV * u
    mul!(jvec, D2, v)
    mul!(p1, CV, u)
    p1 .*= 20.0
    p1 .*= v
    jvec .+= p1
    #    p1 .= CV * v
    mul!(p1, CV, v)
    p1 .*= 20.0
    p1 .*= u
    jvec .+= p1
    return jvec
end

"""
hardleft!(FV, u, pdata)
Convection-diffusion equation with left preconditioning hard-wired in

"""
function hardleft!(FV, u, pdata)
    fdata = pdata.fdata
    # Call the nonlinear function
    FV = pdeF!(FV, u, pdata)
    # and apply the preconditioner.
    FV .= Pfish2d(FV, fdata)
    return FV
end

"""
hardleftFix!(FV, u, pdata)
Fixed point form of the left preconditioned nonlinear
convection-diffusion equation
"""
function hardleftFix!(FV, u, pdata)
    FV = hardleft!(FV, u, pdata)
    # G(u) = u - FV
    axpby!(1.0, u, -1.0, FV)
    return FV
end


"""
pdeinit(n)

collects the precomputed data for the elliptic pde example. This 
includes 

- the sparse matrix representation of the operators, 
- the right side of the equation,
- the exact solution,
- the data that the fft-based fast Poisson solver (fish2d) needs
"""
function pdeinit(n)
    # Make the grids
    n2 = n * n
    h = 1.0 / (n + 1.0)
    x = collect(h:h:1.0-h)
    # collect the operators
    D2 = Lap2d(n)
    DX = Dx2d(n)
    DY = Dy2d(n)
    CV = (DX + DY)
    # I need a spare sparse matrix to save allocations in the Jacobian computation
    CT = copy(CV)
    # Exact solution and its derivatives
    uexact = solexact(x)
    dxe = dxexact(x)
    dye = dyexact(x)
    d2e = l2dexact(x)
    dxv = reshape(dxe, n2)
    dyv = reshape(dye, n2)
    d2v = reshape(d2e, n2)
    uv = reshape(uexact, n2)
    fdata = fishinit(n)
    # The right side of the equation
    RHS = d2v + 20.0 * uv .* (dxv + dyv)
    # preallocate a few vectors
    jvec = zeros(n2)
    jvect1 = zeros(n2)
    # Pack it and ship it.
    pdedata =
        (D2 = D2, CV = CV, CT = CT, RHS = RHS, jvec, jvect1, fdata = fdata, uexact = uexact)
end

"""
This collection of functions 
builds u, u_x, u_y, and the negative Laplacian for the 
example problem in the book. Here
u(x,y) = 10 x y (1-x)(1-y) exp(x^4.5)

which is the example from FA01.
"""

function w(x)
    w = 10.0 * x .* (1.0 .- x) .* exp.(x .^ (4.5))
end

function wx(x)
    wx = 4.5 * (x .^ (3.5)) .* w(x) + 10.0 * exp.(x .^ (4.5)) .* (1.0 .- 2.0 * x)
end

function wxx(x)
    wxx =
        (4.5 * 3.5) * (x .^ (2.5)) .* w(x) +
        4.5 * (x .^ (3.5)) .* wx(x) +
        +10.0 * 4.5 * (x .^ (3.5)) .* exp.(x .^ (4.5)) .* (1.0 .- 2.0 * x) +
        -20.0 * exp.(x .^ (4.5))
end

function v(x)
    v = x .* (1.0 .- x)
end

function vx(x)
    vx = 1.0 .- 2.0 * x
end

function vxx(x)
    vxx = -2.0 * ones(size(x))
end

function solexact(x)
    solexact = w(x) * v(x)'
end

function l2dexact(x)
    l2dexact = -(w(x) * vxx(x)') - (wxx(x) * v(x)')
end

function dxexact(x)
    dxexact = wx(x) * v(x)'
end

function dyexact(x)
    dxexact = w(x) * vx(x)'
end



function P7(n)
    
    pdata = pdeinit(n)
    fdata = pdata.fdata
    fone = ones(n * n)
    u0 = zeros(n * n)
    FV = copy(u0)
    JV = zeros(n * n, 20)

    return AAProblem((G,u) -> hardleftFix!(G,u,pdata),
            u0,
            AAConvParams(1e-10, 0))
end
    
