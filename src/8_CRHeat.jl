using QuadGK: gauss


"""
Orthogonalize!(V, hv, vv, orth; verbose=false)

C. T. Kelley, 2022

Orthogonalize the Krylov vectors using your (my) choice of
methods. Anything other than classical Gram-Schmidt twice (cgs2) is
likely to become an undocumented and UNSUPPORTED option. Methods other 
than cgs2 are mostly for CI for the linear solver.

This orthogonalizes the vector vv against the columns of V
and stores the coefficients in the vector hv. You preallocate
hv. The length of hv is the number of columns of VV + 1. vv is 
overwritten by the orthogonalized unit vector.

DO NOT use anything other than "cgs2" with Anderson acceleration.

I do not export this function, but use it in my own work and in 
the new MultiPrecisionArrays package. 
"""
function Orthogonalize!(V, hv, vv, orth = "cgs2"; verbose = false)
    orthopts = ["mgs1", "mgs2", "cgs1", "cgs2"]
    orth in orthopts || error("Impossible orth spec in Orthogonalize!")
    if orth == "mgs1"
        mgs!(V, hv, vv; verbose = verbose)
    elseif orth == "mgs2"
        mgs!(V, hv, vv, "twice"; verbose = verbose)
    elseif orth == "cgs1"
        cgs!(V, hv, vv, "once"; verbose = verbose)
    else
        cgs!(V, hv, vv, "twice"; verbose = verbose)
    end
end

"""
mgs!(V, hv, vv, orth; verbose=false)
"""
function mgs!(V, hv, vv, orth = "once"; verbose = false)
    k = length(hv) - 1
    normin = norm(vv)
    #p=copy(vv)
    @views for j = 1:k
        p = vec(V[:, j])
        hv[j] = p' * vv
        vv .-= hv[j] * p
    end
    hv[k+1] = norm(vv)
    if (normin + 0.001 * hv[k+1] == normin) && (orth == "twice")
        @views for j = 1:k
            p = vec(V[:, j])
            hr = p' * vv
            hv[j] += hr
            vv .-= hr * p
        end
        hv[k+1] = norm(vv)
    end
    nv = hv[k+1]
    #
    # Watch out for happy breakdown
    #
    #if hv[k+1] != 0
    #@views vv .= vv/hv[k+1]
    (nv != 0) || (verbose && (println("breakdown in mgs1")))
    if nv != 0
        vv ./= nv
    end
end

"""
cgs!(V, hv, vv, orth="twice"; verbose=false)

Classical Gram-Schmidt.
"""
function cgs!(V, hv, vv, orth = "twice"; verbose = false)
    #
    #   no explicit BLAS calls. mul! seems faster than BLAS 
    #   since 1.6 and allocates far less memory.
    #
    k = length(hv)
    T = eltype(V)
    onep = T(1.0)
    zerop = T(0.0)
    @views rk = hv[1:k-1]
    pk = zeros(T, size(rk))
    qk = vv
    Qkm = V
    # Orthogonalize
    # New low allocation stuff
    mul!(rk, Qkm', qk, 1.0, 1.0)
    ###    mul!(pk, Qkm', qk)
    ###    rk .+= pk
    ##    rk .+= Qkm' * qk
    #    qk .-= Qkm * rk
    mul!(qk, Qkm, rk, -1.0, 1.0)
    if orth == "twice"
        # Orthogonalize again
        # New low allocation stuff
        mul!(pk, Qkm', qk)
        ##        pk .= Qkm' * qk
        #        qk .-= Qkm * pk
        mul!(qk, Qkm, pk, -1.0, 1.0)
        rk .+= pk
    end
    # Keep track of what you did.
    nqk = norm(qk)
    (nqk != 0) || (verbose && (println("breakdown in cgs")))
    (nqk > 0.0) && (qk ./= nqk)
    hv[k] = nqk
end

"""
kl\\_gmres(x0, b, atv, V, eta, ptv=nothing; kl_store=nothing; 
             orth = "cgs2", side="right", lmaxit=-1, pdata=nothing)

C. T. Kelley, 2022

Gmres linear solver. Handles preconditioning and restarts. 
Uses gmres_base which is completely oblivious to these things.

The deal is

Input:\n
x0:  initial iterate, this is usually zero for nonlinear solvers

b: right hand side (duh!)

atv:  matrix-vector product which depends on precomputed data pdta
      I expect you to use pdata most or all of the time, so it is not
      an optional argument, even if it's nothing (at least for now). 
      If your mat-vec is just A*v, you have to write a function where 
      A is the precomputed data.
      API for atv is ```av=atv(v,pdata)```

V:  Preallocated n x K array for the Krylov vectors. I store the initial
    normalized residual in column 1, so  you have at most K-1 iterations
    before gmres\\_base returns a failure. kl\\_gmres will handle the 
    restarts and, if lmaxit > 0, keep going until you hit lmaxit GMRES
    iterations. You may allocate V in Float32 and save on storage. The
    benefit from doing this is not dramatic in terms of CPU time.

eta: Termination happens when ||b - Ax|| <= eta || b ||

ptv:  preconditioner-vector product, which will also use pdata. The
      default is nothing, which is no preconditioning at all.
      API for ptv is px=ptv(x,pdata)

Keyword arguments

kl\\_store: You have the option (don't do it!) of giving me some room
         for the vectors gmres needs. These include copies of x0 and b,
         which I will not overwrite and a couple of vectors I use
         in the iteration. If you're only doing a linear solve, PLEASE
         let me allocate those vectores in kl\\_gmres. For computing a
         Newton step or for repeated solves,
         the way to do this is ```kl_store=kstore(n,"gmres")``` where n
         is the number of unknows. I call this myself in the initialization
         phase if you don't do it ahead of me.

         Be very careful with this. kl_store is use to store the solution
         to avoid overwriting the initial iterate. This means that
         two calls to kl_gmres with the same kl_store will step on the
         solution coming from the first call. If you let me allocate it
         then it happens in local scope and will do no harm.

pdata: precomputed data. The default is nothing, but that ain't gonna
        work well for nonlinear equations.

orth: your choice of the wise default, classical Gram-Schmidt twice,
       or something slower and less stable. Those are classical once (really
       bad) or a couple variants of modified Gram-Schmidt. mgs2 is what I
       used in my old matlab codes. Not terrible, but far from great.

side: left or right preconditioning. The default is "right".

lmaxit: maximum number of linear iterations. The default is -1, which
        means that the maximum number of linear iterations is K-1, which
        is all V will allow without restarts. If lmaxit > K-1, then the
        iteration will restart until you consume lmaxit iterations or
        terminate successfully.

Other parameters on the way.

Output:\n
A named tuple (sol, reshist, lits, idid)

where

sol= final result
reshist = residual norm history
lits = number of iterations
idid = status of the iteration
       true -> converged 
       false -> failed to converge
              
 
"""
function kl_gmres(
    x0,
    b,
    atv,
    V,
    eta,
    ptv = nothing;
    kl_store = nothing,
    orth = "cgs2",
    side = "right",
    lmaxit = -1,
    pdata = nothing,
)

    # Build some precomputed data to inform KL_atv about 
    # preconditioning ...
    # Do not overwrite the initial iterate or the right hand side.
    n = length(x0)
    # Get the vectors GMRES needs internally and make room to
    # copy the initial iterate and right side
    (kl_store !== nothing) || (kl_store = kstore(n, "gmres"))
    y0 = kl_store[1]
    y0 .= x0
    rhs = kl_store[2]
    rhs .= b
    # Two vectors for internals
    linsol = kl_store[3]
    restmp = kl_store[4]
    #
    if side == "right" || ptv == nothing
        itsleft = false
    else
        itsleft = true
        rhs .= ptv(rhs, pdata)
    end
    (n, K) = size(V)
    K > 1 || error("Must allocate for GMRES iterations. V must have 
                   at least two columns")
    klmaxit = lmaxit
    lmaxit > 0 || (lmaxit = K - 1)
    #
    itvec = maxitvec(K, lmaxit)
    ip = 1
    idid = false
    Kpdata =
        (pdata = pdata, side = side, ptv = ptv, atv = atv, linsol = linsol, restmp = restmp)
    gout = []
    #
    # Restarted GMRES loop. 
    #
    while ip <= length(itvec) && idid == false
        localout =
            gmres_base(y0, rhs, Katv, V, eta, Kpdata; lmaxit = itvec[ip], orth = orth)
        idid = localout.idid
        gout = outup(gout, localout, ip, klmaxit)
        reslen = length(localout.reshist)
        #
        ip += 1
    end
    #
    # Fixup the solution if preconditioning from the right.
    #
    sol = y0
    if side == "left" || ptv == nothing
        return (sol = sol, reshist = gout.reshist, lits = gout.lits, idid = gout.idid)
    else
        sol .= ptv(sol, pdata)
        return (sol = sol, reshist = gout.reshist, lits = gout.lits, idid = gout.idid)
    end
end

"""
Katv(x,Kpdata)

Builds a matrix-vector product to hand to gmres_base or bicgstab_base. 
Puts the preconditioner in there on the correct side.
"""
function Katv(x, Kpdata)
    #    y=copy(x)
    y = Kpdata.linsol
    pdata = Kpdata.pdata
    ptv = Kpdata.ptv
    atv = Kpdata.atv
    side = Kpdata.side
    sideok = (side == "left") || (side == "right")
    sideok || error(
        "Bad preconditioner side in Krylov solver, input side = ",
        side,
        ". Side must be \"left\" or \"right\" ",
    )
    if ptv == nothing
        y .= atv(x, pdata)
        return y
    elseif side == "left"
        y .= atv(x, pdata)
        return ptv(y, pdata)
    elseif side == "right"
        y .= ptv(x, pdata)
        return atv(y, pdata)
    end
end

"""
gmres_base(x0, b, atv, V, eta, pdata; orth="cgs2", lmaxit=-1)

Base GMRES solver. This is GMRES(m) with no restarts and no preconditioning.
The idea for the future is that it'll be called by kl_gmres (linear
solver) which is the backend of klgmres.

gmres_base overwrites x0 with the solution. This is one of many reasons
that you should not invoke it directly.
"""
function gmres_base(x0, b, atv, V, eta, pdata; orth = "cgs2", lmaxit = -1)

    (n, m) = size(V)
    #
    # Allocate for Givens
    #
    #    kmax = m - 1
    kmax = m
    lmaxit == -1 || (kmax = lmaxit)
    kmax > m - 1 && error("lmaxit error in gmres_base")
    r = pdata.restmp
    r .= b
    T = eltype(V)
    h = zeros(T, kmax + 1, kmax + 1)
    c = zeros(kmax + 1)
    s = zeros(kmax + 1)
    #
    # Don't do the mat-vec if the intial iterate is zero
    #
    #    y = pdata.linsol
    (norm(x0) == 0.0) || (r .-= atv(x0, pdata))
    #    (norm(x0) == 0.0) || (y .= atv(x0, pdata); r .-=y;)
    #
    #
    rho0 = norm(r)
    rho = rho0
    #
    # Initial residual = 0? This can't be good.
    #
    rho == 0.0 && error("Initial resdiual in kl_gmres is zero. Why?")
    #
    g = zeros(size(c))
    g[1] = rho
    errtol = eta * norm(b)
    reshist = []
    #
    # Initialize
    #
    idid = true
    push!(reshist, rho)
    k = 0
    #
    # Showtime!
    #
    #    @views V[:, 1] .= r / rho
    @views v1 = V[:, 1]
    copy!(v1, r)
    rhoinv = 1.0 / rho
    v1 .*= rhoinv
    #    @views V[:,1] ./= rho
    beta = rho
    while (rho > errtol) && (k < kmax)
        k += 1
        @views V[:, k+1] .= atv(V[:, k], pdata)
        @views vv = vec(V[:, k+1])
        @views hv = vec(h[1:k+1, k])
        @views Vkm = V[:, 1:k]
        #
        # Don't mourn. Orthogonalize!
        #
        Orthogonalize!(Vkm, hv, vv, orth)
        #
        # Build information for new Givens rotations.
        #   
        if k > 1
            hv = @view h[1:k, k]
            giveapp!(c[1:k-1], s[1:k-1], hv, k - 1)
        end
        nu = norm(h[k:k+1, k])
        if nu != 0
            c[k] = conj(h[k, k] / nu)
            s[k] = -h[k+1, k] / nu
            h[k, k] = c[k] * h[k, k] - s[k] * h[k+1, k]
            h[k+1, k] = 0.0
            gv = @view g[k:k+1]
            giveapp!(c[k], s[k], gv, 1)
        end
        #
        # Update the residual norm.
        #
        rho = abs(g[k+1])
        (nu > 0.0) || (println("near breakdown"); rho = 0.0)
        push!(reshist, rho)
    end
    #
    # At this point either k = kmax or rho < errtol.
    # It's time to compute x and check out.
    #
    y = h[1:k, 1:k] \ g[1:k]
    #    qmf = view(V, 1:n, 1:k)
    @views qmf = V[:, 1:k]
    #    mul!(r, qmf, y)
    #    r .= qmf*y    
    #    x .+= r
    #    sol = x0
    #    mul!(sol, qmf, y, 1.0, 1.0)
    mul!(x0, qmf, y, 1.0, 1.0)
    (rho <= errtol) || (idid = false)
    k > 0 || println("GMRES iteration terminates on entry.")
    return (rho0 = rho0, reshist = Float64.(reshist), lits = k, idid = idid)
end

function giveapp!(c, s, vin, k)
    for i = 1:k
        w1 = c[i] * vin[i] - s[i] * vin[i+1]
        w2 = s[i] * vin[i] + c[i] * vin[i+1]
        vin[i:i+1] .= [w1, w2]
    end
    return vin
end

#
# The functions maxitvec and outup manage the restarts.
# There is no reason to look at them or fiddle with them.
#

function maxitvec(K, lmaxit)
    levels = Int.(ceil(lmaxit / (K - 1)))
    itvec = ones(Int, levels)
    itvec[1:levels-1] .= K - 1
    remainder = lmaxit - (levels - 1) * (K - 1)
    itvec[levels] = remainder
    return itvec
end

function outup(gout, localout, ip, klmaxit)
    idid = localout.idid
    #
    # If I'm doing restarts I won't store the last residual
    # unless the iteration is successful. The reason is that
    # I will add that residual to the list when I restart.
    #
    if idid || klmaxit == -1
        lreshist = localout.reshist
    else
        lk = length(localout.reshist)
        lreshist = localout.reshist[1:lk-1]
    end
    if ip == 1
        reshist = lreshist
        lits = localout.lits
    else
        reshist = gout.reshist
        append!(reshist, lreshist)
        lits = gout.lits + localout.lits
    end
    gout = (reshist = reshist, lits = lits, idid = idid)
    return gout
end

"""
FCR_heat!(FS, x, hdata)

Nonlinear equation form of conductive-radiative heat transfer problem.
"""
function FCR_heat!(FS, x, hdata)
    FS = p5_f!(FS, x, hdata)
    FS .= x - FS
    #axpy!(-1.0, x, FS)
    return FS
end

"""
p5_f!(theta, thetain, hn_data)

Fixed point map for the conductive-radiative heat transfer problem.
"""
function p5_f!(theta, thetain, hn_data)
    epsl = 1.0
    epsr = 1.0
    sn_data = hn_data.sn_data
    nx = length(thetain)
    theta .= thetain
    source = sn_data.tmphf
    source .*= 0.0
    rhsd2 = hn_data.rhsd2
    bcfix = hn_data.bcfix
    D2 = hn_data.D2
    Nc = hn_data.Nc
    omega = hn_data.omega
    source .= theta
    source .^= 4
    source .*= (1.0 - omega)
    ltol = 1.e-12
    flux = flux_solve(source, hn_data, ltol)
    @views copy!(rhsd2, flux[2:nx-1])
    rhsd2 .*= (1.0 - omega)
    @views axpy!(-2.0, source[2:nx-1], rhsd2)
    pn = 1.0 / (2.0 * Nc)
    rhsd2 .*= pn
    ldiv!(D2, rhsd2)
    theta[1] = 0.0
    theta[nx] = 0.0
    @views theta[2:nx-1] .= rhsd2
    axpy!(1.0, bcfix, theta)
    return theta
end

"""
heat_init(nx, na, thetal, thetar, omega, tau, Nc)

Set up the conductive-radiative heat transfer problem

I pass a named tuple of precomputed and preallocated data to
all the functions and solvers. 
"""
function heat_init(nx, na, thetal, thetar, omega, tau, Nc)
    # Get the 1D Laplacian at the interior nodes. Form and store the LDLt
    # facorization
    np = nx - 2
    D2M = Lap1d(np)
    D2 = ldlt(D2M)
    # Preallocate some room. I'm using kstore to store the internal
    # vectors for kl_gmres since I do a complete GMRES iteration
    # for every call to the fixed point map. Kids, don't try this at home!
    rhsd2 = zeros(np)
    h = tau / (nx - 1.0)
    kl_store = kstore(nx, "gmres")
    xv = collect(0:h:tau)
    bcfix = thetal .+ (thetar - thetal) * xv
    #
    # Precomputed data for the transport problem.
    #
    sn_data = sn_init(nx, na, x -> omega, tau, thetal^4, thetar^4)
    #
    # Stuff it all in one place.
    #
    hn_data = (
        sn_data = sn_data,
        bcfix = bcfix,
        D2 = D2,
        rhsd2 = rhsd2,
        omega = omega,
        Nc = Nc,
        kl_store = kl_store,
        thetal = thetal,
        thetar = thetar,
    )
    return hn_data
end


"""
sn_init(nx, na2, fs, tau, vleft, vright; siewert=false)

I pass a named tuple of precomputed and preallocated data to
all the functions and solvers. 

The input to this is obvious stuff.

nx = number of spatial grid points

na2 = number of angles. The angular mesh is (na2/2) Gaussian quadaratures
      on [-1,0) and (0,1]

fs:function ; scattering coefficient is fs(x)


Boundary conditions for the transport problem are constant vectors
filled with vleft/vright.

phi_left, phi_right = ones(na2/2) * vleft/vright
"""
function sn_init(nx, na2, fs, tau, vleft, vright; siewert = false)
    #
    # Set up the quadrature rule in angle
    #
    # Only used for CI
    if siewert
        #
        # I don't need the weights to make tables, but I need
        # to return something.
        #
        angles = [-0.05; collect(-0.1:-0.1:-1.0); 0.05; collect(0.1:0.1:1.0)]
        weights = angles
        # the real deal
    else
        #        (angles, weights) = hard_gauss()
        (angles, weights) = sn_angles(na2)
    end
    na = floor(Int, na2 / 2)
    #
    # scattering coefficient
    #
    dx = tau / (nx - 1)
    x = collect(0:dx:tau)
    c = fs.(x)
    #
    # Preallocated storage for intermediate results
    #
    phi0 = zeros(nx)
    tmpf = zeros(nx)
    tmp1 = zeros(nx)
    tmphf = zeros(nx)
    rhsg = zeros(nx)
    ptmp = zeros(na)
    #
    # Preallocated storage for source iteration
    #
    psi_left = vleft * ones(na)
    psi_right = vright * ones(na)
    # Preallocating the angular flux is not really necessary
    # since you can compute the scalar flux on the fly as you do it.
    # However, the preallocation makes the code much easier to understand
    # and map to/from the text.
    psi = zeros(na2, nx)
    source_average = zeros(nx - 1)
    source_total = zeros(nx)
    #
    # Preallocated storage for the Krylov basis in the GMRES solve
    #
    V = zeros(nx, 13)
    #
    return sn_data = (
        c = c,
        dx = dx,
        psi = psi,
        angles = angles,
        weights = weights,
        phi0 = phi0,
        tmp1 = tmp1,
        tmpf = tmpf,
        tmphf = tmphf,
        rhsg = rhsg,
        source_average = source_average,
        source_total = source_total,
        nx = nx,
        ptmp = ptmp,
        psi_left = psi_left,
        psi_right = psi_right,
        V = V,
    )
end


#function hard_gauss()
#
# Return the weights/nodes for double 20 pt gauss
# I could use FastGaussQuadrature.jl for this but am
# trying to avoid dependencies, especially for big things
# like StaticArrays.jl
#
# If you want to try FastGaussQuadrature.jl, see the function below,
# which I have commented out.
#
#    m = 40
#    ri = zeros(40)
#    wi = zeros(40)
#    r = zeros(40)
#    w = zeros(40)
#    ri[20] = 0.993128599185095
#    ri[19] = 0.963971927277914
#    ri[18] = 0.912234428251326
#    ri[17] = 0.839116971822218
#    ri[16] = 0.746331906460151
#    ri[15] = 0.636053680726515
#    ri[14] = 0.510867001950827
#    ri[13] = 0.373706088715420
#    ri[12] = 0.227785851141645
#    ri[11] = 0.076526521133497
#    wi[20] = 0.017614007139152
#    wi[19] = 0.040601429800387
#    wi[18] = 0.062672048334109
#    wi[17] = 0.083276741576705
#    wi[16] = 0.101930119817240
#    wi[15] = 0.118194531961518
#    wi[14] = 0.131688638449177
#    wi[13] = 0.142096109318382
#    wi[12] = 0.149172986472604
#    wi[11] = 0.152753387130726
#    for i = 1:10, ri[i] in -ri[21-i]
#        wi[i] = wi[21-i]
#    end
#    mm = floor(Int, m / 2)
#    for i = 1:mm
#        r[i+mm] = (1.0 + ri[i]) * 0.5
#        w[i+mm] = wi[i] * 0.5
#        r[i] = -r[i+mm]
#        w[i] = wi[i] * 0.5
#    end
#    return (r, w)
#end


"""
sn_angles(na2=40)

Get double Gauss nodes and weights for SN
This function uses FastGaussQuadrature
"""
function sn_angles(na2 = 40)
    na = floor(Int, na2 / 2)
    2 * na == na2 || error("odd number of angles")
    baseangles, baseweights = gauss(na)
    posweights = baseweights * 0.5
    negweights = copy(posweights)
    posangles = (baseangles .+ 1.0) * 0.5
    negangles = -copy(posangles)
    weights = [negweights; posweights]
    angles = [negangles; posangles]
    angles, weights
end


"""
flux_solve(source, hn_data, tol)

Solve the transport equation with the source from the heat
conduction problem. The output is what kl_gmres returns, so
the solution is kout.sol
"""
function flux_solve(source, hn_data, tol)
    sn_data = hn_data.sn_data
    b = getrhs(source, sn_data)
    kl_store = hn_data.kl_store
    kout =
        kl_gmres(sn_data.phi0, b, AxB, sn_data.V, tol; pdata = sn_data, kl_store = kl_store)
    return kout.sol
end

function AxB(flux, sn_data)
    nx = length(flux)
    angles = sn_data.angles
    na2 = length(angles)
    na = floor(Int, na2 / 2)
    #tmp1=zeros(nx)
    #tmpf=zeros(nx)
    tmpf = sn_data.tmpf
    tmp1 = sn_data.tmp1
    tmp1 .*= 0.0
    tmpf .= flux
    tmp2 = zeros(na)
    tmpf = source_iteration!(tmpf, tmp2, tmp2, tmp1, sn_data)
    axpy!(-1.0, flux, tmpf)
    tmpf .*= -1.0
    return tmpf
end

function getrhs(source, sn_data)
    nx = sn_data.nx
    #rhs=zeros(nx)
    rhs = sn_data.rhsg
    rhs .*= 0.0
    angles = sn_data.angles
    na2 = length(angles)
    na = floor(Int, na2 / 2)
    rhs = source_iteration!(rhs, sn_data.psi_left, sn_data.psi_right, source, sn_data)
    return rhs
end

function source_iteration!(flux, psi_left, psi_right, source, sn_data)
    psi = sn_data.psi
    psi = transport_sweep!(psi, flux, psi_left, psi_right, source, sn_data)
    weights = sn_data.weights
    nx = sn_data.nx
    na2 = length(weights)
    #
    # Take the 0th moment to get the flux.
    #
    g = reshape(flux, 1, nx)
    wt = reshape(weights, 1, na2)
    mul!(g, wt, psi)
    return flux
end

"""
transport_sweep!(psi, phi, psi_left, psi_right, source, sn_data)

Take a single transport sweep.
"""
function transport_sweep!(psi, phi, psi_left, psi_right, source, sn_data)
    angles = sn_data.angles
    #
    c = sn_data.c
    dx = sn_data.dx
    #
    na2 = length(angles)
    na = floor(Int, na2 / 2)
    nx = length(phi)
    source_average = sn_data.source_average
    source_total = sn_data.source_total
    copy!(source_total, phi)
    source_total .*= 0.5
    source_total .*= c
    axpy!(1.0, source, source_total)
    @views copy!(source_average, source_total[2:nx])
    @views source_average .+= source_total[1:nx-1]
    source_average .*= 0.5
    @views forward_angles = angles[na+1:na2]
    @views backward_angles = angles[1:na]
    vfl = (forward_angles / dx) .+ 0.5
    vfl = 1.0 ./ vfl
    vfr = (forward_angles / dx) .- 0.5
    psi .*= 0.0
    @views psi[1:na, nx] .= psi_right
    @views psi[na+1:na2, 1] .= psi_left
    #
    # Forward sweep
    #
    @views for ix = 2:nx
        copy!(psi[na+1:na2, ix], psi[na+1:na2, ix-1])
        psi[na+1:na2, ix] .*= vfr
        psi[na+1:na2, ix] .+= source_average[ix-1]
        psi[na+1:na2, ix] .*= vfl
    end
    #
    # Backward sweep
    #
    @views for ix = nx-1:-1:1
        copy!(psi[1:na, ix], psi[1:na, ix+1])
        psi[1:na, ix] .*= vfr
        psi[1:na, ix] .+= source_average[ix]
        psi[1:na, ix] .*= vfl
    end
    return psi
end



"""
Lap1d(n)

returns -d^2/dx^2 on [0,1] zero BC
"""
function Lap1d(n; beam=false)
    dx = 1 / (n + 1)
    d = 2.0 * ones(n)
    sup = -ones(n - 1)
    D2 = SymTridiagonal(d, sup)
    D2 ./= (dx*dx)
    return D2
end

"""
kstore(n, lsolver)

Preallocates the vectors a Krylov method uses internally.
"""
function kstore(n, lsolver)
    tmp1 = zeros(n)
    tmp2 = zeros(n)
    tmp3 = zeros(n)
    tmp4 = zeros(n)
    if lsolver == "gmres"
        return (tmp1, tmp2, tmp3, tmp4)
    else
        tmp5 = zeros(n)
        tmp6 = zeros(n)
        tmp7 = zeros(n)
        return (tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7)
    end
end

function P5(Nc,omega,tau,thetal,thetar)

    p = 3                # Grid size exponent: nx = 10^p + 1
    nx = (10^p) + 1
    na = 40

    hn_data = heat_init(nx, na, thetal, thetar, omega, tau, Nc)

    x0 = hn_data.bcfix

    return AAProblem((G,u) -> p5_f!(G,u, hn_data),
            x0,
            AAConvParams(1e-10, 0))
end
