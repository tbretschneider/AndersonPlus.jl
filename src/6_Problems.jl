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
    G .= p3_f_helper!(u, k0, ε, N)
end

u_re = cos.(k0 * x)
u_im = sin.(k0 * x)
x_0 = vcat(u_re, u_im); # Concatenate into a single vector
