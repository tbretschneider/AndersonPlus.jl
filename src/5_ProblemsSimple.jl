# Define the function G(u)
function p1_f!(out::Vector{Float64}, u::Vector{Float64})
    out[1] = cos((u[1] + u[2]) / 2)
    out[2] = cos((u[1] + u[2]) / 2) + 1e-8 * sin(u[1]^2)
    return nothing
end

function p1_f(u::Vector{Float64})
    out = copy(u)
    out[1] = cos((u[1] + u[2]) / 2)
    out[2] = cos((u[1] + u[2]) / 2) + 1e-8 * sin(u[1]^2)
    return out
end

# Problem parameters
n = 2  # Dimension of the problem
u_sol = [0.0, 0.0]  # Solution is assumed for reference (if known)
u_start = [1.0, 1.0]  # Initial iterate

# Dictionary to store problem-specific details
p1_dict = Dict(
    "n" => n,
    "start" => u_start,
    "sol" => u_sol,
    "title" => "G(u) from Table 3.1 of the problem statement"
)

########################################################################################

function p2_f!(H_next, H; ω=0.5)
    N = length(H)
    Δ = 1 / N

    # Calculate the integral part using composite midpoint rule
    integral_term = zeros(N)
    for i = 1:N
        μ_i = (i-1/2)/N
        integral_term[i] = sum([μ_i/(μ_i+(j-1/2)/N)*H[j]*Δ for j in 1:N])
    end

    # Compute the next iteration of H using the fixed point iteration formula
    H_next .= (ones(N).-ω/2*integral_term).^-1

    return nothing
end
