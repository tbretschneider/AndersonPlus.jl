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
