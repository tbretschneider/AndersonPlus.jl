"""
    p1_f!(out::Vector{Float64}, u::Vector{Float64})

This function computes the function `G(u)` for a given input vector `u` and stores the results in the output vector `out`. The function applies the following operations:

- `out[1]` is the cosine of the average of `u[1]` and `u[2]`.
- `out[2]` is a modified version of `out[1]` with a small perturbation from `sin(u[1]^2)`.

# Arguments
- `out`: A vector to store the computed results.
- `u`: A vector containing the input values for the function.

# Example:
```julia
p1_f!(out, [1.0, 1.0])
```
"""
function p1_f!(out::Vector{Float64}, u::Vector{Float64})
    out[1] = cos((u[1] + u[2]) / 2)
    out[2] = cos((u[1] + u[2]) / 2) + 1e-8 * sin(u[1]^2)
    return nothing
end

"""
    p1_f(u::Vector{Float64})

This function computes the function `G(u)` for a given input vector `u` and returns a new vector with the computed results. The function applies the following operations:

- `out[1]` is the cosine of the average of `u[1]` and `u[2]`.
- `out[2]` is a modified version of `out[1]` with a small perturbation from `sin(u[1]^2)`.

# Arguments
- `u`: A vector containing the input values for the function.

# Returns
A new vector containing the computed values.

# Example:
```julia
out = p1_f([1.0, 1.0])
```
"""
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

"""
    p2_f!(H_next, H; ω=0.5)

This function performs a fixed-point iteration to compute the next iteration of the function `H` based on the current value of `H`. It uses the composite midpoint rule to calculate an integral term, which is then used to update `H_next`.

- The integral term is calculated by summing the contributions from each grid point using the midpoint rule.
- The next iteration of `H` is computed using the fixed-point iteration formula with a relaxation parameter `ω`.

# Arguments
- `H_next`: A vector to store the computed next iteration of `H`.
- `H`: A vector containing the current values of `H`.
- `ω`: The relaxation parameter, default is 0.5.

# Example:
```julia
p2_f!(H_next, H, ω=0.5)
```
"""
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
