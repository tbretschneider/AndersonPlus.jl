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


function p2_f_helper_generate_samples(;μ = [0.0, 2.0, 4.0])
    α = [0.3, 0.3, 0.4]
    σ = [1.0, 1.0, 1.0]
    
    N = 100000
    samples = similar(Vector{Float64}, N)
    
    for k = 1:N
        # Choose a component i based on mixture proportions α
        i = wsample(1:3, weights(α))
        # Generate a sample from the selected normal distribution
        samples[k] = μ[i] + σ[i] * randn()
    end
    
    return samples
end

p2_f_helper_samples = p2_f_helper_generate_samples()

# Function to perform one iteration of the EM algorithm
function p2_f!(G, u; samples = p2_f_helper_samples)
    α = [0.3, 0.3, 0.4]
    σ = [1.0, 1.0, 1.0]
    
    num_components = length(u)
    N = length(samples)
    
    μ_plus = zeros(num_components)
    
    for i = 1:num_components
        sum_numer = 0.0
        sum_denom = 0.0
        
        for k = 1:N
            xk = samples[k]
            pi_xk = (1 / sqrt(2 * π * σ[i])) * exp(-(xk - u[i])^2 / (2 * σ[i]^2))
            p_xk = 0.0
            for j = 1:num_components
                   pj_xk = (1 / sqrt(2 * π * σ[j])) * exp(-(xk - u[j])^2 / (2 * σ[j]^2))
                   p_xk += α[j] * pj_xk
            end
            sum_numer += α[i] * pi_xk * xk / p_xk
            sum_denom += α[i] * pi_xk / p_xk
        end
        
        μ_plus[i] = sum_numer / sum_denom
    end
    
    G .= μ_plus
end

function p3_f!(H_next, H; ω=0.5)
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