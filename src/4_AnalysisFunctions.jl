function create_midanalysis_function(midanalysis::Vector{Symbol})
    return function(input::NamedTuple)
    
        result = NamedTuple()

        # Dynamically add fields based on the symbols in `midanalysis`
        for sym in midanalysis
            if sym == :G_k_cond
                result = merge(result, (G_k_cond = cond(input.G_k),))
            elseif sym == :gamma_k_norm
                result = merge(result, (gamma_k_norm = norm(input.gamma_k),))
            end
        end

        return result
    end
end

function create_liveanalysis_function(liveanalysis::Vector{Symbol})
    return function(input::NamedTuple)
    
        result = NamedTuple()

        # Dynamically add fields based on the symbols in `midanalysis`
        for sym in liveanalysis
            if sym == :residual
                result = merge(result, (residual = norm(input.x_k.-input.x_kp1),))
            elseif sym == :residual_ratio
                result = merge(result, (residual_ratio = norm(input.x_k.-input.x_kp1)/input.residual[end],))
            end
        end

        return result
    end

end

using Printf

function output_liveanalysis(liveanalysis::NamedTuple, iterations::Int, updatefreq::Int, startwalltime::Float64)
    if (iterations % updatefreq == 0) && (iterations > 2)
        # Start the format string and values with iteration count
        log_format = "Iteration: %04d"
        log_values = [iterations]

        # Dynamically append each field in liveanalysis
        for (field, value) in pairs(liveanalysis)
            log_format *= @sprintf(", %s: %.4f", string(field))  # Format as a floating point
            push!(log_values, value)
        end

        # Add walltime to the end
        log_format *= ", walltime: %.2f min"
        push!(log_values, (time() - startwalltime) / 60)

        # Generate the final log string
        log = @sprintf(log_format, log_values...)
        println(log)
    end
end
