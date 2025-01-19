function create_midanalysis_function(midanalysis::Vector{Symbol})
    return function(input::NamedTuple)
    
        result = NamedTuple()

        # Dynamically add fields based on the symbols in `midanalysis`
        for sym in midanalysis
            if sym == :G_k_cond
                result = merge(result, (G_k_cond = cond(input.G_k),))
            elseif sym == :gamma_k_norm
                result = merge(result, (gamma_k_norm = norm(input.gamma_k),))
            elseif sym == :residual
                result = merge(result, (residual = input.residual,))
            elseif sym == :residualnorm
                result = merge(result, (residualnorm = norm(input.residual),))
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
            if sym == :residualnorm
                result = merge(result, (residualnorm = norm(input.x_k.-input.x_kp1),))
            elseif sym == :residual_ratio
                result = merge(result, (residual_ratio = norm(input.x_k.-input.x_kp1)/input.residual[end],))
            end
        end

        return result
    end

end

using Printf

const AD = Dict(
    :residualnorm => "Res Norm",
    :residual_ratio => "Res Ratio"
)


function output_liveanalysis(liveanalysis::NamedTuple, iterations::Int, updatefreq::Int, startwalltime::Float64)
    if updatefreq != 0
        if (iterations % updatefreq == 0) && (iterations > 1)
            # Start the format string and values with iteration count
            log_format = "%04d"
            log_values = []
            push!(log_values,iterations)

            # Dynamically append each field in liveanalysis
            for (field, value) in pairs(liveanalysis)
                log_format *= " | $(AD[field]): %.4f"  # Add field name and placeholder
                push!(log_values, value)
            end

            # Add walltime to the end
            log_format *= ", time: %.2f min"
            push!(log_values, (time() - startwalltime) / 60)

            # Create the final log string
            fmt = Printf.Format(log_format)  # Convert dynamic string to a Printf.Format object
            log = Printf.format(fmt, log_values...)  # Apply the format with the values

            println(log)
        end
    end
end

const OUTPUT_DICT = Dict(
    :methodname => "Method Name",
    :methodparams => "Method Parameters",
    :algorithmparams => "Algorithm Parameters",
    :convparams => "Convergence Parameters",
    :iterations => "Iterations"
)

# Function to output the summary
function output_postanalysis(postanalysis::NamedTuple, summary; line_width::Int = 80)
    if summary
        # Print a line of asterisks
        println("*" ^ line_width)

        # Nicely formatted title for the summary
        println("Summary:")
        println("*" ^ line_width)

        # Print method name directly
        println("Method Name: $(postanalysis.methodname)")

        # Print Method Parameters
        methodparams = postanalysis.methodparams
        if !isempty(methodparams)
            println("Method Parameters: $(methodparams)")
        end

        # Print Algorithm Parameters
        algorithmparams = postanalysis.algorithmparams
        if !isempty(algorithmparams)
            println("Algorithm Parameters: $(algorithmparams)")
        end

        # Print Convergence Parameters (atol and rtol, only if non-zero)
        convparams = postanalysis.convparams
        if convparams.atol != 0.0 || convparams.rtol != 0.0
            println("Convergence Parameters:")
            if convparams.atol != 0.0
                println("  atol: $(convparams.atol)")
            end
            if convparams.rtol != 0.0
                println("  rtol: $(convparams.rtol)")
            end
        end

        # Print Iterations
        println("Iterations: $(lpad(postanalysis.iterations, 4, "0"))")

        # Iterate through the remaining fields and print them
        for (field, value) in pairs(postanalysis)
            if !(field in [:methodname, :methodparams, :algorithmparams, :convparams, :iterations])
                field_name = get(OUTPUT_DICT, field, string(field))  # Get a descriptive name or fallback to the symbol
                if value isa Float64
                    println("$(field_name): $(@sprintf("%.4f", value))")
                elseif value isa Int
                    println("$(field_name): $(lpad(value, 4, "0"))")
                elseif value isa AbstractDict || value isa NamedTuple || value isa Vector
                    # For structured data, print a readable summary
                    println("$(field_name): $(string(value))")
                else
                    # Fallback for other types
                    println("$(field_name): $(string(value))")
                end
            end
        end

        # Print closing line of asterisks
        println("*" ^ line_width)
    end
end