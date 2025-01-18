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


function output_liveanalysis(liveanalysis::NamedTuple, iterations::Int, updatefreq::Int, startwalltime::Float64)
    if updatefreq != 0
        if (iterations % updatefreq == 0) && (iterations > 2)
            # Start the format string and values with iteration count
            log_format = "Iteration: %04d"
            log_values = []
            push!(log_values,iterations)

            # Dynamically append each field in liveanalysis
            for (field, value) in pairs(liveanalysis)
                log_format *= ", $(string(field)): %.4f"  # Add field name and placeholder
                push!(log_values, value)
            end

            # Add walltime to the end
            log_format *= ", walltime: %.2f min"
            push!(log_values, (time() - startwalltime) / 60)

            # Create the final log string
            fmt = Printf.Format(log_format)  # Convert dynamic string to a Printf.Format object
            log = Printf.format(fmt, log_values...)  # Apply the format with the values

            println(log)
        end
    end
end

function output_postanalysis(postanalysis::NamedTuple,summary; line_width::Int = 80)
    if summary
        # Start the summary with a general description
        summary = "Summary: "

        # Iterate through the fields of the NamedTuple
        for (field, value) in pairs(postanalysis)
		if value isa Float64
            # Append each field's name and value to the summary
            summary *= @sprintf("%s is %.4f, ", string(field), value)
	    	end
		if value isa Int
            # Append each field's name and value to the summary
            summary *= @sprintf("%s is %04d, ", string(field), value)
	    	end
        end

        # Remove the trailing comma and space, and add a period

        # Line wrapping: Break the summary into chunks of size `line_width`
        lines = []
        current_line = ""
        for word in split(summary, " ")
            if length(current_line) + length(word) + 1 <= line_width
                current_line *= (current_line == "" ? word : " " * word)
            else
                push!(lines, current_line)
                current_line = word
            end
        end
        push!(lines, current_line)  # Add the last line

        # Print each line
        for line in lines
            println(line)
        end
    end
end

