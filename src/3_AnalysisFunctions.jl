
"""
    create_midanalysis_function(midanalysis::Vector{Symbol})

Generates a function for intermediate analysis during the iterative process. The function processes the input `NamedTuple` based on the provided `midanalysis` symbols and returns a `NamedTuple` with the computed results.

# Arguments
- `midanalysis`: A vector of symbols representing the fields to be included in the intermediate analysis. Each symbol corresponds to a specific field, such as `:G_k_cond` or `:gamma_k_norm`.

# Returns
A function that takes a `NamedTuple` as input and returns a `NamedTuple` with the calculated intermediate analysis results.

The function dynamically adds fields to the result based on the specified symbols. For each field, a specific calculation is performed, such as computing the norm of `gamma_k` or evaluating the condition number of a matrix.

# Example:
```julia
midanalysis_func = create_midanalysis_function([:G_k_cond, :gamma_k_norm])
result = midanalysis_func(input_data)
```
"""
function create_midanalysis_function(midanalysis::Vector{Symbol})
    return function(input::NamedTuple)
    
        result = NamedTuple()

        # Dynamically add fields based on the symbols in `midanalysis`
        for sym in midanalysis
            if sym == :G_k_cond
                try
                result = merge(result, (G_k_cond = cond(input.G_k),))
                catch e
                    result = merge(result, (G_k_cond = NaN,))
                end
            elseif sym == :G_k_geocond
                try
                result = merge(result, (G_k_geocond = geometriccond(input.G_k),))
                catch e
                    result = merge(result, (G_k_cond = NaN,))
                end
            elseif sym == :gamma_k_norm
                result = merge(result, (gamma_k_norm = norm(input.gamma_k),))
            elseif sym == :alpha_k_norm
                if haskey(input, :gamma_k)
                    alpha_k_norm = norm(gamma_to_alpha(input.gamma_k))            
                end
                if haskey(input, :alpha_k)
                    alpha_k_norm = norm(input.alpha_k)
                end   
                result = merge(result, (alpha_k_norm = alpha_k_norm,))
            elseif sym == :alpha_k_norm_l1
                if haskey(input, :gamma_k)
                    alpha_k_norm_l1 = sum(abs.(gamma_to_alpha(input.gamma_k)))       
                end
                if haskey(input, :alpha_k)
                    alpha_k_norm_l1 = sum(abs.((input.alpha_k)))
                end   
                result = merge(result, (alpha_k_norm_l1 = alpha_k_norm_l1,))
            elseif sym == :truehistlength
                if haskey(input, :deleted)
                    truehistlength = sum(.!input.deleted)
                end
                if haskey(input, :filtered)
                    truehistlength = sum(.!input.filtered)
                end
                result = merge(result, (truehistlength = truehistlength,))
		elseif sym == :filtered
		result = merge(result, (filtered = input.filtered,))
            elseif sym == :G_cond
                result = merge(result, (G_cond = cond(hcat(input.G...)),))
            elseif sym == :G_geocond
                result = merge(result, (G_geocond = geometriccond(hcat(input.G...)),))
            elseif sym == :Gcal_k_cond
                try
                result = merge(result, (Gcal_k_cond = cond(input.Gcal_k),))
                catch e
                    result = merge(result, (Gcal_k_cond = NaN,))
                end
            elseif sym == :Gcal_k_geocond
                try
                result = merge(result, (Gcal_k_geocond = geometriccond(input.Gcal_k),))
                catch e
                    result = merge(result, (Gcal_k_geocond = NaN,))
                end
            elseif sym == :residual
                result = merge(result, (residual = input.residual,))
            elseif sym == :residualnorm
                result = merge(result, (residualnorm = norm(input.residual),))
            elseif sym == :positions
                result = merge(result, (positions = copy(input.positions), ))
            elseif sym == :alpha
                result = merge(result, (alpha = copy(input.alpha),))
            elseif sym == :HisStuf
                result = merge(result, (HisStuf = copy(input.HisStuf),))
            end
        end

        return result
    end
end

"""
    create_liveanalysis_function(liveanalysis::Vector{Symbol})

Generates a function for live analysis during the iterative process. The function processes the input `NamedTuple` based on the provided `liveanalysis` symbols and returns a `NamedTuple` with the computed live analysis results.

# Arguments
- `liveanalysis`: A vector of symbols representing the fields to be included in the live analysis. Each symbol corresponds to a specific field, such as `:residualnorm` or `:alpha_k_norm`.

# Returns
A function that takes a `NamedTuple` as input and returns a `NamedTuple` with the calculated live analysis results.

The function dynamically adds fields to the result based on the specified symbols. For each field, a specific calculation is performed, such as computing the norm of the difference between `x_k` and `x_kp1` or evaluating the condition number of a matrix.

# Example:
```julia
liveanalysis_func = create_liveanalysis_function([:residualnorm, :alpha_k_norm])
result = liveanalysis_func(input_data)
```
"""

function create_liveanalysis_function(liveanalysis::Vector{Symbol})
    return function(input::NamedTuple)
    
        result = NamedTuple()

        # Dynamically add fields based on the symbols in `midanalysis`
        for sym in liveanalysis
            if sym == :residualnorm
                result = merge(result, (residualnorm = norm(input.x_k.-input.x_kp1),))
            elseif sym == :residual_ratio
                result = merge(result, (residual_ratio = norm(input.x_k.-input.x_kp1)/input.residual[end],))
            elseif sym == :alpha_k_norm
                if haskey(input, :gamma_k)
                    alpha_k_norm = norm(gamma_to_alpha(input.gamma_k))            
                end
                if haskey(input, :alpha_k)
                    alpha_k_norm = norm(input.alpha_k)
                end   
                result = merge(result, (alpha_k_norm = alpha_k_norm,))
            elseif sym == :alpha_k_norm_l1
                if haskey(input, :gamma_k)
                    alpha_k_norm_l1 = sum(abs.(gamma_to_alpha(input.gamma_k)))       
                end
                if haskey(input, :alpha_k)
                    alpha_k_norm_l1 = sum(abs.((input.alpha_k)))
                end   
                result = merge(result, (alpha_k_norm_l1 = alpha_k_norm_l1,))
            elseif sym == :truehistlength
                if haskey(input, :deleted)
                    truehistlength = sum(.!input.deleted)
                end
                if haskey(input, :filtered)
                    truehistlength = sum(.!input.filtered)
                end
                if haskey(input, :positions)
                    truehistlength = sum(input.positions .!= -1)
                end
                result = merge(result, (truehistlength = truehistlength,))
            elseif sym == :G_cond
                result = merge(result, (G_cond = cond(hcat(input.G...)),))
            elseif sym == :G_geocond
                result = merge(result, (G_geocond = geometriccond(hcat(input.G...)),))
            elseif sym == :G_k_cond
                    try
                    result = merge(result, (G_k_cond = cond(input.G_k),))
                    catch e
                        result = merge(result, (G_k_cond = NaN,))
                    end
            elseif sym == :G_k_geocond
                    try
                    result = merge(result, (G_k_geocond = geometriccond(input.G_k),))
                    catch e
                        result = merge(result, (G_k_cond = NaN,))
                    end
            elseif sym == :Gcal_k_cond
                try
                result = merge(result, (Gcal_k_cond = cond(input.Gcal_k),))
                catch e
                    result = merge(result, (Gcal_k_cond = NaN,))
                end
            elseif sym == :Gcal_k_geocond
                try
                result = merge(result, (Gcal_k_geocond = geometriccond(input.Gcal_k),))
                catch e
                    result = merge(result, (Gcal_k_geocond = NaN,))
                end
            end
        end

        return result
    end

end

using Printf

AD = Dict(
    :residualnorm => "Res Norm",
    :residual_ratio => "Res Ratio",
    :alpha_k_norm => "Coeff. Norm",
    :alpha_k_norm => "Coeff. Norm L1",
    :truehistlength => "Effective m",
    :G_cond => "G Condition",
    :G_geocond => "G Geometric Condition",
    :G_k_cond => "G Condition",
    :G_k_geocond => "G Geometric Condition",
    :Gcal_k_cond => "Gcal Condition",
    :Gcal_k_geocond => "Gcal Geometric Condition",
    :positions => "Positions",
)

"""
    output_postanalysis(postanalysis::NamedTuple, summary; line_width::Int = 80)

Outputs a formatted post-analysis summary, including method parameters, algorithm parameters, convergence information, and additional fields. The output is printed with line wrapping for better readability.

# Arguments
- `postanalysis`: A `NamedTuple` containing the post-analysis data to be output.
- `summary`: A boolean flag to indicate whether to print a summary (including method, parameters, and iterations).
- `line_width`: The maximum width of lines for wrapping the printed summary (default is 80).

# Example:
```julia
output_postanalysis(postanalysis_data, summary=true)
```

"""
function output_liveanalysis(liveanalysis::NamedTuple, iterations::Int, updatefreq::Int, startwalltime::Float64)
    if updatefreq != 0
        if (iterations % updatefreq == 0) && (iterations > 1)
            # Start the format string and values with iteration count
            log_format = "%04d"
            log_values = []
            push!(log_values,iterations)

            # Dynamically append each field in liveanalysis
            for (field, value) in pairs(liveanalysis)
                log_format *= " | $(AD[field]): %.4g"  # Add field name and placeholder
                push!(log_values, value)
            end

            # Add walltime to the end
            log_format *= " | time: %.2f min"
            push!(log_values, (time() - startwalltime) / 60)

            # Create the final log string
            fmt = Printf.Format(log_format)  # Convert dynamic string to a Printf.Format object
            log = Printf.format(fmt, log_values...)  # Apply the format with the values

            println(log)
        end
    end
end



"""
    output_postanalysis(postanalysis::NamedTuple, summary; line_width::Int = 80)

Outputs a formatted post-analysis summary, including method parameters, algorithm parameters, convergence information, and additional fields. The output is printed with line wrapping for better readability.

# Arguments
- `postanalysis`: A `NamedTuple` containing the post-analysis data to be output.
- `summary`: A boolean flag to indicate whether to print a summary (including method, parameters, and iterations).
- `line_width`: The maximum width of lines for wrapping the printed summary (default is 80).

# Example:
```julia
output_postanalysis(postanalysis_data, summary=true)
```

"""
function output_postanalysis(postanalysis::NamedTuple, summary; line_width::Int = 80)
    if summary
        # Print a line of asterisks
        println("*" ^ line_width)

        # Nicely formatted title for the summary
        summary_text = "Summary: $(SD[postanalysis.methodname]), with parameters: $(postanalysis.methodparams) and $(postanalysis.algorithmparams)."
        
        # Line wrapping: Break the summary into chunks of size `line_width`
        lines = []
        current_line = ""
        for word in split(summary_text, " ")
            if length(current_line) + length(word) + 1 <= line_width
                current_line *= (current_line == "" ? word : " " * word)
            else
                push!(lines, current_line)
                current_line = word
            end
        end
        push!(lines, current_line)  # Add the last line

        # Print each wrapped line
        for line in lines
            println(line)
        end

        # Print Iterations
        println("Iterations: $(lpad(postanalysis.iterations, 4, "0"))")

        # Print Convergence Parameters (atol and rtol, only if non-zero)
        convparams = postanalysis.convparams
        if convparams.atol != 0.0 || convparams.rtol != 0.0
            conv_str = "Convergence Parameters:"
            if convparams.atol != 0.0
                conv_str *= " atol: $(convparams.atol)"
            end
            if convparams.rtol != 0.0
                conv_str *= " rtol: $(convparams.rtol)"
            end

            # Line wrap the convergence parameters
            lines = []
            current_line = ""
            for word in split(conv_str, " ")
                if length(current_line) + length(word) + 1 <= line_width
                    current_line *= (current_line == "" ? word : " " * word)
                else
                    push!(lines, current_line)
                    current_line = word
                end
            end
            push!(lines, current_line)  # Add the last line

            # Print each wrapped line of convergence parameters
            for line in lines
                println(line)
            end
        end

        # Print closing line of asterisks
        println("*" ^ line_width)

        # Iterate through the remaining fields and print them
        for (field, value) in pairs(postanalysis)
            if !(field in [:methodname, :methodparams, :algorithmparams, :convparams, :iterations])
                field_name = get(SD, field, string(field))  # Get a descriptive name or fallback to the symbol
                if value isa Float64
                    println("$(field_name): $(@sprintf("%.4f", value))")
                elseif value isa Int
                    println("$(field_name): $(lpad(value, 4, "0"))")
                end
            end
        end

        # Print closing line of asterisks
        println("*" ^ line_width)
    end
end
