function create_next_iterate_function(GFix!, aamethod::AAMethod, liveanalysisfunc::Function, midanalysisfunc::Function)
    if aamethod.methodname == :vanilla
        # Define the function for the :vanilla method
        return function(historicalstuff::VanillaHistoricalStuff, x_kp1::Vector{Float64}, x_k::Vector{Float64})
            # Update `x_kp1` and `x_k` using GFix!
            GFix!(x_kp1, x_k)

            m = aamethod.methodparams.m

            residual = historicalstuff.residual
            solhist = historicalstuff.solhist
            iterations = historicalstuff.iterations

            GFix!(x_kp1,x_k)
            g_k = x_kp1 .- x_k
            push!(residual, copy(g_k))
            if length(residual) > m
                popfirst!(residual)
            end

            if iterations > 1
                G_k = hcat([residual[i] .- residual[i-1] for i in
                2:length(residual)]...)
                X_k = hcat([solhist[i] .- solhist[i-1] for i in 2:length(solhist)]...)
                try
                    gamma_k = G_k \ residual[end]
		    println("noridgeregression")
                catch e
			println("ridgeregressionused")
                    gamma_k = ridge_regression(G_k, residual[end])
                end
                x_kp1 = x_k .+ g_k .- (X_k + G_k) * gamma_k

            else 
                G_k = nothing
                gamma_k = nothing
                X_k = nothing
                push!(solhist,x_k)
            end

            push!(solhist,x_kp1)

            if length(solhist) > m
                popfirst!(solhist)
            end

            iterations += 1

            midanalysisin = (G_k = G_k, gamma_k = gamma_k, X_k = X_k,residual = residual[end])

            liveanalysisin = (iterations = iterations, x_kp1 = x_kp1, x_k = solhist[end-1],residual = residual)

            midanalysis = midanalysisfunc(midanalysisin)

            liveanalysis = liveanalysisfunc(liveanalysisin)

            return midanalysis, liveanalysis
        end
    else
        error("Unsupported methodname: $(aamethod.methodname)")
    end
end
