function create_next_iterate_function(GFix!, aamethod::AAMethod, liveanalysisfunc::Function, midanalysisfunc::Function)
    if aamethod.methodname == :vanilla
        # Define the function for the :vanilla method
        return function(HS::VanillaHistoricalStuff, x_kp1::Vector{Float64}, x_k::Vector{Float64})
            # Update `x_kp1` and `x_k` using GFix!
            GFix!(x_kp1, x_k)

            m = aamethod.methodparams.m

            GFix!(x_kp1,x_k)
            g_k = x_kp1 .- x_k
            push!(HS.residual, copy(g_k))
            if length(HS.residual) > (m + 1)
                popfirst!(HS.residual)
            end

            @bp

            if HS.iterations > 1
                G_k = hcat([HS.residual[i] .- HS.residual[i-1] for i in
                2:length(HS.residual)]...)
                X_k = hcat([HS.solhist[i] .- HS.solhist[i-1] for i in 2:length(HS.solhist)]...)
                try
                    gamma_k = G_k \ HS.residual[end]
		    println("noridgeregression")
                catch e
			println("ridgeregressionused")
                    gamma_k = ridge_regression(G_k, HS.residual[end])
                end
                x_kp1 = x_k .+ g_k .- (X_k + G_k) * gamma_k

            else 
                G_k = nothing
                gamma_k = nothing
                X_k = nothing
                push!(HS.solhist,x_k)
            end

            push!(HS.solhist,x_kp1)

            if length(HS.solhist) > (m + 1)
                popfirst!(HS.solhist)
            end

            HS.iterations += 1

            midanalysisin = (G_k = G_k, gamma_k = gamma_k, X_k = X_k,residual = HS.residual[end])

            liveanalysisin = (iterations = HS.iterations, x_kp1 = x_kp1, x_k = HS.solhist[end-1],residual = HS.residual)

            midanalysis = midanalysisfunc(midanalysisin)

            liveanalysis = liveanalysisfunc(liveanalysisin)

            return midanalysis, liveanalysis
        end
    else
        error("Unsupported methodname: $(aamethod.methodname)")
    end
end
