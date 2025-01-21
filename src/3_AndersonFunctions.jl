using Debugger

function create_next_iterate_function(GFix!, aamethod::AAMethod, liveanalysisfunc::Function, midanalysisfunc::Function)
    if aamethod.methodname == :vanilla
        # Define the function for the :vanilla method
        return function(HS::VanillaHistoricalStuff, x_kp1::Vector{Float64}, x_k::Vector{Float64})
            # Update `x_kp1` and `x_k` using GFix!
            m = aamethod.methodparams.m

            GFix!(x_kp1,x_k)
            g_k = x_kp1 .- x_k
            push!(HS.residual, copy(g_k))
            if length(HS.residual) > (m + 1)
                popfirst!(HS.residual)
            end

            @bp

            if HS.iterations > 0
                G_k = hcat([HS.residual[i] .- HS.residual[i-1] for i in 2:length(HS.residual)]...)
                X_k = hcat([HS.solhist[i] .- HS.solhist[i-1] for i in 2:length(HS.solhist)]...)

                @bp

                try
                    gamma_k = G_k \ HS.residual[end]
                catch e
                    gamma_k = ridge_regression(G_k, HS.residual[end])
                end

                x_kp1 .= x_k .+ g_k .- (X_k + G_k) * gamma_k

            else 
                G_k = NaN
                gamma_k = NaN
                X_k = NaN
                push!(HS.solhist,copy(x_k))
            end

            push!(HS.solhist,copy(x_kp1))

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
    elseif aamethod.methodname == :paqr
        return function(HS::PAQRHistoricalStuff,x_kp1::Vector{Float64}, x_k::Vector{Float64})  
                #Selecting parameters...
            tolerance = aamethod.methodparams.threshold
            
                GFix!(x_kp1,x_k)
                g_k = x_kp1 .- x_k
                push!(HS.residual, copy(g_k))
                insert!(HS.G,1,copy(g_k))
                insert!(HS.F,1,copy(x_kp1))
                    
            #For larger iterations.
            if (iterations > 0)
                
                QRP,deleted = paqr_piv!(hcat(HS.G...);tol = tolerance)
                
                deleteat!(HS.G,deleted)
                deleteat!(HS.F,deleted)
                
                R = QRP.R
                
                RT_R = R' * R
                ones_vec = ones(size(QRP.R, 1))
                α = RT_R \ ones_vec  # Solves the system without computing the inverse
        
                # Step 2: Normalize α
                α_normalized = α / sum(α)
                
                x_kp1 .= hcat(HS.F...)*α_normalized
            
            else
                deleted = falses(1)
                alpha_k = [1.0]
            end
                
            # Now updating the x_k and the x_kp1
            x_k .= copy(x_kp1)
            push!(HS.solhist,x_k)

            HS.iterations += 1

            midanalysisin = (residual = HS.residual[end],G = HS.G,deleted = deleted,alpha_k = α_normalized)

            liveanalysisin = (iterations = HS.iterations, x_kp1 = x_kp1, x_k = HS.solhist[end-1],residual = HS.residual,G = HS.G,deleted = deleted,alpha_k= α_normalized)

            midanalysis = midanalysisfunc(midanalysisin)

            liveanalysis = liveanalysisfunc(liveanalysisin)
        
            #Updates Go Here - ridge doesn't really make sense for this. Need to simplify code to get rid of it!
            return (midanalysis,liveanalysis)
        end
    else
        error("Unsupported methodname: $(aamethod.methodname)")
    end
end
