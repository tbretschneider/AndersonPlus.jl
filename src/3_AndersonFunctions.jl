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
            if (HS.iterations > 0)
                
                QRP,deleted = paqr_piv!(hcat(HS.G...);tol = tolerance)

		@bp
                
                deleteat!(HS.G,deleted)
                deleteat!(HS.F,deleted)
                
                R = QRP.R
                
                RT_R = R' * R
                ones_vec = ones(size(QRP.R, 1))
                α = RT_R \ ones_vec  # Solves the system without computing the inverse
        
                # Step 2: Normalize α
                alpha_k = α / sum(α)
                
                x_kp1 .= hcat(HS.F...) * alpha_k
            
            else
                deleted = falses(1)
                alpha_k = [1.0]
                push!(HS.solhist,copy(x_k))
            end
                
            # Now updating the x_k and the x_kp1
            push!(HS.solhist,copy(x_kp1))

            HS.iterations += 1

            midanalysisin = (residual = HS.residual[end],G = HS.G,deleted = deleted,alpha_k = alpha_k)

            liveanalysisin = (iterations = HS.iterations, x_kp1 = x_kp1, x_k = HS.solhist[end-1],residual = HS.residual,G = HS.G,deleted = deleted,alpha_k= alpha_k)

            midanalysis = midanalysisfunc(midanalysisin)

            liveanalysis = liveanalysisfunc(liveanalysisin)
        
            #Updates Go Here - ridge doesn't really make sense for this. Need to simplify code to get rid of it!
            return (midanalysis,liveanalysis)
        end
    elseif aamethod.methodname == :faa
        return function(HS::FAAHistoricalStuff,x_kp1::Vector{Float64}, x_k::Vector{Float64})

            GFix!(x_kp1,x_k)
            g_k = x_kp1 .- x_k
            m = aamethod.methodparams.m

            if HS.iterations > 1
                HS.G_k .= hcat(g_k - HS.g_km1,HS.G_k)

                if size(HS.G_k,2) > m - 1
                    HS.G_k .= HS.G_k[:,1:m-1]
                    HS.X_k .= HS.X_k[:,1:m-1]
                end

                LengthFiltering!(HS.G_k, HS.X_k, 
                aamethod.methodparams.cs, aamethod.methodparams.kappabar)
                AngleFiltering!(HS.G_k, HS.X_k, aamethod.methodparams.cs)

                gamma_k = HS.G_k \ g_k
                        
                x_kp1 .= x_k .+ g_k .- (X_k + G_k) * gamma_k 

            elseif HS.iterations == 1
                HS.G_k .= hcat(g_k - HS.g_km1,HS.G_k)

                gamma_k = (HS.G_k'*HS.G_k)^(-1)*(HS.G_k'*g_k)

                x_kp1 .= x_kp1 - (HS.X_k + HS.G_k)*gamma_k

            elseif HS.iterations == 0

            end
            
            HS.X_k .= hcat(x_kp1 - x_k,HS.X_k)
            HS.g_km1 .= g_k

            midanalysisin = (gamma_k = gamma_k, residual = g_k)

            liveanalysisin = (iterations = HS.iterations, x_kp1 = x_kp1, x_k = x_k, residual = g_k)

            midanalysis = midanalysisfunc(midanalysisin)

            liveanalysis = liveanalysisfunc(liveanalysisin)

            return midanalysis, liveanalysis

        end
    

    else
        error("Unsupported methodname: $(aamethod.methodname)")
    end
end
