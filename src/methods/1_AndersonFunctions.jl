using Debugger
using Wavelets

"""
    create_next_iterate_function(GFix!, aamethod::AAMethod, liveanalysisfunc::Function, midanalysisfunc::Function)

Generates an iteration function based on the provided `aamethod` (which can be one of `:vanilla`, `:paqr`, or `:faa`).
The function performs iterative updates for the given `HS` historical data structure and updates the solution vectors `x_k` and `x_kp1`.

# Arguments
- `GFix!`: A function to apply some transformation or fix to the solution vectors `x_k` and `x_kp1`.
- `aamethod`: An `AAMethod` object containing the method type and its parameters.
- `liveanalysisfunc`: A function that takes live analysis data (such as the current iteration, residuals, and solution vectors) and returns updated live analysis results.
- `midanalysisfunc`: A function that processes intermediate analysis data during the iteration.

# Returns
Returns a function that, given the current historical data (`HS`), and the new solution vector `x_kp1` along with the previous solution vector `x_k`, computes the next iterate, performs necessary updates, and returns the mid-analysis and live-analysis results.

The returned function updates `HS` with new residuals, solution histories, and computes intermediate data depending on the specified method in `aamethod`.

### Method Specifics:
1. **For `:vanilla` method**:
   - Updates `x_kp1` based on previous residuals and solution history.
   - Uses ridge regression if necessary for solving the system.
   
2. **For `:paqr` method**:
   - Updates `x_kp1` using the PAQR method and solves using QR decomposition.
   - Performs residual updates and stores relevant data.

3. **For `:faa` method**:
   - Uses filtered historical data (through `LengthFiltering!` and `AngleFiltering!`).
   - Solves using `gamma_k` based on the historical `G_k` and `X_k` data.

# Example:
```julia
next_iterate_func = create_next_iterate_function(GFix!, aamethod, liveanalysisfunc, midanalysisfunc)
midanalysis, liveanalysis = next_iterate_func(HS, x_kp1, x_k)
```
"""
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
                HS.G_k = hcat(g_k - HS.g_km1,HS.G_k)
                if size(HS.G_k,2) > m - 1
                    HS.G_k = HS.G_k[:,1:m-1]
                    HS.X_k = HS.X_k[:,1:m-1]
                end
                filtered = LengthFiltering!(HS, 
                aamethod.methodparams.cs, aamethod.methodparams.kappabar)
                anglefiltered = AngleFiltering!(HS, aamethod.methodparams.cs)
		filtered[.!filtered] .= anglefiltered

                gamma_k = HS.G_k \ g_k
                        
                x_kp1 .= x_k .+ g_k .- (HS.X_k + HS.G_k) * gamma_k 

            elseif HS.iterations == 1
                HS.G_k = hcat(g_k - HS.g_km1,HS.G_k)

                gamma_k = (HS.G_k'*HS.G_k)^(-1)*(HS.G_k'*g_k)

                x_kp1 = x_kp1 - (HS.X_k + HS.G_k)*gamma_k

		filtered = NaN

            elseif HS.iterations == 0
                gamma_k = NaN
		        filtered = NaN
            end
            
            HS.X_k = hcat(x_kp1 - x_k,HS.X_k)
            HS.g_km1 = g_k
            HS.iterations += 1

            midanalysisin = (gamma_k = gamma_k, residual = g_k,filtered = filtered,G_k = HS.G_k)

            liveanalysisin = (X_k = HS.X_k, filtered = filtered, iterations = HS.iterations, x_kp1 = x_kp1, x_k = x_k, residual = g_k)

            midanalysis = midanalysisfunc(midanalysisin)

            liveanalysis = liveanalysisfunc(liveanalysisin)

            return midanalysis, liveanalysis

        end
    elseif aamethod.methodname == :fftaa
        return function(HS::FFTAAHistoricalStuff,x_kp1::Vector{Float64}, x_k::Vector{Float64})

            GFix!(x_kp1,x_k)
            g_k = x_kp1 .- x_k
            m = aamethod.methodparams.m
            tf = aamethod.methodparams.tf

            truncation = Int(ceil(length(x_k)*tf))
            fftg_k = FFTW.fft(g_k)[1:truncation]

            if HS.iterations > 1
                newFFTG_kcol = FFTW.fft(g_k - HS.g_km1)[1:truncation]
                HS.FFTG_k = hcat(newFFTG_kcol,HS.FFTG_k)
                if size(HS.FFTG_k,2) > m - 1
                    HS.FFTG_k = HS.FFTG_k[:,1:m-1]
                    HS.FFTX_k = HS.FFTX_k[:,1:m-1]
                end

                gamma_k = HS.FFTG_k \ fftg_k
                
                compressed = (HS.FFTX_k + HS.FFTG_k) * gamma_k

                # Reconstruct full spectrum before inverse FFT
                reconstructed = zeros(ComplexF64, length(x_k))  # Allocate memory only before inverse FFT
                reconstructed[1:truncation] .= compressed
                reconstructed[end-truncation+2:end] .= conj.(reverse(compressed[2:end]))  # Restore symmetry

                # Compute inverse FFT
                update = real(FFTW.ifft(reconstructed))

                x_kp1 .= x_kp1 - update

            elseif HS.iterations == 1
                newFFTG_kcol = FFTW.fft(g_k - HS.g_km1)[1:truncation]
                HS.FFTG_k = hcat(newFFTG_kcol,HS.FFTG_k)

                gamma_k = (HS.FFTG_k'*HS.FFTG_k)^(-1)*(HS.FFTG_k'*fftg_k)

                compressed = (HS.FFTX_k + HS.FFTG_k) * gamma_k

                # Reconstruct full spectrum before inverse FFT
                reconstructed = zeros(ComplexF64, length(x_k))  # Allocate memory only before inverse FFT
                reconstructed[1:truncation] .= compressed
                reconstructed[end-truncation+2:end] .= conj.(reverse(compressed[2:end]))  # Restore symmetry

                update = real(FFTW.ifft(reconstructed))

                x_kp1 = x_kp1 - update

            elseif HS.iterations == 0
                gamma_k = NaN
            end
            
            newFFTX_kcol = FFTW.fft(x_kp1 - x_k)[1:truncation]
            HS.FFTX_k = hcat(newFFTX_kcol,HS.FFTX_k)
            HS.g_km1 = g_k
            HS.iterations += 1

            midanalysisin = (gamma_k = gamma_k, residual = g_k)

            liveanalysisin = (X_k = HS.FFTX_k, iterations = HS.iterations, x_kp1 = x_kp1, x_k = x_k, residual = g_k)

            midanalysis = midanalysisfunc(midanalysisin)

            liveanalysis = liveanalysisfunc(liveanalysisin)

            return midanalysis, liveanalysis

        end
    elseif aamethod.methodname == :dwtaa
        return function(HS::DWTAAHistoricalStuff,x_kp1::Vector{Float64}, x_k::Vector{Float64})

            GFix!(x_kp1,x_k)
            g_k = x_kp1 .- x_k
	    residualratio = norm(g_k)/HS.residual
		HS.residual = norm(g_k)
            m = aamethod.methodparams.m

            waveletx_kp1, orglength = wavelet_compress(x_kp1,wavelet(WT.db2),1.0)
            HS.DWTF_k = hcat(waveletx_kp1,HS.DWTF_k)

	    waveletg_k,orglength = wavelet_compress(g_k,wavelet(WT.db2),compute_compression_ratio(residualratio,HS.iterations))
            HS.DWTG_k = hcat(waveletg_k,HS.DWTG_k)

            if HS.iterations > 0

                if size(HS.DWTG_k,2) > m
                    HS.DWTG_k = HS.DWTG_k[:,1:m]
                    HS.DWTF_k = HS.DWTF_k[:,1:m]
                end
                
                GT_G = HS.DWTG_k' * HS.DWTG_k
		ones_vec = ones(size(GT_G,1))
                α = GT_G \ ones_vec  # Solves the system without computing the inverse
        
                # Step 2: Normalize α
                alpha_k = α / sum(α)

                waveletx_kp1 .= HS.DWTF_k * alpha_k
                
                x_kp1 .= wavelet_decompress(waveletx_kp1,wavelet(WT.db2),orglength)

            elseif HS.iterations == 0

            end
            
            HS.iterations += 1

            midanalysisin = (residual = g_k, )

            liveanalysisin = (iterations = HS.iterations, x_kp1 = x_kp1, x_k = x_k, residual = g_k)

            midanalysis = midanalysisfunc(midanalysisin)

            liveanalysis = liveanalysisfunc(liveanalysisin)

            return midanalysis, liveanalysis

        end
    elseif aamethod.methodname == :rfaa
        return function(HS::RFAAHistoricalStuff,x_kp1::Vector{Float64}, x_k::Vector{Float64})

            GFix!(x_kp1,x_k)
            g_k = x_kp1 .- x_k
            m = aamethod.methodparams.m

            filtered = RandomFilter!(HS,aamethod.methodparams.probfun)

            HS.F_k = hcat(x_kp1,HS.F_k)

            HS.G_k = hcat(g_k,HS.G_k)

            if HS.iterations > 0

                if size(HS.G_k,2) > m
                    HS.G_k = HS.G_k[:,1:m]
                    HS.F_k = HS.F_k[:,1:m]
                end
                
                GT_G = HS.G_k' * HS.G_k
		        ones_vec = ones(size(GT_G,1))
                α = GT_G \ ones_vec  # Solves the system without computing the inverse
        
                # Step 2: Normalize α
                alpha_k = α / sum(α)
          
                x_kp1 .= HS.F_k * alpha_k

            elseif HS.iterations == 0

            end
            
            HS.iterations += 1

            midanalysisin = (residual = g_k, filtered = filtered, G_k = HS.G_k)

            liveanalysisin = (iterations = HS.iterations, filtered = filtered, x_kp1 = x_kp1, x_k = x_k, residual = g_k, G_k = HS.G_k)

            midanalysis = midanalysisfunc(midanalysisin)

            liveanalysis = liveanalysisfunc(liveanalysisin)

            return midanalysis, liveanalysis

        end
    elseif aamethod.methodname == :rflsaa
        return function(HS::RFLSAAHistoricalStuff,x_kp1::Vector{Float64}, x_k::Vector{Float64})

            GFix!(x_kp1,x_k)
            g_k = x_kp1 .- x_k
            m = aamethod.methodparams.m

            if HS.iterations > 0
                HS.Gcal_k = hcat(g_k - HS.g_km1,HS.Gcal_k)
                if size(HS.Gcal_k,2) > m - 1
                    HS.Gcal_k = HS.Gcal_k[:,1:m-1]
                    HS.Xcal_k = HS.Xcal_k[:,1:m-1]
                end

                filtered = RandomFilterLS!(HS, aamethod.methodparams.probfun)

                gamma_k = HS.Gcal_k \ g_k
                        
                x_kp1 .= x_k .+ g_k .- (HS.Xcal_k + HS.Gcal_k) * gamma_k 

            elseif HS.iterations == 0
                gamma_k = NaN
		        filtered = NaN
            end
            
            HS.Xcal_k = hcat(x_kp1 - x_k,HS.Xcal_k)
            HS.g_km1 = g_k
            HS.iterations += 1

            midanalysisin = (gamma_k = gamma_k, residual = g_k,filtered = filtered,Gcal_k = HS.Gcal_k)

            liveanalysisin = (Gcal_k = HS.Gcal_k, filtered = filtered, iterations = HS.iterations, x_kp1 = x_kp1, x_k = x_k, residual = g_k)

            midanalysis = midanalysisfunc(midanalysisin)

            liveanalysis = liveanalysisfunc(liveanalysisin)

            return midanalysis, liveanalysis

        end
    elseif aamethod.methodname == :quickaa
        return function(HS::quickAAHistoricalStuff,x_kp1::Vector{Float64}, x_k::Vector{Float64})

            GFix!(x_kp1,x_k)
            g_k = x_kp1 .- x_k
            n_kinv = inv(norm(g_k))
            gtilde_k = g_k * n_kinv

            AnglesUpdate!(HS, gtilde_k)

            Filtering!(HS,aamethod.methodparams)

            AddNew!(HS,n_kinv,x_kp1,gtilde_k)

            alpha = HS.Ninv*HS.GtildeTGtildeinv*HS.Ninv*ones(length(HS.sin_k))

            alpha /= sum(alpha)

            x_kp1 = HS.F_k * alpha

            HS.iterations += 1

            midanalysisin = (residual = g_k, GtildeTGtildeinv = HS.GtildeTGtildeinv,positions = HS.positions,alpha = alpha,HS=HS)

            liveanalysisin = (iterations = HS.iterations, x_kp1 = x_kp1, x_k = x_k, residual = g_k,positions = HS.positions)

            midanalysis = midanalysisfunc(midanalysisin)

            liveanalysis = liveanalysisfunc(liveanalysisin)

            return midanalysis, liveanalysis

        end
    else
        error("Unsupported methodname: $(aamethod.methodname)")
    end
end
