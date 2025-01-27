using AndersonPlus

@testset "Vanilla Toth Kelly 2015 Vanilla" begin
    residual_norm = [6.501e-01, 4.487e-01, 2.615e-02, 7.254e-02, 1.531e-04, 1.185e-05, 1.825e-08, 1.048e-13]
    condition_number = [1.0, 1.000e+00, 2.016e+10, 1.378e+09, 3.613e+10, 2.549e+11, 3.677e+10, 1.574e+11]  # NaN for missing value at k=0
    coefficient_norm = [NaN, 1.000e+00, 4.617e+00, 2.157e+00, 1.184e+00, 1.000e+00, 1.002e+00, 1.092e+00]  # NaN for missing value at k=0

    x0 = [1.0,1.0]

    Problem = AAProblem(p1_f!,
                        x0,
                        AAConvParams(1e-10, 0))
    
    Algorithm = AAAlgorithm(AAMethod(:vanilla,(m=2, )),
                            (maxit = 20, ))

    Analyses = AAAnalysis([],
                        [:residualnorm,:G_k_cond,:alpha_k_norm_l1],
                        0,false)
    
    Output = AASolve(AAInput(Problem,Algorithm,Analyses))

    @test isapprox(Output.analysis.output.residualnorm,residual_norm,rtol=5e-3)
    @test isapprox(Output.analysis.output.G_k_cond[1:7],condition_number[1:7],rtol=5e-3)
    @test isapprox(Output.analysis.output.alpha_k_norm_l1[2:7],coefficient_norm[2:7],atol=5e-3)
end

@testset "PAQR Analysis" begin
    x0 = [1.0,1.0]

    Problem = AAProblem(p1_f!,
                        x0,
                        AAConvParams(1e-10, 0))
    
    Algorithm = AAAlgorithm(AAMethod(:paqr,(threshold=1e-5, )),
                            (maxit = 20, ))

    Analyses = AAAnalysis([:truehistlength,:G_geocond,:residual_norm],
                        [:residualnorm,:G_cond,:alpha_k_norm_l1,:G_geocond],
                        0,false)
                    
    Output = AASolve(AAInput(Problem,Algorithm,Analyses))
end

@testset "FAA Analysis" begin
    x0 = ones(500)

    Problem = AAProblem(p2_f!,
                        x0,
                        AAConvParams(1e-10, 0))
    
    Algorithm = AAAlgorithm(AAMethod(:faa,(m=10,cs = 0.1, kappabar = 1 )),
                            (maxit = 20, ))

    Analyses = AAAnalysis([:residualnorm],
                        [:residualnorm],
                        0,false)
                    
    Output = AASolve(AAInput(Problem,Algorithm,Analyses))
end

@testset "FAA Analysis Test 3" begin
    # Parameters
    k0 = 8.0
    N = 2000
    x_start = 0.0
    x_end = 10.0
    x = range(x_start, x_end, length=N+1)

    u_re = cos.(k0 * x)
    u_im = sin.(k0 * x)
    x0 = vcat(u_re, u_im); # Concatenate into a single vector

    Problem = AAProblem(p3_f!,
                        x0,
                        AAConvParams(1e-10, 0))
    
    Algorithm = AAAlgorithm(AAMethod(:faa,(m=10,cs = 0.01, kappabar = 1 )),
                            (maxit = 20, ))

    Analyses = AAAnalysis([:residualnorm],
                        [:residualnorm],
                        0,false)
                    
    Output = AASolve(AAInput(Problem,Algorithm,Analyses))
end


@testset "GPE Equation" begin

    Problem = P4("GPE32.msh")
    
    Algorithm = AAAlgorithm(AAMethod(:vanilla,(m=2, )),
                            (maxit = 5, ))

    Analyses = AAAnalysis([],
                        [:residualnorm,:G_k_cond,:alpha_k_norm_l1],
                        0,false)
    
    Output = AASolve(AAInput(Problem,Algorithm,Analyses))
end



@testset "CR_Heat Equation" begin
    # Harder problem parameters
    Nc = 0.05
    omega = 0.5          # Scattering parameter
    tau = 2.0            # Absorption/emission parameter
    thetal = 1.0         # Left boundary temperature
    thetar = 1.8         # Right boundary temperature
    # Initialize the heat transfer problem
    Problem = P5(Nc,omega,tau,thetal,thetar)
    
    Algorithm = AAAlgorithm(AAMethod(:vanilla,(m=10, )),
                            (maxit = 20, ))

    Analyses = AAAnalysis([],
                        [:residualnorm,:G_k_cond,:alpha_k_norm_l1],
                        0,false)
    
    Output = AASolve(AAInput(Problem,Algorithm,Analyses))
end
   


@testset "H-Equation" begin
    n = 128
    c = 0.99

    Problem = P6(n,c)
    
    Algorithm = AAAlgorithm(AAMethod(:vanilla,(m=2, )),
                            (maxit = 10, ))

    Analyses = AAAnalysis([],
                        [:residualnorm,:G_k_cond,:alpha_k_norm_l1],
                        0,false)
    
    Output = AASolve(AAInput(Problem,Algorithm,Analyses))
end
