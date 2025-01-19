using AndersonPlus

@testset "Vanilla Toth Kelly 2015" begin
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
    @test isapprox(Output.analysis.output.alpha_k_norm_l1[1:7],coefficient_norm[1:7],rtol=5e-2)
end
