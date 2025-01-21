using AndersonPlus: ridge_regression, gamma_to_alpha

@testset "Ridge Regression and Gamma to Alpha Tests" begin
    @testset "ridge_regression" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [7.0, 8.0, 9.0]
        λ = 0.1
        expected_result = [-4.11960133,  5.01359106] # Precomputed expected values
        result = ridge_regression(A, b, λ)
        @test isapprox(result, expected_result, atol=1e-6)
    end

    @testset "gamma_to_alpha" begin
        gamma = [0.2, 0.5, 0.7]
        expected_alpha = [0.2, 0.3, 0.2, 0.3]
        result = gamma_to_alpha(gamma)
        @test result ≈ expected_alpha atol=1e-8
    end
end

using AndersonPlus: checktolerances


@testset "checktolerances tests" begin
    # Define test inputs
    tolparams1 = AAConvParams(1e-5, 1e-3)
    tolparams2 = AAConvParams(0.0, 0.0)
    tolparams3 = AAConvParams(1e-5, 0.0)

    x1 = [1.0, 2.0]
    y1 = [1.0, 2.000001]  # Within atol
    x2 = [1.0, 2.0]
    y2 = [1.1, 2.1]       # Exceeds atol and rtol
    x3 = [1.0, 2.0]
    y3 = [1.0, 2.0]       # Exact match
    x4 = [0.0, 0.0]
    y4 = [1.0e-6, 1.0e-6] # Within rtol only

    # Tests
    @test checktolerances(x1, y1, tolparams1) == true
    @test checktolerances(x2, y2, tolparams1) == false
    @test checktolerances(x3, y3, tolparams1) == true
    @test checktolerances(x4, y4, tolparams1) == true
    @test checktolerances(x1, y1, tolparams2) == false  # Both tolerances are zero
    @test checktolerances(x1, y1, tolparams3) == true   # Only atol is non-zero
end

using AndersonPlus: geometriccond

@testset "Geometric Cond Tests" begin
    # Test 1: Basic functionality with a 2x2 matrix
    A = [1.0 2.0; 3.0 4.0]
    col_norms = norm.(eachcol(A))
    A_normalized = A ./ reshape(col_norms, (1, size(A, 2)))
    @test geometriccond(A) ≈ cond(A_normalized)
    
    # Test 2: Identity matrix (condition number should be 1)
    I = Matrix(I, 3, 3)
    @test geometriccond(I) ≈ 1.0
    
    # Test 3: Singular matrix
    B = [1.0 2.0; 2.0 4.0]  # Rank-deficient
    @test geometriccond(B) ≈ Inf
    
    # Test 6: Single-column matrix
    E = [1.0; 3.0]
    col_norms = norm.(eachcol(E))
    E_normalized = E ./ reshape(col_norms, (1, size(E, 2)))
    @test geometriccond(E) ≈ cond(E_normalized)
    
    # Test 7: Large random matrix
    F = rand(10, 20)
    col_norms = norm.(eachcol(F))
    F_normalized = F ./ reshape(col_norms, (1, size(F, 2)))
    @test geometriccond(F) ≈ cond(F_normalized)
end
