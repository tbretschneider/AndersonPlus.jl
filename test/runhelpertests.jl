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
using LinearAlgebra

@testset "Geometric Cond Tests" begin
    # Test 1: Basic functionality with a 2x2 matrix
    A = [1.0 2.0; 3.0 4.0]
    col_norms = norm.(eachcol(A))
    A_normalized = A ./ reshape(col_norms, (1, size(A, 2)))
    @test geometriccond(A) ≈ cond(A_normalized)
    
    # Test 2: Identity matrix (condition number should be 1)
    Iden = I(3)
    @test geometriccond(Iden) ≈ 1.0
    
    # Test 7: Large random matrix
    F = rand(10, 20)
    col_norms = norm.(eachcol(F))
    F_normalized = F ./ reshape(col_norms, (1, size(F, 2)))
    @test geometriccond(F) ≈ cond(F_normalized)
end

using AndersonPlus: nl_reflector!

@testset "nl_reflector! Tests" begin
    # Test 1: Zero-length vector
    x = Float64[]
    result = nl_reflector!(x)
    @test result == zero(Float64)
    @test length(x) == 0  # Ensure the vector remains empty

    # Test 3: Multi-element vector with no zeros
    x = [3.0, 4.0, 5.0]
    original_norm = norm(x)
    result = nl_reflector!(x)
    @test norm(x) ≈ original_norm  # Norm should be preserved
    @test x[1] == -result  # Ensure the first element matches the result
    
    # Test 4: Multi-element vector with zeros
    x = [0.0, 3.0, 4.0]
    original_norm = norm(x)
    result = nl_reflector!(x)
    @test norm(x) ≈ original_norm  # Norm should still be preserved

    # Test 5: Vector with very small values
    x = [1e-300, 2e-300, 3e-300]
    original_norm = norm(x)
    result = nl_reflector!(x)
    @test norm(x) ≈ original_norm  # Norm should be preserved despite small values

    # Test 6: Vector with very large values
    x = [1e300, 2e300, 3e300]
    original_norm = norm(x)
    result = nl_reflector!(x)
    @test norm(x) ≈ original_norm  # Norm should be preserved

    # Test 7: Complex-valued vector
    x = ComplexF64[1.0 + 2.0im, 3.0 - 4.0im, 5.0 + 6.0im]
    original_norm = norm(x)
    result = nl_reflector!(x)
    @test norm(x) ≈ original_norm  # Norm should be preserved for complex vectors

end
