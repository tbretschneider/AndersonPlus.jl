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
end

function verify_qrp_reconstruction(QRP, deleted, A, tol)
    Q, R, P = QRP.Q, QRP.R, QRP.P
    reconstructed = Q * R * P'
    norm_diff = norm(reconstructed - A) / norm(A)
    return norm_diff < tol && sum(deleted) == size(A, 2) - size(QRP.R, 2)
end

using AndersonPlus: paqr_piv!, paqr_piv

# Tests
@testset "PAQR Pivoting Tests" begin
    # Test 1: Simple square matrix
    A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]
    QRP, deleted = paqr_piv!(A; tol=eps(eltype(A)))
    @test size(QRP.R) == (3, 3) # Check R shape
    @test verify_qrp_reconstruction(QRP, deleted, A, tol=1e-12)

    # Test 2: Tall matrix (m > n)
    A = [1.0 2.0; 4.0 5.0; 7.0 8.0]
    QRP, deleted = paqr_piv!(A; tol=eps(eltype(A)))
    @test size(QRP.R) == (3, 2) # Check R shape
    @test verify_qrp_reconstruction(QRP, deleted, A, tol=1e-12)

    # Test 3: Wide matrix (m < n)
    A = [1.0 2.0 3.0; 4.0 5.0 6.0]
    QRP, deleted = paqr_piv!(A; tol=eps(eltype(A)))
    @test size(QRP.R) == (2, 3) # Check R shape
    @test verify_qrp_reconstruction(QRP, deleted, A, tol=1e-12)

    # Test 4: Rank-deficient matrix
    A = [1.0 2.0 3.0; 2.0 4.0 6.0; 3.0 6.0 9.0]
    QRP, deleted = paqr_piv!(A; tol=eps(eltype(A)))
    @test size(QRP.R, 1) == 3
    @test size(QRP.R, 2) == 1 # Only one independent column
    @test sum(deleted) == 2 # Two columns should be deleted

    # Test 5: Matrix with zeros
    A = [0.0 0.0; 0.0 0.0]
    QRP, deleted = paqr_piv!(A; tol=eps(eltype(A)))
    @test size(QRP.R, 2) == 0 # All columns should be rejected
    @test all(deleted) == true

    # Test 6: Single column matrix
    A = [1.0; 2.0; 3.0]
    QRP, deleted = paqr_piv!(A; tol=eps(eltype(A)))
    @test size(QRP.R) == (3, 1)
    @test verify_qrp_reconstruction(QRP, deleted, A, tol=1e-12)

    # Test 7: Tolerance affects rejection
    A = [1.0 2.0; 0.0 1e-15]
    QRP, deleted = paqr_piv!(A; tol=1e-10)
    @test sum(deleted) == 1 # Second column should be rejected
    QRP, deleted = paqr_piv!(A; tol=1e-20)
    @test sum(deleted) == 0 # No column should be rejected

    # Test 8: Non-square, random matrix
    A = rand(5, 3)
    QRP, deleted = paqr_piv!(A; tol=eps(eltype(A)))
    @test size(QRP.R) == (5, 3)
    @test verify_qrp_reconstruction(QRP, deleted, A, tol=1e-12)

    # Test 9: paqr_piv non-mutating version
    A = [1.0 2.0; 3.0 4.0]
    QRP1, deleted1 = paqr_piv!(copy(A); tol=eps(eltype(A)))
    QRP2, deleted2 = paqr_piv(A; tol=eps(eltype(A)))
    @test QRP1.R ≈ QRP2.R
    @test deleted1 == deleted2
    @test A == [1.0 2.0; 3.0 4.0] # Ensure the original matrix is not modified
end
