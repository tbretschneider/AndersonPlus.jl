using AndersonPlus: ridge_regression, gamma_to_alpha

Lots = false

if Lots

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

using AndersonPlus: removeinverse!, Random

@testset "removeinverse! Tests" begin
    Random.seed!(42)  # For reproducibility
    n = 10  # Matrix size
    A = randn(n, n)
    A = Symmetric(A'A)  # Ensure A is symmetric positive definite
    A_inv = inv(A)  # Compute its inverse
	for index in 1:10
        A_inv_copy = copy(A_inv)  # Ensure in-place updates don't affect other tests
        removeinverse!(A_inv_copy,index)

        # Compute expected result by removing row/column and inverting
	A_inv_exact = inv(A[setdiff(1:n,index),setdiff(1:n,index)])

        # Check correctness
        @test isapprox(A_inv_copy[setdiff(1:n, index), setdiff(1:n, index)], A_inv_exact)
        # Ensure symmetry is preserved
        @test A_inv_copy ≈ A_inv_copy'
	end


    # Testing if we can just work with big matrix????? Would be very cool!
	
    Random.seed!(42)  # For reproducibility
    n = 10  # Matrix size
    A = randn(n, n)
    A = Symmetric(A'A)  # Ensure A is symmetric positive definite

    A_inv = copy(A)

    A_inv.data[setdiff(1:10,3),setdiff(1:10,3)] = inv(@view A[setdiff(1:10,3),setdiff(1:10,3)])

    A_inv.data[3,1:10] .= 0.0
    
    A_inv.data[1:10,3] .= 0.0

    index = 5
    A_inv_copy = copy(A_inv)  # Ensure in-place updates don't affect other tests
    removeinverse!(A_inv_copy,index)

        # Compute expected result by removing row/column and inverting
	A_inv_exact = inv(A[setdiff(1:n,[index,3]),setdiff(1:n,[index,3])])

        # Check correctness
	@test isapprox(A_inv_copy[setdiff(1:n, [index,3]), setdiff(1:n, [index,3])], A_inv_exact)
        # Ensure symmetry is preserved
	@test isapprox((A_inv_copy * A)[setdiff(1:n,[index,3]),setdiff(1:n,[index,3])],I(8))


end

using AndersonPlus: addinverse!

@testset "Test addinverse!" begin
    Random.seed!(35)  # For reproducibility
    n = 10  # Matrix size
    History = randn(30, n)

    for i in 1:size(History, 2)
        History[:, i] /= norm(History[:, i])  # Normalize column i
    end

    XTX = Symmetric(History'History)  # Ensure A is symmetric positive definite

    XTX_copy = copy(XTX)


    for index in 1:10
        XTXloop = copy(XTX)

        #First make small...

        XTXloopinv = Symmetric(inv(XTXloop))  # Ensure in-place updates don't affect other tests
        removeinverse!(XTXloopinv,index)

        newidentity = I(10)
        newidentity[index,index] = 0

        # Check we have small inverse
	@test isapprox(XTXloopinv,XTXloopinv')
	@test isapprox((XTXloopinv*XTX)[setdiff(1:n,index),setdiff(1:n,index)],I(10)[setdiff(1:n,index),setdiff(1:n,index)];atol = 1e-8)
        @test isapprox((XTXloopinv[setdiff(1:n,index),setdiff(1:n,index)]*XTX[setdiff(1:n,index),setdiff(1:n,index)]),I(9))

        #Now add back in the column and hopefully we get back to what we had...

        addinverse!(XTXloopinv,index,XTXloop[index,1:n])



        @test isapprox((XTXloopinv*XTX),I(10))

    end

end


#Remove multiple then add them back in...


@testset "Testing removing then adding back..." begin
    Random.seed!(35)  # For reproducibility
    n = 10  # Matrix size
    History = randn(30, n)

    for i in 1:size(History, 2)
        History[:, i] /= norm(History[:, i])  # Normalize column i
    end

    XTX = Symmetric(History'History)  # Ensure A is symmetric positive definite

    XTX_copy = copy(XTX)


    XTXloop = copy(XTX)

    #First make small...

    XTXloopinv = Symmetric(inv(XTXloop))
    removeinverse!(XTXloopinv,5)

    @test isapprox(XTXloopinv,XTXloopinv')
    @test isapprox((XTXloopinv*XTX)[setdiff(1:n,5),setdiff(1:n,5)],I(10)[setdiff(1:n,5),setdiff(1:n,5)])
    @test isapprox((XTXloopinv[setdiff(1:n,5),setdiff(1:n,5)]*XTX[setdiff(1:n,5),setdiff(1:n,5)]),I(9))

    removeinverse!(XTXloopinv,6)


    @test isapprox(XTXloopinv,XTXloopinv')
    @test isapprox((XTXloopinv*XTX)[setdiff(1:n,[5,6]),setdiff(1:n,[5,6])],I(10)[setdiff(1:n,[5,6]),setdiff(1:n,[5,6])])
    @test isapprox((XTXloopinv[setdiff(1:n,[5,6]),setdiff(1:n,[5,6])]*XTX[setdiff(1:n,[5,6]),setdiff(1:n,[5,6])]),I(8))


    removeinverse!(XTXloopinv,8)

    @test isapprox(XTXloopinv,XTXloopinv')
    @test isapprox((XTXloopinv*XTX)[setdiff(1:n,[5,6,8]),setdiff(1:n,[5,6,8])],I(10)[setdiff(1:n,[5,6,8]),setdiff(1:n,[5,6,8])])
    @test isapprox((XTXloopinv[setdiff(1:n,[5,6,8]),setdiff(1:n,[5,6,8])]*XTX[setdiff(1:n,[5,6,8]),setdiff(1:n,[5,6,8])]),I(7))


    #Now add back in the column and hopefully we get back to what we had...

    addinverse!(XTXloopinv,6,XTXloop[6,1:n])

    @test isapprox(XTXloopinv,XTXloopinv')
    @test isapprox((XTXloopinv*XTX)[setdiff(1:n,[5,8]),setdiff(1:n,[5,8])],I(10)[setdiff(1:n,[5,8]),setdiff(1:n,[5,8])])
    @test isapprox((XTXloopinv[setdiff(1:n,[5,8]),setdiff(1:n,[5,8])]*XTX[setdiff(1:n,[5,8]),setdiff(1:n,[5,8])]),I(8))


    addinverse!(XTXloopinv,5,XTXloop[5,1:n])

    @test isapprox(XTXloopinv,XTXloopinv')
    @test isapprox((XTXloopinv*XTX)[setdiff(1:n,[8]),setdiff(1:n,[8])],I(10)[setdiff(1:n,[8]),setdiff(1:n,[8])])
    @test isapprox((XTXloopinv[setdiff(1:n,[8]),setdiff(1:n,[8])]*XTX[setdiff(1:n,[8]),setdiff(1:n,[8])]),I(9))


    addinverse!(XTXloopinv,8,XTXloop[8,1:n])

    @test isapprox(XTXloopinv,XTXloopinv')
    @test isapprox(XTXloopinv*XTX,I(10))

end

end

using AndersonPlus: quickAAHistoricalStuff, AnglesUpdate!
using LinearAlgebra

@testset "AnglesUpdate!" begin
    numrows = 5
    m = 3

    # Create a quickAAHistoricalStuff object
    HS = quickAAHistoricalStuff(numrows, m)

    # Initialize some values
    HS.Gtilde_k .= reshape(collect(1:(numrows*m)), numrows, m)  # Fill Gtilde_k with known values
    HS.positions .= [1, -1, 2]  # Only first and third should be updated
    HS.sin_k .= zeros(m)  # Ensure it's initialized

    gtilde_k = [0.5, 1.0, -0.5, 2.0, -1.0]  # Some arbitrary test vector

    # Compute expected dot products for valid indices
    expected_sin_k = zeros(m)
    expected_sin_k[1] = dot(gtilde_k, HS.Gtilde_k[:, 1])
    expected_sin_k[3] = dot(gtilde_k, HS.Gtilde_k[:, 3])

    # Run the function
    AnglesUpdate!(HS, gtilde_k)

    # Test that only the expected indices were updated
    @test HS.sin_k[1] ≈ expected_sin_k[1]
    @test HS.sin_k[3] ≈ expected_sin_k[3]
    @test HS.sin_k[2] == 0  # Should remain unchanged
end


using AndersonPlus: filteringindices, Filtering!

@testset "Filtering Functions" begin
    numrows = 5
    m = 3

    # Define a simple threshold function for testing
    threshold_func = (positions, iterations) -> fill(0.5, length(positions))

    # Create method parameters
    methodparams = Dict(:threshold_func => threshold_func, :m => m)

    # Initialize a quickAAHistoricalStuff object
    HS = quickAAHistoricalStuff(numrows, m)
    HS.positions .= [1, 2, -1]  # Third entry should be ignored
    HS.sin_k .= [0.6, 0.4, 0.8]  # First and third exceed threshold

    # Check filteringindices
    expected_filtered = [true, false, true]  # Based on sin_k > 0.5
    @test filteringindices(HS, methodparams) == expected_filtered

    # Apply Filtering!
    Filtering!(HS, methodparams)

    # First and third positions should be set to -1
    @test HS.positions == [-1, 2, -1]

    # Check if the GtildeTGtildeinv matrix is updated correctly (i.e., reflects the removal of the positions)
    # Let's assume it reduces the size by the number of filtered positions.

    # Check that sin_k values that are filtered are removed (this assumes Filtering! works as intended)
end