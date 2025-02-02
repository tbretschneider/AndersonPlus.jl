using AndersonPlus
using Test

include("runhelpertests.jl")

#include("runmethodtests.jl")

@testset "AndersonPlus.jl" begin
    @test AndersonPlus.greet_your_package_name() == "Hello YourPackageName!"
    @test AndersonPlus.greet_your_package_name() != "Hello world!"
end
