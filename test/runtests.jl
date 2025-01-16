using AndersonPlus
using Test

include("runhelpertests.jl")

@testset "AndersonPlus.jl" begin
    @test AndersonPlus.greet_your_package_name() == "Hello YourPackageName!"
    @test AndersonPlus.greet_your_package_name() != "Hello world!"
end
