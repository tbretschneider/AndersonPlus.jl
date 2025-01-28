# Inside make.jl
push!(LOAD_PATH,"../src/")
using AndersonPlus
using Documenter

makedocs(
         sitename = "AndersonPlus.jl",
         modules  = [AndersonPlus],
         pages=[
                "Home" => "index.md"
               ],
	 debug=true,
	 checkdocs = :none,)
