# Inside make.jl
# Activate the parent project directory
push!(LOAD_PATH,"./src/")
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

deploydocs(
    repo = "github.com/tbretschneider/AndersonPlus.jl.git",
)
