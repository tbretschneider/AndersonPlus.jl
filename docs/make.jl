# Inside make.jl
# Activate the parent project directory
Pkg.activate(".")  # Activate the project in the current directory (which is the parent directory)

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
