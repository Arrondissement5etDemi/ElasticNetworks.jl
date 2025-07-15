import Pkg; Pkg.add("Documenter")
Pkg.add("Graphs")
using Documenter, ElasticityAD, Graphs
push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "ElasticityAD.jl",
    modules = [ElasticityAD],
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(;
    repo = "github.com/Arrondissement5etDemi/ElasticityAD.jl",
    devbranch = "main"
)


