import Pkg; Pkg.add("Documenter")
Pkg.add("Graphs")
using Documenter, ElasticNetworks, Graphs
push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "ElasticNetworks.jl",
    modules = [ElasticNetworks],
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(;
    repo = "github.com/Arrondissement5etDemi/ElasticNetworks.jl",
    devbranch = "main"
)


