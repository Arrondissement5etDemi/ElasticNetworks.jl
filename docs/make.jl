import Pkg; Pkg.add("Documenter")
using Documenter, ElasticityAD

makedocs(
    sitename = "ElasticityAD.jl",
    modules = [ElasticityAD],
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(;
    repo = "github.com/Arrondissement5etDemi/ElasticityAD.jl"
)
