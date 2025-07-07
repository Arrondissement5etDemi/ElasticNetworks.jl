module ElasticityAD
    export(Network)
    export(moduli)
    export(prestrained_network)
    export(relax)
    export(relax!)
    export(diamond1000)
    include("moduliAD.jl")
    include("zoo.jl")
end