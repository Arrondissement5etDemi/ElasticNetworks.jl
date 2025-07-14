# ElasticityAD.jl Documentation

```@contents
```

## Data structure for 3D elastic networks made of harmonic springs.
```@docs
mutable struct Network
```

## Network discriptors
```@docs
mean_degree(net::Network) → Float64
strains(net::Network) → Vector{Float64}
tensions(net::Network) → Vector{Float64}
elastic_energy(net::Network) → Float64
energy_gradient(net::Network) → Vector{Float64}
energy_hessian(net::Network) → Matrix{Float64}
```