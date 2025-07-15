# ElasticNetworks.jl Documentation

```@contents
```

## Data structure for 3D elastic networks made of harmonic springs.
```@docs
Network
```

## Network descriptors
```@docs
mean_degree(net::Network)
strains(net::Network)
tensions(net::Network)
elastic_energy(net::Network)
energy_gradient(net::Network)
energy_hessian(net::Network)
```

## Network modifiers: geometry only
```@docs
relax(net; show_trace=false, g_tol=1e-6)
```

## Compute elastic moduli
```@docs
moduli(net::Network)
```

## Network modifiers: topology
```@docs
ElasticNetworks.add_edge!(net::Network, s::Int, d::Int, rl::Float64, y::Float64 = 1.0)
ElasticNetworks.rem_edge!(net::Network, s::Int, d::Int)
ElasticNetworks.rem_vertex!(net::Network, v::Int)
pluck_out_edge!(net::Network, e::Graphs.SimpleGraphs.SimpleEdge, direction::Function)
plug_in_edge!(net::Network, deg1node::Int, accepting_node::Int)
simplify_net!(net::Network)
```

## Visualizing a network
```@docs
visualize_net(net::Network)
```

## Library of common networks
```@docs
prestrained_network(g, basis, points, ϵ, default_youngs=1.0)
diamond1000(l::Float64, ϵ::Float64)
cubic_network(l::Float64, n_layers::Int, ϵ::Float64 = 0.0)
disordered_cubic_network(l::Float64, n_layers::Int, disorder_param::Float64, ϵ::Float64)
er(l::Float64, points::Matrix{Float64}, z::Float64, ϵ::Float64, max_rl::Float64)
```

## IO functions
```@docs
net_info_primitive(net::Network)
load_network(filename::String)
save_network(net::Network, filename::String)
```