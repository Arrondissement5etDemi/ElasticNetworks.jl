# ElasticityAD.jl

**ElasticityAD.jl** provides numerically EXACT linear-response computations for elastic networks of harmonic springs via automatic differentiation.  
Designed for theoretical and computational research, it supports periodic boundary conditions, topology editing (e.g. edge pruning, node merging), geometric disorder, and 3D visualization, within a unified, easy-to-use API.

---

## 🧠 Features

- **Automatic differentiation** for exact linear-response tensors  
- **Periodic networks** with customizable unit cells and image vectors  
- **Topology tools**: `pluck_out_edge!`, `plug_in_edge!`, `simplify_net!`, and more  
- **Lattice generators**: `cubic_network`, `diamond1000`, `disordered_cubic_network`, `er`  
- **Visualization** with Makie (switchable between `GLMakie` locally and `CairoMakie` in CI)  
- **Network I/O**: load/save routines for sharing and reproducibility

---

## 🚀 Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Arrondissement5etDemi/ElasticityAD.jl")
```
## 📘 Quick Start

```julia
using ElasticityAD

net = diamond1000(10.0, 0.05)
fig = visualize_net(net)
```

You can also explore:

```julia
#create erdos-renyi network with 1300 nodes at random 3D positions, with box side length 9.4766, mean degree = 4, prestrain on edges = 0.3, and max edge rest length 3.0.
ern = er(9.4766, rand(3, 1300), 4.0, 0.2, 3.0) 
#minimize the elastic energy
relax!(ern) 
#create a copy of the network 
erncopy = deepcopy(ern)
#remove contributions to soft modes such as degree-0, degree-1 nodes, and degree-2 nodes joining 2 edges bent at an angle.
simplify_net!(erncopy) 
#compute elastic moduli with automatic differentiation.
moduli(erncopy) 
```
## 📚 Documentation

Comprehensive documentation for the elastic network infrastructure is available at:

👉 [ElasticityAD.jl Documentation](https://arrondissement5etdemi.github.io/ElasticityAD.jl)
