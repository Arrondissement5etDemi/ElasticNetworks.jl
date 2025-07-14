# ElasticityAD.jl

**ElasticityAD.jl** provides exact linear-response computations for elastic networks via automatic differentiation.  
Designed for theoretical and computational research, it supports periodic boundary conditions, topology editing, geometric disorder, and 3D visualization—within a unified, easy-to-use API.

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
ern = er(9.4766, rand(3, 1000), 4.0, 0.2) #A erdos-renyi network with 1000 nodes at random 3D positions, with box side length 9.4766, mean degree = 4 and prestrain on edges = 0.2.
relax!(ern) #minimize the elastic energy
erncopy = deepcopy(ern)
simplify_net!(erncopy) #remove soft modes
moduli(erncopy)
```
## 📚 Documentation

Comprehensive documentation for the elastic network infrastructure is available at:

👉 [ElasticityAD.jl Documentation](https://arrondissement5etdemi.github.io/ElasticityAD.jl)

## 🧪 Citation

```bibtex
@misc{ElasticityAD,
  author       = {Haina Wang},
  title        = {ElasticityAD.jl: Exact Linear Response in Elastic Networks via Automatic Differentiation},
  year         = {2025},
  howpublished = {\url{https://github.com/Arrondissement5etDemi/ElasticityAD.jl}},
  note         = {GitHub repository}
}