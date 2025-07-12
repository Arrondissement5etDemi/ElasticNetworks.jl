include("elastic_network.jl")
using IterativeSolvers

"""
    moduli(net)

Computes the elastic moduli of a network by applying small deformations and extracting the stiffness components.

# Description
This function performs an energy minimization (`relax!`) and then computes elastic moduli using automatic differentiation. The moduli components are obtained by introducing small strain perturbations in various deformation modes and measuring the corresponding energy response.

# Arguments
- `net::Network` : The network structure containing connectivity, node positions, and edge properties.

# Behavior
1. **Relaxation** - Minimizes the elastic energy of the network to find a stable configuration.
2. **Deformation Basis Construction** - Defines strain modes using deformation basis functions.
3. **Hessian Computation** - Computes the Hessian of the system at equilibrium.
4. **Energy Perturbation & Differentiation** - Uses automatic differentiation (`ForwardDiff.hessian`) to compute energy responses to small deformations.
5. **Moduli Extraction** - Computes bulk modulus (`B`), shear modulus (`G`), and individual elastic constants.

# Returns
- `B::Float64` : Bulk modulus.
- `G::Float64` : Shear modulus.
- `c1111::Float64`, `c2222::Float64`, `c3333::Float64` : Normal stress components.
- `c1212::Float64`, `c1313::Float64`, `c2323::Float64` : Shear stress components.
- `c1122::Float64`, `c1133::Float64`, `c2233::Float64` : Mixed stress components.

"""
function moduli(net::Network)
    relax!(net)
    #simplify_net!(net)
    basis, points, edge_nodes, rls, iis, youngs = net_info_primitive(net)
    deformed_bases = Dict{String, Function}()
    deformed_bases["1111"] = ϵ -> ([ϵ 0 0; 0 0 0; 0 0 0] + I)*basis
    deformed_bases["2222"] = ϵ -> ([0 0 0; 0 ϵ 0; 0 0 0] + I)*basis
    deformed_bases["3333"] = ϵ -> ([0 0 0; 0 0 0; 0 0 ϵ] + I)*basis
    deformed_bases["1212"] = ϵ -> ([0 ϵ 0; ϵ 0 0; 0 0 0] + I)*basis
    deformed_bases["1313"] = ϵ -> ([0 0 ϵ; 0 0 0; ϵ 0 0] + I)*basis
    deformed_bases["2323"] = ϵ -> ([0 0 0; 0 0 ϵ; 0 ϵ 0] + I)*basis
    deformed_bases["1122"] = ϵ -> ([ϵ 0 0; 0 ϵ 0; 0 0 0] + I)*basis
    deformed_bases["1133"] = ϵ -> ([ϵ 0 0; 0 0 0; 0 0 ϵ] + I)*basis
    deformed_bases["2233"] = ϵ -> ([0 0 0; 0 ϵ 0; 0 0 ϵ] + I)*basis
    n = nv(net.g)
    H = zeros(3*n, 3*n)
    hessian!(H, basis, points, edge_nodes, rls, iis, youngs)
    function make_curry_function(component)
        function curry(ϵ)
            deformed_basis = deformed_bases[component](ϵ[1])
            F = -gradient(deformed_basis, points, edge_nodes, rls, iis, youngs)
            nonaffine_displacements = qr(H, Val(true)) \ F
            return elastic_energy(deformed_basis, points + nonaffine_displacements, edge_nodes, rls, iis, youngs)
        end
        return curry
    end
    v = det(basis)
    c1111 = ForwardDiff.hessian(make_curry_function("1111"), [0])[1]/v
    c2222 = ForwardDiff.hessian(make_curry_function("2222"), [0])[1]/v
    c3333 = ForwardDiff.hessian(make_curry_function("3333"), [0])[1]/v
    c1212 = ForwardDiff.hessian(make_curry_function("1212"), [0])[1]/(4*v)
    c1313 = ForwardDiff.hessian(make_curry_function("1313"), [0])[1]/(4*v)
    c2323 = ForwardDiff.hessian(make_curry_function("2323"), [0])[1]/(4*v)
    c1122 = ForwardDiff.hessian(make_curry_function("1122"), [0])[1]/(2*v) - (c1111 + c2222)/2
    c1133 = ForwardDiff.hessian(make_curry_function("1133"), [0])[1]/(2*v) - (c1111 + c3333)/2
    c2233 = ForwardDiff.hessian(make_curry_function("2233"), [0])[1]/(2*v) - (c2222 + c3333)/2
    B = 1/9*(c1111 + c2222 + c3333 + 2*(c1122 + c1133 + c2233))
    G = 1/15*(3*(c1212 + c1313 + c2323) + c1111 + c2222 + c3333 - c1122 - c1133 - c2233)
    println("B = $B \n G = $G \n c1111 = $c1111 \n c2222 = $c2222 \n c3333 = $c3333 \n c1212 = $c1212 \n c1313 = $c1313 \n c2323 = $c2323 \n c1122 = $c1122 \n c1133 = $c1133 \n c2233 = $c2233")
    return B, G, c1111, c2222, c3333, c1212, c1313, c2323, c1122, c1133, c2233
end




