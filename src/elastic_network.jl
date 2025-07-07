using Graphs, LoopVectorization, Optim, LinearAlgebra, Statistics, ForwardDiff, ReverseDiff, JLD2
import Graphs: rem_vertex!, add_edge!, rem_edge!

quick_euclidean_graph(N::Int, cutoff) = euclidean_graph(N, 3; cutoff = cutoff, bc = :periodic)
quick_euclidean_graph(points::Matrix{Float64}, cutoff) = euclidean_graph(points; cutoff = cutoff, bc = :periodic)

"""
    mutable struct Network

Represents a network of interconnected nodes with physical properties such as rest lengths and elasticity.

# Fields
- `g::SimpleGraph` : Graph specifying the connectivity of the network.
- `basis::Matrix{Float64}` : Basis for the nodes.
- `points::Matrix{Float64}` : Coordinates of the nodes in space.
- `rest_lengths::Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64}` : Rest lengths of the spring edges in the network.
- `image_info::Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Vector{Int}}` : Specifies which image of node `j` node `i` is connected to in periodic boundary conditions.
- `youngs::Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64}` : Young’s modulus of the spring edges, defining their stiffness.

"""
mutable struct Network
    g :: SimpleGraph 
    basis :: Matrix{Float64} 
    points :: Matrix{Float64} 
    rest_lengths :: Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64} 
    image_info :: Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Vector{Int}} 
    youngs :: Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64} 
end

"""
    prestrained_network(g, basis, points, ϵ, default_youngs=1.0)

Creates a `Network` with prestrained edges based on the provided graph, basis vectors, and node coordinates.

# Arguments
- `g::SimpleGraph` : Graph specifying the connectivity of the network.
- `basis::Matrix{Float64}` : Basis vectors defining the space in which the nodes are embedded.
- `points::Matrix{Float64}` : Coordinates of the nodes.
- `ϵ::Float64` : Prestrain factor, adjusting the rest lengths of spring edges.
- `default_youngs::Float64` (optional, default = 1.0) : Default Young’s modulus assigned to all edges.

# Returns
- `Network` : A network with computed rest lengths, image information for periodic boundaries, and Young's modulus values for each edge.

"""
function prestrained_network(g, basis, points, ϵ, default_youngs = 1.0)
    rest_lengths = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64}()
    image_info = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Vector{Int}}()
    youngs = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64}()
    for e in edges(g)
        image_info[e] = get_image_info(points[:, src(e)], points[:, dst(e)])
        rest_lengths[e] = norm(basis*min_image_vector_rel(points[:, src(e)], points[:, dst(e)]))*(1/(1 + ϵ))
        youngs[e] = default_youngs
    end
    return Network(g, basis, points, rest_lengths, image_info, youngs)
end

function strains(net)
    euclidean_dist(v1::Int, v2::Int) = norm(net.basis*min_image_vector_rel(net.points[:, v1], net.points[:, v2]))
    return [abs((euclidean_dist(k.src, k.dst) - net.rest_lengths[k])/net.rest_lengths[k]) for k in edges(net.g)]
end

function tensions(net)
    σij = strains(net)
    yij = [net.youngs[e] for e in edges(net.g)]
    return [σij[k]*yij[k] for k in eachindex(σij)]
end

"""
    net_info_primitive(net::Network)

Converts network data into numerical matrices containing only primitive types, enabling efficient SIMD (Single Instruction, Multiple Data) operations.

# What is SIMD?
SIMD is a parallel computing technique that allows the same operation to be applied to multiple data points simultaneously. By structuring data for SIMD, numerical computations can be executed faster by leveraging modern processor optimizations.

# Arguments
- `net::Network` : The network structure containing connectivity, node positions, and edge properties.

# Returns
- `basis::Matrix{Float64}` : Basis vectors defining the network space.
- `points::Vector{Float64}` : Flattened node coordinates.
- `edges::Matrix{Int64}` : Edge connections formatted for numerical processing.
- `rls::Vector{Float64}` : Rest lengths of edges.
- `iis::Matrix{Int}` : Image information for periodic boundary conditions.
- `youngs::Vector{Float64}` : Young’s modulus values for edges.

"""
function net_info_primitive(net::Network)
    egs = hcat(([src(e), dst(e)] for e in edges(net.g))...)
    rls = [net.rest_lengths[e] for e in edges(net.g)]
    iis = hcat((net.image_info[e] for e in edges(net.g))...)
    youngs = [net.youngs[e] for e in edges(net.g)]
    return net.basis, collect(Iterators.flatten(net.points)), egs, rls, iis, youngs
end
#__________________________________________________________________________
"""
    elastic_energy(basis, points, egs, rls, iis, youngs)
    elastic_energy(net)

Computes the total elastic potential energy of a network based on edge deformations.

# Methods:
- `elastic_energy(basis, points, egs, rls, iis, youngs) → Float64`  
  Computes the energy using explicitly provided network parameters.
  
- `elastic_energy(net) → Float64`  
  A convenience wrapper that extracts the network parameters using `net_info_primitive(net)`.

# Arguments
- `basis::Matrix{Float64}` : Basis vectors defining the spatial representation.
- `points::Vector{Float64}` : Flattened node coordinates.
- `egs::Matrix{Int64}` : Edge connections within the network.
- `rls::Vector{Float64}` : Rest lengths of edges.
- `iis::Matrix{Int}` : Image offsets for periodic boundary conditions.
- `youngs::Vector{Float64}` : Young’s modulus values for each edge.
- `net::Network` : The network structure containing connectivity, node positions, and edge properties.

# Behavior
- Iterates over edges, computing their current length and energy contribution.
- Uses a quadratic potential energy model to determine stored elastic energy.
- The wrapper function `elastic_energy(net)` simplifies calling the function without manual data extraction.

# Returns
- `Float64` : The total elastic potential energy of the network.

"""
function elastic_energy(basis, points, egs, rls, iis, youngs)
    result = 0
    for k in axes(egs, 2)
        i = egs[1, k]
        j = egs[2, k]
        i_ind, j_ind = (i - 1)*3, (j - 1)*3#because points are flattened
        dx = points[1 + j_ind] + iis[1, k] - points[1 + i_ind]
        dy = points[2 + j_ind] + iis[2, k] - points[2 + i_ind]
        dz = points[3 + j_ind] + iis[3, k] - points[3 + i_ind]
        r1 = dx*basis[1, 1] + basis[1, 2]*dy + basis[1, 3]*dz
        r2 = dx*basis[2, 1] + basis[2, 2]*dy + basis[2, 3]*dz
        r3 = dx*basis[3, 1] + basis[3, 2]*dy + basis[3, 3]*dz
        edge_length_ij = √(r1^2 + r2^2 + r3^2)
        edge_energy_ij = youngs[k]*(rls[k] - edge_length_ij)^2/rls[k]
        result += edge_energy_ij
    end
    return result/2
end

elastic_energy(net) = elastic_energy(net_info_primitive(net)...)

"""
    gradient!(result, basis, points, egs, rls, iis, youngs)

Computes the gradient of the elastic potential energy with respect to node positions and stores the result in-place.

# Arguments
- `result::Vector{Float64}` : Pre-allocated vector to store the computed gradient.
- `basis::Matrix{Float64}` : Basis vectors defining the spatial representation.
- `points::Vector{Float64}` : Flattened node coordinates for efficient access.
- `egs::Matrix{Int64}` : Edge connections within the network.
- `rls::Vector{Float64}` : Rest lengths of the edges.
- `iis::Matrix{Int}` : Image offsets for periodic boundary conditions.
- `youngs::Vector{Float64}` : Young’s modulus values for each edge.

# Behavior
- Initializes `result` to zero before computing forces.
- Iterates over edges, calculating displacements and forces using the basis transformation.
- Accumulates forces for each node pair and stores the final transformed values in `result`.

# Returns
- Modifies `result` in-place with the computed gradient values.

"""
function gradient!(result, basis::AbstractArray{T}, points, egs, rls, iis, youngs) where T
    g = gradient(basis, points, egs, rls, iis, youngs)
    for i in eachindex(result) 
        result[i] = g[i] 
    end
end

"""
    gradient(basis, points, edge_nodes, rls, iis, youngs)

Computes the gradient of the elastic energy with respect to node positions, formatted for automatic differentiation (autodiff).

# Purpose
This function avoids mutable arrays to ensure compatibility with autodiff tools. Instead of modifying values in-place, it constructs updated matrices at each iteration.

# Arguments
- `basis::Matrix{Float64}` : Basis vectors defining the spatial representation.
- `points::Matrix{Float64}` : Node coordinates, stored column-wise.
- `edge_nodes::Matrix{Int64}` : Connectivity of the network, where each column defines an edge.
- `rls::Vector{Float64}` : Rest lengths for each edge.
- `iis::Matrix{Int}` : Image offsets for periodic boundary conditions.
- `youngs::Vector{Float64}` : Young’s modulus values for each edge.

# Returns
- `Matrix{Float64}` : The negative gradient of the elastic energy with respect to node positions.

"""

function gradient(basis::AbstractArray{T}, points::AbstractArray, edge_nodes, rls, iis, youngs) where T
    result = zeros(T, size(points))
    n = (Int)(length(points)/3)
    for k in axes(edge_nodes, 2)
        i, j = edge_nodes[1, k], edge_nodes[2, k]
        i_ind, j_ind = (i - 1)*3, (j - 1)*3 #because points are flattened
        dx = points[1 + j_ind] + iis[1, k] - points[1 + i_ind]
        dy = points[2 + j_ind] + iis[2, k] - points[2 + i_ind]
        dz = points[3 + j_ind] + iis[3, k] - points[3 + i_ind]
        r1 = basis[1, 1]*dx + basis[1, 2]*dy + basis[1, 3]*dz
        r2 = basis[2, 1]*dx + basis[2, 2]*dy + basis[2, 3]*dz
        r3 = basis[3, 1]*dx + basis[3, 2]*dy + basis[3, 3]*dz
        edge_length_ij = √(r1^2 + r2^2 + r3^2)
        factor = youngs[k]*(edge_length_ij - rls[k])/rls[k]/edge_length_ij 
        result[1 + i_ind] -= factor*r1
        result[2 + i_ind] -= factor*r2
        result[3 + i_ind] -= factor*r3
        result[1 + j_ind] += factor*r1
        result[2 + j_ind] += factor*r2
        result[3 + j_ind] += factor*r3
    end
    for i in 1:n
        i_ind = (i - 1)*3
        result[1 + i_ind] = basis[1, 1]*result[1 + i_ind] + basis[1, 2]*result[2 + i_ind] + basis[1, 3]*result[3 + i_ind]
        result[2 + i_ind] = basis[2, 1]*result[1 + i_ind] + basis[2, 2]*result[2 + i_ind] + basis[2, 3]*result[3 + i_ind]
        result[3 + i_ind] = basis[3, 1]*result[1 + i_ind] + basis[3, 2]*result[2 + i_ind] + basis[3, 3]*result[3 + i_ind]
    end
    return result
end

"""
    hessian!(H, basis, points, egs, rls, iis, youngs)

Computes the Hessian matrix (the second derivative with respect to node positions) of the elastic potential energy in place.

# Arguments
- `H::Matrix{Float64}` : Pre-allocated (3N × 3N) matrix that will be modified in-place.
- `basis::Matrix{Float64}` : Basis vectors defining the spatial representation.
- `points::Vector{Float64}` : Flattened node coordinates (length = 3N).
- `egs::Matrix{Int64}` : Edge connections within the network.
- `rls::Vector{Float64}` : Rest lengths for each edge.
- `iis::Matrix{Int}` : Image offsets for periodic boundary conditions.
- `youngs::Vector{Float64}` : Young’s modulus values for each edge.

# Behavior
- Modifies the Hessian `H` directly instead of allocating a new matrix.
- Each edge contributes a 3×3 block that is scattered into the global matrix.

# Returns
- The Hessian matrix `H` is updated in-place.

"""
function hessian!(H, basis, points, egs, rls, iis, youngs)
    N = div(length(points), 3)  # Number of nodes
    fill!(H, 0.0)  # Ensure H starts from zero
    I3 = Matrix{Float64}(I, 3, 3)  # 3×3 identity matrix
    for k in axes(egs, 2)
        i, j = egs[1, k], egs[2, k]
        i_ind, j_ind = (i - 1) * 3, (j - 1) * 3
        # Compute displacement with periodic adjustments
        dx = points[1 + j_ind] + iis[1, k] - points[1 + i_ind]
        dy = points[2 + j_ind] + iis[2, k] - points[2 + i_ind]
        dz = points[3 + j_ind] + iis[3, k] - points[3 + i_ind]
        # Transform displacement into basis coordinates
        r_local = [
            basis[1, 1] * dx + basis[1, 2] * dy + basis[1, 3] * dz,
            basis[2, 1] * dx + basis[2, 2] * dy + basis[2, 3] * dz,
            basis[3, 1] * dx + basis[3, 2] * dy + basis[3, 3] * dz
        ]
        L = norm(r_local)  
        # Avoid division by zero
        if L == 0.0
            continue
        end       
        # Compute unit vector
        u = r_local / L
        # Compute local Hessian
        H_local = (youngs[k] / rls[k]) * (I3 + (rls[k] / L) * (u * u' - I3))
        # Transform to global coordinates
        H_block = basis * H_local * transpose(basis)
        # Scatter into pre-allocated Hessian
        for α in 1:3, β in 1:3
            H[i_ind + α, i_ind + β] += H_block[α, β]
            H[j_ind + α, j_ind + β] += H_block[α, β]
            H[i_ind + α, j_ind + β] -= H_block[α, β]
            H[j_ind + α, i_ind + β] -= H_block[α, β]
        end
    end
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
    relax(net; show_trace=false, g_tol=1e-6)
    relax!(net; show_trace=false, g_tol=1e-6)

Minimizes the elastic energy of a network using Newton's optimization method, adjusting node positions to a lower-energy configuration.

# Methods:
- `relax(net; show_trace, g_tol) → (Matrix{Float64}, Float64)`  
  Returns the optimized node positions and the minimized elastic energy value.
  
- `relax!(net; show_trace, g_tol) → nothing`  
  Performs relaxation in-place by updating `net.points` directly.

# Arguments
- `net::Network` : The network structure containing connectivity, node positions, and edge properties.
- `show_trace::Bool` (optional, default=`false`) : Whether to display optimization progress.
- `g_tol::Float64` (optional, default=`1e-6`) : Gradient tolerance for termination criteria.

# Behavior
- Extracts network data into primitive form using `net_info_primitive(net)`.
- Defines the elastic energy function (`f`), its gradient (`g!`), and Hessian (`h!`).
- Uses Newton's method (`Optim.optimize`) for minimization.
- `relax!` modifies the network in-place, updating `net.points`.

"""
function relax(net; show_trace = false, g_tol = 1e-6)
    basis, initial_points, edge_nodes, rls, iis, youngs = net_info_primitive(net)
    f(points) = elastic_energy(basis, points, edge_nodes, rls, iis, youngs)
    g!(G, points) = gradient!(G, basis, points, edge_nodes, rls, iis, youngs)
    h!(H, points) = hessian!(H, basis, points, edge_nodes, rls, iis, youngs)
    opt = optimize(f, g!, h!, initial_points, Optim.Options(show_trace = show_trace, iterations = 500, g_tol = g_tol); inplace = true) #Newton's method
    return reshape(Optim.minimizer(opt), size(net.points)), Optim.minimum(opt)
end

function relax!(net; show_trace = false, g_tol = 1e-6)
    net.points = relax(net; show_trace = show_trace, g_tol = g_tol)[1]
end

#_____________________________________________________________________________network modifiers

function add_edge!(net::Network, s::Int, d::Int, rl::Float64)
    trues = min(s, d)
    trued = max(s, d)
    add_edge!(net.g, trues, trued)
    e = Edge(trues, trued)
    net.rest_lengths[e] = rl
    net.image_info[e] = get_image_info(net.points[:, trues], net.points[:, trued])
end

function rem_edge!(net::Network, s::Int, d::Int)
    rem_edge!(net.g, min(s, d), max(s, d))
    e = Edge(min(s, d), max(s, d))
    pop!(net.rest_lengths, e)
    pop!(net.image_info, e)
end

function rem_vertex!(net::Network, v::Int)
    original_n = nv(net.g)
    Graphs.rem_vertex!(net.g, v) #Graphs.jl moves the last vertex to index v, so there are now just n - 1 vertices.
    net.points = hcat(net.points[:, 1:v - 1], net.points[:, original_n], net.points[:, v + 1: original_n - 1])
    original_edges = deepcopy(keys(net.rest_lengths))
    new_ind(x) = x > v ? x - 1 : x
    for e in original_edges
        s, d = src(e), dst(e)
        if s == v || d == v
            delete!(net.rest_lengths, e)
            delete!(net.image_info, e)
            continue
        end
        if s == original_n
            rl = pop!(net.rest_lengths, e)
            ii = pop!(net.image_info, e)
            new_s, new_d = min(v, d), max(v, d)
            net.rest_lengths[Edge(new_s, new_d)] = rl
            iisign = new_d == d ? 1 : -1
            net.image_info[Edge(new_s, new_d)] = iisign*ii
        end
        if d == original_n
            rl = pop!(net.rest_lengths, e)
            ii = pop!(net.image_info, e)
            new_s, new_d = min(s, v), max(s, v)
            net.rest_lengths[Edge(new_s, new_d)] = rl
            iisign = new_s == s ? 1 : -1 
            net.image_info[Edge(new_s, new_d)] = iisign*ii
        end
    end
end
#_________________________________________________________________________________________________
#__________________________________________________________________________________________________helper functions below_____________________

min_direction(x) = findmin(abs.([x - 1, x, x + 1]))[2] - 2

function get_image_info(src, dst)
    return min_direction.(dst - src)
end

function min_image_vector_rel(v1, v2)
    vec_diff = v2 - v1
    min_directions = min_direction.(vec_diff)
    return vec_diff + min_directions
end
