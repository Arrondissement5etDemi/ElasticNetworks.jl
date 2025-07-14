using Graphs, LoopVectorization, Optim, LinearAlgebra, Statistics, ForwardDiff, ReverseDiff, JLD2, CairoMakie

"""
    mutable struct Network

Models a Hookean spring network.

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
    mean_degree(net::Network) → Float64

Returns the average degree of nodes in the elastic network `net` that have degree ≥ 3.

# Arguments
- `net::Network` : An elastic network 

# Returns
- `Float64` : The mean degree of qualifying nodes, or `0.0` if none meet the threshold.

"""
function mean_degree(net::Network)
    degs = degree(net.g)
    filter!(x -> x ≥ 3, degs)
    if length(degs) ≥ 1
        return mean(degs)
    else
        return 0 
    end
end

"""
    strains(net::Network) → Vector{Float64}

Computes the edge-wise strain magnitudes (absolute values) in the elastric network `net`, defined as the relative deviation from rest length for each edge.

# Arguments
- `net::Network` :  An elastic network 

# Returns
- `Vector{Float64}` : A vector of strain values for each edge in `net`.

"""
function strains(net::Network)
    euclidean_dist(v1::Int, v2::Int) = norm(net.basis*min_image_vector_rel(net.points[:, v1], net.points[:, v2]))
    return [abs((euclidean_dist(k.src, k.dst) - net.rest_lengths[k])/net.rest_lengths[k]) for k in edges(net.g)]
end

"""
    tensions(net::Network) → Vector{Float64}

Computes the edge-wise tensions in the elastic network `net`, defined as the product of strain and Young’s modulus for each edge.

# Arguments
- `net::Network` : An elastic network

# Returns
- `Vector{Float64}` : A vector of tension values for each edge in `net`.

"""
function tensions(net::Network)
    σij = strains(net)
    yij = [net.youngs[e] for e in edges(net.g)]
    return [σij[k]*yij[k] for k in eachindex(σij)]
end

#__________________________________________________________________________
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

"""
    elastic_energy(net::Network) → Float64

Computes the total elastic potential energy of the elastic network `net`, based on pairwise edge deformations relative to their rest lengths.

# Arguments
- `net::Network` : An elastic network

# Returns
- `Float64` : Total elastic energy stored in the network

"""
function elastic_energy(net::Network) 
    return elastic_energy(net_info_primitive(net)...)
end

function gradient!(result, basis::AbstractArray{T}, points, egs, rls, iis, youngs) where T
    g = gradient(basis, points, egs, rls, iis, youngs)
    for i in eachindex(result) 
        result[i] = g[i] 
    end
end

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
    return collect(Iterators.flatten(basis*reshape(result, (3, n))))
end

"""
    energy_gradient(net::Network) → Vector{Float64}

Computes the gradient of the elastic energy in the elastic network `net`, evaluated with respect to node positions.

# Arguments
- `net::Network` : An elastic network

# Returns
- `Vector{Float64}` : The energy gradient vector over all nodes in `net`

"""
energy_gradient(net) = gradient(net_info_primitive(net)...)

function hessian!(H, basis, points, egs, rls, iis, youngs)
    N = div(length(points), 3)  # Number of nodes
    fill!(H, 0.0)  # Ensure H starts from zero
    I3 = Matrix{Float64}(I, 3, 3)  # 3×3 identity matrix
    bt = transpose(basis)
    ne = size(egs, 2)
    u_array = zeros(4, ne)
    for k in 1:ne
        i, j = egs[1, k], egs[2, k]
        i_ind, j_ind = (i - 1) * 3, (j - 1) * 3
        # Compute displacement with periodic adjustments
        dx = points[1 + j_ind] + iis[1, k] - points[1 + i_ind]
        dy = points[2 + j_ind] + iis[2, k] - points[2 + i_ind]
        dz = points[3 + j_ind] + iis[3, k] - points[3 + i_ind]
        # Transform displacement into basis coordinates
        r_local_x = basis[1, 1] * dx + basis[1, 2] * dy + basis[1, 3] * dz
        r_local_y = basis[2, 1] * dx + basis[2, 2] * dy + basis[2, 3] * dz
        r_local_z = basis[3, 1] * dx + basis[3, 2] * dy + basis[3, 3] * dz
        L = √(r_local_x^2 + r_local_y^2 + r_local_z^2)
        # Compute unit vector
        u_array[1, k] = r_local_x / L
        u_array[2, k] = r_local_y / L
        u_array[3, k] = r_local_z / L
        u_array[4, k] = L
    end
    for k in 1:ne
        i, j = egs[1, k], egs[2, k]
        i_ind, j_ind = (i - 1) * 3, (j - 1) * 3
        # Compute local Hessian
        u = u_array[1:3, k]
        L = u_array[4, k]
        H_local = (youngs[k] / rls[k]) * (I3 + (rls[k] / L) * (u * transpose(u) - I3))
        # Transform to global coordinates
        H_block = basis * H_local * bt
        # Scatter into pre-allocated Hessian
        for α in 1:3, β in 1:3
            H[i_ind + α, i_ind + β] += H_block[α, β]
            H[j_ind + α, j_ind + β] += H_block[α, β]
            H[i_ind + α, j_ind + β] -= H_block[α, β]
            H[j_ind + α, i_ind + β] -= H_block[α, β]
        end
    end
end

"""
    energy_hessian(net::Network) → Matrix{Float64}

Computes the Hessian matrix of the elastic energy in the elastic network `net`, representing second derivatives of the energy with respect to node positions.

# Arguments
- `net::Network` : An elastic network

# Returns
- `Matrix{Float64}` : The elastic energy Hessian matrix for `net`

"""
function energy_hessian(net)
    n = length(net.points)
    H = zeros(n, n)
    hessian!(H, net_info_primitive(net)...)
    return H
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
"""
    add_edge!(net::Network, s::Int, d::Int, rl::Float64, y::Float64 = 1.0) → Nothing

Adds an undirected harmonic spring edge to the elastic network `net` between nodes `s` (source) and `d` (destination), with specified rest length `rl` and optional Young’s modulus `y`.

Although the edge is physically symmetric, the `(s, d)` convention provides directional consistency in the data structure and indexing.

# Arguments
- `net::Network` : An elastic network
- `s::Int` : Source node index
- `d::Int` : Destination node index
- `rl::Float64` : Rest length of the edge
- `y::Float64 = 1.0` : Young’s modulus of the edge (default = 1.0)

# Returns
- `Nothing` : Modifies `net` in place

"""
function add_edge!(net::Network, s::Int, d::Int, rl::Float64, y = 1.0)
    trues = min(s, d)
    trued = max(s, d)
    Graphs.add_edge!(net.g, trues, trued)
    e = Edge(trues, trued)
    net.rest_lengths[e] = rl
    net.image_info[e] = get_image_info(net.points[:, trues], net.points[:, trued])
    net.youngs[e] = y
end

"""
    rem_edge!(net::Network, s::Int, d::Int) → Nothing

Removes an edge from the elastic network `net` between nodes `s` (source) and `d` (destination).

This operation deletes the edge from the underlying graph and also removes associated entries from the network’s dictionaries: rest lengths, periodic image information, and Young’s moduli.

# Arguments
- `net::Network` : An elastic network
- `s::Int` : Source node index
- `d::Int` : Destination node index

# Returns
- `Nothing` : Modifies `net` in place

"""
function rem_edge!(net::Network, s::Int, d::Int)
    Graphs.rem_edge!(net.g, min(s, d), max(s, d))
    e = Edge(min(s, d), max(s, d))
    pop!(net.rest_lengths, e)
    pop!(net.image_info, e)
    pop!(net.youngs, e)
end

"""
    rem_vertex!(net::Network, v::Int) → Nothing

Removes a vertex from the elastic network `net`, updating node positions and edge data to reflect internal graph indexing changes.

In `Graphs.jl`, removing a vertex causes the last vertex to be moved into its place. This function adjusts the `net.points` array and updates associated edge dictionaries—`rest_lengths`, `image_info`, and `youngs`—to ensure consistency with the reshuffled indices.

# Arguments
- `net::Network` : An elastic network
- `v::Int` : Index of the vertex to remove

# Returns
- `Nothing` : Modifies `net` in place

"""
function rem_vertex!(net::Network, v::Int)
    original_n = nv(net.g)
    original_edges = deepcopy(edges(net.g))
    Graphs.rem_vertex!(net.g, v) #Graphs.jl moves the last vertex to index v, so there are now just n - 1 vertices.
    if v ≠ original_n
        net.points = hcat(net.points[:, 1:v - 1], net.points[:, original_n], net.points[:, v + 1: original_n - 1])
    else
        net.points = net.points[:, 1:v - 1]
    end
    for e in original_edges
        s, d = src(e), dst(e)
        if s == v || d == v
            delete!(net.rest_lengths, e)
            delete!(net.image_info, e)
            delete!(net.youngs, e)
            continue
        end
        if s == original_n || d == original_n
            rl = pop!(net.rest_lengths, e)
            ii = pop!(net.image_info, e)
            y = pop!(net.youngs, e)
            if s == original_n
                new_s, new_d = min(v, d), max(v, d)
                iisign = new_d == d ? 1 : -1
            else
                new_s, new_d = min(s, v), max(s, v)
                iisign = new_s == s ? 1 : -1 
            end
            net.rest_lengths[Edge(new_s, new_d)] = rl
            net.image_info[Edge(new_s, new_d)] = iisign*ii
            net.youngs[Edge(new_s, new_d)] = y
        end
    end
end

"""
    pluck_out_edge!(net::Network, e::Graphs.SimpleGraphs.SimpleEdge, direction::Function) → Nothing

Replaces an edge `e` in the elastic network `net` by duplicating one of its endpoints and reconnecting the edge to the new node.

This transformation detaches one node of the edge (chosen via the `direction` function) and moves it to a freshly added vertex with identical coordinates. The original edge is removed, and a new edge—retaining the same rest length and Young’s modulus—is added between the anchoring node and the duplicated node. This structurally alters the network by "plucking out" one endpoint, allowing localized rewiring or boundary conditioning.

# Arguments
- `net::Network` : An elastic network
- `e::Edge` : Edge to be replaced
- `direction::Function` : A function (`src` or `dst`) specifying which node to duplicate

# Returns
- `Nothing` : Modifies `net` in place

"""
function pluck_out_edge!(net::Network, e::Graphs.SimpleGraphs.SimpleEdge, direction::Function)
    affected_node = direction(e)
    anchoring = setdiff([src, dst], [direction])[1]
    anchoring_node = anchoring(e)
    affected_coord = net.points[:, affected_node]
    add_vertex!(net.g)
    net.points = hcat(net.points, affected_coord)
    add_edge!(net, anchoring_node, nv(net.g), net.rest_lengths[e], net.youngs[e])
    rem_edge!(net, src(e), dst(e))
end

"""
    plug_in_edge!(net::Network, deg1node::Int, accepting_node::Int) → Nothing

Merges a dangling node into an existing node in the elastic network `net` by removing the temporary degree-1 node (`deg1node`) and attaching its edge to `accepting_node`.

This operation:
- Verifies that `deg1node` has exactly one neighbor (i.e. was previously created via `pluck_out_edge!`).
- Recreates the edge between the anchoring node and `accepting_node`, using the original rest length and Young’s modulus.
- Removes `deg1node`, effectively merging its connection into `accepting_node`.

# Arguments
- `net::Network` : An elastic network
- `deg1node::Int` : Index of the dangling node to merge
- `accepting_node::Int` : Index of the node receiving the merged edge

# Returns
- `Nothing` : Modifies `net` in place

"""
function plug_in_edge!(net::Network, deg1node::Int, accepting_node::Int)
    if degree(net.g)[deg1node] ≠ 1
        return nothing
    end
    anchoring_node = neighbors(net.g, deg1node)[1] 
    if anchoring_node ≠ accepting_node
        e = Edge(deg1node, anchoring_node)
        if !(e in keys(net.rest_lengths))
            e = Edge(anchoring_node, deg1node)
        end
        add_edge!(net, anchoring_node, accepting_node, net.rest_lengths[e], net.youngs[e])
        rem_vertex!(net, deg1node)
    end
end

"""
    simplify_net!(net::Network) → Nothing

Simplifies the elastic network `net` by iteratively pruning geometrically and topologically unstable nodes that may contribute soft modes.

This function:
- Removes nodes with degree 0 or 1, which are disconnected or dangling.
- Removes degree-2 nodes whose neighbor vectors form a bending angle below 179.99°, as these configurations introduce floppy or under-constrained regions.
- Repeats the simplification until no further changes occur.

# Arguments
- `net::Network` : An elastic network

# Returns
- `Nothing` : Modifies `net` in place

"""
function simplify_net!(net::Network)
    nv_old = nv(net.g)
    nv_new = nv_old - 1
    while nv_old > nv_new
        i = 1
        while i ≤ nv(net.g)
            deg_i = degree(net.g, i)
            if deg_i in [0, 1]
                rem_vertex!(net, i)
            elseif deg_i == 2
                nbs = neighbors(net.g, i)
                v1 = min_image_vector_rel(net.points[:, i], net.points[:, nbs[1]])
                v2 = min_image_vector_rel(net.points[:, i], net.points[:, nbs[2]])
                angle = abs(acosd(clamp(v1⋅v2/(norm(v1)*norm(v2)), -1, 1)))
                if angle < 179.99
                    rem_vertex!(net, i)
                else
                    i += 1
                end
            else
                i += 1
            end
        end
        nv_old = nv_new
        nv_new = nv(net.g)
    end
end

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
