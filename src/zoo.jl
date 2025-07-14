using Roots, Statistics, StatsBase, Random



"""
    prestrained_network(g, basis, points, ϵ, default_youngs=1.0) → Network

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

"""
    diamond1000(l::Float64, ϵ::Float64) → Network

Constructs a 1000-node elastic network based on a diamond lattice within a cubic periodic cell of side length `l`.

This synthetic crystal-like geometry includes eight points per unit cell arranged to approximate tetrahedral bonding motifs. Nodes are placed within a 5×5×5 grid of unit cells, and edges connect nearest neighbors using a minimum image criterion. The returned network includes prestrain `ϵ` applied uniformly to all edges, enabling perturbative analysis or tuning of mechanical response.

# Arguments
- `l::Float64` : Side length of the periodic simulation box
- `ϵ::Float64` : Uniform prestrain factor applied to all edges

# Returns
- `Network` : A prestrained elastic network with diamond-like connectivity

"""
function diamond1000(l, ϵ)
    g = SimpleGraph(1000)
    basis = [l 0 0; 0 l 0; 0 0 l]
    points = zeros(3, 1000)
    for i = 0:4, j = 0:4, k = 0:4
        ind = i*25 + j*5 + k + 1
        corner = [i, j, k]
        points[:, ind] = corner
        points[:, ind + 125] = corner + [1/2, 1/2, 0]
        points[:, ind + 250] = corner + [1/2, 0, 1/2]
        points[:, ind + 375] = corner + [0, 1/2, 1/2]
        points[:, ind + 500] = corner + [1/4, 1/4, 1/4]
        points[:, ind + 625] = corner + [3/4, 3/4, 1/4]
        points[:, ind + 750] = corner + [1/4, 3/4, 3/4]
        points[:, ind + 875] = corner + [3/4, 1/4, 3/4]
    end
    points /= 5
    nnd = √3/20 #nearest neighbor distance
    for i in 1:1000, j in i + 1:1000
        dij = min_image_vector_rel(points[:, i], points[:, j])
        if norm(dij) - nnd ≤ 1e-4
            Graphs.add_edge!(g, i, j)
        end
    end
    return prestrained_network(g, basis, points, ϵ)
end

"""
    cubic_network(l::Float64, n_layers::Int, ϵ::Float64 = 0.0) → Network

Constructs a periodic elastic network based on a cubic lattice of `n_layers × n_layers × n_layers` nodes within a simulation cell of side length `l`.

Nodes are evenly spaced along integer grid points, and edges are added between nearest neighbors using a minimum image distance check. The entire network is embedded in a unit cube scaled by `l`, and a uniform prestrain `ϵ` is applied to all edges to model mechanical perturbations or stress conditioning.

# Arguments
- `l::Float64` : Side length of the periodic simulation box
- `n_layers::Int` : Number of lattice layers along each axis
- `ϵ::Float64 = 0.0` : Uniform prestrain factor applied to all edges (default = 0)

# Returns
- `Network` : A prestrained cubic elastic network

"""
function cubic_network(l, n_layers::Int, ϵ = 0)
    n = n_layers^3
    g = SimpleGraph(n)
    basis = [l 0 0; 0 l 0; 0 0 l]
    points = zeros(3, n_layers^3)
    for i = 0:n_layers - 1, j = 0:n_layers - 1, k = 0:n_layers - 1
        ind = i*n_layers^2 + j*n_layers + k + 1
        points[:, ind] = [i, j, k]/n_layers
    end
    nnd = 1/n_layers
    for i in 1:n, j in i + 1:n
        dij = min_image_vector_rel(points[:, i], points[:, j])
        if abs(norm(dij) - nnd) ≤ 1e-4
            Graphs.add_edge!(g, i, j)
        end
    end
    return prestrained_network(g, basis, points, ϵ)
end

"""
    disordered_cubic_network(l::Float64, n_layers::Int, disorder_param::Float64, ϵ::Float64) → Network

Generates a disordered elastic network by perturbing a regular cubic lattice and recalculating rest lengths to incorporate both geometric disorder and prestrain.

This function starts from a uniform cubic network and introduces positional disorder to each node by adding random displacements scaled by `disorder_param`. It then updates the edge rest lengths based on the new geometry, applying a uniform prestrain `ϵ`. The network is relaxed afterward to remove spurious stresses and achieve mechanical consistency.

# Arguments
- `l::Float64` : Side length of the periodic simulation box
- `n_layers::Int` : Number of lattice layers along each axis
- `disorder_param::Float64` : Magnitude of positional disorder relative to unit cell size
- `ϵ::Float64` : Uniform prestrain factor applied to edge rest lengths

# Returns
- `Network` : A relaxed elastic network with cubic topology and geometric disorder

"""
function disordered_cubic_network(l, n_layers, disorder_param, ϵ)
    result = cubic_network(l, n_layers, ϵ)
    unit_cell_length = 1/n_layers
    for i in axes(result.points, 2)
        result.points[:, i] += rand(3)*unit_cell_length*disorder_param
    end
    for e in edges(result.g)
        s, d = src(e), dst(e)
        result.rest_lengths[e] = norm(result.basis*(result.points[:, d] - result.points[:, s] + result.image_info[e]))*(1 - ϵ)
    end
    relax!(result)
    return result
end

"""
    er(l::Float64, points::Matrix{Float64}, z::Float64, ϵ::Float64) → Network

Generates a randomized elastic network (Erdős–Rényi-style) embedded in a periodic cube of side length `l`, with connectivity tuned to achieve an average degree `z`.

Starting from a graph with no edges, the function randomly samples pairs of distinct nodes from the provided `points`, adding edges until the network reaches the target mean degree. Each added edge obeys a distance cutoff (`< l/2`) and is assigned a rest length reduced by the prestrain factor `ϵ`.

# Arguments
- `l::Float64` : Side length of the periodic simulation box
- `points::Matrix{Float64}` : Node coordinates in reduced (unit-cell) space
- `z::Float64` : Target average degree for the network
- `ϵ::Float64` : Uniform prestrain factor applied to edge rest lengths

# Returns
- `Network` : A randomized elastic network with approximately degree `z`

"""
function er(l, points, z, ϵ)
    basis = I(3)*l
    nv = size(points, 2)
    g = SimpleGraph(nv, 0)
    z_now = 0
    rest_lengths = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64}()
    image_info = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Vector{Int}}()
    youngs = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64}()
    result = Network(g, basis, points, rest_lengths, image_info, youngs)
    while z_now < z
        n1 = rand(1:nv)
        n2 = rand(setdiff(1:nv, [n1]))
        n1, n2 = min(n1, n2), max(n1, n2)
        v1 = points[:, n1]
        v2 = points[:, n2]
        rl = norm(basis*min_image_vector_rel(v1, v2))*(1 - ϵ)
        if rl < l/2
            add_edge!(result, n1, n2, rl)
        end
        z_now = mean_degree(result)
    end
    return result
end

function diamond_smallworld(diamond_base::Network, p::Float64, ϵ::Float64, α::Float64 = 3.0)
    result = deepcopy(diamond_base)
    diamond_copy = deepcopy(diamond_base)
    diam_edges = collect(keys(diamond_base.rest_lengths))
    g = diamond_base.g
    inds = shuffle(1:nv(g))
    for ii in eachindex(inds)
        i = inds[ii]
        shortests = dijkstra_shortest_paths(diamond_base.g, [i]).dists
        for jj in ii + 1:nv(g)
            j = inds[jj]
            path_length = shortests[j]
            if path_length >= 2 && rand() < p/path_length^α
                add_edge!(result, i, j, euclidean_dist(diamond_base, i, j)*(1 - ϵ))
                println("$i $j")
            end
        end
    end
    return result
end

function diamond_smallworld_degree_preserving(diamond_base::Network, p::Float64, ϵ::Float64, α::Float64 = 3.0)
    result = deepcopy(diamond_base)
    diamond_copy = deepcopy(diamond_base)
    diam_edges = collect(keys(diamond_base.rest_lengths))
    g = diamond_base.g
    inds = shuffle(1:nv(g))
    defect_count = 0
    for ii in eachindex(inds)
        i = inds[ii]
        shortests = dijkstra_shortest_paths(diamond_base.g, [i]).dists
        for jj in ii + 1:nv(g)
            j = inds[jj]
            path_length = shortests[j]
            if path_length >= 2 && rand() < p/path_length^α
                if rand() < 0.5
                    temp = i
                    i = j
                    j = temp
                end
                i_nbs = neighbors(diamond_copy.g, i)
                if length(i_nbs) > 0
                    add_edge!(result, i, j, euclidean_dist(diamond_base, i, j)*(1 - ϵ))
                    i_nb = rand(i_nbs)
                    rem_edge!(result, i, i_nb)
                    rem_edge!(diamond_copy, i, i_nb)
                    defect_count += 1
                end
            end
        end
    end
    println(defect_count)
    return result
end

function diamond_smallworld_degree_preserving2(diamond_base::Network, p::Float64, ϵ::Float64, α::Float64 = 3.0)
    result = deepcopy(diamond_base)
    diamond_copy = deepcopy(diamond_base)
    diam_edges = collect(keys(diamond_base.rest_lengths))
    g = diamond_base.g
    inds = shuffle(1:nv(g))
    for ii in eachindex(inds)
        i = inds[ii]
        shortests = dijkstra_shortest_paths(diamond_base.g, [i]).dists
        for jj in ii + 1:nv(g)
            j = inds[jj]
            path_length = shortests[j]
            if path_length >= 2 && rand() < p/path_length^α
                if rand() < 0.5
                    temp = i
                    i = j
                    j = temp
                end
                i_nbs = neighbors(diamond_copy.g, i)
                j_nbs = neighbors(diamond_copy.g, j)
                if length(i_nbs) > 0 && length(j_nbs) > 0
                    add_edge!(result, i, j, euclidean_dist(diamond_base, i, j)*(1 - ϵ))
                    i_nb = rand(i_nbs)
                    j_nb = rand(setdiff(j_nbs, [i_nb]))
                    add_edge!(result, i_nb, j_nb, euclidean_dist(diamond_base, i_nb, j_nb)*(1 - ϵ))
                    rem_edge!(result, i, i_nb)
                    rem_edge!(diamond_copy, i, i_nb)
                    rem_edge!(result, j, j_nb)
                    rem_edge!(diamond_copy, j, j_nb)
                end
            end
        end
    end
    return result
end

function match_mean_tension!(net::Network, targ_tension::Float64)
    ori_net = deepcopy(net)
    function f(factor)
        for e in edges(net.g)
            net.rest_lengths[e] = ori_net.rest_lengths[e]*factor
        end
        relax!(net)
        return mean(values(tensions(net))) - targ_tension
    end
    opt_fac = find_zero(f, (0.99, 1.01), Bisection())
    for e in edges(net.g)
        net.rest_lengths[e] = ori_net.rest_lengths[e]*opt_fac
    end 
end

function survey_dsw(p::Float64, targ_tension::Float64, samples::Int)
    dnet = diamond1000(9.4766, 0.812571915)
    for _ = 1:samples
        dsw = diamond_smallworld_degree_preserving(dnet, p, 0.01)
        relax!(dsw)
        match_mean_tension!(dsw, targ_tension)
        B, G = moduli(dsw)[1:2]
        println("$B $G $(maximum(values(tensions(dsw))))")
    end
end


