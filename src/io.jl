"""
    net_info_primitive(net::Network)

Converts network data into numerical matrices containing only primitive types, enabling efficient SIMD (Single Instruction, Multiple Data) operations if desired.

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

"""
    save_network(net::Network, filename::String) → Nothing

Saves the elastic network `net` to a `.jld2` file containing the core structural and mechanical data.

The output file includes:
- `basis` : 3×3 matrix representing the periodic unit cell
- `nodes` : 3×N array of reduced node coordinates
- `edge_info` : M×7 array where each row stores:
    [src, dst, img_x, img_y, img_z, rest_length, Young's modulus]

This format is compatible with `load_network` for later retrieval or sharing.  
Ensure that `filename` ends with `.jld2` for consistency.

# Arguments
- `net::Network` : The elastic network to serialize
- `filename::String` : Path to the output file (must end in `.jld2`)

# Returns
- `Nothing` : Writes the specified `.jld2` file to disk

"""
function save_network(net::Network, filename::String)
    basis = net.basis
    nodes = net.points
    edge_info = zeros(0, 7)
    for e in edges(net.g)
        edge_info = vcat(edge_info, hcat([src(e), dst(e)]', net.image_info[e]', [net.rest_lengths[e], net.youngs[e]]'))
    end
    jldsave(filename; basis, nodes, edge_info)
end

"""
    load_network(filename::String) → Network

Loads an elastic network from a `.jld2` file containing precomputed graph geometry and edge attributes.

The input `filename.jld2` file, upon loading with `JLD2.load(filename)`, is expected to be a dictionary that contains:
- `"basis"` : A 3×3 matrix representing the unit cell basis
- `"nodes"` : A 3×N array of node coordinates relative to the basis
- `"edge_info"` : A matrix where each row describes an edge, including:
    - node indices,
    - image displacement vector,
    - rest length,
    - (optional) Young’s modulus

This function reconstructs the graph topology, assigns mechanical parameters, and builds the full `Network` object.

# Arguments
- `filename::String` : Path to the file containing serialized network data

# Returns
- `Network` : Reconstructed elastic network with geometry and physics

"""
function load_network(filename)
    b_data = load(filename)
    basis = b_data["basis"]
    points = b_data["nodes"]
    nv = size(points, 2)
    rest_lengths = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64}()
    image_info = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Vector{Int}}()
    youngs = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64}()
    g = SimpleGraph(nv)
    for row in eachrow(b_data["edge_info"])
        true_src, true_dst = sort(Int.(row[1:2]))
        Graphs.add_edge!(g, true_src, true_dst)
        e = Edge(true_src, true_dst)
        rest_lengths[e] = row[6]
        image_info[e] = sign(row[2] - row[1])*row[3:5]
        if length(row) == 6
            youngs[e] = 1.0
        else
            youngs[e] = row[7]
        end
    end
    return Network(g, basis, points, rest_lengths, image_info, youngs)
end

