using Graphs, LoopVectorization, Optim, LinearAlgebra, Statistics, ForwardDiff
import Graphs: rem_vertex!, add_edge!, rem_edge!

quick_euclidean_graph(N::Int, cutoff) = euclidean_graph(N, 3; cutoff = cutoff, bc = :periodic)
quick_euclidean_graph(points::Matrix{Float64}, cutoff) = euclidean_graph(points; cutoff = cutoff, bc = :periodic)

mutable struct Network
    g :: SimpleGraph
    basis :: Matrix{Float64}
    points :: Matrix{Float64}
    rest_lengths :: Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64}
    image_info :: Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Vector{Int}}
    youngs :: Float64
end

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
    Graphs.rem_vertex!(net.g, v)
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

function euclidean_dist(net::Network, v1::Int, v2::Int)
    return norm(net.basis*min_image_vector_rel(net.points[:, v1], net.points[:, v2]))
end

"""creates an elastic network from an euclidean graph

     network_from_graph(euc_gra, l, ϵ, youngs = 1)
"""
function network_from_graph(euc_gra, l, ϵ, youngs = 1)
    g, edge_dict, points = euc_gra
    basis = [l 0 0; 0 l 0; 0 0 l]
    rest_lengths = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64}()
    image_info = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Vector{Int}}()
    for e in edges(g)
        rest_lengths[e] = edge_dict[e]*(1 - ϵ)*l
        image_info[e] = get_image_info(points[:, src(e)], points[:, dst(e)])
    end
    return Network(g, basis, points, rest_lengths, image_info, youngs)
end

"""creates numerical matrices for a network, i.e., only primitive types, so that SIMD algorithms can operate on the network

"""
function net_info_primitive(net)
    edge_nodes = zeros(Int, (0, 2))
    rls = zeros(0)
    iis = zeros(Int, (0, 3))
    for e in edges(net.g)
        edge_nodes = vcat(edge_nodes, [src(e) dst(e)])
        push!(rls, net.rest_lengths[e])
        iis = vcat(iis, net.image_info[e]')
    end
    return net.basis, collect(Iterators.flatten(net.points)), edge_nodes, rls, iis, net.youngs
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function tensions(net)
    result = deepcopy(net.rest_lengths)
    for k in keys(net.rest_lengths)
        result[k] = abs((euclidean_dist(net, k.src, k.dst) - net.rest_lengths[k])/net.rest_lengths[k])
    end
    return result
end

function elastic_energy(basis, points, edge_nodes, rls, iis, youngs)
    result = 0
    for k in axes(edge_nodes, 1)
        i = edge_nodes[k, 1]
        j = edge_nodes[k, 2]
        i_ind, j_ind = (i - 1)*3, (j - 1)*3#because points are flattened
        dx = points[1 + j_ind] + iis[k, 1] - points[1 + i_ind]
        dy = points[2 + j_ind] + iis[k, 2] - points[2 + i_ind]
        dz = points[3 + j_ind] + iis[k, 3] - points[3 + i_ind]
        r1 = dx*basis[1, 1] + basis[1, 2]*dy + basis[1, 3]*dz
        r2 = dx*basis[2, 1] + basis[2, 2]*dy + basis[2, 3]*dz
        r3 = dx*basis[3, 1] + basis[3, 2]*dy + basis[3, 3]*dz
        edge_length_ij = √(r1^2 + r2^2 + r3^2)
        edge_energy_ij = (rls[k] - edge_length_ij)^2/rls[k]
        result += edge_energy_ij
    end
    return youngs*result/2
end

"""computes the elastic energy of the network

"""
elastic_energy(net) = elastic_energy(net_info_primitive(net)...)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

function gradient!(result, basis, points, edge_nodes, rls, iis, youngs)
    for i in eachindex(result)
        result[i] = 0 #initialize
    end
    l = basis[1, 1]
    for k in axes(edge_nodes, 1)
        i = edge_nodes[k, 1]
        j = edge_nodes[k, 2]
        i_ind, j_ind = (i - 1)*3, (j - 1)*3#because points are flattened
        dx = points[1 + j_ind] + iis[k, 1] - points[1 + i_ind]
        dy = points[2 + j_ind] + iis[k, 2] - points[2 + i_ind]
        dz = points[3 + j_ind] + iis[k, 3] - points[3 + i_ind] 
        du = √(dx^2 + dy^2 + dz^2)
        factor = youngs*l*(1/du - l/rls[k])
        myx = factor*dx
        myy = factor*dy
        myz = factor*dz
        result[1 + i_ind] += myx
        result[2 + i_ind] += myy
        result[3 + i_ind] += myz
        result[1 + j_ind] -= myx
        result[2 + j_ind] -= myy
        result[3 + j_ind] -= myz
    end
end

function nonaffine_displacement(basis, points, edge_nodes, rls, iis, youngs, deformed_basis)
    G = similar(points)
    n_compo = length(points)
    H = zeros(n_compo, n_compo) #above are initializations
    gradient_notcube!(G, deformed_basis, points, edge_nodes, rls, iis, youngs)
    hessian!(H, basis, points, edge_nodes, rls, iis, youngs)
    return -(qr(H, Val(true)) \ G)
end

function gradient_notcube!(result, basis, points, edge_nodes, rls, iis, youngs)
    for i in eachindex(result)
        result[i] = 0 #initialize
    end
    n = (Int)(length(points)/3)
    forces = zeros(3, n)
    for k in axes(edge_nodes, 1)
        i, j = edge_nodes[k, 1], edge_nodes[k, 2]
        i_ind, j_ind = (i - 1)*3, (j - 1)*3 #because points are flattened
        dx = points[1 + j_ind] + iis[k, 1] - points[1 + i_ind]
        dy = points[2 + j_ind] + iis[k, 2] - points[2 + i_ind]
        dz = points[3 + j_ind] + iis[k, 3] - points[3 + i_ind]
        r1 = basis[1, 1]*dx + basis[1, 2]*dy + basis[1, 3]*dz
        r2 = basis[2, 1]*dx + basis[2, 2]*dy + basis[2, 3]*dz
        r3 = basis[3, 1]*dx + basis[3, 2]*dy + basis[3, 3]*dz
        edge_length_ij = √(r1^2 + r2^2 + r3^2)
        factor = youngs*(edge_length_ij - rls[k])/rls[k]/edge_length_ij 
        forces[1, i] -= factor*r1
        forces[2, i] -= factor*r2
        forces[3, i] -= factor*r3
        forces[1, j] += factor*r1
        forces[2, j] += factor*r2
        forces[3, j] += factor*r3
    end
    res = basis*forces
    for i in eachindex(result)
        result[i] = res[i]
    end
end


function hessian!(result, basis, points, edge_nodes, rls, iis, youngs) #analytical hessian computed using mathematica
    for i in eachindex(result)
        result[i] = 0 #initialize
    end
    l = basis[1, 1]
    for k in axes(edge_nodes, 1)
        i = edge_nodes[k, 1]
        j = edge_nodes[k, 2]
        i_ind, j_ind = (i - 1)*3, (j - 1)*3#because points are flattened
        dx = points[1 + j_ind] + iis[k, 1] - points[1 + i_ind]
        dy = points[2 + j_ind] + iis[k, 2] - points[2 + i_ind]
        dz = points[3 + j_ind] + iis[k, 3] - points[3 + i_ind]
        du = √(dx^2 + dy^2 + dz^2)
        factor = youngs*l/du^3
        constant = youngs*l^2/rls[k]
        my12 = -dx*dy*factor
        my13 = -dx*dz*factor
        my23 = -dy*dz*factor
        #off-diagonal block ij
        result[i_ind + 1, j_ind + 1] = factor*(du^2 - dx^2) - constant
        result[i_ind + 1, j_ind + 2] = my12
        result[i_ind + 1, j_ind + 3] = my13
        result[i_ind + 2, j_ind + 2] = factor*(du^2 - dy^2) - constant
        result[i_ind + 2, j_ind + 1] = my12
        result[i_ind + 2, j_ind + 3] = my23
        result[i_ind + 3, j_ind + 3] = factor*(du^2 - dz^2) - constant
        result[i_ind + 3, j_ind + 1] = my13
        result[i_ind + 3, j_ind + 2] = my23
        #symmetric of the block above
        result[j_ind + 1, i_ind + 1] = factor*(du^2 - dx^2) - constant
        result[j_ind + 2, i_ind + 1] = my12
        result[j_ind + 3, i_ind + 1] = my13
        result[j_ind + 2, i_ind + 2] = factor*(du^2 - dy^2) - constant
        result[j_ind + 1, i_ind + 2] = my12
        result[j_ind + 3, i_ind + 2] = my23
        result[j_ind + 3, i_ind + 3] = factor*(du^2 - dz^2) - constant
        result[j_ind + 1, i_ind + 3] = my13
        result[j_ind + 2, i_ind + 3] = my23
    end
    #diagonal block for i and j
    @inbounds @fastmath for edge in eachrow(edge_nodes)
        i_ind = (edge[1] - 1)*3
        j_ind = (edge[2] - 1)*3
        result[i_ind + 1:i_ind + 3, i_ind + 1:i_ind + 3] -= result[i_ind + 1:i_ind + 3, j_ind + 1:j_ind + 3]
        result[j_ind + 1:j_ind + 3, j_ind + 1:j_ind + 3] -= result[i_ind + 1:i_ind + 3, j_ind + 1:j_ind + 3]
    end
end

elastic_energy(net) = elastic_energy(net_info_primitive(net)...)

min_direction(x) = findmin(abs.([x - 1, x, x + 1]))[2] - 2

function get_image_info(src, dst)
    return min_direction.(dst - src)
end

function min_image_vector_rel(v1, v2)
    vec_diff = v2 - v1
    min_directions = min_direction.(vec_diff)
    return vec_diff + min_directions
end
