using Roots, Statistics, StatsBase, Random
include("elastic_network.jl")

function diamond1000(l, rest_length, youngs)
    g = SimpleGraph(1000)
    basis = [l 0 0; 0 l 0; 0 0 l]
    points = zeros(3, 1000)
    rest_lengths = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Float64}()
    image_info = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Vector{Int}}() 
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
            add_edge!(g, i, j)
            e = Edge(i, j)
            rest_lengths[e] = rest_length
            image_info[e] = get_image_info(points[:, i], points[:, j])
        end
    end
    return Network(g, basis, points, rest_lengths, image_info, youngs)
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
    dnet = diamond1000(9.4766, 0.812571915, 1)
    for _ = 1:samples
        dsw = diamond_smallworld_degree_preserving(dnet, p, 0.01)
        relax!(dsw)
        match_mean_tension!(dsw, targ_tension)
        B, G = moduli(dsw)[1:2]
        println("$B $G $(maximum(values(tensions(dsw))))")
    end
end
