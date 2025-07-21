using ElasticNetworks, Statistics, Random, Graphs, LinearAlgebra, Combinatorics

function min_ei_pruning!(net::Network, θ::Float64 = 0.001, randrate::Float64 = 1.0, catchprob::Float64 = 1.0)
    catch_count, rand_count = 0, 0
    edges = shuffle(collect(Graphs.edges(net.g)))
    max_rand = rand() < randrate % 1 ? Int(ceil(randrate)) : Int(floor(randrate))
    for e in edges
        i, j = src(e), dst(e)
        if degree(net.g, i) ≥ 1 && degree(net.g, i) ≥ 1
            if rand_count < max_rand
                ElasticNetworks.rem_edge!(net, i, j)
                rand_count += 1
            else
                forceij = norm(ElasticNetworks.force(net, i, j))
                if forceij < θ && rand() < catchprob #0.001
                    ElasticNetworks.rem_edge!(net, i, j)
                    catch_count += 1
                end
            end
        end
        if catch_count + rand_count ≥ 50
            break
        end
    end
    return catch_count, rand_count
end

function edge_midpoints(net::Network)
    result = Dict{Graphs.SimpleGraphs.SimpleEdge{Int64}, Vector{Float64}}()
    for e in edges(net.g)
        i, j = src(e), dst(e)
        result[e] = mod.((net.points[:, i] + net.points[:, j] + net.image_info[e])/2, 1)
    end
    return result
end

function sterics_score(net::Network, em, p, steric_rad)
    result = 0
    for midpoint in values(em)
        dist_pk = ElasticNetworks.euclidean_dist(net, p, midpoint)
        result += (dist_pk < steric_rad ? 1 : 0)
    end
    return result
end

function nucleate(net::Network, em, steric_rad)
    result = rand(3)
    ss_min = 100
    for _ = 1:30
        nucleus_proposed = rand(3)
        ss = sterics_score(net, em, nucleus_proposed, steric_rad)
        if ss < ss_min
            ss_min = ss
            result = nucleus_proposed
        end
    end
    return result
end

function closest_edge_center(net::Network, em, p::Vector{Float64})
    min_dist = 1000
    result = Edge(1, 2)
    for e in keys(em)
        dist_pm = ElasticNetworks.euclidean_dist(net, p, em[e])
        if dist_pm < min_dist
            min_dist = dist_pm
            result = e
        end
    end
    return result
end

function hop_zeros!(net, em)
    hop_count = 0
    zero_nodes = filter(i -> degree(net.g, i) == 0, 1:nv(net.g))
    for i in zero_nodes
        e = closest_edge_center(net, em, net.points[:, i])
        j1, j2 = src(e), dst(e)
        net.points[:, i] = em[e]
        ElasticNetworks.add_edge!(net, min(j1, i), max(j1, i), net.rest_lengths[e]/2)
        ElasticNetworks.add_edge!(net, min(j2, i), max(j2, i), net.rest_lengths[e]/2)
        ElasticNetworks.rem_edge!(net, j1, j2)
        pop!(em, e)
        hop_count += 1
    end
    return hop_count
end

function bubble_edge_addition!(net::Network, steric_rad, seed_conc, max_rl, ϵ)
    n_active = round(Int, seed_conc*det(net.basis))
    degrees = degree(net.g)
    available_nodes = filter(i -> degrees[i] < 4, 1:nv(net.g))
    add_count = 0
    em = edge_midpoints(net)
    for _ in 1:n_active
        nucleus = nucleate(net, em, steric_rad)
        nodes_in_bubble = Vector{Int}()
        for node in available_nodes
            d_node_nuc = ElasticNetworks.euclidean_dist(net, nucleus, net.points[:, node])
            if d_node_nuc < max_rl/(1 - ϵ)/2
                push!(nodes_in_bubble, node)
            end
        end
        if length(nodes_in_bubble) ≥ 2
            pairs = shuffle(collect(combinations(nodes_in_bubble, 2)))
            for pair in pairs
                i, j = pair
                if !(i in neighbors(net.g, j)) 
                    dij = ElasticNetworks.euclidean_dist(net, i, j)
                    ElasticNetworks.add_edge!(net, i, j, dij/(1 + ϵ))
                    add_count += 1
                    degrees[i] += 1
                    degrees[j] += 1
                    for k in pair
                        if degrees[k] ≥ 4
                            deleteat!(available_nodes, findfirst(x -> x == k, available_nodes))
                        end
                    end
                    break
                end
            end
        end
    end
    return add_count
end

function evolve_w_hops(net, t_range, θ, rand_rate, steric_rad, seed_conc, max_rl, ϵ, file_tag)
    relax!(net)
    for t in t_range
        if mod(t, 50) == 0
            save_network(net, file_tag * "$t.jld2")
        end
        #changes to the network happen here
        catch_count, rand_count = min_ei_pruning!(net, θ, rand_rate)
        em = edge_midpoints(net)
        hop_count = hop_zeros!(net, em)
        add_count = bubble_edge_addition!(net, steric_rad, seed_conc, max_rl, ϵ)
        relax!(net)
        ElasticNetworks.recenter!(net)
        #end of changes
        print("$t $(ne(net.g)) $(mean_degree(net)) $(elastic_energy(net)) $catch_count $rand_count $hop_count $add_count \n")
        flush(stdout)
        GC.gc()
    end
end

#=net = load_network("data/edge_model_sheared_50.jld2")
evolve_w_hops(net, 50:20000, 0.001, 1.0, 1.0, 0.5, 3.0, 0.05, "research/edge_model_configs/")=#


net = load_network("data/threshold0x001_conc0.5_maxrl3_epsilon0x05_17500.jld2")
net.basis *= ([0 0.1 0; 0 0 0; 0 0 0] + I)
relax!(net, show_trace = true)
evolve_w_hops(net, 0:20000, 0.001, 1.0, 1.0, 0.5, 3.0, 0.05, "research/edge_model_configs/")
