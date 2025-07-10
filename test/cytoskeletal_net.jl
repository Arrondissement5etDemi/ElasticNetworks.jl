include("../src/elastic_network.jl")

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
        youngs[e] = 1.0
    end
    return Network(g, basis, points, rest_lengths, image_info, youngs)
end

function cytoskeleton_net()
    return load_network("test/threshold0x001_conc0.5_maxrl3_epsilon0x05_17500.jld2")
end