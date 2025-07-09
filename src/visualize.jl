include("elastic_network.jl")

function visualize_net(net::Network, edgecolors = "tension")
    es = collect(edges(net.g))
    ps = net.points
    f = Figure(size = (600, 600))
    tens = tensions(net)
    tension_range = mean(tensions(net))
    data = []
    Axis3(f[1, 1], limits = (0, 9.4766, 0, 9.4766, 0, 9.4766), aspect = (1, 1, 1))
    for i in eachindex(es)
        e = es[i]
        v1, v2 = src(e), dst(e)
        p1 = Point3d(net.basis*ps[:, v1])
        p2 = Point3d(net.basis*(ps[:, v2] + net.image_info[e]))
        push!(data, (p1, p2))
    end
    linesegments!(f[1, 1], data, fxaa = false, color = tens/(tension_range*2)*10, colorrange = (1, 15), linewidth = 3, colormap = :sunset)
    return f
end
#=ps = rand(Point3f, 500)
cs = rand(500)
f = Figure(size = (600, 650))
Label(f[1, 1], "base", tellwidth = false)
linesegments(f[2, 1], ps, color = cs, fxaa = false)
f=#