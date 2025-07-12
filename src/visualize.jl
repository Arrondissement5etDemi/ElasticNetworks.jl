function visualize_net(net::Network, edgecolors = "tension")
    es = collect(edges(net.g))
    ps = net.points
    set_theme!(backgroundcolor = :gray)
    f = Figure(size = (600, 600))
    tens = tensions(net)
    tension_range = mean(tensions(net))
    data = []
    tension_data = zeros(0)
    Axis3(f[1, 1], limits = (0, net.basis[1,1], 0, net.basis[2,2], 0, net.basis[3,3]), aspect = (1, 1, 1))
    for i in eachindex(es)
        e = es[i]
        v1, v2 = src(e), dst(e)
        p1 = Point3d(net.basis*ps[:, v1])
        p2 = Point3d(net.basis*(ps[:, v2] + net.image_info[e]))
        push!(data, (p1, p2))
        push!(tension_data, tens[i])
        if net.image_info[e] ≠ [0, 0, 0]
            p3 = Point3d(net.basis*ps[:, v2])
            p4 = Point3d(net.basis*(ps[:, v1] - net.image_info[e]))
            push!(data, (p3, p4))
            push!(tension_data, tens[i])
        end
    end
    linesegments!(f[1, 1], data, fxaa = false, color = 10 .- tension_data*10, colormap = :sunset, linewidth = 3)
    return f
end