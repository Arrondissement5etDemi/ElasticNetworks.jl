"""
    visualize_net(net::Network) → Figure

Generates a 3D visualization of the elastic network `net`, displaying nodes and edges embedded in the simulation cell defined by `net.basis`.

Edges are colored by normalized tension (default), derived from the product of strain and Young’s modulus. Periodic image contributions are shown by mirrored edge segments when applicable.

# Arguments
- `net::Network` : An elastic network to visualize

# Returns
- `Figure` : A Makie figure object showing the elastic network in 3D

"""
function visualize_net(net::Network)
    es = collect(edges(net.g))
    ps = net.points
    set_theme!(backgroundcolor = :gray)
    f = Figure(size = (600, 600))
    tens = tensions(net)
    tension_range = maximum(tens)
    tens /= tension_range
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