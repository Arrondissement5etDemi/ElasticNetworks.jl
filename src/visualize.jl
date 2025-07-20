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
    recenter!(net)
    es = collect(edges(net.g))
    ps = net.points
    set_theme!(backgroundcolor = :gray)
    f = Figure(size = (600, 600))
    tens = tensions(net)
    tension_range = max(maximum(tens), 1e-5)*2
    tens /= tension_range
    data = []
    tension_data = zeros(0)
    long_diagonal = [sum(col) for col in eachcol(net.basis)]
    Axis3(f[1, 1], limits = (0, long_diagonal[1], 0, long_diagonal[2], 0, long_diagonal[3]), aspect = (long_diagonal[1], long_diagonal[2], long_diagonal[3]))
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

function visualize_eigenmode(net::Network, lvl::Int)
    f = visualize_net(net)
    if lvl ≤ 3
        throw(DomainError(lvl, "lvl <= 3 is a translational mode"))
    end
    h = energy_hessian(net)
    modes = eigvecs(h)
    mode = reshape(modes[:, lvl], size(net.points))
    ps = []
    ns = []
    for i in 1:nv(net.g)
        push!(ps, Point3d(net.basis*net.points[:, i]))
        push!(ns, Point3d(net.basis*mode[:, i]))
    end
    lengths = norm.(ns)
    arrows3d!(ps, ns, color = lengths, lengthscale = 2)
    return f
end

function recenter!(net::Network)
    self_images = Int.(fld.(net.points, 1))
    net.points = mod.(net.points, 1)
    for e in edges(net.g)
        net.image_info[e] += (self_images[:, dst(e)] - self_images[:, src(e)]) 
    end
end