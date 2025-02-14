include("elastic_network.jl")
function main()
euc_gra = quick_euclidean_graph(1000, 0.13)
net = network_from_graph(euc_gra, 10, 0.1)
end

function moduli(net)
    relax!(net)
    basis, points, edge_nodes, rls, iis, youngs = net_info_primitive(net)
    deformed_bases = Dict{String, Function}()
    deformed_bases["1111"] = ϵ -> ([ϵ 0 0; 0 0 0; 0 0 0] + I)*basis
    deformed_bases["2222"] = ϵ -> ([0 0 0; 0 ϵ 0; 0 0 0] + I)*basis
    deformed_bases["3333"] = ϵ -> ([0 0 0; 0 0 0; 0 0 ϵ] + I)*basis
    deformed_bases["1212"] = ϵ -> ([0 ϵ 0; ϵ 0 0; 0 0 0] + I)*basis
    deformed_bases["1313"] = ϵ -> ([0 0 ϵ; 0 0 0; ϵ 0 0] + I)*basis
    deformed_bases["2323"] = ϵ -> ([0 0 0; 0 0 ϵ; 0 ϵ 0] + I)*basis
    deformed_bases["1122"] = ϵ -> ([ϵ 0 0; 0 ϵ 0; 0 0 0] + I)*basis
    deformed_bases["1133"] = ϵ -> ([ϵ 0 0; 0 0 0; 0 0 ϵ] + I)*basis
    deformed_bases["2233"] = ϵ -> ([0 0 0; 0 ϵ 0; 0 0 ϵ] + I)*basis
    points_array = reshape(points, size(net.points))
    n = nv(net.g)
    H = zeros(3*n, 3*n)
    hessian!(H, basis, points, edge_nodes, rls, iis, youngs)
    function make_curry_function(component)
        function curry(ϵ)
            deformed_basis = deformed_bases[component](ϵ[1])
            F = forces(deformed_basis, points_array, edge_nodes, rls, iis, youngs)
            u = qr(H, Val(true)) \ collect(Iterators.flatten(F))
            return elastic_energy(deformed_basis, points + u, edge_nodes, rls, iis, youngs)
        end
        return curry
    end
    v = det(basis)
    c1111 = ForwardDiff.hessian(make_curry_function("1111"), [0])[1]/v
    c2222 = ForwardDiff.hessian(make_curry_function("2222"), [0])[1]/v
    c3333 = ForwardDiff.hessian(make_curry_function("3333"), [0])[1]/v
    c1212 = ForwardDiff.hessian(make_curry_function("1212"), [0])[1]/(4*v)
    c1313 = ForwardDiff.hessian(make_curry_function("1313"), [0])[1]/(4*v)
    c2323 = ForwardDiff.hessian(make_curry_function("2323"), [0])[1]/(4*v)
    c1122 = ForwardDiff.hessian(make_curry_function("1122"), [0])[1]/(2*v) - (c1111 + c2222)/2
    c1133 = ForwardDiff.hessian(make_curry_function("1133"), [0])[1]/(2*v) - (c1111 + c3333)/2
    c2233 = ForwardDiff.hessian(make_curry_function("2233"), [0])[1]/(2*v) - (c2222 + c3333)/2
    B = 1/9*(c1111 + c2222 + c3333 + 2*(c1122 + c1133 + c2233))
    G = 1/15*(3*(c1212 + c1313 + c2323) + c1111 + c2222 + c3333 - c1122 - c1133 - c2233)
    return B, G, c1111, c2222, c3333, c1212, c1313, c2323, c1122, c1133, c2233
end

function forces(deformed_basis, points, edge_nodes, rls, iis, youngs)
    f = zeros(size(points))
    n = size(points, 2)
    for k in axes(edge_nodes, 1)
        i, j = edge_nodes[k, 1], edge_nodes[k, 2]
        dr = deformed_basis*(points[:, j] + iis[k, :] - points[:, i])
        edge_length_ij = norm(dr)
        factor = youngs*(edge_length_ij - rls[k])/rls[k]/edge_length_ij
        miau = factor*dr
        g = hcat(f[:, 1: i - 1], f[:, i] + miau, f[:, i + 1:n])
        h = hcat(g[:, 1: j - 1], g[:, j] - miau, g[:, j + 1:n])
        f = deepcopy(h)
    end
    return deformed_basis*f
end

