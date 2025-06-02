include("../elastic_network.jl")

function grad_autodiff(net)
    net_info = net_info_primitive(net)
    basis, ini_points, egs, rls, iis, youngs = net_info
    energy(points) = elastic_energy(basis, points, egs, rls, iis, youngs)
    return ForwardDiff.gradient(energy, ini_points)
end

function hessian_autodiff(net)
    net_info = net_info_primitive(net)
    basis, ini_points, egs, rls, iis, youngs = net_info
    energy(points) = elastic_energy(basis, points, egs, rls, iis, youngs)
    return ForwardDiff.hessian(energy, ini_points)
end