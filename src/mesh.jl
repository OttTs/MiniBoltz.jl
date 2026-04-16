struct SimulationMesh
    limits::Tuple{Float64, Float64}
    num_elements::Int64
end

element_volume(mesh::SimulationMesh) = (mesh.limits[2] - mesh.limits[1]) / mesh.num_elements

function element(x, m::SimulationMesh)
    x == m.limits[2] && return m.num_elements
    floor(Int, (x - m.limits[1]) / element_volume(m)) + 1
end
to_reference(x, element, m::SimulationMesh) = 2((x - m.limits[1]) / element_volume(m) - element) + 1
to_physical(ξ, element, m::SimulationMesh) = m.limits[1] + ((ξ - 1)/2 + element) * element_volume(m)


struct Variable{N,T}
    _dofs::Vector{T}
    _mesh::SimulationMesh
    function Variable(T::Type; N::Integer, mesh::SimulationMesh)
        @assert 0 ≤ N ≤ 1 "Only constant (N=0) and linear (N=1) basis functions are supported."
        new{N,T}(zeros(T, (N + 1) * mesh.num_elements), mesh)
    end
end

evaluate_basis(f::Variable{0}, x) = ((element(x, f._mesh), 1),)
function evaluate_basis(f::Variable{1}, x)
    e = element(x, f._mesh)
    ξ = to_reference(x, e, f._mesh)

    return (
        (2*e - 1, (1 - √3 * ξ)/2),
        (2*e, (1 + √3 * ξ)/2)
    )
end

# For constant and linear basis functions, the Gauss weights are constant.
gauss_weight(f::Variable{0}, _) = 2.0
gauss_weight(f::Variable{1}, _) = 1.0

function (f::Variable)(x)
    return sum(f._dofs[i] * φᵢ for (i, φᵢ) in evaluate_basis(f, x))
end

function project_particle!(x, weight, f::Variable)
    for (i, φⱼ) in evaluate_basis(f, x)
        ωⱼ = gauss_weight(f, i)
        f._dofs[i] += 2 * weight * φⱼ / (ωⱼ * element_volume(f._mesh))
    end
end