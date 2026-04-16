
struct Species
    weighting::Float64
    mass::Float64
    reference_temperature::Float64
    reference_viscosity::Float64
    reference_exponent::Float64
    prandtl_number::Float64
    function Species(
        weighting::Real,
        mass::Real,
        reference_temperature::Real,
        reference_exponent::Real,
        prandtl_number::Real,
        reference_diameter::Real)
        μ = reference_viscosity(mass, reference_temperature, reference_diameter, reference_exponent)
        new(weighting, mass, reference_temperature, μ, reference_exponent, prandtl_number)
    end
end

function reference_viscosity(m, T, d, ω)
    return 30 * √(m * BOLTZMANN_CONSTANT * T / π) / (4 * (5 - 2 * ω) * (7 - 2 * ω) * d^2)
end

struct Particle
    position::Float64
    velocity::SVector{3, Float64}
end

function init_uniform(n::Real, u::AbstractVector{<:Real}, T::Real, mesh::SimulationMesh, species::Species)
    L = mesh.limits[2] - mesh.limits[1]
    Nₚ = round(Int, n * L / species.weighting)
    σ² = BOLTZMANN_CONSTANT * T / species.mass
    particles = [Particle(mesh.limits[1] + L*rand(), √σ²*randn(SVector{3,Float64}) + u) for _ in 1:Nₚ]
    return particles
end