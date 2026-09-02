"""
    RelaxationMethod

Abstract type for particle relaxation method.

# Functions
```julia
relax(v, time_step, species, N, μ, Σ, method)
```
"""
abstract type RelaxationMethod end

"""
    relax(v, time_step, species, N, μ, Σ, method)

This function updates the velocity of a particle based on the given relaxation method.
"""
function relax end


"""
    StandardESFP <: RelaxationMethod

Standard ES-FP collision step assuming a constant Pressure tensor during the time step.
"""
struct StandardESFP <: RelaxationMethod end

function relax(v, Δt, species::Species, n, u, Σ, ::StandardESFP)
    Pr = species.prandtl_number
    T = temperature(Σ, species.mass)
    τ = relaxation_time(n, T, species)

    c = v - u

    ν = 1 - 3 / (2 * Pr)
    D = Symmetric(ν * Σ + (1-ν) * tr(Σ)/3 * I)
    d = try
        cholesky(D).L
    catch
        # Fallback: Standard FP relaxation
        cholesky(Symmetric(0 * D + tr(Σ)/3 * I)).L
    end
    ξ = randn(typeof(u))

    c = exp(-Δt / τ) * c + √(1 - exp(-2Δt / τ)) * d * ξ
    return u + c
end


"""
    ExactESFP <: RelaxationMethod

Exact integration of the ES-FP collision operator.
"""
struct ExactESFP <: RelaxationMethod end

function relax(v, Δt, species::Species, n, u, Σ, ::ExactESFP)
    Pr = species.prandtl_number
    T = temperature(Σ, species.mass)
    τ = relaxation_time(n, T, species)

    c = v - u

    ν = 1 - 3 / (2 * Pr)
    D = Symmetric((exp(-2 * (1 - ν) * Δt/τ) - exp(-2Δt/τ)) * Σ + (1 - exp(-2 * (1 - ν) * Δt/τ)) * tr(Σ)/3 * I)
    d = try
        cholesky(D).L
    catch
        # Fallback: Standard FP relaxation
        √(1 - exp(-2Δt / τ)) * cholesky(Symmetric(0 * D + tr(Σ)/3 * I)).L
    end
    ξ = randn(typeof(u))

    c = exp(-Δt / τ) * c + d * ξ
    return u + c
end

"""
    MidpointESFP <: RelaxationMethod

Midpoint ES-FP collision step.
"""
struct MidpointESFP <: RelaxationMethod end

function relax(v, Δt, species::Species, n, u, Σ, ::MidpointESFP)
    Pr = species.prandtl_number
    T = temperature(Σ, species.mass)
    τ = relaxation_time(n, T, species)

    c = v - u

    ν = 1 - 3 / (2 * Pr)

    γ⁻ = 1 - Δt / (2 * τ)
    γ⁺ = 1 + Δt / (2 * τ)

    β⁻ = γ⁺^2 - Δt * ν / τ
    β⁺ = γ⁻^2 + Δt * ν / τ
    Σⁿ⁺¹ = (β⁺ * Σ + 2 * (1 - ν) * Δt / τ * tr(Σ)/3 * I) / β⁻

    D = Symmetric((1-ν) * tr(Σ) / 3 * I + ν / 2 * (Σ + Σⁿ⁺¹)) * 2 * Δt / τ
    d = cholesky(D).L
    ξ = randn(typeof(u))

    c = (γ⁻ * c + d * ξ) / γ⁺
    return u + c
end

"""
    USPESFP <: RelaxationMethod

Unified Stochastic Particle ES-FP collision step.
"""
struct USPESFP <: RelaxationMethod end

function relax(v, Δt, species::Species, n, u, Σ, ::USPESFP)
    Pr = species.prandtl_number
    T = temperature(Σ, species.mass)
    p = n * BOLTZMANN_CONSTANT * T
    μ = dynamic_viscosity(T, species.reference_viscosity, species.reference_temperature, species.reference_exponent)
    ε = μ / p

    c = v - u

    if Δt / ε ≤ 2 / Pr
        α = ((2 - Pr * Δt / ε) / (2 + Pr * Δt / ε))^(1/3)
    else
        α = -((Pr * Δt / ε - 2) / (2 + Pr * Δt / ε))^(1/3)
    end
    β = 1 / (1 - α^2) * ((2 - Δt / ε) / (2 + Δt / ε) - α^2)

    R = BOLTZMANN_CONSTANT / species.mass
    ρ = n * species.mass
    σ = ρ * (Σ - tr(Σ) / 3 * I)

    Π = Symmetric(R * T * I + β * σ / ρ)
    L = try
        cholesky(Π).L
    catch
        P = ρ * Σ
        if β ≤ 0
            λₘₐₓ = maximum(eigvals(P))
            # (1. - 1.e-12) makes it strictly positive definite
            β = - (1. - 1.e-12) * R * T / (λₘₐₓ / ρ - R * T)
        else
            λₘᵢₙ = minimum(eigvals(P))
            β = (1. - 1.e-12) * R * T / (R * T - λₘᵢₙ / ρ)
        end

        α² = (β - (2 - Δt / ε) / (2 + Δt / ε)) / (β - 1)
        α₁ = √α²
        α₂ = -α₁
        Pr₁ = 2 * (1 - α₁^3) / ((Δt / ε) * (1 + α₁^3))
        Pr₂ = 2 * (1 - α₂^3) / ((Δt / ε) * (1 + α₂^3))
        if abs(Pr₁ - Pr) < abs(Pr₂ - Pr)
            α = α₁
        else
            α = α₂
        end

        Π = Symmetric(R * T * I + β * σ / ρ)
        cholesky(Π).L
    end
    ξ = randn(typeof(u))

    c = α * c + sqrt(1 - α^2) * L * ξ
    return u + c
end
