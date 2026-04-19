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
        cholesky(Symmetric(tr(Σ)/3 * I)).L
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
        √(1 - exp(-2Δt / τ)) * cholesky(Symmetric(tr(Σ)/3 * I)).L
    end
        ξ = randn(typeof(u))

        c = exp(-Δt / τ) * c + d * ξ
        return u + c
end