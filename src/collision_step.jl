struct RawMoments{N}
    m₀ :: Variable{N,Float64}
    m₁ :: Variable{N,SVector{3,Float64}}
    m₂ :: Variable{N,SMatrix{3,3,Float64,9}}
    function RawMoments(mesh, N)
        new{N}(Variable(Float64; N=N, mesh=mesh),
            Variable(SVector{3,Float64}; N=N, mesh=mesh),
            Variable(SMatrix{3,3,Float64,9}; N=N, mesh=mesh))
    end
end

struct MomentContainer{N}
    before::Vector{RawMoments{N}}
    tmp::Vector{RawMoments{N}}
    averaged::RawMoments{N}
    function MomentContainer(mesh, N)
        new{N}([RawMoments(mesh, N) for _ in 1:Threads.nthreads()],
            [RawMoments(mesh, N) for _ in 1:Threads.nthreads()],
            RawMoments(mesh, N))
    end
end

function collision_step!(particles, species, time_step, mesh, relaxation_method;
    do_stochastic_interpolation, moment_container)

    raw_moments!(moment_container.before, particles, species.weighting)

    # Relax particles
    @batch for i in eachindex(particles)
        xₚ = particles[i].position
        vₚ = particles[i].velocity

        if do_stochastic_interpolation
            xₚ += (rand() - 0.5) * element_volume(mesh)
            xₚ = clamp(xₚ, mesh.limits...)
        end

        n, μ, Σ = central_moments(xₚ, moment_container.before[1])

        vₚ = relax(vₚ, time_step, species, n, μ, Σ, relaxation_method)
        particles[i] = Particle(particles[i].position, vₚ)
    end

    raw_moments!(moment_container.tmp, particles, species.weighting)

    # Conserve momentum and energy
    # Conservation is only enforced on the mean values
    # For raw moments, just evaluate at the moments at the middle of the element
    @batch for i in eachindex(particles)
        xₚ = particles[i].position
        vₚ = particles[i].velocity

        x_mid = to_physical(0.0, element(xₚ, mesh), mesh)

        _, μ, Σ = central_moments(x_mid, moment_container.before[1])
        _, μⁿ, Σⁿ = central_moments(x_mid, moment_container.tmp[1])

        r = √(tr(Σ) / tr(Σⁿ))
        vₚ = μ + r * (vₚ - μⁿ)

        particles[i] = Particle(xₚ, vₚ)
    end
end

function raw_moments!(moments, particles, weighting)

    for m in moments
        for i in eachindex(m.m₀._dofs)
            m.m₀._dofs[i] = zero(eltype(m.m₀._dofs))
            m.m₁._dofs[i] = zero(eltype(m.m₁._dofs))
            m.m₂._dofs[i] = zero(eltype(m.m₂._dofs))
        end
    end

    # Instead of paper, use raw moments (conserves mean values exactly)
    @batch for p in particles
        id = Threads.threadid()
        project_particle!(p.position, weighting, moments[id].m₀)
        project_particle!(p.position, weighting * p.velocity, moments[id].m₁)
        project_particle!(p.position, weighting * SMatrix{3,3,Float64,9}(p.velocity * p.velocity'), moments[id].m₂)
    end

    for i in 2:length(moments)
        moments[1].m₀._dofs .+= moments[i].m₀._dofs
        moments[1].m₁._dofs .+= moments[i].m₁._dofs
        moments[1].m₂._dofs .+= moments[i].m₂._dofs
    end
end

function central_moments(x, moments)
    n = moments.m₀(x)
    μ = moments.m₁(x) / n
    Σ = moments.m₂(x) / n - μ * μ'
    return n, μ, Σ
end