using MiniBoltz
using Test
using Statistics
using StaticArrays: SVector, SMatrix
using LinearAlgebra: tr

@testset "MiniBoltz.jl" begin
    @testset "SimulationMesh" begin
        mesh = SimulationMesh((0.0, 1.0), 10)

        @test MiniBoltz.element_volume(mesh) ≈ 0.1

        @test MiniBoltz.element(0.0, mesh) == 1
        @test MiniBoltz.element(0.48, mesh) == 5
        @test MiniBoltz.element(1.0, mesh) == 10

        @test MiniBoltz.to_reference(0.44, 5, mesh) ≈ -0.2
        @test MiniBoltz.to_physical(0.8, 5, mesh) ≈ 0.49
    end

    @testset "Variables" begin
        mesh = SimulationMesh((0.0, 1.0), 10)

        f = MiniBoltz.Variable(Float64; N=1, mesh=mesh)
        MiniBoltz.project_particle!(0.44, MiniBoltz.element_volume(mesh), f)
        @test f._dofs[9] ≈ 1 - MiniBoltz.to_reference(0.44, 5, mesh) * √3
        @test f._dofs[10] ≈ 1 + MiniBoltz.to_reference(0.44, 5, mesh) * √3
        @test f(0.45) ≈ 1.0
        @test f(0.44) > f(0.46)
        @test f(0.51) ≈ 0.0

        f = MiniBoltz.Variable(Float64; N=0, mesh=mesh)
        MiniBoltz.project_particle!(0.44, MiniBoltz.element_volume(mesh), f)
        @test f._dofs[5] ≈ 1
        @test f(0.48) ≈ 1.0
        @test f(0.51) ≈ 0.0
    end

    @testset "Particle init" begin
        mesh = SimulationMesh((0.0, 1.0), 10)
        species = Species(1e16, 6.63E-26, 273, 0.77, 2/3, 4.05E-10)
        n, u, T = 1e20, SVector(200.0, 0.0, 0.0), 273
        particles = init_uniform(n, u, T, mesh, species)

        @test length(particles) == round(Int, n * (mesh.limits[2] - mesh.limits[1]) / species.weighting)
        @test mean(p.position for p in particles) ≈ 0.5 atol=0.01
        @test mean(p.velocity for p in particles) ≈ u atol=10
    end

    @testset "Particle relaxation" begin
        # Relaxation of two gaussians
        species = Species(1, 6.63E-26, 273, 0.77, 2/3, 4.05E-10)
        n, u, T = 1e20, SVector(100.0, 150.0, 200.0), 273
        τ = MiniBoltz.relaxation_time(n, T, species)

        # Sample from two gaussians with mean u and -u
        N = 100000
        σ² = BOLTZMANN_CONSTANT * T / species.mass
        v = √σ²*randn(SVector{3}, N)
        for i in 1:N÷2
            v[i] = v[i] + u
            v[i+N÷2] = v[i+N÷2] - u
        end

        # The distribution should have zero mean
        μ = SVector(0.0, 0.0, 0.0)

        # After τ/10, Check single step exact with small steps
        Σ_exact = cov(v)
        Σ_exact = cov(collect(MiniBoltz.relax(i, τ/4, species, n, μ, Σ_exact, ExactESFP()) for i in v))

        # Relax with both methods and check that the covariance is close to the expected value
        v_standard, v_exact = copy(v), copy(v)
        nsteps = 100
        for _ in 1:nsteps
            Δt = τ/4 / nsteps

            Σ_iter = cov(v_standard)
            MiniBoltz.@batch for i in eachindex(v_standard)
                v_standard[i] = MiniBoltz.relax(v_standard[i], Δt, species, n, μ, Σ_iter, StandardESFP())
            end

            Σ_iter = cov(v_exact)
            MiniBoltz.@batch for i in eachindex(v_exact)
                v_exact[i] = MiniBoltz.relax(v_exact[i], Δt, species, n, μ, Σ_iter, ExactESFP())
            end
        end
        @test cov(v_standard) ≈ Σ_exact atol=2000
        @test cov(v_exact) ≈ Σ_exact atol=2000
    end

    @testset "Particle reflection" begin
        u = SVector(0.0, 0.0, 100.0)
        T = 300.0
        bc = DiffuseWall(u, T)
        species = Species(1, 6.63E-26, 273, 0.77, 2/3, 4.05E-10)
        samples = [MiniBoltz.reflect(-ones(SVector{3,Float64}), bc, species) for _ in 1:100000]
        σ² = BOLTZMANN_CONSTANT * T / species.mass
        Σ = SMatrix{3}(σ² * (4 - π) / 2, 0, 0, 0, σ², 0, 0, 0, σ²)
        @test mean(samples) ≈ u + SVector(√(σ² * π / 2), 0, 0) rtol=0.01
        @test cov(samples) ≈ Σ rtol=0.01

        specular = SpecularWall()
        v = SVector(-123.0, 4.5, -6.7)
        @test MiniBoltz.reflect(v, specular, species) == SVector(123.0, 4.5, -6.7)
    end

    @testset "Particle collision" begin
        # TODO For now, just check that the collision step conserves momentum and energy
        species = Species(1, 6.63E-26, 273, 0.77, 2/3, 4.05E-10)
        mesh = SimulationMesh((0.0, 1.0), 1)

        u, T = SVector(200.0, 0.0, 0.0), 273

        N = 10000
        σ² = BOLTZMANN_CONSTANT * T / species.mass
        particles = [Particle(rand(), √σ²*randn(SVector{3}) + u) for _ in 1:N]

        moment_container = MiniBoltz.MomentContainer(mesh, 0)
        MiniBoltz.raw_moments!(moment_container.before, particles, species.weighting)
        n, μ, Σ = MiniBoltz.central_moments(0.5, moment_container.before[1])

        τ = MiniBoltz.relaxation_time(n, T, species)

        MiniBoltz.collision_step!(particles, species, τ/4, mesh, StandardESFP();
            do_stochastic_interpolation=true, moment_container)

        MiniBoltz.raw_moments!(moment_container.before, particles, species.weighting)
        n_new, μ_new, Σ_new = MiniBoltz.central_moments(0.5, moment_container.before[1])

        @test n_new ≈ n
        @test μ_new ≈ μ
        @test tr(Σ_new) ≈ tr(Σ)
    end

    @testset "Particle movement" begin
        particles = [Particle(0.5, SVector(1.0, 0.0, 0.0))]
        species = Species(1, 6.63E-26, 273, 0.77, 2/3, 4.05E-10)
        time_step = 2.2
        mesh = SimulationMesh((0.0, 1.0), 1)
        bcs = (SpecularWall(), DiffuseWall(SVector(-1.0, 0.0, 0.0), 0.0))
        MiniBoltz.movement_step!(particles, species, time_step, mesh, bcs)
        @test particles[1].position ≈ 0.7
    end

    @testset "Simple simulation" begin
        species = Species(1e16, 6.63E-26, 273, 0.77, 2/3, 4.05E-10)
        mesh = SimulationMesh((0.0, 1.0), 10)

        particles = init_uniform(1e20, SVector(200.0, 0.0, 0.0), 273, mesh, species)
        time_step = 1e-6
        num_steps = 100
        num_averaging_steps = 10
        boundary_conditions = (SpecularWall(), DiffuseWall(SVector(-200.0, 0.0, 0.0), 273))
        relaxation_method = StandardESFP()
        file_name = "test_simulation"

        simulate!(particles, species, time_step, num_steps, mesh, boundary_conditions, relaxation_method;
        do_stochastic_interpolation=true,
        polynomial_degree=0,
        num_averaging_steps,
        write_interval=2,
        file_name,
        show_progress=true)

        @test isfile(file_name * ".csv")
        @test isfile(file_name * ".jld2")
    end
end
