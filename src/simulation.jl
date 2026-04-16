
function simulate!(
    particles::Vector{Particle},
    species::Species,
    time_step::Real,
    num_steps::Integer,
    mesh::Mesh,
    boundary_conditions::Tuple{<:AbstractBoundaryCondition, <:AbstractBoundaryCondition},
    relaxation_method::RelaxationMethod;
    do_stochastic_interpolation::Bool=false,
    polynomial_degree::Integer=0,
    num_averaging_steps::Real=1,
    write_interval::Integer=1,
    file_name::String,
    show_progress::Bool=true)

    write_interval = max(1, write_interval)
    progress = show_progress ? Progress(num_steps) : nothing

    moment_container = MomentContainer(mesh, polynomial_degree)

    df = DataFrame(step=Int[], time=Float64[], u_x=Float64[], u_y=Float64[], u_z=Float64[], T=Float64[])

    for istep in 1:num_steps
        movement_step!(particles, species, time_step, mesh, boundary_conditions)
        collision_step!(particles, species, time_step, mesh, relaxation_method;
            do_stochastic_interpolation, moment_container)

        if istep > num_steps - num_averaging_steps
            moment_container.averaged.m₀._dofs .+= moment_container.before[1].m₀._dofs
            moment_container.averaged.m₁._dofs .+= moment_container.before[1].m₁._dofs
            moment_container.averaged.m₂._dofs .+= moment_container.before[1].m₂._dofs
        end

        if istep % write_interval == 0
            u_x, u_y, u_z, T = 0.0, 0.0, 0.0, 0.0
            for e in 1:mesh.num_elements
                x = to_physical(0, e, mesh)
                _, μ, Σ = central_moments(x, moment_container.before[1])
                u_x += μ[1] / mesh.num_elements
                u_y += μ[2] / mesh.num_elements
                u_z += μ[3] / mesh.num_elements
                T += species.mass / BOLTZMANN_CONSTANT * tr(Σ) / 3 / mesh.num_elements
            end
            push!(df, (istep, istep * time_step, u_x, u_y, u_z, T))
        end

        isnothing(progress) || next!(progress)
    end

    # Finalize averaging
    moment_container.averaged.m₀._dofs ./= num_averaging_steps
    moment_container.averaged.m₁._dofs ./= num_averaging_steps
    moment_container.averaged.m₂._dofs ./= num_averaging_steps

    # Output results
    jldopen(file_name * ".jld2", "w") do file
        write(file, "mesh", mesh)
        write(file, "zeroth_moment", moment_container.averaged.m₀)
        write(file, "first_moment", moment_container.averaged.m₁)
        write(file, "second_moment", moment_container.averaged.m₂)
    end

    CSV.write(file_name * ".csv", df)

    isnothing(progress) || finish!(progress)

end