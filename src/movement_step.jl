function movement_step!(particles, species, time_step, mesh, boundary_conditions)
    @batch for i in eachindex(particles)
        xₚ = particles[i].position
        vₚ = particles[i].velocity

        Δt = time_step

        part_is_done = false
        while !part_is_done
            xₚ += vₚ[1] * Δt
            if xₚ < mesh.limits[1]
                Δt = (xₚ - mesh.limits[1]) / vₚ[1] # vₚ[1] is negative, thus Δt is positive
                xₚ = mesh.limits[1]
                vₚ = reflect(vₚ, boundary_conditions[1], species)
            elseif xₚ > mesh.limits[2]
                Δt = (xₚ - mesh.limits[2]) / vₚ[1]
                xₚ = mesh.limits[2]
                vₚ = reflect(vₚ, boundary_conditions[2], species)
            else
                part_is_done = true
            end
        end

        particles[i] = Particle(xₚ, vₚ)
    end
end