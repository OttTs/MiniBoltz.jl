function relaxation_time(n, T, species)
    Pr = species.prandtl_number
    μ = dynamic_viscosity(T, species.reference_viscosity, species.reference_temperature, species.reference_exponent)
    p = n * BOLTZMANN_CONSTANT * T
    return 3 / Pr * μ / p
end

number_density(N, ω, V) = ω * N / V

temperature(Σ, m) = tr(Σ) * m / (3 * BOLTZMANN_CONSTANT)

dynamic_viscosity(T, μᵣ, Tᵣ, ω) = μᵣ * (T / Tᵣ)^ω
