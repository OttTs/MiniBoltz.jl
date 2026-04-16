# Check which sampling method conserves mean values
using MiniBoltz

particles = Particle[]
for _ in 1:10^6
    x = -1 + 2*rand()
    v = 238*randn(MiniBoltz.SVector{3,Float64}) + MiniBoltz.SVector((x+20)^2, 0.0, 0.0)
    push!(particles, Particle(x, v))
end

m = MiniBoltz.Mesh((-1, 1), 1)

# 1. Constant
n = MiniBoltz.Variable(Float64; N=0, mesh=m)
u = MiniBoltz.Variable(MiniBoltz.SVector{3,Float64}; N=0, mesh=m)
Σ = MiniBoltz.Variable(MiniBoltz.SMatrix{3,3,Float64,9}; N=0, mesh=m)

for p in particles
    MiniBoltz.project_particle!(p.position, 1.0, n)
end
for p in particles
    MiniBoltz.project_particle!(p.position, p.velocity / n(p.position), u)
end
for p in particles
    MiniBoltz.project_particle!(p.position, (p.velocity - u(p.position)) * (p.velocity - u(p.position))' / n(p.position), Σ)
end

# 2. Linear central
μ₀ = MiniBoltz.Variable(Float64; N=1, mesh=m)
μ₁ = MiniBoltz.Variable(MiniBoltz.SVector{3,Float64}; N=1, mesh=m)
μ₂ = MiniBoltz.Variable(MiniBoltz.SMatrix{3,3,Float64,9}; N=1, mesh=m)

for p in particles
    MiniBoltz.project_particle!(p.position, 1.0, μ₀)
end
for p in particles
    MiniBoltz.project_particle!(p.position, p.velocity / μ₀(p.position), μ₁)
end
for p in particles
    MiniBoltz.project_particle!(p.position, (p.velocity - μ₁(p.position)) * (p.velocity - μ₁(p.position))' / μ₀(p.position), μ₂)
end

# 3. Linear raw
m₀ = MiniBoltz.Variable(Float64; N=1, mesh=m)
m₁ = MiniBoltz.Variable(MiniBoltz.SVector{3,Float64}; N=1, mesh=m)
m₂ = MiniBoltz.Variable(MiniBoltz.SMatrix{3,3,Float64,9}; N=1, mesh=m)

for p in particles
    MiniBoltz.project_particle!(p.position, 1.0, m₀)
    MiniBoltz.project_particle!(p.position, p.velocity, m₁)
    MiniBoltz.project_particle!(p.position, p.velocity * p.velocity', m₂)
end

# 4. Constant raw
nn = MiniBoltz.Variable(Float64; N=0, mesh=m)
nu = MiniBoltz.Variable(MiniBoltz.SVector{3,Float64}; N=0, mesh=m)
nuu = MiniBoltz.Variable(MiniBoltz.SMatrix{3,3,Float64,9}; N=0, mesh=m)

for p in particles
    MiniBoltz.project_particle!(p.position, 1.0, nn)
    MiniBoltz.project_particle!(p.position, p.velocity, nu)
    MiniBoltz.project_particle!(p.position, p.velocity * p.velocity', nuu)
end

# Check mean values
println("Constant basis:")
println("Mean velocity: ", u(0.0))
println("Mean temperature: ", tr(Σ(0.0)) / 3)
println("Linear central basis:")
println("Mean velocity: ", μ₁(0.0))
println("Mean temperature: ", tr(μ₂(0.0)) / 3)
println("Linear raw basis:")
println("Mean velocity: ", m₁(0.0) / m₀(0.0))
println("Mean temperature: ", tr(m₂(0.0) / m₀(0.0) - m₁(0.0) * m₁(0.0)' / m₀(0.0)^2) / 3)

# Plot the velocities:
using GLMakie

x = range(-1, 1, length=100)

u_const = [u(xi) for xi in x]
u_central = [μ₁(xi) for xi in x]
u_raw = [m₁(xi) / m₀(xi) for xi in x]

fig = Figure()
ax = Axis(fig[1, 1], title="Mean velocity", xlabel="x", ylabel="u")
lines!(ax, x, [ui[1] for ui in u_const], label="Constant")
lines!(ax, x, [ui[1] for ui in u_central], label="Linear central")
lines!(ax, x, [ui[1] for ui in u_raw], label="Linear raw")
axislegend(ax)
fig

# Plot the temperatures:
T_const = [tr(Σ(xi)) * 6.63E-26 / (3 * BOLTZMANN_CONSTANT) for xi in x]
T_central = [tr(μ₂(xi))* 6.63E-26 / (3 * BOLTZMANN_CONSTANT) for xi in x]
T_raw = [tr(m₂(xi) / m₀(xi) - m₁(xi) * m₁(xi)' / (m₀(xi)*m₀(xi))) * 6.63E-26 / (3 * BOLTZMANN_CONSTANT) for xi in x]
fig = Figure()
ax = Axis(fig[1, 1], title="Mean temperature", xlabel="x", ylabel="T")
lines!(ax, x, T_const, label="Constant")
lines!(ax, x, T_central, label="Linear central")
lines!(ax, x, T_raw, label="Linear raw")
axislegend(ax)
fig

# Plot only trace
F1 = [tr(m₂(xi) / m₀(xi)) for xi in x]
F2 = [tr(m₁(xi) * m₁(xi)' / m₀(xi)^2) for xi in x]
fig = Figure()
ax = Axis(fig[1, 1], title="Trace of raw moments", xlabel="x", ylabel="Trace")
lines!(ax, x, F1, label="tr(m₂ / m₀)")
lines!(ax, x, F2, label="tr(m₁ * m₁' / m₀^2)")
axislegend(ax)
fig

# Plot n
n_const = [n(xi) for xi in x]
n_central = [μ₀(xi) for xi in x]
n_raw = [m₀(xi) for xi in x]
fig = Figure()
ax = Axis(fig[1, 1], title="Number density", xlabel="x", ylabel="n")
lines!(ax, x, n_const, label="Constant")
lines!(ax, x, n_central, label="Linear central")
lines!(ax, x, n_raw, label="Linear raw")
axislegend(ax)
fig

# Plot raw trace
m2_const = [tr(n(xi) * (Σ(xi) + u(xi)*u(xi)')) for xi in x]
m2_central = [tr(μ₀(xi) * (μ₂(xi) + μ₁(xi)*μ₁(xi)')) for xi in x]
m2_raw = [tr(m₂(xi)) for xi in x]
m2_constraw = [tr(nuu(xi)) for xi in x]
fig = Figure()
ax = Axis(fig[1, 1], title="Trace of m₂", xlabel="x", ylabel="Trace")
lines!(ax, x, m2_const, label="Constant")
lines!(ax, x, m2_central, label="Linear central")
lines!(ax, x, m2_raw, label="Linear raw")
lines!(ax, x, m2_constraw, label="Constant raw")
axislegend(ax)
fig