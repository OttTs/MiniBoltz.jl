module MiniBoltz

using Polyester: @batch
using DataFrames: DataFrame
using CSV, JLD2
using ProgressMeter
#using Statistics: mean, var
using StaticArrays: SVector, SMatrix
using LinearAlgebra: norm, normalize, tr, I, cholesky, Symmetric

"""
    BOLTZMANN_CONSTANT

Boltzmann constant in SI units (J/K).

``k_B = 1.380649 \\times 10^{-23}`` J/K
"""
const BOLTZMANN_CONSTANT = 1.380649e-23

include("mesh.jl")
include("species_particles.jl")
include("physical_variables.jl")
include("boundary_condition.jl")
include("relaxation.jl")
include("movement_step.jl")
include("collision_step.jl")
include("simulation.jl")

export BOLTZMANN_CONSTANT
export Parameters, simulate!
export read_particles, read_parameters
export Particle, Species, SimulationMesh, RelaxationMethod
export DiffuseWall, SpecularWall
export StandardESFP, ExactESFP, MidpointESFP
export init_uniform
export Variable, SimulationMesh

end
