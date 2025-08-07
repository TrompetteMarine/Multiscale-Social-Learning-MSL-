#!/usr/bin/env julia
# basic_simulation.jl - Simple BeliefSim example

using Pkg
Pkg.activate(@__DIR__)

include("../src/BeliefSim.jl")
using ..BeliefSim
using Plots
using Statistics

println("🌊 Basic BeliefSim Example")
println("=========================")

# Simple parameters
params = MSLParams(
    N = 50,           # 50 agents
    T = 30.0,         # 30 time units
    cognitive = CognitiveParams(
        λ = 1.0,      # Mean reversion
        α = 0.6,      # Social influence (moderate)
        σ = 0.3       # Noise
    ),
    network_type = :small_world
)

println("\nRunning simulation with $(params.N) agents...")

# Run simulation
t_vec, trajectories, analysis = run_msl_simulation(params; seed=42)

# Print results
println("\n📊 Results:")
println("   Final consensus: $(round(analysis[:final_consensus], digits=3))")
println("   Final polarization: $(round(analysis[:final_polarization], digits=3))")
println("   Mean belief: $(round(analysis[:mean_belief], digits=3))")
println("   Std belief: $(round(analysis[:std_belief], digits=3))")
println("   Detected regime: $(analysis[:regime])")

# Create output directory
mkpath("output")

# Plot belief evolution
p = plot(xlabel="Time", ylabel="Belief", title="Belief Evolution")
for i in 1:min(10, params.N)
    plot!(p, t_vec, trajectories[:beliefs][i], alpha=0.5, label="")
end
mean_beliefs = [mean([trajectories[:beliefs][i][t] for i in 1:params.N]) for t in 1:length(t_vec)]
plot!(p, t_vec, mean_beliefs, lw=3, color=:red, label="Mean")

savefig(p, "output/basic_beliefs.png")
println("\n   Plot saved: output/basic_beliefs.png")

println("\n✅ Basic example complete!")
println("\nTry modifying:")
println("  • α (social influence): 0.2 → 1.5")
println("  • N (number of agents): 10 → 200")
println("  • network_type: :fully_connected, :scale_free, :random")
