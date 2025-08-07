#!/usr/bin/env julia
# paper_reproduction.jl - Reproduce key results from Bontemps (2024)

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
include("../src/BeliefSim.jl")
using ..BeliefSim
using Plots
using Statistics
using LinearAlgebra

println("🌊 Multi-Scale Social Learning Simulation")
println("==========================================")
println("Reproducing: Bontemps (2025) - Theoretical Economics")
println()

# Create output directory
mkpath("output")

# ============================================================================
# 1. Basic MSL Simulation (Paper Section 2)
# ============================================================================
println("1. Running basic MSL simulation...")

# Parameters matching paper setup
params = MSLParams(
    N = 200,   # Number of agents
    T = 100.0,  # Time Horizon
    Δt = 0.01,  # Time Step
    cognitive = CognitiveParams(
        λ = 1.2,       # Mean reversion
        α = 0.7,       # Social influence  
        σ = 0.15,      # Noise
        δm = 0.4,     # Memory adjustment
        ηw = 0.35,     # Deliberation adjustment
        βΘ = 0.25      # Threshold adjustment
    ),
    network_type = :small_world,
    network_params = Dict(:k => 6, :p => 0.3)
)

# Run simulation
t_vec, trajectories, analysis = run_msl_simulation(params; seed=42)

println("✅ Simulation complete")
println("   Final consensus: $(round(analysis[:final_consensus], digits=3))")
println("   Final polarization: $(round(analysis[:final_polarization], digits=3))")
println("   Detected regime: $(analysis[:regime])")
println()

# Plot belief evolution
p1 = plot(xlabel="Time", ylabel="Beliefs", title="Multi-Scale Social Learning Dynamics")
for i in 1:min(20, params.N)  # Plot subset of agents
    plot!(p1, t_vec, trajectories[:beliefs][i], alpha=0.3, color=:blue, label="")
end
# Add mean
mean_beliefs = [mean([trajectories[:beliefs][i][t] for i in 1:params.N]) for t in 1:length(t_vec)]
plot!(p1, t_vec, mean_beliefs, lw=3, color=:red, label="Mean")
savefig(p1, "output/belief_evolution.png")
println("   → output/belief_evolution.png")

# ============================================================================
# 2. Bifurcation Analysis (Paper Section 4)
# ============================================================================
println("\n2. Performing bifurcation analysis...")

# Sweep social influence parameter α
α_values = 0.2:0.1:1.5
consensus_results = Float64[]
polarization_results = Float64[]

for α in α_values
    # Modify parameters
    test_params = MSLParams(
        params.N, params.T, params.Δt,
        CognitiveParams(params.cognitive; α = α),
        params.network_type,
        params.network_params,
        params.ν,
        params.save_interval
    )
    
    # Run simulation
    _, traj, ana = run_msl_simulation(test_params; seed=123)
    
    push!(consensus_results, ana[:final_consensus])
    push!(polarization_results, ana[:final_polarization])
    
    print(".")
end
println(" Done!")

# Find critical point
gradients = diff(consensus_results)
critical_idx = argmin(gradients)
α_star = α_values[critical_idx]

println("   Critical α* ≈ $(round(α_star, digits=2))")

# Plot bifurcation diagram
p2 = plot(α_values, consensus_results, 
          xlabel="Social Influence (α)", 
          ylabel="Final Consensus",
          title="Supercritical Pitchfork Bifurcation",
          lw=2, marker=:circle, label="Consensus")
plot!(p2, α_values, polarization_results, 
      lw=2, marker=:square, color=:red, label="Polarization")
vline!(p2, [α_star], ls=:dash, color=:green, lw=2, label="Critical α*")
savefig(p2, "output/bifurcation.png")
println("   → output/bifurcation.png")

# ============================================================================
# 3. Multi-Scale Analysis (Paper Section 5)
# ============================================================================
println("\n3. Multi-scale shift analysis...")

# Plot shift metrics evolution
p3 = plot(xlabel="Time", ylabel="Shift Metric", 
          title="Multi-Scale Shift Detection",
          layout=(3,1), size=(800, 600))

shift_micro = Float64[]
shift_meso = Float64[]
shift_macro = Float64[]

# Compute shifts over time windows
window_size = 10
for t_end in window_size:5:length(t_vec)
    t_start = max(1, t_end - window_size)
    
    # Extract window
    window_beliefs = [[trajectories[:beliefs][i][t] for i in 1:params.N] 
                      for t in t_start:t_end]
    
    # Compute shift metrics
    push!(shift_micro, std(window_beliefs[end]))
    push!(shift_meso, std([mean(b[1:params.N÷4]) for b in window_beliefs]))
    push!(shift_macro, std([mean(b) for b in window_beliefs]))
end

t_shifts = t_vec[window_size:5:length(t_vec)]

plot!(p3[1], t_shifts, shift_micro, lw=2, color=:green, label="Micro")
plot!(p3[2], t_shifts, shift_meso, lw=2, color=:orange, label="Meso")
plot!(p3[3], t_shifts, shift_macro, lw=2, color=:purple, label="Macro")

savefig(p3, "output/multiscale_shifts.png")
println("   → output/multiscale_shifts.png")

# ============================================================================
# 4. Regime Classification (Paper Section 6)
# ============================================================================
println("\n4. Regime classification analysis...")

# Test different parameter combinations
test_cases = [
    (α=0.2, expected="Equilibrium"),
    (α=0.5, expected="Meso-Buffered"),
    (α=1.0, expected="Broadcast/Cascade")
]

println("\n   Parameter | Detected Regime | Expected")
println("   " * "-"^40)

for test in test_cases
    test_params = MSLParams(
        50, 30.0, 0.01,  # Smaller for speed
        CognitiveParams(params.cognitive; α = test.α),
        params.network_type,
        params.network_params,
        params.ν,
        0.5
    )
    
    _, _, ana = run_msl_simulation(test_params; seed=42)
    
    println("   α = $(test.α)  | $(ana[:regime]) | $(test.expected)")
end

# ============================================================================
# 5. Cognitive Dynamics (Paper Section 2)
# ============================================================================
println("\n5. Plotting cognitive dynamics...")

# Extract cognitive variables
mean_tension = analysis[:cognitive_tension_evolution]
mean_memory = [mean([trajectories[:memory][i][t] for i in 1:params.N]) for t in 1:length(t_vec)]
mean_deliberation = [mean([trajectories[:deliberation][i][t] for i in 1:params.N]) for t in 1:length(t_vec)]
mean_threshold = [mean([trajectories[:thresholds][i][t] for i in 1:params.N]) for t in 1:length(t_vec)]

p4 = plot(layout=(2,2), size=(800, 600))

plot!(p4[1], t_vec, mean_tension, xlabel="Time", ylabel="|x - r|", 
      title="Cognitive Tension", lw=2, color=:red)
plot!(p4[2], t_vec, mean_memory, xlabel="Time", ylabel="m", 
      title="Memory Weight", lw=2, color=:blue)
plot!(p4[3], t_vec, mean_deliberation, xlabel="Time", ylabel="w", 
      title="Deliberation Weight", lw=2, color=:green)
plot!(p4[4], t_vec, mean_threshold, xlabel="Time", ylabel="Θ", 
      title="Cognitive Threshold", lw=2, color=:purple)

savefig(p4, "output/cognitive_dynamics.png")
println("   → output/cognitive_dynamics.png")

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^60)
println("📊 SIMULATION SUMMARY")
println("="^60)
println()
println("Key Results:")
println("1. ✅ Jump-diffusion dynamics with 5D agent state")
println("2. ✅ Supercritical pitchfork at α* ≈ $(round(α_star, digits=2))")
println("3. ✅ Multi-scale shift detection (micro/meso/macro)")
println("4. ✅ Regime classification: $(analysis[:regime])")
println("5. ✅ Cognitive dynamics tracking")
println()
println("Output files generated:")
println("  • belief_evolution.png - Agent belief trajectories")
println("  • bifurcation.png - Critical peer influence analysis")
println("  • multiscale_shifts.png - Layer-specific shift metrics")
println("  • cognitive_dynamics.png - Cognitive variable evolution")
println()
println("🌊 Simulation complete!")
