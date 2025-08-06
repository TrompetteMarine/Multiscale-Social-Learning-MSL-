#!/usr/bin/env julia
# simple_demo.jl - Basic BeliefSim demonstration without complex visualizations

using Pkg
Pkg.activate(@__DIR__)

include("src/Kernel.jl"); using .Kernel
include("src/Metrics.jl"); using .Metrics  
using Statistics, Random, Plots

println("🌊 BeliefSim Simple Demo")
println("========================\n")

# Create output directory
mkpath("output/demo")

# ============================================================================
# Demo 1: Basic Network Comparison
# ============================================================================
println("📊 Demo 1: Network Effects on Consensus")
println("-" * repeat("-", 40))

N = 50  # Smaller network for faster demo
pars = SimPars(N=N, κ=1.2, β=0.6, σ=0.25, T=15.0, Δt=0.01)

# Three different networks
networks = Dict(
    "Fully Connected" => fully_connected(N),
    "Small World" => watts_strogatz_W(N; k=6, p=0.3),
    "Scale Free" => barabasi_albert_W(N; m=3)
)

println("Running simulations...")
results = []

for (name, W) in networks
    println("  • $name network...")
    
    # Run simulation
    t_vec, traj = run_one_path(pars; W=W, seed=42)
    
    # Calculate metrics
    final_consensus = consensus_metrics(traj[end])[:consensus]
    final_polarization = polarization_metrics(traj[end])[:group_polarization]
    
    push!(results, (network=name, consensus=final_consensus, polarization=final_polarization))
    
    # Simple trajectory plot
    mean_traj = [mean(state) for state in traj]
    std_traj = [std(state) for state in traj]
    
    p = plot(t_vec, mean_traj, ribbon=std_traj, 
             xlabel="Time", ylabel="Mean Belief ± Std", 
             title="$name Network Dynamics", legend=false)
    
    safe_name = replace(name, " " => "_")
    savefig(p, "output/demo/$(safe_name)_simple.png")
    println("    → output/demo/$(safe_name)_simple.png")
end

println("\n📋 Network Comparison Results:")
for r in results
    println("  $(r.network):")
    println("    Consensus: $(round(r.consensus, digits=3))")
    println("    Polarization: $(round(r.polarization, digits=3))")
end

# ============================================================================
# Demo 2: Parameter Effects
# ============================================================================
println("\n🧠 Demo 2: Cognitive Cost Effects")
println("-" * repeat("-", 40))

W_test = watts_strogatz_W(40; k=4, p=0.2)
κ_values = [0.5, 1.0, 1.5, 2.0]  # Different cognitive costs

println("Testing different cognitive costs...")
param_results = []

for κ in κ_values
    test_pars = SimPars(N=40, κ=κ, β=0.5, σ=0.3, T=12.0, Δt=0.01)
    
    print("  κ = $κ... ")
    t_test, traj_test = run_one_path(test_pars; W=W_test, seed=123)
    
    final_consensus = consensus_metrics(traj_test[end])[:consensus]
    final_std = std(traj_test[end])
    
    push!(param_results, (κ=κ, consensus=final_consensus, diversity=final_std))
    println("Consensus: $(round(final_consensus, digits=3))")
end

# Plot parameter effects
κ_plot = [r.κ for r in param_results]
consensus_plot = [r.consensus for r in param_results]
diversity_plot = [r.diversity for r in param_results]

p1 = plot(κ_plot, consensus_plot, marker=:circle, lw=2, 
          xlabel="Cognitive Cost (κ)", ylabel="Final Consensus",
          title="Consensus vs Cognitive Cost", legend=false)

p2 = plot(κ_plot, diversity_plot, marker=:square, lw=2, color=:red,
          xlabel="Cognitive Cost (κ)", ylabel="Belief Diversity (Std)",
          title="Diversity vs Cognitive Cost", legend=false)

combined = plot(p1, p2, layout=(2,1), size=(600, 800))
savefig(combined, "output/demo/parameter_effects.png")
println("→ output/demo/parameter_effects.png")

# ============================================================================
# Demo 3: Simple Heterogeneous Agents
# ============================================================================
println("\n👥 Demo 3: Agent Heterogeneity")
println("-" * repeat("-", 40))

N_small = 30
# Create two types of agents
κ_stubborn = fill(2.0, N_small÷2)    # Stubborn agents
κ_flexible = fill(0.5, N_small÷2)    # Flexible agents

κ_mixed = vcat(κ_stubborn, κ_flexible)
β_values = fill(0.5, N_small)
σ_values = fill(0.3, N_small)

het_pars = HeterogeneousSimPars(N_small, κ_mixed, β_values, σ_values, 20.0, 0.01)
W_het = ring_lattice_W(N_small; k=4)

println("Running heterogeneous simulation...")
t_het, traj_het = run_heterogeneous_path(het_pars; W=W_het, seed=456)

# Compare with homogeneous
hom_pars = SimPars(N=N_small, κ=mean(κ_mixed), β=0.5, σ=0.3, T=20.0, Δt=0.01)
t_hom, traj_hom = run_one_path(hom_pars; W=W_het, seed=456)

# Plot comparison
het_mean = [mean(state) for state in traj_het]
hom_mean = [mean(state) for state in traj_hom]

p = plot(t_het, het_mean, label="Mixed Agents", lw=2)
plot!(p, t_hom, hom_mean, label="Uniform Agents", lw=2, ls=:dash)
xlabel!(p, "Time")
ylabel!(p, "Mean Belief")
title!(p, "Heterogeneous vs Homogeneous Dynamics")
savefig(p, "output/demo/heterogeneous_simple.png")
println("→ output/demo/heterogeneous_simple.png")

println("\nFinal states:")
println("  Heterogeneous mean: $(round(het_mean[end], digits=3))")
println("  Homogeneous mean: $(round(hom_mean[end], digits=3))")

# ============================================================================
# Demo 4: Basic Analysis Metrics
# ============================================================================
println("\n📈 Demo 4: Analysis Metrics")
println("-" * repeat("-", 40))

# Use one of our previous simulations
sample_traj = traj_het
final_beliefs = sample_traj[end]

# Calculate various metrics
consensus_data = consensus_metrics(final_beliefs)
polarization_data = polarization_metrics(final_beliefs)

println("Analysis of final belief state:")
println("  Mean belief: $(round(consensus_data[:mean_belief], digits=3))")
println("  Consensus level: $(round(consensus_data[:consensus], digits=3))")
println("  Disagreement (std): $(round(consensus_data[:disagreement], digits=3))")
println("  Group polarization: $(round(polarization_data[:group_polarization], digits=3))")
println("  Extremism index: $(round(polarization_data[:extremism], digits=3))")

# Show belief distribution
histogram(final_beliefs, bins=15, alpha=0.7, 
          xlabel="Final Belief Value", ylabel="Number of Agents",
          title="Final Belief Distribution", legend=false)
savefig("output/demo/belief_distribution.png")
println("→ output/demo/belief_distribution.png")

# ============================================================================
# Demo Summary
# ============================================================================
println("\n✅ Simple Demo Complete!")
println("=" * repeat("=", 40))
println("📁 Generated files in output/demo/:")

demo_files = [
    "Fully_Connected_simple.png",
    "Small_World_simple.png", 
    "Scale_Free_simple.png",
    "parameter_effects.png",
    "heterogeneous_simple.png",
    "belief_distribution.png"
]

for file in demo_files
    println("  • $file")
end

println("\n🎯 Key Observations:")
println("• Network structure affects consensus formation speed")
println("• Higher cognitive cost (κ) preserves belief diversity") 
println("• Heterogeneous agents can show different dynamics")
println("• Multiple analysis metrics reveal different aspects")

println("\n🚀 Next Steps:")
println("• Try 'julia bs.jl' for basic simulation")
println("• Run 'julia scripts/advanced_analysis.jl' for comprehensive analysis")
println("• Modify parameters in this script to explore further")
println("• Check README.md for detailed documentation")

println("\n🌊 Simple demo completed successfully!")