#!/usr/bin/env julia
# demo_example.jl - Interactive demonstration of BeliefSim capabilities

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

include("src/Kernel.jl"); using .Kernel
include("src/Metrics.jl"); using .Metrics  
include("src/Viz.jl"); using .Viz
using Statistics, Random

println("🌊 Welcome to BeliefSim Interactive Demo!")
println("=====================================\n")

# Check if advanced plotting packages are available
plotting_available = true
try
    using GraphPlot, NetworkLayout
catch e
    plotting_available = false
    println("ℹ️  Note: Advanced plotting packages not available.")
    println("   Some network visualizations will use simpler alternatives.")
    println("   For a streamlined experience, try 'julia simple_demo.jl'\n")
end

# Create output directory
mkpath("output/demo")

# ============================================================================
# Demo 1: Basic Simulation Comparison
# ============================================================================
println("📊 Demo 1: Comparing Network Effects on Consensus Formation")
println("----------------------------------------------------------")

N = 60
pars = SimPars(N=N, κ=1.0, β=0.6, σ=0.25, T=20.0, Δt=0.01)

# Three different networks
networks = Dict(
    "Fully Connected" => fully_connected(N),
    "Small World" => watts_strogatz_W(N; k=6, p=0.3),
    "Scale Free" => barabasi_albert_W(N; m=3)
)

println("Running simulations on $(length(networks)) network types...")

results = []
for (name, W) in networks
    println("  • $name network...")
    
    # Run simulation
    t_vec, traj = run_one_path(pars; W=W, seed=42)
    
    # Calculate final consensus
    final_consensus = consensus_metrics(traj[end])[:consensus]
    final_polarization = polarization_metrics(traj[end])[:group_polarization]
    
    push!(results, (network=name, consensus=final_consensus, polarization=final_polarization))
    
    # Generate individual plots
    safe_name = replace(name, " " => "_")
    trajectory_plot(t_vec, traj; fname="output/demo/$(safe_name)_trajectory.png")
    
    # Try to generate network plot, skip if it fails
    try
        network_graph_plot(W; fname="output/demo/$(safe_name)_network.png")
    catch e
        println("    (Skipping network plot due to: $e)")
        # Create simple heatmap instead
        heatmap(W, title="$name Network Adjacency", color=:blues)
        savefig("output/demo/$(safe_name)_heatmap.png")
        println("    → output/demo/$(safe_name)_heatmap.png (heatmap)")
    end
end

println("\n📋 Results:")
for r in results
    println("  $(r.network): Consensus = $(round(r.consensus, digits=3)), Polarization = $(round(r.polarization, digits=3))")
end

# ============================================================================
# Demo 2: Heterogeneous Agents
# ============================================================================
println("\n👥 Demo 2: Heterogeneous vs Homogeneous Agents")
println("---------------------------------------------")

# Create three types of agents
N_type = N ÷ 3
κ_stubborn = fill(2.5, N_type)      # Stubborn agents (high cognitive cost)
κ_moderate = fill(1.0, N_type)      # Moderate agents
κ_flexible = fill(0.3, N_type)      # Flexible agents (low cognitive cost)

κ_mixed = vcat(κ_stubborn, κ_moderate, κ_flexible)
β_values = fill(0.5, N)
σ_values = fill(0.3, N)

# Heterogeneous simulation
het_pars = HeterogeneousSimPars(N, κ_mixed, β_values, σ_values, 25.0, 0.01)
W_demo = watts_strogatz_W(N; k=4, p=0.2)

println("Running heterogeneous agent simulation...")
t_het, traj_het = run_heterogeneous_path(het_pars; W=W_demo, seed=123)

# Homogeneous comparison
hom_pars = SimPars(N=N, κ=mean(κ_mixed), β=0.5, σ=0.3, T=25.0, Δt=0.01)
println("Running homogeneous agent simulation...")
t_hom, traj_hom = run_one_path(hom_pars; W=W_demo, seed=123)

# Compare consensus evolution
het_consensus = [consensus_metrics(state)[:consensus] for state in traj_het]
hom_consensus = [consensus_metrics(state)[:consensus] for state in traj_hom]

# Create comparison plot
using Plots
p = plot(t_het, het_consensus, label="Heterogeneous Agents", lw=2, color=:blue)
plot!(p, t_hom, hom_consensus, label="Homogeneous Agents", lw=2, color=:red, ls=:dash)
xlabel!(p, "Time")
ylabel!(p, "Consensus Level")
title!(p, "Heterogeneous vs Homogeneous Agent Dynamics")
savefig(p, "output/demo/heterogeneous_comparison.png")

println("  Final consensus (heterogeneous): $(round(het_consensus[end], digits=3))")
println("  Final consensus (homogeneous): $(round(hom_consensus[end], digits=3))")

# ============================================================================
# Demo 3: Cognitive Cost Function Effects
# ============================================================================
println("\n🧠 Demo 3: Different Cognitive Cost Functions")
println("-------------------------------------------")

cost_functions = Dict(
    "Linear" => f_linear,
    "Cubic" => f_cubic,
    "Tanh" => f_tanh
)

test_pars = SimPars(N=40, κ=1.2, β=0.5, σ=0.2, T=15.0, Δt=0.01)
W_test = ring_lattice_W(40; k=4)

cost_results = []
println("Testing $(length(cost_functions)) cognitive cost functions...")

for (name, cost_func) in cost_functions
    println("  • $name cost function...")
    
    t_cost, traj_cost = run_one_path(test_pars; W=W_test, cost_func=cost_func, seed=456)
    
    # Analysis
    final_beliefs = traj_cost[end]
    stability_data = stability_analysis(traj_cost)
    entropy_data = entropy_metrics(final_beliefs)
    
    push!(cost_results, (
        function_name=name, 
        stable=stability_data[:stable],
        entropy=entropy_data[:shannon_entropy],
        spread=std(final_beliefs)
    ))
    
    # Save trajectory
    safe_name = lowercase(replace(name, " " => "_"))
    trajectory_plot(t_cost, traj_cost; fname="output/demo/$(safe_name)_cost_trajectory.png")
end

println("\n📋 Cost Function Results:")
for r in cost_results
    println("  $(r.function_name): Stable=$(r.stable), Entropy=$(round(r.entropy, digits=3)), Spread=$(round(r.spread, digits=3))")
end

# ============================================================================
# Demo 4: Time-Varying Parameters
# ============================================================================
println("\n⏰ Demo 4: Time-Varying Social Influence")
println("--------------------------------------")

# Define time-varying social influence that increases over time
β_increasing(t) = 0.2 + 0.6 * (1 - exp(-t/8))  # Starts low, increases to 0.8
κ_constant(t) = 1.0
σ_constant(t) = 0.3

tv_pars = TimeVaryingPars(50, κ_constant, β_increasing, σ_constant, 20.0, 0.01)
W_tv = erdos_renyi_W(50; p=0.08)

println("Running time-varying parameter simulation...")
t_tv, traj_tv = run_time_varying_path(tv_pars; W=W_tv, seed=789)

# Compare with constant parameters
const_pars = SimPars(N=50, κ=1.0, β=0.5, σ=0.3, T=20.0, Δt=0.01)
t_const, traj_const = run_one_path(const_pars; W=W_tv, seed=789)

# Plot parameter evolution and results
p1 = plot(t_tv, β_increasing.(t_tv), xlabel="Time", ylabel="β(t)", 
          title="Time-Varying Social Influence", lw=2, legend=false)

tv_means = [mean(state) for state in traj_tv]
const_means = [mean(state) for state in traj_const]

p2 = plot(t_tv, tv_means, label="Time-varying β", lw=2)
plot!(p2, t_const, const_means, label="Constant β", lw=2, ls=:dash)
xlabel!(p2, "Time")
ylabel!(p2, "Mean Belief")
title!(p2, "Belief Evolution")

combined = plot(p1, p2, layout=(2,1), size=(800, 600))
savefig(combined, "output/demo/time_varying_demo.png")

println("  Time-varying final mean: $(round(tv_means[end], digits=3))")
println("  Constant parameter final mean: $(round(const_means[end], digits=3))")

# ============================================================================
# Demo Summary
# ============================================================================
println("\n✅ Demo Complete! Summary of Generated Files:")
println("==========================================")
println("📁 All outputs saved to: output/demo/")
println("\nGenerated visualizations:")
demo_files = [
    "Fully_Connected_trajectory.png - Trajectory on complete network",
    "Small_World_trajectory.png - Trajectory on small-world network", 
    "Scale_Free_trajectory.png - Trajectory on scale-free network",
    "heterogeneous_comparison.png - Heterogeneous vs homogeneous agents",
    "linear_cost_trajectory.png - Linear cognitive cost dynamics",
    "cubic_cost_trajectory.png - Cubic cognitive cost dynamics", 
    "tanh_cost_trajectory.png - Tanh cognitive cost dynamics",
    "time_varying_demo.png - Time-varying parameter effects"
]

for file in demo_files
    println("  • $file")
end

println("\n🎯 Key Insights from Demo:")
println("• Network structure significantly affects consensus formation")
println("• Heterogeneous agents can show different dynamics than homogeneous ones")
println("• Cognitive cost functions shape the belief evolution process")
println("• Time-varying parameters can lead to emergent behaviors")

println("\n🚀 Next Steps:")
println("• Run 'julia scripts/advanced_analysis.jl' for comprehensive analysis")
println("• Modify parameters in this script to explore different scenarios")
println("• Check the README.md for detailed documentation")
println("• Explore the src/ modules for customization options")

println("\n🌊 Thank you for trying BeliefSim!")