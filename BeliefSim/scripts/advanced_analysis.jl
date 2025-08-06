#!/usr/bin/env julia
# advanced_analysis.jl - Comprehensive analysis suite for BeliefSim

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("../src/Kernel.jl"); using .Kernel
include("../src/Metrics.jl"); using .Metrics  
include("../src/Viz.jl"); using .Viz
using CSV, DataFrames, Statistics, Random, Plots

# Create output directory
mkpath("output/advanced")

println("🚀 Starting Advanced Analysis Suite...")

# ============================================================================
# 1. Network Topology Comparison
# ============================================================================

function compare_network_topologies()
    println("\n📊 Comparing Network Topologies...")
    
    N = 100
    pars = SimPars(N=N, κ=1.2, β=0.5, σ=0.3, T=30.0, Δt=0.01)
    
    # Define different network types
    networks = Dict(
        :fully_connected => fully_connected(N),
        :small_world => watts_strogatz_W(N; k=6, p=0.3),
        :scale_free => barabasi_albert_W(N; m=3),
        :random => erdos_renyi_W(N; p=0.05),
        :caveman => caveman_W(N; group_size=20, p_within=0.8, p_between=0.02)
    )
    
    results = DataFrame()
    
    for (net_name, W) in networks
        println("  Processing: $net_name")
        
        # Run multiple simulations
        consensus_vals = Float64[]
        polarization_vals = Float64[]
        sync_vals = Float64[]
        
        for seed in 1:20
            t_vec, traj = run_one_path(pars; W=W, seed=seed)
            
            # Final state analysis
            final_beliefs = traj[end]
            consensus_data = consensus_metrics(final_beliefs)
            polarization_data = polarization_metrics(final_beliefs)
            sync_data = synchronization_metrics(traj)
            
            push!(consensus_vals, consensus_data[:consensus])
            push!(polarization_vals, polarization_data[:group_polarization])
            push!(sync_vals, sync_data[:final_synchronization])
        end
        
        # Store summary statistics
        push!(results, (
            network = net_name,
            mean_consensus = mean(consensus_vals),
            std_consensus = std(consensus_vals),
            mean_polarization = mean(polarization_vals),
            mean_synchronization = mean(sync_vals)
        ))
        
        # Generate network visualization
        network_graph_plot(W; fname="output/advanced/network_$(net_name).png")
    end
    
    # Save results
    CSV.write("output/advanced/network_comparison.csv", results)
    
    # Create comparison plot
    p = groupedbar([results.mean_consensus results.mean_polarization results.mean_synchronization],
                   group=repeat(["Consensus", "Polarization", "Synchronization"], outer=nrow(results)),
                   xlabel="Network Type", ylabel="Metric Value",
                   title="Network Topology Comparison",
                   xticks=(1:nrow(results), string.(results.network)))
    savefig(p, "output/advanced/network_comparison.png")
    
    println("→ Network topology comparison complete")
end

# ============================================================================
# 2. Cognitive Cost Function Analysis
# ============================================================================

function analyze_cognitive_cost_functions()
    println("\n🧠 Analyzing Cognitive Cost Functions...")
    
    N = 80
    pars = SimPars(N=N, κ=1.0, β=0.5, σ=0.2, T=25.0, Δt=0.01)
    W = watts_strogatz_W(N; k=4, p=0.2)
    
    cost_functions = Dict(
        :linear => f_linear,
        :cubic => f_cubic, 
        :tanh => f_tanh,
        :sigmoid => f_sigmoid
    )
    
    results = DataFrame()
    
    for (func_name, cost_func) in cost_functions
        println("  Processing: $func_name")
        
        stability_count = 0
        final_entropies = Float64[]
        
        for seed in 1:15
            t_vec, traj = run_one_path(pars; W=W, seed=seed, cost_func=cost_func)
            
            # Stability analysis
            stability_data = stability_analysis(traj)
            if stability_data[:stable]
                stability_count += 1
            end
            
            # Entropy analysis
            entropy_data = entropy_metrics(traj[end])
            push!(final_entropies, entropy_data[:shannon_entropy])
        end
        
        push!(results, (
            cost_function = func_name,
            stability_rate = stability_count / 15,
            mean_entropy = mean(final_entropies),
            std_entropy = std(final_entropies)
        ))
        
        # Generate sample trajectory plot
        t_vec, sample_traj = run_one_path(pars; W=W, seed=42, cost_func=cost_func)
        trajectory_plot(t_vec, sample_traj; 
                       fname="output/advanced/trajectory_$(func_name).png")
    end
    
    CSV.write("output/advanced/cost_function_analysis.csv", results)
    
    # Visualization
    p = scatter(results.stability_rate, results.mean_entropy,
                xlabel="Stability Rate", ylabel="Mean Final Entropy",
                title="Cost Function Comparison", legend=false,
                annotations=[(results.stability_rate[i], results.mean_entropy[i], 
                            string(results.cost_function[i])) for i in 1:nrow(results)])
    savefig(p, "output/advanced/cost_function_scatter.png")
    
    println("→ Cognitive cost function analysis complete")
end

# ============================================================================
# 3. Heterogeneous Agent Analysis
# ============================================================================

function analyze_heterogeneous_agents()
    println("\n👥 Analyzing Heterogeneous Agents...")
    
    N = 100
    
    # Create heterogeneous parameters with different agent types
    rng = MersenneTwister(123)
    
    # Three groups: conservatives (high κ), moderates, progressives (low κ)
    κ_values = vcat(
        rand(rng, Normal(2.0, 0.3), N÷3),    # conservatives
        rand(rng, Normal(1.0, 0.2), N÷3),    # moderates  
        rand(rng, Normal(0.5, 0.2), N÷3)     # progressives
    )
    β_values = rand(rng, Normal(0.5, 0.1), N)
    σ_values = rand(rng, Normal(0.3, 0.05), N)
    
    het_pars = HeterogeneousSimPars(N, κ_values, β_values, σ_values, 30.0, 0.01)
    W = watts_strogatz_W(N; k=6, p=0.2)
    
    # Run heterogeneous simulation
    t_vec, het_traj = run_heterogeneous_path(het_pars; W=W, seed=42)
    
    # Compare with homogeneous case
    hom_pars = SimPars(N=N, κ=mean(κ_values), β=mean(β_values), 
                       σ=mean(σ_values), T=30.0, Δt=0.01)
    t_vec_hom, hom_traj = run_one_path(hom_pars; W=W, seed=42)
    
    # Analysis
    het_consensus = [consensus_metrics(traj)[:consensus] for traj in het_traj]
    hom_consensus = [consensus_metrics(traj)[:consensus] for traj in hom_traj]
    
    # Influence analysis
    het_influence = network_influence_metrics(het_traj, W)
    
    # Create comparison plot
    p = plot(t_vec, het_consensus, label="Heterogeneous", lw=2)
    plot!(p, t_vec_hom, hom_consensus, label="Homogeneous", lw=2, ls=:dash)
    xlabel!(p, "Time")
    ylabel!(p, "Consensus Level")
    title!(p, "Heterogeneous vs Homogeneous Agents")
    savefig(p, "output/advanced/heterogeneous_comparison.png")
    
    # Influence network plot
    influence_network_plot(W, het_influence[:influence_scores];
                          fname="output/advanced/influence_network.png")
    
    # Save agent parameters and final beliefs
    agent_data = DataFrame(
        agent_id = 1:N,
        kappa = κ_values,
        beta = β_values,
        sigma = σ_values,
        final_belief = het_traj[end],
        influence_score = het_influence[:influence_scores]
    )
    CSV.write("output/advanced/agent_heterogeneity.csv", agent_data)
    
    println("→ Heterogeneous agent analysis complete")
end

# ============================================================================
# 4. Time-Varying Parameter Analysis
# ============================================================================

function analyze_time_varying_parameters()
    println("\n⏰ Analyzing Time-Varying Parameters...")
    
    N = 60
    T = 40.0
    
    # Define time-varying functions
    κ_decreasing(t) = 2.0 * exp(-t/10)  # Cognitive cost decreases over time
    β_increasing(t) = 0.2 + 0.6 * (1 - exp(-t/15))  # Social influence increases
    σ_constant(t) = 0.3
    
    tv_pars = TimeVaryingPars(N, κ_decreasing, β_increasing, σ_constant, T, 0.01)
    W = watts_strogatz_W(N; k=4, p=0.3)
    
    # Run time-varying simulation
    t_vec, tv_traj = run_time_varying_path(tv_pars; W=W, seed=42)
    
    # Compare with constant parameters
    const_pars = SimPars(N=N, κ=1.0, β=0.5, σ=0.3, T=T, Δt=0.01)
    t_vec_const, const_traj = run_one_path(const_pars; W=W, seed=42)
    
    # Analysis over time
    tv_consensus = [consensus_metrics(traj)[:consensus] for traj in tv_traj]
    const_consensus = [consensus_metrics(traj)[:consensus] for traj in const_traj]
    
    tv_polarization = [polarization_metrics(traj)[:group_polarization] for traj in tv_traj]
    const_polarization = [polarization_metrics(traj)[:group_polarization] for traj in const_traj]
    
    # Create multi-panel plot
    p1 = plot(t_vec, tv_consensus, label="Time-varying", lw=2)
    plot!(p1, t_vec_const, const_consensus, label="Constant", lw=2, ls=:dash)
    ylabel!(p1, "Consensus")
    title!(p1, "Consensus Evolution")
    
    p2 = plot(t_vec, tv_polarization, label="Time-varying", lw=2, color=:red)
    plot!(p2, t_vec_const, const_polarization, label="Constant", lw=2, ls=:dash, color=:red)
    xlabel!(p2, "Time")
    ylabel!(p2, "Polarization")
    title!(p2, "Polarization Evolution")
    
    # Parameter evolution
    p3 = plot(t_vec, κ_decreasing.(t_vec), label="κ(t)", lw=2)
    plot!(p3, t_vec, β_increasing.(t_vec), label="β(t)", lw=2)
    xlabel!(p3, "Time")
    ylabel!(p3, "Parameter Value")
    title!(p3, "Parameter Evolution")
    
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 900))
    savefig(combined_plot, "output/advanced/time_varying_analysis.png")
    
    println("→ Time-varying parameter analysis complete")
end

# ============================================================================
# 5. Phase Space and Stability Analysis
# ============================================================================

function phase_space_stability_analysis()
    println("\n🌊 Phase Space and Stability Analysis...")
    
    N = 50
    pars = SimPars(N=N, κ=1.5, β=0.4, σ=0.2, T=50.0, Δt=0.001)  # Higher resolution
    W = ring_lattice_W(N; k=4)
    
    # Generate trajectory with high temporal resolution
    t_vec, traj = run_one_path(pars; W=W, seed=42, save_interval=0.1)
    
    # Phase space analysis
    phase_data = phase_space_analysis(traj, 0.1)
    
    # Extract phase space coordinates for visualization (first agent)
    beliefs = [state[1] for state in traj[1:end-1]]
    velocities = [v[1] for v in phase_data[:velocity_trajectory]]
    
    # Create phase space plot
    phase_space_plot(beliefs, velocities; fname="output/advanced/phase_space.png")
    
    # Stability analysis
    stability_data = stability_analysis(traj, 20)
    
    # Lyapunov analysis over parameter range
    κ_range = 0.5:0.2:2.5
    lyapunov_estimates = Float64[]
    
    for κ in κ_range
        test_pars = SimPars(N=N, κ=κ, β=0.4, σ=0.2, T=20.0, Δt=0.01)
        t_test, traj_test = run_one_path(test_pars; W=W, seed=42, save_interval=0.1)
        phase_test = phase_space_analysis(traj_test, 0.1)
        push!(lyapunov_estimates, phase_test[:lyapunov_estimate])
    end
    
    # Lyapunov spectrum plot
    p = plot(κ_range, lyapunov_estimates, xlabel="κ", ylabel="Lyapunov Estimate",
             title="Stability Analysis", lw=2, marker=:circle, legend=false)
    hline!(p, [0], ls=:dash, color=:red, alpha=0.5)
    savefig(p, "output/advanced/lyapunov_analysis.png")
    
    # Save phase space data
    phase_df = DataFrame(
        belief = beliefs,
        velocity = velocities
    )
    CSV.write("output/advanced/phase_space_data.csv", phase_df)
    
    println("→ Phase space and stability analysis complete")
    println("  Lyapunov estimate: $(phase_data[:lyapunov_estimate])")
    println("  System stability: $(stability_data[:stable])")
end

# ============================================================================
# Main execution
# ============================================================================

function main()
    # Run all analyses
    compare_network_topologies()
    analyze_cognitive_cost_functions() 
    analyze_heterogeneous_agents()
    analyze_time_varying_parameters()
    phase_space_stability_analysis()
    
    println("\n✅ Advanced Analysis Complete!")
    println("📁 Results saved in output/advanced/")
    
    # Generate summary report
    println("\n📋 Generating Summary Report...")
    
    summary = """
    # BeliefSim Advanced Analysis Summary
    
    ## Analyses Completed:
    1. **Network Topology Comparison** - Tested 5 different network structures
    2. **Cognitive Cost Functions** - Compared 4 different cost functions
    3. **Heterogeneous Agents** - Mixed agent types with different parameters
    4. **Time-Varying Parameters** - Dynamic system parameters over time
    5. **Phase Space Analysis** - Stability and dynamical systems analysis
    
    ## Key Files Generated:
    - network_comparison.csv - Network topology performance metrics
    - cost_function_analysis.csv - Cognitive cost function comparison
    - agent_heterogeneity.csv - Individual agent characteristics and outcomes
    - phase_space_data.csv - Phase space trajectory data
    
    ## Visualizations:
    - Network structure plots for each topology
    - Trajectory comparisons across conditions
    - Phase space plots and stability analysis
    - Influence network visualizations
    
    Generated on: $(now())
    """
    
    write("output/advanced/ANALYSIS_SUMMARY.md", summary)
    println("→ Summary report saved")
end

# Run the analysis if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
