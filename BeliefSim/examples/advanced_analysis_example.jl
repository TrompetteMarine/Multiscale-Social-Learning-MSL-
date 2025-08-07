#!/usr/bin/env julia
# advanced_analysis_example.jl - Comprehensive analysis of BeliefSim model

using Pkg
Pkg.activate(@__DIR__)

# Load modules
include("../src/BeliefSim.jl")
include("../src/advanced_analysis.jl")

using .BeliefSim
using .AdvancedAnalysis
using Plots, Statistics, DataFrames, CSV
using ProgressMeter

println("🔬 Advanced BeliefSim Analysis")
println("=" ^ 60)
println()

# Create output directory
output_dir = "output/advanced_analysis"
mkpath(output_dir)

# ============================================================================
# 1. PHASE DIAGRAM ANALYSIS
# ============================================================================
println("📊 1. Phase Diagram Analysis")
println("-" * 40)

# Define base parameters
base_params = MSLParams(
    N = 50,  # Smaller for faster computation
    T = 30.0,
    cognitive = CognitiveParams(
        λ = 1.0,
        α = 0.5,
        σ = 0.2,
        δm = 0.1,
        ηw = 0.1,
        βΘ = 0.05
    ),
    network_type = :small_world
)

# Configure phase diagram
phase_params = PhaseDiagramParams(
    param1_name = :α,
    param1_range = (0.1, 2.0),
    param1_steps = 25,
    param2_name = :λ,
    param2_range = (0.1, 2.0),
    param2_steps = 25,
    base_params = base_params,
    analysis_time = 30.0,
    n_realizations = 3
)

println("Computing phase diagram: α ∈ [0.1, 2.0], λ ∈ [0.1, 2.0]")
println("Grid: $(phase_params.param1_steps) × $(phase_params.param2_steps)")

# Run phase diagram analysis
phase_data = run_phase_diagram(phase_params)

# Visualize phase diagram
phase_plot = plot_phase_diagram(phase_data)
savefig(phase_plot, joinpath(output_dir, "phase_diagram.png"))
println("✅ Saved: phase_diagram.png")

# Export phase diagram data
phase_df = DataFrame(
    α = repeat(phase_data[:param1_values], inner=length(phase_data[:param2_values])),
    λ = repeat(phase_data[:param2_values], outer=length(phase_data[:param1_values])),
    n_equilibria = vec(phase_data[:n_equilibria]),
    stability = vec(phase_data[:stability]),
    consensus = vec(phase_data[:consensus]),
    polarization = vec(phase_data[:polarization])
)
CSV.write(joinpath(output_dir, "phase_diagram_data.csv"), phase_df)

# Identify critical boundaries
critical_α = phase_data[:param1_values][findfirst(x -> x > 1, phase_data[:n_equilibria][:, 1])]
println("   Critical α (bifurcation): ≈ $(round(critical_α, digits=2))")

# ============================================================================
# 2. RIGOROUS BIFURCATION ANALYSIS
# ============================================================================
println("\n🔀 2. Bifurcation Analysis")
println("-" * 40)

# Configure bifurcation analysis
bifurc_params = BifurcationParams(
    param_name = :α,
    param_range = (0.0, 2.5),
    n_points = 150,
    continuation_steps = 100,
    stability_check = true,
    track_unstable = true,
    max_period = 5
)

println("Tracking branches: α ∈ [0.0, 2.5] with $(bifurc_params.n_points) points")

# Run bifurcation analysis
bifurc_data = run_bifurcation_analysis(bifurc_params, base_params)

# Plot bifurcation diagram
bifurc_plot = plot_bifurcation_2d(bifurc_data)
savefig(bifurc_plot, joinpath(output_dir, "bifurcation_diagram.png"))
println("✅ Saved: bifurcation_diagram.png")

# Report bifurcation points
if !isempty(bifurc_data[:bifurcation_points])
    println("\n   Bifurcation points detected:")
    for (point, type) in zip(bifurc_data[:bifurcation_points], 
                             bifurc_data[:bifurcation_types])
        println("     α = $(round(point, digits=3)): $type")
    end
end

# Check for period-doubling route to chaos
if !isempty(bifurc_data[:period_doublings])
    println("\n   Period-doubling cascade:")
    for pd in bifurc_data[:period_doublings]
        println("     α = $(round(pd, digits=3))")
    end
end

# ============================================================================
# 3. BASIN OF ATTRACTION ANALYSIS
# ============================================================================
println("\n🎯 3. Basin of Attraction Analysis")
println("-" * 40)

# Choose specific parameter values for basin analysis
test_α_values = [0.3, 0.7, 1.2]  # Sub-critical, near-critical, super-critical

basin_results = Dict()

for α_val in test_α_values
    println("   Analyzing basins for α = $α_val...")
    
    # Update parameters
    test_params = MSLParams(
        base_params.N, base_params.T, base_params.Δt,
        CognitiveParams(base_params.cognitive; α = α_val),
        base_params.network_type,
        base_params.network_params,
        base_params.ν,
        base_params.save_interval
    )
    
    # Configure basin analysis
    basin_params = BasinAnalysisParams(
        x_range = (-3.0, 3.0),
        y_range = (-3.0, 3.0),
        grid_resolution = 80,  # Reduced for speed
        integration_time = 50.0,
        convergence_threshold = 0.01
    )
    
    # Analyze basins
    basin_data = analyze_basins_of_attraction(basin_params, test_params)
    basin_results[α_val] = basin_data
    
    # Plot basin portrait
    basin_plot = plot_basin_portrait(basin_data)
    savefig(basin_plot, joinpath(output_dir, "basins_alpha_$(α_val).png"))
    
    # Report statistics
    println("     → $(basin_data[:n_attractors]) attractors found")
    println("     → Fractal dimension: $(round(basin_data[:fractal_dimensions], digits=3))")
    println("     → Basin sizes: $(round.(collect(values(basin_data[:basin_sizes])), digits=3))")
end

println("✅ Basin portraits saved")

# ============================================================================
# 4. MONTE CARLO PARAMETER EXPLORATION
# ============================================================================
println("\n🎲 4. Monte Carlo Parameter Exploration")
println("-" * 40)

# Define parameter ranges for exploration
param_ranges = Dict(
    :α => (0.1, 2.0),
    :λ => (0.5, 1.5),
    :σ => (0.1, 0.5),
    :δm => (0.05, 0.2),
    :ηw => (0.05, 0.2)
)

n_mc_samples = 500
println("Sampling $(n_mc_samples) parameter combinations...")
println("Parameters: $(join(keys(param_ranges), ", "))")

# Run Monte Carlo exploration
mc_results = monte_carlo_phase_exploration(n_mc_samples, param_ranges, base_params)

# Save results
CSV.write(joinpath(output_dir, "monte_carlo_results.csv"), mc_results[:data])
println("✅ Saved: monte_carlo_results.csv")

# Display sensitivity analysis
println("\n   Parameter Sensitivity (correlation with outcomes):")
println("   " * "-" * 35)
for (param, importance) in mc_results[:param_importance]
    sens = mc_results[:sensitivity][param]
    println("   $param:")
    println("     → Consensus: $(round(sens[:consensus], digits=3))")
    println("     → Stability: $(round(sens[:stability], digits=3))")
    println("     → N equilibria: $(round(sens[:n_equilibria], digits=3))")
end

# ============================================================================
# 5. ADVANCED VISUALIZATIONS
# ============================================================================
println("\n🎨 5. Creating Advanced Visualizations")
println("-" * 40)

# 5.1 3D Phase Space Projection
println("   Creating 3D phase space projection...")

# Extract data for 3D plot
α_vals = unique(phase_df.α)
λ_vals = unique(phase_df.λ)
Z_consensus = reshape(phase_df.consensus, length(λ_vals), length(α_vals))
Z_polarization = reshape(phase_df.polarization, length(λ_vals), length(α_vals))

p3d = plot(
    surface(α_vals, λ_vals, Z_consensus, 
            xlabel="α", ylabel="λ", zlabel="Consensus",
            title="Consensus Landscape", camera=(30, 30),
            c=:viridis),
    surface(α_vals, λ_vals, Z_polarization,
            xlabel="α", ylabel="λ", zlabel="Polarization",
            title="Polarization Landscape", camera=(30, 30),
            c=:thermal),
    layout=(1, 2), size=(1200, 500)
)
savefig(p3d, joinpath(output_dir, "3d_phase_landscape.png"))

# 5.2 Parameter correlation heatmap
println("   Creating parameter correlation matrix...")

# Compute correlation matrix for MC results
param_cols = collect(keys(param_ranges))
outcome_cols = [:n_equilibria, :stability, :consensus, :polarization]

cor_matrix = zeros(length(param_cols), length(outcome_cols))
for (i, param) in enumerate(param_cols)
    for (j, outcome) in enumerate(outcome_cols)
        cor_matrix[i, j] = cor(mc_results[:data][!, param], 
                              mc_results[:data][!, outcome])
    end
end

cor_plot = heatmap(
    string.(outcome_cols), string.(param_cols), cor_matrix,
    xlabel="Outcomes", ylabel="Parameters",
    title="Parameter-Outcome Correlations",
    c=:RdBu, clims=(-1, 1), size=(600, 500)
)

# Add values to heatmap
for i in 1:length(param_cols), j in 1:length(outcome_cols)
    annotate!(cor_plot, j, i, text(round(cor_matrix[i,j], digits=2), 8))
end

savefig(cor_plot, joinpath(output_dir, "parameter_correlations.png"))

# 5.3 Critical manifold visualization
println("   Identifying critical manifolds...")

# Find parameter combinations at criticality
critical_data = filter(row -> abs(row.n_equilibria - 2) < 0.1, mc_results[:data])

if nrow(critical_data) > 0
    critical_plot = scatter(
        critical_data.α, critical_data.λ,
        xlabel="α", ylabel="λ",
        title="Critical Manifold (n_eq ≈ 2)",
        label="Critical points",
        ms=3, alpha=0.6, color=:red
    )
    
    # Add background phase diagram
    contour!(critical_plot, α_vals, λ_vals, Z_consensus',
            levels=10, alpha=0.3, color=:gray, linewidth=1)
    
    savefig(critical_plot, joinpath(output_dir, "critical_manifold.png"))
end

println("✅ Advanced visualizations saved")

# ============================================================================
# 6. SUMMARY REPORT
# ============================================================================
println("\n" * "=" * 60)
println("📋 ANALYSIS SUMMARY")
println("=" * 60)

# Compile key findings
summary = Dict(
    "Phase Diagram" => Dict(
        "Parameter ranges" => "α ∈ [0.1, 2.0], λ ∈ [0.1, 2.0]",
        "Grid resolution" => "$(phase_params.param1_steps) × $(phase_params.param2_steps)",
        "Regimes found" => unique(vec(phase_data[:regime_map]))
    ),
    "Bifurcation Analysis" => Dict(
        "Number of bifurcations" => length(bifurc_data[:bifurcation_points]),
        "Types detected" => unique(bifurc_data[:bifurcation_types]),
        "Critical α" => critical_α
    ),
    "Basin Analysis" => Dict(
        "α values tested" => test_α_values,
        "Max attractors" => maximum([d[:n_attractors] for d in values(basin_results)]),
        "Fractal dimensions" => [round(d[:fractal_dimensions], digits=3) 
                                 for d in values(basin_results)]
    ),
    "Monte Carlo" => Dict(
        "Samples" => n_mc_samples,
        "Parameters varied" => collect(keys(param_ranges)),
        "Most influential" => mc_results[:param_importance][1][1]
    )
)

# Print summary
for (section, data) in summary
    println("\n$section:")
    for (key, value) in data
        println("  • $key: $value")
    end
end

# Save summary to file
open(joinpath(output_dir, "analysis_summary.txt"), "w") do io
    for (section, data) in summary
        println(io, "\n$section:")
        for (key, value) in data
            println(io, "  • $key: $value")
        end
    end
end

println("\n" * "=" * 60)
println("✅ Advanced analysis complete!")
println("📁 Results saved in: $output_dir")
println("\nKey outputs:")
println("  • phase_diagram.png - Complete 2D phase diagram")
println("  • bifurcation_diagram.png - Detailed bifurcation structure")
println("  • basins_alpha_*.png - Basin portraits for different α")
println("  • 3d_phase_landscape.png - 3D parameter landscape")
println("  • parameter_correlations.png - Sensitivity heatmap")
println("  • monte_carlo_results.csv - Full MC dataset")
println("  • analysis_summary.txt - Text summary of findings")
