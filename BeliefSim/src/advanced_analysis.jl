# AdvancedAnalysis.jl - Main advanced analysis module
"""
    AdvancedAnalysis

Main module for advanced analysis of BeliefSim dynamics.
Coordinates submodules for phase diagrams, bifurcation analysis, 
basin mapping, and Monte Carlo exploration.

# Submodules
- `Types`: Shared type definitions
- `Utils`: Common utility functions
- `PhaseDiagrams`: 2D parameter space exploration
- `BifurcationAnalysis`: Continuation and bifurcation detection
- `BasinAnalysis`: Basin of attraction mapping
- `MonteCarloAnalysis`: Large-scale parameter exploration
- `Visualization`: Plotting and visualization functions
- `Sensitivity`: Sensitivity analysis utilities

# Usage
```julia
using AdvancedAnalysis

# Phase diagram
params = PhaseDiagramParams(param1_name=:α, param2_name=:λ)
result = run_phase_diagram(params)

# Monte Carlo
mc_params = MonteCarloParams(n_samples=1000, param_ranges=ranges)
mc_result = monte_carlo_exploration(mc_params)
```
"""
module AdvancedAnalysis

# Load submodules
include("advanced_analysis/Types.jl")
include("advanced_analysis/Utils.jl")
include("advanced_analysis/PhaseDiagrams.jl")
include("advanced_analysis/BifurcationAnalysis.jl")
include("advanced_analysis/BasinAnalysis.jl")
include("advanced_analysis/MonteCarloAnalysis.jl")
include("advanced_analysis/Sensitivity.jl")
include("advanced_analysis/Visualization.jl")

# Import submodules
using .Types
using .Utils
using .PhaseDiagrams
using .BifurcationAnalysis
using .BasinAnalysis
using .MonteCarloAnalysis
using .Sensitivity
using .Visualization

# Re-export main types and functions
# From Types
export PhaseDiagramParams, BifurcationParams, BasinAnalysisParams, MonteCarloParams
export PhaseDiagramResult, BifurcationResult, BasinResult, MonteCarloResult
export SimulationSummary

# From Utils
export update_params, run_single_simulation
export find_fixed_points, check_stability, compute_jacobian

# From PhaseDiagrams
export run_phase_diagram, classify_regime, find_phase_boundaries

# From BifurcationAnalysis
export run_bifurcation_analysis, detect_bifurcation_type
export continuation_analysis, find_codimension_two_points

# From BasinAnalysis
export analyze_basins_of_attraction, compute_basin_sizes
export find_basin_boundaries, estimate_fractal_dimension

# From MonteCarloAnalysis
export monte_carlo_exploration, latin_hypercube_sample
export compute_sensitivities, rank_parameter_importance

# From Sensitivity
export global_sensitivity_analysis, compute_sobol_indices
export parameter_screening, morris_method

# From Visualization
export plot_phase_diagram, plot_bifurcation_diagram
export plot_basin_portrait, plot_sensitivity_heatmap
export create_summary_report

# ============================================================================
# Module Configuration
# ============================================================================

"""
    configure_analysis(; kwargs...)

Configure global settings for advanced analysis.

# Keywords
- `parallel::Bool=false`: Enable parallel computation
- `verbose::Bool=true`: Enable verbose output
- `output_dir::String="output/advanced_analysis"`: Output directory
- `save_intermediate::Bool=false`: Save intermediate results
"""
function configure_analysis(;
    parallel::Bool = false,
    verbose::Bool = true,
    output_dir::String = "output/advanced_analysis",
    save_intermediate::Bool = false
)
    # Store configuration in module-level variables
    global ANALYSIS_CONFIG = Dict(
        :parallel => parallel,
        :verbose => verbose,
        :output_dir => output_dir,
        :save_intermediate => save_intermediate
    )
    
    # Create output directory if needed
    mkpath(output_dir)
    
    if verbose
        println("Advanced Analysis configured:")
        println("  Parallel: $parallel")
        println("  Output: $output_dir")
        println("  Save intermediate: $save_intermediate")
    end
    
    return ANALYSIS_CONFIG
end

# ============================================================================
# Integration Functions
# ============================================================================

"""
    integrate_with_beliefsim(base_params)

Integrate advanced analysis with BeliefSim simulation engine.

# Arguments
- `base_params`: MSLParams from BeliefSim

# Example
```julia
using BeliefSim, AdvancedAnalysis

params = MSLParams(N=100, T=50.0)
integrate_with_beliefsim(params)
```
"""
function integrate_with_beliefsim(base_params)
    # Override the mock run_single_simulation in Utils
    Utils.eval(quote
        function run_single_simulation(params; seed::Int=42)
            # Call the actual BeliefSim simulation
            # This requires integration_patch or direct BeliefSim call
            if isdefined(Main, :IntegrationPatch)
                return Main.IntegrationPatch.run_integrated_simulation(params; seed=seed)
            elseif isdefined(Main, :BeliefSim)
                t_vec, traj, ana = Main.BeliefSim.run_msl_simulation(params; seed=seed)
                # Convert to expected format
                return (
                    n_equilibria = get(ana, :n_equilibria, 1),
                    stability_index = get(ana, :stability_index, 0.5),
                    consensus = get(ana, :final_consensus, 0.5),
                    polarization = get(ana, :final_polarization, 0.0),
                    lyapunov = get(ana, :lyapunov, 0.0),
                    regime = string(get(ana, :regime, "Unknown")),
                    oscillatory = get(ana, :oscillatory, false),
                    variance = get(ana, :variance, 0.0),
                    convergence_time = get(ana, :convergence_time, params.T)
                )
            else
                error("BeliefSim module not found. Please load it first.")
            end
        end
    end)
    
    println("✅ Advanced Analysis integrated with BeliefSim")
    return true
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    run_complete_analysis(base_params; kwargs...)

Run a complete analysis pipeline including phase diagrams, bifurcations, and Monte Carlo.

# Arguments
- `base_params`: Base simulation parameters

# Keywords
- `phase_params::PhaseDiagramParams`: Phase diagram configuration
- `bifurc_params::BifurcationParams`: Bifurcation configuration
- `mc_params::MonteCarloParams`: Monte Carlo configuration
- `skip_phase::Bool=false`: Skip phase diagram
- `skip_bifurc::Bool=false`: Skip bifurcation analysis
- `skip_mc::Bool=false`: Skip Monte Carlo

# Returns
- Dictionary with all analysis results
"""
function run_complete_analysis(base_params;
    phase_params::Union{PhaseDiagramParams, Nothing} = nothing,
    bifurc_params::Union{BifurcationParams, Nothing} = nothing,
    mc_params::Union{MonteCarloParams, Nothing} = nothing,
    skip_phase::Bool = false,
    skip_bifurc::Bool = false,
    skip_mc::Bool = false
)
    results = Dict{Symbol, Any}()
    
    # Phase diagram
    if !skip_phase && phase_params !== nothing
        println("📊 Running phase diagram analysis...")
        results[:phase_diagram] = run_phase_diagram(phase_params)
    end
    
    # Bifurcation analysis
    if !skip_bifurc && bifurc_params !== nothing
        println("🔀 Running bifurcation analysis...")
        results[:bifurcation] = run_bifurcation_analysis(bifurc_params)
    end
    
    # Monte Carlo
    if !skip_mc && mc_params !== nothing
        println("🎲 Running Monte Carlo exploration...")
        results[:monte_carlo] = monte_carlo_exploration(mc_params)
    end
    
    # Generate summary
    results[:summary] = create_analysis_summary(results)
    
    return results
end

"""
    create_analysis_summary(results::Dict)

Create a summary of all analysis results.
"""
function create_analysis_summary(results::Dict)
    summary = Dict{Symbol, Any}()
    
    if haskey(results, :phase_diagram)
        pd = results[:phase_diagram]
        summary[:phase_diagram] = Dict(
            :n_regimes => length(unique(pd.regime_map)),
            :max_n_equilibria => maximum(pd.n_equilibria),
            :mean_consensus => mean(pd.consensus),
            :mean_polarization => mean(pd.polarization)
        )
    end
    
    if haskey(results, :bifurcation)
        bf = results[:bifurcation]
        summary[:bifurcation] = Dict(
            :n_bifurcation_points => length(bf.bifurcation_points),
            :bifurcation_types => unique(bf.bifurcation_types),
            :n_stable_branches => length(bf.stable_branches),
            :n_unstable_branches => length(bf.unstable_branches)
        )
    end
    
    if haskey(results, :monte_carlo)
        mc = results[:monte_carlo]
        summary[:monte_carlo] = Dict(
            :n_successful => mc.n_successful,
            :n_failed => mc.n_failed,
            :top_parameters => length(mc.param_importance) > 0 ? 
                               mc.param_importance[1:min(3, end)] : [],
            :n_critical_regions => length(mc.critical_regions)
        )
    end
    
    return summary
end

# ============================================================================
# Module Initialization
# ============================================================================

function __init__()
    # Set default configuration
    global ANALYSIS_CONFIG = Dict(
        :parallel => false,
        :verbose => true,
        :output_dir => "output/advanced_analysis",
        :save_intermediate => false
    )
end

end # module AdvancedAnalysis
