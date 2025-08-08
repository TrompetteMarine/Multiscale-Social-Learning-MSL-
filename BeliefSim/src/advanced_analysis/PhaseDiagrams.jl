# PhaseDiagrams.jl - Phase diagram analysis module
"""
    PhaseDiagrams

Module for 2D phase diagram analysis of the belief dynamics model.
Explores parameter space to identify different dynamical regimes.
"""
module PhaseDiagrams

using Statistics, LinearAlgebra, ProgressMeter
using ..Types, ..Utils

export run_phase_diagram, classify_regime
export find_phase_boundaries, analyze_phase_point

# ============================================================================
# Main Phase Diagram Function
# ============================================================================

"""
    run_phase_diagram(params::PhaseDiagramParams)

Generate a comprehensive 2D phase diagram exploring parameter space.

# Arguments
- `params::PhaseDiagramParams`: Configuration for phase diagram

# Returns
- `PhaseDiagramResult`: Complete phase diagram data
"""
function run_phase_diagram(params::PhaseDiagramParams)
    # Create parameter grids
    p1_vals = range(params.param1_range[1], params.param1_range[2], 
                    length=params.param1_steps)
    p2_vals = range(params.param2_range[1], params.param2_range[2], 
                    length=params.param2_steps)
    
    # Initialize result matrices
    n_equilibria = zeros(Int, params.param1_steps, params.param2_steps)
    stability_index = zeros(params.param1_steps, params.param2_steps)
    regime_map = fill("", params.param1_steps, params.param2_steps)
    consensus_strength = zeros(params.param1_steps, params.param2_steps)
    polarization_index = zeros(params.param1_steps, params.param2_steps)
    lyapunov_exponents = zeros(params.param1_steps, params.param2_steps)
    
    # Progress tracking
    total_sims = params.param1_steps * params.param2_steps
    progress = Progress(total_sims, desc="Computing phase diagram...")
    
    # Main computation loop
    for (i, p1) in enumerate(p1_vals)
        for (j, p2) in enumerate(p2_vals)
            # Analyze this parameter point
            point_results = analyze_phase_point(
                params.base_params, 
                params.param1_name => p1,
                params.param2_name => p2,
                n_realizations=params.n_realizations
            )
            
            # Store results
            n_equilibria[i,j] = point_results.n_equilibria
            stability_index[i,j] = point_results.stability
            consensus_strength[i,j] = point_results.consensus
            polarization_index[i,j] = point_results.polarization
            lyapunov_exponents[i,j] = point_results.lyapunov
            regime_map[i,j] = point_results.regime
            
            next!(progress)
        end
    end
    
    # Return structured results
    return PhaseDiagramResult(
        collect(p1_vals),
        collect(p2_vals),
        params.param1_name,
        params.param2_name,
        n_equilibria,
        stability_index,
        consensus_strength,
        polarization_index,
        regime_map,
        lyapunov_exponents
    )
end

# ============================================================================
# Phase Point Analysis
# ============================================================================

"""
    analyze_phase_point(base_params, param_updates...; n_realizations=3)

Analyze a single point in parameter space with multiple realizations.

# Returns
- NamedTuple with aggregated results
"""
function analyze_phase_point(base_params, param_updates...; n_realizations::Int=3)
    # Create parameter set for this point
    params = Utils.update_params(base_params, param_updates...)
    
    # Run multiple realizations
    results = []
    for r in 1:n_realizations
        result = Utils.run_single_simulation(params, seed=r*1000 + hash(param_updates))
        push!(results, result)
    end
    
    # Aggregate results
    n_eq_vals = [r.n_equilibria for r in results]
    
    return (
        n_equilibria = round(Int, median(n_eq_vals)),
        stability = mean([r.stability_index for r in results]),
        consensus = mean([r.consensus for r in results]),
        polarization = mean([r.polarization for r in results]),
        lyapunov = mean([r.lyapunov for r in results]),
        regime = classify_regime(results[1])
    )
end

# ============================================================================
# Regime Classification
# ============================================================================

"""
    classify_regime(result)

Classify the dynamical regime based on simulation results.

# Arguments
- `result`: Simulation result with various metrics

# Returns
- String describing the regime
"""
function classify_regime(result)
    # Classification based on paper criteria
    if result.n_equilibria == 1 && result.stability_index > 0.8
        return "Stable Consensus"
    elseif result.n_equilibria == 2
        return "Bistable"
    elseif result.n_equilibria > 2
        return "Multistable"
    elseif result.lyapunov > 0
        return "Chaotic"
    elseif result.oscillatory
        return "Oscillatory"
    else
        return "Transient"
    end
end

# ============================================================================
# Phase Boundary Detection
# ============================================================================

"""
    find_phase_boundaries(phase_result::PhaseDiagramResult; threshold=0.1)

Identify boundaries between different phases in the diagram.

# Arguments
- `phase_result`: Completed phase diagram result
- `threshold`: Sensitivity for boundary detection

# Returns
- Matrix indicating boundary locations
"""
function find_phase_boundaries(phase_result::PhaseDiagramResult; threshold::Float64=0.1)
    nx, ny = size(phase_result.n_equilibria)
    boundaries = falses(nx, ny)
    
    # Check for changes in number of equilibria
    for i in 2:nx-1
        for j in 2:ny-1
            current = phase_result.n_equilibria[i,j]
            
            # Check neighbors
            neighbors = [
                phase_result.n_equilibria[i-1,j],
                phase_result.n_equilibria[i+1,j],
                phase_result.n_equilibria[i,j-1],
                phase_result.n_equilibria[i,j+1]
            ]
            
            # Mark as boundary if different from any neighbor
            if any(n != current for n in neighbors)
                boundaries[i,j] = true
            end
            
            # Also check for regime changes
            current_regime = phase_result.regime_map[i,j]
            neighbor_regimes = [
                phase_result.regime_map[i-1,j],
                phase_result.regime_map[i+1,j],
                phase_result.regime_map[i,j-1],
                phase_result.regime_map[i,j+1]
            ]
            
            if any(r != current_regime for r in neighbor_regimes)
                boundaries[i,j] = true
            end
        end
    end
    
    return boundaries
end

"""
    extract_critical_lines(phase_result::PhaseDiagramResult)

Extract critical parameter values where transitions occur.

# Returns
- Dictionary mapping parameter names to critical values
"""
function extract_critical_lines(phase_result::PhaseDiagramResult)
    critical_lines = Dict{Symbol, Vector{Float64}}()
    
    # Find transitions along param1 axis (for each param2 value)
    critical_lines[phase_result.param1_name] = Float64[]
    for j in 1:length(phase_result.param2_values)
        n_eq_slice = phase_result.n_equilibria[:, j]
        
        for i in 2:length(n_eq_slice)
            if n_eq_slice[i] != n_eq_slice[i-1]
                # Transition found
                critical_val = (phase_result.param1_values[i] + 
                               phase_result.param1_values[i-1]) / 2
                push!(critical_lines[phase_result.param1_name], critical_val)
            end
        end
    end
    
    # Find transitions along param2 axis
    critical_lines[phase_result.param2_name] = Float64[]
    for i in 1:length(phase_result.param1_values)
        n_eq_slice = phase_result.n_equilibria[i, :]
        
        for j in 2:length(n_eq_slice)
            if n_eq_slice[j] != n_eq_slice[j-1]
                critical_val = (phase_result.param2_values[j] + 
                               phase_result.param2_values[j-1]) / 2
                push!(critical_lines[phase_result.param2_name], critical_val)
            end
        end
    end
    
    # Remove duplicates and sort
    for (param, vals) in critical_lines
        critical_lines[param] = sort(unique(vals))
    end
    
    return critical_lines
end

# ============================================================================
# Phase Space Metrics
# ============================================================================

"""
    compute_phase_diversity(phase_result::PhaseDiagramResult)

Compute diversity metrics for the phase diagram.

# Returns
- Dictionary with diversity statistics
"""
function compute_phase_diversity(phase_result::PhaseDiagramResult)
    # Count unique regimes
    unique_regimes = unique(vec(phase_result.regime_map))
    n_regimes = length(unique_regimes)
    
    # Compute regime proportions
    regime_counts = Dict{String, Int}()
    total_points = length(phase_result.regime_map)
    
    for regime in vec(phase_result.regime_map)
        regime_counts[regime] = get(regime_counts, regime, 0) + 1
    end
    
    regime_proportions = Dict(
        regime => count / total_points 
        for (regime, count) in regime_counts
    )
    
    # Compute Shannon entropy of regime distribution
    entropy = -sum(p * log(p) for p in values(regime_proportions) if p > 0)
    
    # Compute spatial correlation length
    correlation_length = estimate_correlation_length(phase_result.n_equilibria)
    
    return Dict(
        :n_regimes => n_regimes,
        :unique_regimes => unique_regimes,
        :regime_proportions => regime_proportions,
        :entropy => entropy,
        :correlation_length => correlation_length
    )
end

"""
    estimate_correlation_length(matrix::Matrix)

Estimate spatial correlation length in a 2D field.
"""
function estimate_correlation_length(matrix::Matrix)
    nx, ny = size(matrix)
    
    # Compute autocorrelation function
    correlations = Float64[]
    
    for lag in 1:min(10, nx÷2, ny÷2)
        corr_sum = 0.0
        count = 0
        
        for i in 1:nx-lag
            for j in 1:ny-lag
                corr_sum += matrix[i,j] * matrix[i+lag,j]
                corr_sum += matrix[i,j] * matrix[i,j+lag]
                count += 2
            end
        end
        
        push!(correlations, corr_sum / count)
    end
    
    # Find where correlation drops to 1/e
    if !isempty(correlations)
        threshold = correlations[1] / ℯ
        for (i, c) in enumerate(correlations)
            if c < threshold
                return Float64(i)
            end
        end
    end
    
    return Float64(length(correlations))
end

end # module PhaseDiagrams
