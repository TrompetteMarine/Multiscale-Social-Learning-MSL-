# MonteCarloAnalysis.jl - Monte Carlo parameter exploration module
"""
    MonteCarloAnalysis

Module for Monte Carlo exploration of parameter space.
Includes Latin Hypercube sampling, sensitivity analysis, and critical region identification.
"""
module MonteCarloAnalysis

using Random, Statistics, LinearAlgebra
using DataFrames, ProgressMeter
using ..Types, ..Utils

export monte_carlo_exploration, latin_hypercube_sample, sobol_sequence
export compute_sensitivities, identify_critical_regions, rank_parameter_importance

# ============================================================================
# Main Monte Carlo Function
# ============================================================================

"""
    monte_carlo_exploration(params::MonteCarloParams)

Run comprehensive Monte Carlo exploration of parameter space.

# Arguments
- `params::MonteCarloParams`: Configuration for Monte Carlo analysis

# Returns
- `MonteCarloResult`: Complete results including data, sensitivity, and critical regions
"""
function monte_carlo_exploration(params::MonteCarloParams)
    # Generate parameter samples
    param_samples = generate_samples(
        params.param_ranges,
        params.n_samples,
        params.sampling_method
    )
    
    # Initialize results storage
    results = DataFrame()
    successful_runs = 0
    failed_runs = 0
    
    # Progress tracking
    progress = Progress(params.n_samples, desc="Monte Carlo sampling...")
    
    # Run simulations
    for i in 1:params.n_samples
        try
            # Create parameter set for this sample
            sim_params = create_params_from_sample(
                params.base_params,
                param_samples[i, :]
            )
            
            # Run simulation
            sim_result = Utils.run_single_simulation(sim_params, seed=i*100)
            
            # Build result row
            result_row = build_result_row(param_samples[i, :], sim_result)
            
            # Add to results
            push!(results, result_row)
            successful_runs += 1
            
        catch e
            @warn "Simulation $i failed: $(typeof(e))"
            failed_runs += 1
        end
        
        next!(progress)
    end
    
    println("\nMonte Carlo complete: $successful_runs successful, $failed_runs failed")
    
    # Compute analysis if we have results
    if nrow(results) > 0
        param_names = collect(keys(params.param_ranges))
        sensitivity = compute_sensitivities(results, param_names)
        critical_regions = identify_critical_regions(results)
        param_importance = rank_parameter_importance(sensitivity)
    else
        sensitivity = Dict()
        critical_regions = Dict()
        param_importance = Pair{Symbol, Float64}[]
    end
    
    return MonteCarloResult(
        results,
        sensitivity,
        critical_regions,
        param_importance,
        successful_runs,
        failed_runs
    )
end

# ============================================================================
# Sampling Methods
# ============================================================================

"""
    generate_samples(param_ranges, n_samples, method)

Generate parameter samples using specified method.

# Arguments
- `param_ranges`: Dictionary of parameter ranges
- `n_samples`: Number of samples
- `method`: Sampling method (:latin_hypercube, :random, :sobol, :grid)

# Returns
- DataFrame with parameter samples
"""
function generate_samples(param_ranges::Dict{Symbol, Tuple{Float64, Float64}}, 
                         n_samples::Int, 
                         method::Symbol)
    if method == :latin_hypercube
        return latin_hypercube_sample(param_ranges, n_samples)
    elseif method == :random
        return random_sample(param_ranges, n_samples)
    elseif method == :sobol
        return sobol_sequence(param_ranges, n_samples)
    elseif method == :grid
        return grid_sample(param_ranges, n_samples)
    else
        @warn "Unknown sampling method: $method, using Latin Hypercube"
        return latin_hypercube_sample(param_ranges, n_samples)
    end
end

"""
    latin_hypercube_sample(param_ranges, n_samples)

Generate Latin Hypercube samples for efficient parameter space coverage.
"""
function latin_hypercube_sample(param_ranges::Dict{Symbol, Tuple{Float64, Float64}}, 
                               n_samples::Int)
    samples = DataFrame()
    
    for (param_name, (min_val, max_val)) in param_ranges
        # Create stratified intervals
        intervals = range(0, 1, length=n_samples+1)
        points = Float64[]
        
        for i in 1:n_samples
            # Random point within interval
            u = intervals[i] + rand() * (intervals[i+1] - intervals[i])
            # Scale to parameter range
            val = min_val + u * (max_val - min_val)
            push!(points, val)
        end
        
        # Shuffle for Latin Hypercube property
        shuffle!(points)
        samples[!, param_name] = points
    end
    
    return samples
end

"""
    random_sample(param_ranges, n_samples)

Generate uniform random samples.
"""
function random_sample(param_ranges::Dict{Symbol, Tuple{Float64, Float64}}, 
                      n_samples::Int)
    samples = DataFrame()
    
    for (param_name, (min_val, max_val)) in param_ranges
        samples[!, param_name] = min_val .+ rand(n_samples) .* (max_val - min_val)
    end
    
    return samples
end

"""
    sobol_sequence(param_ranges, n_samples)

Generate quasi-random Sobol sequence for low-discrepancy sampling.
"""
function sobol_sequence(param_ranges::Dict{Symbol, Tuple{Float64, Float64}}, 
                       n_samples::Int)
    n_params = length(param_ranges)
    samples = DataFrame()
    
    # Simple Sobol sequence approximation using van der Corput
    for (j, (param_name, (min_val, max_val))) in enumerate(param_ranges)
        points = Float64[]
        base = prime_number(j)
        
        for i in 1:n_samples
            u = van_der_corput(i, base)
            val = min_val + u * (max_val - min_val)
            push!(points, val)
        end
        
        samples[!, param_name] = points
    end
    
    return samples
end

"""
    grid_sample(param_ranges, n_samples)

Generate regular grid samples.
"""
function grid_sample(param_ranges::Dict{Symbol, Tuple{Float64, Float64}}, 
                    n_samples::Int)
    n_params = length(param_ranges)
    points_per_dim = Int(floor(n_samples^(1/n_params)))
    
    # Create grid points for each parameter
    grid_points = []
    param_names = []
    
    for (param_name, (min_val, max_val)) in param_ranges
        push!(param_names, param_name)
        push!(grid_points, range(min_val, max_val, length=points_per_dim))
    end
    
    # Create all combinations
    samples = DataFrame()
    idx = 1
    
    for combo in Iterators.product(grid_points...)
        if idx <= n_samples
            for (i, param_name) in enumerate(param_names)
                if idx == 1
                    samples[!, param_name] = [combo[i]]
                else
                    push!(samples[!, param_name], combo[i])
                end
            end
            idx += 1
        else
            break
        end
    end
    
    return samples
end

# ============================================================================
# Sensitivity Analysis
# ============================================================================

"""
    compute_sensitivities(results::DataFrame, param_names::Vector{Symbol})

Calculate parameter sensitivities using correlation analysis.

# Returns
- Dictionary mapping parameters to sensitivity metrics
"""
function compute_sensitivities(results::DataFrame, param_names::Vector{Symbol})
    sensitivity_dict = Dict{Symbol, Dict{Symbol, Float64}}()
    
    for param in param_names
        if !(param in names(results, Symbol))
            @warn "Parameter $param not found in results"
            continue
        end
        
        param_sens = Dict{Symbol, Float64}()
        
        # Calculate correlations with outcomes
        for outcome in [:n_equilibria, :stability, :consensus, :polarization]
            if outcome in names(results, Symbol)
                param_sens[outcome] = safe_correlation(
                    results[!, param],
                    results[!, outcome]
                )
            end
        end
        
        sensitivity_dict[param] = param_sens
    end
    
    return sensitivity_dict
end

"""
    safe_correlation(x, y)

Compute correlation handling missing and NaN values safely.
"""
function safe_correlation(x, y)
    # Convert to numeric if needed
    x_num = try
        Float64.(x)
    catch
        return 0.0
    end
    
    y_num = try
        Float64.(y)
    catch
        return 0.0
    end
    
    # Find valid indices
    valid_idx = .!isnan.(x_num) .& .!isnan.(y_num) .& .!isinf.(x_num) .& .!isinf.(y_num)
    
    if sum(valid_idx) < 3
        return 0.0
    end
    
    # Compute correlation
    try
        return cor(x_num[valid_idx], y_num[valid_idx])
    catch
        return 0.0
    end
end

"""
    rank_parameter_importance(sensitivity::Dict)

Rank parameters by their total influence on system behavior.

# Returns
- Sorted vector of (parameter, importance) pairs
"""
function rank_parameter_importance(sensitivity::Dict{Symbol, Dict{Symbol, Float64}})
    if isempty(sensitivity)
        return Pair{Symbol, Float64}[]
    end
    
    importance_scores = Dict{Symbol, Float64}()
    
    for (param, sens_data) in sensitivity
        # Calculate mean absolute correlation
        correlations = Float64[]
        
        for (_, corr_val) in sens_data
            if !isnan(corr_val)
                push!(correlations, abs(corr_val))
            end
        end
        
        if !isempty(correlations)
            importance_scores[param] = mean(correlations)
        else
            importance_scores[param] = 0.0
        end
    end
    
    # Sort by importance
    return sort(collect(importance_scores), by=x->x[2], rev=true)
end

# ============================================================================
# Critical Region Identification
# ============================================================================

"""
    identify_critical_regions(results::DataFrame)

Find parameter values where qualitative transitions occur.

# Returns
- Dictionary mapping parameters to critical values
"""
function identify_critical_regions(results::DataFrame)
    critical_regions = Dict{Symbol, Vector{Float64}}()
    
    # Check each parameter
    for param in names(results, Symbol)
        # Skip non-parameter columns
        if param in [:n_equilibria, :stability, :consensus, :polarization, 
                    :regime, :lyapunov, :variance, :oscillatory]
            continue
        end
        
        # Sort by parameter
        sorted_df = sort(results, param)
        
        # Find transitions in n_equilibria
        if :n_equilibria in names(results, Symbol)
            n_eq = sorted_df.n_equilibria
            transitions = findall(diff(n_eq) .!= 0)
            
            if !isempty(transitions)
                critical_vals = Float64[]
                
                for idx in transitions
                    # Get parameter value at transition
                    val = (sorted_df[idx, param] + sorted_df[idx+1, param]) / 2
                    push!(critical_vals, val)
                end
                
                critical_regions[param] = unique(critical_vals)
            end
        end
    end
    
    return critical_regions
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
    create_params_from_sample(base_params, sample::DataFrameRow)

Create simulation parameters from a parameter sample.
"""
function create_params_from_sample(base_params, sample::DataFrameRow)
    # Extract current cognitive parameters
    cognitive_dict = Dict{Symbol, Any}(
        :λ => base_params.cognitive.λ,
        :α => base_params.cognitive.α,
        :σ => base_params.cognitive.σ,
        :δm => base_params.cognitive.δm,
        :ηw => base_params.cognitive.ηw,
        :βΘ => base_params.cognitive.βΘ,
        :Δ => base_params.cognitive.Δ,
        :m_min => base_params.cognitive.m_min,
        :m_max => base_params.cognitive.m_max,
        :w_min => base_params.cognitive.w_min,
        :w_max => base_params.cognitive.w_max,
        :Θ_min => base_params.cognitive.Θ_min,
        :Θ_max => base_params.cognitive.Θ_max,
        :ε_max => base_params.cognitive.ε_max
    )
    
    # Update with sampled values
    for col_name in names(sample)
        param_sym = Symbol(col_name)
        if haskey(cognitive_dict, param_sym)
            cognitive_dict[param_sym] = sample[col_name]
        end
    end
    
    # Create new parameters
    # This assumes BeliefSim module is available
    new_cognitive = Main.BeliefSim.CognitiveParams(; cognitive_dict...)
    
    return Main.BeliefSim.MSLParams(
        N = base_params.N,
        T = base_params.T,
        Δt = base_params.Δt,
        cognitive = new_cognitive,
        network_type = base_params.network_type,
        network_params = base_params.network_params,
        ν = base_params.ν,
        save_interval = base_params.save_interval
    )
end

"""
    build_result_row(sample::DataFrameRow, sim_result)

Build a result row combining parameters and simulation outcomes.
"""
function build_result_row(sample::DataFrameRow, sim_result)
    result_row = Dict{Symbol, Any}()
    
    # Add parameter values
    for col_name in names(sample)
        result_row[Symbol(col_name)] = sample[col_name]
    end
    
    # Add simulation results with proper types
    result_row[:n_equilibria] = Int(sim_result.n_equilibria)
    result_row[:stability] = Float64(sim_result.stability_index)
    result_row[:consensus] = Float64(sim_result.consensus)
    result_row[:polarization] = Float64(sim_result.polarization)
    result_row[:regime] = String(sim_result.regime)
    result_row[:lyapunov] = Float64(sim_result.lyapunov)
    result_row[:variance] = Float64(sim_result.variance)
    
    return result_row
end

"""
    van_der_corput(n::Int, base::Int)

Generate van der Corput sequence value.
"""
function van_der_corput(n::Int, base::Int)
    result = 0.0
    denom = 1.0
    
    while n > 0
        denom *= base
        result += (n % base) / denom
        n ÷= base
    end
    
    return result
end

"""
    prime_number(n::Int)

Get the n-th prime number (simple implementation).
"""
function prime_number(n::Int)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    return n <= length(primes) ? primes[n] : primes[end]
end

end # module MonteCarloAnalysis