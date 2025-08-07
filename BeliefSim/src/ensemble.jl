# ensemble.jl - Ensemble simulation and Monte Carlo analysis
module Ensemble

using DifferentialEquations, Distributions, Random
using Statistics, StatsBase, LinearAlgebra
using DataFrames, ProgressMeter

include("agents.jl")
include("networks.jl")
include("simulation.jl")
include("metrics.jl")

using .Agents, .Simulation, .Metrics, .Networks

export EnsembleParams, ParameterRange, QualitativeState
export run_ensemble, run_monte_carlo
export detect_equilibria, find_bifurcations, analyze_stability
export sample_parameters, latin_hypercube_sampling

# ============================================================================
# Parameter Sampling
# ============================================================================

"""
    ParameterRange
Defines the range for parameter sampling in Monte Carlo simulations.
"""
struct ParameterRange
    name::Symbol
    min::Float64
    max::Float64
    distribution::Symbol  # :uniform, :normal, :loguniform
end

"""
    EnsembleParams
Configuration for ensemble simulations.
"""
@kwdef struct EnsembleParams
    base_params::MSLParams
    n_runs::Int = 100
    param_ranges::Vector{ParameterRange} = ParameterRange[]
    parallel::Bool = true
    save_trajectories::Bool = false
    detect_equilibria::Bool = true
    detect_bifurcations::Bool = true
end

"""
    QualitativeState
Stores qualitative behavior analysis results.
"""
struct QualitativeState
    n_equilibria::Int
    equilibrium_positions::Vector{Float64}
    stability::Vector{Bool}
    regime::Symbol
    bifurcation_detected::Bool
    lyapunov_exponent::Float64
    basin_sizes::Vector{Float64}
end

# ============================================================================
# Parameter Sampling Strategies
# ============================================================================

"""
    sample_parameters(ranges, n_samples; method=:latin_hypercube)
Generate parameter samples using various sampling strategies.
"""
function sample_parameters(ranges::Vector{ParameterRange}, n_samples::Int; 
                          method::Symbol=:latin_hypercube, seed::Int=42)
    Random.seed!(seed)
    n_params = length(ranges)
    samples = zeros(n_samples, n_params)
    
    if method == :random
        # Random uniform sampling
        for (i, range) in enumerate(ranges)
            if range.distribution == :uniform
                samples[:, i] = rand(Uniform(range.min, range.max), n_samples)
            elseif range.distribution == :normal
                μ = (range.min + range.max) / 2
                σ = (range.max - range.min) / 6  # 99.7% within range
                samples[:, i] = clamp.(rand(Normal(μ, σ), n_samples), range.min, range.max)
            elseif range.distribution == :loguniform
                samples[:, i] = exp.(rand(Uniform(log(range.min), log(range.max)), n_samples))
            end
        end
        
    elseif method == :latin_hypercube
        # Latin Hypercube Sampling for better coverage
        samples = latin_hypercube_sampling(ranges, n_samples)
        
    elseif method == :grid
        # Regular grid sampling
        points_per_dim = Int(floor(n_samples^(1/n_params)))
        grid_points = [range(r.min, r.max, length=points_per_dim) for r in ranges]
        
        idx = 1
        for combo in Iterators.product(grid_points...)
            if idx <= n_samples
                samples[idx, :] = collect(combo)
                idx += 1
            else
                break
            end
        end
        
    elseif method == :sobol
        # Quasi-random Sobol sequence for low-discrepancy sampling
        samples = sobol_sequence(ranges, n_samples)
    end
    
    return samples
end

"""
    latin_hypercube_sampling(ranges, n_samples)
Implement Latin Hypercube Sampling for efficient parameter space exploration.
"""
function latin_hypercube_sampling(ranges::Vector{ParameterRange}, n_samples::Int)
    n_params = length(ranges)
    samples = zeros(n_samples, n_params)
    
    for (i, range) in enumerate(ranges)
        # Create stratified intervals
        intervals = range(0, 1, length=n_samples+1)
        
        # Random point in each interval
        points = Float64[]
        for j in 1:n_samples
            push!(points, intervals[j] + rand() * (intervals[j+1] - intervals[j]))
        end
        
        # Shuffle and scale to actual range
        shuffle!(points)
        samples[:, i] = range.min .+ points .* (range.max - range.min)
    end
    
    return samples
end

"""
    sobol_sequence(ranges, n_samples)
Generate Sobol quasi-random sequence for low-discrepancy sampling.
"""
function sobol_sequence(ranges::Vector{ParameterRange}, n_samples::Int)
    # Simplified Sobol sequence (would use Sobol.jl in practice)
    n_params = length(ranges)
    samples = zeros(n_samples, n_params)
    
    # Van der Corput sequence as approximation
    for i in 1:n_samples
        for j in 1:n_params
            samples[i, j] = van_der_corput(i, prime(j))
        end
    end
    
    # Scale to ranges
    for (j, range) in enumerate(ranges)
        samples[:, j] = range.min .+ samples[:, j] .* (range.max - range.min)
    end
    
    return samples
end

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

prime(n::Int) = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][min(n, 10)]

# ============================================================================
# Equilibrium Detection
# ============================================================================

"""
    detect_equilibria(trajectory, threshold=0.01, window=20)
Detect number and positions of equilibria from trajectory.
"""
function detect_equilibria(trajectory::Dict; threshold::Float64=0.01, window::Int=20)
    N = length(trajectory[:beliefs])
    T = length(trajectory[:beliefs][1])
    
    if T < window
        return (n_equilibria=0, positions=Float64[], stable=Bool[])
    end
    
    # Get final portion of trajectory
    final_beliefs = [trajectory[:beliefs][i][end-window+1:end] for i in 1:N]
    
    # Compute variance over time for each agent
    agent_variances = [var(beliefs) for beliefs in final_beliefs]
    
    # Agents with low variance are at equilibrium
    at_equilibrium = agent_variances .< threshold
    
    if !any(at_equilibrium)
        return (n_equilibria=0, positions=Float64[], stable=Bool[])
    end
    
    # Cluster equilibrium positions
    equilibrium_values = [mean(beliefs) for (i, beliefs) in enumerate(final_beliefs) if at_equilibrium[i]]
    
    # Use k-means style clustering to find distinct equilibria
    clusters = cluster_positions(equilibrium_values, threshold*10)
    
    # Analyze stability (simplified Jacobian eigenvalue check)
    stability = [analyze_local_stability(pos, trajectory) for pos in clusters.centers]
    
    return (n_equilibria=clusters.n_clusters, 
            positions=clusters.centers,
            stable=stability)
end

"""
    cluster_positions(values, threshold)
Cluster nearby values to identify distinct equilibria.
"""
function cluster_positions(values::Vector{Float64}, threshold::Float64)
    if isempty(values)
        return (n_clusters=0, centers=Float64[])
    end
    
    sorted_vals = sort(values)
    clusters = Vector{Float64}[]
    current_cluster = [sorted_vals[1]]
    
    for val in sorted_vals[2:end]
        if val - mean(current_cluster) < threshold
            push!(current_cluster, val)
        else
            push!(clusters, current_cluster)
            current_cluster = [val]
        end
    end
    push!(clusters, current_cluster)
    
    centers = [mean(cluster) for cluster in clusters]
    
    return (n_clusters=length(clusters), centers=centers)
end

"""
    analyze_local_stability(equilibrium_pos, trajectory)
Determine if an equilibrium is stable using local linearization.
"""
function analyze_local_stability(equilibrium_pos::Float64, trajectory::Dict)
    # Simplified stability analysis
    N = length(trajectory[:beliefs])
    T = length(trajectory[:beliefs][1])
    
    # Find trajectories that pass near this equilibrium
    distances_to_eq = Float64[]
    for t in max(1, T-100):T
        beliefs_t = [trajectory[:beliefs][i][t] for i in 1:N]
        mean_belief = mean(beliefs_t)
        push!(distances_to_eq, abs(mean_belief - equilibrium_pos))
    end
    
    # Check if distance decreases over time (stable) or increases (unstable)
    if length(distances_to_eq) > 10
        trend = cor(1:length(distances_to_eq), distances_to_eq)
        return trend < 0  # Negative correlation means converging (stable)
    end
    
    return false
end

# ============================================================================
# Bifurcation Detection
# ============================================================================

"""
    find_bifurcations(ensemble_results, param_name, param_values)
Detect bifurcation points in parameter space.
"""
function find_bifurcations(ensemble_results::Vector, param_values::Vector{Float64})
    n_equilibria_series = [r.qualitative.n_equilibria for r in ensemble_results]
    
    bifurcations = Float64[]
    bifurcation_types = Symbol[]
    
    for i in 2:length(n_equilibria_series)
        if n_equilibria_series[i] != n_equilibria_series[i-1]
            # Change in number of equilibria
            push!(bifurcations, param_values[i])
            
            if n_equilibria_series[i] > n_equilibria_series[i-1]
                # Gained equilibria
                if n_equilibria_series[i-1] == 1 && n_equilibria_series[i] == 2
                    push!(bifurcation_types, :pitchfork)
                else
                    push!(bifurcation_types, :saddle_node)
                end
            else
                # Lost equilibria
                push!(bifurcation_types, :saddle_node_reverse)
            end
        end
    end
    
    return (points=bifurcations, types=bifurcation_types)
end

"""
    detect_hopf_bifurcation(trajectory)
Check for Hopf bifurcation (transition to oscillatory behavior).
"""
function detect_hopf_bifurcation(trajectory::Dict; window::Int=50)
    N = length(trajectory[:beliefs])
    T = length(trajectory[:beliefs][1])
    
    if T < window * 2
        return false
    end
    
    # Check for periodic oscillations in final portion
    final_beliefs = [trajectory[:beliefs][1][end-window+1:end]]  # Use first agent
    
    # Compute autocorrelation
    autocorr = StatsBase.autocor(final_beliefs[1], 1:window÷2)
    
    # Look for significant peaks in autocorrelation (indicates periodicity)
    peaks = findall(x -> x > 0.5, autocorr[5:end])  # Skip early lags
    
    return !isempty(peaks)
end

# ============================================================================
# Ensemble Simulation Runner
# ============================================================================

"""
    run_ensemble(ensemble_params; show_progress=true)
Run ensemble of simulations with parameter variations.
"""
function run_ensemble(ensemble_params::EnsembleParams; show_progress::Bool=true)
    base_params = ensemble_params.base_params
    n_runs = ensemble_params.n_runs
    
    # Sample parameters
    param_samples = if !isempty(ensemble_params.param_ranges)
        sample_parameters(ensemble_params.param_ranges, n_runs)
    else
        # Use base parameters with different random seeds
        nothing
    end
    
    # Initialize results storage
    results = []
    
    # Progress bar
    prog = show_progress ? Progress(n_runs, "Running ensemble...") : nothing
    
    # Run simulations (could be parallelized with Threads.@threads)
    for run_idx in 1:n_runs
        # Create parameters for this run
        run_params = if param_samples !== nothing
            create_params_from_sample(base_params, ensemble_params.param_ranges, param_samples[run_idx, :])
        else
            base_params
        end
        
        # Generate network (could vary network params too)
        W = create_network(run_params.N, run_params.network_type, run_params.network_params)
        
        # Run simulation
        t_vec, trajectory = simulate_msl(run_params, W, run_idx)
        
        # Analyze results
        analysis = analyze_trajectories(trajectory, W, run_params)
        
        # Detect qualitative features
        qualitative = if ensemble_params.detect_equilibria
            eq_info = detect_equilibria(trajectory)
            hopf = detect_hopf_bifurcation(trajectory)
            
            QualitativeState(
                eq_info.n_equilibria,
                eq_info.positions,
                eq_info.stable,
                analysis[:regime],
                hopf,
                get(analysis, :lyapunov, NaN),
                estimate_basin_sizes(eq_info, trajectory)
            )
        else
            nothing
        end
        
        # Store results
        push!(results, (
            params = run_params,
            param_values = param_samples !== nothing ? param_samples[run_idx, :] : Float64[],
            analysis = analysis,
            qualitative = qualitative,
            trajectory = ensemble_params.save_trajectories ? trajectory : nothing
        ))
        
        show_progress && next!(prog)
    end
    
    return results
end

"""
    create_params_from_sample(base_params, ranges, sample)
Create MSLParams from parameter sample.
"""
function create_params_from_sample(base_params::MSLParams, ranges::Vector{ParameterRange}, sample::Vector{Float64})
    cognitive_dict = Dict(
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
    for (i, range) in enumerate(ranges)
        if range.name in keys(cognitive_dict)
            cognitive_dict[range.name] = sample[i]
        end
    end
    
    return MSLParams(
        N = base_params.N,
        T = base_params.T,
        Δt = base_params.Δt,
        cognitive = CognitiveParams(; cognitive_dict...),
        network_type = base_params.network_type,
        network_params = base_params.network_params,
        ν = base_params.ν,
        save_interval = base_params.save_interval
    )
end

"""
    estimate_basin_sizes(eq_info, trajectory)
Estimate basin of attraction sizes for each equilibrium.
"""
function estimate_basin_sizes(eq_info::NamedTuple, trajectory::Dict)
    if eq_info.n_equilibria == 0
        return Float64[]
    end
    
    N = length(trajectory[:beliefs])
    final_beliefs = [trajectory[:beliefs][i][end] for i in 1:N]
    
    # Assign each agent to nearest equilibrium
    basin_counts = zeros(eq_info.n_equilibria)
    
    for belief in final_beliefs
        distances = [abs(belief - pos) for pos in eq_info.positions]
        nearest_eq = argmin(distances)
        basin_counts[nearest_eq] += 1
    end
    
    return basin_counts / N  # Normalized basin sizes
end

# ============================================================================
# Monte Carlo Analysis
# ============================================================================

"""
    run_monte_carlo(param_ranges, n_samples; base_params=MSLParams())
Run Monte Carlo analysis over parameter space.
"""
function run_monte_carlo(param_ranges::Vector{ParameterRange}, n_samples::Int;
                        base_params::MSLParams=MSLParams(),
                        sampling_method::Symbol=:latin_hypercube,
                        show_progress::Bool=true)
    
    println("🎲 Monte Carlo Analysis")
    println("   Parameters: $(join([r.name for r in param_ranges], ", "))")
    println("   Samples: $n_samples")
    println("   Method: $sampling_method")
    println()
    
    # Create ensemble configuration
    ensemble_params = EnsembleParams(
        base_params = base_params,
        n_runs = n_samples,
        param_ranges = param_ranges,
        detect_equilibria = true,
        detect_bifurcations = true,
        save_trajectories = false
    )
    
    # Run ensemble
    results = run_ensemble(ensemble_params; show_progress=show_progress)
    
    # Analyze results
    mc_analysis = analyze_monte_carlo_results(results, param_ranges)
    
    return (results=results, analysis=mc_analysis)
end

"""
    analyze_monte_carlo_results(results, param_ranges)
Statistical analysis of Monte Carlo simulation results.
"""
function analyze_monte_carlo_results(results::Vector, param_ranges::Vector{ParameterRange})
    n_samples = length(results)
    n_params = length(param_ranges)
    
    # Extract key metrics
    n_equilibria = [r.qualitative.n_equilibria for r in results]
    final_consensus = [r.analysis[:final_consensus] for r in results]
    final_polarization = [r.analysis[:final_polarization] for r in results]
    regimes = [r.analysis[:regime] for r in results]
    
    # Parameter sensitivity (correlation with outcomes)
    param_matrix = hcat([r.param_values for r in results]...)'
    
    sensitivity = Dict()
    for (i, range) in enumerate(param_ranges)
        sensitivity[range.name] = Dict(
            :consensus_correlation => cor(param_matrix[:, i], final_consensus),
            :polarization_correlation => cor(param_matrix[:, i], final_polarization),
            :n_equilibria_correlation => cor(param_matrix[:, i], Float64.(n_equilibria))
        )
    end
    
    # Regime statistics
    regime_counts = countmap(regimes)
    regime_probs = Dict(k => v/n_samples for (k, v) in regime_counts)
    
    # Bifurcation detection across parameter space
    bifurcation_regions = detect_bifurcation_regions(results, param_ranges)
    
    return Dict(
        :n_equilibria_dist => countmap(n_equilibria),
        :consensus_stats => (mean=mean(final_consensus), std=std(final_consensus)),
        :polarization_stats => (mean=mean(final_polarization), std=std(final_polarization)),
        :regime_probabilities => regime_probs,
        :parameter_sensitivity => sensitivity,
        :bifurcation_regions => bifurcation_regions
    )
end

"""
    detect_bifurcation_regions(results, param_ranges)
Identify regions in parameter space where bifurcations occur.
"""
function detect_bifurcation_regions(results::Vector, param_ranges::Vector{ParameterRange})
    # Simplified: identify parameter values where n_equilibria changes
    n_equilibria = [r.qualitative.n_equilibria for r in results]
    param_matrix = hcat([r.param_values for r in results]...)'
    
    bifurcation_regions = Dict()
    
    for (i, range) in enumerate(param_ranges)
        param_vals = param_matrix[:, i]
        
        # Sort by parameter value
        sorted_idx = sortperm(param_vals)
        sorted_params = param_vals[sorted_idx]
        sorted_n_eq = n_equilibria[sorted_idx]
        
        # Find transitions
        transitions = Float64[]
        for j in 2:length(sorted_n_eq)
            if sorted_n_eq[j] != sorted_n_eq[j-1]
                push!(transitions, (sorted_params[j] + sorted_params[j-1])/2)
            end
        end
        
        bifurcation_regions[range.name] = transitions
    end
    
    return bifurcation_regions
end

end # module

