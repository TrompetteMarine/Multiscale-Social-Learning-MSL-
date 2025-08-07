# integration_patch.jl - Connects advanced analysis with BeliefSim
module IntegrationPatch

using ..BeliefSim
using ..AdvancedAnalysis
using Statistics, LinearAlgebra, Random

export integrate_advanced_analysis, run_integrated_simulation
export extract_qualitative_features, compute_order_parameters

"""
    run_integrated_simulation(params; seed=42)
Bridge function that runs BeliefSim simulation and extracts features for advanced analysis.
"""
function run_integrated_simulation(params::MSLParams; seed::Int=42)
    # Run the actual BeliefSim simulation
    t_vec, trajectories, analysis = BeliefSim.run_msl_simulation(params; seed=seed)
    
    # Extract qualitative features
    qualitative = extract_qualitative_features(trajectories, analysis)
    
    # Compute order parameters
    order_params = compute_order_parameters(trajectories, t_vec)
    
    return (
        n_equilibria = qualitative[:n_equilibria],
        stability_index = qualitative[:stability_index],
        consensus = order_params[:consensus],
        polarization = order_params[:polarization],
        lyapunov = qualitative[:lyapunov],
        regime = string(analysis[:regime]),
        oscillatory = qualitative[:oscillatory],
        variance = order_params[:variance],
        trajectories = trajectories,
        analysis = analysis
    )
end

"""
    extract_qualitative_features(trajectories, analysis)
Extract qualitative dynamical features from simulation results.
"""
function extract_qualitative_features(trajectories::Dict, analysis::Dict)
    N = length(trajectories[:beliefs])
    T = length(trajectories[:beliefs][1])
    
    # Detect equilibria
    eq_info = detect_equilibria_from_trajectories(trajectories)
    
    # Compute stability index
    stability_index = compute_stability_index(trajectories)
    
    # Estimate Lyapunov exponent
    lyapunov = estimate_lyapunov_exponent(trajectories)
    
    # Check for oscillations
    oscillatory = detect_oscillations(trajectories)
    
    return Dict(
        :n_equilibria => eq_info[:n_equilibria],
        :equilibrium_positions => eq_info[:positions],
        :stability_index => stability_index,
        :lyapunov => lyapunov,
        :oscillatory => oscillatory,
        :convergence_time => eq_info[:convergence_time]
    )
end

"""
    compute_order_parameters(trajectories, t_vec)
Compute consensus, polarization, and other order parameters.
"""
function compute_order_parameters(trajectories::Dict, t_vec::Vector)
    N = length(trajectories[:beliefs])
    T = length(t_vec)
    
    # Final beliefs for order parameters
    final_window = max(1, T-20):T
    final_beliefs = [mean([trajectories[:beliefs][i][t] for t in final_window]) 
                     for i in 1:N]
    
    # Consensus strength (normalized)
    belief_std = std(final_beliefs)
    max_possible_std = 3.0  # Assuming beliefs roughly in [-3, 3]
    consensus = 1.0 - min(1.0, belief_std / max_possible_std)
    
    # Polarization index
    polarization = compute_polarization_index(final_beliefs)
    
    # Variance
    variance = var(final_beliefs)
    
    # Clustering coefficient
    clustering = compute_belief_clustering(final_beliefs)
    
    return Dict(
        :consensus => consensus,
        :polarization => polarization,
        :variance => variance,
        :clustering => clustering,
        :mean_belief => mean(final_beliefs),
        :belief_range => maximum(final_beliefs) - minimum(final_beliefs)
    )
end

"""
    detect_equilibria_from_trajectories(trajectories)
Identify equilibrium states from trajectory data.
"""
function detect_equilibria_from_trajectories(trajectories::Dict; 
                                            window::Int=20, 
                                            threshold::Float64=0.01)
    N = length(trajectories[:beliefs])
    T = length(trajectories[:beliefs][1])
    
    if T < window
        return Dict(:n_equilibria => 0, :positions => Float64[], 
                   :convergence_time => NaN)
    end
    
    # Check convergence in final window
    final_beliefs_series = [[trajectories[:beliefs][i][t] for i in 1:N] 
                            for t in T-window+1:T]
    
    # Compute variance over time
    belief_variances = [var(beliefs) for beliefs in final_beliefs_series]
    
    # Check if converged (low variance in variance)
    if var(belief_variances) < threshold
        # Extract equilibrium positions
        final_beliefs = final_beliefs_series[end]
        
        # Cluster to find distinct equilibria
        equilibria = cluster_equilibria(final_beliefs, threshold * 10)
        
        # Find convergence time
        convergence_time = find_convergence_time(trajectories, threshold)
        
        return Dict(
            :n_equilibria => length(equilibria),
            :positions => equilibria,
            :convergence_time => convergence_time
        )
    else
        return Dict(:n_equilibria => 0, :positions => Float64[], 
                   :convergence_time => NaN)
    end
end

"""
    cluster_equilibria(beliefs, threshold)
Cluster belief values to identify distinct equilibrium positions.
"""
function cluster_equilibria(beliefs::Vector{Float64}, threshold::Float64)
    if isempty(beliefs)
        return Float64[]
    end
    
    sorted_beliefs = sort(beliefs)
    clusters = Vector{Float64}[]
    current_cluster = [sorted_beliefs[1]]
    
    for belief in sorted_beliefs[2:end]
        if abs(belief - mean(current_cluster)) < threshold
            push!(current_cluster, belief)
        else
            push!(clusters, current_cluster)
            current_cluster = [belief]
        end
    end
    push!(clusters, current_cluster)
    
    # Return cluster centers
    return [mean(cluster) for cluster in clusters if length(cluster) >= 2]
end

"""
    compute_stability_index(trajectories)
Compute overall stability measure from trajectories.
"""
function compute_stability_index(trajectories::Dict)
    N = length(trajectories[:beliefs])
    T = length(trajectories[:beliefs][1])
    
    if T < 50
        return 0.5  # Default for short simulations
    end
    
    # Measure convergence rate
    mid_point = T ÷ 2
    
    mid_variance = var([trajectories[:beliefs][i][mid_point] for i in 1:N])
    final_variance = var([trajectories[:beliefs][i][end] for i in 1:N])
    
    if mid_variance > 1e-10
        convergence_rate = 1.0 - final_variance / mid_variance
    else
        convergence_rate = 1.0
    end
    
    # Measure oscillation damping
    oscillation_measure = 0.0
    for i in 1:min(N, 10)  # Sample agents
        belief_series = trajectories[:beliefs][i][mid_point:end]
        oscillation_measure += compute_oscillation_strength(belief_series)
    end
    oscillation_measure /= min(N, 10)
    
    # Combined stability index
    stability = 0.7 * max(0, convergence_rate) + 0.3 * (1.0 - oscillation_measure)
    
    return clamp(stability, 0.0, 1.0)
end

"""
    estimate_lyapunov_exponent(trajectories)
Estimate largest Lyapunov exponent from trajectories.
"""
function estimate_lyapunov_exponent(trajectories::Dict; dt::Float64=0.1)
    beliefs = trajectories[:beliefs]
    N = length(beliefs)
    T = length(beliefs[1])
    
    if T < 100
        return NaN
    end
    
    # Use first few agents for estimation
    n_test = min(N, 5)
    lyapunov_estimates = Float64[]
    
    for i in 1:n_test
        belief_series = beliefs[i]
        
        # Compute discrete derivatives
        velocities = diff(belief_series) / dt
        
        # Estimate divergence rate
        valid_idx = findall(abs.(velocities) .> 1e-10)
        
        if length(valid_idx) > 10
            log_ratios = Float64[]
            for j in 2:length(valid_idx)
                ratio = abs(velocities[valid_idx[j]] / velocities[valid_idx[j-1]])
                if ratio > 0
                    push!(log_ratios, log(ratio))
                end
            end
            
            if !isempty(log_ratios)
                push!(lyapunov_estimates, mean(log_ratios) / dt)
            end
        end
    end
    
    return isempty(lyapunov_estimates) ? NaN : median(lyapunov_estimates)
end

"""
    detect_oscillations(trajectories)
Check for persistent oscillatory behavior.
"""
function detect_oscillations(trajectories::Dict; min_period::Int=2, max_period::Int=20)
    N = length(trajectories[:beliefs])
    T = length(trajectories[:beliefs][1])
    
    if T < max_period * 3
        return false
    end
    
    # Check multiple agents
    oscillating_agents = 0
    n_check = min(N, 10)
    
    for i in 1:n_check
        belief_series = trajectories[:beliefs][i][end-max_period*3:end]
        
        # Compute autocorrelation
        autocorr = compute_autocorrelation(belief_series, max_period)
        
        # Look for peaks indicating periodicity
        peaks = findall(x -> x > 0.5, autocorr[min_period:end])
        
        if !isempty(peaks)
            oscillating_agents += 1
        end
    end
    
    return oscillating_agents > n_check / 2
end

"""
    compute_autocorrelation(series, max_lag)
Compute autocorrelation function up to max_lag.
"""
function compute_autocorrelation(series::Vector{Float64}, max_lag::Int)
    n = length(series)
    if n < max_lag * 2
        return zeros(max_lag)
    end
    
    mean_val = mean(series)
    var_val = var(series)
    
    if var_val < 1e-10
        return zeros(max_lag)
    end
    
    autocorr = Float64[]
    for lag in 1:max_lag
        cov_val = mean((series[1:n-lag] .- mean_val) .* (series[1+lag:n] .- mean_val))
        push!(autocorr, cov_val / var_val)
    end
    
    return autocorr
end

"""
    compute_oscillation_strength(series)
Measure the strength of oscillations in a time series.
"""
function compute_oscillation_strength(series::Vector{Float64})
    if length(series) < 10
        return 0.0
    end
    
    # Count zero-crossings of derivative
    derivatives = diff(series)
    sign_changes = sum(diff(sign.(derivatives)) .!= 0)
    
    # Normalize by length
    oscillation_freq = sign_changes / length(series)
    
    # Compute amplitude
    amplitude = std(series)
    
    return min(1.0, oscillation_freq * amplitude * 10)
end

"""
    compute_polarization_index(beliefs)
Compute polarization measure based on distribution modality.
"""
function compute_polarization_index(beliefs::Vector{Float64})
    N = length(beliefs)
    
    if N < 10
        return 0.0
    end
    
    # Simple bimodality check
    sorted_beliefs = sort(beliefs)
    median_val = median(beliefs)
    
    lower_group = beliefs[beliefs .< median_val]
    upper_group = beliefs[beliefs .> median_val]
    
    if length(lower_group) > 0 && length(upper_group) > 0
        separation = abs(mean(upper_group) - mean(lower_group))
        
        # Normalize by total spread
        total_spread = maximum(beliefs) - minimum(beliefs)
        
        if total_spread > 0
            return separation / total_spread
        end
    end
    
    return 0.0
end

"""
    compute_belief_clustering(beliefs)
Measure degree of belief clustering.
"""
function compute_belief_clustering(beliefs::Vector{Float64})
    N = length(beliefs)
    
    if N < 3
        return 0.0
    end
    
    # Compute pairwise distances
    distances = Float64[]
    for i in 1:N-1, j in i+1:N
        push!(distances, abs(beliefs[i] - beliefs[j]))
    end
    
    # Clustering: ratio of small to large distances
    threshold = quantile(distances, 0.25)
    close_pairs = sum(distances .< threshold)
    total_pairs = length(distances)
    
    return close_pairs / total_pairs
end

"""
    find_convergence_time(trajectories, threshold)
Estimate time to convergence.
"""
function find_convergence_time(trajectories::Dict, threshold::Float64)
    N = length(trajectories[:beliefs])
    T = length(trajectories[:beliefs][1])
    
    # Compute variance at each time
    variances = Float64[]
    for t in 1:T
        beliefs_t = [trajectories[:beliefs][i][t] for i in 1:N]
        push!(variances, var(beliefs_t))
    end
    
    # Find when variance stabilizes
    window = min(10, T ÷ 10)
    
    for t in window:T-window
        recent_var = variances[t:t+window]
        if std(recent_var) / mean(recent_var) < threshold
            return t
        end
    end
    
    return T  # Didn't converge
end

"""
    integrate_advanced_analysis(base_params::MSLParams)
Main integration function that bridges BeliefSim with advanced analysis.
"""
function integrate_advanced_analysis(base_params::MSLParams)
    # Update the run_single_simulation function in AdvancedAnalysis
    AdvancedAnalysis.eval(quote
        function run_single_simulation(params; seed=42)
            return IntegrationPatch.run_integrated_simulation(params; seed=seed)
        end
    end)
    
    # Update parameter update function
    AdvancedAnalysis.eval(quote
        function update_params(base_params, updates...)
            modified = deepcopy(base_params)
            
            for (param_name, value) in updates
                if param_name in [:α, :λ, :σ, :δm, :ηw, :βΘ]
                    # Update cognitive parameters
                    cog_dict = Dict(
                        :λ => modified.cognitive.λ,
                        :α => modified.cognitive.α,
                        :σ => modified.cognitive.σ,
                        :δm => modified.cognitive.δm,
                        :ηw => modified.cognitive.ηw,
                        :βΘ => modified.cognitive.βΘ,
                        :Δ => modified.cognitive.Δ,
                        :m_min => modified.cognitive.m_min,
                        :m_max => modified.cognitive.m_max,
                        :w_min => modified.cognitive.w_min,
                        :w_max => modified.cognitive.w_max,
                        :Θ_min => modified.cognitive.Θ_min,
                        :Θ_max => modified.cognitive.Θ_max,
                        :ε_max => modified.cognitive.ε_max
                    )
                    
                    cog_dict[param_name] = value
                    modified = BeliefSim.MSLParams(
                        modified.N, modified.T, modified.Δt,
                        BeliefSim.CognitiveParams(; cog_dict...),
                        modified.network_type,
                        modified.network_params,
                        modified.ν,
                        modified.save_interval
                    )
                elseif param_name in [:N, :T, :Δt, :ν, :save_interval]
                    setfield!(modified, param_name, value)
                end
            end
            
            return modified
        end
    end)
    
    println("✅ Advanced analysis integrated with BeliefSim")
    return true
end

end # module
