module Metrics
using Statistics, StatsBase, LinearAlgebra, Distances

export ShiftPars, shift_estimator, layer_feature
export consensus_metrics, polarization_metrics, synchronization_metrics
export phase_space_analysis, stability_analysis, entropy_metrics
export correlation_analysis, network_influence_metrics

# ============================================================================ 
# Shift estimator (original functionality)
# ============================================================================ 

struct ShiftPars
    kernel::Function      # K(u)
    hN::Float64           # bandwidth h_N
end

# Default: Epanechnikov kernel, h_N = c / √N  (c chosen later)
ShiftPars(N; c = 1.06) = ShiftPars(u -> max(0, 0.75 * (1 - u^2)), c / sqrt(N))

function layer_feature(x::Vector{Float64}, ℓ::Symbol, W::AbstractMatrix)
    if ℓ == :micro
        return x
    elseif ℓ == :meso                 # mean belief of each agent's neighbourhood
        return (W * x)                # W is row-stochastic, so W*x is local mean
    elseif ℓ == :macro
        return fill(mean(x), length(x))
    else
        error("layer ℓ must be :micro, :meso or :macro")
    end
end

function shift_estimator(t_vec::Vector{Float64}, g_vec::Vector{Float64},
                         T::Float64, pars::ShiftPars)
    N = length(g_vec)
    K, h = pars.kernel, pars.hN
    return sum(i -> K((t_vec[i] - T)/h) * g_vec[i], 1:N) / (N * h)
end

# ============================================================================ 
# Consensus metrics
# ============================================================================ 

function consensus_metrics(beliefs::Vector{Float64})
    N = length(beliefs)
    μ = mean(beliefs)
    
    # Standard deviation as disagreement measure
    disagreement = std(beliefs)
    
    # Consensus strength (1 - normalized disagreement)
    consensus = 1 - (disagreement / sqrt(N))  # normalized by theoretical max
    
    # Range-based consensus
    range_consensus = 1 - (maximum(beliefs) - minimum(beliefs)) / 4  # assuming beliefs in [-2,2]
    
    return Dict(
        :mean_belief => μ,
        :disagreement => disagreement,
        :consensus => max(0, consensus),
        :range_consensus => max(0, range_consensus),
        :iqr => quantile(beliefs, 0.75) - quantile(beliefs, 0.25)
    )
end

# ============================================================================ 
# Polarization metrics
# ============================================================================ 

function polarization_metrics(beliefs::Vector{Float64}; threshold=0.1)
    N = length(beliefs)
    
    # Bimodality coefficient
    skew = skewness(beliefs)
    kurt = kurtosis(beliefs)
    bimodality = (skew^2 + 1) / (kurt + 3 * (N-1)^2 / ((N-2)*(N-3)))
    
    # Esteban-Ray polarization index
    sorted_beliefs = sort(beliefs)
    er_polarization = 0.0
    for i in 1:N, j in 1:N
        if i != j
            er_polarization += abs(sorted_beliefs[i] - sorted_beliefs[j])
        end
    end
    er_polarization /= (N * (N-1))
    
    # Group polarization (split at median)
    median_belief = median(beliefs)
    lower_group = beliefs[beliefs .<= median_belief]
    upper_group = beliefs[beliefs .> median_belief]
    
    group_polarization = if length(lower_group) > 0 && length(upper_group) > 0
        abs(mean(upper_group) - mean(lower_group))
    else
        0.0
    end
    
    return Dict(
        :bimodality => bimodality,
        :er_polarization => er_polarization,
        :group_polarization => group_polarization,
        :extremism => mean(abs.(beliefs))
    )
end

# ============================================================================ 
# Synchronization metrics
# ============================================================================ 

function synchronization_metrics(trajectory::Vector{Vector{Float64}})
    T_steps = length(trajectory)
    N = length(trajectory[1])
    
    # Order parameter (phase synchronization)
    order_params = Float64[]
    for t in 1:T_steps
        beliefs = trajectory[t]
        # Map beliefs to phases [0, 2π]
        phases = 2π * (beliefs .- minimum(beliefs)) ./ (maximum(beliefs) - minimum(beliefs) + 1e-10)
        
        # Complex order parameter
        z = mean(exp.(im * phases))
        push!(order_params, abs(z))
    end
    
    # Kuramoto order parameter time series
    kuramoto_sync = mean(order_params[end-10:end])  # final synchronization
    
    # Pairwise correlations
    belief_matrix = hcat(trajectory...)  # N × T matrix
    correlations = cor(belief_matrix')   # correlation between agents
    avg_correlation = mean(correlations[triu(ones(Bool, N, N), 1)])
    
    return Dict(
        :order_parameter_series => order_params,
        :final_synchronization => kuramoto_sync,
        :avg_pairwise_correlation => avg_correlation,
        :correlation_matrix => correlations
    )
end

# ============================================================================ 
# Phase space analysis
# ============================================================================ 

function phase_space_analysis(trajectory::Vector{Vector{Float64}}, dt::Float64)
    T_steps = length(trajectory)
    N = length(trajectory[1])
    
    # Compute velocities (discrete derivatives)
    velocities = Vector{Vector{Float64}}()
    for t in 2:T_steps
        v = (trajectory[t] - trajectory[t-1]) / dt
        push!(velocities, v)
    end
    
    # Phase space points (position, velocity)
    phase_points = [(trajectory[t], velocities[t-1]) for t in 2:T_steps]
    
    # Lyapunov exponent estimate (largest)
    if length(velocities) > 10
        # Simple estimate: log of velocity divergence
        vel_norms = [norm(v) for v in velocities]
        if all(vel_norms .> 1e-10)
            lyapunov_est = mean(log.(vel_norms[2:end] ./ vel_norms[1:end-1])) / dt
        else
            lyapunov_est = -Inf
        end
    else
        lyapunov_est = NaN
    end
    
    return Dict(
        :phase_points => phase_points,
        :lyapunov_estimate => lyapunov_est,
        :velocity_trajectory => velocities
    )
end

# ============================================================================ 
# Stability analysis
# ============================================================================ 

function stability_analysis(trajectory::Vector{Vector{Float64}}, equilibrium_window=10)
    T_steps = length(trajectory)
    
    if T_steps < equilibrium_window + 1
        return Dict(:stable => false, :equilibrium => NaN)
    end
    
    # Check for equilibrium (small changes in final steps)
    final_states = trajectory[end-equilibrium_window:end]
    changes = [norm(final_states[i+1] - final_states[i]) for i in 1:equilibrium_window]
    
    max_change = maximum(changes)
    is_stable = max_change < 0.01  # threshold for stability
    
    equilibrium_point = if is_stable
        mean(final_states)
    else
        NaN
    end
    
    return Dict(
        :stable => is_stable,
        :equilibrium => equilibrium_point,
        :max_change => max_change,
        :final_changes => changes
    )
end

# ============================================================================ 
# Entropy metrics
# ============================================================================ 

function entropy_metrics(beliefs::Vector{Float64}; bins=20)
    # Discretize beliefs for entropy calculation
    hist = fit(Histogram, beliefs, bins)
    probs = hist.weights / sum(hist.weights)
    probs = probs[probs .> 0]  # remove zero probabilities
    
    # Shannon entropy
    shannon = -sum(probs .* log2.(probs))
    
    # Normalized entropy
    max_entropy = log2(length(probs))
    normalized_entropy = shannon / max_entropy
    
    # Gini coefficient (alternative inequality measure)
    sorted_beliefs = sort(abs.(beliefs))
    n = length(sorted_beliefs)
    gini = if n > 1
        sum((2*i - n - 1) * sorted_beliefs[i] for i in 1:n) / (n * sum(sorted_beliefs))
    else
        0.0
    end
    
    return Dict(
        :shannon_entropy => shannon,
        :normalized_entropy => normalized_entropy,
        :gini_coefficient => gini,
        :effective_states => 2^shannon
    )
end

# ============================================================================ 
# Correlation analysis
# ============================================================================ 

function correlation_analysis(trajectory::Vector{Vector{Float64}}, lags::Vector{Int}=[1,2,5,10])
    T_steps = length(trajectory)
    N = length(trajectory[1])
    
    # Autocorrelation for each agent
    autocorrs = Dict{Int, Vector{Float64}}()
    
    for lag in lags
        if lag < T_steps
            corr_vals = Float64[]
            for i in 1:N
                agent_series = [trajectory[t][i] for t in 1:T_steps]
                if T_steps > lag + 1
                    lag_corr = cor(agent_series[1:end-lag], agent_series[lag+1:end])
                    push!(corr_vals, isnan(lag_corr) ? 0.0 : lag_corr)
                end
            end
            autocorrs[lag] = corr_vals
        end
    end
    
    # Cross-correlation between agents
    belief_matrix = hcat(trajectory...)  # N × T matrix
    cross_corr = cor(belief_matrix')
    
    return Dict(
        :autocorrelations => autocorrs,
        :cross_correlations => cross_corr,
        :mean_autocorr => Dict(lag => mean(vals) for (lag, vals) in autocorrs)
    )
end

# ============================================================================ 
# Network influence metrics
# ============================================================================ 

function network_influence_metrics(trajectory::Vector{Vector{Float64}}, W::AbstractMatrix)
    T_steps = length(trajectory)
    N = size(W, 1)
    
    # Centrality measures
    degrees = vec(sum(W .> 0, dims=2))
    
    # Influence of each agent on final consensus
    final_beliefs = trajectory[end]
    consensus = mean(final_beliefs)
    
    # Agent influence = correlation with final consensus weighted by degree
    influence_scores = Float64[]
    for i in 1:N
        agent_series = [trajectory[t][i] for t in 1:T_steps]
        
        # Influence = how much agent's trajectory predicts final consensus
        # weighted by network centrality
        if std(agent_series) > 1e-10
            influence = abs(cor(agent_series, [mean(trajectory[t]) for t in 1:T_steps])) * degrees[i]
        else
            influence = 0.0
        end
        push!(influence_scores, isnan(influence) ? 0.0 : influence)
    end
    
    # Network efficiency
    distances = pairwise(Euclidean(), hcat(trajectory[end]...)')
    avg_distance = mean(distances[triu(ones(Bool, N, N), 1)])
    
    return Dict(
        :influence_scores => influence_scores,
        :top_influencers => sortperm(influence_scores, rev=true)[1:min(5, N)],
        :network_efficiency => 1 / (1 + avg_distance),
        :degree_centrality => degrees
    )
end

end # module
