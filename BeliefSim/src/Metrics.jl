module Metrics

using Statistics, StatsBase, LinearAlgebra, Distributions

export ShiftDetectionPars, RegimeClassification
export shift_estimator, detect_regime_transition
export consensus_metrics, polarization_metrics, multi_scale_analysis
export bifurcation_analysis, critical_peer_influence
export layer_shift_metrics, propagation_indicators
export mean_field_analysis, lyapunov_analysis

# ============================================================================
# Multi-Scale Shift Detection (Paper Section 5)
# ============================================================================

@kwdef struct ShiftDetectionPars
    # Kernel parameters
    kernel::Function = epanechnikov_kernel
    bandwidth_factor::Float64 = 1.06     # Silverman's rule factor
    
    # Detection thresholds
    significance_level::Float64 = 0.05
    critical_mass_threshold::Float64 = 0.1
    
    # Temporal parameters  
    observation_window::Int = 50
    stability_window::Int = 20
end

# Regime classification (Paper Section 6)
@enum RegimeType Equilibrium Buffered Broadcast Cascade

struct RegimeClassification
    regime::RegimeType
    detection_flags::Dict{Symbol, Bool}
    propagation_indicators::Dict{Symbol, Float64}
    confidence::Float64
end

# ============================================================================
# Layer-Specific Feature Extraction
# ============================================================================

function extract_layer_features(trajectories::Dict, layer::Symbol, t_idx::Int)
    """Extract features for different aggregation layers"""
    beliefs = [trajectories[:beliefs][i][t_idx] for i in 1:length(trajectories[:beliefs])]
    
    if layer == :micro
        return beliefs  # Individual beliefs
    elseif layer == :meso
        # Community-level averages (simplified to spatial clusters)
        N = length(beliefs)
        community_size = max(5, N ÷ 10)
        meso_features = Float64[]
        
        for start_idx in 1:community_size:N
            end_idx = min(start_idx + community_size - 1, N)
            community_mean = mean(beliefs[start_idx:end_idx])
            append!(meso_features, fill(community_mean, end_idx - start_idx + 1))
        end
        return meso_features[1:N]  # Ensure same length
    elseif layer == :macro
        # Global average for all agents
        global_mean = mean(beliefs)
        return fill(global_mean, length(beliefs))
    else
        throw(ArgumentError("Layer must be :micro, :meso, or :macro"))
    end
end

# ============================================================================
# Shift Estimators (Paper Section 5.1)
# ============================================================================

function epanechnikov_kernel(u::Float64)
    return abs(u) ≤ 1.0 ? 0.75 * (1.0 - u^2) : 0.0
end

function shift_estimator(t_vec::Vector{Float64}, features::Vector{Float64}, 
                        T_obs::Float64, params::ShiftDetectionPars)
    """
    Kernel-smoothed shift estimator δ̂_ℓ(T) from Equation (29)
    """
    N = length(features)
    h_N = params.bandwidth_factor / sqrt(N)
    
    estimator = 0.0
    weight_sum = 0.0
    
    for i in 1:length(t_vec)
        weight = params.kernel((t_vec[i] - T_obs) / h_N)
        estimator += weight * features[i]
        weight_sum += weight
    end
    
    return weight_sum > 0 ? estimator / weight_sum : 0.0
end

function layer_shift_metrics(trajectories::Dict, layers::Vector{Symbol}=[:micro, :meso, :macro])
    """
    Compute shift metrics for all layers over time
    """
    T_steps = length(trajectories[:beliefs][1])
    shift_data = Dict{Symbol, Vector{Float64}}()
    
    for layer in layers
        layer_shifts = Float64[]
        
        for t in 1:T_steps
            features = extract_layer_features(trajectories, layer, t)
            # Use IQR as shift metric (robust to outliers)
            shift_val = quantile(features, 0.75) - quantile(features, 0.25)
            push!(layer_shifts, shift_val)
        end
        
        shift_data[layer] = layer_shifts
    end
    
    return shift_data
end

# ============================================================================
# Consensus and Polarization Analysis (Paper Section 4)
# ============================================================================

function consensus_metrics(beliefs::Vector{Float64})
    """
    Multi-dimensional consensus analysis
    """
    N = length(beliefs)
    μ = mean(beliefs)
    σ = std(beliefs)
    
    # Consensus strength (normalized disagreement)
    max_possible_std = sqrt(N) * std(beliefs) / sqrt(N-1)  # Theoretical maximum
    consensus_strength = 1.0 - min(1.0, σ / max_possible_std)
    
    # Range-based consensus
    belief_range = maximum(beliefs) - minimum(beliefs)
    range_consensus = 1.0 - min(1.0, belief_range / 6.0)  # Assume beliefs in [-3,3]
    
    # Distribution concentration
    mad_val = median(abs.(beliefs .- median(beliefs)))
    concentration = 1.0 / (1.0 + mad_val)
    
    return Dict(
        :mean => μ,
        :std => σ,
        :consensus_strength => consensus_strength,
        :range_consensus => range_consensus,
        :concentration => concentration,
        :agreement_rate => sum(abs.(beliefs .- μ) .< σ) / N
    )
end

function polarization_metrics(beliefs::Vector{Float64})
    """
    Multi-modal polarization analysis following paper methodology
    """
    N = length(beliefs)
    
    # Bimodality coefficient
    kurt_val = kurtosis(beliefs)
    skew_val = skewness(beliefs) 
    bimodality = (skew_val^2 + 1) / (kurt_val + 3 * (N-1)^2 / ((N-2)*(N-3)))
    
    # Esteban-Ray polarization index
    er_polarization = 0.0
    for i in 1:N, j in 1:N
        if i != j
            er_polarization += abs(beliefs[i] - beliefs[j])
        end
    end
    er_polarization /= (N * (N-1))
    
    # Group polarization (median split)
    median_belief = median(beliefs)
    lower_group = beliefs[beliefs .<= median_belief]
    upper_group = beliefs[beliefs .> median_belief]
    
    group_separation = if length(lower_group) > 0 && length(upper_group) > 0
        abs(mean(upper_group) - mean(lower_group))
    else
        0.0
    end
    
    # Fragmentation index
    sorted_beliefs = sort(beliefs)
    gaps = diff(sorted_beliefs)
    fragmentation = length(gaps[gaps .> 0.5]) / (N-1)  # Fraction of large gaps
    
    return Dict(
        :bimodality => bimodality,
        :er_polarization => er_polarization, 
        :group_separation => group_separation,
        :fragmentation => fragmentation,
        :extremism => mean(abs.(beliefs)),
        :variance_ratio => var(beliefs) / (var(beliefs) + 1e-10)
    )
end

# ============================================================================
# Critical Point Analysis (Paper Section 4.1)
# ============================================================================

function critical_peer_influence(consensus_data::Vector{Float64}, κ_values::Vector{Float64})
    """
    Estimate critical peer influence κ* from bifurcation data
    """
    # Find the point where consensus drops rapidly
    consensus_gradient = diff(consensus_data)
    
    # Look for largest negative gradient (steepest consensus drop)
    critical_idx = argmin(consensus_gradient)
    κ_star = κ_values[critical_idx]
    
    # Estimate confidence interval
    gradient_threshold = quantile(abs.(consensus_gradient), 0.8)
    critical_region = findall(abs.(consensus_gradient) .>= gradient_threshold)
    
    κ_confidence = if length(critical_region) > 1
        (κ_values[minimum(critical_region)], κ_values[maximum(critical_region)])
    else
        (κ_star, κ_star)
    end
    
    return Dict(
        :κ_star => κ_star,
        :confidence_interval => κ_confidence,
        :critical_gradient => consensus_gradient[critical_idx],
        :bifurcation_strength => abs(consensus_gradient[critical_idx])
    )
end

function bifurcation_analysis(trajectories::Dict, κ_values::Vector{Float64})
    """
    Full bifurcation analysis across parameter range
    """
    consensus_vals = Float64[]
    polarization_vals = Float64[]
    stability_vals = Float64[]
    
    for traj in trajectories
        final_beliefs = [traj[:beliefs][i][end] for i in 1:length(traj[:beliefs])]
        
        consensus_data = consensus_metrics(final_beliefs)
        polarization_data = polarization_metrics(final_beliefs)
        
        push!(consensus_vals, consensus_data[:consensus_strength])
        push!(polarization_vals, polarization_data[:group_separation])
        
        # Stability: variance of final portion of trajectory
        final_window = max(1, length(traj[:beliefs][1]) - 20):length(traj[:beliefs][1])
        belief_vars = [var([traj[:beliefs][i][t] for i in 1:length(traj[:beliefs])]) 
                      for t in final_window]
        push!(stability_vals, std(belief_vars))
    end
    
    # Identify critical point
    critical_analysis = critical_peer_influence(consensus_vals, κ_values)
    
    return Dict(
        :κ_values => κ_values,
        :consensus => consensus_vals,
        :polarization => polarization_vals,
        :stability => stability_vals,
        :critical_point => critical_analysis,
        :supercritical => consensus_vals[end] < consensus_vals[1] * 0.5
    )
end

# ============================================================================
# Regime Detection (Paper Section 6)
# ============================================================================

function propagation_indicators(trajectories::Dict, W::AbstractMatrix)
    """
    Compute network mixing time and propagation conditions
    """
    N = size(W, 1)
    
    # Network mixing time (spectral gap approximation)
    eigenvals = eigvals(W)
    spectral_gap = 1.0 - maximum(real.(eigenvals[2:end]))  # Skip largest eigenvalue
    mixing_time = spectral_gap > 1e-10 ? -1.0 / log(1.0 - spectral_gap) : Inf
    
    # Fast mixing indicator
    γ_net = mixing_time < 10.0 ? 1.0 : 0.0
    
    # Mass propagation indicator (based on belief variance)
    final_beliefs = [trajectories[:beliefs][i][end] for i in 1:length(trajectories[:beliefs])]
    belief_spread = std(final_beliefs)
    critical_mass = 0.5  # Threshold for significant spread
    γ_mass = belief_spread > critical_mass ? 1.0 : 0.0
    
    return Dict(
        :mixing_time => mixing_time,
        :spectral_gap => spectral_gap,
        :γ_net => γ_net,
        :γ_mass => γ_mass,
        :belief_spread => belief_spread
    )
end

function detect_regime_transition(trajectories::Dict, W::AbstractMatrix, 
                                params::ShiftDetectionPars=ShiftDetectionPars())
    """
    Classify behavioral regime following Paper Section 6
    """
    # Compute detection flags for each layer
    shift_data = layer_shift_metrics(trajectories)
    
    # Detection thresholds (could be calibrated from data)
    threshold_micro = 0.1
    threshold_meso = 0.08
    threshold_macro = 0.05
    
    μ_micro = maximum(shift_data[:micro]) > threshold_micro
    μ_meso = maximum(shift_data[:meso]) > threshold_meso  
    μ_macro = maximum(shift_data[:macro]) > threshold_macro
    
    detection_flags = Dict(
        :micro => μ_micro,
        :meso => μ_meso,
        :macro => μ_macro
    )
    
    # Compute propagation indicators
    prop_indicators = propagation_indicators(trajectories, W)
    
    # Regime classification logic (Paper Table 1)
    regime = if !μ_micro && !μ_meso && !μ_macro
        Equilibrium
    elseif μ_micro && μ_meso && !μ_macro && prop_indicators[:γ_net] == 0.0
        Buffered
    elseif !μ_micro && !μ_meso && μ_macro
        Broadcast  
    elseif μ_micro && μ_meso && μ_macro && prop_indicators[:γ_net] == 1.0 && prop_indicators[:γ_mass] == 1.0
        Cascade
    else
        Equilibrium  # Default fallback
    end
    
    # Confidence based on clarity of detection
    confidence = if regime == Cascade
        min(prop_indicators[:γ_net], prop_indicators[:γ_mass])
    elseif regime == Buffered
        1.0 - prop_indicators[:γ_net]
    else
        0.8  # Default confidence
    end
    
    return RegimeClassification(regime, detection_flags, prop_indicators, confidence)
end

# ============================================================================
# Advanced Analysis Functions
# ============================================================================

function multi_scale_analysis(trajectories::Dict, W::AbstractMatrix, t_vec::Vector{Float64})
    """
    Comprehensive multi-scale analysis combining all metrics
    """
    N = length(trajectories[:beliefs])
    T_steps = length(trajectories[:beliefs][1])
    
    # Time series analysis
    consensus_evolution = Float64[]
    polarization_evolution = Float64[]
    cognitive_load_evolution = Float64[]
    
    for t in 1:T_steps
        # Extract beliefs at time t
        beliefs_t = [trajectories[:beliefs][i][t] for i in 1:N]
        
        # Consensus metrics
        consensus_data = consensus_metrics(beliefs_t)
        push!(consensus_evolution, consensus_data[:consensus_strength])
        
        # Polarization metrics  
        polarization_data = polarization_metrics(beliefs_t)
        push!(polarization_evolution, polarization_data[:group_separation])
        
        # Average cognitive load (memory * deliberation * threshold)
        if haskey(trajectories, :memory)
            cognitive_load = mean([trajectories[:memory][i][t] * trajectories[:deliberation][i][t] * 
                                  trajectories[:thresholds][i][t] for i in 1:N])
            push!(cognitive_load_evolution, cognitive_load)
        end
    end
    
    # Regime detection
    regime_classification = detect_regime_transition(trajectories, W)
    
    # Stability analysis
    final_window = max(1, T_steps-20):T_steps
    final_consensus = consensus_evolution[final_window]
    is_stable = std(final_consensus) < 0.01
    
    return Dict(
        :time_series => Dict(
            :consensus => consensus_evolution,
            :polarization => polarization_evolution,
            :cognitive_load => cognitive_load_evolution
        ),
        :regime => regime_classification,
        :stability => Dict(
            :is_stable => is_stable,
            :final_consensus => mean(final_consensus),
            :consensus_volatility => std(final_consensus)
        ),
        :shift_metrics => layer_shift_metrics(trajectories),
        :network_effects => propagation_indicators(trajectories, W)
    )
end

function lyapunov_analysis(trajectories::Dict, dt::Float64=0.1)
    """
    Estimate largest Lyapunov exponent for stability analysis
    """
    beliefs_series = trajectories[:beliefs]
    N = length(beliefs_series)
    T = length(beliefs_series[1])
    
    if T < 50
        return Dict(:lyapunov_exponent => NaN, :reliable => false)
    end
    
    # Compute velocity (discrete derivative) for first agent
    velocities = diff([beliefs_series[1][t] for t in 1:T]) ./ dt
    
    # Lyapunov exponent estimate
    velocity_magnitudes = abs.(velocities)
    valid_velocities = velocity_magnitudes[velocity_magnitudes .> 1e-12]
    
    if length(valid_velocities) < 10
        return Dict(:lyapunov_exponent => NaN, :reliable => false)
    end
    
    # Log-average growth rate
    lyapunov_est = mean(log.(valid_velocities[2:end] ./ valid_velocities[1:end-1])) / dt
    
    return Dict(
        :lyapunov_exponent => lyapunov_est,
        :reliable => length(valid_velocities) > 20,
        :stability_class => lyapunov_est < 0 ? :stable : :unstable
    )
end

function mean_field_analysis(trajectories_ensemble::Vector, κ_values::Vector{Float64})
    """
    Analyze convergence to mean-field limit across ensemble
    """
    ensemble_size = length(trajectories_ensemble)
    
    # Compute empirical distribution evolution
    consensus_trajectories = []
    
    for traj_data in trajectories_ensemble
        traj = traj_data.trajectories
        T_steps = length(traj[:beliefs][1])
        consensus_t = [consensus_metrics([traj[:beliefs][i][t] for i in 1:length(traj[:beliefs])])[:consensus_strength] 
                      for t in 1:T_steps]
        push!(consensus_trajectories, consensus_t)
    end
    
    # Mean-field limit (ensemble average)
    T_steps = length(consensus_trajectories[1])
    mean_field_consensus = [mean([traj[t] for traj in consensus_trajectories]) for t in 1:T_steps]
    mean_field_variance = [var([traj[t] for traj in consensus_trajectories]) for t in 1:T_steps]
    
    # Convergence rate (should be O(N^{-1/2}))
    N = length(trajectories_ensemble[1].trajectories[:beliefs])
    theoretical_variance = 1.0 / N
    empirical_variance = mean(mean_field_variance[end-10:end])  # Final variance
    
    convergence_rate = empirical_variance > 0 ? sqrt(theoretical_variance / empirical_variance) : NaN
    
    return Dict(
        :mean_field_trajectory => mean_field_consensus,
        :variance_trajectory => mean_field_variance,
        :convergence_rate => convergence_rate,
        :n_realizations => ensemble_size,
        :theoretical_rate => 1.0 / sqrt(N)
    )
end

end # module
