# metrics.jl - Multi-scale metrics and regime classification
module Metrics

using Statistics, LinearAlgebra, StatsBase
using ..Agents

export analyze_trajectories, RegimeType, compute_shift_metrics
export consensus_strength, polarization_index, critical_peer_influence

# Regime types (Paper Section 6)
@enum RegimeType begin
    Equilibrium
    MesoBuffered
    Broadcast
    Cascade
end

"""
    compute_shift_metrics(trajectories, layer)

Compute shift metric δ̂_ℓ(T) for given layer (Equation 29).
"""
function compute_shift_metrics(trajectories::Dict, layer::Symbol)
    N = length(trajectories[:beliefs])
    T = length(trajectories[:beliefs][1])
    
    # Extract features based on layer
    if layer == :micro
        # Individual beliefs
        final_beliefs = [trajectories[:beliefs][i][end] for i in 1:N]
        return std(final_beliefs)
        
    elseif layer == :meso
        # Community averages (simplified)
        final_beliefs = [trajectories[:beliefs][i][end] for i in 1:N]
        communities = reshape(final_beliefs[1:N÷4*4], 4, :)
        community_means = mean(communities, dims=1)
        return std(community_means)
        
    elseif layer == :macro
        # Population average
        belief_means = [mean([trajectories[:beliefs][i][t] for i in 1:N]) for t in 1:T]
        return std(belief_means[end-min(10,T÷2):end])
    end
end

"""
    consensus_strength(beliefs)

Measure consensus as 1 - normalized disagreement.
"""
function consensus_strength(beliefs::Vector{Float64})
    if length(beliefs) < 2
        return 1.0
    end
    
    # Normalized standard deviation
    σ = std(beliefs)
    max_σ = maximum(beliefs) - minimum(beliefs)
    
    return max_σ > 0 ? 1.0 - σ / max_σ : 1.0
end

"""
    polarization_index(beliefs)

Compute polarization following paper methodology.
"""
function polarization_index(beliefs::Vector{Float64})
    N = length(beliefs)
    
    # Bimodality test
    sorted = sort(beliefs)
    median_val = median(sorted)
    
    lower = beliefs[beliefs .< median_val]
    upper = beliefs[beliefs .> median_val]
    
    if length(lower) < 2 || length(upper) < 2
        return 0.0
    end
    
    # Group separation
    separation = abs(mean(upper) - mean(lower))
    
    # Within-group homogeneity
    homogeneity = 1.0 - (std(lower) + std(upper)) / (2 * std(beliefs))
    
    return separation * homogeneity
end

"""
    detect_regime(trajectories, W, params)

Classify behavioral regime following Paper Section 6.
"""
function detect_regime(trajectories::Dict, W::Matrix{Float64}, params::MSLParams)
    # Detection flags μ_ℓ
    μ_micro = compute_shift_metrics(trajectories, :micro) > 0.1
    μ_meso = compute_shift_metrics(trajectories, :meso) > 0.08
    μ_macro = compute_shift_metrics(trajectories, :macro) > 0.05
    
    # Propagation indicators
    eigenvals = eigvals(W)
    spectral_gap = 1.0 - maximum(abs.(eigenvals[2:end]))
    γ_net = spectral_gap > 0.1
    
    final_beliefs = [trajectories[:beliefs][i][end] for i in 1:length(trajectories[:beliefs])]
    γ_mass = std(final_beliefs) > 0.5
    
    # Regime classification (Table 1 from paper)
    if !μ_micro && !μ_meso && !μ_macro
        return Equilibrium
    elseif μ_micro && μ_meso && !μ_macro && !γ_net
        return MesoBuffered
    elseif !μ_micro && !μ_meso && μ_macro
        return Broadcast
    elseif μ_micro && μ_meso && μ_macro && γ_net && γ_mass
        return Cascade
    else
        return Equilibrium  # Default
    end
end

"""
    critical_peer_influence(trajectories, α_values)

Estimate critical peer influence α* from bifurcation data.
"""
function critical_peer_influence(consensus_values::Vector{Float64}, α_values::Vector{Float64})
    # Find steepest drop in consensus
    gradients = diff(consensus_values)
    critical_idx = argmin(gradients)
    
    α_star = α_values[critical_idx]
    
    return (α_star = α_star, 
            gradient = gradients[critical_idx],
            bifurcation_strength = abs(gradients[critical_idx]))
end

"""
    analyze_trajectories(trajectories, W, params)

Complete multi-scale analysis of simulation results.
"""
function analyze_trajectories(trajectories::Dict, W::Matrix{Float64}, params::MSLParams)
    N = params.N
    T_steps = length(trajectories[:beliefs][1])
    
    # Time series analysis
    consensus_evolution = Float64[]
    polarization_evolution = Float64[]
    cognitive_tension_evolution = Float64[]
    
    for t in 1:T_steps
        beliefs_t = [trajectories[:beliefs][i][t] for i in 1:N]
        references_t = [trajectories[:references][i][t] for i in 1:N]
        
        push!(consensus_evolution, consensus_strength(beliefs_t))
        push!(polarization_evolution, polarization_index(beliefs_t))
        
        # Average cognitive tension |x - r|
        tensions = abs.(beliefs_t .- references_t)
        push!(cognitive_tension_evolution, mean(tensions))
    end
    
    # Final state analysis
    final_beliefs = [trajectories[:beliefs][i][end] for i in 1:N]
    
    # Regime detection
    regime = detect_regime(trajectories, W, params)
    
    # Multi-scale shifts
    shifts = Dict(
        :micro => compute_shift_metrics(trajectories, :micro),
        :meso => compute_shift_metrics(trajectories, :meso),
        :macro => compute_shift_metrics(trajectories, :macro)
    )
    
    return Dict(
        :consensus_evolution => consensus_evolution,
        :polarization_evolution => polarization_evolution,
        :cognitive_tension_evolution => cognitive_tension_evolution,
        :final_consensus => consensus_strength(final_beliefs),
        :final_polarization => polarization_index(final_beliefs),
        :regime => regime,
        :shifts => shifts,
        :mean_belief => mean(final_beliefs),
        :std_belief => std(final_beliefs)
    )
end

end # module
