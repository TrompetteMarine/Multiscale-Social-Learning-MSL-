# agents.jl - Agent dynamics following paper equations exactly
module Agents

using Parameters, Distributions, LinearAlgebra

export AgentState, MSLParams, CognitiveParams
export φ, ξ, ψw, ε, Υ
export compute_drift!, apply_jump!, check_jump_condition

# ============================================================================
# Agent State (Paper Section 2.1)
# ============================================================================

"""
    AgentState

Complete agent state Z_i(t) = (x_i, r_i, m_i, w_i, Θ_i) as in paper.
"""
mutable struct AgentState
    x::Float64   # Belief
    r::Float64   # Reference point
    m::Float64   # Memory weight
    w::Float64   # Deliberation weight
    Θ::Float64   # Cognitive tension threshold
end

AgentState() = AgentState(0.0, 0.0, 1.0, 0.5, 1.0)
AgentState(x₀::Float64) = AgentState(x₀, x₀, 1.0, 0.5, 1.0)

# ============================================================================
# Parameters
# ============================================================================

@with_kw mutable struct CognitiveParams
    # Mean reversion and social influence (Equation 2)
    λ::Float64 = 1.0      # Mean reversion strength
    α::Float64 = 0.5      # Social influence scale
    σ::Float64 = 0.3      # Idiosyncratic noise
    Δ::Float64 = 0.1      # Minimum jump slack ratio
    
    # Memory dynamics (Equations 3-4)
    δm::Float64 = 0.1     # Memory adjustment speed
    
    # Deliberation dynamics (Equation 5)
    ηw::Float64 = 0.1     # Deliberation adjustment speed
    
    # Threshold dynamics (Equation 6)
    βΘ::Float64 = 0.05    # Threshold adjustment speed
    
    # Bounds
    m_min::Float64 = 0.1
    m_max::Float64 = 2.0
    w_min::Float64 = 0.1
    w_max::Float64 = 1.0
    Θ_min::Float64 = 0.5
    Θ_max::Float64 = 3.0
    
    # Attention radius parameters
    ε_max::Float64 = 2.0
end

@with_kw struct MSLParams
    # System size and time
    N::Int = 100
    T::Float64 = 50.0
    Δt::Float64 = 0.01
    
    # Cognitive parameters
    cognitive::CognitiveParams = CognitiveParams()
    
    # Network parameters
    network_type::Symbol = :small_world
    network_params::Dict = Dict(:k => 6, :p => 0.3)
    
    # Jump intensity
    ν::Float64 = 1.0      # Jump rate when |x - r| ≥ Θ
    
    # Saving
    save_interval::Float64 = 0.1
end

# ============================================================================
# Cognitive Functions (Paper Section 2.1)
# ============================================================================

"""
    φ(m)
Memory sensitivity function - decreasing (controls reference adjustment rate)
"""
φ(m::Float64) = 1.0 / (1.0 + m)

"""
    ξ(m, Θ)
Memory reset map - increasing in Θ (memory refresh after jump)
"""
ξ(m::Float64, Θ::Float64, params::CognitiveParams) = 
    clamp(m - 0.1 * Θ, params.m_min, params.m_max)

"""
    ψw(w)
Deliberation adjustment function
"""
ψw(w::Float64) = w * (1.0 - w)

"""
    ε(Θ)
Attention radius - strictly increasing and concave
"""
ε(Θ::Float64, params::CognitiveParams) = 
    params.ε_max * (1.0 - exp(-Θ / params.ε_max))

"""
    Υ(m, w)
Optimal tension function
"""
Υ(m::Float64, w::Float64) = 0.5 + 0.3 * m + 0.2 * w

# ============================================================================
# Social Influence (Paper Section 2.2)
# ============================================================================

"""
    compute_attention_set(i, agents, params)
Compute N_i(t) = {j : |x_j - x_i| ≤ ε(Θ_i)}
"""
function compute_attention_set(i::Int, agents::Vector{AgentState}, params::CognitiveParams)
    xi = agents[i].x
    Θi = agents[i].Θ
    radius = ε(Θi, params)
    
    neighbors = Int[]
    for j in 1:length(agents)
        if i != j && abs(agents[j].x - xi) ≤ radius
            push!(neighbors, j)
        end
    end
    return neighbors
end

"""
    compute_social_influence(i, agents, W, params)
Compute Π_i(t) from Equation (8)
"""
function compute_social_influence(i::Int, agents::Vector{AgentState}, W::Matrix{Float64}, params::CognitiveParams)
    neighbors = compute_attention_set(i, agents, params)
    
    if isempty(neighbors)
        return 0.0
    end
    
    xi = agents[i].x
    Πi = 0.0
    
    for j in neighbors
        xj = agents[j].x
        distance = abs(xj - xi)
        # Kernel K(|x_j - x_i|) - using Gaussian for smoothness
        kernel_val = exp(-distance^2 / (2 * 0.5^2))
        Πi += W[i,j] * kernel_val * (xj - xi)
    end
    
    return Πi
end

"""
    compute_reference_influence(i, agents, W, params)
Compute Ψ_i(t) from Equation (9)
"""
function compute_reference_influence(i::Int, agents::Vector{AgentState}, W::Matrix{Float64}, params::CognitiveParams)
    neighbors = compute_attention_set(i, agents, params)
    
    if isempty(neighbors)
        return 0.0
    end
    
    xi = agents[i].x
    ri = agents[i].r
    Ψi = 0.0
    
    for j in neighbors
        xj = agents[j].x
        rj = agents[j].r
        distance = abs(xj - xi)
        kernel_val = exp(-distance^2 / (2 * 0.5^2))
        Ψi += W[i,j] * kernel_val * (rj - ri)
    end
    
    # Normalize by number of neighbors
    return Ψi / max(1, length(neighbors))
end

# ============================================================================
# Dynamics (Paper Equations 2-6)
# ============================================================================

"""
    compute_drift!(du, agents, W, params, t)
Compute drift for all state variables following Equations 2-6
"""
function compute_drift!(du::Vector{Float64}, agents::Vector{AgentState}, W::Matrix{Float64}, params::MSLParams, t::Float64)
    N = params.N
    cognitive = params.cognitive
    
    # Compute average reference influence for deliberation dynamics
    Ψ_bar = mean([compute_reference_influence(i, agents, W, cognitive) for i in 1:N])
    
    for i in 1:N
        idx = 5*(i-1)
        agent = agents[i]
        
        # Social influence terms
        Πi = compute_social_influence(i, agents, W, cognitive)
        Ψi = compute_reference_influence(i, agents, W, cognitive)
        
        # Equation 2: dx/dt
        du[idx + 1] = -cognitive.λ * (agent.x - agent.r) + 
                      cognitive.α * agent.m * agent.w * Πi
        
        # Equation 3: dr/dt
        du[idx + 2] = φ(agent.m) * (agent.x - agent.r)
        
        # Equation 4: dm/dt
        target_m = ξ(agent.m, agent.Θ, cognitive)
        du[idx + 3] = cognitive.δm * (target_m - agent.m)
        
        # Equation 5: dw/dt (replicator dynamics)
        du[idx + 4] = cognitive.ηw * agent.w * (ψw(agent.w) - agent.w) * (Ψi - Ψ_bar)
        
        # Equation 6: dΘ/dt
        target_Θ = Υ(agent.m, agent.w)
        du[idx + 5] = cognitive.βΘ * (target_Θ - agent.Θ)
    end
end

"""
    check_jump_condition(agent)
Check if cognitive tension |x - r| exceeds threshold Θ
"""
check_jump_condition(agent::AgentState) = abs(agent.x - agent.r) ≥ agent.Θ

"""
    apply_jump!(agent, params)
Apply jump: x → r, partial memory reset
"""
function apply_jump!(agent::AgentState, params::CognitiveParams)
    # Jump: belief resets to reference
    agent.x = agent.r
    
    # Memory gets partially reset
    agent.m = ξ(agent.m, agent.Θ, params)
end

end # module
