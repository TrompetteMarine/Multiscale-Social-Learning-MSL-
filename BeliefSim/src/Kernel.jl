module Kernel

using DifferentialEquations, Random, LinearAlgebra, Statistics
using SparseArrays, Graphs, Distributions

export AgentState, MSLSimPars, CognitiveParams, NetworkParams
export run_msl_simulation, run_ensemble_msl
export fully_connected, watts_strogatz_W, barabasi_albert_W, erdos_renyi_W
export φ_memory, ξ_memory_reset, ψw_deliberation, ε_attention_radius
export cognitive_tension, attention_set, social_influence

# ============================================================================
# Multi-Scale Learning Parameters (Following Paper Exactly)
# ============================================================================

# Agent cognitive parameters
@kwdef struct CognitiveParams
    λ::Float64 = 1.0      # mean reversion strength  
    α::Float64 = 0.5      # social influence scale
    σ::Float64 = 0.3      # idiosyncratic noise
    δm::Float64 = 0.1     # memory adjustment speed
    ηw::Float64 = 0.1     # deliberation adjustment speed  
    βΘ::Float64 = 0.05    # threshold adjustment speed
    
    # Bounds
    mmin::Float64 = 0.1
    mmax::Float64 = 2.0
    wmin::Float64 = 0.1
    wmax::Float64 = 1.0
    Θmin::Float64 = 0.5
    Θmax::Float64 = 3.0
    
    # Attention radius parameters
    εmax::Float64 = 2.0
    κ_attention::Float64 = 1.0    # attention radius curvature
end

# Network parameters
@kwdef struct NetworkParams
    type::Symbol = :fully_connected
    k::Int = 6                    # degree for small-world/regular
    p::Float64 = 0.2             # rewiring probability
    m::Int = 3                   # preferential attachment parameter
end

# Complete agent state: (x, r, m, w, Θ) as in paper
mutable struct AgentState
    x::Float64        # belief
    r::Float64        # reference point  
    m::Float64        # memory weight
    w::Float64        # deliberation weight
    Θ::Float64        # cognitive tension threshold
end

AgentState(x=0.0) = AgentState(x, x, 1.0, 0.5, 1.0)

# Main simulation parameters
@kwdef struct MSLSimPars
    N::Int = 100
    T::Float64 = 50.0
    Δt::Float64 = 0.01
    save_interval::Float64 = 0.1
    cognitive::CognitiveParams = CognitiveParams()
    network::NetworkParams = NetworkParams()
end

# ============================================================================
# Cognitive Functions (From Paper Section 2.1)
# ============================================================================

# Memory sensitivity function φ(m) - decreasing
φ_memory(m::Float64) = 1.0 / (1.0 + m)

# Memory reset function ξ(m,Θ) - increasing in Θ  
ξ_memory_reset(m::Float64, Θ::Float64) = min(2.0, m + 0.1 * Θ)

# Deliberation adjustment ψw(w)
ψw_deliberation(w::Float64) = w * (1.0 - w) + 0.1

# Attention radius ε(Θ) - strictly increasing and concave
function ε_attention_radius(Θ::Float64, εmax::Float64=2.0, κ::Float64=1.0)
    return εmax * (1.0 - exp(-κ * Θ / εmax))
end

# Optimal tension function Υ(m,w) 
function Υ_optimal_tension(m::Float64, w::Float64)
    return 0.5 + 0.3 * m + 0.2 * w
end

# ============================================================================
# Network Generation (Enhanced)
# ============================================================================

function create_network(N::Int, params::NetworkParams)
    if params.type == :fully_connected
        return fully_connected(N)
    elseif params.type == :small_world
        return watts_strogatz_W(N; k=params.k, p=params.p)
    elseif params.type == :scale_free
        return barabasi_albert_W(N; m=params.m)
    elseif params.type == :random
        return erdos_renyi_W(N; p=params.p)
    else
        @warn "Unknown network type $(params.type), using fully connected"
        return fully_connected(N)
    end
end

fully_connected(N) = fill(1.0/N, N, N)

function watts_strogatz_W(N; k=6, p=0.2, seed=123)
    rng = Xoshiro(seed)
    g = watts_strogatz(N, k, p; rng=rng)
    W = adjacency_matrix(g)
    row_sums = sum(W, dims=2)
    return Array(W ./ max.(row_sums, 1))
end

function barabasi_albert_W(N; m=3, seed=123)
    rng = Xoshiro(seed)
    g = barabasi_albert(N, m; rng=rng)
    W = adjacency_matrix(g)
    W = (W + W') .> 0  # Make symmetric
    row_sums = sum(W, dims=2)
    return Array(W ./ max.(row_sums, 1))
end

function erdos_renyi_W(N; p=0.1, seed=123)
    rng = Xoshiro(seed)
    g = erdos_renyi(N, p; rng=rng)
    W = adjacency_matrix(g)
    row_sums = sum(W, dims=2)
    return Array(W ./ max.(row_sums, 1))
end

# ============================================================================
# Social Learning Dynamics (Paper Section 2.2)
# ============================================================================

# Cognitive tension |xi - ri| vs threshold Θi
function cognitive_tension(agent::AgentState)
    return abs(agent.x - agent.r)
end

# Attention set Ni(t) based on belief proximity and threshold  
function attention_set(i::Int, agents::Vector{AgentState}, εmax::Float64=2.0)
    xi = agents[i].x
    Θi = agents[i].Θ
    radius = ε_attention_radius(Θi, εmax)
    
    neighbors = Int[]
    for (j, agent) in enumerate(agents)
        if i != j && abs(agent.x - xi) ≤ radius
            push!(neighbors, j)
        end
    end
    return neighbors
end

# Social influence Πi(t) using Epanechnikov kernel
function social_influence(i::Int, agents::Vector{AgentState}, K_func=epanechnikov_kernel)
    neighbors = attention_set(i, agents)
    xi = agents[i].x
    
    influence = 0.0
    for j in neighbors
        xj = agents[j].x
        distance = abs(xj - xi)
        influence += K_func(distance) * (xj - xi)
    end
    
    return influence
end

# Epanechnikov kernel K(r)
function epanechnikov_kernel(r::Float64, bandwidth::Float64=1.0)
    u = r / bandwidth
    return u ≤ 1.0 ? 0.75 * (1.0 - u^2) / bandwidth : 0.0
end

# ============================================================================
# Jump-Diffusion SDE System (Paper Equations 2-6) 
# ============================================================================

function msl_drift!(du, u, p, t)
    agents, W, params = p
    N = length(agents)
    
    # Update agent states from u vector
    for i in 1:N
        idx = (i-1)*5
        agents[i].x = u[idx + 1]
        agents[i].r = u[idx + 2] 
        agents[i].m = u[idx + 3]
        agents[i].w = u[idx + 4]
        agents[i].Θ = u[idx + 5]
    end
    
    # Compute derivatives for each agent
    for i in 1:N
        idx = (i-1)*5
        agent = agents[i]
        
        # Belief dynamics dx/dt (Equation 2)
        Πi = social_influence(i, agents)
        du[idx + 1] = -params.cognitive.λ * (agent.x - agent.r) + 
                      params.cognitive.α * agent.m * agent.w * Πi
        
        # Reference point dr/dt (Equation 3)
        du[idx + 2] = φ_memory(agent.m) * (agent.x - agent.r)
        
        # Memory weight dm/dt (Equation 4)
        du[idx + 3] = params.cognitive.δm * (ξ_memory_reset(agent.m, agent.Θ) - agent.m)
        
        # Deliberation weight dw/dt (Equation 5) - simplified
        du[idx + 4] = params.cognitive.ηw * agent.w * (ψw_deliberation(agent.w) - agent.w) * 0.1
        
        # Threshold dΘ/dt (Equation 6)
        du[idx + 5] = params.cognitive.βΘ * (Υ_optimal_tension(agent.m, agent.w) - agent.Θ)
    end
end

function msl_noise!(du, u, p, t)
    agents, W, params = p
    N = length(agents)
    
    fill!(du, 0.0)
    
    # Only beliefs have noise (σ dW_i)
    for i in 1:N
        idx = (i-1)*5
        du[idx + 1] = params.cognitive.σ
    end
end

# Jump condition: cognitive tension exceeds threshold
function jump_condition(u, t, integrator)
    agents, W, params = integrator.p
    N = length(agents)
    
    # Check each agent for cognitive tension overflow
    for i in 1:N
        idx = (i-1)*5
        x, r, m, w, Θ = u[idx+1], u[idx+2], u[idx+3], u[idx+4], u[idx+5]
        
        if abs(x - r) ≥ Θ
            return true
        end
    end
    return false
end

# Jump effect: reset belief to reference point
function jump_effect!(integrator)
    agents, W, params = integrator.p
    N = length(agents)
    
    for i in 1:N
        idx = (i-1)*5
        x, r, m, w, Θ = integrator.u[idx+1], integrator.u[idx+2], 
                         integrator.u[idx+3], integrator.u[idx+4], integrator.u[idx+5]
        
        if abs(x - r) ≥ Θ
            # Jump: x → r, memory gets partially reset
            integrator.u[idx + 1] = r
            integrator.u[idx + 3] = ξ_memory_reset(m, Θ)
        end
    end
end

# ============================================================================
# Main Simulation Function
# ============================================================================

function run_msl_simulation(params::MSLSimPars; seed=42, W=nothing)
    rng = Xoshiro(seed)
    N = params.N
    
    # Generate network if not provided
    if W === nothing
        W = create_network(N, params.network)
    end
    
    # Initialize agent states
    agents = [AgentState(randn(rng) * 0.5) for _ in 1:N]
    
    # Pack initial conditions into vector: [x1,r1,m1,w1,Θ1, x2,r2,m2,w2,Θ2, ...]
    u0 = Float64[]
    for agent in agents
        append!(u0, [agent.x, agent.r, agent.m, agent.w, agent.Θ])
    end
    
    # Setup SDE problem
    p = (agents, W, params)
    tspan = (0.0, params.T)
    
    # Create SDE with jumps
    prob = SDEProblem(msl_drift!, msl_noise!, u0, tspan, p)
    
    # Add jump callback
    jump_cb = DiscreteCallback(jump_condition, jump_effect!)
    
    # Solve with appropriate algorithm for jump-diffusion
    sol = solve(prob, SOSRI(), callback=jump_cb, 
                dt=params.Δt, saveat=params.save_interval,
                maxiters=1e6, abstol=1e-6, reltol=1e-6)
    
    # Extract trajectories for each agent and variable
    trajectories = extract_trajectories(sol, N)
    
    return sol.t, trajectories
end

function extract_trajectories(sol, N)
    T_steps = length(sol.t)
    
    # Initialize storage: trajectories[variable][agent][time]
    trajectories = Dict(
        :beliefs => [Float64[] for _ in 1:N],
        :references => [Float64[] for _ in 1:N], 
        :memory => [Float64[] for _ in 1:N],
        :deliberation => [Float64[] for _ in 1:N],
        :thresholds => [Float64[] for _ in 1:N]
    )
    
    for t_idx in 1:T_steps
        u = sol.u[t_idx]
        for i in 1:N
            idx = (i-1)*5
            push!(trajectories[:beliefs][i], u[idx + 1])
            push!(trajectories[:references][i], u[idx + 2])
            push!(trajectories[:memory][i], u[idx + 3])
            push!(trajectories[:deliberation][i], u[idx + 4])
            push!(trajectories[:thresholds][i], u[idx + 5])
        end
    end
    
    return trajectories
end

# ============================================================================
# Ensemble Simulations
# ============================================================================

function run_ensemble_msl(params::MSLSimPars; n_runs=10, seeds=nothing)
    if seeds === nothing
        seeds = 1:n_runs
    end
    
    W = create_network(params.N, params.network)  # Fixed network across runs
    results = []
    
    @info "Running ensemble of $(n_runs) MSL simulations..."
    
    for (i, seed) in enumerate(seeds)
        if i % 10 == 0
            @info "  Completed $i/$(n_runs) runs"
        end
        
        try
            t_vec, traj = run_msl_simulation(params; seed=seed, W=W)
            push!(results, (t=t_vec, trajectories=traj, seed=seed))
        catch e
            @warn "Simulation failed for seed $seed: $e"
        end
    end
    
    @info "Ensemble complete: $(length(results))/$(n_runs) successful runs"
    return results, W
end

# ============================================================================
# Utility Functions
# ============================================================================

function network_stats(W::AbstractMatrix)
    N = size(W, 1)
    A = W .> 0  # Binary adjacency
    
    density = sum(A) / (N * (N - 1))
    degrees = vec(sum(A, dims=2))
    avg_degree = mean(degrees)
    clustering = local_clustering_coefficient(A)
    
    return Dict(
        :density => density,
        :avg_degree => avg_degree, 
        :global_clustering => mean(clustering),
        :degree_distribution => degrees
    )
end

function local_clustering_coefficient(A::AbstractMatrix)
    N = size(A, 1)
    clustering = zeros(N)
    
    for i in 1:N
        neighbors = findall(A[i, :])
        ki = length(neighbors)
        
        if ki < 2
            clustering[i] = 0.0
            continue
        end
        
        triangles = 0
        for j in neighbors, k in neighbors
            if j < k && A[j, k]
                triangles += 1
            end
        end
        
        clustering[i] = 2 * triangles / (ki * (ki - 1))
    end
    
    return clustering
end

end # module
