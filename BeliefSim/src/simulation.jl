# simulation.jl - Jump-diffusion SDE solver
module Simulation

using DifferentialEquations, Random, LinearAlgebra
using ..Agents

export simulate_msl, extract_trajectories

"""
    simulate_msl(params, W, seed)

Simulate the jump-diffusion SDE system from the paper.
"""
function simulate_msl(params::MSLParams, W::Matrix{Float64}, seed::Int)
    Random.seed!(seed)
    N = params.N
    
    # Initialize agents
    agents = [AgentState(randn() * 0.5) for _ in 1:N]
    
    # Pack into state vector: [x₁, r₁, m₁, w₁, Θ₁, x₂, r₂, ...]
    u0 = zeros(5N)
    for i in 1:N
        idx = 5*(i-1)
        u0[idx + 1] = agents[i].x
        u0[idx + 2] = agents[i].r
        u0[idx + 3] = agents[i].m
        u0[idx + 4] = agents[i].w
        u0[idx + 5] = agents[i].Θ
    end
    
    # Define drift function
    function drift!(du, u, p, t)
        agents, W, params = p
        
        # Update agents from state vector
        for i in 1:N
            idx = 5*(i-1)
            agents[i].x = u[idx + 1]
            agents[i].r = u[idx + 2]
            agents[i].m = u[idx + 3]
            agents[i].w = u[idx + 4]
            agents[i].Θ = u[idx + 5]
        end
        
        # Compute drift
        compute_drift!(du, agents, W, params, t)
    end
    
    # Define noise function (only beliefs have noise)
    function noise!(du, u, p, t)
        agents, W, params = p
        fill!(du, 0.0)
        
        for i in 1:N
            idx = 5*(i-1)
            du[idx + 1] = params.cognitive.σ
        end
    end
    
    # Jump condition: any agent has |x - r| ≥ Θ
    function condition(u, t, integrator)
        agents, W, params = integrator.p
        
        for i in 1:N
            idx = 5*(i-1)
            x = u[idx + 1]
            r = u[idx + 2]
            Θ = u[idx + 5]
            
            if abs(x - r) ≥ Θ
                return true
            end
        end
        return false
    end
    
    # Jump effect
    function affect!(integrator)
        agents, W, params = integrator.p
        
        for i in 1:N
            idx = 5*(i-1)
            x = integrator.u[idx + 1]
            r = integrator.u[idx + 2]
            m = integrator.u[idx + 3]
            Θ = integrator.u[idx + 5]
            
            if abs(x - r) ≥ Θ
                # Apply jump
                integrator.u[idx + 1] = r  # x → r
                integrator.u[idx + 3] = ξ(m, Θ, params.cognitive)  # Memory reset
            end
        end
    end
    
    # Setup problem
    p = (agents, W, params)
    tspan = (0.0, params.T)
    
    prob = SDEProblem(drift!, noise!, u0, tspan, p)
    jump_callback = DiscreteCallback(condition, affect!)
    
    # Solve
    sol = solve(prob, SOSRA(),
                dt=params.Δt,
                saveat=params.save_interval,
                callback=jump_callback,
                abstol=1e-3,
                reltol=1e-3)
    
    # Extract trajectories
    trajectories = extract_trajectories(sol, N)
    
    return sol.t, trajectories
end

"""
    extract_trajectories(sol, N)

Extract individual agent trajectories from solution.
"""
function extract_trajectories(sol, N::Int)
    T_steps = length(sol.t)
    
    # Initialize storage
    trajectories = Dict(
        :beliefs => [Float64[] for _ in 1:N],
        :references => [Float64[] for _ in 1:N],
        :memory => [Float64[] for _ in 1:N],
        :deliberation => [Float64[] for _ in 1:N],
        :thresholds => [Float64[] for _ in 1:N]
    )
    
    # Extract time series for each agent
    for t_idx in 1:T_steps
        u = sol.u[t_idx]
        
        for i in 1:N
            idx = 5*(i-1)
            push!(trajectories[:beliefs][i], u[idx + 1])
            push!(trajectories[:references][i], u[idx + 2])
            push!(trajectories[:memory][i], u[idx + 3])
            push!(trajectories[:deliberation][i], u[idx + 4])
            push!(trajectories[:thresholds][i], u[idx + 5])
        end
    end
    
    return trajectories
end

end # module
