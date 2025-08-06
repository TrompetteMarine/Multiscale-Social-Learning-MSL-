module Kernel
using DifferentialEquations, Random, LinearAlgebra, Statistics
using SparseArrays, Graphs, Distributions

export SimPars, HeterogeneousSimPars, run_one_path, run_heterogeneous_path
export fully_connected, watts_strogatz_W, barabasi_albert_W, erdos_renyi_W
export scale_free_W, ring_lattice_W, caveman_W
export f_linear, f_cubic, f_tanh, f_sigmoid
export TimeVaryingPars, run_time_varying_path

# ============================================================================ 
# Parameter structures
# ============================================================================ 

struct SimPars
    N::Int; κ::Float64; β::Float64; σ::Float64; T::Float64; Δt::Float64
end

# Allow keyword constructor
SimPars(; N, κ, β, σ, T, Δt) = SimPars(N, κ, β, σ, T, Δt)

# For heterogeneous agents
struct HeterogeneousSimPars
    N::Int
    κ::Vector{Float64}    # agent-specific cognitive costs
    β::Vector{Float64}    # agent-specific social influence weights
    σ::Vector{Float64}    # agent-specific noise levels
    T::Float64
    Δt::Float64
end

# For time-varying parameters
struct TimeVaryingPars
    N::Int
    κ_func::Function     # κ(t)
    β_func::Function     # β(t)
    σ_func::Function     # σ(t)
    T::Float64
    Δt::Float64
end

# ============================================================================ 
# Cognitive cost functions
# ============================================================================ 

f_linear(x) = x
f_cubic(x) = x^3
f_tanh(x) = tanh(x)
f_sigmoid(x) = 2/(1 + exp(-2x)) - 1  # scaled sigmoid to [-1,1]

# Default cognitive cost function
f = f_linear

# ============================================================================ 
# Network generation functions
# ============================================================================ 

fully_connected(N) = fill(1/N, N, N)

function watts_strogatz_W(N; k=6, p=0.2, seed=123)
    g = watts_strogatz(N, k, p; rng=Xoshiro(seed))
    W = adjacency_matrix(g)
    return Array(W ./ sum(W, dims=2))
end

function barabasi_albert_W(N; m=3, seed=123)
    g = barabasi_albert(N, m; rng=Xoshiro(seed))
    W = adjacency_matrix(g)
    # Make symmetric for undirected case
    W = (W + W') .> 0
    return Array(W ./ sum(W, dims=2))
end

function erdos_renyi_W(N; p=0.1, seed=123)
    g = erdos_renyi(N, p; rng=Xoshiro(seed))
    W = adjacency_matrix(g)
    return Array(W ./ sum(W, dims=2))
end

function scale_free_W(N; gamma=2.5, seed=123)
    # Create a scale-free network using configuration model
    rng = Xoshiro(seed)
    degrees = round.(Int, rand(rng, Pareto(1, gamma-1), N))
    degrees = max.(degrees, 1)  # ensure minimum degree 1
    
    # Make total degree even
    if sum(degrees) % 2 == 1
        degrees[1] += 1
    end
    
    g = configuration_model(degrees; rng=rng)
    g = simplify(g)  # remove self-loops and multiple edges
    
    W = adjacency_matrix(g)
    return Array(W ./ sum(W, dims=2))
end

function ring_lattice_W(N; k=4)
    g = Graph(N)
    for i in 1:N
        for j in 1:(k÷2)
            add_edge!(g, i, mod1(i+j, N))
            add_edge!(g, i, mod1(i-j, N))
        end
    end
    W = adjacency_matrix(g)
    return Array(W ./ sum(W, dims=2))
end

function caveman_W(N; group_size=10, p_within=0.9, p_between=0.01, seed=123)
    # Caveman graph: dense clusters with sparse connections between
    rng = Xoshiro(seed)
    n_groups = N ÷ group_size
    g = Graph(N)
    
    # Within-group connections
    for group in 0:(n_groups-1)
        start_idx = group * group_size + 1
        end_idx = min((group + 1) * group_size, N)
        
        for i in start_idx:end_idx
            for j in (i+1):end_idx
                if rand(rng) < p_within
                    add_edge!(g, i, j)
                end
            end
        end
    end
    
    # Between-group connections
    for i in 1:N
        for j in (i+1):N
            group_i = (i-1) ÷ group_size
            group_j = (j-1) ÷ group_size
            if group_i != group_j && rand(rng) < p_between
                add_edge!(g, i, j)
            end
        end
    end
    
    W = adjacency_matrix(g)
    return Array(W ./ max.(sum(W, dims=2), 1))  # avoid division by zero
end

# ============================================================================ 
# Simulation functions
# ============================================================================ 

function run_one_path(pars::SimPars; W=fully_connected(pars.N), seed=0, 
                      cost_func=f_linear, save_interval=1.0)
    rng = Xoshiro(seed)
    B0 = randn(rng, pars.N)
    
    function drift!(dB, B, _, _)
        mul!(dB, W, B)
        @. dB = -pars.κ * cost_func(B) + pars.β * (dB - B)
    end
    
    noise!(dB, _, _, _) = (@. dB = pars.σ)
    
    prob = SDEProblem(drift!, noise!, B0, (0, pars.T))
    sol = solve(prob, EM(); dt=pars.Δt, saveat=save_interval)
    
    return sol.t, sol.u
end

function run_heterogeneous_path(pars::HeterogeneousSimPars; W=fully_connected(pars.N), 
                                seed=0, cost_func=f_linear, save_interval=1.0)
    rng = Xoshiro(seed)
    B0 = randn(rng, pars.N)
    
    function drift!(dB, B, _, _)
        mul!(dB, W, B)
        for i in 1:pars.N
            dB[i] = -pars.κ[i] * cost_func(B[i]) + pars.β[i] * (dB[i] - B[i])
        end
    end
    
    function noise!(dB, _, _, _)
        for i in 1:pars.N
            dB[i] = pars.σ[i]
        end
    end
    
    prob = SDEProblem(drift!, noise!, B0, (0, pars.T))
    sol = solve(prob, EM(); dt=pars.Δt, saveat=save_interval)
    
    return sol.t, sol.u
end

function run_time_varying_path(pars::TimeVaryingPars; W=fully_connected(pars.N), 
                               seed=0, cost_func=f_linear, save_interval=1.0)
    rng = Xoshiro(seed)
    B0 = randn(rng, pars.N)
    
    function drift!(dB, B, _, t)
        mul!(dB, W, B)
        κ_t = pars.κ_func(t)
        β_t = pars.β_func(t)
        @. dB = -κ_t * cost_func(B) + β_t * (dB - B)
    end
    
    function noise!(dB, _, _, t)
        σ_t = pars.σ_func(t)
        @. dB = σ_t
    end
    
    prob = SDEProblem(drift!, noise!, B0, (0, pars.T))
    sol = solve(prob, EM(); dt=pars.Δt, saveat=save_interval)
    
    return sol.t, sol.u
end

# ============================================================================ 
# Utility functions
# ============================================================================ 

function network_stats(W::AbstractMatrix)
    N = size(W, 1)
    # Convert to binary adjacency matrix
    A = (W .> 0)
    
    # Basic stats
    density = sum(A) / (N * (N - 1))
    avg_degree = mean(sum(A, dims=2))
    
    # Clustering coefficient (local)
    clustering = zeros(N)
    for i in 1:N
        neighbors = findall(A[i, :])
        k_i = length(neighbors)
        if k_i < 2
            clustering[i] = 0.0
        else
            triangles = 0
            for j in neighbors, k in neighbors
                if j < k && A[j, k]
                    triangles += 1
                end
            end
            clustering[i] = 2 * triangles / (k_i * (k_i - 1))
        end
    end
    
    return Dict(
        :density => density,
        :avg_degree => avg_degree,
        :global_clustering => mean(clustering),
        :degree_distribution => vec(sum(A, dims=2))
    )
end

end # module
