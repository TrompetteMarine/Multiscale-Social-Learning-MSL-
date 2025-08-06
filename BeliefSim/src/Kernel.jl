module Kernel
using DifferentialEquations, Random, LinearAlgebra, Statistics,
      SparseArrays, Graphs    # ← Graphs is now here, at top level

export SimPars, run_one_path, fully_connected, watts_strogatz_W

struct SimPars
    N::Int; κ::Float64; β::Float64; σ::Float64; T::Float64; Δt::Float64
end

# allow SimPars(; N=…, κ=…, …) keyword syntax
SimPars(; N, κ, β, σ, T, Δt) = SimPars(N, κ, β, σ, T, Δt)

f(x) = x
fully_connected(N) = fill(1/N, N, N)

function watts_strogatz_W(N; k=6, p=0.2, seed=123)
    g  = watts_strogatz(N, k, p; rng=Xoshiro(seed))
    W  = adjacency_matrix(g)
    return Array(W ./ sum(W, dims=2))
end

function run_one_path(pars::SimPars; W=fully_connected(pars.N), seed=0)
    rng = Xoshiro(seed);  B0 = randn(rng, pars.N)
    function drift!(dB, B, _, _)
        mul!(dB, W, B);  @. dB = -pars.κ * f(B) + pars.β * (dB - B)
    end
    noise!(dB, _, _, _) = (@. dB = pars.σ)
    prob = SDEProblem(drift!, noise!, B0, (0, pars.T))
    sol  = solve(prob, EM(); dt=pars.Δt, saveat=1.0)
    return sol.t, sol.u
end
end # module
