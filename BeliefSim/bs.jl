################################################################################
# belief_sim.jl
# A minimal Euler–Maruyama simulation of the paper's belief-dynamics SDE
# Requires: DifferentialEquations, Distributions, Plots  (already in your env)
################################################################################

using Pkg
Pkg.activate(@__DIR__)       # makes sure we are inside the BeliefSim env
Pkg.instantiate()            # downloads packages if env is copied elsewhere

using DifferentialEquations
using Distributions
using Random
using LinearAlgebra
using Plots                    # default backend is fine

# ------------------------------------------------------------------ parameters
const N       = 150            # number of agents
const κ       = 0.9            # cognitive-cost parameter  (tune this!)
const β       = 0.5            # social-influence weight
const σ       = 0.3            # idiosyncratic noise
const T       = 50.0           # simulation horizon
const Δt      = 0.01           # step size for Euler–Maruyama

# cognitive-cost function f(x); swap in the paper’s exact form if different
f(x) = x                       # odd, Lipschitz; linear keeps code simple

# ------------------------------------------------------------------ network W
# Fully connected, row-stochastic matrix by default
#W = fill(1 / N, N, N)

# Example: small-world network (uncomment 4 lines below)
 using Graphs, Random
 g = watts_strogatz(N, 6, 0.2; rng = Xoshiro(123))
 W = adjacency_matrix(g)
 W = W ./ sum(W, dims = 2)     # row-normalise

# ------------------------------------------------------------------ SDE callbacks
function drift!(dB, B, _, _)       # μ(B) dt
    mul!(dB, W, B)                 # dB = W * B
    @. dB = -κ * f(B) + β * (dB - B)
end

function noise!(dB, _, _, _)       # σ dW
    @. dB = σ
end

# ------------------------------------------------------------------ initial state
rng = Xoshiro(42)                  # reproducible RNG
B0  = randn(rng, N)                # Normal(0,1) beliefs

prob = SDEProblem(drift!, noise!, B0, (0.0, T))
sol  = solve(prob, EM(); dt = Δt, saveat = 1.0)   # EM = Euler–Maruyama

# ------------------------------------------------------------------ quick plots
histogram(sol.u[end];
          bins = 40,
          title = "Belief distribution at T = $(T)",
          xlabel = "belief",
          ylabel = "frequency",
          legend = false)
savefig("belief_hist.png")         # saved in the same folder

# Optional: time-series of mean belief
mean_b = [mean(u) for u in sol.u]
plot(sol.t, mean_b;
     title = "Mean belief over time",
     xlabel = "time",
     ylabel = "E[Bᵢ(t)]")
savefig("mean_belief.png")

println("Simulation finished ✔︎  (plots saved to belief_hist.png & mean_belief.png)")
