#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))   # <- points to BeliefSim/
Pkg.instantiate()

using LinearAlgebra
include("../src/Kernel.jl");  using .Kernel:SimPars, fully_connected, f, run_one_path 
include("../src/Metrics.jl"); using .Metrics:ShiftPars, shift_estimator, layer_feature
using DifferentialEquations, Random, CSV, DataFrames, Plots

# ------------------------------- experiment set-up ----------------------------
pars  = SimPars(N = 400, κ = 1.45, β = 0.5, σ = 0.3, T = 50.0, Δt = 0.01)
#pars = SimPars(400, 1.45, 0.5, 0.3, 50.0, 0.01)

W     = fully_connected(pars.N)                      # or watts_strogatz_W(…)
Nrun  = 1_000                                        # ensemble size
seeds = rand(1:10^9, Nrun)                          # reproducible seeds

# shift-metric bandwidth constant c (Silverman rule-of-thumb for Epanechnikov)
shift_pars = ShiftPars(pars.N; c = 1.06)

# ------------------------------ ensemble solver -------------------------------
################################################################################
# 1.  Create ONE prototype problem (same as run_one_path but without the seed)
################################################################################
t_vec, B0_dummy = run_one_path(pars; W, seed = 0)   # one shot just to get B0 size
prob_prototype = let
    # drift and noise closures capture `pars` and `W`
    function drift!(dB, B, _, _)
        mul!(dB, W, B);  @. dB = -pars.κ * f(B) + pars.β * (dB - B)
    end
    noise!(dB, _, _, _) = (@. dB = pars.σ)
    SDEProblem(drift!, noise!, randn(pars.N), (0, pars.T))
end

################################################################################
# 2.  Customise each trajectory with a different seed
################################################################################
function prob_func(prob, i, repeat)
    reinit = remake(prob; u0 = randn(Xoshiro(seeds[i]), pars.N))
    return reinit
end

ensemble_prob = EnsembleProblem(prob_prototype; prob_func)
@time sol_set = solve(ensemble_prob,EM();
                      dt= pars.Δt,
                      saveat= 1.0,
                      trajectories = Nrun,
                      ensemblealg = EnsembleThreads())


###############################################################################
# ----- compute shift metrics for every trajectory ----------------------------
###############################################################################
layers  = [:micro, :meso, :macro]
results = DataFrame(run = Int[], layer = Symbol[], delta_hat = Float64[])

T_obs   = pars.T
shift_p = ShiftPars(pars.N)                 # bandwidth + kernel

for (k, sol) in enumerate(sol_set)
    local tgrid = sol.t                     # <- local to suppress the warning
    local Bgrid = sol.u                     # Vector{Vector{Float64}}
    B_final     = Bgrid[end]                # beliefs at time T

    for ℓ in layers
        g_vec = layer_feature(B_final, ℓ, W)            # feature vector length N
        δ̂     = shift_estimator(tgrid, g_vec, T_obs, shift_p)
        push!(results, (k, ℓ, δ̂))
    end
end


CSV.write("output/shift_metrics.csv", results)
println("🚀 Monte-Carlo finished → shift_metrics.csv")

# ------------------------------ quick visual: PDF of δ̂_micro -----------------
micro_vals = results[results.layer .== :micro, :delta_hat]
histogram(micro_vals; bins = 40,
          title = "Distribution of \\hat{δ}_micro(T)",
          xlabel = "shift metric",
          ylabel = "frequency",
          legend = false)
savefig("output/delta_micro_hist.png")
println("→ output/delta_micro_hist.png saved")
