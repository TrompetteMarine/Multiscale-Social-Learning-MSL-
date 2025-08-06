#!/usr/bin/env julia
using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using ..Kernel, ..Viz, Statistics

# ---- sweep parameters -------------------------------------------------------
κ_grid = 0.6:0.2:2.0
pars0  = SimPars(N=200, κ=0, β=0.5, σ=0.3, T=50, Δt=0.01)
W      = fully_connected(pars0.N)
plateau = Float64[]

for κ in κ_grid
    pars = SimPars(; pars0.N, κ, pars0.β, pars0.σ, pars0.T, pars0.Δt)
    t, Bs = run_one_path(pars; W, seed=42)      # single path suffices for plateau
    Δt_series = shift_metric.(Bs)
    push!(plateau, mean(Δt_series[end-9:end]))  # average last 10 points
end

bifurcation_plot(collect(κ_grid), plateau; fname="output/bifurcation.png")
