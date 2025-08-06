module Viz
using Statistics, StatsBase, Plots, Colors, LinearAlgebra

export shift_metric, ribbon_plot!, bifurcation_plot, network_heatmap

shift_metric(B) = quantile(B, 0.75) - quantile(B, 0.25)   # Δ

"""
    ribbon_plot!(ax, t, runs; α=0.1)

Overlay mean curve with an 80 % ribbon on existing axis.
"""
function ribbon_plot!(t, Δstore; α=0.1, kwargs...)
    Δmean = mean(Δstore, dims=1) |> vec
    Δlow  = mapslices(v -> quantile(v, α),  Δstore; dims=1) |> vec
    Δhigh = mapslices(v -> quantile(v, 1-α), Δstore; dims=1) |> vec
    plot!(t, Δmean; ribbon=(Δmean .- Δlow, Δhigh .- Δmean), kwargs...)
end

"""
    bifurcation_plot(κs, plateau_vals; fname)

Scatter + join showing steady-state Δ vs κ.
"""
function bifurcation_plot(κs, vals; fname="output/bifurcation.png")
    scatter(κs, vals; xlabel="κ", ylabel="steady Δ", legend=false,
            title="Bifurcation diagram")
    plot!(κs, vals)
    savefig(fname); println("→ $fname")
end

"""
    network_heatmap(W; fname)

Quick heat-map of adjacency / weight matrix.
"""
function network_heatmap(W; fname="output/network_heatmap.png")
    heatmap(Matrix(W); c=:dense, colorbar=false, axis=nothing)
    savefig(fname); println("→ $fname")
end
end # module
