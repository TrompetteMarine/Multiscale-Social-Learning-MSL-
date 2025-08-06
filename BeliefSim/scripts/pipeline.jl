#!/usr/bin/env julia
using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
include("bifurcation.jl")
include("heatmap.jl")
println("🎯 All plots generated under output/")
