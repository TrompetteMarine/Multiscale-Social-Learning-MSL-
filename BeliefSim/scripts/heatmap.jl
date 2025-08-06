#!/usr/bin/env julia
using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using ..Kernel, ..Viz

W = watts_strogatz_W(60, k=4, p=0.2)   # small network for clarity
network_heatmap(W; fname="output/network_heatmap.png")
