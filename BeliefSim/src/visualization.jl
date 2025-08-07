# visualization.jl - Visualization utilities
module Visualization

using Plots, Statistics

export plot_results, plot_trajectories, plot_bifurcation, plot_network

"""
    plot_trajectories(t_vec, trajectories; agents_to_plot=20)

Plot belief trajectories for a sample of agents.
"""
function plot_trajectories(t_vec, trajectories; agents_to_plot=20)
    N = length(trajectories[:beliefs])
    n_plot = min(agents_to_plot, N)
    
    p = plot(xlabel="Time", ylabel="Belief", 
             title="Agent Belief Trajectories",
             legend=false)
    
    # Plot individual trajectories
    for i in 1:n_plot
        plot!(p, t_vec, trajectories[:beliefs][i], 
              alpha=0.4, color=:blue, lw=1)
    end
    
    # Add mean trajectory
    mean_beliefs = [mean([trajectories[:beliefs][i][t] for i in 1:N]) 
                   for t in 1:length(t_vec)]
    plot!(p, t_vec, mean_beliefs, 
          lw=3, color=:red, label="Mean")
    
    return p
end

"""
    plot_bifurcation(α_values, consensus, polarization; α_star=nothing)

Plot bifurcation diagram.
"""
function plot_bifurcation(α_values, consensus, polarization; α_star=nothing)
    p = plot(xlabel="Social Influence (α)", 
             ylabel="Order Parameter",
             title="Bifurcation Diagram",
             legend=:topright)
    
    plot!(p, α_values, consensus, 
          lw=2, marker=:circle, ms=3,
          label="Consensus", color=:blue)
    
    plot!(p, α_values, polarization,
          lw=2, marker=:square, ms=3,
          label="Polarization", color=:red)
    
    if α_star !== nothing
        vline!(p, [α_star], ls=:dash, lw=2, 
               color=:green, label="Critical α*")
    end
    
    return p
end

"""
    plot_network(W; layout=:spring)

Visualize network structure.
"""
function plot_network(W::Matrix{Float64}; layout=:spring)
    # Simple heatmap visualization
    p = heatmap(W, 
                xlabel="Agent j", ylabel="Agent i",
                title="Network Adjacency Matrix",
                c=:viridis, clims=(0, maximum(W)))
    
    return p
end

"""
    plot_results(analysis)

Create multi-panel summary plot.
"""
function plot_results(analysis::Dict)
    # Extract data
    consensus_evol = get(analysis, :consensus_evolution, Float64[])
    polarization_evol = get(analysis, :polarization_evolution, Float64[])
    tension_evol = get(analysis, :cognitive_tension_evolution, Float64[])
    
    T = length(consensus_evol)
    t_vec = 1:T
    
    # Create subplots
    p1 = plot(t_vec, consensus_evol,
              xlabel="Time", ylabel="Consensus",
              title="Consensus Evolution",
              lw=2, color=:blue, legend=false)
    
    p2 = plot(t_vec, polarization_evol,
              xlabel="Time", ylabel="Polarization",
              title="Polarization Evolution",
              lw=2, color=:red, legend=false)
    
    p3 = plot(t_vec, tension_evol,
              xlabel="Time", ylabel="|x - r|",
              title="Cognitive Tension",
              lw=2, color=:purple, legend=false)
    
    # Regime indicator
    regime = get(analysis, :regime, "Unknown")
    p4 = plot([], [], framestyle=:none, legend=false)
    annotate!(p4, [(0.5, 0.5, text("Regime:\n$regime", 16, :center))])
    
    # Combine
    return plot(p1, p2, p3, p4, 
                layout=(2,2), size=(800, 600))
end

end # module
