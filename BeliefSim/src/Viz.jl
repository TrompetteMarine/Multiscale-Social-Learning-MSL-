module Viz
using Statistics, StatsBase, Plots, Colors, LinearAlgebra
using GraphPlot, Graphs, NetworkLayout

export shift_metric, ribbon_plot!, bifurcation_plot, network_heatmap
export trajectory_plot, phase_space_plot, network_graph_plot
export consensus_evolution_plot, polarization_heatmap, influence_network_plot
export parameter_sweep_plot, synchronization_plot, belief_distribution_animation

# ============================================================================ 
# Original functionality
# ============================================================================ 

shift_metric(B) = quantile(B, 0.75) - quantile(B, 0.25)   # Δ

function ribbon_plot!(t, Δstore; α=0.1, kwargs...)
    Δmean = mean(Δstore, dims=1) |> vec
    Δlow  = mapslices(v -> quantile(v, α),  Δstore; dims=1) |> vec
    Δhigh = mapslices(v -> quantile(v, 1-α), Δstore; dims=1) |> vec
    plot!(t, Δmean; ribbon=(Δmean .- Δlow, Δhigh .- Δmean), kwargs...)
end

function bifurcation_plot(κs, vals; fname="output/bifurcation.png")
    scatter(κs, vals; xlabel="κ", ylabel="steady Δ", legend=false,
            title="Bifurcation diagram")
    plot!(κs, vals)
    savefig(fname); println("→ $fname")
end

function network_heatmap(W; fname="output/network_heatmap.png")
    heatmap(Matrix(W); c=:dense, colorbar=true, axis=nothing,
            title="Network Adjacency Matrix")
    savefig(fname); println("→ $fname")
end

# ============================================================================ 
# Enhanced trajectory visualization
# ============================================================================ 

function trajectory_plot(t_vec::Vector{Float64}, trajectories::Vector{Vector{Float64}};
                        sample_agents=10, fname="output/trajectories.png")
    N = length(trajectories[1])
    T_steps = length(t_vec)
    
    # Sample agents to avoid overcrowding
    if N > sample_agents
        agent_indices = rand(1:N, sample_agents)
    else
        agent_indices = 1:N
    end
    
    p = plot(xlabel="Time", ylabel="Belief", title="Individual Agent Trajectories")
    
    for i in agent_indices
        agent_traj = [trajectories[t][i] for t in 1:T_steps]
        plot!(p, t_vec, agent_traj, alpha=0.6, lw=1, label="")
    end
    
    # Add ensemble mean
    mean_traj = [mean(trajectories[t]) for t in 1:T_steps]
    plot!(p, t_vec, mean_traj, lw=3, color=:red, label="Ensemble Mean")
    
    savefig(p, fname)
    println("→ $fname")
    return p
end

# ============================================================================ 
# Phase space visualization
# ============================================================================ 

function phase_space_plot(beliefs::Vector{Float64}, velocities::Vector{Float64};
                         fname="output/phase_space.png")
    p = scatter(beliefs, velocities, alpha=0.6, xlabel="Belief", ylabel="Velocity",
                title="Phase Space Plot", legend=false, ms=2)
    savefig(p, fname)
    println("→ $fname")
    return p
end

# ============================================================================ 
# Network graph visualization
# ============================================================================ 

function network_graph_plot(W::AbstractMatrix; fname="output/network_graph.png",
                           node_colors=nothing, node_sizes=nothing)
    # Convert to Graph object
    A = W .> 0  # Binary adjacency
    g = Graph(A)
    
    # Default node properties
    N = size(W, 1)
    if node_colors === nothing
        node_colors = fill(:lightblue, N)
    end
    if node_sizes === nothing
        degrees = vec(sum(A, dims=2))
        node_sizes = degrees .> 0 ? 10 .+ 5 * degrees / maximum(degrees) : fill(10, N)
    end
    
    # Use spring layout - returns (x_coords, y_coords) tuple
    try
        pos = spring_layout(g)
        x_coords, y_coords = pos  # Unpack the coordinate vectors
        
        p = plot(size=(600, 600), aspect_ratio=:equal, showaxis=false, grid=false)
        
        # Draw edges
        for edge in edges(g)
            i, j = src(edge), dst(edge)
            plot!(p, [x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 
                  color=:gray, alpha=0.5, lw=1, label="")
        end
        
        # Draw nodes
        scatter!(p, x_coords, y_coords, 
                 color=node_colors, ms=node_sizes, 
                 strokewidth=1, strokecolor=:black, label="")
        
        title!(p, "Network Structure")
        savefig(p, fname)
        println("→ $fname")
        return p
        
    catch e
        # Fallback to simple circular layout if spring_layout fails
        println("  Warning: Spring layout failed, using circular layout")
        θ = 2π * (0:N-1) / N
        x_coords = cos.(θ)
        y_coords = sin.(θ)
        
        p = plot(size=(600, 600), aspect_ratio=:equal, showaxis=false, grid=false)
        
        # Draw edges with circular layout
        for edge in edges(g)
            i, j = src(edge), dst(edge)
            plot!(p, [x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 
                  color=:gray, alpha=0.5, lw=1, label="")
        end
        
        # Draw nodes
        scatter!(p, x_coords, y_coords, 
                 color=node_colors, ms=node_sizes, 
                 strokewidth=1, strokecolor=:black, label="")
        
        title!(p, "Network Structure (Circular Layout)")
        savefig(p, fname)
        println("→ $fname")
        return p
    end
end

# ============================================================================ 
# Consensus evolution
# ============================================================================ 

function consensus_evolution_plot(t_vec::Vector{Float64}, 
                                 trajectories::Vector{Vector{Float64}};
                                 fname="output/consensus_evolution.png")
    T_steps = length(t_vec)
    
    # Calculate consensus metrics over time
    mean_beliefs = [mean(trajectories[t]) for t in 1:T_steps]
    std_beliefs = [std(trajectories[t]) for t in 1:T_steps]
    iqr_beliefs = [quantile(trajectories[t], 0.75) - quantile(trajectories[t], 0.25) 
                   for t in 1:T_steps]
    
    p1 = plot(t_vec, mean_beliefs, lw=2, label="Mean Belief", 
              xlabel="Time", ylabel="Mean Belief", title="Consensus Evolution")
    
    p2 = plot(t_vec, std_beliefs, lw=2, color=:red, label="Standard Deviation",
              xlabel="Time", ylabel="Disagreement")
    plot!(p2, t_vec, iqr_beliefs, lw=2, color=:blue, label="IQR")
    
    p = plot(p1, p2, layout=(2,1), size=(800, 600))
    savefig(p, fname)
    println("→ $fname")
    return p
end

# ============================================================================ 
# Polarization heatmap
# ============================================================================ 

function polarization_heatmap(parameter_grid, polarization_values; 
                             param_names=["Parameter 1", "Parameter 2"],
                             fname="output/polarization_heatmap.png")
    p = heatmap(parameter_grid[1], parameter_grid[2], polarization_values,
                xlabel=param_names[1], ylabel=param_names[2],
                title="Polarization Landscape", c=:viridis)
    savefig(p, fname)
    println("→ $fname")
    return p
end

# ============================================================================ 
# Influence network plot
# ============================================================================ 

function influence_network_plot(W::AbstractMatrix, influence_scores::Vector{Float64};
                               fname="output/influence_network.png")
    # Normalize influence scores for visualization
    max_influence = maximum(influence_scores)
    norm_influence = max_influence > 0 ? influence_scores / max_influence : fill(0.5, length(influence_scores))
    
    # Color nodes by influence (red = high influence, blue = low influence)
    colors = [:red for _ in norm_influence]  # Simple red for high influence
    sizes = 8 .+ 15 * norm_influence
    
    return network_graph_plot(W; fname=fname, node_colors=colors, node_sizes=sizes)
end

# ============================================================================ 
# Parameter sweep visualization
# ============================================================================ 

function parameter_sweep_plot(param_values, metrics; metric_name="Metric",
                             param_name="Parameter", fname="output/parameter_sweep.png")
    
    if isa(metrics[1], Dict)
        # Multiple metrics case
        p = plot(xlabel=param_name, ylabel=metric_name, title="Parameter Sweep")
        for (key, values) in pairs(first(metrics))
            metric_series = [m[key] for m in metrics]
            plot!(p, param_values, metric_series, label=string(key), lw=2, marker=:circle)
        end
    else
        # Single metric case
        p = plot(param_values, metrics, xlabel=param_name, ylabel=metric_name,
                 title="Parameter Sweep", lw=2, marker=:circle, legend=false)
    end
    
    savefig(p, fname)
    println("→ $fname")
    return p
end

# ============================================================================ 
# Synchronization plot
# ============================================================================ 

function synchronization_plot(t_vec::Vector{Float64}, order_parameters::Vector{Float64};
                             fname="output/synchronization.png")
    p = plot(t_vec, order_parameters, xlabel="Time", ylabel="Order Parameter",
             title="Synchronization Evolution", lw=2, legend=false)
    hline!(p, [0.5], ls=:dash, color=:red, label="Sync Threshold")
    savefig(p, fname)
    println("→ $fname")
    return p
end

# ============================================================================ 
# Belief distribution animation (saves frames)
# ============================================================================ 

function belief_distribution_animation(t_vec::Vector{Float64}, 
                                     trajectories::Vector{Vector{Float64}};
                                     frames_dir="output/frames/")
    mkpath(frames_dir)
    T_steps = length(t_vec)
    
    # Find global bounds for consistent axes
    all_beliefs = vcat(trajectories...)
    xlims = (minimum(all_beliefs) - 0.1, maximum(all_beliefs) + 0.1)
    ylims = (0, maximum([length(findall(x -> x_min <= x <= x_max, traj)) 
                        for traj in trajectories 
                        for x_min in minimum(all_beliefs):0.1:maximum(all_beliefs)
                        for x_max in (x_min+0.1):0.1:maximum(all_beliefs)]) * 1.1)
    
    for (i, t) in enumerate(t_vec[1:5:end])  # Every 5th frame to reduce size
        beliefs_t = trajectories[min(i*5, T_steps)]
        
        p = histogram(beliefs_t, bins=30, alpha=0.7, xlabel="Belief", ylabel="Count",
                     title="Belief Distribution at t=$(round(t, digits=2))",
                     xlims=xlims, legend=false, color=:steelblue)
        
        # Add vertical line for mean
        vline!(p, [mean(beliefs_t)], lw=2, color=:red, ls=:dash)
        
        savefig(p, joinpath(frames_dir, "frame_$(lpad(i, 4, '0')).png"))
    end
    
    println("→ Animation frames saved in $frames_dir")
    println("  Use: ffmpeg -framerate 10 -i frame_%04d.png -pix_fmt yuv420p animation.mp4")
end

end # module