module Viz

using Statistics, StatsBase, Plots, Colors, LinearAlgebra
using GraphPlot, Graphs, NetworkLayout

export plot_msl_trajectories, plot_regime_evolution, plot_bifurcation_diagram
export plot_network_structure, plot_consensus_evolution, plot_agent_state_space
export plot_multi_scale_shifts, plot_attention_radius, plot_cognitive_dynamics
export animate_belief_evolution, plot_phase_portrait, plot_lyapunov_spectrum

# ============================================================================
# Multi-Scale Learning Trajectory Visualization
# ============================================================================

function plot_msl_trajectories(t_vec::Vector{Float64}, trajectories::Dict; 
                               sample_agents=10, fname="output/msl_trajectories.png")
    """Plot trajectories of all agent state variables"""
    
    N = length(trajectories[:beliefs])
    T_steps = length(t_vec)
    
    # Sample agents to avoid overcrowding
    if N > sample_agents
        agent_indices = rand(1:N, sample_agents)
    else
        agent_indices = 1:N
    end
    
    # Create multi-panel plot for all state variables
    p1 = plot(xlabel="Time", ylabel="Belief (x)", title="Beliefs")
    p2 = plot(xlabel="Time", ylabel="Reference (r)", title="Reference Points")
    p3 = plot(xlabel="Time", ylabel="Memory (m)", title="Memory Weights")
    p4 = plot(xlabel="Time", ylabel="Deliberation (w)", title="Deliberation Weights")
    p5 = plot(xlabel="Time", ylabel="Threshold (Θ)", title="Cognitive Thresholds")
    
    # Plot individual agent trajectories
    for i in agent_indices
        plot!(p1, t_vec, trajectories[:beliefs][i], alpha=0.6, lw=1, label="")
        plot!(p2, t_vec, trajectories[:references][i], alpha=0.6, lw=1, label="")
        plot!(p3, t_vec, trajectories[:memory][i], alpha=0.6, lw=1, label="")
        plot!(p4, t_vec, trajectories[:deliberation][i], alpha=0.6, lw=1, label="")
        plot!(p5, t_vec, trajectories[:thresholds][i], alpha=0.6, lw=1, label="")
    end
    
    # Add ensemble means
    beliefs_mean = [mean([trajectories[:beliefs][i][t] for i in 1:N]) for t in 1:T_steps]
    refs_mean = [mean([trajectories[:references][i][t] for i in 1:N]) for t in 1:T_steps]
    memory_mean = [mean([trajectories[:memory][i][t] for i in 1:N]) for t in 1:T_steps]
    delib_mean = [mean([trajectories[:deliberation][i][t] for i in 1:N]) for t in 1:T_steps]
    thresh_mean = [mean([trajectories[:thresholds][i][t] for i in 1:N]) for t in 1:T_steps]
    
    plot!(p1, t_vec, beliefs_mean, lw=3, color=:red, label="Mean")
    plot!(p2, t_vec, refs_mean, lw=3, color=:red, label="Mean")
    plot!(p3, t_vec, memory_mean, lw=3, color=:red, label="Mean")
    plot!(p4, t_vec, delib_mean, lw=3, color=:red, label="Mean")
    plot!(p5, t_vec, thresh_mean, lw=3, color=:red, label="Mean")
    
    # Combine into single plot
    combined = plot(p1, p2, p3, p4, p5, layout=(3,2), size=(1000, 800))
    
    savefig(combined, fname)
    println("→ $fname")
    return combined
end

# ============================================================================
# Regime Evolution Visualization
# ============================================================================

function plot_regime_evolution(t_vec::Vector{Float64}, analysis::Dict; 
                               fname="output/regime_evolution.png")
    """Plot the evolution of regime indicators over time"""
    
    # Extract time series data
    consensus_series = analysis[:time_series][:consensus]
    polarization_series = analysis[:time_series][:polarization]
    
    # Shift metrics
    shift_data = analysis[:shift_metrics]
    
    p1 = plot(t_vec, consensus_series, xlabel="Time", ylabel="Consensus Strength",
              title="Consensus Evolution", lw=2, color=:blue, label="Consensus")
    plot!(p1, t_vec, polarization_series, lw=2, color=:red, label="Polarization")
    
    p2 = plot(t_vec, shift_data[:micro], xlabel="Time", ylabel="Shift Metric",
              title="Multi-Scale Shift Detection", lw=2, label="Micro", color=:green)
    plot!(p2, t_vec, shift_data[:meso], lw=2, label="Meso", color=:orange)
    plot!(p2, t_vec, shift_data[:macro], lw=2, label="Macro", color=:purple)
    
    # Add regime classification
    regime = analysis[:regime]
    regime_text = "Regime: $(regime.regime)\nConfidence: $(round(regime.confidence, digits=2))"
    annotate!(p1, [(t_vec[end]*0.7, maximum(consensus_series)*0.8, text(regime_text, 10, :left))])
    
    combined = plot(p1, p2, layout=(2,1), size=(800, 600))
    savefig(combined, fname)
    println("→ $fname")
    return combined
end

# ============================================================================
# Bifurcation Diagram
# ============================================================================

function plot_bifurcation_diagram(κ_values::Vector{Float64}, bifurcation_data::Dict;
                                 fname="output/bifurcation_diagram.png")
    """Plot supercritical pitchfork bifurcation following paper Figure"""
    
    consensus_vals = bifurcation_data[:consensus]
    polarization_vals = bifurcation_data[:polarization]
    stability_vals = bifurcation_data[:stability]
    critical_point = bifurcation_data[:critical_point]
    
    p = plot(xlabel="Social Influence Parameter (α)", ylabel="Final State Metric",
             title="Supercritical Pitchfork Bifurcation", size=(800, 600))
    
    # Plot consensus and polarization
    plot!(p, κ_values, consensus_vals, lw=3, marker=:circle, label="Consensus", color=:blue)
    plot!(p, κ_values, polarization_vals, lw=3, marker=:square, label="Polarization", color=:red)
    
    # Add critical point
    κ_star = critical_point[:κ_star]
    vline!(p, [κ_star], lw=2, ls=:dash, color=:green, label="Critical α* = $(round(κ_star, digits=3))")
    
    # Add stability shading
    stable_region = κ_values .<= κ_star
    unstable_region = κ_values .> κ_star
    
    # Annotate regions
    annotate!(p, [(κ_star*0.5, maximum(consensus_vals)*0.9, text("Consensus\nRegime", 10, :center))])
    annotate!(p, [(κ_star*1.5, maximum(polarization_vals)*0.9, text("Polarization\nRegime", 10, :center))])
    
    savefig(p, fname)
    println("→ $fname")
    return p
end

# ============================================================================
# Network Structure Visualization
# ============================================================================

function plot_network_structure(W::AbstractMatrix, influence_scores=nothing; 
                               fname="output/network_structure.png")
    """Enhanced network visualization with influence coloring"""
    
    N = size(W, 1)
    
    # Convert to Graph object
    A = W .> 0  # Binary adjacency
    g = Graph(A)
    
    # Node properties
    if influence_scores !== nothing
        # Color by influence (red = high, blue = low)
        max_influence = maximum(influence_scores)
        norm_influence = max_influence > 0 ? influence_scores / max_influence : fill(0.5, N)
        node_colors = [RGB(r, 0.2, 1-r) for r in norm_influence]
        node_sizes = 8 .+ 15 * norm_influence
    else
        # Default coloring by degree
        degrees = vec(sum(A, dims=2))
        max_degree = maximum(degrees)
        norm_degrees = max_degree > 0 ? degrees / max_degree : fill(0.5, N)
        node_colors = [RGB(0.2, 0.6, 1-d*0.8) for d in norm_degrees]
        node_sizes = 8 .+ 10 * norm_degrees
    end
    
    try
        # Use spring layout
        pos = spring_layout(g)
        x_coords, y_coords = pos
        
        p = plot(size=(800, 800), aspect_ratio=:equal, showaxis=false, grid=false)
        
        # Draw edges with weight-based transparency
        for edge in edges(g)
            i, j = src(edge), dst(edge)
            edge_weight = W[i, j]
            alpha_val = 0.1 + 0.7 * edge_weight  # Scale transparency by weight
            
            plot!(p, [x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 
                  color=:gray, alpha=alpha_val, lw=1 + 2*edge_weight, label="")
        end
        
        # Draw nodes
        scatter!(p, x_coords, y_coords, 
                 color=node_colors, ms=node_sizes, 
                 strokewidth=1, strokecolor=:black, label="")
        
        # Add title and colorbar info
        title_text = influence_scores !== nothing ? "Social Learning Network\n(Node Color = Influence)" : 
                                                   "Social Learning Network\n(Node Color = Degree)"
        title!(p, title_text)
        
        savefig(p, fname)
        println("→ $fname")
        return p
        
    catch e
        @warn "Network visualization failed: $e. Creating adjacency heatmap instead."
        
        # Fallback to heatmap
        p = heatmap(Matrix(W), xlabel="Agent", ylabel="Agent", 
                    title="Network Adjacency Matrix", c=:viridis)
        savefig(p, fname)
        println("→ $fname (heatmap fallback)")
        return p
    end
end

# ============================================================================
# Consensus Evolution with Confidence Bands
# ============================================================================

function plot_consensus_evolution(t_vec::Vector{Float64}, ensemble_results::Vector;
                                 fname="output/consensus_evolution.png")
    """Plot consensus evolution with confidence bands from ensemble"""
    
    T_steps = length(t_vec)
    
    # Extract consensus trajectories from ensemble
    consensus_trajectories = []
    for result in ensemble_results
        traj = result.trajectories
        N = length(traj[:beliefs])
        consensus_t = Float64[]
        
        for t in 1:T_steps
            beliefs_t = [traj[:beliefs][i][t] for i in 1:N]
            consensus_val = 1.0 - std(beliefs_t) / sqrt(N)  # Normalized consensus
            push!(consensus_t, max(0, consensus_val))
        end
        
        push!(consensus_trajectories, consensus_t)
    end
    
    # Compute statistics
    consensus_mean = [mean([traj[t] for traj in consensus_trajectories]) for t in 1:T_steps]
    consensus_std = [std([traj[t] for traj in consensus_trajectories]) for t in 1:T_steps]
    consensus_q25 = [quantile([traj[t] for traj in consensus_trajectories], 0.25) for t in 1:T_steps]
    consensus_q75 = [quantile([traj[t] for traj in consensus_trajectories], 0.75) for t in 1:T_steps]
    
    # Create plot with confidence bands
    p = plot(t_vec, consensus_mean, lw=3, color=:blue, label="Mean Consensus")
    plot!(p, t_vec, consensus_mean, ribbon=(consensus_mean - consensus_q25, consensus_q75 - consensus_mean),
          alpha=0.3, color=:blue, label="25%-75% Range")
    plot!(p, t_vec, consensus_mean, ribbon=consensus_std, alpha=0.2, color=:lightblue, label="±1 Std")
    
    xlabel!(p, "Time")
    ylabel!(p, "Consensus Strength")
    title!(p, "Consensus Evolution (N=$(length(ensemble_results)) realizations)")
    
    savefig(p, fname)
    println("→ $fname")
    return p
end

# ============================================================================
# Agent State Space Visualization
# ============================================================================

function plot_agent_state_space(trajectories::Dict, t_indices=[1, length(trajectories[:beliefs][1])÷2, length(trajectories[:beliefs][1])];
                               fname="output/agent_state_space.png")
    """Plot 3D agent state space evolution (beliefs vs memory vs thresholds)"""
    
    N = length(trajectories[:beliefs])
    
    # Extract states at different time points
    colors = [:blue, :orange, :red]
    labels = ["Initial", "Mid-simulation", "Final"]
    
    p = plot(xlabel="Belief (x)", ylabel="Memory (m)", zlabel="Threshold (Θ)",
             title="Agent State Space Evolution", camera=(45, 30))
    
    for (i, t_idx) in enumerate(t_indices)
        beliefs_t = [trajectories[:beliefs][j][t_idx] for j in 1:N]
        memory_t = [trajectories[:memory][j][t_idx] for j in 1:N]
        thresholds_t = [trajectories[:thresholds][j][t_idx] for j in 1:N]
        
        scatter!(p, beliefs_t, memory_t, thresholds_t, 
                 color=colors[i], alpha=0.6, ms=4, label=labels[i])
    end
    
    savefig(p, fname)
    println("→ $fname")
    return p
end

# ============================================================================
# Multi-Scale Shift Detection Visualization
# ============================================================================

function plot_multi_scale_shifts(t_vec::Vector{Float64}, shift_data::Dict;
                                fname="output/multi_scale_shifts.png")
    """Plot shift metrics across all scales"""
    
    p1 = plot(t_vec, shift_data[:micro], xlabel="Time", ylabel="Micro Shift",
              title="Individual Agent Level", lw=2, color=:green)
    
    p2 = plot(t_vec, shift_data[:meso], xlabel="Time", ylabel="Meso Shift", 
              title="Community Level", lw=2, color=:orange)
    
    p3 = plot(t_vec, shift_data[:macro], xlabel="Time", ylabel="Macro Shift",
              title="Population Level", lw=2, color=:purple)
    
    # Add detection thresholds
    threshold_micro = 0.1
    threshold_meso = 0.08
    threshold_macro = 0.05
    
    hline!(p1, [threshold_micro], ls=:dash, color=:red, alpha=0.7, label="Threshold")
    hline!(p2, [threshold_meso], ls=:dash, color=:red, alpha=0.7, label="Threshold")
    hline!(p3, [threshold_macro], ls=:dash, color=:red, alpha=0.7, label="Threshold")
    
    combined = plot(p1, p2, p3, layout=(3,1), size=(800, 800))
    suptitle!(combined, "Multi-Scale Shift Detection")
    
    savefig(combined, fname)
    println("→ $fname")
    return combined
end

# ============================================================================
# Attention Radius Visualization
# ============================================================================

function plot_attention_radius(trajectories::Dict; fname="output/attention_radius.png")
    """Plot attention radius function and agent thresholds"""
    
    # Attention radius function ε(Θ)
    Θ_range = 0.1:0.01:3.0
    ε_values = [2.0 * (1.0 - exp(-Θ / 2.0)) for Θ in Θ_range]  # Example concave function
    
    p1 = plot(Θ_range, ε_values, xlabel="Cognitive Threshold (Θ)", ylabel="Attention Radius ε(Θ)",
              title="Attention Radius Function", lw=3, color=:blue, label="ε(Θ)")
    
    # Show agent threshold distribution
    final_thresholds = [trajectories[:thresholds][i][end] for i in 1:length(trajectories[:thresholds])]
    
    p2 = histogram(final_thresholds, bins=20, xlabel="Final Threshold (Θ)", ylabel="Count",
                   title="Agent Threshold Distribution", alpha=0.7, color=:lightblue, legend=false)
    
    combined = plot(p1, p2, layout=(2,1), size=(800, 600))
    savefig(combined, fname)
    println("→ $fname")
    return combined
end

# ============================================================================
# Cognitive Dynamics Visualization
# ============================================================================

function plot_cognitive_dynamics(trajectories::Dict, t_vec::Vector{Float64};
                                fname="output/cognitive_dynamics.png")
    """Plot cognitive load and tension evolution"""
    
    N = length(trajectories[:beliefs])
    T_steps = length(t_vec)
    
    # Compute cognitive tension |x - r|
    cognitive_tensions = []
    cognitive_loads = []
    
    for t in 1:T_steps
        tensions_t = [abs(trajectories[:beliefs][i][t] - trajectories[:references][i][t]) for i in 1:N]
        loads_t = [trajectories[:memory][i][t] * trajectories[:deliberation][i][t] * trajectories[:thresholds][i][t] for i in 1:N]
        
        push!(cognitive_tensions, mean(tensions_t))
        push!(cognitive_loads, mean(loads_t))
    end
    
    p1 = plot(t_vec, cognitive_tensions, xlabel="Time", ylabel="Mean Cognitive Tension |x-r|",
              title="Cognitive Tension Evolution", lw=2, color=:red)
    
    p2 = plot(t_vec, cognitive_loads, xlabel="Time", ylabel="Mean Cognitive Load (m×w×Θ)",
              title="Cognitive Load Evolution", lw=2, color=:purple)
    
    # Add threshold exceedance events
    threshold_mean = mean([mean(trajectories[:thresholds][i]) for i in 1:N])
    exceedance_times = t_vec[cognitive_tensions .> threshold_mean]
    
    if !isempty(exceedance_times)
        scatter!(p1, exceedance_times, cognitive_tensions[cognitive_tensions .> threshold_mean],
                 color=:orange, ms=4, label="Threshold Exceedance", alpha=0.7)
    end
    
    combined = plot(p1, p2, layout=(2,1), size=(800, 600))
    savefig(combined, fname)
    println("→ $fname")
    return combined
end

# ============================================================================
# Phase Portrait Visualization
# ============================================================================

function plot_phase_portrait(trajectories::Dict, agent_idx=1; fname="output/phase_portrait.png")
    """Plot phase portrait (belief vs velocity) for single agent"""
    
    beliefs = trajectories[:beliefs][agent_idx]
    
    # Compute velocities (discrete derivatives)
    dt = 0.1  # Assume standard time step
    velocities = diff(beliefs) ./ dt
    
    p = scatter(beliefs[2:end], velocities, alpha=0.6, ms=2, xlabel="Belief", ylabel="Belief Velocity",
                title="Phase Portrait (Agent $agent_idx)", legend=false, color=:blue)
    
    # Add trajectory direction arrows (subsample for clarity)
    arrow_indices = 1:max(1, length(velocities)÷20):length(velocities)
    for i in arrow_indices
        if i < length(velocities)
            quiver!(p, [beliefs[i+1]], [velocities[i]], 
                   quiver=([beliefs[i+2] - beliefs[i+1]], [velocities[i+1] - velocities[i]]),
                   color=:red, alpha=0.5)
        end
    end
    
    savefig(p, fname)
    println("→ $fname")
    return p
end

# ============================================================================
# Animation Functions
# ============================================================================

function animate_belief_evolution(t_vec::Vector{Float64}, trajectories::Dict;
                                 frames_dir="output/animation_frames/", fps=10)
    """Create animation frames for belief evolution"""
    
    mkpath(frames_dir)
    N = length(trajectories[:beliefs])
    T_steps = length(t_vec)
    
    # Find bounds for consistent axes
    all_beliefs = vcat([trajectories[:beliefs][i] for i in 1:N]...)
    xlims = (minimum(all_beliefs) - 0.5, maximum(all_beliefs) + 0.5)
    
    # Create frames (every 5th time step to reduce size)
    frame_indices = 1:5:T_steps
    
    println("Creating $(length(frame_indices)) animation frames...")
    
    for (frame_num, t_idx) in enumerate(frame_indices)
        beliefs_t = [trajectories[:beliefs][i][t_idx] for i in 1:N]
        
        p = histogram(beliefs_t, bins=25, alpha=0.7, xlabel="Belief", ylabel="Number of Agents",
                     title="Belief Distribution at t=$(round(t_vec[t_idx], digits=1))",
                     xlims=xlims, ylims=(0, N÷3), legend=false, color=:steelblue)
        
        # Add mean line
        vline!(p, [mean(beliefs_t)], lw=3, color=:red, ls=:dash, label="Mean")
        
        # Save frame
        frame_file = joinpath(frames_dir, "frame_$(lpad(frame_num, 4, '0')).png")
        savefig(p, frame_file)
        
        if frame_num % 20 == 0
            println("  Created $frame_num/$(length(frame_indices)) frames")
        end
    end
    
    println("→ Animation frames saved in $frames_dir")
    println("  Create video with: ffmpeg -framerate $fps -i frame_%04d.png -pix_fmt yuv420p belief_evolution.mp4")
    
    return frames_dir
end

# ============================================================================
# Utility Functions
# ============================================================================

function suptitle!(p, title_text::String)
    """Add super title to multi-panel plot"""
    plot!(p, title=title_text, titlelocation=:center, titlefontsize=16)
end

# Set default plotting parameters for publication quality
function set_publication_theme!()
    """Set plotting defaults for publication-quality figures"""
    default(
        fontfamily="Computer Modern",
        titlefontsize=14,
        guidefontsize=12,
        tickfontsize=10,
        legendfontsize=10,
        linewidth=2,
        markersize=4,
        dpi=300,
        size=(800, 600)
    )
end

end # module
