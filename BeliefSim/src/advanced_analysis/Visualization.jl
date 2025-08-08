# Visualization.jl - Visualization module for advanced analysis
module Visualization

using Plots, ColorSchemes, StatsPlots
using DataFrames, Statistics
using ..Types

export plot_phase_diagram, plot_bifurcation_diagram, plot_basin_portrait
export plot_sensitivity_heatmap, plot_monte_carlo_results
export create_summary_report, plot_time_series_ensemble

function plot_phase_diagram(result::PhaseDiagramResult)
    # Create comprehensive phase diagram visualization
    
    p1 = heatmap(result.param1_values, result.param2_values, 
                 result.n_equilibria',
                 xlabel=string(result.param1_name), 
                 ylabel=string(result.param2_name),
                 title="Number of Equilibria",
                 c=:viridis, clims=(1, maximum(result.n_equilibria)))
    
    p2 = heatmap(result.param1_values, result.param2_values,
                 result.stability',
                 xlabel=string(result.param1_name), 
                 ylabel=string(result.param2_name),
                 title="Stability Index",
                 c=:RdYlBu, clims=(0, 1))
    
    p3 = heatmap(result.param1_values, result.param2_values,
                 result.consensus',
                 xlabel=string(result.param1_name), 
                 ylabel=string(result.param2_name),
                 title="Consensus Strength",
                 c=:plasma, clims=(0, 1))
    
    p4 = heatmap(result.param1_values, result.param2_values,
                 result.polarization',
                 xlabel=string(result.param1_name), 
                 ylabel=string(result.param2_name),
                 title="Polarization Index",
                 c=:thermal, clims=(0, maximum(result.polarization)))
    
    # Create regime map with discrete colors
    regime_colors = create_regime_colormap(result.regime_map)
    
    p5 = heatmap(result.param1_values, result.param2_values,
                 regime_colors',
                 xlabel=string(result.param1_name), 
                 ylabel=string(result.param2_name),
                 title="Dynamical Regimes",
                 c=:Set1_6, clims=(1, 6))
    
    p6 = heatmap(result.param1_values, result.param2_values,
                 result.lyapunov',
                 xlabel=string(result.param1_name), 
                 ylabel=string(result.param2_name),
                 title="Lyapunov Exponent",
                 c=:RdBu, 
                 clims=(-maximum(abs.(result.lyapunov)), 
                        maximum(abs.(result.lyapunov))))
    
    return plot(p1, p2, p3, p4, p5, p6, 
                layout=(3,2), size=(1200, 900),
                plot_title="Phase Diagram Analysis")
end

function plot_bifurcation_diagram(result::BifurcationResult)
    p = plot(xlabel=string(result.param_name), 
             ylabel="Fixed Point Value",
             title="Bifurcation Diagram",
             legend=:topright,
             grid=true)
    
    # Plot stable branches
    for (branch_id, branch) in result.stable_branches
        if !isempty(branch)
            param_vals = [b[1] for b in branch]
            state_vals = [b[2] for b in branch]
            
            plot!(p, param_vals, state_vals, 
                  lw=2, color=:blue, 
                  label=(branch_id == 1 ? "Stable" : ""))
        end
    end
    
    # Plot unstable branches
    for (branch_id, branch) in result.unstable_branches
        if !isempty(branch)
            param_vals = [b[1] for b in branch]
            state_vals = [b[2] for b in branch]
            
            plot!(p, param_vals, state_vals, 
                  lw=2, ls=:dash, color=:red, 
                  label=(branch_id == 1 ? "Unstable" : ""))
        end
    end
    
    # Mark bifurcation points
    for (point, bif_type) in zip(result.bifurcation_points, 
                                 result.bifurcation_types)
        marker_props = get_bifurcation_marker(bif_type)
        scatter!(p, [point], [0], 
                marker=marker_props.shape,
                ms=marker_props.size,
                color=marker_props.color,
                label=string(bif_type))
    end
    
    return p
end

function plot_basin_portrait(result::BasinResult)
    # Create basin visualization with attractors
    
    # Main basin plot
    p1 = heatmap(result.x_grid, result.y_grid, result.basin_map',
                 xlabel="x (belief)", ylabel="r (reference)",
                 title="Basins of Attraction",
                 c=:Set1_9, 
                 aspect_ratio=:equal)
    
    # Mark attractors
    for (idx, attractor) in enumerate(result.attractors)
        marker = result.attractor_types[idx] == :fixed_point ? :star : :circle
        scatter!(p1, [attractor[1]], [attractor[2]],
                marker=marker, ms=10, color=:white,
                markerstrokecolor=:black, markerstrokewidth=2,
                label=(idx == 1 ? "Attractors" : ""))
    end
    
    # Basin size pie chart
    sizes = collect(values(result.basin_sizes))
    labels = ["Basin $k" for k in keys(result.basin_sizes)]
    
    p2 = pie(sizes, labels=labels, 
             title="Basin Sizes",
             legend=:outertopright)
    
    # Info panel
    p3 = plot([], [], framestyle=:none, legend=false)
    annotate!(p3, [
        (0.5, 0.8, text("Basin Analysis", 14, :center)),
        (0.5, 0.6, text("Fractal Dimension: $(round(result.fractal_dimensions, digits=3))", 12)),
        (0.5, 0.4, text("N Attractors: $(result.n_attractors)", 12)),
        (0.5, 0.2, text("Largest Basin: $(round(maximum(sizes)*100, digits=1))%", 12))
    ])
    
    return plot(p1, p2, p3, 
                layout=@layout([a{0.6w} [b; c]]), 
                size=(1000, 600))
end

function plot_sensitivity_heatmap(sensitivity::Dict)
    # Create sensitivity heatmap
    params = collect(keys(sensitivity))
    metrics = [:consensus, :polarization, :n_equilibria, :stability]
    
    # Build matrix
    sens_matrix = zeros(length(params), length(metrics))
    
    for (i, param) in enumerate(params)
        for (j, metric) in enumerate(metrics)
            if haskey(sensitivity[param], metric)
                sens_matrix[i, j] = abs(sensitivity[param][metric])
            end
        end
    end
    
    p = heatmap(string.(metrics), string.(params), sens_matrix,
                xlabel="Output Metric", ylabel="Parameter",
                title="Parameter Sensitivity",
                c=:Blues, clims=(0, 1))
    
    # Add values
    for i in 1:length(params), j in 1:length(metrics)
        annotate!(p, j, i, text(round(sens_matrix[i,j], digits=2), 8))
    end
    
    return p
end

function plot_monte_carlo_results(result::MonteCarloResult)
    # Create multi-panel Monte Carlo visualization
    plots = []
    
    # Parameter importance
    if !isempty(result.param_importance)
        params = [p[1] for p in result.param_importance[1:min(10, end)]]
        importance = [p[2] for p in result.param_importance[1:min(10, end)]]
        
        p1 = bar(string.(params), importance,
                 xlabel="Parameter", ylabel="Importance",
                 title="Parameter Importance Ranking",
                 legend=false, rotation=45)
        push!(plots, p1)
    end
    
    # Distribution plots if data available
    if isa(result.data, DataFrame) && nrow(result.data) > 0
        # Consensus distribution
        if :consensus in names(result.data)
            p2 = histogram(result.data.consensus,
                          xlabel="Consensus", ylabel="Count",
                          title="Consensus Distribution",
                          bins=20, legend=false)
            push!(plots, p2)
        end
        
        # Regime distribution
        if :regime in names(result.data)
            regime_counts = countmap(result.data.regime)
            p3 = bar(collect(keys(regime_counts)), 
                    collect(values(regime_counts)),
                    xlabel="Regime", ylabel="Count",
                    title="Regime Distribution",
                    legend=false, rotation=45)
            push!(plots, p3)
        end
    end
    
    if length(plots) > 0
        return plot(plots..., layout=(length(plots), 1), size=(800, 300*length(plots)))
    else
        return plot(title="No Monte Carlo data to visualize")
    end
end

function create_summary_report(all_results::Dict)
    # Create comprehensive summary visualization
    plots = []
    
    # Add each analysis type if present
    if haskey(all_results, :phase_diagram)
        push!(plots, plot_phase_diagram(all_results[:phase_diagram]))
    end
    
    if haskey(all_results, :bifurcation)
        push!(plots, plot_bifurcation_diagram(all_results[:bifurcation]))
    end
    
    if haskey(all_results, :basins)
        push!(plots, plot_basin_portrait(all_results[:basins]))
    end
    
    if haskey(all_results, :monte_carlo)
        push!(plots, plot_monte_carlo_results(all_results[:monte_carlo]))
    end
    
    return plots
end

# Helper functions
function create_regime_colormap(regime_map::Matrix{String})
    unique_regimes = unique(vec(regime_map))
    regime_to_color = Dict(r => i for (i, r) in enumerate(unique_regimes))
    
    color_matrix = zeros(size(regime_map))
    for i in 1:size(regime_map, 1)
        for j in 1:size(regime_map, 2)
            color_matrix[i,j] = regime_to_color[regime_map[i,j]]
        end
    end
    
    return color_matrix
end

function get_bifurcation_marker(bif_type::Symbol)
    if bif_type == :saddle_node
        return (shape=:circle, size=6, color=:green)
    elseif bif_type == :pitchfork
        return (shape=:square, size=6, color=:orange)
    elseif bif_type == :hopf
        return (shape=:diamond, size=6, color=:purple)
    else
        return (shape=:star, size=5, color=:gray)
    end
end

function countmap(x)
    counts = Dict{eltype(x), Int}()
    for val in x
        counts[val] = get(counts, val, 0) + 1
    end
    return counts
end

end # module Visualization