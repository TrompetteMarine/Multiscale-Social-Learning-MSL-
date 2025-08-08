# advanced_analysis.jl - Advanced bifurcation, phase diagram, and basin analysis
module AdvancedAnalysis

using DifferentialEquations, Statistics, LinearAlgebra, StatsBase
using Distributions, Random, ProgressMeter
using Plots, ColorSchemes, Contour
using NearestNeighbors, Clustering
using Optim, ForwardDiff
using DataFrames, CSV
using UUIDs

export PhaseDiagramParams, BifurcationParams, BasinAnalysisParams
export run_phase_diagram, run_bifurcation_analysis, analyze_basins_of_attraction
export plot_phase_diagram, plot_bifurcation_2d, plot_basin_portrait
export compute_stability_indicators, find_fixed_points, continuation_analysis
export monte_carlo_phase_exploration, sensitivity_analysis

# ============================================================================
# Parameter Structures
# ============================================================================

"""
    PhaseDiagramParams
Configuration for 2D phase diagram analysis.
"""
@kwdef struct PhaseDiagramParams
    param1_name::Symbol = :α
    param1_range::Tuple{Float64, Float64} = (0.1, 2.0)
    param1_steps::Int = 20
    
    param2_name::Symbol = :λ
    param2_range::Tuple{Float64, Float64} = (0.1, 2.0)
    param2_steps::Int = 20
    
    base_params::Any = nothing  # MSLParams
    analysis_time::Float64 = 50.0
    transient_time::Float64 = 20.0
    n_realizations::Int = 3
    parallel::Bool = true
end

"""
    BifurcationParams
Configuration for bifurcation analysis.
"""
@kwdef struct BifurcationParams
    param_name::Symbol = :α
    param_range::Tuple{Float64, Float64} = (0.0, 2.0)
    n_points::Int = 100
    continuation_steps::Int = 200
    stability_check::Bool = true
    track_unstable::Bool = true
    max_period::Int = 10
    tolerance::Float64 = 1e-6
end

"""
    BasinAnalysisParams
Configuration for basin of attraction analysis.
"""
@kwdef struct BasinAnalysisParams
    x_range::Tuple{Float64, Float64} = (-3.0, 3.0)
    y_range::Tuple{Float64, Float64} = (-3.0, 3.0)
    grid_resolution::Int = 100
    integration_time::Float64 = 100.0
    convergence_threshold::Float64 = 0.01
    max_attractors::Int = 10
end

# ============================================================================
# Phase Diagram Analysis
# ============================================================================

"""
    run_phase_diagram(phase_params::PhaseDiagramParams)
Generate comprehensive 2D phase diagram exploring parameter space.
"""
function run_phase_diagram(phase_params::PhaseDiagramParams)
    p1_vals = range(phase_params.param1_range[1], phase_params.param1_range[2], 
                    length=phase_params.param1_steps)
    p2_vals = range(phase_params.param2_range[1], phase_params.param2_range[2], 
                    length=phase_params.param2_steps)
    
    # Initialize result matrices
    n_equilibria = zeros(Int, phase_params.param1_steps, phase_params.param2_steps)
    stability_index = zeros(phase_params.param1_steps, phase_params.param2_steps)
    regime_map = fill("", phase_params.param1_steps, phase_params.param2_steps)
    consensus_strength = zeros(phase_params.param1_steps, phase_params.param2_steps)
    polarization_index = zeros(phase_params.param1_steps, phase_params.param2_steps)
    lyapunov_exponents = zeros(phase_params.param1_steps, phase_params.param2_steps)
    
    total_sims = phase_params.param1_steps * phase_params.param2_steps
    progress = Progress(total_sims, desc="Computing phase diagram...")
    
    for (i, p1) in enumerate(p1_vals), (j, p2) in enumerate(p2_vals)
        # Create parameter set
        params = update_params(phase_params.base_params, 
                              phase_params.param1_name => p1,
                              phase_params.param2_name => p2)
        
        # Run multiple realizations for statistics
        results = []
        for r in 1:phase_params.n_realizations
            result = run_single_simulation(params, seed=r*1000+i*10+j)
            push!(results, result)
        end
        
        # Aggregate results
        n_eq_vals = [r.n_equilibria for r in results]
        n_equilibria[i,j] = round(Int, median(n_eq_vals))
        
        stability_index[i,j] = mean([r.stability_index for r in results])
        consensus_strength[i,j] = mean([r.consensus for r in results])
        polarization_index[i,j] = mean([r.polarization for r in results])
        lyapunov_exponents[i,j] = mean([r.lyapunov for r in results])
        
        # Classify regime
        regime_map[i,j] = classify_regime(results[1])
        
        next!(progress)
    end
    
    return Dict(
        :param1_values => collect(p1_vals),
        :param2_values => collect(p2_vals),
        :param1_name => phase_params.param1_name,
        :param2_name => phase_params.param2_name,
        :n_equilibria => n_equilibria,
        :stability => stability_index,
        :regime_map => regime_map,
        :consensus => consensus_strength,
        :polarization => polarization_index,
        :lyapunov => lyapunov_exponents
    )
end

"""
    classify_regime(result)
Classify the dynamical regime based on simulation results.
"""
function classify_regime(result)
    if result.n_equilibria == 1 && result.stability_index > 0.8
        return "Stable Consensus"
    elseif result.n_equilibria == 2
        return "Bistable"
    elseif result.n_equilibria > 2
        return "Multistable"
    elseif result.lyapunov > 0
        return "Chaotic"
    elseif result.oscillatory
        return "Oscillatory"
    else
        return "Transient"
    end
end

# ============================================================================
# Rigorous Bifurcation Analysis
# ============================================================================

"""
    run_bifurcation_analysis(bifurc_params::BifurcationParams, base_params)
Perform detailed bifurcation analysis with continuation methods.
"""
function run_bifurcation_analysis(bifurc_params::BifurcationParams, base_params)
    param_vals = range(bifurc_params.param_range[1], bifurc_params.param_range[2],
                      length=bifurc_params.n_points)
    
    # Storage for branches
    stable_branches = Dict{Int, Vector{Vector{Float64}}}()
    unstable_branches = Dict{Int, Vector{Vector{Float64}}}()
    bifurcation_points = Float64[]
    bifurcation_types = Symbol[]
    
    # Initial sweep to find branches
    println("Finding equilibrium branches...")
    
    for (idx, p_val) in enumerate(param_vals)
        params = update_params(base_params, bifurc_params.param_name => p_val)
        
        # Find fixed points
        fixed_points = find_fixed_points(params)
        
        # Check stability
        for fp in fixed_points
            is_stable = check_stability(fp, params)
            
            # Assign to branch
            branch_id = assign_to_branch(fp, stable_branches, unstable_branches, 
                                        bifurc_params.tolerance)
            
            if is_stable
                if !haskey(stable_branches, branch_id)
                    stable_branches[branch_id] = Vector{Float64}[]
                end
                push!(stable_branches[branch_id], [p_val, fp...])
            else
                if !haskey(unstable_branches, branch_id)
                    unstable_branches[branch_id] = Vector{Float64}[]
                end
                push!(unstable_branches[branch_id], [p_val, fp...])
            end
        end
        
        # Detect bifurcations
        if idx > 1
            bif_type = detect_bifurcation_type(idx, param_vals, stable_branches, 
                                              unstable_branches)
            if bif_type !== nothing
                push!(bifurcation_points, p_val)
                push!(bifurcation_types, bif_type)
            end
        end
    end
    
    # Continuation for unstable branches if requested
    if bifurc_params.track_unstable
        println("Tracking unstable manifolds...")
        for (branch_id, branch) in unstable_branches
            extended_branch = continuation_method(branch, params, bifurc_params)
            unstable_branches[branch_id] = extended_branch
        end
    end
    
    # Period-doubling cascade detection
    period_doublings = detect_period_doublings(param_vals, base_params, bifurc_params)
    
    return Dict(
        :param_values => param_vals,
        :stable_branches => stable_branches,
        :unstable_branches => unstable_branches,
        :bifurcation_points => bifurcation_points,
        :bifurcation_types => bifurcation_types,
        :period_doublings => period_doublings,
        :codim2_points => find_codimension_two_points(bifurcation_points, bifurcation_types)
    )
end

"""
    find_fixed_points(params)
Find all fixed points using multiple initial conditions and Newton's method.
"""
function find_fixed_points(params; n_attempts=20)
    fixed_points = Vector{Float64}[]
    
    # Define the system's vector field
    function vector_field!(du, u, p, t)
        # Simplified for mean-field approximation
        x = u[1]
        r = u[2]
        
        du[1] = -params.cognitive.λ * (x - r)  # Simplified
        du[2] = (1.0 / (1.0 + 1.0)) * (x - r)  # φ(m) * (x - r)
    end
    
    # Try multiple initial conditions
    for _ in 1:n_attempts
        x0 = randn() * 2.0
        r0 = x0 + randn() * 0.5
        
        # Use Newton's method via Optim
        result = optimize(u -> norm(vector_field_residual(u, params)), [x0, r0], 
                         Newton(), autodiff=:forward)
        
        if Optim.converged(result) && Optim.minimum(result) < 1e-8
            fp = Optim.minimizer(result)
            
            # Check if this is a new fixed point
            is_new = true
            for existing_fp in fixed_points
                if norm(fp - existing_fp) < 1e-4
                    is_new = false
                    break
                end
            end
            
            if is_new
                push!(fixed_points, fp)
            end
        end
    end
    
    return fixed_points
end

"""
    check_stability(fixed_point, params)
Check stability via eigenvalues of the Jacobian.
"""
function check_stability(fixed_point::Vector{Float64}, params)
    # Compute Jacobian at fixed point
    J = compute_jacobian(fixed_point, params)
    
    # Check eigenvalues
    eigenvals = eigvals(J)
    max_real_part = maximum(real.(eigenvals))
    
    return max_real_part < 0
end

"""
    continuation_method(branch, params, bifurc_params)
Numerical continuation to track solution branches.
"""
function continuation_method(branch::Vector{Vector{Float64}}, params, bifurc_params)
    if length(branch) < 2
        return branch
    end
    
    extended_branch = copy(branch)
    
    # Predictor-corrector steps
    for step in 1:bifurc_params.continuation_steps
        # Predictor: linear extrapolation
        if length(extended_branch) >= 2
            tangent = extended_branch[end] - extended_branch[end-1]
            predicted = extended_branch[end] + tangent * 0.1
            
            # Corrector: Newton's method
            corrected = newton_correction(predicted, params)
            
            if corrected !== nothing
                push!(extended_branch, corrected)
            else
                break  # Branch terminated
            end
        end
    end
    
    return extended_branch
end

# ============================================================================
# Basin of Attraction Analysis
# ============================================================================

"""
    analyze_basins_of_attraction(basin_params::BasinAnalysisParams, params)
Map basins of attraction in 2D projection of phase space.
"""
function analyze_basins_of_attraction(basin_params::BasinAnalysisParams, params)
    # Create grid of initial conditions
    x_grid = range(basin_params.x_range[1], basin_params.x_range[2], 
                   length=basin_params.grid_resolution)
    y_grid = range(basin_params.y_range[1], basin_params.y_range[2], 
                   length=basin_params.grid_resolution)
    
    # Storage
    basin_map = zeros(Int, basin_params.grid_resolution, basin_params.grid_resolution)
    attractors = Vector{Float64}[]
    attractor_types = Symbol[]
    
    println("Mapping basins of attraction...")
    progress = Progress(basin_params.grid_resolution^2)
    
    for (i, x0) in enumerate(x_grid), (j, r0) in enumerate(y_grid)
        # Integrate from this initial condition
        final_state = integrate_to_attractor([x0, r0], params, basin_params)
        
        # Classify the attractor
        attractor_id, is_new = classify_attractor(final_state, attractors, 
                                                  basin_params.convergence_threshold)
        
        if is_new && attractor_id <= basin_params.max_attractors
            push!(attractors, final_state.state)
            push!(attractor_types, final_state.type)
        end
        
        basin_map[i,j] = attractor_id
        next!(progress)
    end
    
    # Compute basin properties
    basin_sizes = compute_basin_sizes(basin_map)
    basin_boundaries = find_basin_boundaries(basin_map)
    
    # Estimate basin dimensions
    fractal_dimensions = estimate_fractal_dimensions(basin_map, basin_boundaries)
    
    return Dict(
        :x_grid => collect(x_grid),
        :y_grid => collect(y_grid),
        :basin_map => basin_map,
        :attractors => attractors,
        :attractor_types => attractor_types,
        :basin_sizes => basin_sizes,
        :basin_boundaries => basin_boundaries,
        :fractal_dimensions => fractal_dimensions,
        :n_attractors => length(attractors)
    )
end

"""
    integrate_to_attractor(ic, params, basin_params)
Integrate system from initial condition to find attractor.
"""
function integrate_to_attractor(ic::Vector{Float64}, params, basin_params)
    # Simplified 2D system for visualization
    function dynamics!(du, u, p, t)
        x, r = u
        du[1] = -params.cognitive.λ * (x - r) + params.cognitive.α * 0.5 * tanh(x)
        du[2] = (1.0 / (1.0 + 1.0)) * (x - r)
    end
    
    prob = ODEProblem(dynamics!, ic, (0.0, basin_params.integration_time), params)
    sol = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8)
    
    # Extract final state
    final_state = sol.u[end]
    
    # Determine attractor type
    if length(sol.t) > 100
        # Check for periodicity
        period = detect_period(sol.u[end-100:end])
        if period > 1
            type = :limit_cycle
        else
            type = :fixed_point
        end
    else
        type = :fixed_point
    end
    
    return (state=final_state, type=type, trajectory=sol)
end

"""
    classify_attractor(state, existing_attractors, threshold)
Determine which attractor basin a state belongs to.
"""
function classify_attractor(final_state, existing_attractors::Vector, threshold::Float64)
    for (idx, attractor) in enumerate(existing_attractors)
        if norm(final_state.state - attractor) < threshold
            return idx, false
        end
    end
    
    # New attractor
    return length(existing_attractors) + 1, true
end

"""
    compute_basin_sizes(basin_map)
Calculate relative sizes of each basin.
"""
function compute_basin_sizes(basin_map::Matrix{Int})
    n_total = length(basin_map)
    basin_counts = countmap(vec(basin_map))
    
    basin_sizes = Dict{Int, Float64}()
    for (basin_id, count) in basin_counts
        basin_sizes[basin_id] = count / n_total
    end
    
    return basin_sizes
end

"""
    find_basin_boundaries(basin_map)
Identify cells on basin boundaries.
"""
function find_basin_boundaries(basin_map::Matrix{Int})
    nx, ny = size(basin_map)
    boundaries = falses(nx, ny)
    
    for i in 2:nx-1, j in 2:ny-1
        current = basin_map[i,j]
        # Check neighbors
        neighbors = [basin_map[i-1,j], basin_map[i+1,j], 
                    basin_map[i,j-1], basin_map[i,j+1]]
        
        if any(n != current for n in neighbors)
            boundaries[i,j] = true
        end
    end
    
    return boundaries
end

"""
    estimate_fractal_dimensions(basin_map, boundaries)
Estimate fractal dimension of basin boundaries using box-counting.
"""
function estimate_fractal_dimensions(basin_map::Matrix{Int}, boundaries::BitMatrix)
    # Box-counting method
    box_sizes = [2, 4, 8, 16, 32]
    counts = Float64[]
    
    for box_size in box_sizes
        count = 0
        nx, ny = size(boundaries)
        
        for i in 1:box_size:nx-box_size+1, j in 1:box_size:ny-box_size+1
            # Check if box contains boundary
            box = boundaries[i:min(i+box_size-1, nx), j:min(j+box_size-1, ny)]
            if any(box)
                count += 1
            end
        end
        
        push!(counts, count)
    end
    
    # Fit log-log relationship
    if length(counts) > 2 && all(c > 0 for c in counts)
        log_sizes = log.(box_sizes)
        log_counts = log.(counts)
        
        # Linear regression
        A = [ones(length(log_sizes)) log_sizes]
        slope = (A' * A) \ (A' * log_counts)
        
        fractal_dim = -slope[2]
        return fractal_dim
    else
        return NaN
    end
end

# ============================================================================
# Monte Carlo Phase Space Exploration
# ============================================================================

function cut(x::Vector{T}, breaks::Vector{T}; labels::Vector{String}=String[]) where T <: Real
    # Ensure the breaks are sorted
    sort!(breaks)

    # Determine the number of breaks
    n = length(breaks)

    # Check if labels are provided, if not, create default labels
    if isempty(labels)
        labels = ["$(uuid4())_$(breaks[i])_$(breaks[i+1])" for i in 1:n-1]
    else
        if length(labels) != n - 1
            throw(ArgumentError("Number of labels must be one less than the number of breaks."))
        end
        # Append a unique UUID to each label
        labels = ["$(uuid4())_$(labels[i])" for i in 1:length(labels)]
    end

    # Initialize the result vector
    result = Vector{String}(undef, length(x))

    # Assign each element of x to its corresponding interval
    for i in eachindex(x)
        found = false
        for j in 1:n-1
            if x[i] <= breaks[j+1]
                result[i] = labels[j]
                found = true
                break
            end
        end
        if !found
            result[i] = "NA" # Assign NA if x[i] is greater than the largest break
        end
    end

    return result
end

"""
    compute_sensitivities(results, param_names)
Calculate parameter sensitivities using various methods.
"""
function compute_sensitivities(results::DataFrame, param_names::Vector)
    sensitivity_dict = Dict()
    
    for param in param_names
        sensitivity_dict[param] = Dict(
            :n_equilibria => cor(results[!, param], results.n_equilibria),
            :stability => cor(results[!, param], results.stability),
            :consensus => cor(results[!, param], results.consensus),
            :polarization => cor(results[!, param], results.polarization),
            :variance_explained => compute_variance_explained(results, param)
        )
    end
    
    return sensitivity_dict
end
"""
    monte_carlo_phase_exploration(n_samples::Int, param_ranges::Dict, base_params)
Comprehensive Monte Carlo exploration of parameter space.
"""

function create_params_from_sample(base_params, sample::DataFrameRow)
    # Create a deep copy of the base parameters to avoid modifying the original
    modified_params = deepcopy(base_params)

    # Iterate through each field in the sample and update the corresponding parameter
    for param_name in names(sample)
        param_value = sample[param_name]

        # Check if the parameter is a field of the cognitive sub-structure
        if hasfield(typeof(modified_params.cognitive), Symbol(param_name))
            setfield!(modified_params.cognitive, Symbol(param_name), param_value)
        # Check if the parameter is a field of the base_params itself
        elseif hasfield(typeof(modified_params), Symbol(param_name))
            setfield!(modified_params, Symbol(param_name), param_value)
        else
            @warn "Parameter $param_name not found in the structure."
        end
    end

    return modified_params
end


function monte_carlo_phase_exploration(n_samples::Int, param_ranges::Dict, base_params)
    # Sample parameters using Latin Hypercube
    param_samples = latin_hypercube_sample(param_ranges, n_samples)
    
    results = DataFrame()
    
    println("Running Monte Carlo exploration with $n_samples samples...")
    progress = Progress(n_samples)
    
    for i in 1:n_samples
        # Create parameter set
        params = create_params_from_sample(base_params, param_samples[i, :])
        
        # Run simulation
        sim_result = run_single_simulation(params, seed=i)
        
        # Store results
        push!(results, merge(
            param_samples[i, :],
            Dict(
                :n_equilibria => sim_result.n_equilibria,
                :stability => sim_result.stability_index,
                :consensus => sim_result.consensus,
                :polarization => sim_result.polarization,
                :regime => sim_result.regime,
                :lyapunov => sim_result.lyapunov,
                :final_variance => sim_result.variance
            )
        ))
        
        next!(progress)
    end
    
    # Compute correlations and sensitivities
    sensitivity = compute_sensitivities(results, collect(keys(param_ranges)))
    
    # Identify critical regions
    critical_regions = identify_critical_regions(results)
    
    return Dict(
        :data => results,
        :sensitivity => sensitivity,
        :critical_regions => critical_regions,
        :param_importance => rank_parameter_importance(sensitivity)
    )
end



# ============================================================================
# Visualization Functions
# ============================================================================

"""
    plot_phase_diagram(phase_data::Dict)
Create comprehensive phase diagram visualization.
"""
function plot_phase_diagram(phase_data::Dict)
    p1_name = phase_data[:param1_name]
    p2_name = phase_data[:param2_name]
    
    # Create subplots
    p1 = heatmap(phase_data[:param1_values], phase_data[:param2_values], 
                 phase_data[:n_equilibria]',
                 xlabel=string(p1_name), ylabel=string(p2_name),
                 title="Number of Equilibria",
                 c=:viridis, clims=(1, maximum(phase_data[:n_equilibria])))
    
    p2 = heatmap(phase_data[:param1_values], phase_data[:param2_values],
                 phase_data[:stability]',
                 xlabel=string(p1_name), ylabel=string(p2_name),
                 title="Stability Index",
                 c=:RdYlBu, clims=(0, 1))
    
    p3 = heatmap(phase_data[:param1_values], phase_data[:param2_values],
                 phase_data[:consensus]',
                 xlabel=string(p1_name), ylabel=string(p2_name),
                 title="Consensus Strength",
                 c=:plasma, clims=(0, 1))
    
    p4 = heatmap(phase_data[:param1_values], phase_data[:param2_values],
                 phase_data[:polarization]',
                 xlabel=string(p1_name), ylabel=string(p2_name),
                 title="Polarization Index",
                 c=:thermal, clims=(0, maximum(phase_data[:polarization])))
    
    # Add regime boundaries
    regime_colors = Dict(
        "Stable Consensus" => 1,
        "Bistable" => 2,
        "Multistable" => 3,
        "Oscillatory" => 4,
        "Chaotic" => 5,
        "Transient" => 6
    )
    
    regime_matrix = zeros(size(phase_data[:regime_map]))
    for i in 1:size(regime_matrix, 1), j in 1:size(regime_matrix, 2)
        regime_matrix[i,j] = get(regime_colors, phase_data[:regime_map][i,j], 0)
    end
    
    p5 = heatmap(phase_data[:param1_values], phase_data[:param2_values],
                 regime_matrix',
                 xlabel=string(p1_name), ylabel=string(p2_name),
                 title="Dynamical Regimes",
                 c=:Set1_6, clims=(1, 6))
    
    p6 = heatmap(phase_data[:param1_values], phase_data[:param2_values],
                 phase_data[:lyapunov]',
                 xlabel=string(p1_name), ylabel=string(p2_name),
                 title="Lyapunov Exponent",
                 c=:RdBu, clims=(-maximum(abs.(phase_data[:lyapunov])), 
                                  maximum(abs.(phase_data[:lyapunov]))))
    
    return plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1200, 900))
end

"""
    plot_bifurcation_2d(bifurc_data::Dict)
Create detailed bifurcation diagram with stable and unstable branches.
"""
function plot_bifurcation_2d(bifurc_data::Dict)
    p = plot(xlabel="Parameter", ylabel="Fixed Point Value",
             title="Bifurcation Diagram", legend=:topright)
    
    # Plot stable branches
    for (branch_id, branch) in bifurc_data[:stable_branches]
        if !isempty(branch)
            param_vals = [b[1] for b in branch]
            state_vals = [b[2] for b in branch]
            
            plot!(p, param_vals, state_vals, 
                  lw=2, color=:blue, label=(branch_id == 1 ? "Stable" : ""))
        end
    end
    
    # Plot unstable branches
    for (branch_id, branch) in bifurc_data[:unstable_branches]
        if !isempty(branch)
            param_vals = [b[1] for b in branch]
            state_vals = [b[2] for b in branch]
            
            plot!(p, param_vals, state_vals, 
                  lw=2, ls=:dash, color=:red, label=(branch_id == 1 ? "Unstable" : ""))
        end
    end
    
    # Mark bifurcation points
    for (point, bif_type) in zip(bifurc_data[:bifurcation_points], 
                                 bifurc_data[:bifurcation_types])
        if bif_type == :saddle_node
            scatter!(p, [point], [0], marker=:circle, ms=6, color=:green, 
                    label="Saddle-Node")
        elseif bif_type == :pitchfork
            scatter!(p, [point], [0], marker=:square, ms=6, color=:orange,
                    label="Pitchfork")
        elseif bif_type == :hopf
            scatter!(p, [point], [0], marker=:diamond, ms=6, color=:purple,
                    label="Hopf")
        end
    end
    
    return p
end

"""
    plot_basin_portrait(basin_data::Dict)
Visualize basins of attraction with boundaries and attractors.
"""
function plot_basin_portrait(basin_data::Dict)
    # Create colormap for basins
    n_attractors = basin_data[:n_attractors]
    colors = distinguishable_colors(n_attractors + 1, [RGB(1,1,1)])[2:end] # Ensure enough colors

    # Ensure we have at least one color, even if no attractors are found
    if isempty(colors)
        colors = [RGB(0,0,0)]  # Default to black if no attractors
    end

    # Main basin plot
    p1 = heatmap(basin_data[:x_grid], basin_data[:y_grid], basin_data[:basin_map]',  # Ensure transposition matches
                 xlabel="x (belief)", ylabel="r (reference)",
                 title="Basins of Attraction",
                 c=palette(colors[1:min(n_attractors, end)]), clims=(1, n_attractors),  # Safeguard against out-of-bounds
                 aspect_ratio=:equal)

    # Mark attractors
    for (idx, attractor) in enumerate(basin_data[:attractors])
        if idx <= length(colors)  # Ensure index safety
            if basin_data[:attractor_types][idx] == :fixed_point
                scatter!(p1, [attractor[1]], [attractor[2]],
                        marker=:star, ms=10, color=:white,
                        markerstrokecolor=:black, markerstrokewidth=2,
                        label=(idx == 1 ? "Fixed Points" : ""))
            else
                scatter!(p1, [attractor[1]], [attractor[2]],
                        marker=:circle, ms=8, color=:white,
                        markerstrokecolor=:black, markerstrokewidth=2,
                        label=(idx == 1 ? "Limit Cycles" : ""))
            end
        end
    end

    # Basin size pie chart
    sizes = collect(values(basin_data[:basin_sizes]))
    labels = ["Basin $k" for k in keys(basin_data[:basin_sizes])]

    p2 = pie(sizes, labels=labels, title="Basin Sizes",
            color=colors[1:min(n_attractors, length(colors))])

    # Fractal dimension info
    p3 = plot([], [], framestyle=:none, legend=false)
    fractal_dim = basin_data[:fractal_dimensions]
    annotate!(p3, [(0.5, 0.7, text("Basin Boundary Analysis", 14, :center)),
                   (0.5, 0.5, text("Fractal Dimension: $(round(fractal_dim, digits=3))", 12)),
                   (0.5, 0.3, text("N Attractors: $(n_attractors)", 12))])

    return plot(p1, p2, p3, layout=@layout([a{0.6w} [b; c]]), size=(1000, 600))
end

# ============================================================================
# Helper Functions
# ============================================================================

function update_params(base_params, updates...)
    # Create modified parameter set
    # Implementation depends on your MSLParams structure
    modified = deepcopy(base_params)
    
    for (param_name, value) in updates
        if param_name in [:α, :λ, :σ, :δm, :ηw, :βΘ]
            setfield!(modified.cognitive, param_name, value)
        else
            setfield!(modified, param_name, value)
        end
    end
    
    return modified
end

function run_single_simulation(params; seed=42)
    # Placeholder - integrate with your actual simulation
    # This should return a summary of the simulation results
    
    Random.seed!(seed)
    
    # Simplified result for demonstration
    return (
        n_equilibria = rand(1:3),
        stability_index = rand(),
        consensus = rand(),
        polarization = rand(),
        lyapunov = randn() * 0.1,
        regime = rand(["Stable Consensus", "Bistable", "Oscillatory"]),
        oscillatory = rand() < 0.2,
        variance = rand()
    )
end

function vector_field_residual(u, params)
    du = similar(u)
    
    x, r = u
    du[1] = -params.cognitive.λ * (x - r)
    du[2] = (1.0 / (1.0 + 1.0)) * (x - r)
    
    return du
end

function compute_jacobian(point, params)
    # Numerical Jacobian computation
    f(u) = vector_field_residual(u, params)
    return ForwardDiff.jacobian(f, point)
end

function newton_correction(predicted, params; max_iter=20, tol=1e-8)
    u = copy(predicted)
    
    for _ in 1:max_iter
        residual = vector_field_residual(u, params)
        
        if norm(residual) < tol
            return u
        end
        
        J = compute_jacobian(u, params)
        Δu = -J \ residual
        u += Δu
        
        if norm(Δu) < tol
            return u
        end
    end
    
    return nothing  # Failed to converge
end

function detect_period(trajectory; max_period=20)
    n = length(trajectory)
    if n < max_period * 2
        return 1
    end
    
    for period in 2:max_period
        is_periodic = true
        for i in 1:period
            if norm(trajectory[end-period+i] - trajectory[end-2*period+i]) > 1e-3
                is_periodic = false
                break
            end
        end
        
        if is_periodic
            return period
        end
    end
    
    return 1  # No periodicity detected
end

function latin_hypercube_sample(param_ranges::Dict{Symbol, Tuple{Float64, Float64}}, n_samples::Int)
    n_params = length(param_ranges)
    samples = DataFrame()

    for (param_name, range_tuple) in param_ranges
        # Extract the range values from the tuple
        range_start, range_end = range_tuple

        # Create stratified samples
        intervals = range(range_start, range_end, length=n_samples+1)
        points = Float64[]

        for i in 1:n_samples
            # Generate a random sample within each interval
            sample = intervals[i] + rand() * (intervals[i+1] - intervals[i])
            push!(points, sample)
        end

        # Shuffle to ensure randomness across the hypercube
        shuffle!(points)
        samples[!, param_name] = points
    end

    return samples
end
  

function detect_bifurcation_type(idx, param_vals, stable_branches, unstable_branches)
    # Simplified bifurcation type detection
    n_stable_before = count_branches_at_param(param_vals[idx-1], stable_branches)
    n_stable_after = count_branches_at_param(param_vals[idx], stable_branches)
    
    if n_stable_before != n_stable_after
        if n_stable_after > n_stable_before
            return :pitchfork
        else
            return :saddle_node
        end
    end
    
    return nothing
end

function count_branches_at_param(param_val, branches)
    count = 0
    for (_, branch) in branches
        for point in branch
            if abs(point[1] - param_val) < 1e-6
                count += 1
                break
            end
        end
    end
    return count
end

function find_codimension_two_points(bifurcation_points, bifurcation_types)
    # Find points where multiple bifurcations coincide
    codim2 = Float64[]
    
    for i in 2:length(bifurcation_points)
        if abs(bifurcation_points[i] - bifurcation_points[i-1]) < 0.1
            push!(codim2, (bifurcation_points[i] + bifurcation_points[i-1])/2)
        end
    end
    
    return codim2
end

function detect_period_doublings(param_vals, base_params, bifurc_params)
    period_doublings = Float64[]
    
    # Simplified period-doubling detection
    for p_val in param_vals[1:10:end]  # Sample subset for efficiency
        params = update_params(base_params, bifurc_params.param_name => p_val)
        
        # Check for period-2 orbit
        # This would need actual simulation in practice
        if rand() < 0.1  # Placeholder
            push!(period_doublings, p_val)
        end
    end
    
    return period_doublings
end

function assign_to_branch(fp, stable_branches, unstable_branches, tolerance)
    # Find which branch this fixed point belongs to
    all_branches = merge(stable_branches, unstable_branches)
    
    for (branch_id, branch) in all_branches
        if !isempty(branch)
            # Check distance to last point in branch
            last_point = branch[end][2:end]  # Skip parameter value
            if norm(fp - last_point) < tolerance * 10
                return branch_id
            end
        end
    end
    
    # New branch
    return maximum([0; collect(keys(all_branches))]) + 1
end

function identify_critical_regions(results::DataFrame)
    # Find regions where qualitative changes occur
    critical_regions = Dict()
    
    # Identify transition boundaries
    for col in [:n_equilibria, :regime]
        transitions = findall(diff(results[!, col]) .!= 0)
        if !isempty(transitions)
            critical_regions[col] = transitions
        end
    end
    
    return critical_regions
end

function rank_parameter_importance(sensitivity::Dict)
    # Rank parameters by total influence
    importance_scores = Dict()
    
    for (param, sens_data) in sensitivity
        # Average absolute correlation across all metrics
        total_influence = mean(abs.([
            sens_data[:n_equilibria],
            sens_data[:stability],
            sens_data[:consensus],
            sens_data[:polarization]
        ]))
        
        importance_scores[param] = total_influence
    end
    
    # Sort by importance
    return sort(collect(importance_scores), by=x->x[2], rev=true)
end

function compute_variance_explained(results::DataFrame, param::Symbol)
    # Simple variance decomposition
    total_var = var(results.consensus)
    
    # Group by parameter quintiles
    param_vals = results[!, param]
    quintiles = quantile(param_vals, [0.2, 0.4, 0.6, 0.8])
    
    groups = cut(param_vals, [-Inf; quintiles; Inf])
    group_means = combine(groupby(results, groups), :consensus => mean).consensus_mean
    
    between_var = var(group_means) * length(groups)
    
    return between_var / total_var
end

end # module