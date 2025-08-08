# BasinAnalysis.jl - Basin of attraction analysis module
module BasinAnalysis

using LinearAlgebra, Statistics, ProgressMeter
using DifferentialEquations
using ..Types, ..Utils

export analyze_basins_of_attraction, compute_basin_sizes
export find_basin_boundaries, estimate_fractal_dimension
export integrate_to_attractor, classify_attractor

function analyze_basins_of_attraction(params::BasinAnalysisParams, sim_params)
    # Create grid of initial conditions
    x_grid = range(params.x_range[1], params.x_range[2], 
                   length=params.grid_resolution)
    y_grid = range(params.y_range[1], params.y_range[2], 
                   length=params.grid_resolution)
    
    # Storage
    basin_map = zeros(Int, params.grid_resolution, params.grid_resolution)
    attractors = Vector{Float64}[]
    attractor_types = Symbol[]
    
    println("Mapping basins of attraction...")
    progress = Progress(params.grid_resolution^2)
    
    for (i, x0) in enumerate(x_grid)
        for (j, r0) in enumerate(y_grid)
            # Integrate from this initial condition
            final_state = integrate_to_attractor([x0, r0], sim_params, params)
            
            # Classify the attractor
            attractor_id, is_new = classify_attractor(
                final_state, attractors, params.convergence_threshold
            )
            
            if is_new && attractor_id <= params.max_attractors
                push!(attractors, final_state.state)
                push!(attractor_types, final_state.type)
            end
            
            basin_map[i,j] = attractor_id
            next!(progress)
        end
    end
    
    # Compute basin properties
    basin_sizes = compute_basin_sizes(basin_map)
    basin_boundaries = find_basin_boundaries(basin_map)
    fractal_dim = Utils.estimate_fractal_dimension(basin_boundaries)
    
    return BasinResult(
        collect(x_grid),
        collect(y_grid),
        basin_map,
        attractors,
        attractor_types,
        basin_sizes,
        basin_boundaries,
        fractal_dim,
        length(attractors)
    )
end

function integrate_to_attractor(ic::Vector{Float64}, params, basin_params)
    # Simplified 2D dynamics for basin analysis
    function dynamics!(du, u, p, t)
        x, r = u
        du[1] = -params.cognitive.λ * (x - r) + params.cognitive.α * 0.5 * tanh(x)
        du[2] = (1.0 / (1.0 + 1.0)) * (x - r)
    end
    
    prob = ODEProblem(dynamics!, ic, (0.0, basin_params.integration_time), params)
    
    try
        sol = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8)
        
        # Extract final state
        final_state = sol.u[end]
        
        # Determine attractor type
        period = Utils.detect_period(sol.u[max(1, end-100):end])
        type = period > 1 ? :limit_cycle : :fixed_point
        
        return (state=final_state, type=type, trajectory=sol)
    catch
        # If integration fails, return initial condition as "escaped"
        return (state=ic, type=:escaped, trajectory=nothing)
    end
end

function classify_attractor(final_state, existing_attractors::Vector, threshold::Float64)
    for (idx, attractor) in enumerate(existing_attractors)
        if norm(final_state.state - attractor) < threshold
            return idx, false
        end
    end
    
    # New attractor
    return length(existing_attractors) + 1, true
end

function compute_basin_sizes(basin_map::Matrix{Int})
    n_total = length(basin_map)
    basin_counts = Dict{Int, Int}()
    
    for val in vec(basin_map)
        basin_counts[val] = get(basin_counts, val, 0) + 1
    end
    
    basin_sizes = Dict{Int, Float64}()
    for (basin_id, count) in basin_counts
        basin_sizes[basin_id] = count / n_total
    end
    
    return basin_sizes
end

function find_basin_boundaries(basin_map::Matrix{Int})
    nx, ny = size(basin_map)
    boundaries = falses(nx, ny)
    
    for i in 2:nx-1
        for j in 2:ny-1
            current = basin_map[i,j]
            neighbors = [
                basin_map[i-1,j], basin_map[i+1,j],
                basin_map[i,j-1], basin_map[i,j+1]
            ]
            
            if any(n != current for n in neighbors)
                boundaries[i,j] = true
            end
        end
    end
    
    return boundaries
end

function compute_wada_property(basin_map::Matrix{Int}, boundaries::BitMatrix)
    # Check if basins have the Wada property
    # (every boundary point is on the boundary of all basins)
    # Simplified check
    n_basins = length(unique(basin_map))
    return n_basins > 2  # Placeholder
end

function estimate_basin_stability(basin_map::Matrix{Int}, n_perturbations::Int=100)
    # Estimate how stable basin assignments are to perturbations
    # Placeholder implementation
    return 0.5
end

end # module BasinAnalysis
