# ============================================================================
# FILE 1: BifurcationAnalysis.jl
# ============================================================================

# BifurcationAnalysis.jl - Bifurcation and continuation analysis module
module BifurcationAnalysis

using LinearAlgebra, Statistics, ProgressMeter
using Optim, ForwardDiff
using ..Types, ..Utils

export run_bifurcation_analysis, continuation_analysis
export detect_bifurcation_type, find_codimension_two_points
export track_unstable_branches, detect_period_doublings

# Main bifurcation analysis function
function run_bifurcation_analysis(params::BifurcationParams, base_params)
    param_vals = range(params.param_range[1], params.param_range[2],
                      length=params.n_points)
    
    # Storage for branches
    stable_branches = Dict{Int, Vector{Vector{Float64}}}()
    unstable_branches = Dict{Int, Vector{Vector{Float64}}}()
    bifurcation_points = Float64[]
    bifurcation_types = Symbol[]
    
    println("Finding equilibrium branches...")
    progress = Progress(params.n_points, desc="Bifurcation analysis...")
    
    for (idx, p_val) in enumerate(param_vals)
        # Update parameters
        sim_params = Utils.update_params(base_params, params.param_name => p_val)
        
        # Find fixed points
        fixed_points = Utils.find_fixed_points(sim_params)
        
        # Check stability and assign to branches
        for fp in fixed_points
            is_stable = Utils.check_stability(fp, sim_params)
            branch_id = assign_to_branch(fp, stable_branches, unstable_branches, 
                                        params.tolerance)
            
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
            bif_type = detect_bifurcation_transition(
                idx, param_vals, stable_branches, unstable_branches
            )
            if bif_type !== nothing
                push!(bifurcation_points, p_val)
                push!(bifurcation_types, bif_type)
            end
        end
        
        next!(progress)
    end
    
    # Track unstable branches if requested
    if params.track_unstable
        for (branch_id, branch) in unstable_branches
            extended = continuation_analysis(branch, base_params, params)
            unstable_branches[branch_id] = extended
        end
    end
    
    # Detect period-doubling cascade
    period_doublings = detect_period_doublings(param_vals, base_params, params)
    
    # Find codimension-two points
    codim2 = find_codimension_two_points(bifurcation_points, bifurcation_types)
    
    return BifurcationResult(
        param_vals,
        params.param_name,
        stable_branches,
        unstable_branches,
        bifurcation_points,
        bifurcation_types,
        period_doublings,
        codim2
    )
end

function detect_bifurcation_transition(idx, param_vals, stable_branches, unstable_branches)
    # Count branches at current and previous parameter values
    n_stable_before = count_branches_at_param(param_vals[idx-1], stable_branches)
    n_stable_after = count_branches_at_param(param_vals[idx], stable_branches)
    
    if n_stable_before != n_stable_after
        if n_stable_after > n_stable_before
            return :pitchfork
        else
            return :saddle_node
        end
    end
    
    # Check for Hopf bifurcation (stability change without branch change)
    # This would require eigenvalue analysis
    
    return nothing
end

function continuation_analysis(branch, base_params, bifurc_params)
    if length(branch) < 2
        return branch
    end
    
    extended = copy(branch)
    
    # Predictor-corrector continuation
    for step in 1:bifurc_params.continuation_steps
        if length(extended) >= 2
            # Predictor: tangent prediction
            tangent = extended[end] - extended[end-1]
            predicted = extended[end] + tangent * 0.01
            
            # Corrector: Newton's method
            corrected = newton_correction(predicted, base_params, bifurc_params)
            
            if corrected !== nothing
                push!(extended, corrected)
            else
                break
            end
        end
    end
    
    return extended
end

function newton_correction(predicted, base_params, bifurc_params)
    # Simplified Newton correction
    # In practice, would solve F(u, λ) = 0 with continuation constraint
    return predicted  # Placeholder
end

function detect_period_doublings(param_vals, base_params, bifurc_params)
    # Detect period-doubling bifurcations
    period_doublings = Float64[]
    
    # Sample parameter values
    for p_val in param_vals[1:10:end]
        sim_params = Utils.update_params(base_params, bifurc_params.param_name => p_val)
        
        # Run simulation and check for period-2 orbits
        # This is a placeholder - would need actual simulation
        if rand() < 0.1  # Placeholder condition
            push!(period_doublings, p_val)
        end
    end
    
    return period_doublings
end

function find_codimension_two_points(bifurcation_points, bifurcation_types)
    # Find points where multiple bifurcations coincide
    codim2 = Float64[]
    
    for i in 2:length(bifurcation_points)
        if abs(bifurcation_points[i] - bifurcation_points[i-1]) < 0.01
            push!(codim2, (bifurcation_points[i] + bifurcation_points[i-1])/2)
        end
    end
    
    return unique(codim2)
end

function assign_to_branch(fp, stable_branches, unstable_branches, tolerance)
    all_branches = merge(stable_branches, unstable_branches)
    
    for (branch_id, branch) in all_branches
        if !isempty(branch)
            last_point = branch[end][2:end]
            if norm(fp - last_point) < tolerance * 10
                return branch_id
            end
        end
    end
    
    # New branch
    return maximum([0; collect(keys(all_branches))]) + 1
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

function track_unstable_branches(unstable_branches, base_params, bifurc_params)
    # Enhanced tracking of unstable manifolds
    tracked = Dict{Int, Vector{Vector{Float64}}}()
    
    for (branch_id, branch) in unstable_branches
        extended = continuation_analysis(branch, base_params, bifurc_params)
        tracked[branch_id] = extended
    end
    
    return tracked
end

function detect_bifurcation_type(eigenvalues_before, eigenvalues_after)
    # Classify bifurcation based on eigenvalue changes
    # Placeholder implementation
    return :unknown
end

end # module BifurcationAnalysis.jl