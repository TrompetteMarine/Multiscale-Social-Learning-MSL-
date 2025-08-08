# Utils.jl - Shared utilities for advanced analysis
"""
    Utils

Shared utility functions for advanced analysis modules.
Provides parameter manipulation, data processing, and numerical utilities.
"""
module Utils

using LinearAlgebra, Statistics, Random
using ForwardDiff, Optim

export update_params, run_single_simulation
export find_fixed_points, check_stability, compute_jacobian
export detect_period, cluster_positions, estimate_fractal_dimension
export simple_bin, moving_average, normalize_data

# ============================================================================
# Parameter Manipulation
# ============================================================================

"""
    update_params(base_params, updates...)

Create a modified parameter set with specified updates.

# Arguments
- `base_params`: Base MSLParams object
- `updates...`: Pairs of (param_name, value)

# Returns
- Modified MSLParams object
"""
function update_params(base_params, updates...)
    modified = deepcopy(base_params)
    
    for (param_name, value) in updates
        if param_name in [:α, :λ, :σ, :δm, :ηw, :βΘ]
            # Update cognitive parameters
            cog_dict = Dict{Symbol, Any}(
                :λ => modified.cognitive.λ,
                :α => modified.cognitive.α,
                :σ => modified.cognitive.σ,
                :δm => modified.cognitive.δm,
                :ηw => modified.cognitive.ηw,
                :βΘ => modified.cognitive.βΘ,
                :Δ => modified.cognitive.Δ,
                :m_min => modified.cognitive.m_min,
                :m_max => modified.cognitive.m_max,
                :w_min => modified.cognitive.w_min,
                :w_max => modified.cognitive.w_max,
                :Θ_min => modified.cognitive.Θ_min,
                :Θ_max => modified.cognitive.Θ_max,
                :ε_max => modified.cognitive.ε_max
            )
            
            cog_dict[param_name] = value
            
            # Reconstruct with updated cognitive params
            # This assumes BeliefSim module is available
            modified = Main.BeliefSim.MSLParams(
                modified.N, modified.T, modified.Δt,
                Main.BeliefSim.CognitiveParams(; cog_dict...),
                modified.network_type,
                modified.network_params,
                modified.ν,
                modified.save_interval
            )
        elseif param_name in [:N, :T, :Δt, :ν, :save_interval]
            setfield!(modified, param_name, value)
        end
    end
    
    return modified
end

"""
    run_single_simulation(params; seed=42)

Run a single simulation and extract summary statistics.
This is a placeholder that should be connected to the actual simulation.
"""
function run_single_simulation(params; seed::Int=42)
    # This function should be overridden by integration_patch.jl
    # For now, return mock data for testing
    Random.seed!(seed)
    
    return (
        n_equilibria = rand(1:3),
        stability_index = rand(),
        consensus = rand(),
        polarization = rand(),
        lyapunov = randn() * 0.1,
        regime = rand(["Stable Consensus", "Bistable", "Oscillatory"]),
        oscillatory = rand() < 0.2,
        variance = rand(),
        convergence_time = rand() * params.T
    )
end

# ============================================================================
# Fixed Point Analysis
# ============================================================================

"""
    find_fixed_points(params; n_attempts=20)

Find fixed points of the dynamical system using multiple initial conditions.

# Arguments
- `params`: System parameters
- `n_attempts`: Number of random initial conditions to try

# Returns
- Vector of fixed point positions
"""
function find_fixed_points(params; n_attempts::Int=20)
    fixed_points = Vector{Float64}[]
    
    # Define residual function for the system
    function vector_field_residual(u)
        x, r = u
        du = zeros(2)
        du[1] = -params.cognitive.λ * (x - r)  # Simplified
        du[2] = (1.0 / (1.0 + 1.0)) * (x - r)  # φ(m) * (x - r)
        return norm(du)
    end
    
    # Try multiple initial conditions
    for _ in 1:n_attempts
        x0 = randn() * 2.0
        r0 = x0 + randn() * 0.5
        
        # Use optimization to find zeros
        result = optimize(vector_field_residual, [x0, r0], 
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
    check_stability(fixed_point::Vector{Float64}, params)

Check stability of a fixed point via eigenvalue analysis.

# Returns
- `true` if stable (all eigenvalues have negative real parts)
"""
function check_stability(fixed_point::Vector{Float64}, params)
    J = compute_jacobian(fixed_point, params)
    eigenvals = eigvals(J)
    max_real_part = maximum(real.(eigenvals))
    return max_real_part < 0
end

"""
    compute_jacobian(point, params)

Compute the Jacobian matrix at a given point.
"""
function compute_jacobian(point::Vector{Float64}, params)
    function vector_field(u)
        x, r = u
        du = zeros(2)
        du[1] = -params.cognitive.λ * (x - r)
        du[2] = (1.0 / (1.0 + 1.0)) * (x - r)
        return du
    end
    
    return ForwardDiff.jacobian(vector_field, point)
end

# ============================================================================
# Time Series Analysis
# ============================================================================

"""
    detect_period(trajectory; max_period=20, tolerance=1e-3)

Detect the period of oscillations in a trajectory.

# Arguments
- `trajectory`: Time series data
- `max_period`: Maximum period to check
- `tolerance`: Tolerance for periodicity detection

# Returns
- Period (1 if no periodicity detected)
"""
function detect_period(trajectory; max_period::Int=20, tolerance::Float64=1e-3)
    n = length(trajectory)
    if n < max_period * 2
        return 1
    end
    
    for period in 2:max_period
        is_periodic = true
        for i in 1:min(period, n - 2*period)
            if norm(trajectory[end-period+i] - trajectory[end-2*period+i]) > tolerance
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

"""
    moving_average(x::Vector{Float64}, window::Int)

Compute moving average of a time series.
"""
function moving_average(x::Vector{Float64}, window::Int)
    n = length(x)
    if n < window
        return [mean(x)]
    end
    
    ma = zeros(n - window + 1)
    for i in 1:length(ma)
        ma[i] = mean(x[i:i+window-1])
    end
    
    return ma
end

# ============================================================================
# Clustering and Classification
# ============================================================================

"""
    cluster_positions(values::Vector{Float64}, threshold::Float64)

Cluster nearby values to identify distinct groups.

# Returns
- NamedTuple with n_clusters and cluster centers
"""
function cluster_positions(values::Vector{Float64}, threshold::Float64)
    if isempty(values)
        return (n_clusters=0, centers=Float64[])
    end
    
    sorted_vals = sort(values)
    clusters = Vector{Float64}[]
    current_cluster = [sorted_vals[1]]
    
    for val in sorted_vals[2:end]
        if val - mean(current_cluster) < threshold
            push!(current_cluster, val)
        else
            push!(clusters, current_cluster)
            current_cluster = [val]
        end
    end
    push!(clusters, current_cluster)
    
    centers = [mean(cluster) for cluster in clusters]
    
    return (n_clusters=length(clusters), centers=centers)
end

"""
    simple_bin(x::Vector{T}, n_bins::Int=5) where T <: Real

Create simple bins for data without UUID complications.

# Arguments
- `x`: Data vector
- `n_bins`: Number of bins

# Returns
- Vector of bin labels
"""
function simple_bin(x::Vector{T}, n_bins::Int=5) where T <: Real
    if isempty(x)
        return String[]
    end
    
    min_x, max_x = extrema(x)
    if min_x ≈ max_x
        return fill("bin_1", length(x))
    end
    
    bins = String[]
    bin_width = (max_x - min_x) / n_bins
    
    for val in x
        bin_idx = min(n_bins, max(1, ceil(Int, (val - min_x) / bin_width)))
        push!(bins, "bin_$bin_idx")
    end
    
    return bins
end

# ============================================================================
# Fractal Analysis
# ============================================================================

"""
    estimate_fractal_dimension(boundaries::BitMatrix)

Estimate fractal dimension using box-counting method.

# Arguments
- `boundaries`: Binary matrix indicating boundary points

# Returns
- Estimated fractal dimension (NaN if calculation fails)
"""
function estimate_fractal_dimension(boundaries::BitMatrix)
    box_sizes = [2, 4, 8, 16, 32]
    counts = Float64[]
    
    nx, ny = size(boundaries)
    
    for box_size in box_sizes
        count = 0
        
        for i in 1:box_size:nx-box_size+1
            for j in 1:box_size:ny-box_size+1
                # Check if box contains boundary
                box = boundaries[i:min(i+box_size-1, nx), j:min(j+box_size-1, ny)]
                if any(box)
                    count += 1
                end
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
        coeffs = (A' * A) \ (A' * log_counts)
        
        fractal_dim = -coeffs[2]
        return fractal_dim
    else
        return NaN
    end
end

# ============================================================================
# Data Normalization
# ============================================================================

"""
    normalize_data(x::Vector{Float64}; method::Symbol=:minmax)

Normalize data using specified method.

# Arguments
- `x`: Data vector
- `method`: Normalization method (:minmax, :zscore, :unit)

# Returns
- Normalized data vector
"""
function normalize_data(x::Vector{Float64}; method::Symbol=:minmax)
    if isempty(x)
        return x
    end
    
    if method == :minmax
        min_x, max_x = extrema(x)
        if min_x ≈ max_x
            return fill(0.5, length(x))
        end
        return (x .- min_x) ./ (max_x - min_x)
        
    elseif method == :zscore
        μ = mean(x)
        σ = std(x)
        if σ < 1e-10
            return zeros(length(x))
        end
        return (x .- μ) ./ σ
        
    elseif method == :unit
        norm_x = norm(x)
        if norm_x < 1e-10
            return zeros(length(x))
        end
        return x ./ norm_x
        
    else
        error("Unknown normalization method: $method")
    end
end

end # module Utils
