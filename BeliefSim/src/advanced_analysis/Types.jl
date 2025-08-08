# Types.jl - Shared type definitions for advanced analysis
"""
    Types

Shared type definitions for the advanced analysis modules.
Provides parameter structures and result types used across different analysis methods.
"""
module Types

export PhaseDiagramParams, BifurcationParams, BasinAnalysisParams, MonteCarloParams
export AnalysisResult, PhaseDiagramResult, BifurcationResult, BasinResult, MonteCarloResult
export SimulationSummary

# ============================================================================
# Parameter Structures
# ============================================================================

"""
    PhaseDiagramParams

Configuration for 2D phase diagram analysis.

# Fields
- `param1_name::Symbol`: Name of first parameter to vary
- `param1_range::Tuple{Float64, Float64}`: Range for first parameter
- `param1_steps::Int`: Number of steps for first parameter
- `param2_name::Symbol`: Name of second parameter to vary
- `param2_range::Tuple{Float64, Float64}`: Range for second parameter
- `param2_steps::Int`: Number of steps for second parameter
- `base_params::Any`: Base MSLParams object
- `analysis_time::Float64`: Simulation time for each point
- `transient_time::Float64`: Time to discard as transient
- `n_realizations::Int`: Number of realizations per parameter point
- `parallel::Bool`: Whether to use parallel computation
"""
Base.@kwdef struct PhaseDiagramParams
    param1_name::Symbol = :α
    param1_range::Tuple{Float64, Float64} = (0.1, 2.0)
    param1_steps::Int = 20
    
    param2_name::Symbol = :λ
    param2_range::Tuple{Float64, Float64} = (0.1, 2.0)
    param2_steps::Int = 20
    
    base_params::Any = nothing
    analysis_time::Float64 = 50.0
    transient_time::Float64 = 20.0
    n_realizations::Int = 3
    parallel::Bool = false
end

"""
    BifurcationParams

Configuration for bifurcation analysis with continuation methods.

# Fields
- `param_name::Symbol`: Parameter to vary
- `param_range::Tuple{Float64, Float64}`: Parameter range
- `n_points::Int`: Number of points for analysis
- `continuation_steps::Int`: Steps for numerical continuation
- `stability_check::Bool`: Whether to check stability
- `track_unstable::Bool`: Whether to track unstable branches
- `max_period::Int`: Maximum period for cycle detection
- `tolerance::Float64`: Numerical tolerance
"""
Base.@kwdef struct BifurcationParams
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

# Fields
- `x_range::Tuple{Float64, Float64}`: Range for belief dimension
- `y_range::Tuple{Float64, Float64}`: Range for reference point dimension
- `grid_resolution::Int`: Grid points per dimension
- `integration_time::Float64`: Time to integrate trajectories
- `convergence_threshold::Float64`: Threshold for convergence detection
- `max_attractors::Int`: Maximum number of attractors to track
"""
Base.@kwdef struct BasinAnalysisParams
    x_range::Tuple{Float64, Float64} = (-3.0, 3.0)
    y_range::Tuple{Float64, Float64} = (-3.0, 3.0)
    grid_resolution::Int = 100
    integration_time::Float64 = 100.0
    convergence_threshold::Float64 = 0.01
    max_attractors::Int = 10
end

"""
    MonteCarloParams

Configuration for Monte Carlo parameter exploration.

# Fields
- `n_samples::Int`: Number of parameter samples
- `param_ranges::Dict{Symbol, Tuple{Float64, Float64}}`: Parameter ranges
- `sampling_method::Symbol`: Sampling method (:latin_hypercube, :random, :sobol)
- `base_params::Any`: Base MSLParams object
- `save_trajectories::Bool`: Whether to save full trajectories
- `compute_sensitivity::Bool`: Whether to compute sensitivity indices
"""
Base.@kwdef struct MonteCarloParams
    n_samples::Int = 1000
    param_ranges::Dict{Symbol, Tuple{Float64, Float64}} = Dict()
    sampling_method::Symbol = :latin_hypercube
    base_params::Any = nothing
    save_trajectories::Bool = false
    compute_sensitivity::Bool = true
end

# ============================================================================
# Result Structures
# ============================================================================

"""
    SimulationSummary

Summary statistics from a single simulation run.
"""
struct SimulationSummary
    n_equilibria::Int
    stability_index::Float64
    consensus::Float64
    polarization::Float64
    lyapunov::Float64
    regime::String
    oscillatory::Bool
    variance::Float64
    convergence_time::Float64
end

"""
    PhaseDiagramResult

Results from phase diagram analysis.
"""
struct PhaseDiagramResult
    param1_values::Vector{Float64}
    param2_values::Vector{Float64}
    param1_name::Symbol
    param2_name::Symbol
    n_equilibria::Matrix{Int}
    stability::Matrix{Float64}
    consensus::Matrix{Float64}
    polarization::Matrix{Float64}
    regime_map::Matrix{String}
    lyapunov::Matrix{Float64}
end

"""
    BifurcationResult

Results from bifurcation analysis.
"""
struct BifurcationResult
    param_values::Vector{Float64}
    param_name::Symbol
    stable_branches::Dict{Int, Vector{Vector{Float64}}}
    unstable_branches::Dict{Int, Vector{Vector{Float64}}}
    bifurcation_points::Vector{Float64}
    bifurcation_types::Vector{Symbol}
    period_doublings::Vector{Float64}
    codim2_points::Vector{Float64}
end

"""
    BasinResult

Results from basin of attraction analysis.
"""
struct BasinResult
    x_grid::Vector{Float64}
    y_grid::Vector{Float64}
    basin_map::Matrix{Int}
    attractors::Vector{Vector{Float64}}
    attractor_types::Vector{Symbol}
    basin_sizes::Dict{Int, Float64}
    basin_boundaries::BitMatrix
    fractal_dimensions::Float64
    n_attractors::Int
end

"""
    MonteCarloResult

Results from Monte Carlo parameter exploration.
"""
struct MonteCarloResult
    data::Any  # DataFrame
    sensitivity::Dict{Symbol, Dict{Symbol, Float64}}
    critical_regions::Dict{Symbol, Any}
    param_importance::Vector{Pair{Symbol, Float64}}
    n_successful::Int
    n_failed::Int
end

end # module Types