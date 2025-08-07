# BeliefSim.jl - Main module
module BeliefSim

using Reexport

include("agents.jl")
include("networks.jl")
include("simulation.jl")
include("metrics.jl")
include("visualization.jl")

@reexport using .Agents
@reexport using .Networks
@reexport using .Simulation
@reexport using .Metrics
@reexport using .Visualization

export run_msl_simulation, MSLParams, AgentState
export create_network, analyze_results, plot_results

"""
    run_msl_simulation(params::MSLParams; seed=42)

Run a complete Multi-Scale Social Learning simulation following Bontemps (2024).

Returns: (t_vec, trajectories, analysis)
"""
function run_msl_simulation(params::MSLParams; seed=42, W=nothing)
    # Create network if not provided
    if W === nothing
        W = create_network(params.N, params.network_type, params.network_params)
    end
    
    # Run simulation
    t_vec, trajectories = simulate_msl(params, W, seed)
    
    # Analyze results
    analysis = analyze_trajectories(trajectories, W, params)
    
    return t_vec, trajectories, analysis
end

end # module
