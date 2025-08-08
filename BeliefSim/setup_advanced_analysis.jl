#!/usr/bin/env julia
# setup_advanced_analysis.jl - Setup script for advanced BeliefSim analysis

println("🚀 Setting up Advanced Analysis for BeliefSim")
println("=" ^ 50)

using Pkg

# Check Julia version
if VERSION < v"1.6"
    error("Julia 1.6 or higher required. Current version: $VERSION")
end

println("\n📦 Installing additional dependencies...")

# Activate project
Pkg.activate(@__DIR__)

# Add required packages for advanced analysis
additional_packages = [
    "Optim",
    "ForwardDiff", 
    "NearestNeighbors",
    "Clustering",
    "DataFrames",
    "CSV",
    "ColorSchemes",
    "Contour",
    "ProgressMeter"
]

for pkg in additional_packages
    try
        Pkg.add(pkg)
        println("  ✓ $pkg")
    catch e
        println("  ⚠ $pkg already installed or error: $e")
    end
end

println("\n📁 Creating directory structure...")

# Create necessary directories
dirs = [
    "src",
    "output",
    "output/advanced_analysis",
    "output/phase_diagrams",
    "output/bifurcations",
    "output/basins",
    "examples/advanced"
]

for dir in dirs
    mkpath(dir)
    println("  ✓ $dir")
end

println("\n📝 Installing analysis modules...")

# Check if files need to be moved
src_files = [
    ("advanced_analysis.jl", "src/advanced_analysis.jl"),
    ("integration_patch.jl", "src/integration_patch.jl"),
    ("advanced_analysis_example.jl", "examples/advanced/run_analysis.jl")
]

println("\n⚙️  Testing installation...")

# Test basic import
try
    # Test that BeliefSim loads
    include("src/BeliefSim.jl")
    using .BeliefSim
    println("  ✓ BeliefSim loaded successfully")
    
    # Test that advanced analysis loads
    include("src/advanced_analysis.jl")
    using .AdvancedAnalysis
    println("  ✓ AdvancedAnalysis loaded successfully")
    
    # Test integration patch
    include("src/integration_patch.jl")
    using .IntegrationPatch
    println("  ✓ IntegrationPatch loaded successfully")
    
    # Quick functionality test
    test_params = BeliefSim.MSLParams(N=10, T=5.0)
    println("  ✓ Parameter creation successful")
    
    println("\n✅ Installation complete!")
    
catch e
    println("\n❌ Installation test failed:")
    println(e)
    println("\nPlease check that all BeliefSim modules are in place.")
end

println("=" ^ 50)
println("📚 USAGE INSTRUCTIONS")
println("=" ^ 50)

println("""

To use the advanced analysis features:

1. BASIC USAGE:
   ```julia
   include("src/BeliefSim.jl")
   include("src/advanced_analysis.jl")
   include("src/integration_patch.jl")
   
   using .BeliefSim, .AdvancedAnalysis, .IntegrationPatch
   
   # Integrate the modules
   integrate_advanced_analysis(base_params)
   ```

2. RUN PHASE DIAGRAM ANALYSIS:
   ```julia
   phase_params = PhaseDiagramParams(
       param1_name = :α,
       param1_range = (0.1, 2.0),
       param2_name = :λ,
       param2_range = (0.1, 2.0),
       base_params = your_params
   )
   
   phase_data = run_phase_diagram(phase_params)
   plot_phase_diagram(phase_data)
   ```

3. RUN BIFURCATION ANALYSIS:
   ```julia
   bifurc_params = BifurcationParams(
       param_name = :α,
       param_range = (0.0, 2.5),
       n_points = 100
   )
   
   bifurc_data = run_bifurcation_analysis(bifurc_params, base_params)
   plot_bifurcation_2d(bifurc_data)
   ```

4. ANALYZE BASINS OF ATTRACTION:
   ```julia
   basin_params = BasinAnalysisParams(
       x_range = (-3.0, 3.0),
       y_range = (-3.0, 3.0),
       grid_resolution = 100
   )
   
   basin_data = analyze_basins_of_attraction(basin_params, your_params)
   plot_basin_portrait(basin_data)
   ```

5. RUN MONTE CARLO EXPLORATION:
   ```julia
   param_ranges = Dict(
       :α => (0.1, 2.0),
       :λ => (0.5, 1.5),
       :σ => (0.1, 0.5)
   )
   
   mc_results = monte_carlo_phase_exploration(500, param_ranges, base_params)
   ```

6. RUN COMPLETE ANALYSIS:
   ```julia
   julia examples/advanced/run_analysis.jl
   ```

For more examples, see the examples/advanced/ directory.
""")

println("\n🎯 Next steps:")
println("  1. Run: julia examples/advanced/run_analysis.jl")
println("  2. Check output/advanced_analysis/ for results")
println("  3. Modify parameters in the example script as needed")
