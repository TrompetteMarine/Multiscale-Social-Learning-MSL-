#!/usr/bin/env julia
# setup.jl - BeliefSim Environment Setup Script
# Run this once to install all required packages

println("🌊 BeliefSim Environment Setup")
println("==============================\n")

using Pkg
using InteractiveUtils

# Check Julia version
println("📋 System Information:")
println("Julia Version: $(VERSION)")
println("OS: $(Sys.KERNEL)")
println("Architecture: $(Sys.MACHINE)")

if VERSION < v"1.6.0"
    @warn "BeliefSim is tested with Julia 1.6+. Your version is $(VERSION). Consider upgrading for best compatibility."
end

println("\n🔧 Setting up BeliefSim environment...")

# Activate the project environment
println("Activating project environment...")
Pkg.activate(@__DIR__)

# Define all required packages with version constraints
required_packages = [
    # Core simulation packages
    ("DifferentialEquations", "7.0"),
    ("Distributions", "0.25"), 
    ("Graphs", "1.6"),
    
    # Analysis packages
    ("CSV", "0.10"),
    ("DataFrames", "1.3"),
    ("StatsBase", "0.33"),
    ("Distances", "0.10"),
    
    # Visualization packages  
    ("Plots", "1.35"),
    ("GraphPlot", "0.5"),
    ("NetworkLayout", "0.4"),
    ("Colors", "0.12")
]

println("\n📦 Installing required packages...")
println("This may take several minutes on first run...")

# Track installation progress
total_packages = length(required_packages)
installed_count = 0

for (pkg_name, min_version) in required_packages
    try
        print("  Installing $pkg_name (>= $min_version)... ")
        Pkg.add(Pkg.PackageSpec(name=pkg_name, version=min_version))
        installed_count += 1
        println("✅")
    catch e
        println("❌")
        println("    Error installing $pkg_name: $e")
        println("    Trying without version constraint...")
        try
            Pkg.add(pkg_name)
            installed_count += 1
            println("    ✅ Installed without version constraint")
        catch e2
            println("    ❌ Failed completely: $e2")
        end
    end
end

println("\n📈 Installation Progress: $installed_count/$total_packages packages")

# Instantiate the environment
println("\n⚙️  Resolving dependencies and precompiling...")
try
    Pkg.instantiate()
    Pkg.precompile()
    println("✅ Environment setup complete!")
catch e
    println("⚠️  Warning during instantiation: $e")
    println("Continuing with testing...")
end

# Update Project.toml with all dependencies
println("\n📝 Updating Project.toml...")
project_content = """
name = "BeliefSim"
uuid = "12345678-1234-5678-9abc-123456789012"
version = "0.1.0"

[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GraphPlot = "a2cc645c-3eea-5389-862e-a155d0052231"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NetworkLayout = "46757867-2c16-5918-afeb-47bfcb05e46a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
julia = "1.6"
CSV = "0.10"
DataFrames = "1.3"
DifferentialEquations = "7.0"
Distributions = "0.25"
Graphs = "1.6"
Plots = "1.35"
StatsBase = "0.33"
"""

# Write the updated Project.toml
open("Project.toml", "w") do f
    write(f, project_content)
end
println("✅ Project.toml updated")

# Test basic functionality
println("\n🧪 Running installation tests...")

test_results = []

# Test 1: Core simulation packages
print("  Testing core simulation packages... ")
try
    using DifferentialEquations, Distributions, Graphs
    using LinearAlgebra, Statistics, Random, SparseArrays
    push!(test_results, ("Core packages", "✅"))
    println("✅")
catch e
    push!(test_results, ("Core packages", "❌"))
    println("❌")
    println("    Error: $e")
end

# Test 2: Analysis packages  
print("  Testing analysis packages... ")
try
    using CSV, DataFrames, StatsBase, Distances
    push!(test_results, ("Analysis packages", "✅"))
    println("✅")
catch e
    push!(test_results, ("Analysis packages", "❌"))
    println("❌")
    println("    Error: $e")
end

# Test 3: Visualization packages
print("  Testing visualization packages... ")
try
    using Plots, GraphPlot, NetworkLayout, Colors
    push!(test_results, ("Visualization packages", "✅"))
    println("✅")
catch e
    push!(test_results, ("Visualization packages", "❌"))
    println("❌")
    println("    Error: $e")
end

# Test 4: BeliefSim modules
print("  Testing BeliefSim modules... ")
try
    include("src/Kernel.jl")
    using .Kernel
    include("src/Metrics.jl") 
    using .Metrics
    include("src/Viz.jl")
    using .Viz
    push!(test_results, ("BeliefSim modules", "✅"))
    println("✅")
catch e
    push!(test_results, ("BeliefSim modules", "❌"))
    println("❌")
    println("    Error: $e")
end

# Test 5: Basic simulation
print("  Testing basic simulation... ")
try
    pars = SimPars(N=10, κ=1.0, β=0.5, σ=0.3, T=1.0, Δt=0.01)
    W = fully_connected(10)
    t_vec, traj = run_one_path(pars; W=W, seed=42)
    
    # Basic analysis
    final_consensus = consensus_metrics(traj[end])
    
    push!(test_results, ("Basic simulation", "✅"))
    println("✅")
    println("    Sample result: Final consensus = $(round(final_consensus[:consensus], digits=3))")
catch e
    push!(test_results, ("Basic simulation", "❌"))
    println("❌")
    println("    Error: $e")
end

# Test 6: Plotting capability
print("  Testing plotting capability... ")
try
    mkpath("output/setup_test")
    
    # Create a simple test plot
    using Plots
    test_x = 1:10
    test_y = rand(10)
    p = plot(test_x, test_y, title="Setup Test Plot")
    savefig(p, "output/setup_test/test_plot.png")
    
    push!(test_results, ("Plotting", "✅"))
    println("✅")
    println("    Test plot saved to output/setup_test/test_plot.png")
catch e
    push!(test_results, ("Plotting", "❌"))
    println("❌")
    println("    Error: $e")
end

# Display test summary
println("\n📊 Installation Test Summary:")
println("=============================")
all_passed = true
for (test_name, status) in test_results
    println("  $test_name: $status")
    if status == "❌"
        global all_passed = false
    end
end

# Create setup verification script
verification_script = """
# verification.jl - Quick verification that BeliefSim is working
using Pkg; Pkg.activate(@__DIR__)

include("src/Kernel.jl"); using .Kernel
include("src/Metrics.jl"); using .Metrics

println("🧪 BeliefSim Quick Verification")
pars = SimPars(N=20, κ=1.2, β=0.6, σ=0.25, T=5.0, Δt=0.01)
W = watts_strogatz_W(20; k=4, p=0.3)
t, traj = run_one_path(pars; W=W, seed=123)

println("✅ Simulation completed successfully!")
println("Final mean belief: \$(round(mean(traj[end]), digits=3))")
println("Final consensus: \$(round(consensus_metrics(traj[end])[:consensus], digits=3))")
println("\\nBeliefSim is ready to use! 🌊")
"""

write("verification.jl", verification_script)

# Final status report
println("\n" * "="^50)
if all_passed
    println("🎉 SETUP COMPLETE - ALL TESTS PASSED!")
    println("\n✅ BeliefSim is ready to use!")
    println("\n🚀 Quick Start:")
    println("  julia verification.jl           # Quick verification")
    println("  julia demo_example.jl          # Interactive demo")
    println("  julia bs.jl                     # Basic simulation")
    println("  julia scripts/advanced_analysis.jl  # Full analysis suite")
    
    println("\n📚 Documentation:")
    println("  README.md                       # Complete documentation")
    println("  src/                           # Source code modules")
    println("  scripts/                       # Analysis scripts")
    
    println("\n💡 Tip: Run 'julia demo_example.jl' for an interactive introduction!")
    
else
    println("⚠️  SETUP COMPLETED WITH ISSUES")
    println("\nSome components failed installation/testing.")
    println("BeliefSim may still be usable with reduced functionality.")
    println("\n🔧 Troubleshooting:")
    println("  1. Check your Julia version (1.6+ recommended)")
    println("  2. Update your package registry: Pkg.Registry.update()")
    println("  3. Try manual installation: Pkg.add([failed packages])")
    println("  4. Check internet connection for package downloads")
    
    println("\n📧 If problems persist:")
    println("  - Check the GitHub issues page")
    println("  - Verify your system has required dependencies")
    println("  - Try running 'julia verification.jl' to test basic functionality")
end

println("\n📁 Files created/updated:")
println("  Project.toml        # Package dependencies")
println("  Manifest.toml       # Exact package versions (auto-generated)")  
println("  verification.jl     # Quick verification script")
println("  output/setup_test/  # Test outputs")

println("\n🌊 Thank you for using BeliefSim!")
println("="^50)
