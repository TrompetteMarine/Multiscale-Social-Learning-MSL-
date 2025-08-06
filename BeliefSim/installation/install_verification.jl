#!/usr/bin/env julia
# install_verification.jl - Verify BeliefSim installation is working

println("🔍 BeliefSim Installation Verification")
println("=====================================\n")

using Pkg
Pkg.activate(@__DIR__)

# Test package loading
println("📦 Testing package imports...")

packages_to_test = [
    ("DifferentialEquations", "Core SDE solver"),
    ("Distributions", "Random distributions"), 
    ("Graphs", "Network generation"),
    ("CSV", "Data export"),
    ("DataFrames", "Data manipulation"),
    ("Plots", "Visualization"),
    ("StatsBase", "Statistical functions")
]

failed_packages = String[]

for (pkg, description) in packages_to_test
    print("  $pkg ($description)... ")
    try
        eval(Meta.parse("using $pkg"))
        println("✅")
    catch e
        println("❌")
        push!(failed_packages, pkg)
        println("    Error: $e")
    end
end

if !isempty(failed_packages)
    println("\n⚠️  Failed to load packages: $(join(failed_packages, ", "))")
    println("Run 'julia setup.jl' to reinstall missing packages.")
    exit(1)
end

# Test BeliefSim modules
println("\n🌊 Testing BeliefSim modules...")

modules_to_test = [
    ("src/Kernel.jl", "Core simulation engine"),
    ("src/Metrics.jl", "Analysis tools"),
    ("src/Viz.jl", "Visualization functions")
]

for (file, description) in modules_to_test
    print("  $(basename(file)) ($description)... ")
    try
        include(file)
        println("✅")
    catch e
        println("❌")
        println("    Error: $e")
        exit(1)
    end
end

# Load the modules
using .Kernel, .Metrics, .Viz

# Test basic functionality
println("\n🧪 Testing basic simulation...")

try
    # Create simple simulation
    pars = SimPars(N=20, κ=1.0, β=0.5, σ=0.3, T=2.0, Δt=0.01)
    W = fully_connected(20)
    
    print("  Running simulation... ")
    t_vec, trajectory = run_one_path(pars; W=W, seed=42)
    println("✅")
    
    print("  Analyzing results... ")
    final_beliefs = trajectory[end]
    consensus_data = consensus_metrics(final_beliefs)
    println("✅")
    
    print("  Network generation... ")
    W_sw = watts_strogatz_W(20; k=4, p=0.2)
    println("✅")
    
    println("\n📊 Sample Results:")
    println("  Final mean belief: $(round(mean(final_beliefs), digits=3))")
    println("  Consensus level: $(round(consensus_data[:consensus], digits=3))")
    println("  Belief spread (std): $(round(std(final_beliefs), digits=3))")
    
    # Test visualization (basic)
    print("\n🎨 Testing visualization... ")
    mkpath("output/verification")
    
    # Create a simple plot without displaying it
    ENV["GKSwstype"] = "100"  # Use non-interactive backend
    p = plot([1,2,3], [1,4,2], title="Verification Test")
    savefig(p, "output/verification/test_plot.png")
    println("✅")
    println("  Test plot saved to output/verification/test_plot.png")
    
catch e
    println("❌")
    println("Simulation test failed: $e")
    exit(1)
end

# All tests passed
println("\n🎉 ALL TESTS PASSED!")
println("\n✅ BeliefSim is properly installed and working!")

println("\n🚀 You can now run:")
println("  julia demo_example.jl           # Interactive demo")
println("  julia bs.jl                      # Basic simulation") 
println("  julia scripts/montecarlo_shift.jl   # Monte Carlo analysis")
println("  julia scripts/advanced_analysis.jl  # Full analysis suite")

println("\n📚 Check README.md for detailed documentation.")
println("🌊 Happy simulating with BeliefSim!")
