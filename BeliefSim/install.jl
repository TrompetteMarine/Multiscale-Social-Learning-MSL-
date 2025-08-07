#!/usr/bin/env julia
# install.jl - Simple installation script for BeliefSim

println("🌊 BeliefSim Installation")
println("========================")
println()

using Pkg

# Activate project environment
Pkg.activate(@__DIR__)

println("📦 Installing dependencies...")
Pkg.instantiate()

println("⚙️  Precompiling packages...")
Pkg.precompile()

# Test basic functionality
println("\n🧪 Testing installation...")

try
    # Test package loading
    using DifferentialEquations
    using Plots
    using Graphs
    using Statistics
    
    println("✅ Core packages loaded successfully")
    
    # Test BeliefSim modules
    include("src/BeliefSim.jl")
    using .BeliefSim
    
    println("✅ BeliefSim modules loaded successfully")
    
    # Run minimal test
    params = BeliefSim.MSLParams(N=10, T=5.0)
    println("✅ Parameter creation successful")
    
    println("\n🎉 Installation complete!")
    println("\nTo get started:")
    println("  julia --project=. examples/basic_simulation.jl")
    println("\nTo reproduce paper results:")
    println("  julia --project=. examples/paper_reproduction.jl")
    
catch e
    println("\n❌ Installation failed:")
    println(e)
    println("\nTry:")
    println("  1. Update Julia to version 1.6 or later")
    println("  2. Run: julia --project=. -e 'using Pkg; Pkg.update()'")
    println("  3. Check internet connection for package downloads")
end