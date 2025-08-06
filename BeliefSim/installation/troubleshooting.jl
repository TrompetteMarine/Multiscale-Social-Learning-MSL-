#!/usr/bin/env julia
# troubleshoot.jl - BeliefSim troubleshooting and diagnostic tool

println("🔧 BeliefSim Troubleshooting Tool")
println("=================================\n")

using Pkg, InteractiveUtils

# Activate the project environment
Pkg.activate(@__DIR__)

# System information
println("📋 System Information:")
println("-" * repeat("-", 30))
versioninfo()
println()

# Package status
println("📦 Package Status:")
println("-" * repeat("-", 30))
try
    Pkg.status()
    println("✅ Package environment loaded successfully")
except e
    println("❌ Package environment error: $e")
end
println()

# Test core packages
println("🧪 Testing Core Packages:")
println("-" * repeat("-", 30))

core_packages = [
    ("DifferentialEquations", "SDE solver"),
    ("Distributions", "Random distributions"),
    ("Graphs", "Network generation"),
    ("LinearAlgebra", "Matrix operations"),
    ("Statistics", "Statistical functions"),
    ("Random", "Random number generation")
]

failed_core = []
for (pkg, description) in core_packages
    print("  $pkg... ")
    try
        eval(Meta.parse("using $pkg"))
        println("✅")
    catch e
        println("❌")
        println("    Error: $e")
        push!(failed_core, pkg)
    end
end

if !isempty(failed_core)
    println("\n⚠️  Critical packages failed: $(join(failed_core, ", "))")
    println("   Try: julia -e 'using Pkg; Pkg.add([\"$(join(failed_core, "\", \""))\"]))'")
end

# Test optional packages
println("\n🎨 Testing Optional Packages:")
println("-" * repeat("-", 30))

optional_packages = [
    ("CSV", "Data export"),
    ("DataFrames", "Data manipulation"),
    ("Plots", "Basic plotting"),
    ("StatsBase", "Extended statistics"),
    ("GraphPlot", "Network visualization"),
    ("NetworkLayout", "Graph layouts"),
    ("Colors", "Color management")
]

failed_optional = []
for (pkg, description) in optional_packages
    print("  $pkg... ")
    try
        eval(Meta.parse("using $pkg"))
        println("✅")
    catch e
        println("⚠️  (optional)")
        push!(failed_optional, pkg)
    end
end

if !isempty(failed_optional)
    println("\n📝 Optional packages not available: $(join(failed_optional, ", "))")
    println("   Some features may be limited.")
    println("   Install with: julia -e 'using Pkg; Pkg.add([\"$(join(failed_optional, "\", \""))\"]))'")
end

# Test BeliefSim modules
println("\n🌊 Testing BeliefSim Modules:")
println("-" * repeat("-", 30))

modules = [
    ("src/Kernel.jl", "Core simulation engine"),
    ("src/Metrics.jl", "Analysis tools"),
    ("src/Viz.jl", "Visualization functions")
]

module_errors = []
for (file, description) in modules
    print("  $(basename(file))... ")
    try
        include(file)
        println("✅")
    catch e
        println("❌")
        println("    Error: $e")
        push!(module_errors, file)
    end
end

if isempty(module_errors)
    println("\n✅ All BeliefSim modules loaded successfully")
    
    # Test basic functionality
    println("\n🧪 Testing Basic Functionality:")
    println("-" * repeat("-", 30))
    
    try
        using .Kernel, .Metrics
        
        print("  Creating simulation parameters... ")
        pars = SimPars(N=5, κ=1.0, β=0.5, σ=0.3, T=1.0, Δt=0.01)
        println("✅")
        
        print("  Generating network... ")
        W = fully_connected(5)
        println("✅")
        
        print("  Running simulation... ")
        t_vec, traj = run_one_path(pars; W=W, seed=42)
        println("✅")
        
        print("  Computing metrics... ")
        final_metrics = consensus_metrics(traj[end])
        println("✅")
        
        println("\n📊 Sample Results:")
        println("  Simulation steps: $(length(t_vec))")
        println("  Final mean belief: $(round(mean(traj[end]), digits=3))")
        println("  Final consensus: $(round(final_metrics[:consensus], digits=3))")
        
    catch e
        println("❌ Basic functionality test failed: $e")
    end
    
else
    println("\n❌ BeliefSim module errors detected")
    println("   Check that all source files are present in src/")
end

# Test plotting capabilities  
println("\n🎨 Testing Plotting:")
println("-" * repeat("-", 30))

try
    using Plots
    ENV["GKSwstype"] = "100"  # Use non-interactive backend
    
    print("  Basic plotting... ")
    p = plot([1, 2, 3], [1, 4, 2], title="Test Plot")
    println("✅")
    
    print("  Saving plot... ")
    mkpath("output/troubleshoot")
    savefig(p, "output/troubleshoot/test_plot.png")
    println("✅")
    println("    → output/troubleshoot/test_plot.png")
    
catch e
    println("❌ Plotting test failed: $e")
    println("    Try: julia -e 'using Pkg; Pkg.add(\"Plots\")' ")
end

# Diagnostic recommendations
println("\n💡 Recommendations:")
println("-" * repeat("-", 30))

if isempty(failed_core) && isempty(module_errors)
    println("✅ BeliefSim appears to be working correctly!")
    println("\nSuggested next steps:")
    println("  • julia simple_demo.jl      # Try the streamlined demo")
    println("  • julia demo_example.jl     # Full interactive demo")
    println("  • julia bs.jl               # Basic simulation")
    
else
    println("⚠️  Issues detected. Try these fixes:")
    
    if !isempty(failed_core)
        println("  1. Install missing core packages:")
        println("     julia -e 'using Pkg; Pkg.add([\"$(join(failed_core, "\", \""))\"]))'")
    end
    
    if !isempty(module_errors)
        println("  2. Check BeliefSim source files:")
        println("     • Ensure src/Kernel.jl, src/Metrics.jl, src/Viz.jl exist")
        println("     • Check file permissions")
    end
    
    println("  3. Update package registry:")
    println("     julia -e 'using Pkg; Pkg.Registry.update()'")
    
    println("  4. Reinstall environment:")
    println("     julia install.jl")
    
    println("  5. Try minimal version:")
    println("     julia simple_demo.jl")
end

# Environment status summary
println("\n📈 Environment Summary:")
println("-" * repeat("-", 30))
println("Julia Version: $(VERSION)")
println("Core packages: $(length(core_packages) - length(failed_core))/$(length(core_packages)) working")
println("Optional packages: $(length(optional_packages) - length(failed_optional))/$(length(optional_packages)) working") 
println("BeliefSim modules: $(length(modules) - length(module_errors))/$(length(modules)) working")

if isempty(failed_core) && isempty(module_errors)
    println("\n🎉 Status: Ready for BeliefSim simulations!")
else
    println("\n⚠️  Status: Some issues detected - see recommendations above")
end

println("\n🔧 Troubleshooting complete.")
