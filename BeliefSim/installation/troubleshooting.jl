#!/usr/bin/env julia
# troubleshooting.jl - Advanced BeliefSim Diagnostic and Repair Tool

using Pkg, InteractiveUtils, Dates

println("🔧 BeliefSim Advanced Troubleshooting Tool")
println("=========================================")
println("Comprehensive diagnostics for Multi-Scale Social Learning Simulator")
println()

# ============================================================================
# System Diagnostics
# ============================================================================

function system_diagnostics()
    println("📊 System Diagnostics:")
    println("-" ^ 30)
    
    # Julia version
    julia_version = VERSION
    println("Julia Version: $julia_version")
    
    if julia_version < v"1.6.0"
        println("  ⚠️  WARNING: Julia $julia_version detected. BeliefSim requires 1.6+")
        println("     Consider upgrading Julia for optimal compatibility.")
    else
        println("  ✅ Julia version compatible")
    end
    
    # System info
    println("OS: $(Sys.KERNEL)")
    println("Architecture: $(Sys.MACHINE)")  
    println("CPU Cores: $(Sys.CPU_THREADS)")
    println("Total Memory: $(round(Sys.total_memory() / 2^30, digits=2)) GB")
    
    # Package environment
    println("\nPackage Environment:")
    try
        Pkg.activate(@__DIR__)
        println("  Project: $(pwd())")
        
        # Check if Project.toml exists
        if isfile("Project.toml")
            println("  ✅ Project.toml found")
        else
            println("  ❌ Project.toml missing")
        end
        
        # Check Manifest.toml
        if isfile("Manifest.toml")
            println("  ✅ Manifest.toml found")
        else
            println("  ⚠️  Manifest.toml missing (will be created on first instantiate)")
        end
        
    catch e
        println("  ❌ Package environment error: $e")
    end
    
    println()
end

# ============================================================================
# Package Diagnostics
# ============================================================================

function package_diagnostics()
    println("📦 Package Diagnostics:")
    println("-" ^ 30)
    
    essential_packages = [
        "DifferentialEquations" => "SDE solver engine",
        "Distributions" => "Random distributions", 
        "Graphs" => "Network generation",
        "CSV" => "Data export",
        "DataFrames" => "Data manipulation",
        "Plots" => "Visualization",
        "StatsBase" => "Statistical functions"
    ]
    
    optional_packages = [
        "GraphPlot" => "Network visualization",
        "NetworkLayout" => "Graph layouts",
        "Colors" => "Color management",
        "Distances" => "Distance metrics"
    ]
    
    # Test essential packages
    essential_failed = []
    println("Essential packages:")
    for (pkg, desc) in essential_packages
        print("  $pkg... ")
        try
            eval(Meta.parse("using $pkg"))
            println("✅")
        catch e
            println("❌")
            push!(essential_failed, pkg)
            println("    Error: $e")
        end
    end
    
    # Test optional packages  
    optional_failed = []
    println("\nOptional packages:")
    for (pkg, desc) in optional_packages
        print("  $pkg... ")
        try
            eval(Meta.parse("using $pkg"))
            println("✅")
        catch e
            println("⚠️  (optional)")
            push!(optional_failed, pkg)
        end
    end
    
    return essential_failed, optional_failed
end

# ============================================================================
# BeliefSim Module Diagnostics
# ============================================================================

function module_diagnostics()
    println("🌊 BeliefSim Module Diagnostics:")
    println("-" ^ 30)
    
    modules = [
        ("src/Kernel.jl", "Core simulation engine"),
        ("src/Metrics.jl", "Analysis and metrics"),
        ("src/Viz.jl", "Visualization functions")
    ]
    
    module_issues = []
    
    for (file, desc) in modules
        print("  $(basename(file))... ")
        
        if !isfile(file)
            println("❌ File missing")
            push!(module_issues, "$file missing")
            continue
        end
        
        try
            include(file)
            println("✅")
        catch e
            println("❌")
            println("    Error: $e")
            push!(module_issues, "$file: $e")
        end
    end
    
    # Test module integration
    if isempty(module_issues)
        print("  Module integration... ")
        try
            using .Kernel, .Metrics, .Viz
            println("✅")
        catch e
            println("❌")
            println("    Integration error: $e")
            push!(module_issues, "Integration: $e")
        end
    end
    
    return module_issues
end

# ============================================================================
# Simulation Diagnostics
# ============================================================================

function simulation_diagnostics()
    println("🧪 Simulation Diagnostics:")
    println("-" ^ 30)
    
    simulation_issues = []
    
    try
        # Test parameter creation
        print("  Parameter creation... ")
        params = MSLSimPars(N=5, T=1.0, Δt=0.01)
        println("✅")
        
        # Test network generation
        print("  Network generation... ")
        W = create_network(5, NetworkParams())
        if size(W) == (5, 5) && all(W .>= 0)
            println("✅")
        else
            println("❌ Invalid network matrix")
            push!(simulation_issues, "Network generation produces invalid matrix")
        end
        
        # Test basic simulation
        print("  Basic MSL simulation... ")
        t_vec, trajectories = run_msl_simulation(params; seed=42)
        
        if haskey(trajectories, :beliefs) && length(trajectories[:beliefs]) == 5
            println("✅")
        else
            println("❌ Invalid trajectory structure")
            push!(simulation_issues, "Simulation produces invalid trajectories")
        end
        
        # Test analysis
        print("  Multi-scale analysis... ")
        analysis = multi_scale_analysis(trajectories, W, t_vec)
        
        if haskey(analysis, :regime) && haskey(analysis, :stability)
            println("✅")
        else
            println("❌ Invalid analysis results")
            push!(simulation_issues, "Analysis produces invalid results")
        end
        
    catch e
        println("❌ Simulation error: $e")
        push!(simulation_issues, "Simulation failed: $e")
    end
    
    return simulation_issues
end

# ============================================================================
# Performance Diagnostics
# ============================================================================

function performance_diagnostics()
    println("⚡ Performance Diagnostics:")
    println("-" ^ 30)
    
    try
        # Test small simulation speed
        println("  Running performance test...")
        params = MSLSimPars(N=20, T=2.0, Δt=0.01)
        
        start_time = time()
        t_vec, trajectories = run_msl_simulation(params; seed=42)
        end_time = time()
        
        elapsed = end_time - start_time
        println("  Small simulation (20 agents, 2 time units): $(round(elapsed, digits=2))s")
        
        if elapsed < 10.0
            println("  ✅ Performance: Good")
        elseif elapsed < 30.0
            println("  ⚠️  Performance: Acceptable but slow")
        else
            println("  ❌ Performance: Very slow, check system resources")
        end
        
        # Memory usage estimate
        Base.gc()  # Force garbage collection
        mem_before = Base.gc_live_bytes()
        
        # Run larger simulation
        params_large = MSLSimPars(N=50, T=5.0, Δt=0.01)
        t_vec, trajectories = run_msl_simulation(params_large; seed=42)
        
        Base.gc()
        mem_after = Base.gc_live_bytes()
        mem_used = (mem_after - mem_before) / 2^20  # Convert to MB
        
        println("  Memory usage (50 agents): $(round(mem_used, digits=1)) MB")
        
    catch e
        println("  ❌ Performance test failed: $e")
    end
    
    println()
end

# ============================================================================
# File System Diagnostics
# ============================================================================

function filesystem_diagnostics()
    println("📁 File System Diagnostics:")
    println("-" ^ 30)
    
    # Check directory structure
    required_dirs = ["src", "output"]
    missing_dirs = []
    
    for dir in required_dirs
        if isdir(dir)
            println("  ✅ $dir/ directory exists")
        else
            println("  ❌ $dir/ directory missing")
            push!(missing_dirs, dir)
        end
    end
    
    # Check write permissions
    print("  Write permissions... ")
    try
        test_dir = "output/diagnostic_test"
        mkpath(test_dir)
        test_file = joinpath(test_dir, "test.txt")
        write(test_file, "test")
        rm(test_file)
        rm(test_dir)
        println("✅")
    catch e
        println("❌")
        println("    Cannot write to output directory: $e")
    end
    
    # Check essential files
    essential_files = [
        "Project.toml",
        "src/Kernel.jl",
        "src/Metrics.jl", 
        "src/Viz.jl"
    ]
    
    missing_files = []
    for file in essential_files
        if isfile(file)
            println("  ✅ $file")
        else
            println("  ❌ $file missing")
            push!(missing_files, file)
        end
    end
    
    return missing_dirs, missing_files
end

# ============================================================================
# Automated Repairs
# ============================================================================

function attempt_repairs(essential_failed, missing_dirs, missing_files)
    println("🔨 Automated Repairs:")
    println("-" ^ 30)
    
    repairs_made = []
    
    # Create missing directories
    for dir in missing_dirs
        try
            mkpath(dir)
            println("  ✅ Created directory: $dir/")
            push!(repairs_made, "Created $dir/ directory")
        catch e
            println("  ❌ Failed to create $dir/: $e")
        end
    end
    
    # Try to install missing essential packages
    if !isempty(essential_failed)
        println("  Installing missing packages...")
        for pkg in essential_failed
            print("    Installing $pkg... ")
            try
                Pkg.add(pkg)
                println("✅")
                push!(repairs_made, "Installed $pkg")
            catch e
                println("❌")
                println("      Error: $e")
            end
        end
    end
    
    # Update package registry if packages failed
    if !isempty(essential_failed)
        print("  Updating package registry... ")
        try
            Pkg.Registry.update()
            println("✅")
            push!(repairs_made, "Updated package registry")
        catch e
            println("❌: $e")
        end
    end
    
    # Instantiate environment
    print("  Instantiating environment... ")
    try
        Pkg.instantiate()
        println("✅")
        push!(repairs_made, "Instantiated package environment")
    catch e
        println("❌: $e")
    end
    
    return repairs_made
end

# ============================================================================
# Generate Diagnostic Report
# ============================================================================

function generate_report(diagnostics_data)
    println("📄 Generating Diagnostic Report...")
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    report_file = "DIAGNOSTIC_REPORT_$timestamp.md"
    
    report_content = """
# BeliefSim Diagnostic Report

Generated: $(now())
System: $(Sys.KERNEL) $(Sys.MACHINE)
Julia: $(VERSION)

## Summary

$(diagnostics_data[:summary])

## Issues Found

### Critical Issues
$(isempty(diagnostics_data[:critical]) ? "None" : join(diagnostics_data[:critical], "\n"))

### Minor Issues  
$(isempty(diagnostics_data[:minor]) ? "None" : join(diagnostics_data[:minor], "\n"))

## Repairs Attempted

$(isempty(diagnostics_data[:repairs]) ? "No repairs attempted" : join(diagnostics_data[:repairs], "\n"))

## Recommendations

$(diagnostics_data[:recommendations])

## Next Steps

1. Address critical issues if any
2. Run installation verification: `julia verify_install.jl`
3. Try basic simulation: `julia basic_example.jl`
4. If problems persist, consider:
   - Updating Julia to latest version
   - Reinstalling packages: `julia install.jl`
   - Checking system dependencies

---
Generated by BeliefSim Troubleshooting Tool
"""
    
    write(report_file, report_content)
    println("  ✅ Report saved: $report_file")
    
    return report_file
end

# ============================================================================
# Main Diagnostic Flow
# ============================================================================

function main_diagnostics()
    println("🔍 Running comprehensive diagnostics...\n")
    
    # Collect diagnostic data
    diagnostics_data = Dict(
        :critical => [],
        :minor => [],
        :repairs => [],
        :summary => "",
        :recommendations => ""
    )
    
    # 1. System diagnostics
    system_diagnostics()
    
    # 2. Package diagnostics
    essential_failed, optional_failed = package_diagnostics()
    if !isempty(essential_failed)
        push!(diagnostics_data[:critical], "Missing essential packages: $(join(essential_failed, ", "))")
    end
    if !isempty(optional_failed)
        push!(diagnostics_data[:minor], "Missing optional packages: $(join(optional_failed, ", "))")
    end
    println()
    
    # 3. Module diagnostics
    module_issues = module_diagnostics()
    if !isempty(module_issues)
        for issue in module_issues
            push!(diagnostics_data[:critical], "Module issue: $issue")
        end
    end
    println()
    
    # 4. Simulation diagnostics  
    simulation_issues = simulation_diagnostics()
    if !isempty(simulation_issues)
        for issue in simulation_issues
            push!(diagnostics_data[:critical], "Simulation issue: $issue")
        end
    end
    println()
    
    # 5. Performance diagnostics
    performance_diagnostics()
    
    # 6. Filesystem diagnostics
    missing_dirs, missing_files = filesystem_diagnostics()
    if !isempty(missing_dirs)
        push!(diagnostics_data[:minor], "Missing directories: $(join(missing_dirs, ", "))")
    end
    if !isempty(missing_files)
        push!(diagnostics_data[:critical], "Missing files: $(join(missing_files, ", "))")
    end
    println()
    
    # 7. Attempt automated repairs
    repairs = attempt_repairs(essential_failed, missing_dirs, missing_files)
    diagnostics_data[:repairs] = repairs
    println()
    
    # 8. Generate summary and recommendations
    critical_count = length(diagnostics_data[:critical])
    minor_count = length(diagnostics_data[:minor])
    
    if critical_count == 0 && minor_count == 0
        diagnostics_data[:summary] = "✅ All diagnostics passed! BeliefSim is ready to use."
        diagnostics_data[:recommendations] = """
BeliefSim appears to be working correctly. You can:
- Run `julia run.jl` for interactive demos
- Try `julia paper_demo.jl` for core paper results
- Explore `julia bifurcation_analysis.jl` for advanced analysis
"""
    elseif critical_count == 0
        diagnostics_data[:summary] = "⚠️  Minor issues detected but system should work."
        diagnostics_data[:recommendations] = """
BeliefSim should work with current configuration. Minor issues may affect some features:
- Install optional packages for full visualization: `julia -e 'using Pkg; Pkg.add(["GraphPlot", "NetworkLayout"])'`
- Try basic simulation: `julia basic_example.jl`
"""
    else
        diagnostics_data[:summary] = "❌ Critical issues detected. BeliefSim may not work correctly."
        diagnostics_data[:recommendations] = """
Critical issues need attention:
- Run full installation: `julia install.jl`
- Check Julia version (1.6+ required)
- Ensure network connection for package downloads
- Check system permissions for file operations
"""
    end
    
    # 9. Display final summary
    println("="^60)
    println("🏆 DIAGNOSTIC SUMMARY")
    println("="^60)
    println(diagnostics_data[:summary])
    println()
    println("Issues Found:")
    println("  Critical: $critical_count")
    println("  Minor: $minor_count")
    println("  Repairs attempted: $(length(repairs))")
    println()
    println("💡 Recommendations:")
    println(diagnostics_data[:recommendations])
    
    # 10. Generate report
    report_file = generate_report(diagnostics_data)
    println("\n📋 Detailed report saved: $report_file")
    
    # 11. Quick test if no critical issues
    if critical_count == 0
        println("\n🧪 Quick Functionality Test:")
        print("  Running basic simulation... ")
        try
            params = MSLSimPars(N=5, T=0.5, Δt=0.01)
            t_vec, trajectories = run_msl_simulation(params; seed=42)
            println("✅")
            
            println("\n🎉 BeliefSim is operational!")
            println("   Try: julia run.jl")
            
        catch e
            println("❌")
            println("   Error: $e")
            println("\n⚠️  BeliefSim may have runtime issues despite passing diagnostics.")
        end
    end
    
    println("\n🔧 Troubleshooting complete!")
    return critical_count == 0
end

# ============================================================================
# Interactive Repair Menu
# ============================================================================

function interactive_repair_menu()
    println("\n🛠️  Interactive Repair Options:")
    println("1. Reinstall all packages")
    println("2. Update package registry") 
    println("3. Clear package cache")
    println("4. Reset environment")
    println("5. Create missing directories")
    println("6. Exit")
    print("\nSelect repair option (1-6): ")
    
    choice = readline()
    
    if choice == "1"
        println("🔄 Reinstalling packages...")
        try
            include("install.jl")
            println("✅ Reinstallation complete")
        catch e
            println("❌ Reinstallation failed: $e")
        end
        
    elseif choice == "2"
        println("🔄 Updating package registry...")
        try
            Pkg.Registry.update()
            println("✅ Registry updated")
        catch e
            println("❌ Registry update failed: $e")
        end
        
    elseif choice == "3"
        println("🔄 Clearing package cache...")
        try
            Pkg.gc()
            println("✅ Package cache cleared")
        catch e
            println("❌ Cache clearing failed: $e")
        end
        
    elseif choice == "4"
        println("🔄 Resetting environment...")
        try
            if isfile("Manifest.toml")
                rm("Manifest.toml")
                println("  Removed Manifest.toml")
            end
            Pkg.instantiate()
            println("✅ Environment reset complete")
        catch e
            println("❌ Environment reset failed: $e")
        end
        
    elseif choice == "5"
        println("📁 Creating missing directories...")
        dirs = ["src", "output", "output/basic", "output/verification"]
        for dir in dirs
            mkpath(dir)
            println("  Created: $dir/")
        end
        println("✅ Directories created")
        
    elseif choice == "6"
        return false
        
    else
        println("❌ Invalid choice")
    end
    
    return true
end

# ============================================================================
# Entry Point
# ============================================================================

# Run diagnostics if script executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    success = main_diagnostics()
    
    if !success
        println("\n🔧 Would you like to try interactive repairs?")
        print("Enter 'y' for yes, any other key to exit: ")
        
        if lowercase(strip(readline())) == "y"
            while interactive_repair_menu()
                println()
            end
        end
    end
    
    println("\n👋 Troubleshooting session complete!")
end
