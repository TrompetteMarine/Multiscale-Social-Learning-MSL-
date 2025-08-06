#!/usr/bin/env julia
# install.jl - Complete BeliefSim Installation Manager
# Handles different installation scenarios and troubleshooting

using Pkg, InteractiveUtils

const BELIEFBIM_ASCII = """
██████╗ ███████╗██╗     ██╗███████╗███████╗███████╗██╗███╗   ███╗
██╔══██╗██╔════╝██║     ██║██╔════╝██╔════╝██╔════╝██║████╗ ████║
██████╔╝█████╗  ██║     ██║█████╗  █████╗  ███████╗██║██╔████╔██║
██╔══██╗██╔══╝  ██║     ██║██╔══╝  ██╔══╝  ╚════██║██║██║╚██╔╝██║
██████╔╝███████╗███████╗██║███████╗██║     ███████║██║██║ ╚═╝ ██║
╚═════╝ ╚══════╝╚══════╝╚═╝╚══════╝╚═╝     ╚══════╝╚═╝╚═╝     ╚═╝
    Multi-Scale Social Learning Dynamics Simulator
"""

function print_header()
    println(BELIEFBIM_ASCII)
    println("Version 0.1.0 - Installation Manager")
    println("="^65)
    println()
end

function check_julia_version()
    println("🔍 Checking Julia compatibility...")
    println("Julia Version: $(VERSION)")
    
    if VERSION < v"1.6.0"
        println("⚠️  Warning: Julia $(VERSION) detected.")
        println("   BeliefSim is tested with Julia 1.6+")
        println("   Some features may not work correctly.")
        print("Continue anyway? (y/N): ")
        response = readline()
        if lowercase(strip(response)) != "y"
            println("Installation cancelled.")
            exit(0)
        end
    else
        println("✅ Julia version compatible")
    end
    println()
end

function detect_installation_type()
    println("🎯 Detecting installation scenario...")
    
    has_project = isfile("Project.toml")
    has_manifest = isfile("Manifest.toml")
    has_src = isdir("src")
    
    if has_project && has_src
        println("✅ BeliefSim project structure detected")
        return :existing_project
    elseif has_project
        println("📁 Julia project detected, adding BeliefSim dependencies")
        return :add_to_project
    else
        println("🆕 Setting up new BeliefSim project")
        return :new_project
    end
end

function install_packages(mode::Symbol)
    println("\n📦 Installing packages...")
    
    # Activate environment
    Pkg.activate(@__DIR__)
    
    # Core packages that must be installed
    essential_packages = [
        "DifferentialEquations",
        "Distributions", 
        "Graphs",
        "CSV",
        "DataFrames",
        "Plots",
        "StatsBase"
    ]
    
    # Optional packages (install if possible, warn if not)
    optional_packages = [
        "GraphPlot",
        "NetworkLayout", 
        "Colors",
        "Distances"
    ]
    
    installed = []
    failed = []
    
    # Install essential packages
    println("Installing essential packages:")
    for pkg in essential_packages
        print("  $pkg... ")
        try
            Pkg.add(pkg)
            push!(installed, pkg)
            println("✅")
        catch e
            push!(failed, pkg)
            println("❌")
            println("    Error: $e")
        end
    end
    
    # Install optional packages  
    println("\nInstalling optional packages:")
    for pkg in optional_packages
        print("  $pkg... ")
        try
            Pkg.add(pkg)
            push!(installed, pkg)
            println("✅")
        catch e
            push!(failed, pkg)
            println("⚠️  (optional)")
        end
    end
    
    if !isempty(failed)
        essential_failed = filter(p -> p in essential_packages, failed)
        if !isempty(essential_failed)
            println("\n❌ Critical packages failed to install: $(join(essential_failed, ", "))")
            println("BeliefSim may not work correctly.")
        end
    end
    
    println("\n📈 Installation summary:")
    println("  Installed: $(length(installed)) packages")  
    println("  Failed: $(length(failed)) packages")
    
    return length(failed) == 0
end

function create_project_files()
    println("\n📝 Creating project configuration files...")
    
    # Create Project.toml
    project_toml = """
name = "BeliefSim"
uuid = "12345678-1234-5678-9abc-123456789012"  
version = "0.1.0"
authors = ["BeliefSim User"]

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
"""
    
    write("Project.toml", project_toml)
    println("✅ Project.toml created")
end

function run_tests()
    println("\n🧪 Running installation tests...")
    
    test_count = 0
    passed_count = 0
    
    tests = [
        ("Package Loading", test_package_loading),
        ("BeliefSim Modules", test_modules),
        ("Basic Simulation", test_basic_simulation),
        ("Visualization", test_visualization)
    ]
    
    for (test_name, test_func) in tests
        test_count += 1
        print("  $test_name... ")
        try
            if test_func()
                passed_count += 1
                println("✅")
            else
                println("❌")
            end
        catch e
            println("❌ (Error: $e)")
        end
    end
    
    println("\n📊 Test Results: $passed_count/$test_count passed")
    return passed_count == test_count
end

function test_package_loading()
    try
        using DifferentialEquations, Distributions, Graphs
        using CSV, DataFrames, Plots, StatsBase
        using LinearAlgebra, Statistics, Random
        return true
    catch
        return false
    end
end

function test_modules()
    try
        include("src/Kernel.jl")
        include("src/Metrics.jl")
        include("src/Viz.jl")
        using .Kernel, .Metrics, .Viz
        return true
    catch
        return false
    end
end

function test_basic_simulation()
    try
        pars = SimPars(N=5, κ=1.0, β=0.5, σ=0.3, T=0.5, Δt=0.01)
        W = fully_connected(5)
        t_vec, traj = run_one_path(pars; W=W, seed=42)
        consensus_metrics(traj[end])
        return true
    catch
        return false  
    end
end

function test_visualization()
    try
        mkpath("output/install_test")
        ENV["GKSwstype"] = "100"  # Non-interactive backend
        p = plot([1,2,3], [1,4,2])
        savefig(p, "output/install_test/test.png")
        return true
    catch
        return false
    end
end

function create_shortcuts()
    println("\n🔗 Creating convenience scripts...")
    
    # Create run script
    run_script = """
#!/usr/bin/env julia
# run.jl - Quick launcher for BeliefSim

println("🌊 BeliefSim Quick Launcher")
println("==========================")
println()
println("Choose an option:")
println("  1. Simple Demo (recommended for first-time users)")
println("  2. Full Interactive Demo") 
println("  3. Basic Simulation")
println("  4. Monte Carlo Analysis")
println("  5. Advanced Analysis Suite")
println("  6. Installation Verification")
println("  7. Troubleshooting Tool")
println("  8. Exit")
print("\\nEnter choice (1-8): ")

choice = readline()
println()

if choice == "1"
    println("🚀 Running Simple Demo...")
    include("simple_demo.jl")
elseif choice == "2"
    println("🚀 Running Full Interactive Demo...")
    include("demo_example.jl")
elseif choice == "3"
    println("🚀 Running Basic Simulation...")
    include("bs.jl")
elseif choice == "4"
    println("🚀 Running Monte Carlo Analysis...")
    include("scripts/montecarlo_shift.jl")
elseif choice == "5"
    println("🚀 Running Advanced Analysis...")
    include("scripts/advanced_analysis.jl") 
elseif choice == "6"
    println("🚀 Running Installation Verification...")
    include("install_verification.jl")
elseif choice == "7"
    println("🚀 Running Troubleshooting Tool...")
    include("troubleshoot.jl")
elseif choice == "8"
    println("👋 Goodbye!")
else
    println("❌ Invalid choice. Please run 'julia run.jl' again.")
end
"""
    
    write("run.jl", run_script)
    println("✅ run.jl created (quick launcher)")
    
    # Create update script
    update_script = """
#!/usr/bin/env julia  
# update.jl - Update BeliefSim packages

using Pkg
Pkg.activate(@__DIR__)
println("🔄 Updating BeliefSim packages...")
Pkg.update()
println("✅ Update complete!")
"""
    
    write("update.jl", update_script)
    println("✅ update.jl created (package updater)")
end

function print_completion_message(success::Bool)
    println("\n" * "="^65)
    
    if success
        println("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
        println()
        println("✅ BeliefSim is ready to use!")
        println()
        println("🚀 Quick Start Commands:")
        println("  julia run.jl                    # Interactive launcher")
        println("  julia demo_example.jl           # Interactive demo") 
        println("  julia install_verification.jl   # Verify installation")
        println()
        println("📚 Documentation:")
        println("  README.md                       # Complete guide")
        println("  src/                           # Source modules")
        println("  scripts/                       # Analysis scripts")
        println()
        println("💡 Pro tip: Start with 'julia demo_example.jl'!")
        
    else
        println("⚠️  INSTALLATION COMPLETED WITH ISSUES")
        println()
        println("Some components may not work correctly.")
        println()
        println("🔧 Troubleshooting:")
        println("  1. Run 'julia install_verification.jl' for detailed diagnostics")
        println("  2. Check Julia version (1.6+ recommended)")
        println("  3. Update package registry: Pkg.Registry.update()")
        println("  4. Try manual package installation")
        println()
        println("📧 If problems persist, check the GitHub issues page")
    end
    
    println()
    println("🌊 Thank you for choosing BeliefSim!")
    println("="^65)
end

# Main installation flow
function main()
    print_header()
    check_julia_version()
    
    installation_type = detect_installation_type()
    
    # Create project files if needed
    if installation_type == :new_project
        create_project_files()
    end
    
    # Install packages
    package_success = install_packages(installation_type)
    
    # Instantiate environment  
    println("\n⚙️  Finalizing environment...")
    try
        Pkg.instantiate()
        Pkg.precompile()
        println("✅ Environment ready")
    catch e
        println("⚠️  Environment setup warnings: $e")
    end
    
    # Run tests
    test_success = run_tests()
    
    # Create convenience scripts
    create_shortcuts()
    
    # Final report
    overall_success = package_success && test_success
    print_completion_message(overall_success)
    
    return overall_success ? 0 : 1
end

# Run installer
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end