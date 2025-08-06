#!/usr/bin/env julia
# install.jl - Complete BeliefSim Installation Manager (Fixed)

using Pkg, InteractiveUtils

const BELIEFBIM_ASCII = """
🌊 BeliefSim: Multi-Scale Social Learning Simulator 🌊
====================================================
Implementation of "Multi-Scale Social Learning: From Individual 
Bounded Rationality to Collective Dynamics" (Bontemps, 2024)
====================================================
"""

println(BELIEFBIM_ASCII)
println("Julia Version: $(VERSION)")
println()

# ============================================================================
# Robust Package Installation
# ============================================================================

function install_packages_robust()
    println("📦 Installing BeliefSim dependencies...")
    
    # Activate local environment
    Pkg.activate(@__DIR__)
    
    # Core packages (must work for basic functionality)
    essential_packages = [
        "DifferentialEquations",  # SDE solver
        "Distributions",          # Random distributions
        "Graphs",                # Network generation
        "LinearAlgebra",         # Matrix operations (built-in)
        "Statistics",            # Statistics (built-in)
        "Random",               # Random numbers (built-in)
        "SparseArrays"          # Sparse matrices (built-in)
    ]
    
    # Analysis packages (important but not critical)
    analysis_packages = [
        "CSV",                  # Data export
        "DataFrames",          # Data frames
        "StatsBase",           # Extended statistics
        "Distances"            # Distance metrics
    ]
    
    # Visualization packages (optional, can fall back to basic plots)
    viz_packages = [
        "Plots",               # Basic plotting
        "Colors",              # Color management
        "GraphPlot",           # Network visualization  
        "NetworkLayout"        # Graph layouts
    ]
    
    installed = []
    failed = []
    
    # Install essential packages
    println("Installing essential packages:")
    for pkg in essential_packages
        if pkg in ["LinearAlgebra", "Statistics", "Random", "SparseArrays"]
            println("  $pkg... ✅ (built-in)")
            push!(installed, pkg)
            continue
        end
        
        print("  $pkg... ")
        try
            Pkg.add(pkg)
            println("✅")
            push!(installed, pkg)
        catch e
            println("❌")
            push!(failed, pkg)
            @warn "Failed to install $pkg: $e"
        end
    end
    
    # Install analysis packages
    println("\nInstalling analysis packages:")
    for pkg in analysis_packages
        print("  $pkg... ")
        try
            Pkg.add(pkg) 
            println("✅")
            push!(installed, pkg)
        catch e
            println("⚠️  (optional)")
            push!(failed, pkg)
        end
    end
    
    # Install visualization packages
    println("\nInstalling visualization packages:")
    for pkg in viz_packages
        print("  $pkg... ")
        try
            Pkg.add(pkg)
            println("✅") 
            push!(installed, pkg)
        catch e
            println("⚠️  (optional)")
            push!(failed, pkg)
        end
    end
    
    return installed, failed
end

# ============================================================================
# Create/Update Project Files
# ============================================================================

function create_project_toml()
    project_content = """
name = "BeliefSim"
uuid = "12345678-1234-5678-9abc-123456789012"
version = "0.2.0"
authors = ["BeliefSim Contributors"]

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
Colors = "0.12"
GraphPlot = "0.5"
NetworkLayout = "0.4"
Distances = "0.10"
"""
    
    write("Project.toml", project_content)
    println("✅ Project.toml created/updated")
end

# ============================================================================
# Test Installation
# ============================================================================

function test_installation()
    println("\n🧪 Testing installation...")
    
    tests = [
        ("Core packages", test_core_packages),
        ("BeliefSim modules", test_modules),
        ("Basic simulation", test_basic_simulation),
        ("Plotting", test_plotting)
    ]
    
    results = []
    
    for (test_name, test_func) in tests
        print("  $test_name... ")
        try
            success = test_func()
            if success
                println("✅")
                push!(results, (test_name, true, ""))
            else
                println("❌")
                push!(results, (test_name, false, "Test returned false"))
            end
        catch e
            println("❌")
            push!(results, (test_name, false, string(e)))
        end
    end
    
    return results
end

function test_core_packages()
    using DifferentialEquations, Distributions, Graphs
    using LinearAlgebra, Statistics, Random, SparseArrays
    return true
end

function test_modules()
    include("src/Kernel.jl")
    using .Kernel
    include("src/Metrics.jl") 
    using .Metrics
    return true
end

function test_basic_simulation()
    # Test the enhanced MSL simulation
    params = MSLSimPars(N=10, T=1.0, Δt=0.01)
    t_vec, trajectories = run_msl_simulation(params; seed=42)
    
    # Verify we have the right structure
    return haskey(trajectories, :beliefs) && length(trajectories[:beliefs]) == 10
end

function test_plotting()
    try
        ENV["GKSwstype"] = "100"  # Non-interactive backend
        using Plots
        p = plot([1,2,3], [1,4,2])
        
        mkpath("output/test")
        savefig(p, "output/test/installation_test.png")
        return isfile("output/test/installation_test.png")
    catch
        return false
    end
end

# ============================================================================
# Create Demo Scripts
# ============================================================================

function create_demo_scripts()
    println("\n📝 Creating demo scripts...")
    
    # Create enhanced paper demo
    paper_demo = """
#!/usr/bin/env julia
# paper_demo.jl - Demonstrate paper's core results

using Pkg; Pkg.activate(@__DIR__)

include("src/Kernel.jl"); using .Kernel
include("src/Metrics.jl"); using .Metrics
using Plots, Statistics

println("🌊 BeliefSim: Paper Implementation Demo")
println("=====================================")

# Parameters matching paper's setup
cognitive_params = CognitiveParams(
    λ = 1.0,      # Mean reversion strength
    α = 0.5,      # Social influence scale  
    σ = 0.3,      # Idiosyncratic noise
    δm = 0.1,     # Memory adjustment
    ηw = 0.1,     # Deliberation adjustment
    βΘ = 0.05     # Threshold adjustment
)

network_params = NetworkParams(type=:small_world, k=6, p=0.3)

params = MSLSimPars(
    N = 100,
    T = 30.0,
    cognitive = cognitive_params,
    network = network_params
)

println("\\n1. Running single MSL simulation...")
t_vec, trajectories = run_msl_simulation(params; seed=42)

println("✅ Simulation complete!")
println("   Time steps: \$(length(t_vec))")
println("   Agents: \$(length(trajectories[:beliefs]))")

# Multi-scale analysis
println("\\n2. Performing multi-scale analysis...")
W = create_network(params.N, params.network)
analysis = multi_scale_analysis(trajectories, W, t_vec)

println("✅ Analysis complete!")
println("   Regime detected: \$(analysis[:regime].regime)")
println("   Final consensus: \$(round(analysis[:stability][:final_consensus], digits=3))")
println("   System stable: \$(analysis[:stability][:is_stable])")

# Basic visualization
println("\\n3. Creating visualizations...")
mkpath("output/paper_demo")

# Belief evolution
beliefs_mean = [mean([trajectories[:beliefs][i][t] for i in 1:params.N]) for t in 1:length(t_vec)]
p1 = plot(t_vec, beliefs_mean, xlabel="Time", ylabel="Mean Belief", 
          title="Multi-Scale Social Learning Dynamics", lw=2)
savefig(p1, "output/paper_demo/belief_evolution.png")

# Consensus evolution  
p2 = plot(t_vec, analysis[:time_series][:consensus], 
          xlabel="Time", ylabel="Consensus Strength", 
          title="Consensus Formation", lw=2)
savefig(p2, "output/paper_demo/consensus_evolution.png")

println("✅ Plots saved to output/paper_demo/")

println("\\n🎉 Paper demo complete!")
println("\\n📚 This demonstrates:")
println("   • Full 5D agent state dynamics (x,r,m,w,Θ)")
println("   • Jump-diffusion with cognitive tension") 
println("   • Multi-scale social learning")
println("   • Regime classification")
println("   • Network effects on consensus formation")
"""
    
    write("paper_demo.jl", paper_demo)
    println("✅ paper_demo.jl created")
    
    # Create bifurcation analysis script
    bifurcation_script = """
#!/usr/bin/env julia
# bifurcation_analysis.jl - Analyze critical peer influence

using Pkg; Pkg.activate(@__DIR__)

include("src/Kernel.jl"); using .Kernel
include("src/Metrics.jl"); using .Metrics
using Plots, Statistics

println("🔍 Bifurcation Analysis: Critical Peer Influence")
println("================================================")

# Parameter sweep around critical point
κ_values = 0.2:0.1:1.5
base_params = MSLSimPars(N=80, T=25.0, network=NetworkParams(type=:small_world))

results = []
println("Running parameter sweep...")

for (i, α) in enumerate(κ_values)
    print("  α = \$α (\$i/\$(length(κ_values)))... ")
    
    # Modify social influence parameter
    test_params = MSLSimPars(
        N = base_params.N,
        T = base_params.T, 
        cognitive = CognitiveParams(base_params.cognitive; α = α),
        network = base_params.network
    )
    
    t_vec, trajectories = run_msl_simulation(test_params; seed=123)
    
    # Analyze final state
    final_beliefs = [trajectories[:beliefs][i][end] for i in 1:base_params.N]
    consensus_data = consensus_metrics(final_beliefs)
    polarization_data = polarization_metrics(final_beliefs)
    
    push!(results, (
        α = α,
        consensus = consensus_data[:consensus_strength],
        polarization = polarization_data[:group_separation]
    ))
    
    println("✅")
end

# Extract data for plotting
α_vals = [r.α for r in results]
consensus_vals = [r.consensus for r in results] 
polarization_vals = [r.polarization for r in results]

# Find critical point
critical_analysis = critical_peer_influence(consensus_vals, α_vals)
println("\\n📊 Bifurcation Analysis Results:")
println("   Critical α*: \$(round(critical_analysis[:κ_star], digits=3))")
println("   Bifurcation strength: \$(round(critical_analysis[:bifurcation_strength], digits=3))")

# Create bifurcation plot
mkpath("output/bifurcation")

p1 = plot(α_vals, consensus_vals, xlabel="Social Influence (α)", ylabel="Final Consensus",
          title="Consensus-Polarization Bifurcation", lw=2, marker=:circle, label="Consensus")
plot!(p1, α_vals, polarization_vals, lw=2, marker=:square, label="Polarization", color=:red)
vline!(p1, [critical_analysis[:κ_star]], ls=:dash, color=:green, lw=2, label="Critical α*")

savefig(p1, "output/bifurcation/bifurcation_diagram.png")
println("\\n✅ Bifurcation diagram saved to output/bifurcation/")

println("\\n🎯 Key Insight: System undergoes supercritical pitchfork bifurcation")
println("     at α* ≈ \$(round(critical_analysis[:κ_star], digits=2))")
"""
    
    write("bifurcation_analysis.jl", bifurcation_script)
    println("✅ bifurcation_analysis.jl created")
end

# ============================================================================
# Create Launcher Script
# ============================================================================

function create_launcher()
    launcher_script = """
#!/usr/bin/env julia
# run.jl - BeliefSim Quick Launcher

println("🌊 BeliefSim: Multi-Scale Social Learning Simulator")
println("==================================================")
println("Implementation of Bontemps (2024) paper")
println()
println("Choose an option:")
println("  1. Paper Implementation Demo")
println("  2. Bifurcation Analysis") 
println("  3. Installation Verification")
println("  4. Troubleshooting")
println("  5. Exit")
print("\\nEnter choice (1-5): ")

choice = readline()
println()

if choice == "1"
    println("🚀 Running Paper Demo...")
    include("paper_demo.jl")
elseif choice == "2" 
    println("🔍 Running Bifurcation Analysis...")
    include("bifurcation_analysis.jl")
elseif choice == "3"
    println("🧪 Running Installation Verification...")
    include("verify_install.jl")
elseif choice == "4"
    println("🔧 Running Troubleshooting...")
    include("troubleshoot.jl")
elseif choice == "5"
    println("👋 Goodbye!")
else
    println("❌ Invalid choice. Please run 'julia run.jl' again.")
end
"""
    
    write("run.jl", launcher_script)
    println("✅ run.jl launcher created")
end

# ============================================================================
# Installation Verification Script
# ============================================================================

function create_verification()
    verification_script = """
#!/usr/bin/env julia
# verify_install.jl - Verify BeliefSim installation

using Pkg; Pkg.activate(@__DIR__)

println("🔍 BeliefSim Installation Verification")
println("======================================")

# Test core functionality step by step
tests = [
    ("Loading core modules", () -> begin
        include("src/Kernel.jl"); using .Kernel
        include("src/Metrics.jl"); using .Metrics
        return true
    end),
    
    ("Creating simulation parameters", () -> begin
        params = MSLSimPars(N=10, T=2.0, Δt=0.01)
        return params.N == 10
    end),
    
    ("Running MSL simulation", () -> begin
        params = MSLSimPars(N=5, T=1.0, Δt=0.01)
        t_vec, trajectories = run_msl_simulation(params; seed=42)
        return length(t_vec) > 0 && haskey(trajectories, :beliefs)
    end),
    
    ("Multi-scale analysis", () -> begin
        params = MSLSimPars(N=5, T=1.0)
        t_vec, trajectories = run_msl_simulation(params; seed=42)
        W = create_network(5, NetworkParams())
        analysis = multi_scale_analysis(trajectories, W, t_vec)
        return haskey(analysis, :regime)
    end),
    
    ("Basic visualization", () -> begin
        try
            ENV["GKSwstype"] = "100"
            using Plots
            p = plot([1,2,3], [1,4,2])
            mkpath("output/verification")
            savefig(p, "output/verification/test.png")
            return isfile("output/verification/test.png")
        catch
            return false
        end
    end)
]

passed = 0
total = length(tests)

for (test_name, test_func) in tests
    print("\$(test_name)... ")
    try
        if test_func()
            println("✅")
            passed += 1
        else
            println("❌")
        end
    catch e
        println("❌ (\$e)")
    end
end

println("\\n📊 Results: \$passed/\$total tests passed")

if passed == total
    println("\\n🎉 All tests passed! BeliefSim is ready to use.")
    println("\\n🚀 Try: julia paper_demo.jl")
else
    println("\\n⚠️  Some tests failed. Try:")
    println("   julia install.jl        # Reinstall")
    println("   julia troubleshoot.jl   # Diagnose issues")
end
"""
    
    write("verify_install.jl", verification_script)
    println("✅ verify_install.jl created")
end

# ============================================================================
# Main Installation Flow
# ============================================================================

function main()
    println("🚀 Starting BeliefSim installation...\n")
    
    # 1. Install packages
    installed, failed = install_packages_robust()
    
    # 2. Create/update project files
    create_project_toml()
    
    # 3. Instantiate environment
    println("\n⚙️  Finalizing environment...")
    try
        Pkg.instantiate()
        Pkg.precompile()
        println("✅ Environment ready")
    catch e
        @warn "Environment setup issues: $e"
    end
    
    # 4. Test installation
    test_results = test_installation()
    
    # 5. Create demo scripts and launcher
    create_demo_scripts()
    create_launcher()
    create_verification()
    
    # 6. Create directory structure
    println("\n📁 Creating directory structure...")
    for dir in ["output/paper_demo", "output/bifurcation", "output/verification"]
        mkpath(dir)
    end
    println("✅ Directories created")
    
    # 7. Final report
    println("\n" * "="^60)
    successful_tests = sum([r[2] for r in test_results])
    total_tests = length(test_results)
    
    if successful_tests == total_tests && length(failed) == 0
        println("🎉 INSTALLATION SUCCESSFUL!")
        println()
        println("✅ BeliefSim is ready for multi-scale social learning simulations!")
        println()
        println("🚀 Quick Start:")
        println("  julia run.jl                # Interactive launcher")
        println("  julia paper_demo.jl         # Demonstrate paper results")
        println("  julia bifurcation_analysis.jl # Critical point analysis")
        println()
        println("📚 Features:")
        println("  • Full 5D agent dynamics (beliefs, references, memory, deliberation, thresholds)")
        println("  • Jump-diffusion with cognitive tension triggers")
        println("  • Multi-scale shift detection and regime classification")  
        println("  • Bifurcation analysis and critical peer influence")
        println("  • Multiple network topologies and social learning models")
        
    else
        println("⚠️  INSTALLATION COMPLETED WITH ISSUES")
        println()
        println("Test Results:")
        for (test_name, success, error) in test_results
            status = success ? "✅" : "❌"
            println("  $status $test_name")
            if !success && !isempty(error)
                println("      Error: $error")
            end
        end
        
        if !isempty(failed)
            println("\\nFailed packages: $(join(failed, ", "))")
        end
        
        println("\\n🔧 Try:")
        println("  julia verify_install.jl     # Detailed verification")
        println("  julia troubleshoot.jl       # Diagnose issues")
    end
    
    println("\n🌊 Multi-Scale Social Learning awaits!")
    println("="^60)
end

# Run installation
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
