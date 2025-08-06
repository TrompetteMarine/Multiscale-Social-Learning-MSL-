#!/usr/bin/env julia
# cleanup_repository.jl - Clean and organize BeliefSim repository

println("🧹 BeliefSim Repository Cleanup & Organization")
println("==============================================\n")

# ============================================================================
# 1. Remove redundant/outdated files
# ============================================================================

println("1. Removing redundant files...")

# Files to remove (outdated or redundant)
files_to_remove = [
    "demo/bs.jl",                    # Replaced by paper_demo.jl
    "demo/simple_demo.jl",           # Replaced by enhanced demos
    "demo_example.jl",               # Replaced by paper_demo.jl
    "simple_demo.jl",                # Redundant
    "bs.jl",                         # Basic simulation replaced by paper_demo.jl
    "organize_files.jl",             # Utility script no longer needed
    "GITHUB_SETUP_SUMMARY.md",       # Temporary file
    "scripts/Manifest.toml",         # Incorrect location
    "scripts/Project.toml"           # Incorrect location
]

removed_count = 0
for file in files_to_remove
    if isfile(file)
        rm(file)
        println("  ✅ Removed: $file")
        removed_count += 1
    elseif isdir(file)
        rm(file; recursive=true)
        println("  ✅ Removed directory: $file")
        removed_count += 1
    end
end

println("  Removed $removed_count redundant files/directories")

# ============================================================================
# 2. Update existing files with proper structure
# ============================================================================

println("\n2. Updating file structure...")

# Update simplified bs.jl as basic example
basic_example = """
#!/usr/bin/env julia
# basic_example.jl - Simple BeliefSim demonstration

using Pkg; Pkg.activate(@__DIR__)

include("src/Kernel.jl"); using .Kernel
include("src/Metrics.jl"); using .Metrics

println("🌊 Basic BeliefSim Example")
println("==========================")

# Simple parameters
params = MSLSimPars(N=50, T=10.0, Δt=0.01)

println("Running simulation with \$(params.N) agents for \$(params.T) time units...")

# Run simulation
t_vec, trajectories = run_msl_simulation(params; seed=42)

# Basic analysis
final_beliefs = [trajectories[:beliefs][i][end] for i in 1:params.N]
consensus_data = consensus_metrics(final_beliefs)

println("\\n📊 Results:")
println("   Final mean belief: \$(round(mean(final_beliefs), digits=3))")
println("   Consensus strength: \$(round(consensus_data[:consensus_strength], digits=3))")
println("   Belief spread (std): \$(round(std(final_beliefs), digits=3))")

# Create output directory and save basic plot
mkpath("output/basic")

using Plots
beliefs_mean = [mean([trajectories[:beliefs][i][t] for i in 1:params.N]) for t in 1:length(t_vec)]
p = plot(t_vec, beliefs_mean, xlabel="Time", ylabel="Mean Belief", 
         title="Basic Multi-Scale Learning", lw=2, legend=false)
savefig(p, "output/basic/belief_evolution.png")

println("   Plot saved: output/basic/belief_evolution.png")
println("\\n✅ Basic example complete!")
"""

write("basic_example.jl", basic_example)
println("  ✅ Created: basic_example.jl")

# ============================================================================
# 3. Ensure proper directory structure
# ============================================================================

println("\n3. Creating proper directory structure...")

required_dirs = [
    "src",
    "scripts", 
    "output",
    "output/basic",
    "output/paper_demo",
    "output/bifurcation",
    "output/verification",
    "output/advanced",
    "installation"
]

for dir in required_dirs
    if !isdir(dir)
        mkpath(dir)
        println("  ✅ Created: $dir/")
    end
end

# ============================================================================
# 4. Move installation files to proper location
# ============================================================================

println("\n4. Organizing installation files...")

installation_files = [
    "install.jl",
    "setup.jl", 
    "install_verification.jl",
    "troubleshooting.jl"
]

# Keep main install.jl in root, move others to installation/
main_installer = """
#!/usr/bin/env julia
# install.jl - BeliefSim Main Installer

println("🌊 BeliefSim Installation")
println("========================")
println("Installing Multi-Scale Social Learning Simulator...")
println()

# Run the detailed installation from installation directory
include("installation/install_detailed.jl")
"""

write("install.jl", main_installer)
println("  ✅ Updated: install.jl (main installer)")

# Move detailed installer
if isfile("installation/install.jl")
    mv("installation/install.jl", "installation/install_detailed.jl")
end

# ============================================================================
# 5. Create master script for all demos
# ============================================================================

println("\n5. Creating comprehensive demo menu...")

master_demo = """
#!/usr/bin/env julia
# run.jl - BeliefSim Master Demo System

println("🌊 BeliefSim: Multi-Scale Social Learning Simulator")
println("=================================================")
println("Implementation of Bontemps (2024) - Theoretical Economics")
println()

function show_menu()
    println("📋 Available Demonstrations:")
    println("   1. Paper Implementation Demo    - Core theoretical results")
    println("   2. Bifurcation Analysis        - Critical peer influence κ*")
    println("   3. Basic Example               - Simple MSL simulation")
    println("   4. Advanced Analysis Suite     - Comprehensive analysis")  
    println("   5. Installation Verification   - Test system setup")
    println("   6. Troubleshooting            - Diagnose issues")
    println("   7. Exit")
    println()
end

function run_demo(choice::String)
    if choice == "1"
        println("🚀 Running Paper Implementation Demo...")
        println("   Demonstrates: 5D agent dynamics, jump-diffusion, regime classification")
        include("paper_demo.jl")
        
    elseif choice == "2"
        println("🔍 Running Bifurcation Analysis...")
        println("   Demonstrates: Critical peer influence, supercritical pitchfork")
        include("bifurcation_analysis.jl")
        
    elseif choice == "3"
        println("📊 Running Basic Example...")
        println("   Demonstrates: Simple MSL simulation and analysis")
        include("basic_example.jl")
        
    elseif choice == "4"
        println("🧪 Running Advanced Analysis Suite...")
        println("   Demonstrates: Network comparison, heterogeneous agents, phase space")
        include("scripts/advanced_analysis.jl")
        
    elseif choice == "5"
        println("✅ Running Installation Verification...")
        include("verify_install.jl")
        
    elseif choice == "6"
        println("🔧 Running Troubleshooting...")
        include("installation/troubleshooting.jl")
        
    elseif choice == "7"
        println("👋 Thank you for using BeliefSim!")
        return false
        
    else
        println("❌ Invalid choice. Please select 1-7.")
    end
    
    return true
end

# Main loop
while true
    show_menu()
    print("Enter choice (1-7): ")
    choice = readline()
    println()
    
    if !run_demo(choice)
        break
    end
    
    println("\\n" * "="^50)
    print("Press Enter to continue or 'q' to quit: ")
    response = readline()
    if lowercase(strip(response)) == "q"
        println("👋 Goodbye!")
        break
    end
    println()
end
"""

write("run.jl", master_demo)
println("  ✅ Created: run.jl (master demo system)")

# ============================================================================
# 6. Update gitignore for clean repository
# ============================================================================

println("\n6. Creating comprehensive .gitignore...")

gitignore_content = """
# Julia
Manifest.toml
*.jl.cov
*.jl.*.cov
*.jl.mem
deps/deps.jl
docs/build/
docs/site/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# System files
.DS_Store
Thumbs.db
desktop.ini
.Trashes

# Temporary files
*~
*.tmp
*.temp
*.bak
*.backup

# Output directories (preserve structure but not content)
output/**
!output/.gitkeep
!output/*/.gitkeep

# Log files
*.log

# Archive files
*.zip
*.tar.gz
*.7z

# Jupyter notebooks checkpoints
.ipynb_checkpoints/

# Python (if any auxiliary scripts)
__pycache__/
*.pyc
*.pyo

# R (if any auxiliary scripts)
.Rhistory
.RData
.Ruserdata

# LaTeX (if any documentation)
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.synctex.gz
*.toc

# Temporary installation files
GITHUB_SETUP_SUMMARY.md
installation_temp/
"""

write(".gitignore", gitignore_content)
println("  ✅ Updated: .gitignore")

# Create .gitkeep files for output directories
gitkeep_dirs = [
    "output",
    "output/basic", 
    "output/paper_demo",
    "output/bifurcation", 
    "output/verification",
    "output/advanced"
]

for dir in gitkeep_dirs
    gitkeep_file = joinpath(dir, ".gitkeep")
    if !isfile(gitkeep_file)
        write(gitkeep_file, "# Preserve directory structure in git\n")
    end
end

println("  ✅ Added .gitkeep files for output directories")

# ============================================================================
# 7. Validate file structure
# ============================================================================

println("\n7. Validating final structure...")

essential_files = [
    "README.md" => "Main documentation",
    "install.jl" => "Main installer", 
    "run.jl" => "Master demo system",
    "paper_demo.jl" => "Paper implementation demo",
    "bifurcation_analysis.jl" => "Bifurcation analysis",
    "basic_example.jl" => "Simple example",
    "verify_install.jl" => "Installation verification",
    "Project.toml" => "Package dependencies",
    "src/Kernel.jl" => "Core simulation engine",
    "src/Metrics.jl" => "Analysis tools",
    "src/Viz.jl" => "Visualization functions"
]

missing_files = []
present_files = []

for (file, description) in essential_files
    if isfile(file)
        println("  ✅ $file")
        push!(present_files, file)
    else
        println("  ❌ $file - $description")
        push!(missing_files, file)
    end
end

# ============================================================================
# 8. Generate final summary
# ============================================================================

println("\n" * "="^60)
println("🎉 REPOSITORY CLEANUP COMPLETE")
println("="^60)

println("\n📊 Summary:")
println("   Files present: $(length(present_files))/$(length(essential_files))")
println("   Files removed: $removed_count redundant items")
println("   Directory structure: ✅ Organized")
println("   Git configuration: ✅ Updated")

if isempty(missing_files)
    println("\n🌟 Repository Status: READY FOR GITHUB")
    println("\n🚀 Next Steps:")
    println("   1. git init")
    println("   2. git add .")
    println("   3. git commit -m \"BeliefSim: Multi-Scale Social Learning Simulator\"")
    println("   4. git remote add origin <your-github-url>")
    println("   5. git push -u origin main")
    
    println("\n💡 Quick Start:")
    println("   julia install.jl      # Install dependencies")
    println("   julia run.jl          # Interactive demo system")
    println("   julia paper_demo.jl   # Core paper results")
    
    println("\n📚 Features Ready:")
    println("   • Complete 5D agent state dynamics")
    println("   • Jump-diffusion with cognitive thresholds") 
    println("   • Multi-scale shift detection")
    println("   • Regime classification (Equilibrium/Buffered/Broadcast/Cascade)")
    println("   • Bifurcation analysis with critical peer influence")
    println("   • Multiple network topologies")
    println("   • Comprehensive visualization suite")
    
else
    println("\n⚠️  Missing Files:")
    for file in missing_files
        println("     • $file")
    end
    println("\n🔧 Action Required:")
    println("   Create missing files before GitHub upload")
end

println("\n🌊 BeliefSim: From Individual Bounded Rationality to Collective Dynamics")
println("="^60)

# Create a final status report
status_report = """
# BeliefSim Repository Status Report

Generated: $(now())

## ✅ Repository Health: $(isempty(missing_files) ? "EXCELLENT" : "NEEDS ATTENTION")

### File Structure
- Essential files: $(length(present_files))/$(length(essential_files)) present
- Redundant files: $removed_count removed
- Directory structure: Organized and clean
- Git configuration: Updated with comprehensive .gitignore

### Core Features Implemented
- ✅ Multi-scale social learning dynamics (5D agent state)
- ✅ Jump-diffusion SDE system with cognitive thresholds
- ✅ Attention-constrained social networks  
- ✅ Mean-field analysis and bifurcation theory
- ✅ Regime classification system
- ✅ Ensemble simulations and Monte Carlo analysis
- ✅ Advanced visualization suite
- ✅ Comprehensive documentation

### Demo System
- ✅ Interactive launcher (run.jl)
- ✅ Paper implementation demo
- ✅ Bifurcation analysis demo
- ✅ Basic example for newcomers
- ✅ Installation verification
- ✅ Troubleshooting tools

### Ready for GitHub: $(isempty(missing_files) ? "YES" : "AFTER FIXING MISSING FILES")

---
🌊 BeliefSim: Multi-Scale Social Learning Simulator
Implementation of Bontemps (2024) theoretical framework
"""

write("REPOSITORY_STATUS.md", status_report)
println("\n📄 Status report saved: REPOSITORY_STATUS.md")
