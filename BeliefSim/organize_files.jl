#!/usr/bin/env julia
# organize_files.jl - Organize BeliefSim files for GitHub upload

println("📁 BeliefSim File Organization Helper")
println("====================================\n")

# Define the expected file structure
expected_files = [
    # Core documentation
    "README.md" => "Main documentation and user guide",
    "INSTALLATION.md" => "Detailed installation instructions", 
    "Project.toml" => "Julia package dependencies",
    
    # Installation system
    "install.jl" => "One-command installer",
    "setup.jl" => "Package installation script",
    "install_verification.jl" => "Installation verification",
    "troubleshoot.jl" => "Diagnostic and troubleshooting tool",
    
    # Interactive demos
    "demo_example.jl" => "Full interactive demo",
    "simple_demo.jl" => "Streamlined demo (robust)",
    "run.jl" => "Quick launcher menu",
    "bs.jl" => "Basic simulation script",
    
    # Core modules
    "src/Kernel.jl" => "Core simulation engine",
    "src/Metrics.jl" => "Analysis and metrics tools", 
    "src/Viz.jl" => "Visualization functions",
    
    # Research scripts
    "scripts/advanced_analysis.jl" => "Comprehensive analysis suite",
    "scripts/montecarlo_shift.jl" => "Monte Carlo ensemble analysis",
    "scripts/bifurcation.jl" => "Parameter bifurcation analysis",
    "scripts/heatmap.jl" => "Network heatmap visualization",
    "scripts/pipeline.jl" => "Analysis pipeline"
]

# Check file status
println("📋 Checking file organization...")
println("-" ^ 50)

missing_files = []
present_files = []

for (file, description) in expected_files
    if isfile(file)
        println("✅ $file")
        push!(present_files, file)
    else
        println("❌ $file - $description")
        push!(missing_files, file)
    end
end

println("\n📊 Summary:")
println("-" ^ 20)
println("Present: $(length(present_files))/$(length(expected_files)) files")
println("Missing: $(length(missing_files)) files")

if !isempty(missing_files)
    println("\n⚠️  Missing Files:")
    for file in missing_files
        println("  • $file")
    end
    println("\n💡 Action needed:")
    println("   Make sure to create/copy all missing files before uploading to GitHub")
end

# Check directory structure
println("\n📂 Directory Structure Check:")
println("-" ^ 30)

required_dirs = ["src", "scripts"]
for dir in required_dirs
    if isdir(dir)
        println("✅ $dir/ directory exists")
        
        # List files in directory
        files = readdir(dir)
        if !isempty(files)
            println("   Contains: $(join(files, ", "))")
        else
            println("   ⚠️  Directory is empty")
        end
    else
        println("❌ $dir/ directory missing")
        println("   Creating directory...")
        mkdir(dir)
        println("   ✅ Created $dir/")
    end
end

# Create .gitignore if missing
println("\n🚫 Git Configuration:")
println("-" ^ 20)

if isfile(".gitignore")
    println("✅ .gitignore exists")
else
    println("❌ .gitignore missing")
    println("   Creating .gitignore...")
    
    gitignore_content = """
# Julia
Manifest.toml
*.jl.cov
*.jl.*.cov
*.jl.mem
deps/deps.jl
docs/build/
docs/site/

# Temporary files
*~
*.tmp
.DS_Store
Thumbs.db

# Output directories
output/
!output/.gitkeep

# IDE files
.vscode/
.idea/
*.swp
*.swo

# System files
.DS_Store
Thumbs.db
desktop.ini

# Backup files
*.bak
*.backup

# Log files
*.log
"""
    
    write(".gitignore", gitignore_content)
    println("   ✅ .gitignore created")
end

# Create output directory structure
println("\n📁 Output Directory:")
println("-" ^ 20)

if !isdir("output")
    println("❌ output/ directory missing")
    println("   Creating output directory structure...")
    mkpath("output")
    
    # Create subdirectories
    subdirs = ["demo", "advanced", "verification"]
    for subdir in subdirs
        mkpath("output/$subdir")
        # Create .gitkeep to preserve directory in git
        write("output/$subdir/.gitkeep", "")
    end
    
    println("   ✅ Output directories created")
else
    println("✅ output/ directory exists")
end

# Repository readiness check
println("\n🚀 GitHub Readiness Check:")
println("-" ^ 28)

readiness_score = 0
total_checks = 6

# Check 1: Essential files
if all(isfile(f) for (f, _) in expected_files if f in ["README.md", "Project.toml", "install.jl"])
    println("✅ Essential files present")
    readiness_score += 1
else
    println("❌ Missing essential files (README.md, Project.toml, install.jl)")
end

# Check 2: Core modules
if all(isfile(f) for (f, _) in expected_files if startswith(f, "src/"))
    println("✅ Core modules complete")
    readiness_score += 1
else
    println("❌ Core modules incomplete")
end

# Check 3: Demo scripts
demo_files = ["demo_example.jl", "simple_demo.jl", "run.jl"]
if all(isfile(f) for f in demo_files)
    println("✅ Demo scripts ready")
    readiness_score += 1
else
    println("❌ Demo scripts incomplete")
end

# Check 4: Installation system
install_files = ["install.jl", "setup.jl", "install_verification.jl", "troubleshoot.jl"]
if all(isfile(f) for f in install_files)
    println("✅ Installation system complete")
    readiness_score += 1
else
    println("❌ Installation system incomplete")
end

# Check 5: Git configuration
if isfile(".gitignore")
    println("✅ Git configuration ready")
    readiness_score += 1
else
    println("❌ Git configuration incomplete")
end

# Check 6: Directory structure
if all(isdir(d) for d in ["src", "scripts", "output"])
    println("✅ Directory structure organized")
    readiness_score += 1
else
    println("❌ Directory structure needs work")
end

# Final assessment
println("\n🏆 Readiness Score: $readiness_score/$total_checks")

if readiness_score == total_checks
    println("\n🎉 Perfect! Your BeliefSim project is ready for GitHub!")
    println("\n📋 Next steps:")
    println("   1. Run: git init")
    println("   2. Run: git add .")
    println("   3. Run: git commit -m \"Initial BeliefSim release\"")
    println("   4. Create repository on GitHub")
    println("   5. Run: git remote add origin <your-github-url>")
    println("   6. Run: git push -u origin main")
    
elseif readiness_score >= 4
    println("\n✅ Good! Your project is mostly ready for GitHub.")
    println("   Address the missing items above, then proceed with git setup.")
    
else
    println("\n⚠️  Your project needs more work before GitHub upload.")
    println("   Please address the missing files and try again.")
end

# Create a quick setup summary
println("\n📄 Creating GITHUB_SETUP_SUMMARY.md...")

summary_content = """
# BeliefSim GitHub Setup Summary

Generated on: $(now())

## File Status ($(length(present_files))/$(length(expected_files)) complete)

### ✅ Present Files
$(join(["- $f" for f in present_files], "\n"))

### ❌ Missing Files  
$(isempty(missing_files) ? "None! All files present." : join(["- $f" for f in missing_files], "\n"))

## Readiness Score: $readiness_score/$total_checks

## Quick GitHub Commands

```bash
# Initialize and commit
git init
git add .
git commit -m "Initial BeliefSim release"

# Connect to GitHub (replace with your URL)  
git remote add origin https://github.com/username/BeliefSim.git
git push -u origin main
```

## Repository Details
- **Recommended Name**: BeliefSim
- **Description**: Multi-Scale Social Learning Dynamics Simulator in Julia
- **Topics**: julia, simulation, social-networks, belief-dynamics, sde, research-tool
- **License**: MIT (recommended for open source)

## After Upload
1. Add repository description and topics
2. Enable Issues and Discussions  
3. Create first release (v0.1.0)
4. Add CONTRIBUTING.md guidelines

Your BeliefSim project is $(readiness_score == total_checks ? "ready" : "nearly ready") for GitHub! 🌊
"""

write("GITHUB_SETUP_SUMMARY.md", summary_content)
println("✅ Summary saved to GITHUB_SETUP_SUMMARY.md")

println("\n🌊 File organization complete!")
