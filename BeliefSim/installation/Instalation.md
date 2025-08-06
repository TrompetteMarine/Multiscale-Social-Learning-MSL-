# BeliefSim Installation Guide

This guide will help you set up BeliefSim with all required dependencies on your system.

## 🚀 Quick Installation

### Option 1: Automatic Setup (Recommended)

**On Unix/Linux/macOS:**
```bash
julia install.jl
```

**On Windows:**
```cmd
julia install.jl
```

This will automatically detect your system, install all packages, and verify the installation.

### Option 2: Manual Setup

If you prefer manual control:

1. **Install Julia packages:**
   ```bash
   julia setup.jl
   ```

2. **Verify installation:**
   ```bash
   julia install_verification.jl
   ```

### Option 3: Platform-Specific Scripts

**On Unix/Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows:**
```cmd
setup.bat
```

## 📋 Prerequisites

### Julia Installation

BeliefSim requires Julia 1.6 or later. If you don't have Julia installed:

1. **Download Julia**: Visit [julialang.org/downloads](https://julialang.org/downloads/)

2. **Install Julia**: Follow the platform-specific instructions

3. **Add to PATH**: Ensure `julia` command is available in your terminal

4. **Verify installation**: 
   ```bash
   julia --version
   ```

### System Requirements

- **RAM**: Minimum 4GB (8GB+ recommended for large simulations)
- **Storage**: ~2GB for Julia + packages
- **OS**: Windows 7+, macOS 10.9+, or Linux (64-bit)

## 📦 Package Dependencies

BeliefSim automatically installs these packages:

### Essential Packages
- `DifferentialEquations.jl` - SDE solver engine
- `Distributions.jl` - Random distributions
- `Graphs.jl` - Network generation  
- `CSV.jl` - Data export/import
- `DataFrames.jl` - Data manipulation
- `Plots.jl` - Visualization
- `StatsBase.jl` - Statistical functions

### Optional Packages
- `GraphPlot.jl` - Network visualization
- `NetworkLayout.jl` - Graph layouts
- `Colors.jl` - Color management
- `Distances.jl` - Distance metrics

### Built-in Packages (No installation needed)
- `LinearAlgebra` - Matrix operations
- `Statistics` - Basic statistics
- `Random` - Random number generation
- `SparseArrays` - Sparse matrices

## 🔧 Installation Options

### Full Installation (Default)
Installs all packages including optional visualization tools:
```julia
julia install.jl
```

### Minimal Installation
Install only essential packages for basic functionality:
```julia
julia -e 'include("setup.jl"); minimal_install()'
```

### Development Installation
Install with additional development tools:
```julia
julia -e 'include("setup.jl"); dev_install()'
```

## ✅ Verification

After installation, verify everything works:

```bash
julia install_verification.jl
```

This will test:
- ✅ Package loading
- ✅ BeliefSim modules  
- ✅ Basic simulation
- ✅ Visualization
- ✅ File I/O operations

## 🎯 Quick Start After Installation

Once installation is complete:

### Interactive Demo
```bash
julia demo_example.jl
```

### Quick Launcher
```bash
julia run.jl
```

### Basic Simulation
```bash
julia bs.jl
```

### Verify Installation
```bash
julia install_verification.jl
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "Julia not found"
**Problem**: `julia` command not recognized
**Solution**: 
- Add Julia to your system PATH
- On Windows: Use Julia installer's "Add to PATH" option
- On Unix: Add Julia bin directory to `~/.bashrc` or `~/.zshrc`

#### 2. Package Installation Fails
**Problem**: Network or package manager issues
**Solutions**:
```julia
# Update package registry
using Pkg; Pkg.Registry.update()

# Clear package cache  
Pkg.gc()

# Reinstall problematic package
Pkg.add("PackageName")
```

#### 3. "LoadError: Module not found"
**Problem**: BeliefSim modules not loading
**Solutions**:
- Ensure you're in the BeliefSim directory
- Check that `src/` folder contains all module files
- Verify file permissions

#### 4. Plot/Visualization Issues
**Problem**: Plots don't display or save
**Solutions**:
```julia
# Use non-interactive backend
ENV["GKSwstype"] = "100"
using Plots; plot([1,2,3])

# Or install specific backend
using Pkg; Pkg.add("GR")  # or "PlotlyJS", "PyPlot"
```

#### 5. Permission Errors
**Problem**: Cannot write to output directory
**Solutions**:
- Run Julia with appropriate permissions
- Check directory write permissions
- Create output directory manually: `mkdir output`

### Advanced Troubleshooting

#### Check Package Status
```julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.status()  # Show installed packages
```

#### Reinstall All Packages
```julia
using Pkg
Pkg.activate(@__DIR__)
rm("Manifest.toml"; force=true)  # Remove lock file
Pkg.instantiate()  # Reinstall from Project.toml
```

#### Test Individual Components
```julia
# Test SDE solver
using DifferentialEquations
prob = ODEProblem((u,p,t) -> u, 1.0, (0.0, 1.0))
solve(prob)

# Test plotting
using Plots
plot([1,2,3], [1,4,2])
```

### Platform-Specific Issues

#### Windows
- **Long Path Issues**: Use Julia 1.7+ or enable long paths in Windows
- **Antivirus**: Add Julia directory to antivirus exceptions
- **PowerShell**: May need to set execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned`

#### macOS
- **Permissions**: Use `sudo` if needed for global installations
- **Xcode Tools**: Install command line tools: `xcode-select --install`

#### Linux
- **Missing Libraries**: Install system dependencies:
  ```bash
  # Ubuntu/Debian
  sudo apt update && sudo apt install build-essential

  # CentOS/RHEL  
  sudo yum install gcc gcc-c++ make
  ```

## 📊 Installation Verification Checklist

After running the installer, verify these components work:

- [ ] **Julia Environment**: `julia --version` shows 1.6+
- [ ] **Package Loading**: No errors when loading packages
- [ ] **BeliefSim Modules**: `src/Kernel.jl`, `src/Metrics.jl`, `src/Viz.jl` load successfully
- [ ] **Basic Simulation**: Can run simple belief dynamics simulation
- [ ] **Network Generation**: Can create different network topologies
- [ ] **Data Analysis**: Can compute consensus and polarization metrics  
- [ ] **Visualization**: Can create and save plots
- [ ] **File Operations**: Can read/write CSV files

## 🔄 Updating BeliefSim

To update packages to latest versions:

```bash
julia update.jl
```

Or manually:
```julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.update()
```

## 🆘 Getting Help

If you encounter issues:

1. **Check this guide**: Look for your specific error above
2. **Run diagnostics**: `julia install_verification.jl`
3. **Check Julia version**: Ensure you have 1.6+
4. **Update packages**: `julia update.jl`
5. **GitHub Issues**: Open an issue with your error message
6. **Julia Discourse**: Ask on [discourse.julialang.org](https://discourse.julialang.org)

## 📁 Installation Files Created

After successful installation, you'll have:

```
BeliefSim/
├── Project.toml              # Package dependencies
├── Manifest.toml            # Exact package versions  
├── install.jl               # Main installer
├── setup.jl                 # Package installer
├── install_verification.jl  # Installation tester
├── run.jl                   # Quick launcher
├── update.jl               # Package updater
├── demo_example.jl         # Interactive demo
├── src/                    # BeliefSim modules
├── scripts/               # Analysis scripts  
└── output/                # Generated results
```

## 🌊 Ready to Explore!

Once installation is complete:

1. **Start with the demo**: `julia demo_example.jl`
2. **Read the documentation**: `README.md`  
3. **Run your first simulation**: `julia bs.jl`
4. **Explore advanced features**: `julia scripts/advanced_analysis.jl`

Welcome to BeliefSim! 🎉
