# BeliefSim - Multi-Scale Social Learning Simulator

[![Julia 1.6+](https://img.shields.io/badge/Julia-1.6+-blue.svg)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of **"Threshold Attention and Polarised Equilibria in Social Learning"** by Gabriel Bontemps (2025), working paper.

## 🌊 Overview

BeliefSim is a comprehensive simulation and analysis framework for studying belief dynamics in populations of cognitively bounded agents. The system provides multiple layers of analysis depth, from quick exploratory simulations to rigorous mathematical characterization of the model's qualitative behaviors.

### Core Features

- **5D Agent State**: beliefs (x), reference points (r), memory (m), deliberation (w), cognitive thresholds (Θ)
- **Jump-Diffusion Dynamics**: Continuous belief evolution with discrete resets when cognitive tension exceeds thresholds
- **Multi-Scale Analysis**: Micro (individual), meso (community), and macro (population) level metrics
- **Regime Classification**: Equilibrium, Meso-Buffered, Broadcast, and Cascade states
- **Modular Architecture**: Clean separation between core simulation and optional analysis extensions
- **Advanced Analysis Suite**: Phase diagrams, bifurcation analysis, basin mapping, Monte Carlo exploration

## 🏗️ Project Architecture

```
BeliefSim/
│
├── 📁 src/                          # Core simulation engine
│   ├── BeliefSim.jl                # Main module coordinator
│   ├── agents.jl                   # Agent dynamics (5D state, cognitive functions)
│   ├── networks.jl                 # Network topologies (small-world, scale-free, etc.)
│   ├── simulation.jl               # Jump-diffusion SDE solver
│   ├── metrics.jl                  # Analysis metrics (consensus, polarization, shifts)
│   ├── visualization.jl            # Basic plotting utilities
│   ├── advanced_analysis/          # Advanced analysis submodules
│   ├── advanced_analysis.jl        # Module entry point
│   ├── integration_patch.jl        # Bridge between simulation and analysis
│   └── ensemble.jl                 # Ensemble simulation management
│
├── 📁 examples/                     # Usage examples
│   ├── basic_simulation.jl         # Quick start example
│   ├── paper_reproduction.jl       # Reproduce paper figures
│   └── advanced/                   # Advanced analysis demos
│       ├── phase_exploration.jl    # Phase diagram examples
│       ├── bifurcation_study.jl    # Detailed bifurcation analysis
│       └── basin_analysis.jl       # Basin of attraction studies
│
├── 📁 output/                       # Generated results
│   ├── advanced_analysis/          # Advanced analysis outputs
│   ├── phase_diagrams/             # Phase diagram results
│   ├── bifurcations/               # Bifurcation diagrams
│   └── basins/                     # Basin portraits
│
├── 📁 docs/                         # Documentation
│   ├── analysis_guide.md           # Analysis methodology guide
│   └── analysis_guide_advanced.md  # Advanced analysis guide
│
├── Project.toml                    # Package dependencies
├── setup_advanced_analysis.jl      # Setup script for advanced features
└── README.md                       # This file
```

## 🚀 Installation

### Basic Installation
```bash
# Clone repository
git clone https://github.com/TrompetteMarine/BeliefSim.git
cd BeliefSim

# Install core dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Advanced Analysis Installation
```bash
# Install additional dependencies for advanced analysis
julia setup_advanced_analysis.jl
```

## 🎯 Operationalizing the System: Layers of Analysis Depth

BeliefSim is designed as a **multi-layer analysis system** where researchers can choose the appropriate depth of investigation based on their needs. Each layer builds upon the previous, providing progressively deeper insights into the model's behavior:

### **Layer 1: Exploratory Simulation** (Minutes)
Quick single runs to understand basic dynamics and test parameters:
```julia
using BeliefSim
params = MSLParams(N=50, T=30.0, cognitive=CognitiveParams(α=0.5))
t_vec, trajectories, analysis = run_msl_simulation(params)
println("Regime: ", analysis[:regime])
```
*Use for:* Initial exploration, parameter testing, hypothesis generation

### **Layer 2: Statistical Analysis** (Hours)
Ensemble simulations with basic statistical characterization:
```julia
# Run ensemble with different seeds
results = []
for seed in 1:100
    t_vec, traj, ana = run_msl_simulation(params; seed=seed)
    push!(results, ana)
end
# Analyze variance, convergence rates, regime probabilities
```
*Use for:* Robust conclusions, variance estimation, publication figures

### **Layer 3: Parameter Space Mapping** (Hours to Days)
Systematic exploration of parameter space with phase diagrams:
```julia
using AdvancedAnalysis
phase_params = PhaseDiagramParams(
    param1_name=:α, param1_range=(0.1, 2.0),
    param2_name=:λ, param2_range=(0.1, 2.0),
    param1_steps=25, param2_steps=25
)
phase_data = run_phase_diagram(phase_params)
```
*Use for:* Identifying critical regions, regime boundaries, parameter interactions

### **Layer 4: Bifurcation & Stability Analysis** (Days)
Rigorous mathematical characterization using continuation methods:
```julia
bifurc_params = BifurcationParams(
    param_name=:α, param_range=(0.0, 2.5),
    n_points=200, track_unstable=true
)
bifurc_data = run_bifurcation_analysis(bifurc_params, base_params)
```
*Use for:* Precise critical points, stability boundaries, theoretical validation

### **Layer 5: Global Dynamics Characterization** (Days to Weeks)
Complete analysis including basins of attraction and Monte Carlo exploration:
```julia
# Basin analysis for multiple parameter values
basin_params = BasinAnalysisParams(grid_resolution=200)
basin_data = analyze_basins_of_attraction(basin_params, params)

# Large-scale Monte Carlo
mc_results = monte_carlo_phase_exploration(5000, param_ranges, base_params)
```
*Use for:* Complete characterization, sensitivity analysis, prediction of rare events

### **Choosing the Right Layer**
- **Time-constrained research**: Start with Layer 1-2 for quick insights
- **Publication preparation**: Use Layer 2-3 for robust, reproducible results
- **Theoretical investigation**: Employ Layer 4-5 for mathematical rigor
- **Policy applications**: Combine Layer 3-5 for comprehensive understanding

## 📊 Quick Start Examples

The example scripts use a dedicated project to ensure all plotting and data
dependencies are available. Activate this environment before running any
example:

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()' # run once to install deps
julia --project=examples examples/basic_simulation.jl
```

### Basic Simulation
```julia
using BeliefSim

# Create parameters
params = MSLParams(
    N = 100,          # Number of agents
    T = 50.0,         # Time horizon
    cognitive = CognitiveParams(
        λ = 1.0,      # Mean reversion
        α = 0.5,      # Social influence
        σ = 0.3       # Noise level
    )
)

# Run simulation
t_vec, trajectories, analysis = run_msl_simulation(params)

# Check results
println("Final consensus: ", analysis[:final_consensus])
println("Detected regime: ", analysis[:regime])
```

### Advanced Analysis
```julia
# Load advanced modules
include("src/advanced_analysis.jl")
include("src/integration_patch.jl")
using .AdvancedAnalysis, .IntegrationPatch

# Integrate modules
integrate_advanced_analysis(params)

# Run phase diagram
phase_data = run_phase_diagram(PhaseDiagramParams(base_params=params))
plot_phase_diagram(phase_data)
```

### Reproduce Paper Results
Activate the examples project and run the script:

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()' # run once
julia --project=examples examples/paper_reproduction.jl
```

## 📈 Key Results & Insights

### 1. **Supercritical Pitchfork Bifurcation**
- Critical peer influence α* where consensus breaks down
- Transition from unimodal to bimodal belief distribution
- Precise critical point: α* ≈ 0.85 (for default parameters)

### 2. **Multi-Scale Dynamics**
- **Micro**: Individual belief fluctuations
- **Meso**: Community-level buffering
- **Macro**: Population-wide cascades

### 3. **Behavioral Regimes**
| Regime | Characteristics | Parameter Range |
|--------|----------------|-----------------|
| **Equilibrium** | Stable consensus | α < 0.3 |
| **Meso-Buffered** | Local perturbations absorbed | 0.3 < α < 0.8 |
| **Broadcast** | Top-down influence | High λ, low α |
| **Cascade** | Full-scale propagation | α > 1.0 |

### 4. **Basin Structure**
- Multiple attractors emerge for α > α*
- Fractal basin boundaries indicate sensitive dependence
- Basin sizes determine probability of different outcomes

## 🛠️ Advanced Features

### Phase Diagrams
```julia
phase_params = PhaseDiagramParams(
    param1_name = :α,
    param1_range = (0.1, 2.0),
    param1_steps = 50,  # High resolution
    param2_name = :λ,
    param2_range = (0.1, 2.0),
    param2_steps = 50
)
```

### Monte Carlo Analysis
```julia
param_ranges = Dict(
    :α => (0.1, 2.0),
    :λ => (0.5, 1.5),
    :σ => (0.1, 0.5)
)
mc_results = monte_carlo_phase_exploration(1000, param_ranges, base_params)
```

### Custom Network Topologies
```julia
params = MSLParams(
    network_type = :scale_free,  # or :small_world, :random, :fully_connected
    network_params = Dict(:m => 3)
)
```

## 📊 Output Files

The system generates comprehensive outputs organized by analysis type:

```
output/
├── basic_simulations/       # Individual run results
├── ensemble_results/        # Statistical analyses
├── phase_diagrams/         # 2D parameter landscapes
├── bifurcations/           # Bifurcation diagrams
├── basins/                 # Basin portraits
├── monte_carlo/            # MC exploration results
└── summary_reports/        # Aggregated findings
```

## 🔬 Computational Requirements

| Analysis Layer | Time | Memory | Cores |
|---------------|------|--------|-------|
| Layer 1 (Exploratory) | < 1 min | < 1 GB | 1 |
| Layer 2 (Statistical) | 10-60 min | 2-4 GB | 1-4 |
| Layer 3 (Phase Mapping) | 2-12 hours | 4-8 GB | 4-8 |
| Layer 4 (Bifurcation) | 6-24 hours | 8-16 GB | 8-16 |
| Layer 5 (Global) | 1-7 days | 16-32 GB | 16-32 |

## 📖 Citation

If you use BeliefSim in your research, please cite:

```bibtex
@article{bontemps2025multiscale,
  title={Multi-Scale Social Learning: From Individual Bounded Rationality to Collective Dynamics},
  author={Bontemps, Gabriel},
  journal={},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Priorities
- [ ] GPU acceleration for large-scale simulations
- [ ] Interactive web interface
- [ ] Real-time visualization
- [ ] Machine learning integration for pattern detection
- [ ] Empirical data calibration tools

## 📚 Documentation

- **[Analysis Guide](docs/analysis_guide.md)**: Methodology and workflow
- **[Advanced Analysis Guide](docs/analysis_guide_advanced.md)**: Phase diagrams, bifurcations, and basin studies
- **[Examples](examples/)**: Working code examples
- **[Paper](link-to-paper)**: Theoretical background

## 🆘 Support

For help and support:
- **Issues**: [GitHub Issues](https://github.com/yourusername/BeliefSim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/BeliefSim/discussions)
- **Email**: gabriel.bontemps@unice.fr

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Gabriel Bontemps for the theoretical framework
- Julia community for excellent scientific computing tools
- Contributors and users for feedback and improvements

---

<div align="center">

**BeliefSim** - *Understanding how individual cognitive limitations shape collective dynamics*

[Documentation](docs/) • [Examples](examples/) • [Paper](link) • [Cite](#citation)

</div>