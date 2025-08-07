# BeliefSim - Multi-Scale Social Learning Simulator

[![Julia 1.6+](https://img.shields.io/badge/Julia-1.6+-blue.svg)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of **"Multi-Scale Social Learning: From Individual Bounded Rationality to Collective Dynamics"** by Gabriel Bontemps (2025), published in **.

## 🌊 Overview

BeliefSim simulates the evolution of beliefs in a population of cognitively bounded agents who learn from both private signals and social interactions. The model features:

- **5D Agent State**: beliefs (x), reference points (r), memory (m), deliberation (w), cognitive thresholds (Θ)
- **Jump-Diffusion Dynamics**: Continuous belief evolution with discrete resets when cognitive tension exceeds thresholds
- **Multi-Scale Analysis**: Micro (individual), meso (community), and macro (population) level metrics
- **Regime Classification**: Equilibrium, Meso-Buffered, Broadcast, and Cascade states
- **Bifurcation Analysis**: Supercritical pitchfork transition from consensus to polarization

## 🚀 Quick Start

### Installation

```julia
# Clone the repository
git clone https://github.com/TrompetteMarine/BeliefSim.git
cd BeliefSim

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Basic Usage

```julia
using BeliefSim

# Create parameters
params = MSLParams(
    N = 100,          # Number of agents
    T = 50.0,         # Time horizon
    cognitive = CognitiveParams(
        λ = 1.0,      # Mean reversion strength
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

### Reproduce Paper Results

```bash
julia --project=. examples/paper_reproduction.jl
```

This will generate:
- Belief evolution plots
- Bifurcation diagram showing critical α*
- Multi-scale shift metrics
- Cognitive dynamics visualization

## 📊 Model Components

### Agent Dynamics (Equations 2-6)

The model implements the full jump-diffusion system from the paper:

```
dx_i = [-λ(x_i - r_i) + αm_iw_iΠ_i]dt + σdW_i - (x_i - r_i)dN_i
dr_i = φ(m_i)(x_i - r_i)dt
dm_i = δ_m[ξ(m_i,Θ_i) - m_i]dt
dw_i = η_w w_i[ψ_w(w_i) - w_i][Ψ_i - Ψ̄]dt
dΘ_i = β_Θ[Υ(m_i,w_i) - Θ_i]dt
```

Where jumps occur when cognitive tension |x_i - r_i| ≥ Θ_i.

### Network Structures

- Fully connected
- Small-world (Watts-Strogatz)
- Scale-free (Barabási-Albert)
- Random (Erdős-Rényi)

### Analysis Metrics

- **Consensus strength**: 1 - normalized disagreement
- **Polarization index**: Bimodality and group separation
- **Shift metrics**: Layer-specific change detection
- **Regime classification**: Based on multi-scale indicators

## 📁 Repository Structure

```
BeliefSim/
├── src/
│   ├── BeliefSim.jl        # Main module
│   ├── agents.jl           # Agent dynamics
│   ├── networks.jl         # Network generation
│   ├── simulation.jl       # Jump-diffusion solver
│   ├── metrics.jl          # Analysis functions
│   └── visualization.jl    # Plotting utilities
├── examples/
│   ├── paper_reproduction.jl   # Reproduce paper figures
│   └── basic_simulation.jl     # Simple example
├── scripts/
│   ├── bifurcation.jl      # Bifurcation analysis
│   └── regime_analysis.jl  # Regime classification
├── Project.toml            # Dependencies
└── README.md              # This file
```

## 🔬 Key Results

### 1. Supercritical Pitchfork Bifurcation
- Critical peer influence α* where consensus breaks down
- Transition from unimodal to bimodal belief distribution

### 2. Multi-Scale Dynamics
- Individual beliefs evolve continuously
- Communities buffer local perturbations
- Population-level cascades emerge above critical threshold

### 3. Behavioral Regimes
- **Equilibrium**: Stable consensus, no significant shifts
- **Meso-Buffered**: Local changes absorbed by communities
- **Broadcast**: Top-down influence without micro changes
- **Cascade**: Full-scale propagation across all layers

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{bontemps2024multiscale,
  title={Multi-Scale Social Learning: From Individual Bounded Rationality to Collective Dynamics},
  author={Bontemps, Gabriel},
  journal={},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues or questions:
- Open an issue on GitHub
- Check the examples folder for usage patterns
- Refer to the paper for theoretical background

---
*BeliefSim - Understanding how individual cognitive limitations shape collective dynamics*
