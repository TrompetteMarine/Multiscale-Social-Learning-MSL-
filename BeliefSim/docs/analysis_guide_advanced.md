# Advanced Analysis Module for BeliefSim

## 🎯 Overview

The Advanced Analysis module extends BeliefSim with comprehensive tools for studying the qualitative behavior of the multi-scale social learning model. It provides:

- **Phase Diagram Analysis**: 2D parameter space exploration
- **Rigorous Bifurcation Analysis**: Continuation methods and branch tracking
- **Basin of Attraction Mapping**: Visualization of attractor basins
- **Monte Carlo Exploration**: Large-scale parameter sensitivity analysis
- **Advanced Visualizations**: 3D landscapes, correlation matrices, critical manifolds

## 📊 Key Features

### 1. Phase Diagrams
Explore how model behavior changes across two-parameter spaces:
- Automatic regime classification
- Stability analysis
- Number of equilibria tracking
- Consensus/polarization landscapes

### 2. Bifurcation Analysis
Rigorous detection and classification of bifurcations:
- Saddle-node bifurcations
- Pitchfork bifurcations (supercritical/subcritical)
- Hopf bifurcations
- Period-doubling cascades
- Continuation methods for tracking unstable branches
- Codimension-two point detection

### 3. Basin of Attraction Analysis
Map the basins of different attractors:
- Grid-based initial condition sampling
- Attractor classification (fixed points, limit cycles)
- Basin boundary detection
- Fractal dimension estimation
- Basin size quantification

### 4. Monte Carlo Parameter Exploration
Statistical analysis of parameter impacts:
- Latin Hypercube Sampling
- Sobol sequences for low-discrepancy sampling
- Sensitivity analysis
- Critical region identification
- Parameter importance ranking

## 🚀 Quick Start

### Installation

1. Ensure BeliefSim is installed and working
2. Run the setup script:
```bash
julia setup_advanced_analysis.jl
```

### Basic Usage

```julia
# Load modules
include("src/BeliefSim.jl")
include("src/advanced_analysis.jl")
include("src/integration_patch.jl")

using .BeliefSim, .AdvancedAnalysis, .IntegrationPatch

# Create base parameters
base_params = MSLParams(
    N = 50,
    T = 30.0,
    cognitive = CognitiveParams(α = 0.5, λ = 1.0)
)

# Integrate with advanced analysis
integrate_advanced_analysis(base_params)

# Run phase diagram
phase_params = PhaseDiagramParams(
    param1_name = :α,
    param1_range = (0.1, 2.0),
    param1_steps = 20,
    param2_name = :λ,
    param2_range = (0.1, 2.0),
    param2_steps = 20,
    base_params = base_params
)

phase_data = run_phase_diagram(phase_params)
plot_phase_diagram(phase_data)
```

## 📈 Analysis Types

### Phase Diagram Analysis

```julia
phase_params = PhaseDiagramParams(
    param1_name = :α,              # First parameter to vary
    param1_range = (0.1, 2.0),     # Range for first parameter
    param1_steps = 25,             # Resolution
    param2_name = :λ,              # Second parameter
    param2_range = (0.1, 2.0),     
    param2_steps = 25,
    base_params = base_params,     # Your MSLParams
    n_realizations = 3             # Ensemble size per point
)

results = run_phase_diagram(phase_params)
```

**Output includes:**
- Number of equilibria matrix
- Stability index matrix
- Regime classification map
- Consensus/polarization landscapes
- Lyapunov exponent map

### Bifurcation Analysis

```julia
bifurc_params = BifurcationParams(
    param_name = :α,               # Parameter to vary
    param_range = (0.0, 2.5),      # Range
    n_points = 150,                # Resolution
    stability_check = true,        # Check stability
    track_unstable = true,         # Track unstable branches
    max_period = 10                # Max period for cycle detection
)

bifurc_data = run_bifurcation_analysis(bifurc_params, base_params)
```

**Output includes:**
- Stable and unstable solution branches
- Bifurcation points and types
- Period-doubling cascade points
- Codimension-two points

### Basin of Attraction Analysis

```julia
basin_params = BasinAnalysisParams(
    x_range = (-3.0, 3.0),         # Belief range
    y_range = (-3.0, 3.0),         # Reference point range
    grid_resolution = 100,          # Grid points per dimension
    integration_time = 100.0,       # Time to integrate
    convergence_threshold = 0.01    # Convergence criterion
)

basin_data = analyze_basins_of_attraction(basin_params, params)
```

**Output includes:**
- Basin map (which initial conditions lead to which attractor)
- Attractor positions and types
- Basin sizes
- Basin boundary fractal dimension

### Monte Carlo Exploration

```julia
# Define parameter ranges
param_ranges = Dict(
    :α => (0.1, 2.0),
    :λ => (0.5, 1.5),
    :σ => (0.1, 0.5),
    :δm => (0.05, 0.2)
)

# Run Monte Carlo
mc_results = monte_carlo_phase_exploration(
    1000,                          # Number of samples
    param_ranges,                  # Parameter ranges
    base_params                    # Base parameters
)
```

**Output includes:**
- Full dataset (DataFrame)
- Parameter sensitivity analysis
- Critical regions identification
- Parameter importance ranking

## 📊 Visualization Functions

### Phase Diagram Visualization
```julia
phase_plot = plot_phase_diagram(phase_data)
```
Creates a 6-panel plot showing:
- Number of equilibria
- Stability index
- Consensus strength
- Polarization index
- Dynamical regimes
- Lyapunov exponents

### Bifurcation Diagram
```julia
bifurc_plot = plot_bifurcation_2d(bifurc_data)
```
Shows:
- Stable branches (solid lines)
- Unstable branches (dashed lines)
- Bifurcation points (markers)

### Basin Portrait
```julia
basin_plot = plot_basin_portrait(basin_data)
```
Displays:
- Color-coded basin map
- Attractor locations
- Basin size pie chart
- Fractal dimension info

## 🔍 Advanced Features

### Custom Sampling Strategies

The module supports multiple sampling strategies for parameter exploration:

```julia
# Latin Hypercube Sampling (default)
samples = sample_parameters(param_ranges, n_samples, method=:latin_hypercube)

# Random uniform sampling
samples = sample_parameters(param_ranges, n_samples, method=:random)

# Regular grid
samples = sample_parameters(param_ranges, n_samples, method=:grid)

# Quasi-random Sobol sequence
samples = sample_parameters(param_ranges, n_samples, method=:sobol)
```

### Stability Analysis

Detailed stability analysis using:
- Eigenvalue analysis of Jacobian
- Lyapunov exponent estimation
- Convergence rate measurement
- Oscillation detection

### Regime Classification

Automatic classification into:
- **Stable Consensus**: Single stable equilibrium
- **Bistable**: Two stable equilibria
- **Multistable**: Multiple stable equilibria
- **Oscillatory**: Persistent oscillations
- **Chaotic**: Positive Lyapunov exponent
- **Transient**: Non-converged dynamics

## 📁 Output Files

The analysis generates various output files:

```
output/advanced_analysis/
├── phase_diagram.png           # 2D phase diagram
├── phase_diagram_data.csv      # Raw phase diagram data
├── bifurcation_diagram.png     # Bifurcation structure
├── basins_alpha_*.png         # Basin portraits
├── 3d_phase_landscape.png     # 3D parameter landscape
├── parameter_correlations.png  # Sensitivity heatmap
├── critical_manifold.png      # Critical parameter regions
├── monte_carlo_results.csv    # Full MC dataset
└── analysis_summary.txt       # Text summary
```

## 🛠️ Customization

### Adding New Parameters

To analyze additional parameters:

1. Add to parameter ranges:
```julia
param_ranges[:new_param] = (min_val, max_val)
```

2. Update the parameter update function in `integration_patch.jl`

### Custom Order Parameters

Define new order parameters in `integration_patch.jl`:

```julia
function compute_custom_order_parameter(trajectories)
    # Your analysis here
    return value
end
```

### Custom Regime Classification

Modify `classify_regime()` in `advanced_analysis.jl` to add new regime types.

## 🐛 Troubleshooting

### Memory Issues
- Reduce grid resolution for phase diagrams
- Use smaller `grid_resolution` for basin analysis
- Decrease `n_realizations` for faster computation

### Convergence Problems
- Increase `integration_time` for basin analysis
- Adjust `convergence_threshold`
- Check parameter ranges are reasonable

### Visualization Issues
- Ensure all plotting packages are installed
- Check output directory permissions
- Verify data dimensions match

## 📚 References

The methods implemented are based on:

1. **Bifurcation Analysis**: 
   - Kuznetsov, Y. A. (2004). *Elements of Applied Bifurcation Theory*

2. **Basin Analysis**:
   - Nusse, H. E., & Yorke, J. A. (1997). *Dynamics: Numerical Explorations*

3. **Monte Carlo Methods**:
   - McKay, M. D., et al. (1979). "A comparison of three methods for selecting values of input variables"

4. **Phase Diagrams**:
   - Strogatz, S. H. (2014). *Nonlinear Dynamics and Chaos*

## 💡 Tips

1. **Start Small**: Begin with coarse grids and increase resolution
2. **Parameter Ranges**: Keep initial ranges conservative
3. **Parallel Processing**: Use Julia's parallel features for large analyses
4. **Save Intermediates**: Export data frequently for large computations
5. **Visualization**: Customize plots using Plots.jl attributes

## 📧 Support

For issues or questions:
- Check the example scripts in `scripts/advanced/`
- Review the main BeliefSim documentation
- Examine the source code comments

---

*Advanced Analysis Module v1.0 - Exploring the parameter landscape of belief dynamics*