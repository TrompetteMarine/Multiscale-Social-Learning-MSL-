# BeliefSim Analysis Guide: From Quick Exploration to Deep Characterization

## Overview

This guide explains how to operationalize BeliefSim's multi-layered analysis system, helping researchers choose the appropriate depth of investigation for their specific research questions and computational constraints.

## 🎯 Analysis Philosophy

BeliefSim follows a **progressive deepening** approach where each analysis layer provides increasingly detailed insights:

```
Layer 1: Exploratory     →  Quick insights, hypothesis generation
   ↓
Layer 2: Statistical     →  Robust conclusions, variance estimation  
   ↓
Layer 3: Parametric      →  Phase diagrams, critical regions
   ↓
Layer 4: Mathematical    →  Bifurcations, stability analysis
   ↓
Layer 5: Global          →  Complete dynamics, rare events
```

---

## 📊 Layer 1: Exploratory Simulation

### Purpose
Quick exploration to understand basic dynamics and test hypotheses.

### Time Required
Minutes to hours

### Use Cases
- Initial parameter exploration
- Hypothesis generation
- Teaching demonstrations
- Quick visualizations

### Implementation

```julia
using BeliefSim
using Plots

# Quick parameter scan
α_values = [0.2, 0.5, 0.8, 1.2]
results = Dict()

for α in α_values
    params = MSLParams(
        N = 50,  # Small network for speed
        T = 30.0,
        cognitive = CognitiveParams(α = α)
    )
    
    t_vec, traj, ana = run_msl_simulation(params)
    results[α] = ana
    
    # Quick visualization
    p = plot(t_vec, mean([traj[:beliefs][i] for i in 1:params.N]), 
             label="α=$α", title="Mean Belief Evolution")
    display(p)
end

# Compare regimes
for (α, ana) in results
    println("α = $α: Regime = $(ana[:regime]), Consensus = $(round(ana[:final_consensus], digits=3))")
end
```

### Key Outputs
- Belief trajectories
- Final consensus/polarization
- Regime classification
- Basic visualizations

### Decision Points
- If interesting behavior observed → proceed to Layer 2
- If parameters need refinement → iterate at Layer 1
- If specific phenomenon identified → jump to relevant deeper layer

---

## 📈 Layer 2: Statistical Analysis

### Purpose
Establish robust statistical conclusions through ensemble simulations.

### Time Required
Hours

### Use Cases
- Publication-quality results
- Confidence intervals
- Variance analysis
- Reproducible findings

### Implementation

```julia
using Statistics, DataFrames, CSV

# Ensemble configuration
n_runs = 100
params = MSLParams(N = 100, T = 50.0)

# Run ensemble
ensemble_results = []
@time for seed in 1:n_runs
    t_vec, traj, ana = run_msl_simulation(params; seed=seed)
    
    push!(ensemble_results, (
        seed = seed,
        consensus = ana[:final_consensus],
        polarization = ana[:final_polarization],
        regime = ana[:regime],
        convergence_time = ana[:convergence_time]
    ))
end

# Statistical analysis
df = DataFrame(ensemble_results)

# Compute statistics
stats = Dict(
    :consensus_mean => mean(df.consensus),
    :consensus_std => std(df.consensus),
    :consensus_ci => quantile(df.consensus, [0.025, 0.975]),
    :regime_distribution => proportionmap(df.regime),
    :convergence_mean => mean(skipmissing(df.convergence_time))
)

# Save results
CSV.write("output/ensemble_results.csv", df)

# Visualization
using StatsPlots
@df df begin
    p1 = histogram(:consensus, bins=20, label="Consensus Distribution")
    p2 = bar(collect(keys(stats[:regime_distribution])), 
             collect(values(stats[:regime_distribution])),
             label="Regime Frequency")
    plot(p1, p2, layout=(2,1))
end
```

### Statistical Tests

```julia
# Test for parameter significance
using HypothesisTests

# Compare two parameter settings
results_low_α = [run_msl_simulation(MSLParams(cognitive=CognitiveParams(α=0.3)); seed=s)[:analysis][:final_consensus] for s in 1:50]
results_high_α = [run_msl_simulation(MSLParams(cognitive=CognitiveParams(α=1.0)); seed=s)[:analysis][:final_consensus] for s in 1:50]

# Mann-Whitney U test
test_result = MannWhitneyUTest(results_low_α, results_high_α)
println("p-value: ", pvalue(test_result))
```

### Key Outputs
- Distribution statistics
- Confidence intervals
- Regime probabilities
- Ensemble trajectories

---

## 🗺️ Layer 3: Parameter Space Mapping

### Purpose
Systematic exploration of parameter interactions and phase boundaries.

### Time Required
Hours to days

### Use Cases
- Identifying critical parameters
- Finding phase transitions
- Parameter optimization
- Regime boundaries

### Implementation

```julia
using AdvancedAnalysis

# Define parameter space
phase_params = PhaseDiagramParams(
    param1_name = :α,
    param1_range = (0.1, 2.0),
    param1_steps = 30,
    param2_name = :λ,
    param2_range = (0.1, 2.0),
    param2_steps = 30,
    base_params = MSLParams(N=75, T=40.0),
    n_realizations = 5  # Multiple runs per point
)

# Generate phase diagram
@time phase_data = run_phase_diagram(phase_params)

# Analyze critical boundaries
critical_lines = find_phase_boundaries(phase_data)

# Create detailed visualization
using Plots.PlotMeasures
p = plot_phase_diagram(phase_data)
plot!(p, size=(1200, 800), margin=5mm)
savefig("output/phase_diagram_detailed.png")

# Export for further analysis
save_phase_data("output/phase_data.jld2", phase_data)
```

### Advanced Mapping Techniques

```julia
# Adaptive sampling for efficiency
function adaptive_phase_sampling(param_ranges, initial_resolution=10)
    # Start with coarse grid
    coarse_data = run_phase_diagram(
        PhaseDiagramParams(param1_steps=initial_resolution, 
                          param2_steps=initial_resolution)
    )
    
    # Identify regions needing refinement
    gradient_map = compute_gradients(coarse_data)
    high_gradient_regions = find_high_gradient_regions(gradient_map)
    
    # Refine interesting regions
    for region in high_gradient_regions
        fine_params = create_refined_params(region, resolution=50)
        fine_data = run_phase_diagram(fine_params)
        merge_phase_data!(coarse_data, fine_data)
    end
    
    return coarse_data
end
```

### Key Outputs
- 2D/3D phase diagrams
- Critical boundaries
- Regime maps
- Parameter sensitivity maps

---

## 🔬 Layer 4: Mathematical Characterization

### Purpose
Rigorous mathematical analysis of bifurcations and stability.

### Time Required
Days

### Use Cases
- Theoretical validation
- Precise critical points
- Stability boundaries
- Publication in theory journals

### Implementation

```julia
# Detailed bifurcation analysis
bifurc_params = BifurcationParams(
    param_name = :α,
    param_range = (0.0, 2.5),
    n_points = 200,
    continuation_steps = 500,
    stability_check = true,
    track_unstable = true,
    tolerance = 1e-8
)

# Run analysis with continuation
bifurc_data = run_bifurcation_analysis(bifurc_params, base_params)

# Extract critical points
critical_points = extract_bifurcation_points(bifurc_data)

# Normal form analysis near bifurcations
for cp in critical_points
    normal_form = compute_normal_form(cp, base_params)
    println("Bifurcation at α = $(cp.value):")
    println("  Type: $(cp.type)")
    println("  Normal form: $(normal_form.equation)")
    println("  Criticality: $(normal_form.criticality)")
end

# Floquet analysis for periodic orbits
if has_periodic_orbits(bifurc_data)
    floquet_multipliers = compute_floquet_multipliers(bifurc_data)
    plot_floquet_diagram(floquet_multipliers)
end
```

### Advanced Stability Analysis

```julia
# Lyapunov spectrum computation
function compute_lyapunov_spectrum(params, n_exponents=5)
    # Initialize orthonormal perturbations
    perturbations = initialize_perturbations(n_exponents)
    
    # Evolve system with Gram-Schmidt orthogonalization
    lyapunov_exponents = evolve_with_orthogonalization(
        params, perturbations, t_final=1000.0
    )
    
    return sort(lyapunov_exponents, rev=true)
end

# Linear stability analysis
function analyze_linear_stability(fixed_point, params)
    # Compute Jacobian
    J = compute_jacobian(fixed_point, params)
    
    # Eigenvalue analysis
    eigenvals = eigvals(J)
    eigenvecs = eigvecs(J)
    
    # Classify stability
    stability_type = classify_stability(eigenvals)
    
    return Dict(
        :eigenvalues => eigenvals,
        :eigenvectors => eigenvecs,
        :type => stability_type,
        :stable => all(real.(eigenvals) .< 0)
    )
end
```

### Key Outputs
- Bifurcation diagrams with stable/unstable branches
- Critical point classifications
- Normal form coefficients
- Lyapunov spectra
- Floquet multipliers

---

## 🌍 Layer 5: Global Dynamics

### Purpose
Complete characterization including rare events and global structure.

### Time Required
Days to weeks

### Use Cases
- Complete system understanding
- Rare event prediction
- Policy recommendations
- Long-term behavior

### Implementation

```julia
# Comprehensive global analysis
function global_dynamics_analysis(base_params)
    results = Dict()
    
    # 1. Basin of attraction analysis
    println("Analyzing basins of attraction...")
    basin_params = BasinAnalysisParams(
        grid_resolution = 200,
        integration_time = 200.0
    )
    results[:basins] = analyze_basins_of_attraction(basin_params, base_params)
    
    # 2. Large-scale Monte Carlo
    println("Running Monte Carlo exploration...")
    param_ranges = Dict(
        :α => (0.1, 2.0),
        :λ => (0.5, 1.5),
        :σ => (0.1, 0.5),
        :δm => (0.05, 0.2),
        :ηw => (0.05, 0.2)
    )
    results[:monte_carlo] = monte_carlo_phase_exploration(
        5000, param_ranges, base_params
    )
    
    # 3. Invariant measure estimation
    println("Estimating invariant measures...")
    results[:invariant_measure] = estimate_invariant_measure(base_params)
    
    # 4. Transition path analysis
    println("Computing transition paths...")
    results[:transitions] = compute_transition_paths(
        results[:basins][:attractors], base_params
    )
    
    # 5. Extreme event analysis
    println("Analyzing extreme events...")
    results[:extreme_events] = analyze_extreme_events(base_params, n_runs=1000)
    
    return results
end
```

### Rare Event Analysis

```julia
# Importance sampling for rare events
function rare_event_probability(event_condition, params; n_samples=10000)
    # Design importance sampling distribution
    importance_dist = design_importance_distribution(event_condition, params)
    
    # Sample and compute weights
    samples = []
    weights = []
    
    for i in 1:n_samples
        # Sample from importance distribution
        trajectory = sample_trajectory(importance_dist, params)
        
        # Compute likelihood ratio
        weight = compute_likelihood_ratio(trajectory, params, importance_dist)
        
        push!(samples, trajectory)
        push!(weights, weight)
    end
    
    # Estimate probability
    event_indicators = [event_condition(s) for s in samples]
    probability = sum(event_indicators .* weights) / sum(weights)
    
    # Confidence interval
    ci = bootstrap_confidence_interval(event_indicators, weights)
    
    return (probability=probability, ci=ci, effective_samples=1/sum(weights.^2))
end
```

### Key Outputs
- Complete basin portraits
- Global bifurcation structure
- Rare event probabilities
- Transition pathways
- Invariant measures
- Sensitivity indices

---

## 🔄 Workflow Integration

### Recommended Analysis Pipeline

```julia
# Complete analysis workflow
function complete_analysis_pipeline(research_question)
    # Start with exploration
    preliminary = exploratory_analysis(quick_params)
    
    if interesting_behavior_found(preliminary)
        # Statistical validation
        ensemble_results = statistical_analysis(refined_params)
        
        if statistically_significant(ensemble_results)
            # Map parameter space
            phase_diagram = parameter_space_mapping(param_ranges)
            
            if critical_regions_found(phase_diagram)
                # Mathematical characterization
                bifurcation_analysis = mathematical_characterization(critical_params)
                
                if complex_dynamics_detected(bifurcation_analysis)
                    # Global analysis
                    global_results = global_dynamics_analysis(full_params)
                end
            end
        end
    end
    
    return compile_results(all_analyses)
end
```

### Computational Resource Management

```julia
# Adaptive resource allocation
function allocate_computational_resources(analysis_layer)
    if analysis_layer == 1
        return (cores=1, memory="2GB", time="1h")
    elseif analysis_layer == 2
        return (cores=4, memory="8GB", time="4h")
    elseif analysis_layer == 3
        return (cores=16, memory="16GB", time="24h")
    elseif analysis_layer == 4
        return (cores=32, memory="32GB", time="72h")
    elseif analysis_layer == 5
        return (cores=64, memory="64GB", time="168h")
    end
end
```

---

## 📋 Analysis Selection Guide

| Research Question | Recommended Layers | Key Methods |
|------------------|-------------------|-------------|
| "Does the model show consensus?" | 1-2 | Single runs, basic statistics |
| "When does polarization emerge?" | 2-3 | Ensemble, phase diagrams |
| "What causes regime transitions?" | 3-4 | Parameter mapping, bifurcation analysis |
| "How robust are the dynamics?" | 2-4 | Statistical analysis, sensitivity |
| "Can we predict extreme events?" | 4-5 | Global dynamics, rare event analysis |
| "What are all possible behaviors?" | 3-5 | Complete pipeline |

---

## 💡 Best Practices

### 1. **Start Simple**
Always begin with Layer 1 to gain intuition before deeper analysis.

### 2. **Validate Statistically**
Use Layer 2 to ensure results are robust before proceeding.

### 3. **Focus on Interesting Regions**
Use coarse Layer 3 analysis to identify regions worth detailed study.

### 4. **Document Parameters**
Keep detailed records of all parameters used at each layer.

### 5. **Save Intermediate Results**
Store results from each layer for reproducibility and comparison.

### 6. **Use Adaptive Methods**
Employ adaptive sampling and refinement to optimize computational resources.

### 7. **Visualize at Every Step**
Create visualizations at each layer to guide analysis decisions.

---

## 📚 Further Reading

- **Strogatz (2014)**: Nonlinear Dynamics and Chaos
- **Kuznetsov (2004)**: Elements of Applied Bifurcation Theory
- **Guckenheimer & Holmes (1983)**: Nonlinear Oscillations, Dynamical Systems, and Bifurcations
- **Nusse & Yorke (1997)**: Dynamics: Numerical Explorations

---

*This guide is part of the BeliefSim documentation. For specific implementation details, see the API reference.*
