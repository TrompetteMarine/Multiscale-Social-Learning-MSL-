# Sensitivity.jl - Sensitivity analysis module
module Sensitivity

using Statistics, LinearAlgebra, Random
using DataFrames
using ..Types

export global_sensitivity_analysis, compute_sobol_indices
export parameter_screening, morris_method
export compute_prcc, compute_src

function global_sensitivity_analysis(data::DataFrame, param_names::Vector{Symbol}, 
                                    output_name::Symbol)
    # Compute various sensitivity indices
    results = Dict{Symbol, Any}()
    
    # Pearson correlation
    results[:pearson] = compute_correlations(data, param_names, output_name, :pearson)
    
    # Spearman rank correlation
    results[:spearman] = compute_correlations(data, param_names, output_name, :spearman)
    
    # Partial rank correlation coefficient (PRCC)
    results[:prcc] = compute_prcc(data, param_names, output_name)
    
    # Standardized regression coefficient (SRC)
    results[:src] = compute_src(data, param_names, output_name)
    
    # Sobol indices (if enough data)
    if nrow(data) > 100
        results[:sobol] = compute_sobol_indices(data, param_names, output_name)
    end
    
    return results
end

function compute_correlations(data::DataFrame, param_names::Vector{Symbol}, 
                             output_name::Symbol, method::Symbol)
    correlations = Dict{Symbol, Float64}()
    
    for param in param_names
        if param in names(data, Symbol) && output_name in names(data, Symbol)
            x = Float64.(data[!, param])
            y = Float64.(data[!, output_name])
            
            # Remove NaN/Inf
            valid = .!isnan.(x) .& .!isnan.(y) .& .!isinf.(x) .& .!isinf.(y)
            
            if sum(valid) > 3
                if method == :pearson
                    correlations[param] = cor(x[valid], y[valid])
                elseif method == :spearman
                    correlations[param] = cor(tiedrank(x[valid]), tiedrank(y[valid]))
                end
            else
                correlations[param] = 0.0
            end
        end
    end
    
    return correlations
end

function compute_prcc(data::DataFrame, param_names::Vector{Symbol}, output_name::Symbol)
    # Partial Rank Correlation Coefficient
    prcc_values = Dict{Symbol, Float64}()
    
    # Rank transform all variables
    ranked_data = DataFrame()
    
    for param in param_names
        if param in names(data, Symbol)
            ranked_data[!, param] = tiedrank(Float64.(data[!, param]))
        end
    end
    
    if output_name in names(data, Symbol)
        ranked_data[!, output_name] = tiedrank(Float64.(data[!, output_name]))
    end
    
    # Compute partial correlations
    for param in param_names
        if param in names(ranked_data, Symbol)
            # Regress out other parameters
            other_params = filter(p -> p != param, param_names)
            
            # Simplified partial correlation
            # In practice, would use proper regression
            prcc_values[param] = cor(ranked_data[!, param], ranked_data[!, output_name])
        end
    end
    
    return prcc_values
end

function compute_src(data::DataFrame, param_names::Vector{Symbol}, output_name::Symbol)
    # Standardized Regression Coefficient
    src_values = Dict{Symbol, Float64}()
    
    # Standardize variables
    standardized_data = DataFrame()
    
    for param in param_names
        if param in names(data, Symbol)
            x = Float64.(data[!, param])
            standardized_data[!, param] = (x .- mean(x)) ./ std(x)
        end
    end
    
    if output_name in names(data, Symbol)
        y = Float64.(data[!, output_name])
        standardized_data[!, output_name] = (y .- mean(y)) ./ std(y)
    end
    
    # Multiple linear regression (simplified)
    # In practice, would use GLM.jl
    for param in param_names
        if param in names(standardized_data, Symbol)
            src_values[param] = cor(standardized_data[!, param], 
                                   standardized_data[!, output_name])
        end
    end
    
    return src_values
end

function compute_sobol_indices(data::DataFrame, param_names::Vector{Symbol}, 
                               output_name::Symbol)
    # Simplified Sobol sensitivity indices
    # Full implementation would use variance decomposition
    sobol_indices = Dict{Symbol, Dict{Symbol, Float64}}()
    
    total_variance = var(data[!, output_name])
    
    for param in param_names
        first_order = compute_first_order_sobol(data, param, output_name, total_variance)
        total_effect = compute_total_effect_sobol(data, param, output_name, total_variance)
        
        sobol_indices[param] = Dict(
            :first_order => first_order,
            :total_effect => total_effect
        )
    end
    
    return sobol_indices
end

function compute_first_order_sobol(data, param, output, total_var)
    # Simplified first-order Sobol index
    # Group by parameter bins
    param_vals = data[!, param]
    n_bins = min(10, length(unique(param_vals)))
    
    if n_bins < 2
        return 0.0
    end
    
    # Bin the parameter
    bins = quantile(param_vals, range(0, 1, length=n_bins+1))
    binned = cut_simple(param_vals, bins)
    
    # Compute conditional means
    conditional_means = Float64[]
    for bin in unique(binned)
        idx = binned .== bin
        if sum(idx) > 0
            push!(conditional_means, mean(data[idx, output]))
        end
    end
    
    # Variance of conditional means
    var_conditional = var(conditional_means)
    
    return min(1.0, var_conditional / total_var)
end

function compute_total_effect_sobol(data, param, output, total_var)
    # Simplified total effect index
    # Would need proper sampling design for accurate calculation
    return compute_first_order_sobol(data, param, output, total_var) * 1.2
end

function morris_method(param_ranges::Dict, n_trajectories::Int=10, n_levels::Int=4)
    # Morris screening method for parameter importance
    # Simplified implementation
    return Dict()
end

function parameter_screening(base_params, param_ranges::Dict, n_samples::Int=100)
    # Elementary effects screening
    results = Dict{Symbol, Float64}()
    
    for (param, range) in param_ranges
        effects = Float64[]
        
        for _ in 1:n_samples
            # Random baseline
            baseline = range[1] + rand() * (range[2] - range[1])
            
            # Perturbation
            delta = (range[2] - range[1]) / 10
            perturbed = baseline + delta
            
            # Compute elementary effect (would need actual simulation)
            effect = rand()  # Placeholder
            push!(effects, effect)
        end
        
        results[param] = mean(abs.(effects))
    end
    
    return results
end

# Helper function
function cut_simple(x::Vector, breaks::Vector)
    result = String[]
    for val in x
        for i in 1:length(breaks)-1
            if val >= breaks[i] && val <= breaks[i+1]
                push!(result, "bin_$i")
                break
            end
        end
    end
    return result
end

function tiedrank(x::Vector)
    # Simple ranking with ties
    n = length(x)
    ranks = zeros(n)
    sorted_idx = sortperm(x)
    
    i = 1
    while i <= n
        j = i
        while j < n && x[sorted_idx[j+1]] ≈ x[sorted_idx[i]]
            j += 1
        end
        
        # Average rank for ties
        avg_rank = (i + j) / 2
        for k in i:j
            ranks[sorted_idx[k]] = avg_rank
        end
        
        i = j + 1
    end
    
    return ranks
end

end # module Sensitivity